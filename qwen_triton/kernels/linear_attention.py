from __future__ import annotations

import os
import warnings

import torch

try:
    import triton
    import triton.language as tl

    _TRITON_AVAILABLE = True
except Exception:  # pragma: no cover - import guard
    triton = None
    tl = None
    _TRITON_AVAILABLE = False


_TRITON_RUNTIME_OK = _TRITON_AVAILABLE
_FALLBACK_WARNED = False


def _warn_fallback_once(exc: Exception) -> None:
    global _FALLBACK_WARNED
    if _FALLBACK_WARNED:
        return
    _FALLBACK_WARNED = True
    warnings.warn(
        f"[Qwen-Triton fallback] Linear-attention Triton kernel unavailable ({exc.__class__.__name__}: {exc}). Using torch fallback.",
        RuntimeWarning,
        stacklevel=2,
    )


def _torch_gated_delta_rule_step(
    state: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    decay: torch.Tensor,
    beta: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    decay_factor = decay.exp().unsqueeze(-1).unsqueeze(-1)
    state = state * decay_factor
    kv_mem = (state * key.unsqueeze(-1)).sum(dim=-2)
    delta = (value - kv_mem) * beta.unsqueeze(-1)
    state = state + key.unsqueeze(-1) * delta.unsqueeze(-2)
    out = (state * query.unsqueeze(-1)).sum(dim=-2)
    return out, state


if _TRITON_AVAILABLE:
    @triton.jit
    def _gated_delta_rule_step_kernel(
        state_ptr,
        query_ptr,
        key_ptr,
        value_ptr,
        decay_ptr,
        beta_ptr,
        out_ptr,
        k_dim,
        v_dim,
        BLOCK_K: tl.constexpr,
        BLOCK_V: tl.constexpr,
    ):
        bh = tl.program_id(0)
        v_block = tl.program_id(1)
        offs_v = v_block * BLOCK_V + tl.arange(0, BLOCK_V)
        mask_v = offs_v < v_dim

        decay = tl.exp(tl.load(decay_ptr + bh).to(tl.float32))
        beta = tl.load(beta_ptr + bh).to(tl.float32)
        value = tl.load(value_ptr + bh * v_dim + offs_v, mask=mask_v, other=0.0).to(tl.float32)
        kv_mem = tl.zeros((BLOCK_V,), dtype=tl.float32)

        for k_base in range(0, k_dim, BLOCK_K):
            offs_k = k_base + tl.arange(0, BLOCK_K)
            mask_k = offs_k < k_dim
            state_ptrs = state_ptr + (bh * k_dim + offs_k[:, None]) * v_dim + offs_v[None, :]
            state = tl.load(state_ptrs, mask=mask_k[:, None] & mask_v[None, :], other=0.0).to(tl.float32)
            state = state * decay
            tl.store(state_ptrs, state, mask=mask_k[:, None] & mask_v[None, :])
            key = tl.load(key_ptr + bh * k_dim + offs_k, mask=mask_k, other=0.0).to(tl.float32)
            kv_mem += tl.sum(state * key[:, None], axis=0)

        delta = (value - kv_mem) * beta
        out = tl.zeros((BLOCK_V,), dtype=tl.float32)

        for k_base in range(0, k_dim, BLOCK_K):
            offs_k = k_base + tl.arange(0, BLOCK_K)
            mask_k = offs_k < k_dim
            state_ptrs = state_ptr + (bh * k_dim + offs_k[:, None]) * v_dim + offs_v[None, :]
            state = tl.load(state_ptrs, mask=mask_k[:, None] & mask_v[None, :], other=0.0).to(tl.float32)
            key = tl.load(key_ptr + bh * k_dim + offs_k, mask=mask_k, other=0.0).to(tl.float32)
            query = tl.load(query_ptr + bh * k_dim + offs_k, mask=mask_k, other=0.0).to(tl.float32)
            state = state + key[:, None] * delta[None, :]
            tl.store(state_ptrs, state, mask=mask_k[:, None] & mask_v[None, :])
            out += tl.sum(state * query[:, None], axis=0)

        tl.store(out_ptr + bh * v_dim + offs_v, out, mask=mask_v)


def _triton_gated_delta_rule_step(
    state: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    decay: torch.Tensor,
    beta: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    out = torch.empty_like(value, dtype=torch.float32)
    grid = (state.shape[0], triton.cdiv(state.shape[-1], 64))
    _gated_delta_rule_step_kernel[grid](
        state,
        query,
        key,
        value,
        decay,
        beta,
        out,
        state.shape[-2],
        state.shape[-1],
        BLOCK_K=32,
        BLOCK_V=64,
    )
    return out, state


def gated_delta_rule_sequence(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    decay: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor | None = None,
    use_triton: bool | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    global _TRITON_RUNTIME_OK
    """
    Args:
        query/key: [batch, seq, heads, key_dim]
        value: [batch, seq, heads, value_dim]
        decay/beta: [batch, seq, heads]
        initial_state: [batch, heads, key_dim, value_dim]
    Returns:
        outputs: [batch, seq, heads, value_dim]
        final_state: [batch, heads, key_dim, value_dim]
    """
    batch_size, seq_len, num_heads, key_dim = query.shape
    value_dim = value.shape[-1]
    use_triton = (query.is_cuda and _TRITON_AVAILABLE and _TRITON_RUNTIME_OK) if use_triton is None else use_triton
    query = query * (key_dim ** -0.5)

    if initial_state is None:
        state = torch.zeros(
            (batch_size * num_heads, key_dim, value_dim),
            device=query.device,
            dtype=torch.float32,
        )
    else:
        state = initial_state.reshape(batch_size * num_heads, key_dim, value_dim).to(torch.float32)

    outputs = torch.empty((batch_size, seq_len, num_heads, value_dim), device=value.device, dtype=torch.float32)
    step_impl = _triton_gated_delta_rule_step if use_triton else _torch_gated_delta_rule_step

    for token_idx in range(seq_len):
        query_t = query[:, token_idx].reshape(batch_size * num_heads, key_dim).to(torch.float32)
        key_t = key[:, token_idx].reshape(batch_size * num_heads, key_dim).to(torch.float32)
        value_t = value[:, token_idx].reshape(batch_size * num_heads, value_dim).to(torch.float32)
        decay_t = decay[:, token_idx].reshape(batch_size * num_heads).to(torch.float32)
        beta_t = beta[:, token_idx].reshape(batch_size * num_heads).to(torch.float32)
        if use_triton:
            try:
                out_t, state = step_impl(state, query_t, key_t, value_t, decay_t, beta_t)
            except Exception as exc:
                _TRITON_RUNTIME_OK = False
                if os.environ.get("QWEN_TRITON_STRICT") == "1":
                    raise RuntimeError("Linear-attention Triton kernel failed and strict Triton mode is enabled.") from exc
                _warn_fallback_once(exc)
                use_triton = False
                step_impl = _torch_gated_delta_rule_step
                out_t, state = step_impl(state, query_t, key_t, value_t, decay_t, beta_t)
        else:
            out_t, state = step_impl(state, query_t, key_t, value_t, decay_t, beta_t)
        outputs[:, token_idx] = out_t.view(batch_size, num_heads, value_dim)

    final_state = state.view(batch_size, num_heads, key_dim, value_dim)
    return outputs.to(value.dtype), final_state
