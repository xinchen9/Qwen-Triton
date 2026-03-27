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
        f"[Qwen-Triton fallback] KV cache Triton helper unavailable ({exc.__class__.__name__}: {exc}). Using torch fallback.",
        RuntimeWarning,
        stacklevel=2,
    )


if _TRITON_AVAILABLE:
    @triton.jit
    def _append_kv_kernel(
        prefix_ptr,
        suffix_ptr,
        out_ptr,
        batch_size,
        num_heads,
        old_seq,
        new_seq,
        head_dim,
        total_elements,
        BLOCK_SIZE: tl.constexpr,
    ):
        pid = tl.program_id(0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < total_elements

        d = offsets % head_dim
        tmp = offsets // head_dim
        total_seq = old_seq + new_seq
        s = tmp % total_seq
        tmp = tmp // total_seq
        h = tmp % num_heads
        b = tmp // num_heads

        prefix_index = (((b * num_heads + h) * old_seq + s) * head_dim) + d
        suffix_index = (((b * num_heads + h) * new_seq + (s - old_seq)) * head_dim) + d
        out_index = offsets
        is_prefix = s < old_seq

        prefix_val = tl.load(prefix_ptr + prefix_index, mask=mask & is_prefix, other=0.0)
        suffix_val = tl.load(suffix_ptr + suffix_index, mask=mask & (~is_prefix), other=0.0)
        out = tl.where(is_prefix, prefix_val, suffix_val)
        tl.store(out_ptr + out_index, out, mask=mask)


def append_attention_kv(prefix: torch.Tensor | None, suffix: torch.Tensor) -> torch.Tensor:
    global _TRITON_RUNTIME_OK
    if prefix is None:
        return suffix
    if not (prefix.is_cuda and suffix.is_cuda and _TRITON_AVAILABLE and _TRITON_RUNTIME_OK):
        return torch.cat((prefix, suffix), dim=-2)

    if prefix.dim() != 4 or suffix.dim() != 4:
        return torch.cat((prefix, suffix), dim=-2)

    batch_size, num_heads, old_seq, head_dim = prefix.shape
    new_seq = suffix.shape[-2]
    out = torch.empty(
        (batch_size, num_heads, old_seq + new_seq, head_dim),
        device=prefix.device,
        dtype=prefix.dtype,
    )
    total_elements = out.numel()
    try:
        grid = (triton.cdiv(total_elements, 256),)
        _append_kv_kernel[grid](
            prefix.contiguous(),
            suffix.contiguous(),
            out,
            batch_size,
            num_heads,
            old_seq,
            new_seq,
            head_dim,
            total_elements,
            BLOCK_SIZE=256,
        )
    except Exception as exc:
        _TRITON_RUNTIME_OK = False
        if os.environ.get("QWEN_TRITON_STRICT") == "1":
            raise RuntimeError("KV cache Triton helper failed and strict Triton mode is enabled.") from exc
        _warn_fallback_once(exc)
        return torch.cat((prefix, suffix), dim=-2)
    return out
