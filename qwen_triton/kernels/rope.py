from __future__ import annotations

import os
import warnings

import torch

from qwen_triton.ops import apply_rope_cuda_op, get_rope_cuda_op_error, load_rope_cuda_op

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
_CUDA_OP_FALLBACK_WARNED = False
_TORCH_FALLBACK_WARNED = False


def _warn_fallback_once(exc: Exception) -> None:
    global _FALLBACK_WARNED
    if _FALLBACK_WARNED:
        return
    _FALLBACK_WARNED = True
    warnings.warn(
        f"[Qwen-Triton fallback] RoPE Triton kernel unavailable ({exc.__class__.__name__}: {exc}). Using torch fallback.",
        RuntimeWarning,
        stacklevel=2,
    )


def _warn_cuda_op_fallback_once(exc: Exception) -> None:
    global _CUDA_OP_FALLBACK_WARNED
    if _CUDA_OP_FALLBACK_WARNED:
        return
    _CUDA_OP_FALLBACK_WARNED = True
    warnings.warn(
        f"[Qwen-Triton fallback] CUDA RoPE custom op unavailable ({exc.__class__.__name__}: {exc}). Using Triton fallback.",
        RuntimeWarning,
        stacklevel=2,
    )


def _warn_torch_fallback_once(reason: str) -> None:
    global _TORCH_FALLBACK_WARNED
    if _TORCH_FALLBACK_WARNED:
        return
    _TORCH_FALLBACK_WARNED = True
    warnings.warn(f"[Qwen-Triton fallback] {reason}. Using torch fallback.", RuntimeWarning, stacklevel=2)


def _torch_apply_rope_tensor(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    rotary_dim = cos.shape[-1]
    x_rot, x_pass = x[..., :rotary_dim], x[..., rotary_dim:]
    half = rotary_dim // 2
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    x1, x2 = x_rot[..., :half], x_rot[..., half:rotary_dim]
    cos_half = cos[..., :half]
    sin_half = sin[..., :half]
    rotated = torch.cat((x1 * cos_half - x2 * sin_half, x2 * cos_half + x1 * sin_half), dim=-1)
    return torch.cat((rotated, x_pass), dim=-1)


def _torch_apply_rope(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return _torch_apply_rope_tensor(q, cos, sin), _torch_apply_rope_tensor(k, cos, sin)


if _TRITON_AVAILABLE:
    @triton.jit
    def _rope_tensor_kernel(
        x_ptr,
        cos_ptr,
        sin_ptr,
        out_ptr,
        stride_xm,
        stride_om,
        stride_cm,
        half_dim,
        BLOCK_SIZE: tl.constexpr,
    ):
        row = tl.program_id(0)
        offsets = tl.arange(0, BLOCK_SIZE)
        mask = offsets < half_dim
        x1 = tl.load(x_ptr + row * stride_xm + offsets, mask=mask, other=0.0).to(tl.float32)
        x2 = tl.load(x_ptr + row * stride_xm + offsets + half_dim, mask=mask, other=0.0).to(tl.float32)
        cos = tl.load(cos_ptr + row * stride_cm + offsets, mask=mask, other=0.0).to(tl.float32)
        sin = tl.load(sin_ptr + row * stride_cm + offsets, mask=mask, other=0.0).to(tl.float32)

        out_1 = x1 * cos - x2 * sin
        out_2 = x2 * cos + x1 * sin

        tl.store(out_ptr + row * stride_om + offsets, out_1, mask=mask)
        tl.store(out_ptr + row * stride_om + offsets + half_dim, out_2, mask=mask)


def _resolve_backend(use_triton: bool | None, backend: str | None) -> str:
    if backend is None:
        if use_triton is False:
            return "torch"
        return os.environ.get("QWEN_TRITON_ROPE_BACKEND", "triton").lower()
    return backend.lower()


def _triton_apply_rope_tensor(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    rotary_dim = cos.shape[-1]
    half_dim = rotary_dim // 2
    batch, heads, seqlen, _ = x.shape
    cos_half = cos[..., :half_dim].unsqueeze(1).expand(batch, heads, seqlen, half_dim).contiguous().view(-1, half_dim)
    sin_half = sin[..., :half_dim].unsqueeze(1).expand(batch, heads, seqlen, half_dim).contiguous().view(-1, half_dim)
    x_rows = x.contiguous().view(-1, x.shape[-1])
    out = x_rows.clone()
    grid = (x_rows.shape[0],)
    block_size = max(2, 1 << (half_dim - 1).bit_length())
    _rope_tensor_kernel[grid](
        x_rows,
        cos_half,
        sin_half,
        out,
        x_rows.stride(0),
        out.stride(0),
        cos_half.stride(0),
        half_dim,
        BLOCK_SIZE=block_size,
    )
    return out.view_as(x)


def _apply_rope_backend_pair(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    resolved_backend: str,
    strict: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    global _TRITON_RUNTIME_OK
    if resolved_backend == "torch":
        return _torch_apply_rope(q, k, cos, sin)

    if resolved_backend not in {"auto", "cuda_op", "triton"}:
        raise ValueError(f"Unsupported RoPE backend: {resolved_backend}")

    if resolved_backend in {"auto", "cuda_op"} and q.is_cuda and k.is_cuda:
        if load_rope_cuda_op(verbose=os.environ.get("QWEN_TRITON_BUILD_VERBOSE") == "1"):
            try:
                return apply_rope_cuda_op(q, k, cos, sin)
            except Exception as exc:
                if strict and resolved_backend == "cuda_op":
                    raise RuntimeError("CUDA RoPE custom op failed and strict Triton mode is enabled.") from exc
                _warn_cuda_op_fallback_once(exc)
        elif resolved_backend == "cuda_op":
            error = get_rope_cuda_op_error()
            if strict:
                raise RuntimeError("CUDA RoPE custom op failed to build/load and strict Triton mode is enabled.") from error
            _warn_cuda_op_fallback_once(error or RuntimeError("Unknown CUDA custom op load failure"))

    can_use_triton = q.is_cuda and k.is_cuda and _TRITON_AVAILABLE and _TRITON_RUNTIME_OK
    if not can_use_triton:
        if strict and resolved_backend in {"triton", "auto", "cuda_op"}:
            raise RuntimeError("RoPE Triton kernel is unavailable and strict Triton mode is enabled.")
        if q.is_cuda and k.is_cuda:
            _warn_torch_fallback_once("RoPE Triton kernel is unavailable")
        return _torch_apply_rope(q, k, cos, sin)

    try:
        return _triton_apply_rope_tensor(q, cos, sin), _triton_apply_rope_tensor(k, cos, sin)
    except Exception as exc:
        _TRITON_RUNTIME_OK = False
        if strict:
            raise RuntimeError("RoPE Triton kernel failed and strict Triton mode is enabled.") from exc
        _warn_fallback_once(exc)
        return _torch_apply_rope(q, k, cos, sin)


class _RoPEPairFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        resolved_backend: str,
        strict: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        ctx.save_for_backward(cos, sin)
        return _apply_rope_backend_pair(q, k, cos, sin, resolved_backend, strict)

    @staticmethod
    def backward(
        ctx,
        grad_q: torch.Tensor,
        grad_k: torch.Tensor,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, None, None, None, None]:
        cos, sin = ctx.saved_tensors
        grad_q_input = _torch_apply_rope_tensor(grad_q, cos, -sin) if ctx.needs_input_grad[0] else None
        grad_k_input = _torch_apply_rope_tensor(grad_k, cos, -sin) if ctx.needs_input_grad[1] else None
        return grad_q_input, grad_k_input, None, None, None, None


def apply_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    use_triton: bool | None = None,
    backend: str | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    global _TRITON_RUNTIME_OK
    resolved_backend = _resolve_backend(use_triton, backend)
    strict = os.environ.get("QWEN_TRITON_STRICT") == "1"
    if torch.is_grad_enabled() and (q.requires_grad or k.requires_grad):
        return _RoPEPairFunction.apply(q, k, cos, sin, resolved_backend, strict)
    return _apply_rope_backend_pair(q, k, cos, sin, resolved_backend, strict)
