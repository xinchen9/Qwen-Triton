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
        f"[Qwen-Triton fallback] RMSNorm Triton kernel unavailable ({exc.__class__.__name__}: {exc}). Using torch fallback.",
        RuntimeWarning,
        stacklevel=2,
    )


def _torch_rmsnorm(hidden_states: torch.Tensor, weight: torch.Tensor, eps: float, one_plus_weight: bool) -> torch.Tensor:
    input_dtype = hidden_states.dtype
    x = hidden_states.to(torch.float32)
    variance = x.pow(2).mean(dim=-1, keepdim=True)
    scale = 1.0 + weight.to(torch.float32) if one_plus_weight else weight.to(torch.float32)
    y = x * torch.rsqrt(variance + eps)
    y = y * scale
    return y.to(input_dtype)


def _torch_rmsnorm_backward(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    grad_output: torch.Tensor,
    eps: float,
    one_plus_weight: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    x = hidden_states.to(torch.float32)
    grad = grad_output.to(torch.float32)
    scale = 1.0 + weight.to(torch.float32) if one_plus_weight else weight.to(torch.float32)
    variance = x.pow(2).mean(dim=-1, keepdim=True)
    inv_std = torch.rsqrt(variance + eps)
    normalized = x * inv_std
    grad_weight = (grad * normalized).sum(dim=tuple(range(grad.ndim - 1)))
    dot = (grad * scale * x).sum(dim=-1, keepdim=True)
    cols = x.shape[-1]
    grad_x = grad * scale * inv_std - x * (inv_std.pow(3) / cols) * dot
    return grad_x.to(hidden_states.dtype), grad_weight.to(weight.dtype)


def _triton_rmsnorm_forward(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    one_plus_weight: bool,
) -> torch.Tensor:
    original_shape = hidden_states.shape
    x_2d = hidden_states.contiguous().view(-1, original_shape[-1])
    out = torch.empty_like(x_2d)
    grid = (x_2d.shape[0],)
    _rmsnorm_kernel[grid](
        x_2d,
        weight,
        out,
        x_2d.stride(0),
        out.stride(0),
        x_2d.shape[1],
        eps,
        1 if one_plus_weight else 0,
    )
    return out.view(*original_shape)


if _TRITON_AVAILABLE:
    @triton.autotune(
        configs=[
            triton.Config({"BLOCK_SIZE": 64}, num_warps=1),
            triton.Config({"BLOCK_SIZE": 64}, num_warps=2),
            triton.Config({"BLOCK_SIZE": 128}, num_warps=2),
            triton.Config({"BLOCK_SIZE": 128}, num_warps=4),
            triton.Config({"BLOCK_SIZE": 256}, num_warps=4),
            triton.Config({"BLOCK_SIZE": 512}, num_warps=8),
        ],
        key=["n_cols"],
    )
    @triton.jit
    def _rmsnorm_kernel(
        x_ptr,
        weight_ptr,
        out_ptr,
        stride_xm,
        stride_om,
        n_cols,
        eps,
        one_plus_weight,
        BLOCK_SIZE: tl.constexpr,
    ):
        row = tl.program_id(0)
        row_x_ptr = x_ptr + row * stride_xm
        row_out_ptr = out_ptr + row * stride_om
        square_sum = tl.zeros((), dtype=tl.float32)

        for block_start in tl.range(0, n_cols, BLOCK_SIZE):
            offsets = block_start + tl.arange(0, BLOCK_SIZE)
            mask = offsets < n_cols
            x = tl.load(row_x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
            square_sum += tl.sum(x * x, axis=0)

        variance = square_sum / n_cols
        inv_std = tl.rsqrt(variance + eps)

        for block_start in tl.range(0, n_cols, BLOCK_SIZE):
            offsets = block_start + tl.arange(0, BLOCK_SIZE)
            mask = offsets < n_cols
            x = tl.load(row_x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
            weight = tl.load(weight_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
            scale = tl.where(one_plus_weight > 0, 1.0 + weight, weight)
            y = x * inv_std * scale
            tl.store(row_out_ptr + offsets, y, mask=mask)

    @triton.autotune(
        configs=[
            triton.Config({"BLOCK_SIZE": 64}, num_warps=1),
            triton.Config({"BLOCK_SIZE": 64}, num_warps=2),
            triton.Config({"BLOCK_SIZE": 128}, num_warps=2),
            triton.Config({"BLOCK_SIZE": 128}, num_warps=4),
            triton.Config({"BLOCK_SIZE": 256}, num_warps=4),
            triton.Config({"BLOCK_SIZE": 512}, num_warps=8),
        ],
        key=["n_cols"],
        reset_to_zero=["grad_weight_ptr"],
    )
    @triton.jit
    def _rmsnorm_backward_kernel(
        x_ptr,
        weight_ptr,
        grad_out_ptr,
        grad_x_ptr,
        grad_weight_ptr,
        stride_xm,
        stride_gom,
        stride_gxm,
        n_cols,
        eps,
        one_plus_weight,
        BLOCK_SIZE: tl.constexpr,
    ):
        row = tl.program_id(0)
        row_x_ptr = x_ptr + row * stride_xm
        row_go_ptr = grad_out_ptr + row * stride_gom
        row_gx_ptr = grad_x_ptr + row * stride_gxm

        square_sum = tl.zeros((), dtype=tl.float32)
        dot = tl.zeros((), dtype=tl.float32)
        inv_cols = 1.0 / n_cols

        for block_start in tl.range(0, n_cols, BLOCK_SIZE):
            offsets = block_start + tl.arange(0, BLOCK_SIZE)
            mask = offsets < n_cols
            x = tl.load(row_x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
            grad_out = tl.load(row_go_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
            weight = tl.load(weight_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
            scale = tl.where(one_plus_weight > 0, 1.0 + weight, weight)
            square_sum += tl.sum(x * x, axis=0)
            dot += tl.sum(grad_out * scale * x, axis=0)

        variance = square_sum * inv_cols
        inv_std = tl.rsqrt(variance + eps)
        inv_std_cubed = inv_std * inv_std * inv_std
        dot_term = dot * inv_cols

        for block_start in tl.range(0, n_cols, BLOCK_SIZE):
            offsets = block_start + tl.arange(0, BLOCK_SIZE)
            mask = offsets < n_cols
            x = tl.load(row_x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
            grad_out = tl.load(row_go_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
            weight = tl.load(weight_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
            scale = tl.where(one_plus_weight > 0, 1.0 + weight, weight)
            normalized = x * inv_std
            grad_x = grad_out * scale * inv_std - x * inv_std_cubed * dot_term
            tl.store(row_gx_ptr + offsets, grad_x, mask=mask)
            tl.atomic_add(grad_weight_ptr + offsets, grad_out * normalized, mask=mask)


class _RMSNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        hidden_states: torch.Tensor,
        weight: torch.Tensor,
        eps: float,
        one_plus_weight: bool,
    ) -> torch.Tensor:
        ctx.eps = eps
        ctx.one_plus_weight = one_plus_weight
        ctx.save_for_backward(hidden_states, weight)
        return _triton_rmsnorm_forward(hidden_states, weight, eps, one_plus_weight)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, None, None]:
        hidden_states, weight = ctx.saved_tensors
        eps = ctx.eps
        one_plus_weight = ctx.one_plus_weight
        grad_output = grad_output.contiguous()
        try:
            original_shape = hidden_states.shape
            x_2d = hidden_states.contiguous().view(-1, original_shape[-1])
            grad_2d = grad_output.view(-1, original_shape[-1])
            grad_x = torch.empty_like(x_2d)
            grad_weight = torch.zeros((original_shape[-1],), device=weight.device, dtype=torch.float32)
            grid = (x_2d.shape[0],)
            _rmsnorm_backward_kernel[grid](
                x_2d,
                weight,
                grad_2d,
                grad_x,
                grad_weight,
                x_2d.stride(0),
                grad_2d.stride(0),
                grad_x.stride(0),
                x_2d.shape[1],
                eps,
                1 if one_plus_weight else 0,
            )
            return grad_x.view_as(hidden_states), grad_weight.to(weight.dtype), None, None
        except Exception as exc:
            global _TRITON_RUNTIME_OK
            _TRITON_RUNTIME_OK = False
            if os.environ.get("QWEN_TRITON_STRICT") == "1":
                raise RuntimeError("RMSNorm Triton backward kernel failed and strict Triton mode is enabled.") from exc
            _warn_fallback_once(exc)
            grad_x, grad_weight = _torch_rmsnorm_backward(hidden_states, weight, grad_output, eps, one_plus_weight)
            return grad_x, grad_weight, None, None


def rmsnorm(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
    one_plus_weight: bool = False,
    use_triton: bool | None = None,
) -> torch.Tensor:
    global _TRITON_RUNTIME_OK
    use_triton = (hidden_states.is_cuda and weight.is_cuda and _TRITON_AVAILABLE) if use_triton is None else use_triton
    if not use_triton or not _TRITON_RUNTIME_OK:
        return _torch_rmsnorm(hidden_states, weight, eps, one_plus_weight)

    try:
        if torch.is_grad_enabled() and (hidden_states.requires_grad or weight.requires_grad):
            return _RMSNormFunction.apply(hidden_states, weight, eps, one_plus_weight)
        return _triton_rmsnorm_forward(hidden_states, weight, eps, one_plus_weight)
    except Exception as exc:
        _TRITON_RUNTIME_OK = False
        if os.environ.get("QWEN_TRITON_STRICT") == "1":
            raise RuntimeError("RMSNorm Triton kernel failed and strict Triton mode is enabled.") from exc
        _warn_fallback_once(exc)
        return _torch_rmsnorm(hidden_states, weight, eps, one_plus_weight)
