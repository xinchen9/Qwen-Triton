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
        f"[Qwen-Triton fallback] Sigmoid-mul Triton kernel unavailable ({exc.__class__.__name__}: {exc}). Using torch fallback.",
        RuntimeWarning,
        stacklevel=2,
    )


def _torch_sigmoid_mul(x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(gate)


def _torch_sigmoid_mul_backward(
    x: torch.Tensor,
    gate: torch.Tensor,
    grad_output: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    x_fp32 = x.to(torch.float32)
    gate_fp32 = gate.to(torch.float32)
    grad_fp32 = grad_output.to(torch.float32)
    sig = torch.sigmoid(gate_fp32)
    grad_x = grad_fp32 * sig
    grad_gate = grad_fp32 * x_fp32 * sig * (1.0 - sig)
    return grad_x.to(x.dtype), grad_gate.to(gate.dtype)


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
    def _sigmoid_mul_kernel(
        x_ptr,
        gate_ptr,
        out_ptr,
        stride_xm,
        stride_gm,
        stride_om,
        n_cols,
        BLOCK_SIZE: tl.constexpr,
    ):
        row = tl.program_id(0)
        col_block = tl.program_id(1)
        offsets = col_block * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_cols
        x = tl.load(x_ptr + row * stride_xm + offsets, mask=mask, other=0.0).to(tl.float32)
        gate = tl.load(gate_ptr + row * stride_gm + offsets, mask=mask, other=0.0).to(tl.float32)
        out = x * tl.sigmoid(gate)
        tl.store(out_ptr + row * stride_om + offsets, out, mask=mask)

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
    def _sigmoid_mul_backward_kernel(
        x_ptr,
        gate_ptr,
        grad_out_ptr,
        grad_x_ptr,
        grad_gate_ptr,
        stride_xm,
        stride_gm,
        stride_gom,
        stride_gxm,
        stride_ggm,
        n_cols,
        BLOCK_SIZE: tl.constexpr,
    ):
        row = tl.program_id(0)
        col_block = tl.program_id(1)
        offsets = col_block * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_cols
        x = tl.load(x_ptr + row * stride_xm + offsets, mask=mask, other=0.0).to(tl.float32)
        gate = tl.load(gate_ptr + row * stride_gm + offsets, mask=mask, other=0.0).to(tl.float32)
        grad_out = tl.load(grad_out_ptr + row * stride_gom + offsets, mask=mask, other=0.0).to(tl.float32)
        sig = tl.sigmoid(gate)
        tl.store(grad_x_ptr + row * stride_gxm + offsets, grad_out * sig, mask=mask)
        tl.store(grad_gate_ptr + row * stride_ggm + offsets, grad_out * x * sig * (1.0 - sig), mask=mask)


def _triton_sigmoid_mul_forward(x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    x_2d = x.contiguous().view(-1, x.shape[-1])
    gate_2d = gate.contiguous().view(-1, gate.shape[-1])
    out = torch.empty_like(x_2d)
    grid = lambda meta: (x_2d.shape[0], triton.cdiv(x_2d.shape[1], meta["BLOCK_SIZE"]))
    _sigmoid_mul_kernel[grid](
        x_2d,
        gate_2d,
        out,
        x_2d.stride(0),
        gate_2d.stride(0),
        out.stride(0),
        x_2d.shape[1],
    )
    return out.view_as(x)


class _SigmoidMulFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(x, gate)
        return _triton_sigmoid_mul_forward(x, gate)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x, gate = ctx.saved_tensors
        grad_output = grad_output.contiguous()
        try:
            x_2d = x.contiguous().view(-1, x.shape[-1])
            gate_2d = gate.contiguous().view(-1, gate.shape[-1])
            grad_2d = grad_output.view(-1, grad_output.shape[-1])
            grad_x = torch.empty_like(x_2d)
            grad_gate = torch.empty_like(gate_2d)
            grid = lambda meta: (x_2d.shape[0], triton.cdiv(x_2d.shape[1], meta["BLOCK_SIZE"]))
            _sigmoid_mul_backward_kernel[grid](
                x_2d,
                gate_2d,
                grad_2d,
                grad_x,
                grad_gate,
                x_2d.stride(0),
                gate_2d.stride(0),
                grad_2d.stride(0),
                grad_x.stride(0),
                grad_gate.stride(0),
                x_2d.shape[1],
            )
            return grad_x.view_as(x), grad_gate.view_as(gate)
        except Exception as exc:
            global _TRITON_RUNTIME_OK
            _TRITON_RUNTIME_OK = False
            if os.environ.get("QWEN_TRITON_STRICT") == "1":
                raise RuntimeError("Sigmoid-mul Triton backward kernel failed and strict Triton mode is enabled.") from exc
            _warn_fallback_once(exc)
            return _torch_sigmoid_mul_backward(x, gate, grad_output)


def sigmoid_mul(x: torch.Tensor, gate: torch.Tensor, use_triton: bool | None = None) -> torch.Tensor:
    global _TRITON_RUNTIME_OK
    use_triton = (x.is_cuda and gate.is_cuda and _TRITON_AVAILABLE) if use_triton is None else use_triton
    if not use_triton or not _TRITON_RUNTIME_OK:
        return _torch_sigmoid_mul(x, gate)

    try:
        if torch.is_grad_enabled() and (x.requires_grad or gate.requires_grad):
            return _SigmoidMulFunction.apply(x, gate)
        return _triton_sigmoid_mul_forward(x, gate)
    except Exception as exc:
        _TRITON_RUNTIME_OK = False
        if os.environ.get("QWEN_TRITON_STRICT") == "1":
            raise RuntimeError("Sigmoid-mul Triton kernel failed and strict Triton mode is enabled.") from exc
        _warn_fallback_once(exc)
        return _torch_sigmoid_mul(x, gate)
