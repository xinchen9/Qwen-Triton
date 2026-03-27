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
        f"[Qwen-Triton fallback] SiLU-mul Triton kernel unavailable ({exc.__class__.__name__}: {exc}). Using torch fallback.",
        RuntimeWarning,
        stacklevel=2,
    )


def _torch_silu_mul(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.silu(gate) * up


def _torch_silu_mul_backward(
    gate: torch.Tensor,
    up: torch.Tensor,
    grad_output: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    gate_fp32 = gate.to(torch.float32)
    up_fp32 = up.to(torch.float32)
    grad_fp32 = grad_output.to(torch.float32)
    sig = torch.sigmoid(gate_fp32)
    silu = gate_fp32 * sig
    grad_gate = grad_fp32 * up_fp32 * sig * (1.0 + gate_fp32 * (1.0 - sig))
    grad_up = grad_fp32 * silu
    return grad_gate.to(gate.dtype), grad_up.to(up.dtype)


def _triton_silu_mul_forward(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    gate_2d = gate.contiguous().view(-1, gate.shape[-1])
    up_2d = up.contiguous().view(-1, up.shape[-1])
    out = torch.empty_like(gate_2d)
    grid = lambda meta: (gate_2d.shape[0], triton.cdiv(gate_2d.shape[1], meta["BLOCK_SIZE"]))
    _silu_mul_kernel[grid](
        gate_2d,
        up_2d,
        out,
        gate_2d.stride(0),
        up_2d.stride(0),
        out.stride(0),
        gate_2d.shape[1],
    )
    return out.view_as(gate)


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
    def _silu_mul_kernel(gate_ptr, up_ptr, out_ptr, stride_gm, stride_um, stride_om, n_cols, BLOCK_SIZE: tl.constexpr):
        row = tl.program_id(0)
        col_block = tl.program_id(1)
        offsets = col_block * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_cols
        gate = tl.load(gate_ptr + row * stride_gm + offsets, mask=mask, other=0.0).to(tl.float32)
        up = tl.load(up_ptr + row * stride_um + offsets, mask=mask, other=0.0).to(tl.float32)
        out = gate * tl.sigmoid(gate) * up
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
    def _silu_mul_backward_kernel(
        gate_ptr,
        up_ptr,
        grad_out_ptr,
        grad_gate_ptr,
        grad_up_ptr,
        stride_gm,
        stride_um,
        stride_gom,
        stride_ggm,
        stride_gum,
        n_cols,
        BLOCK_SIZE: tl.constexpr,
    ):
        row = tl.program_id(0)
        col_block = tl.program_id(1)
        offsets = col_block * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_cols
        gate = tl.load(gate_ptr + row * stride_gm + offsets, mask=mask, other=0.0).to(tl.float32)
        up = tl.load(up_ptr + row * stride_um + offsets, mask=mask, other=0.0).to(tl.float32)
        grad_out = tl.load(grad_out_ptr + row * stride_gom + offsets, mask=mask, other=0.0).to(tl.float32)
        sig = tl.sigmoid(gate)
        silu = gate * sig
        grad_gate = grad_out * up * sig * (1.0 + gate * (1.0 - sig))
        grad_up = grad_out * silu
        tl.store(grad_gate_ptr + row * stride_ggm + offsets, grad_gate, mask=mask)
        tl.store(grad_up_ptr + row * stride_gum + offsets, grad_up, mask=mask)


class _SiLUMulFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(gate, up)
        return _triton_silu_mul_forward(gate, up)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        gate, up = ctx.saved_tensors
        grad_output = grad_output.contiguous()
        try:
            gate_2d = gate.contiguous().view(-1, gate.shape[-1])
            up_2d = up.contiguous().view(-1, up.shape[-1])
            grad_2d = grad_output.view(-1, grad_output.shape[-1])
            grad_gate = torch.empty_like(gate_2d)
            grad_up = torch.empty_like(up_2d)
            grid = lambda meta: (gate_2d.shape[0], triton.cdiv(gate_2d.shape[1], meta["BLOCK_SIZE"]))
            _silu_mul_backward_kernel[grid](
                gate_2d,
                up_2d,
                grad_2d,
                grad_gate,
                grad_up,
                gate_2d.stride(0),
                up_2d.stride(0),
                grad_2d.stride(0),
                grad_gate.stride(0),
                grad_up.stride(0),
                gate_2d.shape[1],
            )
            return grad_gate.view_as(gate), grad_up.view_as(up)
        except Exception as exc:
            global _TRITON_RUNTIME_OK
            _TRITON_RUNTIME_OK = False
            if os.environ.get("QWEN_TRITON_STRICT") == "1":
                raise RuntimeError("SiLU-mul Triton backward kernel failed and strict Triton mode is enabled.") from exc
            _warn_fallback_once(exc)
            return _torch_silu_mul_backward(gate, up, grad_output)


def silu_mul(gate: torch.Tensor, up: torch.Tensor, use_triton: bool | None = None) -> torch.Tensor:
    global _TRITON_RUNTIME_OK
    use_triton = (gate.is_cuda and up.is_cuda and _TRITON_AVAILABLE) if use_triton is None else use_triton
    if not use_triton or not _TRITON_RUNTIME_OK:
        return _torch_silu_mul(gate, up)

    try:
        if torch.is_grad_enabled() and (gate.requires_grad or up.requires_grad):
            return _SiLUMulFunction.apply(gate, up)
        return _triton_silu_mul_forward(gate, up)
    except Exception as exc:
        _TRITON_RUNTIME_OK = False
        if os.environ.get("QWEN_TRITON_STRICT") == "1":
            raise RuntimeError("SiLU-mul Triton kernel failed and strict Triton mode is enabled.") from exc
        _warn_fallback_once(exc)
        return _torch_silu_mul(gate, up)
