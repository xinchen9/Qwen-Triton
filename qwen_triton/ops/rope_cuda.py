from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.cpp_extension import load

_EXTENSION_NAME = "qwen_triton_rope_cuda"
_LOAD_ATTEMPTED = False
_LOAD_ERROR: Exception | None = None
_LOADED = False


def _source_paths() -> list[str]:
    package_root = Path(__file__).resolve().parents[1]
    return [
        str(package_root / "csrc" / "rope_op.cpp"),
        str(package_root / "csrc" / "rope_op_kernel.cu"),
    ]


def _build_directory() -> Path:
    build_dir = Path(__file__).resolve().parents[2] / ".cache" / "torch_extensions" / _EXTENSION_NAME
    build_dir.mkdir(parents=True, exist_ok=True)
    return build_dir


def get_rope_cuda_op_error() -> Exception | None:
    return _LOAD_ERROR


def load_rope_cuda_op(verbose: bool = False) -> bool:
    global _LOAD_ATTEMPTED, _LOAD_ERROR, _LOADED
    if _LOADED:
        return True
    if _LOAD_ATTEMPTED and _LOAD_ERROR is not None:
        return False

    _LOAD_ATTEMPTED = True
    try:
        load(
            name=_EXTENSION_NAME,
            sources=_source_paths(),
            extra_cflags=["-O3"],
            extra_cuda_cflags=["-O3"],
            build_directory=str(_build_directory()),
            with_cuda=True,
            is_python_module=False,
            verbose=verbose,
        )
    except Exception as exc:
        _LOAD_ERROR = exc
        _LOADED = False
        return False

    _LOAD_ERROR = None
    _LOADED = True
    return True


class _RopeTensorCudaFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        if not load_rope_cuda_op():
            error = get_rope_cuda_op_error()
            raise RuntimeError("Qwen-Triton CUDA RoPE custom op is unavailable.") from error
        ctx.save_for_backward(cos, sin)
        return torch.ops.qwen_triton.rope_tensor_forward(x.contiguous(), cos.contiguous(), sin.contiguous())

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor | None, None, None]:
        if grad_output is None:
            return None, None, None
        cos, sin = ctx.saved_tensors
        grad_input = torch.ops.qwen_triton.rope_tensor_forward(
            grad_output.contiguous(),
            cos.contiguous(),
            (-sin).contiguous(),
        )
        return grad_input, None, None


def apply_rope_cuda_op(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    return _RopeTensorCudaFunction.apply(q, cos, sin), _RopeTensorCudaFunction.apply(k, cos, sin)
