from __future__ import annotations

import torch
from torch import nn

from qwen_triton.kernels import rmsnorm, silu_mul


class QwenRMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6, one_plus_weight: bool = False) -> None:
        super().__init__()
        init_value = 0.0 if one_plus_weight else 1.0
        self.weight = nn.Parameter(torch.full((hidden_size,), init_value))
        self.eps = eps
        self.one_plus_weight = one_plus_weight

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return rmsnorm(hidden_states, self.weight, eps=self.eps, one_plus_weight=self.one_plus_weight)

    def extra_repr(self) -> str:
        return f"{tuple(self.weight.shape)}, eps={self.eps}, one_plus_weight={self.one_plus_weight}"


class QwenRMSNormGated(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, hidden_states: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states_fp32 = hidden_states.to(torch.float32)
        variance = hidden_states_fp32.pow(2).mean(dim=-1, keepdim=True)
        normalized = hidden_states_fp32 * torch.rsqrt(variance + self.eps)
        normalized = self.weight.to(torch.float32) * normalized
        return silu_mul(gate.to(input_dtype), normalized.to(input_dtype))
