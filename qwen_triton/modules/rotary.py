from __future__ import annotations

import torch
from torch import nn

from qwen_triton.configs import QwenTritonConfig
from qwen_triton.kernels import apply_rope


class QwenRotaryEmbedding(nn.Module):
    def __init__(self, config: QwenTritonConfig) -> None:
        super().__init__()
        self.config = config
        rotary_dim = config.rotary_dim
        inv_freq = 1.0 / (
            config.rope_theta ** (torch.arange(0, rotary_dim, 2, dtype=torch.float32) / rotary_dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    @torch.no_grad()
    def forward(self, hidden_states: torch.Tensor, position_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        inv_freq = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(hidden_states.device)
        position_ids = position_ids[:, None, :].float()
        freqs = (inv_freq @ position_ids).transpose(1, 2)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos().to(hidden_states.dtype), emb.sin().to(hidden_states.dtype)


def apply_rotary_pos_emb(
    query: torch.Tensor,
    key: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    return apply_rope(query, key, cos, sin)
