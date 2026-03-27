from __future__ import annotations

from dataclasses import dataclass

import torch

from qwen_triton.kernels import append_attention_kv


@dataclass
class QwenTritonCache:
    num_layers: int
    layer_types: list[str]

    def __post_init__(self) -> None:
        self.key_cache: list[torch.Tensor | None] = [None] * self.num_layers
        self.value_cache: list[torch.Tensor | None] = [None] * self.num_layers
        self.conv_states: list[torch.Tensor | None] = [None] * self.num_layers
        self.recurrent_states: list[torch.Tensor | None] = [None] * self.num_layers

    def update_attention(
        self,
        layer_idx: int,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self.key_cache[layer_idx] = append_attention_kv(self.key_cache[layer_idx], key_states)
        self.value_cache[layer_idx] = append_attention_kv(self.value_cache[layer_idx], value_states)
        return self.key_cache[layer_idx], self.value_cache[layer_idx]

    def get_seq_length(self, layer_idx: int | None = None) -> int:
        if layer_idx is not None and self.key_cache[layer_idx] is not None:
            return int(self.key_cache[layer_idx].shape[-2])
        for key_cache in self.key_cache:
            if key_cache is not None:
                return int(key_cache.shape[-2])
        return 0

    @property
    def has_previous_state(self) -> bool:
        linear_indices = [idx for idx, layer_type in enumerate(self.layer_types) if layer_type == "linear_attention"]
        if not linear_indices:
            return False
        return self.recurrent_states[linear_indices[-1]] is not None
