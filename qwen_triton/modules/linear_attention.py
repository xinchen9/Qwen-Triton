from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from qwen_triton.configs import QwenTritonConfig
from qwen_triton.kernels import gated_delta_rule_sequence
from qwen_triton.modules.cache import QwenTritonCache
from qwen_triton.modules.norms import QwenRMSNormGated


def apply_mask_to_padding_states(hidden_states: torch.Tensor, attention_mask: torch.Tensor | None) -> torch.Tensor:
    if attention_mask is not None and attention_mask.ndim == 2 and attention_mask.shape[1] > 1 and attention_mask.shape[0] > 1:
        hidden_states = hidden_states * attention_mask[:, :, None].to(hidden_states.dtype)
    return hidden_states


def l2norm(hidden_states: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return hidden_states * torch.rsqrt(hidden_states.pow(2).sum(dim=-1, keepdim=True) + eps)


class Qwen3NextLinearAttention(nn.Module):
    def __init__(self, config: QwenTritonConfig, layer_idx: int) -> None:
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_v_heads = config.linear_num_value_heads
        self.num_k_heads = config.linear_num_key_heads
        self.head_k_dim = config.linear_key_head_dim
        self.head_v_dim = config.linear_value_head_dim
        self.key_dim = self.num_k_heads * self.head_k_dim
        self.value_dim = self.num_v_heads * self.head_v_dim
        self.conv_kernel_size = config.linear_conv_kernel_dim

        self.conv_dim = self.key_dim * 2 + self.value_dim
        self.conv1d = nn.Conv1d(
            in_channels=self.conv_dim,
            out_channels=self.conv_dim,
            bias=False,
            kernel_size=self.conv_kernel_size,
            groups=self.conv_dim,
            padding=0,
        )

        projection_size_qkvz = self.key_dim * 2 + self.value_dim * 2
        projection_size_ba = self.num_v_heads * 2
        self.in_proj_qkvz = nn.Linear(self.hidden_size, projection_size_qkvz, bias=False)
        self.in_proj_ba = nn.Linear(self.hidden_size, projection_size_ba, bias=False)
        self.dt_bias = nn.Parameter(torch.ones(self.num_v_heads))
        self.A_log = nn.Parameter(torch.empty(self.num_v_heads).uniform_(0, 16).log_())
        self.norm = QwenRMSNormGated(self.head_v_dim, eps=config.rms_norm_eps)
        self.out_proj = nn.Linear(self.value_dim, self.hidden_size, bias=False)

    def fix_query_key_value_ordering(
        self,
        mixed_qkvz: torch.Tensor,
        mixed_ba: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        qkvz_shape = mixed_qkvz.size()[:-1] + (
            self.num_k_heads,
            2 * self.head_k_dim + 2 * self.head_v_dim * self.num_v_heads // self.num_k_heads,
        )
        ba_shape = mixed_ba.size()[:-1] + (self.num_k_heads, 2 * self.num_v_heads // self.num_k_heads)
        mixed_qkvz = mixed_qkvz.view(*qkvz_shape)
        mixed_ba = mixed_ba.view(*ba_shape)
        split_qkvz = [
            self.head_k_dim,
            self.head_k_dim,
            self.num_v_heads // self.num_k_heads * self.head_v_dim,
            self.num_v_heads // self.num_k_heads * self.head_v_dim,
        ]
        split_ba = [self.num_v_heads // self.num_k_heads, self.num_v_heads // self.num_k_heads]
        query, key, value, z = torch.split(mixed_qkvz, split_qkvz, dim=3)
        b, a = torch.split(mixed_ba, split_ba, dim=3)
        value = value.reshape(value.shape[0], value.shape[1], -1, self.head_v_dim)
        z = z.reshape(z.shape[0], z.shape[1], -1, self.head_v_dim)
        b = b.reshape(b.shape[0], b.shape[1], self.num_v_heads)
        a = a.reshape(a.shape[0], a.shape[1], self.num_v_heads)
        return query, key, value, z, b, a

    def _depthwise_causal_conv(
        self,
        mixed_qkv: torch.Tensor,
        cache_params: QwenTritonCache | None,
    ) -> torch.Tensor:
        batch_size, conv_dim, seq_len = mixed_qkv.shape
        raw_mixed_qkv = mixed_qkv
        prev_state = None if cache_params is None else cache_params.conv_states[self.layer_idx]
        if prev_state is not None:
            conv_input = torch.cat((prev_state, raw_mixed_qkv), dim=-1)
        else:
            conv_input = F.pad(raw_mixed_qkv, (self.conv_kernel_size - 1, 0))

        mixed_qkv = F.conv1d(conv_input, self.conv1d.weight, self.conv1d.bias, groups=self.conv_dim)
        mixed_qkv = F.silu(mixed_qkv[:, :, -seq_len:])

        if cache_params is not None:
            raw_input = raw_mixed_qkv.new_zeros(batch_size, conv_dim, 0)
            if prev_state is not None:
                raw_input = prev_state
            raw_input = torch.cat((raw_input, raw_mixed_qkv), dim=-1)
            cache_params.conv_states[self.layer_idx] = raw_input[:, :, -(self.conv_kernel_size - 1):]
        return mixed_qkv

    def forward(
        self,
        hidden_states: torch.Tensor,
        cache_params: QwenTritonCache | None = None,
        cache_position: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del cache_position
        hidden_states = apply_mask_to_padding_states(hidden_states, attention_mask)
        batch_size, seq_len, _ = hidden_states.shape
        recurrent_state = None if cache_params is None else cache_params.recurrent_states[self.layer_idx]

        projected_qkvz = self.in_proj_qkvz(hidden_states)
        projected_ba = self.in_proj_ba(hidden_states)
        query, key, value, z, b, a = self.fix_query_key_value_ordering(projected_qkvz, projected_ba)
        query, key, value = (tensor.reshape(tensor.shape[0], tensor.shape[1], -1) for tensor in (query, key, value))

        mixed_qkv = torch.cat((query, key, value), dim=-1).transpose(1, 2)
        mixed_qkv = self._depthwise_causal_conv(mixed_qkv, cache_params).transpose(1, 2)
        query, key, value = torch.split(mixed_qkv, [self.key_dim, self.key_dim, self.value_dim], dim=-1)
        query = query.reshape(batch_size, seq_len, -1, self.head_k_dim)
        key = key.reshape(batch_size, seq_len, -1, self.head_k_dim)
        value = value.reshape(batch_size, seq_len, -1, self.head_v_dim)

        beta = b.sigmoid()
        decay = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)
        if self.num_v_heads // self.num_k_heads > 1:
            repeat_factor = self.num_v_heads // self.num_k_heads
            query = query.repeat_interleave(repeat_factor, dim=2)
            key = key.repeat_interleave(repeat_factor, dim=2)

        query = l2norm(query)
        key = l2norm(key)
        core_attn_out, last_recurrent_state = gated_delta_rule_sequence(
            query=query,
            key=key,
            value=value,
            decay=decay,
            beta=beta,
            initial_state=recurrent_state,
        )

        if cache_params is not None:
            cache_params.recurrent_states[self.layer_idx] = last_recurrent_state

        z_shape = z.shape
        core_attn_out = self.norm(core_attn_out.reshape(-1, self.head_v_dim), z.reshape(-1, self.head_v_dim))
        core_attn_out = core_attn_out.reshape(z_shape).reshape(batch_size, seq_len, -1)
        return self.out_proj(core_attn_out)
