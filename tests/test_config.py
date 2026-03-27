from qwen_triton.configs import QwenTritonConfig


def test_qwen35_text_config_normalization() -> None:
    raw = {
        "model_type": "qwen3_5_moe",
        "text_config": {
            "model_type": "qwen3_5_moe_text",
            "vocab_size": 32000,
            "hidden_size": 128,
            "intermediate_size": 256,
            "num_hidden_layers": 4,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 32,
            "max_position_embeddings": 4096,
            "rms_norm_eps": 1e-6,
            "attn_output_gate": True,
            "partial_rotary_factor": 0.25,
            "linear_conv_kernel_dim": 4,
            "linear_key_head_dim": 16,
            "linear_value_head_dim": 16,
            "linear_num_key_heads": 4,
            "linear_num_value_heads": 4,
            "num_experts": 8,
            "num_experts_per_tok": 2,
            "moe_intermediate_size": 64,
            "shared_expert_intermediate_size": 64,
            "layer_types": ["linear_attention", "full_attention", "linear_attention", "full_attention"],
        },
    }
    config = QwenTritonConfig.from_hf_config(raw)
    assert config.family == "qwen35_text_moe"
    assert config.norm_one_plus is True
    assert config.rotary_dim == 8
    assert config.layer_types[0] == "linear_attention"


def test_qwen3_dense_normalization() -> None:
    raw = {
        "model_type": "qwen3",
        "vocab_size": 32000,
        "hidden_size": 128,
        "intermediate_size": 256,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 32,
    }
    config = QwenTritonConfig.from_hf_config(raw)
    assert config.family == "qwen3_dense"
    assert config.norm_one_plus is False
    assert config.layer_types == ["full_attention", "full_attention"]
