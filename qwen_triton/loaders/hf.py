from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

import torch
from huggingface_hub import snapshot_download
from safetensors import safe_open
from transformers import AutoConfig
from transformers.configuration_utils import PretrainedConfig


def ensure_local_model_path(model_id_or_path: str, include_weights: bool = True) -> Path:
    path = Path(model_id_or_path)
    if path.exists():
        return path

    allow_patterns = [
        "config.json",
        "generation_config.json",
        "tokenizer*",
        "special_tokens_map.json",
        "tokenizer_config.json",
        "merges.txt",
        "vocab.json",
        "*.model",
    ]
    if include_weights:
        allow_patterns.extend(["*.safetensors", "*.safetensors.index.json"])

    download_kwargs = {
        "repo_id": model_id_or_path,
        "allow_patterns": allow_patterns,
        "ignore_patterns": ["*.bin", "*.msgpack", "*.h5", "*.ot", "*.onnx"],
    }
    try:
        snapshot_path = snapshot_download(local_files_only=True, **download_kwargs)
    except Exception:
        snapshot_path = snapshot_download(**download_kwargs)
    return Path(snapshot_path)


def load_config_dict(model_id_or_path: str) -> dict:
    path = Path(model_id_or_path)
    if path.exists():
        config_path = path / "config.json" if path.is_dir() else path
        with config_path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    try:
        cfg = AutoConfig.from_pretrained(model_id_or_path, trust_remote_code=True)
        return cfg.to_dict()
    except Exception:
        raw_cfg, _ = PretrainedConfig.get_config_dict(model_id_or_path, trust_remote_code=True)
        return raw_cfg


def _iter_safetensor_files(snapshot_path: Path) -> list[Path]:
    index_path = snapshot_path / "model.safetensors.index.json"
    if index_path.exists():
        with index_path.open("r", encoding="utf-8") as handle:
            weight_map = json.load(handle)["weight_map"]
        files = sorted({snapshot_path / filename for filename in weight_map.values()})
        return files

    files = sorted(snapshot_path.glob("*.safetensors"))
    if files:
        return files
    raise FileNotFoundError(f"No safetensors files found under {snapshot_path}")


def iter_safetensor_tensors(snapshot_path: Path) -> Iterator[tuple[str, torch.Tensor]]:
    for tensor_file in _iter_safetensor_files(snapshot_path):
        with safe_open(str(tensor_file), framework="pt", device="cpu") as handle:
            for key in handle.keys():
                yield key, handle.get_tensor(key)


def _candidate_target_keys(source_key: str, target_keys: set[str]) -> list[str]:
    candidates = [source_key]
    prefixes = [
        "language_model.",
        "model.language_model.",
        "model.language_model.model.",
        "model.model.",
    ]
    for prefix in prefixes:
        if source_key.startswith(prefix):
            stripped = source_key[len(prefix):]
            candidates.append(stripped)
            candidates.append(f"model.{stripped}")
    if source_key.startswith("model.") and source_key[6:] in target_keys:
        candidates.append(source_key[6:])
    if not source_key.startswith("model."):
        candidates.append(f"model.{source_key}")
    return candidates


def _normalize_target_key(source_key: str, target_keys: set[str]) -> str | None:
    for candidate in _candidate_target_keys(source_key, target_keys):
        if candidate in target_keys:
            return candidate
    return None


def load_hf_weights_into_model(
    model: torch.nn.Module,
    snapshot_path: Path,
    strict: bool = False,
) -> dict[str, list[str]]:
    target_tensors = dict(model.named_parameters())
    target_tensors.update(dict(model.named_buffers()))
    target_keys = set(target_tensors)
    loaded: set[str] = set()
    skipped: list[str] = []
    mismatched: list[str] = []

    with torch.no_grad():
        for source_key, tensor in iter_safetensor_tensors(snapshot_path):
            target_key = _normalize_target_key(source_key, target_keys)
            if target_key is None:
                skipped.append(source_key)
                continue

            target = target_tensors[target_key]
            if tuple(target.shape) != tuple(tensor.shape):
                mismatched.append(f"{source_key} -> {target_key}: {tuple(tensor.shape)} != {tuple(target.shape)}")
                continue

            target.copy_(tensor.to(device=target.device, dtype=target.dtype))
            loaded.add(target_key)

    missing = sorted(target_keys - loaded)
    if strict and missing:
        raise RuntimeError(f"Missing keys after HF load: {missing[:20]}")
    return {
        "loaded": sorted(loaded),
        "missing": missing,
        "skipped": skipped,
        "mismatched": mismatched,
    }
