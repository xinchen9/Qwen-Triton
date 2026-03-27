from __future__ import annotations

import json
import math
import random
import time
import gc
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from datasets import load_dataset
from transformers import AutoTokenizer

from qwen_triton.models import QwenTritonForCausalLM


@dataclass
class Batch:
    input_ids: torch.Tensor
    labels: torch.Tensor
    attention_mask: torch.Tensor


@dataclass
class BackendMetrics:
    backend: str
    model_id: str
    dataset: str
    batch_size: int
    seq_len: int
    train_steps: int
    warmup_steps: int
    eval_batches: int
    load_time_s: float
    train_time_s: float
    train_step_time_mean_s: float
    train_tokens_per_s: float
    train_loss_first: float
    train_loss_last: float
    train_loss_mean: float
    eval_time_s: float
    eval_loss_mean: float
    eval_perplexity: float
    eval_token_accuracy: float
    peak_memory_gb: float | None
    total_time_s: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def prepare_tokenizer(model_id: str) -> AutoTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def _build_examples(
    tokenizer: AutoTokenizer,
    dataset_config: str,
    seq_len: int,
    split: str,
    limit: int,
    max_texts: int,
) -> list[torch.Tensor]:
    ds = load_dataset("wikitext", dataset_config, split=split)
    texts = [text for text in ds["text"] if text and not text.isspace()]
    text = "\n\n".join(texts[:max_texts])
    token_ids = tokenizer(text, return_tensors="pt")["input_ids"][0]
    window = seq_len + 1
    examples: list[torch.Tensor] = []
    for start in range(0, max(token_ids.numel() - window + 1, 1), seq_len):
        chunk = token_ids[start : start + window]
        if chunk.numel() == window:
            examples.append(chunk)
        if len(examples) >= limit:
            break
    return examples


def build_batches(
    tokenizer: AutoTokenizer,
    dataset_config: str,
    seq_len: int,
    batch_size: int,
    train_example_count: int,
    eval_example_count: int,
    max_texts: int = 512,
) -> tuple[list[Batch], list[Batch]]:
    train_examples = _build_examples(
        tokenizer=tokenizer,
        dataset_config=dataset_config,
        seq_len=seq_len,
        split="train[:1%]",
        limit=train_example_count,
        max_texts=max_texts,
    )
    eval_examples = _build_examples(
        tokenizer=tokenizer,
        dataset_config=dataset_config,
        seq_len=seq_len,
        split="validation[:1%]",
        limit=eval_example_count,
        max_texts=max_texts,
    )
    return _pack_batches(train_examples, batch_size), _pack_batches(eval_examples, batch_size)


def _pack_batches(examples: list[torch.Tensor], batch_size: int) -> list[Batch]:
    batches: list[Batch] = []
    for start in range(0, len(examples), batch_size):
        group = examples[start : start + batch_size]
        if len(group) != batch_size:
            break
        batch = torch.stack(group, dim=0)
        input_ids = batch[:, :-1].contiguous()
        batches.append(
            Batch(
                input_ids=input_ids,
                labels=input_ids.clone(),
                attention_mask=torch.ones_like(input_ids),
            )
        )
    return batches


def clone_batches_to_device(batches: list[Batch], device: str | torch.device) -> list[Batch]:
    return [
        Batch(
            input_ids=batch.input_ids.to(device),
            labels=batch.labels.to(device),
            attention_mask=batch.attention_mask.to(device),
        )
        for batch in batches
    ]


def _maybe_cuda_sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _peak_memory_gb(device: torch.device) -> float | None:
    if device.type != "cuda":
        return None
    return torch.cuda.max_memory_allocated(device) / (1024**3)


def _evaluate_batches(
    model: QwenTritonForCausalLM,
    batches: list[Batch],
    device: torch.device,
) -> tuple[float, float, float]:
    if not batches:
        return 0.0, 0.0, 0.0

    total_loss = 0.0
    total_correct = 0
    total_tokens = 0
    total_time = 0.0

    model.eval()
    with torch.no_grad():
        for batch in batches:
            _maybe_cuda_sync(device)
            start = time.perf_counter()
            outputs = model(
                input_ids=batch.input_ids,
                attention_mask=batch.attention_mask,
                labels=batch.labels,
                use_cache=False,
            )
            _maybe_cuda_sync(device)
            total_time += time.perf_counter() - start
            if outputs.loss is None or not torch.isfinite(outputs.loss):
                raise RuntimeError("Eval loss is non-finite")
            total_loss += float(outputs.loss.detach())
            preds = outputs.logits[:, :-1].argmax(dim=-1)
            gold = batch.labels[:, 1:]
            total_correct += int((preds == gold).sum().item())
            total_tokens += int(gold.numel())
    mean_loss = total_loss / len(batches)
    token_accuracy = total_correct / total_tokens if total_tokens else 0.0
    return total_time, mean_loss, token_accuracy


def run_backend_workload(
    *,
    model_id: str,
    backend: str,
    dataset: str,
    train_batches: list[Batch],
    eval_batches: list[Batch],
    device: str | torch.device,
    dtype: str,
    lr: float,
    warmup_steps: int,
    seed: int,
) -> BackendMetrics:
    resolved_device = torch.device(device)
    set_seed(seed)

    if resolved_device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(resolved_device)

    total_start = time.perf_counter()
    load_start = time.perf_counter()
    model = QwenTritonForCausalLM.from_pretrained_hf(
        model_id,
        backend=backend,
        device=resolved_device,
        dtype=dtype,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    load_time = time.perf_counter() - load_start

    train_losses: list[float] = []
    measured_train_times: list[float] = []
    measured_train_tokens = 0

    model.train()
    for step_idx, batch in enumerate(train_batches):
        optimizer.zero_grad(set_to_none=True)
        _maybe_cuda_sync(resolved_device)
        start = time.perf_counter()
        outputs = model(
            input_ids=batch.input_ids,
            attention_mask=batch.attention_mask,
            labels=batch.labels,
            use_cache=False,
        )
        if outputs.loss is None or not torch.isfinite(outputs.loss):
            raise RuntimeError(f"Training loss is non-finite for backend={backend} at step={step_idx + 1}")
        outputs.loss.backward()
        optimizer.step()
        _maybe_cuda_sync(resolved_device)
        elapsed = time.perf_counter() - start
        loss_value = float(outputs.loss.detach())
        train_losses.append(loss_value)
        if step_idx >= warmup_steps:
            measured_train_times.append(elapsed)
            measured_train_tokens += int(batch.input_ids.numel())

    eval_time, eval_loss, eval_token_accuracy = _evaluate_batches(model, eval_batches, resolved_device)
    total_time = time.perf_counter() - total_start
    train_time = sum(measured_train_times)
    peak_memory_gb = _peak_memory_gb(resolved_device)
    perplexity = math.exp(eval_loss) if eval_loss < 20 else float("inf")

    metrics = BackendMetrics(
        backend=backend,
        model_id=model_id,
        dataset=dataset,
        batch_size=int(train_batches[0].input_ids.shape[0]) if train_batches else 0,
        seq_len=int(train_batches[0].input_ids.shape[1]) if train_batches else 0,
        train_steps=max(len(train_batches) - warmup_steps, 0),
        warmup_steps=warmup_steps,
        eval_batches=len(eval_batches),
        load_time_s=load_time,
        train_time_s=train_time,
        train_step_time_mean_s=(train_time / len(measured_train_times)) if measured_train_times else 0.0,
        train_tokens_per_s=(measured_train_tokens / train_time) if train_time > 0 else 0.0,
        train_loss_first=train_losses[0] if train_losses else 0.0,
        train_loss_last=train_losses[-1] if train_losses else 0.0,
        train_loss_mean=(sum(train_losses) / len(train_losses)) if train_losses else 0.0,
        eval_time_s=eval_time,
        eval_loss_mean=eval_loss,
        eval_perplexity=perplexity,
        eval_token_accuracy=eval_token_accuracy,
        peak_memory_gb=peak_memory_gb,
        total_time_s=total_time,
    )
    del optimizer
    del model
    gc.collect()
    if resolved_device.type == "cuda":
        torch.cuda.empty_cache()
    return metrics


def save_metrics(path: str | Path, payload: dict[str, Any]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
