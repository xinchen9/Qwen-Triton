from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch

from qwen_triton.models import QwenTritonForCausalLM
from qwen_triton.scripts.wikitext_workload import build_batches, clone_batches_to_device, prepare_tokenizer, save_metrics


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Short Wikitext smoke training for Qwen-Triton")
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--backend", default="triton", choices=["triton", "ref"])
    parser.add_argument("--dataset", default="wikitext-2-raw-v1")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", default="bf16")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--train-steps", type=int, default=1)
    parser.add_argument("--eval-batches", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--output-dir", default="artifacts/wikitext_smoke")
    parser.add_argument("--strict-triton", action="store_true")
    parser.add_argument("--max-texts", type=int, default=512)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.strict_triton:
        os.environ["QWEN_TRITON_STRICT"] = "1"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = prepare_tokenizer(args.model_id)

    model = QwenTritonForCausalLM.from_pretrained_hf(
        args.model_id,
        backend=args.backend,
        device=args.device,
        dtype=args.dtype,
    )
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    train_batches_cpu, eval_batches_cpu = build_batches(
        tokenizer=tokenizer,
        dataset_config=args.dataset,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        train_example_count=args.batch_size * args.train_steps,
        eval_example_count=args.batch_size * args.eval_batches,
        max_texts=args.max_texts,
    )
    train_batches = clone_batches_to_device(train_batches_cpu, args.device)
    eval_batches = clone_batches_to_device(eval_batches_cpu, args.device)

    for step, batch in enumerate(train_batches[: args.train_steps], start=1):
        optimizer.zero_grad(set_to_none=True)
        outputs = model(
            input_ids=batch.input_ids,
            attention_mask=batch.attention_mask,
            labels=batch.labels,
            use_cache=False,
        )
        if outputs.loss is None or not torch.isfinite(outputs.loss):
            raise RuntimeError("Training loss is non-finite")
        outputs.loss.backward()
        optimizer.step()
        print(f"train_step={step} loss={float(outputs.loss.detach()):.6f}")

    model.eval()
    with torch.no_grad():
        for idx, batch in enumerate(eval_batches[: args.eval_batches], start=1):
            outputs = model(
                input_ids=batch.input_ids,
                attention_mask=batch.attention_mask,
                labels=batch.labels,
                use_cache=False,
            )
            if outputs.loss is None or not torch.isfinite(outputs.loss):
                raise RuntimeError("Eval loss is non-finite")
            print(f"eval_batch={idx} loss={float(outputs.loss.detach()):.6f}")

    checkpoint_path = output_dir / "checkpoint.pt"
    torch.save({"state_dict": model.state_dict(), "config": model.config.to_dict()}, checkpoint_path)

    reloaded = QwenTritonForCausalLM.from_config(model.config)
    reloaded.load_state_dict(torch.load(checkpoint_path, map_location="cpu")["state_dict"], strict=False)
    save_metrics(output_dir / "config.json", model.config.to_dict())
    print(f"saved_checkpoint={checkpoint_path}")


if __name__ == "__main__":
    main()
