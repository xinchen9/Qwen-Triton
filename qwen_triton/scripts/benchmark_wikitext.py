from __future__ import annotations

import argparse
from pathlib import Path

from qwen_triton.scripts.wikitext_workload import (
    build_batches,
    clone_batches_to_device,
    prepare_tokenizer,
    run_backend_workload,
    save_metrics,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare Triton and reference Qwen3 fine-tuning on Wikitext")
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--backends", nargs="+", default=["triton", "ref"], choices=["triton", "ref"])
    parser.add_argument("--dataset", default="wikitext-2-raw-v1")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", default="bf16")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--train-steps", type=int, default=8)
    parser.add_argument("--warmup-steps", type=int, default=1)
    parser.add_argument("--eval-batches", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-texts", type=int, default=512)
    parser.add_argument("--output-dir", default="artifacts/benchmarks/wikitext_compare")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = prepare_tokenizer(args.model_id)
    train_batches_cpu, eval_batches_cpu = build_batches(
        tokenizer=tokenizer,
        dataset_config=args.dataset,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        train_example_count=args.batch_size * (args.train_steps + args.warmup_steps),
        eval_example_count=args.batch_size * args.eval_batches,
        max_texts=args.max_texts,
    )
    train_batches = clone_batches_to_device(train_batches_cpu, args.device)
    eval_batches = clone_batches_to_device(eval_batches_cpu, args.device)

    results = []
    for backend in args.backends:
        metrics = run_backend_workload(
            model_id=args.model_id,
            backend=backend,
            dataset=args.dataset,
            train_batches=train_batches,
            eval_batches=eval_batches,
            device=args.device,
            dtype=args.dtype,
            lr=args.lr,
            warmup_steps=args.warmup_steps,
            seed=args.seed,
        )
        results.append(metrics.to_dict())
        print(
            f"backend={backend} train_step_time_mean_s={metrics.train_step_time_mean_s:.6f} "
            f"train_tokens_per_s={metrics.train_tokens_per_s:.2f} eval_loss_mean={metrics.eval_loss_mean:.6f} "
            f"eval_token_accuracy={metrics.eval_token_accuracy:.6f} peak_memory_gb={metrics.peak_memory_gb}"
        )

    payload = {
        "model_id": args.model_id,
        "dataset": args.dataset,
        "batch_size": args.batch_size,
        "seq_len": args.seq_len,
        "train_steps": args.train_steps,
        "warmup_steps": args.warmup_steps,
        "eval_batches": args.eval_batches,
        "results": results,
    }
    metrics_path = output_dir / "metrics.json"
    save_metrics(metrics_path, payload)
    print(f"saved_metrics={metrics_path}")


if __name__ == "__main__":
    main()
