from __future__ import annotations

import argparse

import torch

from qwen_triton.models import QwenTritonForCausalLM
from qwen_triton.scripts.wikitext_workload import build_batches, clone_batches_to_device, prepare_tokenizer, set_seed


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Profile a single backend step with NVTX ranges")
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--backend", required=True, choices=["triton", "ref"])
    parser.add_argument("--dataset", default="wikitext-2-raw-v1")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bf16")
    parser.add_argument("--mode", default="train", choices=["train", "eval"])
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--warmup-steps", type=int, default=1)
    parser.add_argument("--profile-steps", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-texts", type=int, default=512)
    return parser.parse_args()


def _nvtx_range_push(label: str, enabled: bool) -> None:
    if enabled:
        torch.cuda.nvtx.range_push(label)


def _nvtx_range_pop(enabled: bool) -> None:
    if enabled:
        torch.cuda.nvtx.range_pop()


def main() -> None:
    args = _parse_args()
    device = torch.device(args.device)
    use_nvtx = device.type == "cuda"
    set_seed(args.seed)

    tokenizer = prepare_tokenizer(args.model_id)
    total_examples = args.batch_size * (args.warmup_steps + args.profile_steps)
    train_batches_cpu, eval_batches_cpu = build_batches(
        tokenizer=tokenizer,
        dataset_config=args.dataset,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        train_example_count=total_examples,
        eval_example_count=max(total_examples, args.batch_size),
        max_texts=args.max_texts,
    )
    batches = clone_batches_to_device(train_batches_cpu if args.mode == "train" else eval_batches_cpu, device)

    model = QwenTritonForCausalLM.from_pretrained_hf(
        args.model_id,
        backend=args.backend,
        device=device,
        dtype=args.dtype,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr) if args.mode == "train" else None
    if args.mode == "train":
        model.train()
    else:
        model.eval()

    for warmup_idx in range(args.warmup_steps):
        batch = batches[warmup_idx]
        _nvtx_range_push(f"{args.backend}_{args.mode}_warmup", use_nvtx)
        if args.mode == "train":
            optimizer.zero_grad(set_to_none=True)
            outputs = model(
                input_ids=batch.input_ids,
                attention_mask=batch.attention_mask,
                labels=batch.labels,
                use_cache=False,
            )
            outputs.loss.backward()
            optimizer.step()
        else:
            with torch.no_grad():
                model(
                    input_ids=batch.input_ids,
                    attention_mask=batch.attention_mask,
                    labels=batch.labels,
                    use_cache=False,
                )
        _nvtx_range_pop(use_nvtx)

    for prof_idx in range(args.profile_steps):
        batch = batches[args.warmup_steps + prof_idx]
        _nvtx_range_push(f"{args.backend}_{args.mode}_profile", use_nvtx)
        if args.mode == "train":
            optimizer.zero_grad(set_to_none=True)
            outputs = model(
                input_ids=batch.input_ids,
                attention_mask=batch.attention_mask,
                labels=batch.labels,
                use_cache=False,
            )
            outputs.loss.backward()
            optimizer.step()
            loss_value = float(outputs.loss.detach())
            print(f"profile_step={prof_idx + 1} loss={loss_value:.6f}")
        else:
            with torch.no_grad():
                outputs = model(
                    input_ids=batch.input_ids,
                    attention_mask=batch.attention_mask,
                    labels=batch.labels,
                    use_cache=False,
                )
            loss_value = float(outputs.loss.detach()) if outputs.loss is not None else 0.0
            print(f"profile_step={prof_idx + 1} loss={loss_value:.6f}")
        _nvtx_range_pop(use_nvtx)

    if device.type == "cuda":
        torch.cuda.synchronize(device)


if __name__ == "__main__":
    main()
