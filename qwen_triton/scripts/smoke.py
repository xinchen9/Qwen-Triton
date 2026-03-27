from __future__ import annotations

import argparse
import os

import torch
from transformers import AutoTokenizer

from qwen_triton.models import QwenTritonForCausalLM


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Qwen-Triton smoke test")
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--backend", default="triton", choices=["triton", "ref"])
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", default="bf16")
    parser.add_argument("--prompt", default="Write one sentence about Triton kernels.")
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--compare-ref", action="store_true")
    parser.add_argument("--strict-triton", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.strict_triton:
        os.environ["QWEN_TRITON_STRICT"] = "1"
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = QwenTritonForCausalLM.from_pretrained_hf(
        args.model_id,
        backend=args.backend,
        device=args.device,
        dtype=args.dtype,
    ).eval()

    encoded = tokenizer(args.prompt, return_tensors="pt")
    encoded = {key: value.to(args.device) for key, value in encoded.items()}

    with torch.no_grad():
        outputs = model(**encoded, use_cache=True)
        next_token = outputs.logits[:, -1].argmax(dim=-1)
        generated = model.greedy_generate(
            input_ids=encoded["input_ids"],
            attention_mask=encoded.get("attention_mask"),
            max_new_tokens=args.max_new_tokens,
        )

    generated_tokens = generated[0].detach().cpu().to(torch.long).tolist()
    print(f"backend={args.backend}")
    print(f"next_token={next_token.tolist()}")
    print(tokenizer.decode(generated_tokens, skip_special_tokens=True))

    if args.compare_ref and args.backend == "triton":
        ref_model = QwenTritonForCausalLM.from_pretrained_hf(
            args.model_id,
            backend="ref",
            device=args.device,
            dtype=args.dtype,
        ).eval()
        with torch.no_grad():
            ref_outputs = ref_model(**encoded, use_cache=True)
        diff = (outputs.logits.float() - ref_outputs.logits.float()).abs()
        print(f"logit_max_abs_diff={float(diff.max()):.6f}")
        print(f"logit_mean_abs_diff={float(diff.mean()):.6f}")


if __name__ == "__main__":
    main()
