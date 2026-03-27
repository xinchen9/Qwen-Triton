from __future__ import annotations

import argparse

from qwen_triton.ops import get_rope_cuda_op_error, load_rope_cuda_op


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build/load the Qwen-Triton CUDA RoPE custom operator")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if not load_rope_cuda_op(verbose=args.verbose):
        error = get_rope_cuda_op_error()
        raise SystemExit(f"Failed to build/load qwen_triton::rope_tensor_forward: {error}")
    print("loaded=qwen_triton::rope_tensor_forward")


if __name__ == "__main__":
    main()
