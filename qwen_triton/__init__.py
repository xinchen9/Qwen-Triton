from __future__ import annotations

import os
from pathlib import Path


def _prepend_env_path(var_name: str, path: str) -> None:
    if not Path(path).exists():
        return
    current = os.environ.get(var_name, "")
    parts = [part for part in current.split(":") if part]
    if path in parts:
        return
    os.environ[var_name] = ":".join([path, *parts]) if parts else path


for _candidate in (
    "/usr/lib/x86_64-linux-gnu",
    "/usr/local/cuda/targets/x86_64-linux/lib",
    "/usr/local/cuda-13.0/targets/x86_64-linux/lib",
):
    _prepend_env_path("LIBRARY_PATH", _candidate)
    _prepend_env_path("LD_LIBRARY_PATH", _candidate)


from .configs import QwenTritonConfig
from .models import QwenTritonForCausalLM

__all__ = ["QwenTritonConfig", "QwenTritonForCausalLM"]
