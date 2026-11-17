"""Generate a vLLM launch config for an extended model.

T4 and P100 GPUs have no bfloat16, so the dtype is always ``half``. The extended tokenizer
lives in the model folder, so the same path is passed for ``--tokenizer``.
"""

from __future__ import annotations

import shlex
from pathlib import Path


def vllm_options(
    served_name: str,
    max_model_len: int = 4096,
    gpu_memory_utilization: float = 0.9,
    port: int = 8000,
) -> dict[str, object]:
    """Options for ``vllm serve`` as ``{flag name without dashes: value}``."""
    if max_model_len < 256:
        raise ValueError("max_model_len must be at least 256")
    if not 0.1 <= gpu_memory_utilization <= 0.98:
        raise ValueError("gpu_memory_utilization must be between 0.1 and 0.98")
    return {
        "dtype": "half",
        "max-model-len": max_model_len,
        "gpu-memory-utilization": gpu_memory_utilization,
        "served-model-name": served_name,
        "port": port,
    }


def config_yaml(options: dict[str, object]) -> str:
    """YAML for ``vllm serve --config`` (one ``flag: value`` per line)."""
    return "".join(f"{k}: {v}\n" for k, v in options.items())


def serve_command(model_dir: str, config_file: str) -> str:
    """The shell command that starts the server."""
    return (
        f"vllm serve {shlex.quote(model_dir)} --tokenizer {shlex.quote(model_dir)} "
        f"--config {shlex.quote(config_file)}"
    )


def check_no_bfloat16(*texts: str) -> None:
    """Raise if any generated text mentions bfloat16 (unsupported on T4 and P100)."""
    for t in texts:
        if "bfloat16" in t.lower() or "bf16" in t.lower():
            raise ValueError("generated vLLM config must not use bfloat16")


def write_vllm_files(
    out_dir: Path, model_dir: str, served_name: str, max_model_len: int = 4096
) -> dict[str, Path]:
    """Write ``vllm.yaml`` and ``serve_vllm.sh`` into ``out_dir``."""
    cfg = config_yaml(vllm_options(served_name, max_model_len))
    script = "#!/bin/sh\n" + serve_command(model_dir, "vllm.yaml") + "\n"
    check_no_bfloat16(cfg, script)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {"config": out_dir / "vllm.yaml", "script": out_dir / "serve_vllm.sh"}
    paths["config"].write_text(cfg, encoding="utf-8", newline="\n")
    paths["script"].write_text(script, encoding="utf-8", newline="\n")
    return paths
