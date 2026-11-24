# ruff: noqa: E501
"""Writes notebooks/09_adapt_export_qwen3.ipynb (Stage 3, chunk 8: re-export the adapted model).

Run: python notebooks/make_09_notebook.py [COMMIT]   (COMMIT must be pushed to GitHub and contain
src/zubaan/export/check.py)

The install, machine check and llama.cpp cells are notebook 03's, which ran on Kaggle; they are
imported from make_03_notebook.py, and only what is new lives here.
"""

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "make_03_notebook", Path(__file__).parent / "make_03_notebook.py"
)
n3 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(n3)
md, code = n3.md, n3.code

# Notebook 03's self-check and build cover four tools; the quantization check also needs
# llama-perplexity.
LLAMA_SELFCHECK = n3.LLAMA_SELFCHECK.replace(
    '("llama-quantize", "llama-imatrix", "llama-tokenize", "llama-completion")',
    '("llama-quantize", "llama-imatrix", "llama-tokenize", "llama-completion", "llama-perplexity")',
).replace(
    "--target llama-quantize llama-imatrix llama-tokenize llama-completion",
    "--target llama-quantize llama-imatrix llama-tokenize llama-completion llama-perplexity",
)
assert LLAMA_SELFCHECK.count("llama-perplexity") == 2, "notebook 03's self-check cell changed"

CHECK_CODE = n3.CHECK_CODE.replace(
    "import zubaan.export.parity  # noqa: F401",
    "import zubaan.export.check  # noqa: F401  (export check is in this commit)\n"
    "import zubaan.export.parity  # noqa: F401",
)
assert "zubaan.export.check" in CHECK_CODE, "notebook 03's check cell changed"

FIND_MODEL = """
import json
from pathlib import Path

# The adapted model is the output of notebook 07 (Add Input > Notebooks). Its zubaan_extension.json
# says trained: true, which tells it apart from the untrained extended model of notebook 03.
markers = sorted(Path("/kaggle/input").rglob("zubaan_extension.json"))
trained = [m.parent for m in markers if json.loads(m.read_text()).get("trained")]
print("trained model folders:", [str(f) for f in trained])
assert len(trained) == 1, f"need exactly one trained model folder, found {len(trained)}: add notebook 07's output"
MODEL_DIR = trained[0]
for f in ("tokenizer.json", "tokenizer_config.json", "config.json"):
    assert (MODEL_DIR / f).exists(), f"no {f} in {MODEL_DIR}"
assert (MODEL_DIR / "model.safetensors").exists() or list(MODEL_DIR.glob("model-*.safetensors")), "no weights"
meta = json.loads((MODEL_DIR / "zubaan_extension.json").read_text())
print(json.dumps(meta, indent=1)[:2000])
has_template = "chat_template" in (MODEL_DIR / "tokenizer_config.json").read_text() or (MODEL_DIR / "chat_template.jinja").exists()
assert has_template, "no chat template: Ollama and vLLM chat would fail"
print("OK: adapted model found:", MODEL_DIR)
"""

SETTINGS = """
BASE_MODEL = "Qwen/Qwen3-1.7B"   # tokenizer reference for the pre-tokenizer and fingerprint checks
NAME = "qwen3-1.7b-urdu-adapted"
"""

CONFIG = """
import os

import yaml

WORK = Path("/kaggle/working")
config = {
    "urdu_corpus": {"raw_dir": "data/raw", "out_dir": "data/urdu", "flores": "data/flores_urd.txt", "max_mb": 100, "heldout_percent": 1.0},
    "export": {
        "model_dir": str(MODEL_DIR),
        "base_model": BASE_MODEL,
        "output_dir": str(WORK / "export"),
        "name": NAME,
        "llama_cpp_dir": str(LLAMA_DIR),
        "llama_cpp_bin": str(BIN_DIR),
        "quants": ["Q4_K_M", "Q8_0"],
        # Importance matrix on Urdu training text (same as Stage 2). The quantization cost below is
        # measured on FLORES, which the model never trained on.
        "calibration": "data/urdu/urdu_train.txt",
        "calibration_chars": 200000,
        "imatrix_chunks": 40,
        "threads": os.cpu_count(),
        "parity_urdu": "data/flores_urd.txt",
        "parity_english": "data/flores_eng.txt",
        "parity_workers": os.cpu_count(),
        "max_model_len": 4096,
        "check_urdu": "data/flores_urd.txt",
        "check_english": "data/flores_eng.txt",
        "check_chunks": 20,
        "check_ctx": 512,
        "check_n_predict": 80,
        # The f16 logits file is about 3 GB per language: keep it out of /kaggle/working.
        "check_scratch": "/tmp/zubaan_check",
    },
}
Path("export.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
print(Path("export.yaml").read_text())
"""

SUMMARY = """
summary = json.loads(Path("/kaggle/working/export/export_summary.json").read_text())
g = summary["gguf"]
p = g["parity"]
print("llama.cpp", g["llama_cpp_tag"], "| pre-tokenizer", g["pre_tokenizer"], "| converter patched:", g["converter_patched"])
print(f"parity: {p['n_texts'] - p['n_mismatch']}/{p['n_texts']} identical, {p['n_nfc_only']} differ only because the text is not NFC, {p['n_real_mismatch']} real mismatches")
assert p["passed"] and p["n_real_mismatch"] == 0, "tokenizer parity failed: see export_summary.json"
assert {"Q4_K_M", "Q8_0"} <= set(g["quantized"]), "a quantized file is missing"
# Stage 2 saw 2,041 of 2,045 identical (4 NFC-only). Training does not touch the tokenizer, so the
# same counts are expected here.
print("OK: GGUF export of the adapted model passed tokenizer parity")
"""

SHOW = """
check = json.loads(Path("/kaggle/working/export/export_check.json").read_text())
for model, samples in check["samples"].items():
    print("=" * 30, model)
    for s in samples:
        share = s["answer_urdu_letter_share"]
        speed = s["tokens_per_s"]
        print(f"--- {s['prompt']} | think block: {s['has_think_block']} | Urdu letters: {share if share is None else round(share, 2)} | {speed} tok/s")
        print(s["input"])
        print(s["text"])
print()
for lang, res in check["quant_cost"].items():
    print(lang, "f16 PPL", round(res["f16_ppl"]["value"], 3))
    for q, k in res["quants"].items():
        print(f"  {q}: mean KLD {k['mean_kld']['value']:.4f} ± {k['mean_kld']['uncertainty']:.4f}, "
              f"same top token {k['same_top_p_percent']['value']:.1f}%, PPL ratio {k['ppl_ratio']['value']:.4f}")
"""

READABLE = """
# CHECK 4 (success check 5): the Q4_K_M answers Urdu prompts in Urdu script. This only catches drift
# into another script; whether the Urdu reads well is for a person to judge from the text above.
q4 = {s["prompt"]: s for s in check["samples"]["Q4_K_M"]}
for name in ("urdu_completion_capital", "urdu_completion_lahore", "urdu_chat_capital", "urdu_chat_sentences"):
    share = q4[name]["answer_urdu_letter_share"]
    print(name, share)
    assert share is not None and share > 0.8, f"{name}: the Q4_K_M answer is not mostly Urdu script"
print("OK: Q4_K_M answers Urdu prompts in Urdu script. Now read the samples.")
"""

CLEANUP = """
# Kaggle keeps /kaggle/working as output (limit 20 GB). The f16 GGUF was needed for the parity and
# quantization checks; it is not a deliverable, so remove it.
WORK = Path("/kaggle/working")
for f in (WORK / "export").glob("*-F16.gguf"):
    print("removing", f.name, f"{f.stat().st_size / 1e9:.1f} GB")
    f.unlink()
total = 0
for f in sorted(WORK.rglob("*")):
    if f.is_file() and f.stat().st_size > 5e6:
        total += f.stat().st_size
        print(f"{f.stat().st_size / 1e9:6.2f} GB  {f.relative_to(WORK)}")
print(f"total of files over 5 MB: {total / 1e9:.1f} GB")
assert total < 19e9, "output is close to Kaggle's 20 GB limit"
"""


def build(commit: str) -> dict:
    cells = [
        md(
            f"""
# Stage 3: re-export the adapted model (Qwen3-1.7B + 10,000 trained Urdu tokens)

Converts the adapted model from notebook 07 to GGUF, **proves tokenizer parity** (Hugging Face and llama.cpp give identical ids on every FLORES sentence and a stress set; the export stops otherwise), builds Q4_K_M and Q8_0 with the Urdu importance matrix, writes the Ollama `Modelfile` and vLLM config, then runs `zubaan export check`:

- **Generation samples** on f16, Q8_0 and Q4_K_M: Urdu completion, Urdu chat (thinking off), Urdu chat with thinking allowed (to see the `<think>` block), and an English control. Greedy decoding.
- **Quantization cost**: KL divergence of Q8_0 and Q4_K_M against f16 on FLORES Urdu and FLORES English (same tokenizer, so per-token numbers are comparable between the files).

Settings before running:

- Accelerator: **None** (CPU). The GPU is not used: this notebook costs no GPU hours.
- Internet: **on**. No secret is needed.
- Input: **Add Input > Notebooks**: the output of `07_adapt_main_qwen3` (it holds `adapted/model/`). Do not add notebook 03's output; this notebook needs exactly one trained model.

Needs about 12 GB RAM and 20 GB disk. Expected time about 2 hours on 4 CPUs: conversion 5 min, parity 10 to 15, importance matrix 15 to 25, quantizing 5, generation samples 20 to 30, quantization cost 25 to 35.

**Before the first run:** the commit in the install cell must be pushed to GitHub. Use *Save Version > Save & Run All (Commit)*; finished files are skipped if you run it again.

**Download afterwards:** `export/export_summary.json`, `export/export_check.json`, `export/{{NAME}}-Q4_K_M.gguf` (about 1.2 GB, for the local Ollama check) and `export/Modelfile`.

llama.cpp release `{n3.LLAMA_CPP_TAG}`, as in Stage 2.
"""
        ),
        code(n3.INSTALL.format(commit=commit)),
        code(CHECK_CODE),
        code(n3.CHECK_MACHINE),
        code(SETTINGS),
        md("## 1. Find the adapted model"),
        code(FIND_MODEL),
        md("## 2. llama.cpp"),
        code(n3.LLAMA_CPP),
        code(LLAMA_SELFCHECK),
        md(
            "## 3. Data and config\n\nFLORES-200 (parity and quantization-cost texts) and Urdu Wikipedia (importance matrix)."
        ),
        code(CONFIG),
        code("!zubaan data flores --out data\n!zubaan data urdu --config export.yaml"),
        md("## 4. Export with the parity check"),
        code("!zubaan export gguf --config export.yaml"),
        code(
            "!zubaan export ollama --config export.yaml\n!zubaan export vllm --config export.yaml"
        ),
        code(SUMMARY),
        md("## 5. Generation samples and quantization cost"),
        code("!zubaan export check --config export.yaml"),
        code(SHOW),
        code(READABLE),
        md("## 6. Clean up and list the output"),
        code(CLEANUP),
    ]
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main() -> None:
    commit = (
        sys.argv[1]
        if len(sys.argv) > 1
        else subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    )
    out = Path(__file__).parent / "09_adapt_export_qwen3.ipynb"
    nb = build(commit)
    for i, cell in enumerate(nb["cells"]):
        cell["id"] = f"cell-{i}"
    out.write_text(
        json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n"
    )
    print(f"wrote {out} pinned to commit {commit}")


if __name__ == "__main__":
    main()
