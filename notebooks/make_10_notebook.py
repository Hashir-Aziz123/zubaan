# ruff: noqa: E501
"""Writes notebooks/10_adapt_vllm_qwen3.ipynb (Stage 3, chunk 8: vLLM check of the adapted model).

Run: python notebooks/make_10_notebook.py [COMMIT]   (COMMIT must be pushed to GitHub)

The install, GPU check, vLLM install, server and stop cells are notebook 04's, which ran on Kaggle;
they are imported from make_04_notebook.py, and only what is new lives here.
"""

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "make_04_notebook", Path(__file__).parent / "make_04_notebook.py"
)
n4 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(n4)
md, code = n4.md, n4.code

FIND_MODEL = """
from pathlib import Path

# The adapted model is the output of notebook 07 (Add Input > Notebooks); zubaan_extension.json says
# trained: true. vLLM serves this Hugging Face folder directly (no GGUF).
markers = sorted(Path("/kaggle/input").rglob("zubaan_extension.json"))
trained = [m.parent for m in markers if json.loads(m.read_text()).get("trained")]
print("trained model folders:", [str(f) for f in trained])
assert len(trained) == 1, f"need exactly one trained model folder, found {len(trained)}: add notebook 07's output"
MODEL_DIR = trained[0]
assert (MODEL_DIR / "tokenizer.json").exists(), "no tokenizer.json in the model folder"
assert (MODEL_DIR / "model.safetensors").exists() or list(MODEL_DIR.glob("model-*.safetensors")), "no weights"
has_template = "chat_template" in (MODEL_DIR / "tokenizer_config.json").read_text() or (MODEL_DIR / "chat_template.jinja").exists()
assert has_template, "no chat template: /v1/chat/completions would fail"
print("OK: adapted model:", MODEL_DIR)
"""

SETTINGS = """
NAME = "qwen3-urdu-adapted"
MAX_MODEL_LEN = 4096
"""

TEST = """
import time

from tokenizers import Tokenizer

from zubaan.export.check import strip_think, urdu_letter_share

# CHECK 3: the server uses the EXTENDED tokenizer: vLLM's ids for Urdu equal those of tokenizer.json.
tok = Tokenizer.from_file(str(MODEL_DIR / "tokenizer.json"))
sentences = [
    "پاکستان کا دارالحکومت",
    "پاکستان ایک خوبصورت ملک ہے۔",
    "کیا آپ ٹھیک ہیں؟ جی ہاں!",
    "اردو میں 123 اور English کا مرکب mixed script ٹیسٹ",
    "The quick brown fox jumps over the lazy dog.",
]
mismatches = []
for s in sentences:
    want = tok.encode(s, add_special_tokens=False).ids
    got = get("/tokenize", {"model": NAME, "prompt": s, "add_special_tokens": False})["tokens"]
    if got != want:
        mismatches.append((s, want, got))
print(f"tokenization: {len(sentences) - len(mismatches)}/{len(sentences)} identical to tokenizer.json")
assert not mismatches, mismatches

# CHECK 4: it generates Urdu. Greedy, so the output is reproducible.
samples = {}
t0 = time.perf_counter()
out = get("/v1/completions", {"model": NAME, "prompt": sentences[0], "max_tokens": 60, "temperature": 0})
samples["completion"] = {"seconds": round(time.perf_counter() - t0, 2), "usage": out["usage"], "text": out["choices"][0]["text"]}
question = "پاکستان کا دارالحکومت کیا ہے؟"  # "What is the capital of Pakistan?"
for name, kwargs in [("chat_no_think", {"enable_thinking": False}), ("chat_thinking", {})]:
    body = {"model": NAME, "messages": [{"role": "user", "content": question}], "max_tokens": 200, "temperature": 0}
    if kwargs:
        body["chat_template_kwargs"] = kwargs
    r = get("/v1/chat/completions", body)
    samples[name] = {"usage": r["usage"], "text": r["choices"][0]["message"]["content"] or "",
                     "finish_reason": r["choices"][0]["finish_reason"]}
for name, s in samples.items():
    s["answer_urdu_letter_share"] = urdu_letter_share(strip_think(s["text"]))
    print(f"--- {name} | Urdu letters: {s['answer_urdu_letter_share']}")
    print(s["text"])
assert samples["completion"]["usage"]["completion_tokens"] > 0
assert (samples["chat_no_think"]["answer_urdu_letter_share"] or 0) > 0.8, "the chat answer is not mostly Urdu script"

result = {
    "passed": True,
    "vllm": VLLM_VERSION,
    "gpu": name_gpu,
    "compute_capability": cap,
    "dtype": "half",
    "model_dir": str(MODEL_DIR),
    "tokenization_identical": len(sentences),
    "samples": samples,
}
(WORK / "vllm_check_adapted.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))
print("OK: vLLM served the adapted model")
"""


def build(commit: str) -> dict:
    # Notebook 04's GPU cell names the GPU ``name``; the test cell uses ``name`` as a loop variable,
    # so keep the GPU name under its own name.
    check_gpu = n4.CHECK_GPU + "\nname_gpu = name\n"
    cells = [
        md(
            """
# Stage 3: serve the adapted model with vLLM

Checks that **vLLM serves the adapted model** (notebook 07's output) with the config that `zubaan export vllm` generates (`dtype half`): vLLM's tokenization of Urdu equals the extended `tokenizer.json`, and the model answers an Urdu completion and an Urdu chat question, once with thinking off and once with thinking allowed (to see what training on plain text did to the `<think>` behavior).

Settings before running:

- Accelerator: **GPU T4 x2** (one GPU is used). **Not P100**: vLLM needs compute capability 7.0 or higher.
- Internet: **on**. No secret is needed.
- Input: **Add Input > Notebooks**: the output of `07_adapt_main_qwen3`.

Takes about 15 to 25 minutes, mostly installing vLLM (its own Python 3.12 environment) and loading the model. The model is stored as bfloat16 and served as float16 (the T4 has no bfloat16).

Send back `vllm_check_adapted.json`.
"""
        ),
        code(n4.INSTALL.format(commit=commit)),
        code(n4.CHECK_CODE),
        code(check_gpu),
        code(SETTINGS),
        md("## 1. Find the adapted model"),
        code("import json\n" + FIND_MODEL),
        md("## 2. Install vLLM (isolated environment)"),
        code(n4.INSTALL_VLLM),
        md("## 3. Generate the vLLM config and start the server"),
        code(n4.WRITE_FILES),
        code(n4.SERVE),
        md("## 4. Check tokenization and generation"),
        code(TEST),
        md("## 5. Stop the server"),
        code(n4.STOP),
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
    out = Path(__file__).parent / "10_adapt_vllm_qwen3.ipynb"
    nb = build(commit)
    for i, cell in enumerate(nb["cells"]):
        cell["id"] = f"cell-{i}"
    out.write_text(
        json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n"
    )
    print(f"wrote {out} pinned to commit {commit}")


if __name__ == "__main__":
    main()
