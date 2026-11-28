# ruff: noqa: E501
"""Writes notebooks/16_tune_export.ipynb (Stage 4, chunk 8: re-export the chat-tuned model).

Run: python notebooks/make_16_notebook.py [COMMIT]   (COMMIT must be pushed to GitHub)

Notebook 09 re-exported the Stage 3 model on Kaggle CPU; its cells are reused here unchanged
except for the model search. New: the chat check runs on the Q4_K_M file through llama-server,
and checks 1 to 4 of the plan are applied to it.
"""

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).parent


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


n3, n9 = _load("make_03_notebook"), _load("make_09_notebook")
md, code = n3.md, n3.code

# The chat check needs llama-server as well as notebook 09's tools.
LLAMA_SELFCHECK = n9.LLAMA_SELFCHECK.replace(
    '"llama-completion", "llama-perplexity")',
    '"llama-completion", "llama-perplexity", "llama-server")',
).replace("llama-completion llama-perplexity\n", "llama-completion llama-perplexity llama-server\n")
assert LLAMA_SELFCHECK.count("llama-server") == 2, "notebook 09's self-check cell changed"

CHECK_CODE = n9.CHECK_CODE.replace(
    "import zubaan.export.check  # noqa: F401  (export check is in this commit)",
    "import zubaan.export.check  # noqa: F401  (export check is in this commit)\n"
    "from zubaan.tune.evaluate import quantized_checks  # noqa: F401  (chunk 8 check is in this commit)",
)
assert "quantized_checks" in CHECK_CODE, "notebook 09's check cell changed"

FIND_MODEL = """
import json
from pathlib import Path

# The Stage 4 model is the output of notebook 14 (Add Input > Notebooks): tuned/model. Its
# zubaan_extension.json has a "stage4" entry, which tells it apart from the Stage 3 model.
markers = sorted(Path("/kaggle/input").rglob("zubaan_extension.json"))
tuned = [m.parent for m in markers if "stage4" in json.loads(m.read_text())]
print("Stage 4 model folders:", [str(f) for f in tuned])
assert len(tuned) == 1, f"need exactly one Stage 4 model folder, found {len(tuned)}: add notebook 14's output (the main run)"
MODEL_DIR = tuned[0]
for f in ("tokenizer.json", "tokenizer_config.json", "config.json"):
    assert (MODEL_DIR / f).exists(), f"no {f} in {MODEL_DIR}"
assert (MODEL_DIR / "model.safetensors").exists() or list(MODEL_DIR.glob("model-*.safetensors")), "no weights"
meta = json.loads((MODEL_DIR / "zubaan_extension.json").read_text())
print(json.dumps(meta["stage4"], indent=1)[:2000])
has_template = "chat_template" in (MODEL_DIR / "tokenizer_config.json").read_text() or (MODEL_DIR / "chat_template.jinja").exists()
assert has_template, "no chat template: Ollama and llama-server chat would fail"
print("OK: Stage 4 model found:", MODEL_DIR)
"""

SETTINGS = """
BASE_MODEL = "Qwen/Qwen3-1.7B"   # tokenizer reference for the pre-tokenizer and fingerprint checks
NAME = "qwen3-1.7b-urdu-chat"
"""

CHAT_Q4 = """
# The chat check on the Q4_K_M file: the same 60 prompts, greedy, thinking on and off, the same
# limits as notebook 15, served by llama-server with the GGUF's own chat template. About 1 to 1.5
# hours on 4 CPUs (one reply at a time). Replies are saved as they arrive; a rerun resumes.
import os

from zubaan.config import ChatCheckConfig
from zubaan.tune.chat_check import run_from_config
from zubaan.tune.evaluate import checks_markdown, quantized_checks

Q4 = WORK / "export" / f"{NAME}-Q4_K_M.gguf"
assert Q4.exists(), f"{Q4} is missing"
out = WORK / "chat_check_q4.json"
if out.exists():
    result = json.loads(out.read_text(encoding="utf-8"))
    print("already done")
else:
    cfg = ChatCheckConfig(
        label="stage4_q4", gguf=Q4, llama_cpp_dir=LLAMA_DIR, llama_cpp_bin=BIN_DIR,
        threads=os.cpu_count(), output=out,
    )
    result = run_from_config(cfg, log=print)
print(f"{result['seconds'] / 60:.0f} minutes")
checks = quantized_checks(result["summary"])
md_text = checks_markdown(checks, {"stage4_q4": result["summary"]})
(WORK / "chat_check_q4_checks.json").write_text(json.dumps({"checks": checks}, indent=1))
(WORK / "chat_check_q4_checks.md").write_text(md_text, encoding="utf-8")
print(md_text)
print("PASS" if all(c["passed"] for c in checks) else "FAIL: see the table above")
"""

PACK = """
# The small result files in one zip to download. The GGUF files stay in the Output tab.
import shutil

RESULTS = WORK / "export_results"
RESULTS.mkdir(exist_ok=True)
for f in [WORK / "export" / "export_summary.json", WORK / "export" / "export_check.json", WORK / "export" / "Modelfile",
          *WORK.glob("chat_check_q4*")]:
    if f.exists() and f.is_file() and f.suffix != ".jsonl":
        shutil.copy(f, RESULTS / f.name)
shutil.make_archive(str(WORK / "export_results"), "zip", RESULTS)
print(sorted(p.name for p in RESULTS.iterdir()))
print("download /kaggle/working/export_results.zip and export/" + NAME + "-Q4_K_M.gguf from the Output tab")
"""


def build(commit: str) -> dict:
    cells = [
        md(
            f"""
# Stage 4, chunk 8: re-export the chat-tuned model

Converts the Stage 4 model from notebook 14 to GGUF, **proves tokenizer parity** (the export stops if Hugging Face and llama.cpp disagree on any id), builds Q4_K_M and Q8_0 with the Urdu importance matrix, writes the Ollama `Modelfile` and vLLM config, and runs `zubaan export check` (generation samples and the quantization cost), exactly as notebook 09 did for Stage 3. Then:

- **The chat check on the Q4_K_M file** (the 60 prompts of notebook 15, thinking on and off, greedy, through llama-server) with checks 1 to 4 of the plan, against the base model's numbers from notebook 15.

Settings before running:

- Accelerator: **None** (CPU). This notebook costs no GPU hours.
- Internet: **on**. No secret is needed.
- Input: **Add Input > Notebooks**: the output of `14_tune_sft` (the main run, with `tuned/model/`). Only that one.

Expected time about 3 to 3.5 hours on 4 CPUs: the export and its checks about 2 hours (as notebook 09), the chat check 1 to 1.5 hours.

Run with *Save Version > Save & Run All (Commit)*. **Download afterwards:** `export_results.zip`, and `export/{{NAME}}-Q4_K_M.gguf` (about 1.2 GB) for the local Ollama check.

llama.cpp release `{n3.LLAMA_CPP_TAG}`, as in Stages 2 and 3.
"""
        ),
        code(n3.INSTALL.format(commit=commit)),
        code(CHECK_CODE),
        code(n3.CHECK_MACHINE),
        code(SETTINGS),
        md("## 1. Find the Stage 4 model"),
        code(FIND_MODEL),
        md("## 2. llama.cpp"),
        code(n3.LLAMA_CPP),
        code(LLAMA_SELFCHECK),
        md(
            "## 3. Data and config\n\nFLORES-200 (parity and quantization-cost texts) and Urdu Wikipedia (importance matrix)."
        ),
        code(n9.CONFIG),
        code("!zubaan data flores --out data\n!zubaan data urdu --config export.yaml"),
        md("## 4. Export with the parity check"),
        code("!zubaan export gguf --config export.yaml"),
        code(
            "!zubaan export ollama --config export.yaml\n!zubaan export vllm --config export.yaml"
        ),
        code(n9.SUMMARY.replace("of the adapted model", "of the Stage 4 model")),
        md("## 5. Generation samples and quantization cost"),
        code("!zubaan export check --config export.yaml"),
        code(n9.SHOW),
        code(n9.READABLE),
        md("## 6. Chat check on the Q4_K_M file"),
        code(CHAT_Q4),
        md("## 7. Clean up and pack the results"),
        code(n9.CLEANUP),
        code(PACK),
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
    out = HERE / "16_tune_export.ipynb"
    nb = build(commit)
    for i, cell in enumerate(nb["cells"]):
        cell["id"] = f"cell-{i}"
    out.write_text(
        json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n"
    )
    print(f"wrote {out} pinned to commit {commit}")


if __name__ == "__main__":
    main()
