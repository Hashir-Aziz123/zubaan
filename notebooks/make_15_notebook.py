# ruff: noqa: E501
"""Writes notebooks/15_tune_eval.ipynb (Stage 4, chunk 7: evaluation against the success checks).

Run: python notebooks/make_15_notebook.py [COMMIT]   (COMMIT must be pushed to GitHub)

The vLLM install and server cells come from notebooks 04 and 12, the Urdu data cells from notebook
08; all of them ran on Kaggle. Everything else calls zubaan.tune.
"""

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import adapt_cells as c  # noqa: E402


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


n4, n8, n12 = _load("make_04_notebook"), _load("make_08_notebook"), _load("make_12_notebook")

SETTINGS = """
from pathlib import Path

OUT = Path("/kaggle/working/tune_eval")
OUT.mkdir(parents=True, exist_ok=True)
MAX_MODEL_LEN = 4096   # prompt plus the longest reply the check allows (2,048 with thinking)
WORKERS = 16           # chat-check requests in flight; vLLM batches them
"""

FIND_MODELS = """
# Inputs: the output of notebook 07 (Stage 3 model) and of notebook 14 (Stage 4 model). The base
# model is downloaded from the Hub.
INPUT_ROOT = Path("/kaggle/input")
stage3, stage4 = [], []
for m in sorted(INPUT_ROOT.rglob("zubaan_extension.json")):
    meta = json.loads(m.read_text(encoding="utf-8"))
    if meta.get("trained"):
        (stage4 if "stage4" in meta else stage3).append(m.parent)
assert len(stage3) == 1 and len(stage4) == 1, f"need one Stage 3 and one Stage 4 model, found {stage3} and {stage4}"
MODELS = {"base": "Qwen/Qwen3-1.7B", "stage3": str(stage3[0]), "stage4": str(stage4[0])}
a = json.loads((stage3[0] / "tokenizer.json").read_text(encoding="utf-8"))
b = json.loads((stage4[0] / "tokenizer.json").read_text(encoding="utf-8"))
assert a == b, "Stage 3 and Stage 4 tokenizers differ: SFT must not touch the tokenizer"
print(json.dumps(MODELS, indent=1))
"""

CHAT = """
# The same hand-written prompt set, greedy decoding and limits for all three models, each served by
# vLLM in turn. Replies are saved as they arrive; a rerun after a crash resumes each model.
from zubaan.config import ChatCheckConfig
from zubaan.tune.chat_check import run_from_config

chat = {}
for label, model in MODELS.items():
    out_file = OUT / f"chat_check_{label}.json"
    if out_file.exists():
        chat[label] = json.loads(out_file.read_text(encoding="utf-8"))["summary"]
        print(label, "already done")
        continue
    proc = start_teacher(model, label)
    try:
        cfg = ChatCheckConfig(label=label, server_url="http://127.0.0.1:8000", model=label, workers=WORKERS, output=out_file)
        chat[label] = run_from_config(cfg, log=print)["summary"]
    finally:
        stop_teacher(proc)
    for mode, groups in chat[label].items():
        g = groups["all"]
        print(f"{label} {mode}: answered {g['answered']}/{g['n']}, hit limit {g['hit_limit']}, loops {g['loops']}")
"""

QUALITY_CONFIG = """
import yaml

# The Stage 3 protocol (notebook 08): 1000 texts of at most 400 characters per set, all 900 Belebele
# questions per language, float32. The base model is not re-scored (its numbers are recorded); the
# Stage 3 model is, to show the protocol still reproduces notebook 08.
evaluate = {
    "models": {"stage3": MODELS["stage3"], "stage4": MODELS["stage4"]},
    "untrained": None,
    "adapted": None,
    "heldout": "data/urdu/urdu_heldout.txt",
    "eval_urdu": "data/flores_urd.txt",
    "eval_english": "data/flores_eng.txt",
    "quality_texts": 1000,
    "max_chars": 400,
    "belebele_questions": None,
    "dtype": "float32",
    "device": "cuda",
    "batch_size": 4,
    "max_batch_tokens": 2048,
    "output": str(OUT / "tune_quality.json"),
}
config = {
    "urdu_corpus": {"raw_dir": "data/raw", "out_dir": "data/urdu", "flores": "data/flores_urd.txt", "max_mb": 100, "heldout_percent": 1.0},
    "evaluate": evaluate,
}
Path("evaluate.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
print(Path("evaluate.yaml").read_text())
"""

QUALITY = """
# About 25 minutes per model on a T4. Finished models are skipped on a rerun.
!zubaan adapt eval --config evaluate.yaml
"""

VERDICT = """
import shutil

from zubaan.tune.evaluate import checks_markdown, reproduces_stage3, stage4_checks

quality = json.loads((OUT / "tune_quality.json").read_text(encoding="utf-8"))["variants"]
drift = reproduces_stage3(quality["stage3"])
assert not drift, f"the Stage 3 model no longer reproduces notebook 08, so the protocol drifted: {drift}"
print("OK: the Stage 3 model reproduces its notebook 08 numbers")
checks = stage4_checks(chat, quality)
md_text = checks_markdown(checks, chat)
(OUT / "stage4_checks.json").write_text(json.dumps({"checks": checks, "chat": chat, "vllm": VLLM_VERSION, "gpu": gpu_name}, indent=1, ensure_ascii=False))
(OUT / "stage4_checks.md").write_text(md_text, encoding="utf-8")
print(md_text)
shutil.make_archive("/kaggle/working/tune_eval", "zip", OUT)
print("download /kaggle/working/tune_eval.zip from the Output tab")
"""


def build(commit: str) -> dict:
    cells = [
        c.md(
            """
# Stage 4, chunk 7: evaluate the chat-tuned model

1. **Chat behavior** of the base model, the Stage 3 model and the Stage 4 model: the hand-written prompt set (60 prompts), thinking on and off, greedy, each model served by vLLM in turn. This also gives the base and Stage 3 baselines.
2. **Quality**, the Stage 3 protocol: bits per character on held-out Urdu, FLORES Urdu and English, and Belebele, for the Stage 3 and Stage 4 models.
3. **The success checks** of `docs/stage4-plan.md` as PASS or FAIL.

About 1.5 to 2 hours. Settings before running:

- Accelerator: **GPU T4 x2**. Internet: **on**. No secret.
- Inputs (**Add Input > Notebooks**): the output of `07_adapt_main_qwen3` and of `14_tune_sft`.
- Run with **Save Version > Save & Run All (Commit)**, then send back `tune_eval.zip` (Output tab).
"""
        ),
        c.code(c.INSTALL.format(commit=commit)),
        c.code(c.CHECK_CODE),
        c.code(c.CHECK_GPU + n12.TWO_GPUS),
        c.code(SETTINGS),
        c.md("## 1. Models"),
        c.code(FIND_MODELS),
        c.md("## 2. Chat check (vLLM, isolated environment)"),
        c.code(n4.INSTALL_VLLM),
        c.code(n12.SERVER),
        c.code(CHAT),
        c.md("## 3. Quality (Stage 3 protocol)"),
        c.code(QUALITY_CONFIG),
        c.code(n8.DATA),
        c.code(n8.DATA_CHECK),
        c.code(QUALITY),
        c.md("## 4. Success checks"),
        c.code(VERDICT),
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
    out = HERE / "15_tune_eval.ipynb"
    nb = build(commit)
    for i, cell in enumerate(nb["cells"]):
        cell["id"] = f"cell-{i}"
    out.write_text(
        json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n"
    )
    print(f"wrote {out} pinned to commit {commit}")


if __name__ == "__main__":
    main()
