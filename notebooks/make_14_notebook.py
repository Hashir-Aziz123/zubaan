# ruff: noqa: E501
"""Writes notebooks/14_tune_sft.ipynb (Stage 4, chunk 6: chat SFT on a T4).

Run: python notebooks/make_14_notebook.py [COMMIT [HUB_REPO]]   (COMMIT must be pushed to GitHub)

Install, GPU and secret cells come from the Stage 3 notebooks, which ran on Kaggle; everything
else calls zubaan.tune.sft.
"""

import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import adapt_cells as c  # noqa: E402

SETTINGS = """
# Run MODE = "smoke" first (about 30 to 45 minutes), send back smoke_report.json, then run MODE = "main".
MODE = "smoke"
# Private Hub repo for Stage 4 checkpoints (a different one from Stage 3's). Smoke runs use HUB_REPO + "-smoke".
HUB_REPO = ""            # e.g. "your-name/zubaan-qwen3-tune-ckpt"
GRAD_CHECKPOINT = True   # rows are 4,096 tokens (Stage 3 used 2 x 1,024); the smoke run reports peak memory
EPOCHS = 2
LR_LORA = 1e-4
TRAIN_NEW_ROWS = False   # the plan: start with the Stage 2 rows frozen
# Do not change the lines above between sessions of the same run: a resumed run refuses a changed schedule.
import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"  # before torch starts
assert MODE in ("smoke", "main"), "MODE must be smoke or main"
assert HUB_REPO, "set HUB_REPO: without it a crash or the 12-hour limit loses the run"
"""

FIND_INPUTS = """
import json
from pathlib import Path

# Two inputs (Add Input in the sidebar):
#  1. the output of notebook 07 (the merged Stage 3 model, adapted/model), and
#  2. a Kaggle dataset holding conversations.jsonl (from zubaan tune filter on the laptop).
INPUT_ROOT = Path("/kaggle/input")
stage3 = []
for m in sorted(INPUT_ROOT.rglob("zubaan_extension.json")):
    meta = json.loads(m.read_text(encoding="utf-8"))
    if meta.get("trained") and "stage4" not in meta:
        stage3.append(m.parent)
assert len(stage3) == 1, f"need exactly one Stage 3 model (output of notebook 07), found {[str(s) for s in stage3]}"
MODEL_DIR = stage3[0]
assert (MODEL_DIR / "model.safetensors").exists() or list(MODEL_DIR.glob("model-*.safetensors")), "no weights"
data_files = sorted(INPUT_ROOT.rglob("conversations.jsonl"))
assert len(data_files) == 1, f"need exactly one conversations.jsonl (upload it as a dataset), found {data_files}"
TRAIN_FILE = data_files[0]

from transformers import AutoTokenizer

assert AutoTokenizer.from_pretrained(MODEL_DIR).chat_template, "the Stage 3 folder has no chat template"
report = TRAIN_FILE.parent / "curate_report.json"
print("model:", MODEL_DIR)
print("conversations:", TRAIN_FILE, sum(1 for _ in TRAIN_FILE.open(encoding="utf-8")), "lines")
if report.exists():
    print("filter report:", json.loads(report.read_text(encoding="utf-8"))["slots"])
"""

CONFIG = """
import yaml

WORK = Path("/kaggle/working")
OUT = WORK / ("tuned" if MODE == "main" else "tune_smoke")
# Smoke runs get their own Hub repo, so the main run can never resume from a smoke checkpoint.
HUB = HUB_REPO if MODE == "main" else HUB_REPO + "-smoke"
tune = {
    "model_path": str(MODEL_DIR),
    "train_file": str(TRAIN_FILE),
    "max_len": 4096,
    "val_share": 0.02,
    "output_dir": str(OUT),
    "epochs": EPOCHS,
    "micro_batch_size": 1,
    "grad_accum": 16,
    "lr_lora": LR_LORA,
    "train_new_rows": TRAIN_NEW_ROWS,
    "warmup_steps": 20,
    "lora_rank": 64,
    "lora_alpha": 128,
    "precision": "fp16",
    "gradient_checkpointing": GRAD_CHECKPOINT,
    "device": "cuda",
    "seed": 0,
    "log_every": 5,
    "eval_every": 50,
    "checkpoint_every": 25,
    "keep_checkpoints": 2,
    "resume": True,
    "hub_repo": HUB,
    "hub_private": True,
}
Path("tune.yaml").write_text(yaml.safe_dump({"tune": tune}, sort_keys=False))
from zubaan.config import load_config

cfg = load_config("tune.yaml").tune
print(cfg.model_dump_json(indent=1))
"""

HUB_PREFLIGHT = """
# Before training: the repo exists, is private, accepts a write, and says where this run starts.
from huggingface_hub import HfApi

api = HfApi()
api.create_repo(HUB, private=True, exist_ok=True)
assert api.repo_info(HUB).private, f"{HUB} is public: make it private on the Hub first"
api.upload_file(path_or_fileobj=b"ok", path_in_repo="_preflight/ok.txt", repo_id=HUB)
api.delete_folder("_preflight", repo_id=HUB)
files = set(api.list_repo_files(HUB))
if MODE == "smoke" and "last/meta.json" in files:
    # An earlier smoke run's checkpoint; the smoke test must start at step 0 to test the resume.
    api.delete_folder("last", repo_id=HUB)
    print("removed an earlier smoke checkpoint from", HUB)
elif "last/meta.json" in files:
    from huggingface_hub import hf_hub_download

    step = json.loads(Path(hf_hub_download(HUB, "last/meta.json")).read_text())["step"]
    print(f"OK: a checkpoint at step {step} exists in {HUB}: this run RESUMES from it.")
else:
    print(f"OK: {HUB} is private and writable. No checkpoint yet: this run starts from step 0.")
"""

SMOKE = """
# Smoke: 20 steps, delete the local checkpoints, resume from the Hub to step 30. Then speed, memory,
# the held-out loss, and the main run's projected hours.
import gc
import shutil
import statistics

import torch

from zubaan.tune.sft import run_tune


def free_gpu():
    # The model of a finished run_tune call is garbage; release its GPU memory before the next one.
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if MODE == "smoke":
    run_tune(cfg, log=print, stop_at_step=20)
    free_gpu()
    shutil.rmtree(cfg.output_dir / "checkpoints")
    second = run_tune(cfg, log=print, stop_at_step=30)
    assert second["resumed_from_step"] == 20, "the run did not resume from the Hub checkpoint"
    log_rows = [json.loads(x) for x in (cfg.output_dir / "train_log.jsonl").read_text().splitlines()]
    steps = [r for r in log_rows if "loss" in r]
    speed = statistics.median(r["tokens_per_s"] for r in steps[1:])
    train = second["data"]["train"]
    tokens_per_step = train["tokens"] / train["rows"] * cfg.micro_batch_size * cfg.grad_accum
    smoke = {
        "gpu": gpu_name,
        "data": second["data"],
        "packing_max_logit_diff": second["packing_max_logit_diff"],
        "losses": [(r["step"], round(r["loss"], 4)) for r in steps],
        "evals": second["evals"],
        "median_tokens_per_s": round(speed),
        "peak_gpu_mem_gb": round(max(r.get("gpu_mem_gb", 0) for r in steps), 2),
        "total_steps": second["total_steps"],
        "projected_main_hours": round(second["total_steps"] * tokens_per_step / speed / 3600, 2),
        "settings": {"grad_checkpoint": GRAD_CHECKPOINT, "epochs": EPOCHS, "lr_lora": LR_LORA, "train_new_rows": TRAIN_NEW_ROWS},
    }
    (WORK / "smoke_report.json").write_text(json.dumps(smoke, indent=1, ensure_ascii=False))
    print(json.dumps({k: v for k, v in smoke.items() if k != "data"}, indent=1))
    print("Send back smoke_report.json (Output tab).")
"""

MAIN = """
# Main: the full schedule. If the session ends first, run the notebook again with the same settings.
if MODE == "main":
    summary = run_tune(cfg, log=print)
    print({k: summary[k] for k in ("resumed_from_step", "final_step", "total_steps", "finished")})
    assert summary["finished"], "the schedule did not finish: run this notebook again with the same settings to resume"
    for e in summary["evals"]:
        print(e["step"], e["eval"])
"""

MERGE = """
# Fold LoRA into the Stage 3 weights (float32), check the logits did not change, save tuned/model.
if MODE == "main":
    from zubaan.tune.sft import merge_tuned

    free_gpu()
    merged = merge_tuned(cfg, log=print)
    shutil.rmtree(cfg.output_dir / "checkpoints", ignore_errors=True)  # they are on the Hub
"""

PROBE = """
# A first look (not the evaluation): greedy replies to 2 Urdu and 2 English chat-check prompts,
# thinking on and off. The Stage 3 model answered none of these with thinking on.
if MODE == "main":
    from zubaan.tune.sft import quick_probe

    free_gpu()
    probe = quick_probe(Path(merged["model_dir"]), log=print)
    (WORK / "probe.json").write_text(json.dumps(probe, indent=1, ensure_ascii=False))
    for p in probe:
        print("\\n===", p["id"], "thinking" if p["thinking"] else "no thinking", "===")
        print(p["reply"][:1500])
"""


def build(commit: str) -> dict:
    cells = [
        c.md(
            """
# Stage 4, chunk 6: chat SFT of the adapted model

LoRA fine-tuning of the merged Stage 3 model on the filtered teacher conversations, with the loss on the assistant's reply only, so the model answers again with thinking on and replies in Urdu. All logic lives in `zubaan.tune.sft`; decisions in `docs/decisions.md` (2026-11-26).

**Two modes, in this order** (first code cell):

1. `smoke`: 20 steps, local checkpoints deleted, resumed from the Hub to step 30. Reports speed, peak memory, the held-out loss and the main run's projected hours. **Send back `smoke_report.json`.**
2. `main`: the full schedule (2 epochs), then merge, save `tuned/model`, and a first look at a few replies. Checkpoints go to the private Hub repo every 25 steps; if the session ends, run it again with the same settings and it resumes.

Settings before running:

- Accelerator: **GPU T4 x2** (one GPU is used). Internet: **on**.
- Inputs (**Add Input**): the output of `07_adapt_main_qwen3` (the Stage 3 model), and a **dataset** with `conversations.jsonl` (and `curate_report.json`) from `zubaan tune filter`.
- Secret: **HF_TOKEN** (Add-ons > Secrets).
- Run with **Save Version > Save & Run All (Commit)**.
"""
        ),
        c.code(SETTINGS),
        c.code(c.INSTALL.format(commit=commit)),
        c.code(c.CHECK_CODE),
        c.code(c.CHECK_GPU),
        c.code(c.SECRET),
        c.md("## 1. Inputs: the Stage 3 model and the conversations"),
        c.code(FIND_INPUTS),
        c.md("## 2. Configuration and Hub check"),
        c.code(CONFIG),
        c.code(HUB_PREFLIGHT),
        c.md("## 3. Smoke run (MODE = smoke)"),
        c.code(SMOKE),
        c.md("## 4. Main run, merge and a first look (MODE = main)"),
        c.code(MAIN),
        c.code(MERGE),
        c.code(PROBE),
        c.md("## 5. Output size"),
        c.code(c.CLEANUP),
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
    out = Path(__file__).parent / "14_tune_sft.ipynb"
    nb = build(commit)
    if len(sys.argv) > 2:  # fill in the private Hub repo in the settings cell
        first = nb["cells"][1]
        first["source"] = [
            ln.replace('HUB_REPO = ""', f'HUB_REPO = "{sys.argv[2]}"') for ln in first["source"]
        ]
    for i, cell in enumerate(nb["cells"]):
        cell["id"] = f"cell-{i}"
    out.write_text(
        json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n"
    )
    print(f"wrote {out} pinned to commit {commit}")


if __name__ == "__main__":
    main()
