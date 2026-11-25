# ruff: noqa: E501
"""Writes notebooks/11_adapt_control_qwen3.ipynb (Stage 3, decision 6: the compute-matched control).

Base Qwen3-1.7B with its own tokenizer, trained with the main run's recipe (LoRA, Urdu Wikipedia with
10% English, fp16, micro batch 1) for the same 800 steps of 32 blocks of 1,024 tokens. It answers
whether the adapted model's quality gain comes from the extended tokenizer or from the extra Urdu
training alone.

Run: python notebooks/make_11_notebook.py [COMMIT [HUB_REPO]]   (COMMIT must be pushed to GitHub)
"""

import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import adapt_cells as c  # noqa: E402

# The held-out Urdu file the Stage 2 and Stage 3 tables were measured on (see make_08_notebook.py).
HELDOUT_SHA = "19c4f1b63b41b8a91cf8aa345d0f0e83945f33e403cc78c1f6f52ed7e7c34ada"
# Measured on the extended tokenizer (decisions, 2026-11-18): the main run's Urdu token count.
MAIN_URDU_TOKENS = 16053605

SETTINGS = """
import os
import time

T0 = time.time()  # session start, for the time guard before the optional evaluation
# A PRIVATE Hub repo for this run's checkpoints. It must NOT be the main run's repo: use a new name,
# e.g. "your-name/zubaan-qwen3-control". Resume refuses a main-run checkpoint anyway.
HUB_REPO = ""
MICRO_BATCH = 1          # the main run's setting (notebook 06 speed trial)
GRAD_CHECKPOINT = False  # the main run's setting
PHASE_A_STEPS = 0        # no new rows in the base model, so no phase A
PHASE_B_STEPS = 800      # same 800 steps x 32 blocks x 1,024 tokens as the main run (100 + 700)
# Do not change the lines above between sessions of the same run: a resumed run refuses a changed schedule.
STOP_AT_STEP = None      # None = all 800 steps (about 9 h). 400 = about 4.5 h, compared with the main run's step-400 milestone; a later session with None continues to 800
RUN_EVAL = True          # after the merge, score the control on the notebook 08 texts if the session has time left

# Peak memory of the main run was 13.3 GB of about 14.5: let the allocator avoid fragmentation.
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
assert HUB_REPO, "set HUB_REPO: without it a crash or the 12-hour limit loses the run"
assert 32 % MICRO_BATCH == 0, "MICRO_BATCH must divide 32"
assert STOP_AT_STEP is None or 0 < STOP_AT_STEP <= PHASE_B_STEPS, STOP_AT_STEP
OUT_NAME = "control"
HUB = HUB_REPO
"""

BASE_MODEL = """
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download
from tokenizers import Tokenizer

# The base model, with its own tokenizer, downloaded outside /kaggle/working so it is not saved as output.
BASE = "Qwen/Qwen3-1.7B"
BASE_REVISION = HfApi().model_info(BASE).sha
MODEL_DIR = Path("/tmp/qwen3-1.7b-base")
snapshot_download(BASE, revision=BASE_REVISION, local_dir=str(MODEL_DIR))
print("base model:", BASE, "revision", BASE_REVISION, "->", MODEL_DIR)
assert not (MODEL_DIR / "zubaan_extension.json").exists(), "this is an extended folder, not the base"
vocab = Tokenizer.from_file(str(MODEL_DIR / "tokenizer.json")).get_vocab_size()
print("tokenizer vocabulary:", vocab)
assert vocab < 160_000, "the tokenizer has the 10,000 Urdu tokens: this must be the base tokenizer"
print("OK: base model and base tokenizer")
"""

DATA_CHECK = f"""
meta = json.loads(Path("data/adapt_control/blocks_meta.json").read_text())
assert meta["control"] is True and meta["first_new_id"] is None, "blocks were not made for the control run"
assert meta["overlap_guard"]["hits"] == 0
urdu_tokens, n_urdu, n_en = meta["urdu"]["n_tokens"], meta["urdu"]["n_blocks"], meta["english"]["n_blocks"]
print("Urdu blocks", n_urdu, "| tokens", urdu_tokens, "| English blocks", n_en)
print("guard:", meta["overlap_guard"])
# The same Urdu text with the extended tokenizer was {MAIN_URDU_TOKENS:,} tokens; Stage 2 measured 2.29x fewer
# tokens per character with the extension, so expect about 36.7M here.
ratio = urdu_tokens / {MAIN_URDU_TOKENS}
print(f"base / extended tokens on the same Urdu text: {{ratio:.2f}}x")
assert 2.0 < ratio < 2.6, "unexpected token ratio: a different corpus or tokenizer, check before training"
epoch = n_urdu / 0.9  # Urdu blocks plus 10% English
print(f"800 steps x 32 blocks = 25,600 blocks = {{25600 / epoch:.2f}} epochs of this data (main run: 1.47)")
"""

CHECK_TRAIN = """
# CHECK 3: five real fp16 LoRA steps on the base model (no phase A: nothing but LoRA trains).
from zubaan.adapt.pipeline import run_adapt
from zubaan.config import load_config

cfg = load_config("adapt.yaml").adapt
assert cfg.control and cfg.phase_a_steps == 0
check = cfg.model_copy(update={
    "output_dir": WORK / "adapt_check", "hub_repo": None, "resume": False,
    "phase_b_steps": 5, "grad_accum": 2, "warmup_steps": 1,
    "log_every": 1, "eval_every": 5, "eval_texts": 20, "checkpoint_every": 5,
})
run_adapt(check, log=print)
rows = [json.loads(line) for line in (check.output_dir / "train_log.jsonl").read_text().splitlines()]
steps = [r for r in rows if "loss" in r]
skipped = sum(1 for r in steps if r["grad_norm"] == float("inf"))
print(f"steps skipped by the loss scaler: {skipped} of 5")
assert len(steps) == 5 and all(r["phase"] == "B" and r["lr_new"] is None for r in steps)
assert all(r["loss"] == r["loss"] and abs(r["loss"]) < 1e4 for r in steps)
assert skipped < 5, "every step overflowed: use precision fp32"
import shutil

shutil.rmtree(check.output_dir, ignore_errors=True)
print("OK: 5 fp16 LoRA steps, finite losses", [round(r["loss"], 3) for r in steps])
"""

HUB_IS_CONTROL = """
# A checkpoint already in this repo must be from a control run, or the repo is the wrong one.
if "last/meta.json" in files:
    saved = json.loads(Path(hf_hub_download(HUB_REPO, "last/meta.json")).read_text())["config"]
    assert saved.get("control") is True, f"{HUB_REPO} holds a main-run checkpoint: use a new repo for the control"
    print("OK: the existing checkpoint is from this control run")
"""

MAIN = """
from zubaan.adapt.pipeline import run_adapt
from zubaan.config import load_config

cfg = load_config("adapt.yaml").adapt
summary = run_adapt(cfg, log=print, stop_at_step=STOP_AT_STEP)
print({k: summary[k] for k in ("resumed_from_step", "final_step", "total_steps", "finished")})
target = STOP_AT_STEP or summary["total_steps"]
assert summary["final_step"] == target, "the run stopped before its target: run this notebook again with the same settings to resume"
for e in summary["evals"]:
    print(e["step"], e["eval"])
"""

MERGE = """
# Fold LoRA into the weights (float32) and save a plain model with the base tokenizer. The merge checks
# that every embedding row and the tokenizer files are identical to the base model.
!zubaan adapt merge --config adapt.yaml
meta = json.loads((WORK / "control/model/zubaan_adapt.json").read_text())
print(json.dumps(meta, indent=1))
assert meta["control"] is True and meta["stage3"]["step"] == (STOP_AT_STEP or PHASE_B_STEPS)
"""

EVAL = f"""
# Optional: score the control on the texts and protocol of notebook 08 (Stage 2 protocol: 1000 texts of at
# most 400 characters per set, all 900 Belebele questions, float32). The base model took 38 minutes in 08
# and the control has the same tokenizer, so this needs about 45 minutes.
import yaml

hours_used = (time.time() - T0) / 3600
if not RUN_EVAL:
    print("RUN_EVAL is False: evaluate later with the merged model in this notebook's output")
elif hours_used > 10.75:
    print(f"{{hours_used:.1f}} h used: not enough time before the 12 h limit. Evaluate in a new session from this notebook's output")
else:
    sha = json.loads(Path("data/urdu/urdu_meta.json").read_text())["files"]["heldout"]["sha256"]
    assert sha == "{HELDOUT_SHA}", "the held-out Urdu file differs from the one notebook 08 used"
    step = STOP_AT_STEP or PHASE_B_STEPS
    name = f"control-step{{step}}"
    evaluate = {{
        "models": {{name: str(WORK / "control/model")}},
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
        "output": str(WORK / "control_quality.json"),
    }}
    Path("evaluate.yaml").write_text(yaml.safe_dump({{"evaluate": evaluate}}, sort_keys=False))
    !zubaan adapt eval --config evaluate.yaml
    row = json.loads((WORK / "control_quality.json").read_text())["variants"][name]
    print({{k: round(v["bits_per_character"], 3) for k, v in row["bpc"].items()}})
    print("adapted (notebook 08): urdu_heldout 1.486, urdu_flores 1.718, english_flores 1.066; Belebele urd acc_char 0.340")
"""

CLEANUP = (
    """
# Local checkpoints are on the Hub; drop them from the output, keep the merged model and the logs.
import shutil

shutil.rmtree(WORK / "control" / "checkpoints", ignore_errors=True)
"""
    + c.CLEANUP
)


def build(commit: str) -> dict:
    cells = [
        c.md(
            """
# Stage 3: compute-matched control run (base Qwen3-1.7B, base tokenizer)

The adapted model scored 1.718 bits per character on FLORES Urdu against the base model's 2.339, but it also had 16M tokens of Urdu training the base never had. This run separates the two: the **base model with its own tokenizer**, trained with the main run's recipe on the same Urdu Wikipedia and English replay, for the **same number of tokens** (800 steps x 32 blocks x 1,024 tokens). Decision 6 in `docs/decisions.md`.

- Same as notebook 07: LoRA rank 64 on all projections, learning rate 2e-4, warmup 50, cosine to 10%, fp16, micro batch 1, 10% English replay, seed 0, in-run eval every 100 steps on the same 200 held-out Urdu lines.
- Different by design: no new rows and no phase A, so all 800 steps train LoRA (the main run trained LoRA for 700). The base tokenizer needs about 2.29x more tokens for the same Urdu, so the control sees about 0.64 epochs of the Urdu text where the main run saw 1.47. That is what matching on compute means.

Checkpoints go to your private Hub repo every 50 steps, with a permanent milestone every 200. If the session ends, run the notebook again with the same settings and it resumes. `STOP_AT_STEP = 400` gives a shorter run (about 4.5 h) to compare with the main run's step-400 milestone; a later session with `None` continues it to 800.

Settings before running: Accelerator **GPU T4 x2** (one GPU is used), Internet **on**, Secret **HF_TOKEN** attached. **No input is needed** (the base model is downloaded). Set `HUB_REPO` to a **new private repo** in the first cell. Use *Save Version > Save & Run All (Commit)* so it keeps running with the tab closed.

Send back `control/adapt_summary.json`, `control/train_log.jsonl` and, if it ran, `control_quality.json`.
"""
        ),
        c.code(SETTINGS),
        c.code(c.INSTALL.format(commit=commit)),
        c.code(c.CHECK_CODE),
        c.code(c.CHECK_GPU),
        c.code(c.SECRET.replace(', "(smoke runs use", HUB_REPO + "-smoke)"', "")),
        c.md("## 1. Base model"),
        c.code(BASE_MODEL),
        c.md("## 2. Configuration"),
        c.code(
            c.CONFIG.replace(
                '"english_mb": 12,',
                '"english_mb": 20,  # the base tokenizer gives about 2.3x more Urdu blocks, so 10% English needs more replay text',
            ).replace(
                '"data_dir": "data/adapt",',
                '"data_dir": "data/adapt_control",\n    "control": True,',
            )
        ),
        c.md(
            "## 3. Data\n\nThe same Urdu and English text as notebook 07, tokenized with the base tokenizer (about 15 minutes)."
        ),
        c.code(c.DATA),
        c.code(DATA_CHECK),
        c.md("## 4. Five-step check"),
        c.code(CHECK_TRAIN),
        c.code(c.FREE_GPU),
        c.md("## 5. Hub check"),
        c.code(c.HUB_PREFLIGHT + HUB_IS_CONTROL),
        c.md(
            "## 6. Train\n\nRun it again with the same settings if the session ends before it reaches its target step."
        ),
        c.code(MAIN),
        c.md("## 7. Merge, save and verify"),
        c.code(MERGE),
        c.md("## 8. Evaluate (optional, needs about 45 minutes)"),
        c.code(EVAL),
        c.md("## 9. Output size"),
        c.code(CLEANUP),
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
    out = Path(__file__).parent / "11_adapt_control_qwen3.ipynb"
    nb = build(commit)
    if len(sys.argv) > 2:  # fill in the user's private Hub repo in the settings cell
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
