# ruff: noqa: E501
"""Writes notebooks/05_adapt_qwen3.ipynb.

Run: python notebooks/make_05_notebook.py [COMMIT]

COMMIT defaults to the current git HEAD and must be pushed to GitHub, because the notebook
installs zubaan from that exact commit.
"""

import json
import subprocess
import sys
from pathlib import Path


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(True)}


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": text.strip("\n").splitlines(True),
    }


SETTINGS = """
# What to run. Change MODE between sessions: "check", then "smoke", then "main".
MODE = "smoke"       # "check": versions and a 5-step fp16 test | "smoke": check plus a 30-step run with a kill and resume | "main": the full run, then merge and save
HUB_REPO = ""        # a PRIVATE Hub repo for checkpoints, e.g. "your-name/zubaan-qwen3-ckpt". Empty = no Hub (a crash loses the run)
MICRO_BATCH = 2      # blocks of 1024 tokens per forward pass; lower it if the smoke run runs out of GPU memory
assert MODE in ("check", "smoke", "main"), MODE
"""

INSTALL = """
# Install zubaan at one exact commit. Do not edit the next line except to change the commit.
COMMIT = "{commit}"
REPO = f"git+https://github.com/Hashir-Aziz123/zubaan.git@{{COMMIT}}"
!pip install -q "zubaan[audit,train] @ {{REPO}}"
# Kaggle may already hold an older copy with the same version number, which pip keeps.
# Force our commit (dependencies are already in place).
!pip install -q --force-reinstall --no-deps "zubaan @ {{REPO}}"
# Kaggle ships torchao 0.10, which peft refuses ("only versions above 0.16 are supported") when it
# builds LoRA layers. We never use torchao, so remove it; peft then skips its torchao support.
!pip uninstall -y -q torchao
!zubaan version
"""

CHECK_CODE = """
# CHECK 1: the installed code is the commit above, not a cached older copy.
# If this fails, use Run > Restart session and run the cells again from the top.
import json
from importlib import metadata

direct = json.loads(metadata.distribution("zubaan").read_text("direct_url.json"))
installed = direct["vcs_info"]["commit_id"]
print("installed commit:", installed)
assert installed.startswith(COMMIT), f"installed {installed}, wanted {COMMIT}"

import zubaan.adapt.pipeline  # noqa: F401

print("OK: code version verified")
"""

CHECK_GPU = """
import os
import subprocess

# Quiet progress bars: they can overwhelm the notebook front end.
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TQDM_DISABLE"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

# CHECK 2: a GPU with about 16 GB. Training uses one GPU (fp16: a T4 and a P100 have no BF16).
try:
    r = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,compute_cap,memory.total", "--format=csv,noheader"],
        capture_output=True,
        text=True,
    )
except FileNotFoundError:
    raise SystemExit("No GPU in this session (nvidia-smi is missing). Settings > Accelerator > GPU T4 x2, then run again.") from None
assert r.returncode == 0, "no GPU found: turn on a GPU accelerator in the notebook settings"
print(r.stdout)
gpu_name, cap, mem = [x.strip() for x in r.stdout.splitlines()[0].split(",")]
assert float(mem.split()[0]) >= 14000, f"{gpu_name} has {mem}; need about 16 GB"
if float(cap) < 7.0:
    print("WARNING: compute capability", cap, "has no fp16 tensor cores; this will be slow. A T4 is better.")

import peft
import torch
import transformers

print("torch", torch.__version__, "| transformers", transformers.__version__, "| peft", peft.__version__, "| cuda", torch.version.cuda)
assert torch.cuda.is_available(), "torch cannot see the GPU"
print("OK: GPU", gpu_name, "compute capability", cap, mem)
"""

SECRET = """
# The Hub token is a Kaggle secret named HF_TOKEN (Add-ons > Secrets). It is never written in code.
if HUB_REPO:
    from kaggle_secrets import UserSecretsClient

    os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")
    print("Hub checkpoints go to the PRIVATE repo", HUB_REPO, "(smoke runs use", HUB_REPO + "-smoke)")
else:
    print("No Hub repo: checkpoints stay in this session. A crash or the 12-hour limit loses the run.")
"""

FIND_MODEL = """
from pathlib import Path

# The extended model comes from the output of notebook 03, added as an input with "Add Input > Notebooks".
# The folder holds zubaan_extension.json. Prefer one that is not trained yet.
INPUT_ROOT = Path("/kaggle/input")
markers = sorted(INPUT_ROOT.rglob("zubaan_extension.json"))
assert markers, (
    "no extended model found under /kaggle/input. In the sidebar choose Add Input > Notebooks and add "
    "the finished output of 03_extend_export_qwen3"
)
untrained = [m for m in markers if not json.loads(m.read_text()).get("trained")]
marker = (untrained or markers)[0]
MODEL_DIR = marker.parent
print("model folder:", MODEL_DIR)
print(json.dumps(json.loads(marker.read_text()), indent=1))
assert (MODEL_DIR / "model.safetensors").exists() or list(MODEL_DIR.glob("model-*.safetensors")), "no weights in the model folder"
assert (MODEL_DIR / "tokenizer.json").exists(), "no tokenizer.json in the model folder"
"""

CONFIG = """
from pathlib import Path

import yaml

WORK = Path("/kaggle/working")
OUT = WORK / ("adapted" if MODE == "main" else "adapt_smoke")
# Smoke runs get their own Hub repo so a later main run can never resume from a smoke checkpoint.
hub = None if not HUB_REPO else (HUB_REPO if MODE == "main" else HUB_REPO + "-smoke")

adapt = {
    "model_path": str(MODEL_DIR),
    "urdu_corpus": "data/urdu/urdu_train.txt",
    "heldout": "data/urdu/urdu_heldout.txt",
    "english_corpus": "data/english/english_replay.txt",
    "english_mb": 12,
    "guard_urdu": ["data/flores_urd.txt", "data/flores_dev/flores_urd.txt"],
    "guard_english": ["data/flores_eng.txt", "data/flores_dev/flores_eng.txt"],
    "guard_belebele": True,
    "drop_overlaps": True,
    "data_dir": "data/adapt",
    "block_size": 1024,
    "english_replay_ratio": 0.1,
    "output_dir": str(OUT),
    "micro_batch_size": MICRO_BATCH,
    "lr_new_rows": 1e-3,
    "lr_lora": 2e-4,
    "lora_rank": 64,
    "lora_alpha": 128,
    "lora_dropout": 0.05,
    "precision": "fp16",
    "gradient_checkpointing": True,
    "device": "cuda",
    "seed": 0,
    "hub_repo": hub,
    "hub_private": True,
    "merged_dtype": "bfloat16",
}
if MODE == "main":
    # 150 steps of new rows only, then 800 steps of LoRA plus new rows; 32 blocks (33k tokens) per step.
    adapt.update(
        phase_a_steps=150, phase_b_steps=800, grad_accum=16, warmup_steps=50,
        log_every=10, eval_every=100, eval_texts=200, checkpoint_every=100, keep_checkpoints=2,
    )
else:
    # A 30-step schedule with both phases. Fewer blocks per step (grad_accum 4) so it is quick; the
    # memory per forward pass is the same as in the main run, which is what the smoke run checks.
    adapt.update(
        phase_a_steps=15, phase_b_steps=15, grad_accum=4, warmup_steps=3,
        log_every=1, eval_every=10, eval_texts=50, checkpoint_every=10, keep_checkpoints=2,
    )
config = {
    "urdu_corpus": {"raw_dir": "data/raw", "out_dir": "data/urdu", "flores": "data/flores_urd.txt", "max_mb": 100, "heldout_percent": 1.0},
    "adapt": adapt,
}
Path("adapt.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
print(Path("adapt.yaml").read_text())
"""

CHECK_TRAIN = """
# CHECK 3: five real training steps in fp16, covering both phases (2 steps of new rows, 3 with LoRA).
# A non-finite loss stops the run with a message; then set precision to fp32 and rerun.
from zubaan.adapt.pipeline import run_adapt
from zubaan.config import load_config

cfg = load_config("adapt.yaml").adapt
check = cfg.model_copy(update={
    "output_dir": WORK / "adapt_check", "hub_repo": None, "resume": False,
    "phase_a_steps": 2, "phase_b_steps": 3, "grad_accum": 2, "warmup_steps": 1,
    "log_every": 1, "eval_every": 5, "eval_texts": 20, "checkpoint_every": 5,
})
run_adapt(check, log=print)
rows = [json.loads(line) for line in (check.output_dir / "train_log.jsonl").read_text().splitlines()]
losses = [r["loss"] for r in rows if "loss" in r]
# The fp16 loss scaler starts high and skips a step (gradient norm inf) when it overflows. A few at the
# start are normal; every step skipped would mean the run is not learning.
skipped = sum(1 for r in rows if "loss" in r and r["grad_norm"] == float("inf"))
print(f"steps skipped by the loss scaler: {skipped} of 5")
assert len(losses) == 5 and all(x == x and abs(x) < 1e4 for x in losses), losses
assert skipped < 5, "every step overflowed: use precision fp32"
print("OK: 5 fp16 steps, finite losses", [round(x, 3) for x in losses])
"""

DATA = """
# Evaluation text first (FLORES devtest and dev), then the Urdu training text, then the token blocks.
# Prepare drops any training line that overlaps FLORES or Belebele text and counts it.
!zubaan data flores --out data
!zubaan data flores --split dev --out data/flores_dev
!zubaan data urdu --config adapt.yaml
!zubaan adapt prepare --config adapt.yaml
"""

DATA_CHECK = """
meta = json.loads(Path("data/adapt/blocks_meta.json").read_text())
print("Urdu blocks", meta["urdu"]["n_blocks"], "| tokens", meta["urdu"]["n_tokens"], "| English blocks", meta["english"]["n_blocks"])
print("guard:", meta["overlap_guard"])
assert meta["overlap_guard"]["hits"] == 0
# The laptop run (2026-11-18) gave 15,677 Urdu blocks and 16,053,605 tokens. A different number means a
# different Wikipedia dump or tokenizer: not wrong by itself, but write it down.
print("laptop reference: 15677 Urdu blocks, 16053605 tokens")
"""

SMOKE = """
# SMOKE: 20 of 30 steps, then lose the local checkpoints (as a crashed session would), then rerun.
# With a Hub repo the second run resumes from the Hub; without one it resumes from the local files.
import shutil

from zubaan.adapt.pipeline import run_adapt
from zubaan.config import load_config

cfg = load_config("adapt.yaml").adapt
if cfg.hub_repo:  # a smoke run must start from step 0, so clear any older smoke checkpoint
    from huggingface_hub import HfApi

    try:
        HfApi().delete_folder("last", repo_id=cfg.hub_repo, repo_type="model")
        print("cleared an old smoke checkpoint from", cfg.hub_repo)
    except Exception as e:
        print("no old smoke checkpoint to clear:", type(e).__name__)
shutil.rmtree(cfg.output_dir / "checkpoints", ignore_errors=True)
first = run_adapt(cfg, log=print, max_steps=20)
assert first["final_step"] == 20 and not first["finished"]
if cfg.hub_repo:
    shutil.rmtree(cfg.output_dir / "checkpoints")
    print("local checkpoints deleted; resuming from the Hub")
second = run_adapt(cfg, log=print)
assert second["resumed_from_step"] == 20 and second["finished"], second
print("OK: killed at step 20, resumed, finished at step", second["final_step"])
"""

SMOKE_REPORT = """
import statistics

rows = [json.loads(line) for line in (OUT / "train_log.jsonl").read_text().splitlines()]
steps = [r for r in rows if "loss" in r]
evals = [r for r in rows if "eval" in r]
assert all(r["loss"] == r["loss"] for r in steps), "NaN loss in the log"
tps = [r["tokens_per_s"] for r in steps[1:]]  # the first step includes CUDA warm-up
peak = max(r.get("gpu_mem_gb", 0) for r in steps)
tokens_per_step = 2 * 16 * 1024  # the main run: micro batch 2 x accumulation 16 x 1024
total_steps = 150 + 800
n_urdu = meta["urdu"]["n_blocks"]
epoch_blocks = n_urdu + round(n_urdu * 0.1 / 0.9)  # Urdu blocks plus the 10% English replay
median_tps = statistics.median(tps)
hours = total_steps * tokens_per_step / median_tps / 3600
report = {
    "gpu": gpu_name,
    "micro_batch": MICRO_BATCH,
    "median_tokens_per_s": round(median_tps),
    "peak_gpu_gb": round(peak, 2),
    "first_loss": steps[0]["loss"],
    "last_loss": steps[-1]["loss"],
    "evals": [e["eval"] for e in evals],
    "projected_main_run_hours": round(hours, 1),
    "main_run_steps": total_steps,
    "main_run_epochs": round(total_steps * 32 / epoch_blocks, 2),
}
(WORK / "smoke_report.json").write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
assert peak < 15.0, "peak GPU memory is too close to the card's limit: lower MICRO_BATCH"
print("Send back smoke_report.json. The projection above decides the step budget for the main run.")
"""

MAIN = """
from zubaan.adapt.pipeline import run_adapt
from zubaan.config import load_config

cfg = load_config("adapt.yaml").adapt
summary = run_adapt(cfg, log=print)
print({k: summary[k] for k in ("resumed_from_step", "final_step", "total_steps", "finished")})
assert summary["finished"], "the schedule did not finish: run this cell again to resume from the last checkpoint"
for e in summary["evals"]:
    print(e["step"], e["eval"])
"""

MERGE = """
# Fold LoRA and the trained rows into the weights (in float32), save a plain model with the chat template
# files, and verify the folder: English tokenizes as before, Urdu round-trips, the old rows are identical.
!zubaan adapt merge --config adapt.yaml
"""

CLEANUP = """
total = 0
for f in sorted(WORK.rglob("*")):
    if f.is_file() and f.stat().st_size > 5e6:
        total += f.stat().st_size
        print(f"{f.stat().st_size / 1e9:6.2f} GB  {f.relative_to(WORK)}")
print(f"total of files over 5 MB: {total / 1e9:.1f} GB")
assert total < 19e9, "output is close to Kaggle's 20 GB limit: delete data/raw or old checkpoints"
"""


def build(commit: str) -> dict:
    cells = [
        md(
            """
# Stage 3: adapt Qwen3-1.7B to its new Urdu tokens

Continued pretraining of the extended model: first only the 10,000 new embedding rows, then LoRA plus the new rows, on Urdu Wikipedia with 10% English Wikipedia mixed in. All logic lives in the `zubaan` package (`zubaan adapt prepare | run | merge`); this notebook installs it, checks the machine and calls it. Decisions and reasons: `docs/decisions.md`, 2026-11-18.

**Three modes, run in this order** (set `MODE` in the first code cell):

1. `check`: versions, GPU, and 5 real fp16 training steps (both phases). About 5 minutes plus data preparation. Fails loudly if the loss goes NaN.
2. `smoke`: everything in `check`, then a 30-step run that is stopped at step 20, loses its local checkpoints and resumes. It reports tokens per second and peak GPU memory and projects the main run's hours. **Send back `smoke_report.json`.**
3. `main`: the full schedule (950 steps, about 1.75 epochs), then merge, save and verify the model. Several hours; use *Save Version > Save & Run All (Commit)* and give it a Hub repo so it can resume.

Settings before running:

- Accelerator: **GPU T4 x2** (one GPU is used). Internet: **on**.
- Input: **Add Input > Notebooks** and add the finished output of `03_extend_export_qwen3` (the extended model folder).
- Secret: **HF_TOKEN** (Add-ons > Secrets), only needed if `HUB_REPO` is set. Create the repo as **private** first.
- The commit named in the install cell must be pushed to GitHub, because the notebook installs zubaan from it.
- GPU hours are limited (about 30 a week): stop the session when a mode finishes.
"""
        ),
        code(SETTINGS),
        code(INSTALL.format(commit=commit)),
        code(CHECK_CODE),
        code(CHECK_GPU),
        code(SECRET),
        md("## 1. Find the extended model"),
        code(FIND_MODEL),
        md(
            "## 2. Configuration\n\nOne YAML file drives every step, the same file the CLI reads on a laptop."
        ),
        code(CONFIG),
        md(
            "## 3. Data\n\nFLORES, the Urdu text and the token blocks. Takes about 10 minutes (Wikipedia download, English streaming, tokenizing). Training lines that overlap evaluation text are dropped and counted, not hidden."
        ),
        code(DATA),
        code(DATA_CHECK),
        md(
            "## 4. Check: five fp16 training steps\n\nCovers the new-rows phase and the LoRA phase, so an fp16 overflow shows up in minutes instead of hours."
        ),
        code(CHECK_TRAIN),
        md(
            "## 5. Smoke run (MODE = smoke)\n\nA short run that is killed and resumed. Reports speed and memory."
        ),
        code("if MODE == 'smoke':\n" + "\n".join("    " + ln for ln in SMOKE.strip().splitlines())),
        code(
            "if MODE == 'smoke':\n"
            + "\n".join("    " + ln for ln in SMOKE_REPORT.strip().splitlines())
        ),
        md(
            "## 6. Main run (MODE = main)\n\nIf the session ends, run the notebook again with the same settings: it resumes from the newest checkpoint (local, or the Hub repo)."
        ),
        code("if MODE == 'main':\n" + "\n".join("    " + ln for ln in MAIN.strip().splitlines())),
        md("## 7. Merge, save and verify (MODE = main)"),
        code("if MODE == 'main':\n" + "\n".join("    " + ln for ln in MERGE.strip().splitlines())),
        md("## 8. Output size"),
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
    out = Path(__file__).parent / "05_adapt_qwen3.ipynb"
    nb = build(commit)
    for i, cell in enumerate(nb["cells"]):
        cell["id"] = f"cell-{i}"
    out.write_text(
        json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n"
    )
    print(f"wrote {out} pinned to commit {commit}")


if __name__ == "__main__":
    main()
