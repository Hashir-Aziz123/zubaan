# ruff: noqa: E501
"""Writes notebooks/06_adapt_speed_trial.ipynb.

Run: python notebooks/make_06_notebook.py [COMMIT]   (COMMIT must be pushed to GitHub)
"""

import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import adapt_cells as c  # noqa: E402

SETTINGS = """
# Speed and memory are measured without the Hub. If you set HUB_REPO (a PRIVATE repo you made), the
# notebook also tests the checkpoint and milestone upload and download on a throwaway "<repo>-test" repo.
HUB_REPO = ""            # e.g. "your-name/zubaan-qwen3-ckpt"; empty skips the Hub test
MICRO_BATCH = 2          # used by the in-process 5-step check only
GRAD_CHECKPOINT = True
PHASE_A_STEPS = 100      # unused here; the config cell shares its code with notebook 07
PHASE_B_STEPS = 700
OUT_NAME = "adapt_speed"
HUB = None
MODE = "check"           # the shared check cell reads this name
"""


def build(commit: str) -> dict:
    cells = [
        c.md(
            """
# Stage 3: speed trial for the main run

Notebook 05 proved that training works and resumes. It also showed the main run is too slow at micro batch 2 with gradient checkpointing (582 tokens/s, projected 14.9 h for 950 steps). This notebook tries four settings for a few real steps each and tells you which is fastest and still fits in GPU memory:

| setting | micro batch | gradient checkpointing |
| --- | --- | --- |
| `mb2_ckpt_on` | 2 | on (the smoke-run setting) |
| `mb4_ckpt_on` | 4 | on |
| `mb1_ckpt_off` | 1 | off |
| `mb2_ckpt_off` | 2 | off |

Each runs in its own process, so an out-of-memory error in one only marks that setting as failed. Every optimizer step still sees 32 blocks, so the winner does not change what is learned, only how fast.

**Send back `speed_report.json`** (written to the notebook's working folder), then put the winning `MICRO_BATCH` and `GRAD_CHECKPOINT` into notebook 07.

Settings before running: Accelerator **GPU T4 x2** (one is used), Internet **on**, Input **Add Input > Notebooks** with the finished output of `03_extend_export_qwen3`. A Hub repo and the `HF_TOKEN` secret are optional: with them the notebook also proves checkpoint and milestone upload and download work. About 20 minutes including data preparation.
"""
        ),
        c.code(SETTINGS),
        c.code(c.INSTALL.format(commit=commit)),
        c.code(c.CHECK_CODE),
        c.code(c.CHECK_GPU),
        c.code(c.SECRET),
        c.md("## 1. Find the extended model"),
        c.code(c.FIND_MODEL),
        c.md("## 2. Configuration"),
        c.code(c.CONFIG),
        c.md("## 3. Data\n\nThe same preparation as notebook 05."),
        c.code(c.DATA),
        c.code(c.DATA_CHECK),
        c.md(
            "## 4. Five-step fp16 check\n\nQuick proof that this session trains before the trial starts."
        ),
        c.code(c.CHECK_TRAIN),
        c.code(c.FREE_GPU),
        c.md("## 5. The trial"),
        c.code(c.SPEED),
        c.md(
            "## 6. Hub round trip (only if HUB_REPO is set)\n\nUploads a stand-in checkpoint the way the main run does, checks `last/` and the milestone copy exist, downloads both, and removes the test repo."
        ),
        c.code(c.HUB_TEST),
        c.md("## 7. Output size"),
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
    out = Path(__file__).parent / "06_adapt_speed_trial.ipynb"
    nb = build(commit)
    for i, cell in enumerate(nb["cells"]):
        cell["id"] = f"cell-{i}"
    out.write_text(
        json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n"
    )
    print(f"wrote {out} pinned to commit {commit}")


if __name__ == "__main__":
    main()
