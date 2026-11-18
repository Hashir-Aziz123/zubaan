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
# Nothing to set here. This notebook only measures speed and memory; it writes nothing to the Hub.
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

Settings before running: Accelerator **GPU T4 x2** (one is used), Internet **on**, Input **Add Input > Notebooks** with the finished output of `03_extend_export_qwen3`. No Hub repo or secret is needed. About 20 minutes including data preparation.
"""
        ),
        c.code(SETTINGS),
        c.code(c.INSTALL.format(commit=commit)),
        c.code(c.CHECK_CODE),
        c.code(c.CHECK_GPU),
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
        c.md("## 6. Output size"),
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
