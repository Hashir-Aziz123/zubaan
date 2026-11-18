# ruff: noqa: E501
"""Writes notebooks/07_adapt_main_qwen3.ipynb.

Run: python notebooks/make_07_notebook.py [COMMIT [HUB_REPO]]   (COMMIT must be pushed to GitHub)
"""

import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import adapt_cells as c  # noqa: E402

SETTINGS = """
# The Hub repo is required: it is what makes the run safe. Create it as PRIVATE first. It must be a
# different repo from the smoke one (notebook 05 used HUB_REPO + "-smoke", so there is no clash).
HUB_REPO = ""            # e.g. "your-name/zubaan-qwen3-ckpt"
MICRO_BATCH = 1          # from notebook 06 (T4, 2026-11-18): 850 tok/s, peak 13.3 GB; micro batch 2 without checkpointing ran out of memory
GRAD_CHECKPOINT = False  # from notebook 06: off is 44% faster than on (592 tok/s) and still fits
PHASE_A_STEPS = 100      # steps that train only the 10,000 new rows
PHASE_B_STEPS = 700      # steps that train LoRA plus the new rows; the learning rate decays to its minimum over these
# Do not change any of the lines above between sessions of the same run: a resumed run refuses a changed schedule.
import os

# Peak memory is 13.3 GB of about 14.5: let the allocator avoid fragmentation. Must be set before torch starts.
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
assert HUB_REPO, "set HUB_REPO: without it a crash or the 12-hour limit loses the run"
assert 32 % MICRO_BATCH == 0, "MICRO_BATCH must divide 32"
OUT_NAME = "adapted"
HUB = HUB_REPO
MODE = "main"
"""

MERGE_CELL = """
# Fold LoRA and the trained rows into the weights (in float32), save a plain model with the chat
# template files, and verify the folder. Needs the run to have finished.
!zubaan adapt merge --config adapt.yaml
"""


def build(commit: str) -> dict:
    cells = [
        c.md(
            """
# Stage 3: main training run (Qwen3-1.7B + 10,000 Urdu tokens)

Continued pretraining of the extended model: first only the new embedding rows, then LoRA plus the new rows, on Urdu Wikipedia with 10% English mixed in. Notebooks 05 (smoke) and 06 (speed) must have passed first.

**Built to be safe, not fast to finish:**

- A checkpoint goes to your private Hub repo every **50 steps** (`<repo>/last`), and every **200th** is kept for good in `<repo>/milestones/step-NNNNNNN`.
- If the session ends (12-hour limit, crash, disconnect), run this notebook again with the **same settings**. It finds `last/` on the Hub and resumes from there, with the same data order.
- Every 100 steps it measures held-out Urdu bits per character and English bits per character, so you can see learning and any English damage as it happens.
- Any milestone can be merged and evaluated later, e.g. `zubaan adapt merge --config adapt.yaml --milestone 400`. If you stop early, merge the newest checkpoint: usable, but its learning rate has not decayed, so it is a little noisier than a finished run.

Settings before running: Accelerator **GPU T4 x2**, Internet **on**, Input **Add Input > Notebooks** with the output of `03_extend_export_qwen3`, Secret **HF_TOKEN** attached. Use *Save Version > Save & Run All (Commit)* so it keeps running with the tab closed. Set `HUB_REPO` and the speed-trial winner in the first cell.
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
        c.md("## 3. Data\n\nThe same preparation as notebook 05 (about 10 minutes)."),
        c.code(c.DATA),
        c.code(c.DATA_CHECK),
        c.md(
            "## 4. Hub check\n\nProves the repo is private and writable before hours are spent, and says whether this run starts fresh or resumes."
        ),
        c.code(c.HUB_PREFLIGHT),
        c.md(
            "## 5. Train\n\nRun it again with the same settings if the session ends before it prints `finished`."
        ),
        c.code(c.MAIN),
        c.md("## 6. Merge, save and verify"),
        c.code(MERGE_CELL),
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
    out = Path(__file__).parent / "07_adapt_main_qwen3.ipynb"
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
