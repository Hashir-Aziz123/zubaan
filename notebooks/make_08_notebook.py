# ruff: noqa: E501
"""Writes notebooks/08_adapt_eval_qwen3.ipynb.

Run: python notebooks/make_08_notebook.py [COMMIT]   (COMMIT must be pushed to GitHub and contain
src/zubaan/adapt/evaluate.py)
"""

import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import adapt_cells as c  # noqa: E402

# Stage 2 numbers (reports/extend_quality_qwen3-1.7b.json) that this notebook must reproduce for the
# base model, and the hash of the held-out file they were measured on.
HELDOUT_SHA = "19c4f1b63b41b8a91cf8aa345d0f0e83945f33e403cc78c1f6f52ed7e7c34ada"

FIND_MODELS = """
from pathlib import Path

# Two folders written by zubaan come from Kaggle inputs: the extended model before training
# (output of notebook 03) and the adapted model (output of notebook 07). The marker file tells them apart.
INPUT_ROOT = Path("/kaggle/input")
markers = sorted(INPUT_ROOT.rglob("zubaan_extension.json"))
assert markers, "no zubaan model found under /kaggle/input: add the outputs of notebooks 03 and 07 (Add Input > Notebooks)"
trained, untrained = [], []
for m in markers:
    (trained if json.loads(m.read_text()).get("trained") else untrained).append(m.parent)
for name, folders in [("adapted (trained)", trained), ("extended (untrained)", untrained)]:
    print(name, [str(f) for f in folders])
assert len(trained) == 1, f"need exactly one trained model folder, found {len(trained)}: add only notebook 07's output (or remove the others)"
assert len(untrained) >= 1, "need the untrained extended model: add the output of notebook 03"
ADAPTED_DIR, EXTENDED_DIR = trained[0], untrained[0]
for d in (ADAPTED_DIR, EXTENDED_DIR):
    assert (d / "tokenizer.json").exists(), f"no tokenizer.json in {d}"
    assert (d / "model.safetensors").exists() or list(d.glob("model-*.safetensors")), f"no weights in {d}"
print(json.dumps(json.loads((ADAPTED_DIR / "zubaan_extension.json").read_text()), indent=1))

# Both tokenizers must be identical: training never touches the tokenizer.
a = json.loads((ADAPTED_DIR / "tokenizer.json").read_text())
b = json.loads((EXTENDED_DIR / "tokenizer.json").read_text())
assert a == b, "adapted and extended tokenizer.json differ"
print("OK: adapted and extended tokenizers are identical")
"""

CONFIG = """
import yaml

BASE = "Qwen/Qwen3-1.7B"
evaluate = {
    "models": {"base": BASE, "extended-untrained": str(EXTENDED_DIR), "adapted": str(ADAPTED_DIR)},
    "untrained": "extended-untrained",
    "adapted": "adapted",
    "heldout": "data/urdu/urdu_heldout.txt",
    "eval_urdu": "data/flores_urd.txt",
    "eval_english": "data/flores_eng.txt",
    # The Stage 2 protocol: 1000 texts of at most 400 characters per set, all 900 Belebele questions, float32.
    "quality_texts": 1000,
    "max_chars": 400,
    "belebele_questions": None,
    "dtype": "float32",
    "device": "cuda",
    "batch_size": 4,
    "max_batch_tokens": 2048,
    "output": "/kaggle/working/adapt_quality.json",
}
config = {
    "urdu_corpus": {"raw_dir": "data/raw", "out_dir": "data/urdu", "flores": "data/flores_urd.txt", "max_mb": 100, "heldout_percent": 1.0},
    "evaluate": evaluate,
}
Path("evaluate.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
print(Path("evaluate.yaml").read_text())
"""

DATA = """
# The same evaluation text as Stage 2: FLORES devtest, and the held-out Urdu Wikipedia lines.
!zubaan data flores --out data
!zubaan data urdu --config evaluate.yaml
"""

DATA_CHECK = f"""
import hashlib

meta = json.loads(Path("data/urdu/urdu_meta.json").read_text())
sha = meta["files"]["heldout"]["sha256"]
print("held-out sha256:", sha)
# Stage 2 measured the base model on a file with this hash. A different file would make the rows
# incomparable with the Stage 2 table, so stop.
assert sha == "{HELDOUT_SHA}", "the held-out Urdu file differs from the one Stage 2 used"
print("OK: same held-out file as Stage 2")
"""

REHEARSAL = """
# CHECK 3, rehearsal (a few minutes): the adapted model only, on 12 texts and 4 Belebele questions, written
# to a separate file. The full run starts only if this works.
import copy
import math

rh = copy.deepcopy(config)
rh["evaluate"].update({"models": {"adapted": str(ADAPTED_DIR)}, "untrained": None, "adapted": None,
                       "quality_texts": 12, "belebele_questions": 4, "output": "/kaggle/working/rehearsal.json"})
Path("evaluate_rehearsal.yaml").write_text(yaml.safe_dump(rh, sort_keys=False))
Path("/kaggle/working/rehearsal.json").unlink(missing_ok=True)
!zubaan adapt eval --config evaluate_rehearsal.yaml

row = json.loads(Path("/kaggle/working/rehearsal.json").read_text())["variants"]["adapted"]
for name, r in row["bpc"].items():
    assert math.isfinite(r["bits_per_character"]), name
print({k: round(v["bits_per_character"], 2) for k, v in row["bpc"].items()}, row["memory"])
assert row["memory"].get("peak_gpu_gb", 0) < 14, "peak GPU memory too close to 16 GB: lower max_batch_tokens"
print("REHEARSAL PASSED")
"""

FULL = """
# Base (downloaded), extended-untrained and adapted, all on the same texts. Finished models are skipped
# if the session has to be run again.
!zubaan adapt eval --config evaluate.yaml
"""

REPRODUCE = """
# The base model must reproduce its Stage 2 numbers (same texts, same dtype); if it does not, the
# protocol drifted and the comparison below cannot be trusted.
res = json.loads(Path("/kaggle/working/adapt_quality.json").read_text())
base = res["variants"]["base"]
stage2 = {"urdu_heldout": 2.310, "urdu_flores": 2.339, "english_flores": 1.208}
for name, want in stage2.items():
    got = base["bpc"][name]["bits_per_character"]
    print(f"base {name}: now {got:.3f}, Stage 2 {want:.3f}")
    assert abs(got - want) < 0.01, f"base {name} differs from Stage 2 by {got - want:+.3f}"
print("OK: the base model reproduces the Stage 2 table")
"""


def build(commit: str) -> dict:
    cells = [
        c.md(
            """
# Stage 3: evaluate the adapted model (Qwen3-1.7B + 10,000 Urdu tokens)

Scores three models on **the same texts** with the Stage 2 protocol: the base model, the extended model before training, and the adapted model from notebook 07. Bits per character on held-out Urdu, FLORES Urdu and FLORES English, and Belebele accuracy (Urdu and English, 900 questions each, with 95% intervals). It then prints the Stage 3 success checks as PASS or FAIL.

Nothing is trained here. It takes about an hour on a T4.

Settings before running: Accelerator **GPU T4**, Internet **on** (downloads the base model, Belebele and FLORES), Input **Add Input > Notebooks**: the output of `03_extend_export_qwen3` **and** the output of `07_adapt_main_qwen3`. No secret is needed. Use *Save Version > Save & Run All (Commit)*.

Models run in float32 (as in Stage 2). The adapted model is stored as bfloat16; loading it as float32 is exact, so the comparison is fair.
"""
        ),
        c.code(c.INSTALL.format(commit=commit)),
        c.code(
            c.CHECK_CODE
            + '\nimport zubaan.adapt.evaluate  # noqa: F401  (the evaluation code is in this commit)\nprint("OK: evaluation code present")\n'
        ),
        c.code(c.CHECK_GPU),
        c.md("## 1. Find the two local models"),
        c.code("import json\n" + FIND_MODELS),
        c.md("## 2. Configuration"),
        c.code("from pathlib import Path\n" + CONFIG),
        c.md("## 3. Data"),
        c.code(DATA),
        c.code(DATA_CHECK),
        c.md("## 4. Rehearsal"),
        c.code(REHEARSAL),
        c.md("## 5. Full evaluation"),
        c.code(FULL),
        c.code(REPRODUCE),
        c.md(
            "Download `adapt_quality.json` from the Output tab into your local `reports/` folder and send it back. "
            "Lower bits per character is better; Belebele chance is 0.25. The checks above (PASS or FAIL) are printed at the end of the full run."
        ),
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
    out = Path(__file__).parent / "08_adapt_eval_qwen3.ipynb"
    nb = build(commit)
    for i, cell in enumerate(nb["cells"]):
        cell["id"] = f"cell-{i}"
    out.write_text(
        json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n"
    )
    print(f"wrote {out} pinned to commit {commit}")


if __name__ == "__main__":
    main()
