# ruff: noqa: E501
"""Writes notebooks/02_extend_quality_qwen3.ipynb. Run: python notebooks/make_02_notebook.py"""

import json
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


CONFIG = """
from pathlib import Path

config = '''
urdu_corpus:
  raw_dir: data/raw
  out_dir: data/urdu
  flores: data/flores_urd.txt
  max_mb: 100
  heldout_percent: 1.0

extend:
  base_model: Qwen/Qwen3-1.7B
  corpus: data/urdu/urdu_train.txt
  heldout: data/urdu/urdu_heldout.txt
  eval_urdu: data/flores_urd.txt
  eval_english: data/flores_eng.txt
  n_new_tokens: 10000
  sizes: [5000, 10000, 20000, 32000]
  eval_sizes: [10000, 20000]
  init_methods: [mean, random, focus]
  min_count: 2
  merges_file: /kaggle/working/merges.json
  output_dir: /kaggle/working/extended
  dtype: float32
  device: cuda
  quality_texts: 1000
  max_chars: 400
  belebele_questions: null
  batch_size: 4
  max_batch_tokens: 2048
  focus_lines: 30000
  quality_output: /kaggle/working/extend_quality_qwen3-1.7b.json
'''
Path("extend.yaml").write_text(config)
"""

cells = [
    md(
        """
# Stage 2: quality next to cost, Qwen3-1.7B

Extends Qwen3-1.7B with 10,000 and 20,000 Urdu tokens, initializes the new embeddings three ways (mean of pieces, FOCUS-style, random), and measures bits per character and Belebele accuracy for every variant next to the base model. **Nothing is trained**: the new rows are only initialized, so quality is expected to be worse than the base. Stage 3 trains them.

All logic lives in the `zubaan` package; this notebook only calls the CLI.

Settings before running:

- Accelerator: **GPU T4** (one GPU is enough) or P100.
- Internet: **on**.
- No secret is needed: Qwen3, FLORES-200 (Meta archive), Urdu Wikipedia and Belebele are not gated.

The model runs in **float32** (about 7 GB of the 16 GB GPU): T4 and P100 have no bfloat16, and float32 avoids float16 overflow. The run saves after every variant, so if the session stops, run all cells again and finished variants are skipped. Expected run time: about 2 hours.

**Safest way to run it:** use *Save Version > Save & Run All (Commit)*. That runs the notebook top to bottom in a fresh container without a browser attached, and keeps `/kaggle/working` as the output even if you close the tab. If a run stops, the results file records the peak GPU and RAM of every finished variant (`memory`), and running again skips finished variants.
"""
    ),
    code(
        """
# Install zubaan at one exact commit. Do not edit the next line except to change the commit.
COMMIT = "8aefa55"
REPO = f"git+https://github.com/Hashir-Aziz123/zubaan.git@{COMMIT}"
!pip install -q "zubaan[audit] @ {REPO}"
# Kaggle may already hold an older copy with the same version number, which pip keeps.
# Force our commit (dependencies are already in place).
!pip install -q --force-reinstall --no-deps "zubaan @ {REPO}"
!zubaan version
"""
    ),
    code(
        """
# CHECK 1: the installed code is the commit above, not a cached older copy.
# If this fails, use Run > Restart session and run the cells again from the top.
import json
from importlib import metadata

direct = json.loads(metadata.distribution("zubaan").read_text("direct_url.json"))
installed = direct["vcs_info"]["commit_id"]
print("installed commit:", installed)
assert installed.startswith(COMMIT), f"installed {installed}, wanted {COMMIT}"

import inspect

import zubaan.extend.init_emb as _ie
import zubaan.extend.quality_run as _qr

assert ".to(w.device)" in inspect.getsource(_ie.random_rows), "old random_rows loaded: restart the session"
assert ".cpu()" in inspect.getsource(_qr.build_focus_rows), "old build_focus_rows loaded: restart the session"
print("OK: code version verified")
"""
    ),
    code(
        """
import os
import shutil

# Quiet progress bars: thousands of redraw lines can overwhelm the notebook front end.
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TQDM_DISABLE"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch

# CHECK 2: the machine has what the run needs.
assert torch.cuda.is_available(), "Turn on a GPU accelerator in the notebook settings"
print(torch.cuda.get_device_name(0))
gpu_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
with open("/proc/meminfo") as f:
    mem = {line.split(":")[0]: int(line.split()[1]) for line in f}
ram_gb = mem["MemAvailable"] / 1e6
disk_gb = shutil.disk_usage("/kaggle/working").free / 1e9
print(f"GPU {gpu_gb:.1f} GB, RAM available {ram_gb:.1f} GB, disk free {disk_gb:.1f} GB")
assert gpu_gb >= 15, "need a 16 GB GPU (T4 or P100); a smaller GPU will run out of memory"
assert ram_gb >= 20, "need about 20 GB free RAM (the run peaked at 14.4 GB); restart the session to free memory"
assert disk_gb >= 10, "need 10 GB free disk"
print("OK: machine is big enough")
"""
    ),
    md("## 1. Data: FLORES (evaluation) and the Urdu Wikipedia training text"),
    code(CONFIG),
    code(
        """
!zubaan data flores --out data
!zubaan data urdu --config extend.yaml
"""
    ),
    md(
        "## 2. Learn the merges (tokenizer only, about 2 minutes)\n\n"
        "Check the printout: the number of learned merges must be at least the largest `eval_sizes`."
    ),
    code("!zubaan extend learn --config extend.yaml"),
    code("!zubaan extend sweep --config extend.yaml --out /kaggle/working/extend_sweep.json"),
    md(
        "## 3. Quality\n\n"
        "**CHECK 3, rehearsal (about 10 minutes).** It runs the exact full pipeline on a few texts: "
        "the base model and all 6 variants (10k and 20k tokens, mean, random and FOCUS), on the GPU. "
        "The full run below starts only if every variant passes, so a bug shows up here after minutes "
        "instead of after hours."
    ),
    code(
        """
from pathlib import Path

cfg = Path("extend.yaml").read_text()
for old, new in [
    ("quality_texts: 1000", "quality_texts: 12"),
    ("belebele_questions: null", "belebele_questions: 4"),
    ("quality_output: /kaggle/working/extend_quality_qwen3-1.7b.json", "quality_output: /kaggle/working/rehearsal.json"),
]:
    assert old in cfg, old
    cfg = cfg.replace(old, new)
Path("extend_rehearsal.yaml").write_text(cfg)
Path("/kaggle/working/rehearsal.json").unlink(missing_ok=True)
!zubaan extend quality --config extend_rehearsal.yaml
"""
    ),
    code(
        """
import json
import math

rehearsal = json.loads(Path("/kaggle/working/rehearsal.json").read_text())
want = {"base", "n10000-mean", "n10000-random", "n10000-focus", "n20000-mean", "n20000-random", "n20000-focus"}
got = set(rehearsal["variants"])
assert got == want, f"missing variants: {want - got}"
for key, row in rehearsal["variants"].items():
    for name, r in row["bpc"].items():
        assert math.isfinite(r["bits_per_character"]), (key, name)
    print(key, {k: round(v["bits_per_character"], 2) for k, v in row["bpc"].items()}, row["memory"])
worst_gpu = max(r["memory"].get("peak_gpu_gb", 0) for r in rehearsal["variants"].values())
assert worst_gpu < 14, f"peak GPU {worst_gpu} GB is too close to 16 GB; lower max_batch_tokens in the config"
print("REHEARSAL PASSED: every variant builds, scores and saves. Safe to start the full run.")
"""
    ),
    md("## 4. Full run (hours). Finished variants are skipped if you have to rerun it."),
    code("!zubaan extend quality --config extend.yaml"),
    md(
        "Download `extend_quality_qwen3-1.7b.json` and `extend_sweep.json` from the Output tab into "
        "your local `reports/` folder and send them back. Lower bits per character is better; "
        "Belebele chance is 0.25."
    ),
]

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}
out = Path(__file__).with_name("02_extend_quality_qwen3.ipynb")
out.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
print("wrote", out)
