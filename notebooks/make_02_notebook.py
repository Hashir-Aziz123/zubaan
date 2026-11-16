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
  batch_size: 8
  focus_lines: 50000
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
"""
    ),
    code(
        """
# Install zubaan from GitHub. Pin a commit for reproducibility by appending @<commit>.
REPO = "git+https://github.com/Hashir-Aziz123/zubaan.git"
!pip install -q "zubaan[audit] @ {REPO}"
!zubaan version
"""
    ),
    code(
        """
import torch

assert torch.cuda.is_available(), "Turn on a GPU accelerator in the notebook settings"
print(torch.cuda.get_device_name(0))
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
        "A quick check first: the base model and one variant with a few texts. Then the full run."
    ),
    code(
        """
from pathlib import Path

cfg = Path("extend.yaml").read_text()
for old, new in [
    ("quality_texts: 1000", "quality_texts: 20"),
    ("belebele_questions: null", "belebele_questions: 8"),
    ("eval_sizes: [10000, 20000]", "eval_sizes: [10000]"),
    ("init_methods: [mean, random, focus]", "init_methods: [mean]"),
]:
    assert old in cfg, old
    cfg = cfg.replace(old, new)
Path("extend_smoke.yaml").write_text(cfg)
!zubaan extend quality --config extend_smoke.yaml --out /kaggle/working/smoke.json
"""
    ),
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
