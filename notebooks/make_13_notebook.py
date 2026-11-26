# ruff: noqa: E501
"""Writes notebooks/13_tune_generate.ipynb (Stage 4, chunks 2 and 3: the chat training data).

Run: python notebooks/make_13_notebook.py [COMMIT]   (COMMIT must be pushed to GitHub)

Install, GPU and server cells come from notebook 12 (the teacher trial, which ran on Kaggle);
everything else calls zubaan.tune.generate.
"""

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "make_12_notebook", Path(__file__).parent / "make_12_notebook.py"
)
n12 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(n12)
n4, md, code = n12.n4, n12.md, n12.code

SETTINGS = """
from pathlib import Path

# Run this notebook TWICE: once with SHARD = 0 and once with SHARD = 1. Each run handles about half
# of the passages (a fixed split by passage id), so the two outputs together are the whole dataset.
SHARD = 0
N_SHARDS = 2

TEACHER = ("Qwen/Qwen3-14B-AWQ", "qwen3-14b-awq")  # picked after the teacher trial (notebook 12)
PER_FORMAT = 2  # questions per passage
OVERGENERATE = 1.15  # 15% spare for the filters
WORKERS = 128  # requests in flight
MAX_MODEL_LEN = 8192
THINK_MAX_TOKENS = 4096  # the trial's 14B needed at most 2,476
NO_THINK_MAX_TOKENS = 2048  # 1,024 cut off 18% of long grounded answers in the trial
SEED = 0
OUT = Path(f"/kaggle/working/data_shard{SHARD}")
OUT.mkdir(parents=True, exist_ok=True)
assert 0 <= SHARD < N_SHARDS
"""

DATA = """
# Evaluation text first (FLORES devtest and dev), then the Urdu training text, as in Stage 3.
!zubaan data flores --out data
!zubaan data flores --split dev --out data/flores_dev
!zubaan data urdu
"""

ENGLISH = """
# English Wikipedia paragraphs, built exactly as Stage 3's English replay text (streamed, cleaned,
# guarded against FLORES and Belebele English). About 5 minutes.
from zubaan.adapt.data import build_english_replay, iter_english_wikipedia
from zubaan.adapt.pipeline import belebele_passages
from zubaan.io import read_lines

GUARD = {
    "urd": read_lines("data/flores_urd.txt") + read_lines("data/flores_dev/flores_urd.txt") + belebele_passages("urd_Arab"),
    "eng": read_lines("data/flores_eng.txt") + read_lines("data/flores_dev/flores_eng.txt") + belebele_passages("eng_Latn"),
}
english_meta = build_english_replay(iter_english_wikipedia(), Path("data/english/english_replay.txt"), GUARD["eng"], max_mb=12)
print({k: v for k, v in english_meta.items() if k != "source"})
"""

PASSAGES = """
import json

from zubaan.tune.generate import (
    DEFAULT_MIX,
    passage_counts,
    sample_passages,
    shard_of,
    split_by_format,
    to_jsonl,
)

counts = passage_counts(DEFAULT_MIX, PER_FORMAT, OVERGENERATE)
print("passages needed (all shards):", counts)
corpora = {"urd": "data/urdu/urdu_train.txt", "eng": "data/english/english_replay.txt"}
plan, passage_stats = {}, {}
for lang, path in corpora.items():
    need = {fmt: counts.get((lang, fmt), 0) for fmt in ("grounded", "standalone")}
    # Every shard samples the same full list (same seed), then keeps its own part.
    passages, passage_stats[lang] = sample_passages(read_lines(path), sum(need.values()), GUARD[lang], lang=lang, seed=SEED)
    for fmt, ps in split_by_format(passages, need).items():
        plan[(lang, fmt)] = [p for p in ps if shard_of(p.id, N_SHARDS) == SHARD]
mine = [p for ps in plan.values() for p in ps]
(OUT / "passages.jsonl").write_text(to_jsonl(mine), encoding="utf-8")
(OUT / "passage_stats.json").write_text(json.dumps(passage_stats, indent=1))
print({f"{k[0]} {k[1]}": len(v) for k, v in plan.items()}, "passages in this shard")
"""

QUESTIONS = """
from zubaan.tune.client import ChatClient
from zubaan.tune.generate import write_questions

model, served = TEACHER
proc = start_teacher(model, served)
client = ChatClient("http://127.0.0.1:8000", served)
t0 = time.perf_counter()
questions, question_stats = [], {}
for (lang, fmt), ps in plan.items():
    qs, st = write_questions(client, ps, per_format=PER_FORMAT, formats=[fmt], workers=WORKERS)
    questions += qs
    question_stats[f"{lang} {fmt}"] = {k: v for k, v in st.items() if k != "failures"}
question_stats["seconds"] = round(time.perf_counter() - t0, 1)
(OUT / "questions.jsonl").write_text(to_jsonl(questions), encoding="utf-8")
(OUT / "question_stats.json").write_text(json.dumps(question_stats, indent=1))
print(question_stats)
for q in questions[:3]:
    print(q.lang, q.format, q.kind, "|", q.text)
"""

ANSWERS = """
import dataclasses
import threading

from zubaan.tune.client import QWEN3_NO_THINKING, QWEN3_THINKING
from zubaan.tune.generate import assign_modes, generate_answers

jobs = assign_modes(questions, DEFAULT_MIX, seed=SEED)
print(len(jobs), "answers to write;", sum(t for _, t in jobs), "with thinking")
lock = threading.Lock()
progress = {"n": 0, "tokens": 0, "t0": time.perf_counter()}
answers_file = (OUT / "answers.jsonl").open("w", encoding="utf-8")


def save(rec):
    # Written as each answer arrives, so the file holds everything finished if the session stops.
    with lock:
        answers_file.write(json.dumps(rec, ensure_ascii=False) + "\\n")
        answers_file.flush()
        progress["n"] += 1
        progress["tokens"] += rec["completion_tokens"] or 0
        if progress["n"] % 200 == 0:
            secs = time.perf_counter() - progress["t0"]
            print(f"{progress['n']}/{len(jobs)} answers, {progress['tokens'] / secs:.0f} tokens/s, {secs / 60:.0f} min")


try:
    records = generate_answers(
        client, questions, mine, teacher=served, jobs=jobs, workers=WORKERS, on_record=save,
        think_sampling=dataclasses.replace(QWEN3_THINKING, max_tokens=THINK_MAX_TOKENS),
        no_think_sampling=dataclasses.replace(QWEN3_NO_THINKING, max_tokens=NO_THINK_MAX_TOKENS),
    )
finally:
    answers_file.close()
    stop_teacher(proc)
answer_seconds = round(time.perf_counter() - progress["t0"], 1)
"""

REPORT = """
import shutil

from zubaan.tune.generate import summarize_answers

summary = {
    "shard": SHARD,
    "n_shards": N_SHARDS,
    "teacher": TEACHER[0],
    "vllm": VLLM_VERSION,
    "gpu": name,
    "settings": {"per_format": PER_FORMAT, "overgenerate": OVERGENERATE, "think_max_tokens": THINK_MAX_TOKENS,
                 "no_think_max_tokens": NO_THINK_MAX_TOKENS, "workers": WORKERS, "seed": SEED},
    "english_replay": {k: v for k, v in english_meta.items() if k != "source"},
    "questions": question_stats,
    "answers": len(records),
    "answer_seconds": answer_seconds,
    "new_tokens": progress["tokens"],
    "groups": summarize_answers(records),
}
(OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1))
for key, g in summary["groups"].items():
    print(key.ljust(40), {k: g[k] for k in ("n", "failed_requests", "answered", "hit_limit", "loops", "urdu_answers", "source_in_answer")})
shutil.make_archive(f"/kaggle/working/tune_data_shard{SHARD}", "zip", OUT)
print(f"download /kaggle/working/tune_data_shard{SHARD}.zip from the Output tab")
"""


def build(commit: str) -> dict:
    cells = [
        md(
            """
# Stage 4, chunks 2 and 3: generate the chat training data

The teacher picked in the trial (`Qwen/Qwen3-14B-AWQ`, notebook 12) writes questions about Urdu and English Wikipedia passages from the training split and answers each question once, with thinking on or off. The mix (`DEFAULT_MIX`, `docs/decisions.md`, 2026-11-26) keeps 10,000 conversations after filtering: Urdu 4,000 grounded with thinking, 2,000 grounded without, 2,000 standalone without; English 1,000, 500 and 500. 15% extra is generated for the filters.

**Run it twice**, with `SHARD = 0` and then `SHARD = 1` (settings cell). Each run is about 3 to 4 hours.

- Accelerator: **GPU T4 x2** (both are used). Internet: **on**. No input, no secret.
- Run with **Save Version > Save & Run All**.
- Send back `tune_data_shard0.zip` and `tune_data_shard1.zip` (Output tab).
"""
        ),
        code(n4.INSTALL.format(commit=commit)),
        code(n4.CHECK_CODE),
        code(n4.CHECK_GPU + n12.TWO_GPUS),
        code(SETTINGS),
        md("## 1. Evaluation texts, Urdu and English Wikipedia, and this shard's passages"),
        code(DATA),
        code(ENGLISH),
        code(PASSAGES),
        md("## 2. Install vLLM (isolated environment)"),
        code(n4.INSTALL_VLLM),
        md("## 3. Questions, then answers (saved as they arrive)"),
        code(n12.SERVER),
        code(QUESTIONS),
        code(ANSWERS),
        md("## 4. Counts and the zip"),
        code(REPORT),
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
    out = Path(__file__).parent / "13_tune_generate.ipynb"
    nb = build(commit)
    for i, cell in enumerate(nb["cells"]):
        cell["id"] = f"cell-{i}"
    out.write_text(
        json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n"
    )
    print(f"wrote {out} pinned to commit {commit}")


if __name__ == "__main__":
    main()
