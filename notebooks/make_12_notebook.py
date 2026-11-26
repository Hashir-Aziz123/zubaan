# ruff: noqa: E501
"""Writes notebooks/12_tune_teacher_trial.ipynb (Stage 4, chunk 0: pick the teacher).

Run: python notebooks/make_12_notebook.py [COMMIT]   (COMMIT must be pushed to GitHub)

The install, code check, GPU check and vLLM install cells are notebook 04's, which ran on Kaggle;
they are imported from make_04_notebook.py. Everything else calls zubaan.tune.generate.
"""

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "make_04_notebook", Path(__file__).parent / "make_04_notebook.py"
)
n4 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(n4)
md, code = n4.md, n4.code

TWO_GPUS = """
# Both teachers are split over the two T4s (tensor parallel 2): Qwen3-8B in float16 is about 16 GB
# of weights and does not fit one 16 GB T4.
gpus = [x for x in r.stdout.splitlines() if x.strip()]
assert len(gpus) == 2, f"found {len(gpus)} GPU(s); pick GPU T4 x2 in the notebook settings"
print("OK: two GPUs")
"""

SETTINGS = """
# Teachers, in order. The FIRST one also writes the questions, so every teacher answers the same ones.
TEACHERS = [
    ("Qwen/Qwen3-8B", "qwen3-8b"),  # float16, about 16 GB
    ("Qwen/Qwen3-14B-AWQ", "qwen3-14b-awq"),  # 4-bit AWQ, about 10 GB
]
N_PASSAGES = 50  # Urdu Wikipedia paragraphs of 60 to 250 words from the training split
PER_FORMAT = 2  # questions per passage and format (grounded, standalone): 200 questions in all
READ_PASSAGES = 15  # passages in reading.md (all are in the JSONL files and the summary)
MAX_MODEL_LEN = 8192
WORKERS = 64  # requests in flight, so vLLM batches them
SEED = 0
"""

DATA = """
# Evaluation text first (FLORES devtest and dev, Belebele), then the Urdu training text, as in Stage 3.
# Passages come only from urdu_train.txt (never the held-out lines) and are checked against all of it.
!zubaan data flores --out data
!zubaan data flores --split dev --out data/flores_dev
!zubaan data urdu
"""

PASSAGES = """
import json
from pathlib import Path

from zubaan.adapt.pipeline import belebele_passages
from zubaan.io import read_lines
from zubaan.tune.generate import sample_passages, to_jsonl

OUT = Path("/kaggle/working/trial")
OUT.mkdir(parents=True, exist_ok=True)
guard = read_lines("data/flores_urd.txt") + read_lines("data/flores_dev/flores_urd.txt") + belebele_passages("urd_Arab")
print("guard texts:", len(guard))
lines = read_lines("data/urdu/urdu_train.txt")
passages, passage_stats = sample_passages(lines, N_PASSAGES, guard, seed=SEED)
(OUT / "passages.jsonl").write_text(to_jsonl(passages), encoding="utf-8")
(OUT / "passage_stats.json").write_text(json.dumps(passage_stats, indent=1))
print(passage_stats)
print(passages[0].text[:300])
"""

SERVER = """
import os
import signal
import time
import urllib.request

from zubaan.tune.generate import teacher_serve_command

ENV = dict(os.environ, PATH="/tmp/vllm-env/bin:" + os.environ["PATH"])


def gpu_used_mb():
    out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout
    return [int(x) for x in out.split()]


def start_teacher(model, served):
    log = OUT / f"vllm_{served}.log"
    cmd = teacher_serve_command(model, served, tensor_parallel=2, max_model_len=MAX_MODEL_LEN)
    print(" ".join(cmd))
    proc = subprocess.Popen(cmd, stdout=open(log, "w"), stderr=subprocess.STDOUT, env=ENV, start_new_session=True)
    for _ in range(240):  # up to 40 minutes: download, load, CUDA graph capture
        time.sleep(10)
        if proc.poll() is not None:
            break
        try:
            with urllib.request.urlopen("http://127.0.0.1:8000/v1/models", timeout=10) as r:
                if served in r.read().decode():
                    print("server up:", served)
                    return proc
        except Exception:
            pass
    print(log.read_text()[-4000:])
    raise RuntimeError(f"vLLM did not start for {model}: see {log}")


def stop_teacher(proc):
    os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    for _ in range(60):
        time.sleep(5)
        if proc.poll() is not None and max(gpu_used_mb()) < 1000:
            break
    print("stopped; GPU memory in use (MB):", gpu_used_mb())
"""

RUN = """
from zubaan.tune.client import ChatClient
from zubaan.tune.generate import generate_answers, write_questions

questions = None
timings = {}
for model, served in TEACHERS:
    proc = start_teacher(model, served)
    try:
        client = ChatClient("http://127.0.0.1:8000", served)
        if questions is None:
            t0 = time.perf_counter()
            questions, question_stats = write_questions(client, passages, per_format=PER_FORMAT, workers=WORKERS)
            question_stats["teacher"] = model
            question_stats["seconds"] = round(time.perf_counter() - t0, 1)
            (OUT / "questions.jsonl").write_text(to_jsonl(questions), encoding="utf-8")
            (OUT / "question_stats.json").write_text(json.dumps(question_stats, ensure_ascii=False, indent=1))
            for q in questions[:4]:
                print(q.format, q.kind, "|", q.text)
        t0 = time.perf_counter()
        records = generate_answers(client, questions, passages, teacher=served, workers=WORKERS)
        timings[served] = round(time.perf_counter() - t0, 1)
        # Saved per teacher, so a failure in the second teacher keeps the first one's answers.
        (OUT / f"answers_{served}.jsonl").write_text(to_jsonl(records), encoding="utf-8")
        tokens = sum(r["completion_tokens"] or 0 for r in records)
        print(f"{served}: {len(records)} answers, {tokens} new tokens in {timings[served]} s ({tokens / max(timings[served], 0.1):.0f} tokens/s)")
    finally:
        stop_teacher(proc)
"""

REPORT = """
from zubaan.tune.generate import reading_markdown, summarize_answers

records = []
for _, served in TEACHERS:
    path = OUT / f"answers_{served}.jsonl"
    if path.exists():
        records += [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines()]
summary = {
    "teachers": [m for m, _ in TEACHERS],
    "vllm": VLLM_VERSION,
    "gpu": name,
    "passages": passage_stats,
    "questions": {k: v for k, v in question_stats.items() if k != "failures"},
    "answer_seconds": timings,
    "groups": summarize_answers(records),
}
(OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1))
(OUT / "reading.md").write_text(reading_markdown(passages, questions, records, max_passages=READ_PASSAGES), encoding="utf-8")
cols = ["n", "failed_requests", "with_think_block", "think_closed", "answered", "hit_limit", "loops", "urdu_answers", "source_in_answer", "source_in_think", "completion_tokens_median"]
print("group".ljust(42), " ".join(c[:9].rjust(9) for c in cols))
for key, g in summary["groups"].items():
    print(key.ljust(42), " ".join(str(g[c]).rjust(9) for c in cols))
"""

ZIP = """
import shutil

shutil.make_archive("/kaggle/working/teacher_trial", "zip", OUT)
print("download /kaggle/working/teacher_trial.zip from the Output tab")
"""


def build(commit: str) -> dict:
    cells = [
        md(
            """
# Stage 4, chunk 0: teacher trial

Picks the teacher that will write Stage 4's chat training data, **by reading its Urdu**. Two larger Qwen3 models answer the same Urdu questions about the same 50 Urdu Wikipedia passages, each with thinking on and off:

- `Qwen/Qwen3-8B` in float16, split over both T4s;
- `Qwen/Qwen3-14B-AWQ` (4-bit), also split over both T4s.

The first teacher writes the questions: per passage, 2 **grounded** requests (the user shows the passage and asks about it) and 2 **standalone** questions (they make sense without the passage; the teacher sees the passage as hidden background, which is dropped from the training conversation). Both formats are tried so the author can choose the mix.

Nothing here trains. Settings before running:

- Accelerator: **GPU T4 x2** (both are used). **Not P100**: vLLM needs compute capability 7.0 or higher.
- Internet: **on**. No input and no secret are needed (the models are public).

Expected time: about 1 to 1.5 hours (vLLM install about 10 minutes, Urdu Wikipedia about 5, each teacher about 10 to 15 to download and start, then the answers).

Send back `teacher_trial.zip` (Output tab). Read `reading.md` in it; `summary.json` has the automatic counts.
"""
        ),
        code(n4.INSTALL.format(commit=commit)),
        code(n4.CHECK_CODE),
        code(n4.CHECK_GPU + TWO_GPUS),
        code(SETTINGS),
        md("## 1. Urdu passages from the training split, checked against the evaluation texts"),
        code(DATA),
        code(PASSAGES),
        md("## 2. Install vLLM (isolated environment)"),
        code(n4.INSTALL_VLLM),
        md("## 3. Questions from the first teacher, answers from every teacher"),
        code(SERVER),
        code(RUN),
        md(
            """
## 4. Counts and the reading file

Columns: `with_think_block` and `think_closed` (thinking on should close every block), `answered` (a non-empty answer after the block), `hit_limit` (stopped at `max_tokens`), `loops` (an 8-word run repeated more than twice), `urdu_answers` (answers at least 90% Urdu script), `source_in_*` (mentions "the passage", "متن کے مطابق" and similar; it only matters for standalone questions). These are heuristics that flag replies for reading, not verdicts.
"""
        ),
        code(REPORT),
        code(ZIP),
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
    out = Path(__file__).parent / "12_tune_teacher_trial.ipynb"
    nb = build(commit)
    for i, cell in enumerate(nb["cells"]):
        cell["id"] = f"cell-{i}"
    out.write_text(
        json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n"
    )
    print(f"wrote {out} pinned to commit {commit}")


if __name__ == "__main__":
    main()
