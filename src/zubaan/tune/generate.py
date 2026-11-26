"""Generated chat data grounded in Wikipedia passages (Stage 4, chunks 0, 2 and 3).

A larger Qwen3 (the teacher) first writes questions about a passage, then answers each question
twice, with thinking on and off. Two formats of conversation are made:

* ``grounded``: the user shows the passage and asks about it. The passage is part of the
  user turn, in the teacher's request and in the training conversation alike.
* ``standalone``: the question makes sense without the passage. The teacher sees the passage
  as background in a system message, so its facts come from the passage rather than memory;
  the training conversation drops that system message and keeps only the question.

The teacher's text is never tokenized here: it is plain text, tokenized with the extended
tokenizer at training time. Passages come from the training split, never the held-out lines,
and are checked against the evaluation texts (FLORES, Belebele) with the 8-word shingle test.
"""

from __future__ import annotations

import json
import random
import re
import statistics
import zlib
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass

from zubaan.data.urdu import flores_shingles, overlaps_flores
from zubaan.export.check import urdu_letter_share
from zubaan.text.normalize import split_words
from zubaan.tune.client import (
    QWEN3_NO_THINKING,
    QWEN3_THINKING,
    ChatClient,
    Sampling,
    parallel_map,
)
from zubaan.tune.filters import has_loop, mentions_source, split_reply

FORMATS = ("grounded", "standalone")

# Question kinds per format, used in this order and cycled when more questions are asked for.
KINDS = {
    "grounded": ("summarize", "explain", "list", "factual"),
    "standalone": ("factual", "explain", "compare", "open-ended"),
}

KIND_HINTS = {
    "summarize": "ask for a short summary",
    "explain": "ask why or how something in it happens or matters",
    "list": "ask for a list of points, items or steps",
    "factual": "ask for a specific fact (who, what, when, where, how many)",
    "compare": "ask to compare two things it describes",
    "open-ended": "ask an open question that invites a short discussion",
}

LANGUAGES = {"urd": "Urdu", "eng": "English"}


@dataclass(frozen=True)
class Passage:
    """One training-split paragraph. ``id`` is the language and the line number in the file."""

    id: str
    lang: str
    text: str


@dataclass(frozen=True)
class Question:
    """A teacher-written question about one passage."""

    id: str
    passage_id: str
    lang: str
    format: str
    kind: str
    text: str


# passages ---------------------------------------------------------------------------------------


def sample_passages(
    lines: Sequence[str],
    n: int,
    guard_texts: Sequence[str],
    lang: str = "urd",
    min_words: int = 60,
    max_words: int = 250,
    seed: int = 0,
) -> tuple[list[Passage], dict]:
    """Pick ``n`` lines of ``min_words`` to ``max_words`` words that share no 8-word run with
    any evaluation text. Deterministic for a given file and seed.

    Returns the passages and counts: candidate lines in the length range, and how many of the
    lines looked at were skipped for overlapping the evaluation texts.
    """
    if lang not in LANGUAGES:
        raise ValueError(f"unknown language {lang!r}; known: {sorted(LANGUAGES)}")
    if n < 1:
        raise ValueError("n must be at least 1")
    if not 1 <= min_words <= max_words:
        raise ValueError("need 1 <= min_words <= max_words")
    shingles = flores_shingles(guard_texts)
    if not shingles:
        raise ValueError("no evaluation texts given; the contamination guard would do nothing")
    candidates = [i for i, x in enumerate(lines) if min_words <= len(split_words(x)) <= max_words]
    random.Random(seed).shuffle(candidates)
    chosen: list[Passage] = []
    skipped = 0
    for i in candidates:
        if overlaps_flores(lines[i], shingles):
            skipped += 1
            continue
        chosen.append(Passage(id=f"{lang}-{i}", lang=lang, text=lines[i]))
        if len(chosen) == n:
            break
    if len(chosen) < n:
        raise ValueError(
            f"only {len(chosen)} usable passages of {min_words} to {max_words} words; asked for {n}"
        )
    stats = {
        "lines": len(lines),
        "candidates": len(candidates),
        "skipped_overlap": skipped,
        "chosen": n,
        "min_words": min_words,
        "max_words": max_words,
        "seed": seed,
    }
    return chosen, stats


# questions --------------------------------------------------------------------------------------


def question_kinds(fmt: str, n: int) -> list[str]:
    """The kinds of the ``n`` questions asked for one passage in one format."""
    if fmt not in FORMATS:
        raise ValueError(f"unknown format {fmt!r}; known: {FORMATS}")
    kinds = KINDS[fmt]
    return [kinds[i % len(kinds)] for i in range(n)]


def question_messages(passage: Passage, fmt: str, n: int) -> list[dict]:
    """The request that makes the teacher write ``n`` questions of the given format."""
    language = LANGUAGES[passage.lang]
    kinds = question_kinds(fmt, n)
    wanted = "\n".join(f"{i + 1}. {k}: {KIND_HINTS[k]}" for i, k in enumerate(kinds))
    if fmt == "grounded":
        task = (
            f"Write {n} different requests that a user might send to an assistant together with "
            f"this passage. Each must be answerable from the passage alone and may refer to it as "
            f'"this text" (in {language}).'
        )
    else:
        task = (
            f"Write {n} different questions that a curious {language} speaker might ask an "
            "assistant without having seen this passage. Each must make sense on its own and "
            "must not mention a passage, text or article, and the passage must contain what is "
            "needed to answer it."
        )
    content = (
        f"Here is a passage from {language} Wikipedia.\n\n<passage>\n{passage.text}\n</passage>\n\n"
        f"{task} Write them in natural, fluent {language}, as a native speaker would.\n\n"
        f"Kinds, in this order:\n{wanted}\n\n"
        f"Answer with only a JSON array of {n} strings, in the order above, and nothing else."
    )
    return [{"role": "user", "content": content}]


_JSON_ARRAY = re.compile(r"\[.*\]", re.S)


def parse_questions(text: str, n: int) -> list[str]:
    """The ``n`` questions in the teacher's reply. Raises ValueError if they are not there."""
    answer = split_reply(text).answer
    found = _JSON_ARRAY.search(answer)
    if not found:
        raise ValueError("no JSON array in the reply")
    try:
        items = json.loads(found.group(0))
    except json.JSONDecodeError as e:
        raise ValueError(f"the JSON array does not parse: {e}") from e
    if not isinstance(items, list) or not all(isinstance(q, str) and q.strip() for q in items):
        raise ValueError("the array must hold only non-empty strings")
    if len(items) != n:
        raise ValueError(f"asked for {n} questions, got {len(items)}")
    return [q.strip() for q in items]


def request_seed(key: str) -> int:
    """A stable per-request seed, so a rerun samples the same way on the same server."""
    return zlib.crc32(key.encode("utf-8")) & 0x7FFFFFFF


def write_questions(
    client: ChatClient,
    passages: Sequence[Passage],
    per_format: int = 2,
    formats: Sequence[str] = FORMATS,
    sampling: Sampling = QWEN3_NO_THINKING,
    workers: int = 32,
    log: Callable[[str], None] = print,
) -> tuple[list[Question], dict]:
    """Ask the teacher (thinking off) for ``per_format`` questions per passage and format.

    A reply that does not parse is counted under ``failed`` and its passage and format get no
    questions; nothing is retried, so the counts are honest.
    """
    jobs = [(p, f) for p in passages for f in formats]

    def ask(job: tuple[Passage, str]) -> tuple[list[str] | None, str]:
        p, f = job
        try:
            reply = client.chat(
                question_messages(p, f, per_format),
                sampling,
                enable_thinking=False,
                seed=request_seed(f"q|{p.id}|{f}"),
            )
            return parse_questions(reply.text, per_format), ""
        except Exception as e:  # counted, never silently dropped
            return None, f"{type(e).__name__}: {e}"

    results = parallel_map(ask, jobs, workers)
    questions: list[Question] = []
    failures: list[dict] = []
    for (p, f), (texts, err) in zip(jobs, results, strict=True):
        if texts is None:
            failures.append({"passage_id": p.id, "format": f, "error": err[:300]})
            continue
        for i, (kind, text) in enumerate(zip(question_kinds(f, per_format), texts, strict=True)):
            questions.append(Question(f"{p.id}-{f[0]}{i}", p.id, p.lang, f, kind, text))
    log(f"questions: {len(questions)} from {len(jobs)} requests, {len(failures)} failed")
    return questions, {"requests": len(jobs), "failed": len(failures), "failures": failures}


# answers ----------------------------------------------------------------------------------------

BACKGROUND = (
    "Background that may help you answer (the user cannot see it):\n\n{passage}\n\n"
    "Answer the user in the language of their question. Use the background for facts when it "
    "is relevant, but answer as you would from your own knowledge: never mention the "
    "background, a passage, a text or an article."
)


def answer_messages(question: Question, passage: Passage) -> tuple[list[dict], list[dict]]:
    """The teacher's request and the training conversation's prompt for one question.

    They differ only for standalone questions, whose background system message is dropped
    from the training conversation.
    """
    if question.passage_id != passage.id:
        raise ValueError(f"question {question.id} is not about passage {passage.id}")
    if question.format == "grounded":
        user = [{"role": "user", "content": f"{passage.text}\n\n{question.text}"}]
        return user, user
    if question.format == "standalone":
        user = [{"role": "user", "content": question.text}]
        system = [{"role": "system", "content": BACKGROUND.format(passage=passage.text)}]
        return system + user, user
    raise ValueError(f"unknown format {question.format!r}")


def generate_answers(
    client: ChatClient,
    questions: Sequence[Question],
    passages: Sequence[Passage],
    teacher: str,
    thinking_modes: Sequence[bool] = (True, False),
    think_sampling: Sampling = QWEN3_THINKING,
    no_think_sampling: Sampling = QWEN3_NO_THINKING,
    workers: int = 64,
    log: Callable[[str], None] = print,
) -> list[dict]:
    """One record per question and thinking mode, in input order.

    ``messages`` is the training conversation: the prompt plus the teacher's raw reply as the
    assistant turn (with its ``<think>`` block when thinking was on). A failed request gives a
    record with ``error`` set and no assistant turn.
    """
    by_id = {p.id: p for p in passages}
    jobs = [(q, t) for q in questions for t in thinking_modes]

    def answer(job: tuple[Question, bool]) -> dict:
        q, thinking = job
        request, prompt = answer_messages(q, by_id[q.passage_id])
        rec = {
            "id": f"{q.id}-{'think' if thinking else 'nothink'}",
            "question_id": q.id,
            "passage_id": q.passage_id,
            "lang": q.lang,
            "format": q.format,
            "kind": q.kind,
            "thinking": thinking,
            "teacher": teacher,
            "messages": prompt,
            "finish_reason": None,
            "prompt_tokens": None,
            "completion_tokens": None,
            "error": None,
        }
        try:
            reply = client.chat(
                request,
                think_sampling if thinking else no_think_sampling,
                enable_thinking=thinking,
                seed=request_seed(f"a|{teacher}|{rec['id']}"),
            )
        except Exception as e:  # counted in the summary
            rec["error"] = f"{type(e).__name__}: {e}"[:300]
            return rec
        rec["messages"] = prompt + [{"role": "assistant", "content": reply.text}]
        rec.update(
            finish_reason=reply.finish_reason,
            prompt_tokens=reply.prompt_tokens,
            completion_tokens=reply.completion_tokens,
        )
        return rec

    records = parallel_map(answer, jobs, workers)
    errors = sum(r["error"] is not None for r in records)
    log(f"answers from {teacher}: {len(records)} records, {errors} failed requests")
    return records


# reading the results ----------------------------------------------------------------------------


def reply_flags(record: dict) -> dict:
    """The automatic checks on one record, as plain values (None when there is no reply)."""
    if record.get("error") is not None or record["messages"][-1]["role"] != "assistant":
        return {"replied": False}
    parts = split_reply(record["messages"][-1]["content"])
    share = urdu_letter_share(parts.answer) if parts.answer else None
    return {
        "replied": True,
        "has_think": parts.think is not None,
        "think_closed": parts.closed,
        "answered": bool(parts.answer),
        "hit_limit": record.get("finish_reason") == "length",
        "answer_urdu_share": share,
        "loop": has_loop(parts.answer) or (parts.think is not None and has_loop(parts.think)),
        "source_in_answer": mentions_source(parts.answer),
        "source_in_think": parts.think is not None and mentions_source(parts.think),
    }


def summarize_answers(records: Sequence[dict], urdu_share_min: float = 0.9) -> dict:
    """Counts per teacher, format and thinking mode, for comparing teachers.

    ``urdu_answers`` counts Urdu-question answers whose letters are at least ``urdu_share_min``
    Urdu script. ``source_*`` matter for standalone questions only (a grounded user turn shows
    the passage, so mentioning it is fine there).
    """
    groups: dict[str, list[dict]] = {}
    for r in records:
        key = f"{r['teacher']} | {r['format']} | {'think' if r['thinking'] else 'no think'}"
        groups.setdefault(key, []).append(r)
    out = {}
    for key, recs in sorted(groups.items()):
        flags = [reply_flags(r) for r in recs]
        ok = [f for f in flags if f["replied"]]
        urdu = [f for r, f in zip(recs, flags, strict=True) if f["replied"] and r["lang"] == "urd"]
        tokens = [r["completion_tokens"] for r in recs if r.get("completion_tokens")]
        out[key] = {
            "n": len(recs),
            "failed_requests": len(recs) - len(ok),
            "think_closed": sum(f["think_closed"] for f in ok if f["has_think"]),
            "with_think_block": sum(f["has_think"] for f in ok),
            "answered": sum(f["answered"] for f in ok),
            "hit_limit": sum(f["hit_limit"] for f in ok),
            "loops": sum(f["loop"] for f in ok),
            "urdu_questions": len(urdu),
            "urdu_answers": sum((f["answer_urdu_share"] or 0) >= urdu_share_min for f in urdu),
            "source_in_answer": sum(f["source_in_answer"] for f in ok),
            "source_in_think": sum(f["source_in_think"] for f in ok),
            "completion_tokens_median": statistics.median(tokens) if tokens else None,
            "completion_tokens_max": max(tokens) if tokens else None,
        }
    return out


def _quote(text: str) -> str:
    return "\n".join(f"> {line}" if line else ">" for line in text.strip().splitlines())


def reading_markdown(
    passages: Sequence[Passage],
    questions: Sequence[Question],
    records: Sequence[dict],
    max_passages: int | None = None,
) -> str:
    """A Markdown file for reading: each passage, its questions, and every teacher's answers
    with thinking off and on (reasoning folded in a ``<details>`` block)."""
    by_question: dict[str, list[dict]] = {}
    for r in records:
        by_question.setdefault(r["question_id"], []).append(r)
    out = ["# Teacher trial: passages, questions and answers", ""]
    for p in passages[:max_passages]:
        out += [f"## Passage {p.id}", "", _quote(p.text), ""]
        for q in (q for q in questions if q.passage_id == p.id):
            out += [f"### {q.id} ({q.format}, {q.kind})", "", f"**Question:** {q.text}", ""]
            recs = sorted(by_question.get(q.id, []), key=lambda r: (r["teacher"], r["thinking"]))
            for r in recs:
                mode = "thinking on" if r["thinking"] else "thinking off"
                head = f"**{r['teacher']}, {mode}**"
                if r["error"] is not None:
                    out += [f"{head}: request failed ({r['error']})", ""]
                    continue
                parts = split_reply(r["messages"][-1]["content"])
                flags = reply_flags(r)
                note = f"{r['completion_tokens']} tokens, finish: {r['finish_reason']}"
                if flags["loop"]:
                    note += ", LOOP"
                if q.format == "standalone" and (
                    flags["source_in_answer"] or flags["source_in_think"]
                ):
                    note += ", mentions the source"
                out += [f"{head} ({note})", ""]
                if parts.think is not None:
                    state = "" if parts.closed else " (never closed)"
                    out += [
                        f"<details><summary>reasoning{state}</summary>",
                        "",
                        parts.think,
                        "",
                        "</details>",
                        "",
                    ]
                out += [parts.answer or "*(no answer)*", ""]
        out.append("---\n")
    return "\n".join(out)


def to_jsonl(items: Sequence[object]) -> str:
    """One JSON object per line (dataclasses are converted); Urdu is written as is."""
    rows = [asdict(x) if hasattr(x, "__dataclass_fields__") else x for x in items]
    return "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)


# teacher server ---------------------------------------------------------------------------------


def teacher_serve_command(
    model: str,
    served_name: str,
    tensor_parallel: int = 2,
    max_model_len: int = 8192,
    gpu_memory_utilization: float = 0.9,
    port: int = 8000,
) -> list[str]:
    """``vllm serve`` for a teacher on T4s: float16 (no bfloat16 on a T4) and NO reasoning
    parser, so replies keep their raw ``<think>`` block for the training data."""
    if tensor_parallel < 1:
        raise ValueError("tensor_parallel must be at least 1")
    if max_model_len < 256:
        raise ValueError("max_model_len must be at least 256")
    if not 0.1 <= gpu_memory_utilization <= 0.98:
        raise ValueError("gpu_memory_utilization must be between 0.1 and 0.98")
    return [
        "vllm",
        "serve",
        model,
        "--served-model-name",
        served_name,
        "--dtype",
        "half",
        "--tensor-parallel-size",
        str(tensor_parallel),
        "--max-model-len",
        str(max_model_len),
        "--gpu-memory-utilization",
        str(gpu_memory_utilization),
        "--port",
        str(port),
    ]
