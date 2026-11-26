"""Filter, balance and sample the teacher's conversations (Stage 4, chunk 4).

Input: the ``answers.jsonl`` files of the generation shards (notebook 13). Output: the training
conversations, a report that counts every dropped record under one reason, and a sample for a
person to read.

Each record is checked in a fixed order and dropped under the **first** reason that applies:

1. ``failed_request``: the server returned an error, so there is no reply.
2. ``hit_limit``: the reply stopped at the token limit; its ending is missing.
3. ``think_unclosed``: thinking on, but the ``<think>`` block was never closed.
4. ``think_mismatch``: thinking on without a ``<think>`` block, or thinking off with one.
5. ``no_answer``: nothing after the reasoning.
6. ``too_short``: an answer of fewer than ``min_answer_words`` words.
7. ``wrong_language``: an Urdu question answered with under 90% Urdu-script letters, or an
   English question with over 10%.
8. ``mixed_script``: Arabic-script and Latin letters inside one word ("سULTAN"), a decoding
   glitch, in the answer or the reasoning. An English term in brackets is fine.
9. ``loop``: an 8-word run repeated more than twice in the answer or the reasoning.
10. ``source_mention``: a standalone answer that refers to a passage the user never gave.
11. ``duplicate_question``: the same user turn (after normalization) as an earlier record.
12. ``near_chat_check``: the user turn is close to a chat-check prompt (word-set Jaccard at or
    above the threshold), so the evaluation would test the model on its own training data.

Then each slot of the mix (language, format, thinking) keeps at most its target count, chosen in
a fixed pseudo-random order, and the report states any slot that came up short.
"""

from __future__ import annotations

import csv
import io
import json
import zlib
from collections import Counter
from collections.abc import Iterable, Sequence
from pathlib import Path

from zubaan.export.check import urdu_letter_share
from zubaan.text.normalize import normalize_urdu, split_words
from zubaan.tune.filters import has_loop, mentions_source, mixed_script_word, split_reply
from zubaan.tune.generate import DEFAULT_MIX, MixSlot, check_mix

REASONS = (
    "failed_request",
    "hit_limit",
    "think_unclosed",
    "think_mismatch",
    "no_answer",
    "too_short",
    "wrong_language",
    "mixed_script",
    "loop",
    "source_mention",
    "duplicate_question",
    "near_chat_check",
)
KEEP_FIELDS = (
    "id",
    "question_id",
    "passage_id",
    "lang",
    "format",
    "kind",
    "thinking",
    "teacher",
    "messages",
)


def slot_of(record: dict) -> str:
    """Name of a record's mix slot, e.g. ``urd grounded think``."""
    mode = "think" if record["thinking"] else "nothink"
    return f"{record['lang']} {record['format']} {mode}"


def reply_problem(
    record: dict,
    min_answer_words: int = 3,
    urdu_min: float = 0.9,
    english_max: float = 0.1,
) -> str | None:
    """The first of reasons 1 to 10 that applies to one record, or None if it passes."""
    msgs = record.get("messages") or []
    if record.get("error") is not None or not msgs or msgs[-1].get("role") != "assistant":
        return "failed_request"
    if record.get("finish_reason") == "length":
        return "hit_limit"
    parts = split_reply(msgs[-1].get("content") or "")
    if record["thinking"]:
        if parts.think is None:
            return "think_mismatch"
        if not parts.closed:
            return "think_unclosed"
    elif parts.think is not None:
        return "think_mismatch"
    if not parts.answer:
        return "no_answer"
    if len(split_words(parts.answer)) < min_answer_words:
        return "too_short"
    share = urdu_letter_share(parts.answer)
    if record["lang"] == "urd" and (share is None or share < urdu_min):
        return "wrong_language"
    if record["lang"] == "eng" and share is not None and share > english_max:
        return "wrong_language"
    if mixed_script_word(parts.answer) or (parts.think and mixed_script_word(parts.think)):
        return "mixed_script"
    if has_loop(parts.answer) or (parts.think is not None and has_loop(parts.think)):
        return "loop"
    if record["format"] == "standalone" and mentions_source(parts.answer):
        return "source_mention"
    return None


def user_turn(record: dict) -> str:
    """The last user message of a record (the question, with the passage if grounded)."""
    return next(m["content"] for m in reversed(record["messages"]) if m["role"] == "user")


def word_set(text: str) -> frozenset[str]:
    return frozenset(w.lower() for w in split_words(normalize_urdu(text)))


def jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    """|a and b| / |a or b|; 0 for two empty sets.

    Example: {a, b, c} and {b, c, d} share 2 of 4 distinct words: 2 / 4 = 0.5.
    """
    union = len(a | b)
    return len(a & b) / union if union else 0.0


def nearest_prompt(
    text: str, prompts: Sequence[tuple[str, frozenset[str]]]
) -> tuple[str | None, float]:
    """The chat-check prompt with the highest word-set Jaccard to ``text``, and that score."""
    words = word_set(text)
    best, score = None, 0.0
    for pid, pw in prompts:
        s = jaccard(words, pw)
        if s > score:
            best, score = pid, s
    return best, score


def _order_key(record_id: str, seed: int) -> int:
    return zlib.crc32(f"{seed}:{record_id}".encode())


def curate(
    records: Iterable[dict],
    chat_prompts: Sequence[tuple[str, str]] = (),
    mix: Sequence[MixSlot] = DEFAULT_MIX,
    near_threshold: float = 0.6,
    min_answer_words: int = 3,
    seed: int = 0,
) -> tuple[list[dict], dict]:
    """Filter and balance records. Returns the kept conversations and a report.

    ``chat_prompts`` are ``(id, text)`` pairs, one per user turn of the chat-check prompts.
    Records are processed in id order, so which of two duplicates is kept does not depend on
    the order of the input files.
    """
    check_mix(mix)
    prompt_sets = [(pid, word_set(text)) for pid, text in chat_prompts]
    records = sorted(records, key=lambda r: r["id"])
    ids = [r["id"] for r in records]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate record ids in the input (a shard given twice?)")
    dropped: dict[str, Counter] = {}
    near_examples: list[dict] = []
    seen: set[str] = set()
    passing: dict[str, list[dict]] = {}
    for rec in records:
        slot = slot_of(rec)
        reason = reply_problem(rec, min_answer_words)
        if reason is None:
            turn = user_turn(rec)
            key = " ".join(split_words(normalize_urdu(turn).lower()))
            if key in seen:
                reason = "duplicate_question"
            else:
                seen.add(key)
                pid, score = nearest_prompt(turn, prompt_sets) if prompt_sets else (None, 0.0)
                if pid is not None and score >= near_threshold:
                    reason = "near_chat_check"
                    near_examples.append(
                        {"id": rec["id"], "prompt": pid, "jaccard": round(score, 3)}
                    )
        if reason is None:
            passing.setdefault(slot, []).append(rec)
        else:
            dropped.setdefault(slot, Counter())[reason] += 1

    targets = {f"{s.lang} {s.format} {'think' if s.thinking else 'nothink'}": s.count for s in mix}
    kept: list[dict] = []
    slots = {}
    all_slots = sorted(set(targets) | set(passing) | set(dropped))
    for slot in all_slots:
        ok = sorted(passing.get(slot, []), key=lambda r: _order_key(r["id"], seed))
        target = targets.get(slot, 0)
        chosen = ok[:target]
        kept += [{k: r[k] for k in KEEP_FIELDS if k in r} for r in chosen]
        n_drop = sum(dropped.get(slot, Counter()).values())
        slots[slot] = {
            "generated": len(ok) + n_drop,
            "passed": len(ok),
            "target": target,
            "kept": len(chosen),
            "short_by": max(0, target - len(ok)),
            "dropped": {
                r: dropped.get(slot, Counter())[r]
                for r in REASONS
                if dropped.get(slot, Counter())[r]
            },
        }
    kept.sort(key=lambda r: r["id"])
    totals = Counter()
    for c in dropped.values():
        totals.update(c)
    report = {
        "generated": len(records),
        "passed": sum(len(v) for v in passing.values()),
        "kept": len(kept),
        "dropped": {r: totals[r] for r in REASONS if totals[r]},
        "slots": slots,
        "near_chat_check": near_examples,
        "settings": {
            "near_threshold": near_threshold,
            "min_answer_words": min_answer_words,
            "seed": seed,
            "urdu_min_share": 0.9,
            "english_max_share": 0.1,
        },
    }
    return kept, report


def review_sample(kept: Sequence[dict], n: int, seed: int = 0) -> list[dict]:
    """``n`` conversations for a person to read, spread over the slots in proportion to their
    size (at least one per slot), in a fixed pseudo-random order."""
    if n < 1:
        raise ValueError("n must be at least 1")
    by_slot: dict[str, list[dict]] = {}
    for r in kept:
        by_slot.setdefault(slot_of(r), []).append(r)
    total = len(kept)
    sample = []
    for _, recs in sorted(by_slot.items()):
        k = max(1, round(n * len(recs) / total))
        recs = sorted(recs, key=lambda r: _order_key(r["id"], seed + 1))
        sample += recs[:k]
    return sorted(sample, key=lambda r: (slot_of(r), _order_key(r["id"], seed + 1)))


def _quote(text: str) -> str:
    return "\n".join("> " + line if line.strip() else ">" for line in text.strip().splitlines())


def review_markdown(sample: Sequence[dict], think_chars: int = 600) -> str:
    """The sample as Markdown: question, the start of the reasoning, the full answer."""
    out = [
        "# Stage 4 review sample",
        "",
        f"{len(sample)} conversations. For each, mark `pass` or `fail` in `review.csv` (column "
        "`verdict`) and add a short note for a fail: wrong fact, unnatural Urdu, does not answer "
        "the question, wrong honorific, and so on. Judge the answer; the reasoning is shown only "
        f"in part (first {think_chars} characters).",
        "",
    ]
    for i, r in enumerate(sample, 1):
        parts = split_reply(r["messages"][-1]["content"])
        out += [f"## {i}. {r['id']} ({slot_of(r)}, {r.get('kind', '')})", "", "**User**", ""]
        out += [_quote(user_turn(r)), ""]
        if parts.think:
            think = (
                parts.think
                if len(parts.think) <= think_chars
                else parts.think[:think_chars] + " ..."
            )
            out += ["**Reasoning (start)**", "", _quote(think), ""]
        out += ["**Answer**", "", _quote(parts.answer), ""]
    return "\n".join(out) + "\n"


def review_csv(sample: Sequence[dict]) -> str:
    """A sheet to fill in: number, id, slot, verdict (pass or fail), note."""
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(["n", "id", "slot", "verdict", "note"])
    for i, r in enumerate(sample, 1):
        w.writerow([i, r["id"], slot_of(r), "", ""])
    return buf.getvalue()


def read_answers(paths: Sequence[Path]) -> list[dict]:
    """Records from the shards' answers.jsonl files. A cut-off last line (a session that
    stopped mid-write) is skipped and counted, not fatal."""
    records = []
    for path in paths:
        with Path(path).open(encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    records.append({"id": f"{path}:unreadable", "unreadable": True})
    return records


def chat_check_turns(path: Path | None = None) -> list[tuple[str, str]]:
    """Every user turn of the chat-check prompt set, as ``(prompt id, text)``."""
    from zubaan.tune.chat_check import load_prompts

    prompts, _ = load_prompts(path)
    return [(p.id, t) for p in prompts for i, t in enumerate(p.turns) if i % 2 == 0]


def run_curate(
    answers: Sequence[Path],
    out_dir: Path,
    review_n: int = 150,
    near_threshold: float = 0.6,
    min_answer_words: int = 3,
    prompts: Path | None = None,
    seed: int = 0,
) -> dict:
    """Read the shards, curate, and write conversations.jsonl, curate_report.json, review.md
    and review.csv into ``out_dir``. Returns the report."""
    raw = read_answers(answers)
    unreadable = sum(1 for r in raw if r.get("unreadable"))
    records = [r for r in raw if not r.get("unreadable")]
    kept, report = curate(
        records, chat_check_turns(prompts), DEFAULT_MIX, near_threshold, min_answer_words, seed
    )
    report["unreadable_lines"] = unreadable
    report["inputs"] = [str(p) for p in answers]
    sample = review_sample(kept, review_n, seed) if kept else []
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "conversations.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in kept),
        encoding="utf-8",
        newline="\n",
    )
    (out_dir / "curate_report.json").write_text(
        json.dumps(report, indent=1, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n"
    )
    (out_dir / "review.md").write_text(review_markdown(sample), encoding="utf-8", newline="\n")
    (out_dir / "review.csv").write_text(review_csv(sample), encoding="utf-8", newline="\n")
    return report
