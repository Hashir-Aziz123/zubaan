"""The Stage 4 success checks (docs/stage4-plan.md) as PASS or FAIL rows (chunk 7).

Inputs are results the other tools already write:

* ``chat``: label -> the ``summary`` of ``zubaan tune check`` (per mode, per language), for the
  base, Stage 3 and Stage 4 models, all served the same way (vLLM, greedy, same limits).
* ``quality``: the ``variants`` of ``zubaan adapt eval`` for the Stage 3 and Stage 4 models.
* ``base_quality``: the base model's numbers on the same protocol. They are not re-measured:
  notebook 08 measured them (2026-11-19), and re-scoring the Stage 3 model in the same session
  as Stage 4 shows the protocol still gives the same numbers.

Check 6 (a person reads replies next to the base model's) is not computed here.

``quantized_checks`` repeats checks 1 to 4 on the quantized GGUF (chunk 8), against the base
model's numbers from the chunk 7 run.
"""

from __future__ import annotations

from collections.abc import Mapping

# Notebook 08 (Kaggle T4, float32, commit 8d16b47): docs/decisions.md, 2026-11-19.
BASE_QUALITY = {
    "urdu_flores": 2.339,
    "english_flores": 1.208,
    "belebele_urd": 0.334,
    "belebele_urd_ci": (0.304, 0.366),
}
STAGE3_QUALITY = {"urdu_heldout": 1.486, "urdu_flores": 1.718, "english_flores": 1.066}
# Notebook 15 (2x T4, vLLM 0.34.0, float16, greedy, prompt set v1): docs/decisions.md, 2026-11-28.
BASE_CHAT = {"think_answer_rate": 58 / 60, "hit_limit": 44}


def _bpc(row: Mapping, name: str) -> float:
    return row["bpc"][name]["bits_per_character"]


def _rate(group: Mapping, key: str = "answer_rate") -> float:
    value = group[key]
    if value is None:
        raise ValueError("a chat-check group has no prompts")
    return value


def reproduces_stage3(stage3_row: Mapping, tolerance: float = 0.01) -> list[str]:
    """Differences between the Stage 3 model's numbers now and in notebook 08 (empty if none).

    If the protocol drifted (other texts, other code), the Stage 4 rows cannot be set next to the
    recorded base numbers.
    """
    return [
        f"{k}: now {_bpc(stage3_row, k):.3f}, notebook 08 {v:.3f}"
        for k, v in STAGE3_QUALITY.items()
        if abs(_bpc(stage3_row, k) - v) > tolerance
    ]


def _hit_limit(summary: Mapping) -> int:
    return sum(summary[m]["all"]["hit_limit"] for m in ("think", "no_think"))


def chat_checks(
    summary: Mapping,
    base_think_rate: float,
    base_hit_limit: int,
    stage3_think_rate: float | None = None,
) -> list[dict]:
    """Checks 1 to 4 of the plan for one chat-check ``summary`` (see ``stage4_checks``)."""
    checks = []
    on_t = _rate(summary["think"]["all"])
    checks.append(
        {
            "name": "1. Thinking on: answer rate within 10 points of the base",
            "passed": on_t >= base_think_rate - 0.10 - 1e-9,
            "detail": f"{on_t:.1%} vs base {base_think_rate:.1%}"
            + (f" (Stage 3 {stage3_think_rate:.1%})" if stage3_think_rate is not None else ""),
        }
    )
    off_t = _rate(summary["no_think"]["all"])
    checks.append(
        {
            "name": "2. Thinking off: at least 95% answered",
            "passed": off_t >= 0.95 - 1e-9,
            "detail": f"{off_t:.1%}",
        }
    )
    lang = {m: _rate(summary[m]["urd"], "right_language_rate") for m in ("think", "no_think")}
    checks.append(
        {
            "name": "3. Urdu prompts answered in Urdu script (at least 90% in each mode)",
            "passed": all(v >= 0.90 - 1e-9 for v in lang.values()),
            "detail": f"thinking on {lang['think']:.1%}, thinking off {lang['no_think']:.1%}",
        }
    )
    limit_t = _hit_limit(summary)
    checks.append(
        {
            "name": "4. Replies stopped at the token limit: no more than the base",
            "passed": limit_t <= base_hit_limit,
            "detail": f"{limit_t} vs base {base_hit_limit}",
        }
    )
    return checks


def quantized_checks(summary: Mapping, base_chat: Mapping = BASE_CHAT) -> list[dict]:
    """Checks 1 to 4 on a quantized GGUF served by llama-server, against the base model's numbers
    from notebook 15 (served by vLLM in float16). The runtimes differ, so a small gap from the
    float16 Stage 4 numbers is expected; the checks ask whether the file still behaves."""
    return chat_checks(summary, base_chat["think_answer_rate"], base_chat["hit_limit"])


def stage4_checks(
    chat: Mapping[str, Mapping],
    quality: Mapping[str, Mapping],
    base: str = "base",
    stage3: str = "stage3",
    stage4: str = "stage4",
    base_quality: Mapping = BASE_QUALITY,
) -> list[dict]:
    """Checks 1 to 5 of the plan as ``name, passed, detail`` rows.

    1. Thinking on: the share of prompts answered is at most 10 points below the base model's.
    2. Thinking off: at least 95% answered.
    3. Language: at least 90% of Urdu prompts answered in Urdu script, in each mode. An
       unanswered prompt counts as a miss.
    4. Loops: no more replies stopped at the token limit than the base model (both modes).
    5. Stage 3 gains kept: FLORES Urdu bits per character at most 0.05 above the Stage 3 model
       and below the base; FLORES English at most 0.02 above the base; Belebele Urdu
       ``acc_char`` not below the lower end of the base's 95% interval.
    """
    for label in (base, stage4):
        if label not in chat:
            raise ValueError(f"no chat-check results for {label!r}; have {sorted(chat)}")
    for label in (stage3, stage4):
        if label not in quality:
            raise ValueError(f"no quality results for {label!r}; have {sorted(quality)}")
    b, t4 = chat[base], chat[stage4]
    checks = chat_checks(
        t4,
        base_think_rate=_rate(b["think"]["all"]),
        base_hit_limit=_hit_limit(b),
        stage3_think_rate=_rate(chat[stage3]["think"]["all"]) if stage3 in chat else None,
    )
    q3, q4 = quality[stage3], quality[stage4]
    ur3, ur4 = _bpc(q3, "urdu_flores"), _bpc(q4, "urdu_flores")
    en4 = _bpc(q4, "english_flores")
    bel = q4["belebele"]["urd_Arab"]
    low = base_quality["belebele_urd_ci"][0]
    parts = {
        "FLORES Urdu within 0.05 of Stage 3": ur4 <= ur3 + 0.05 + 1e-9,
        "FLORES Urdu below the base": ur4 < base_quality["urdu_flores"],
        "FLORES English within 0.02 of the base": en4
        <= base_quality["english_flores"] + 0.02 + 1e-9,
        "Belebele Urdu not below the base's interval": bel["acc_char"] >= low,
    }
    checks.append(
        {
            "name": "5. Stage 3 gains kept",
            "passed": all(parts.values()),
            "detail": f"FLORES Urdu {ur4:.3f} (Stage 3 {ur3:.3f}, base "
            f"{base_quality['urdu_flores']:.3f}); FLORES English {en4:.3f} (base "
            f"{base_quality['english_flores']:.3f}); Belebele Urdu {bel['acc_char']:.3f} "
            f"[{bel['acc_char_ci'][0]:.3f}, {bel['acc_char_ci'][1]:.3f}] (base interval from "
            f"{low:.3f}). Failed: {[k for k, v in parts.items() if not v] or 'none'}",
        }
    )
    return checks


def checks_markdown(checks: list[dict], chat: Mapping[str, Mapping]) -> str:
    """The checks and a per-model chat table, as Markdown for the decisions log."""
    lines = ["| Check | Result | Detail |", "| --- | --- | --- |"]
    lines += [
        f"| {c['name']} | {'PASS' if c['passed'] else 'FAIL'} | {c['detail']} |" for c in checks
    ]
    lines += [
        "",
        "| Model | Mode | Answered | Urdu prompts in Urdu | Hit limit | Loops |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for label, summary in chat.items():
        for mode in ("think", "no_think"):
            if mode not in summary:
                continue
            a, u = summary[mode]["all"], summary[mode].get("urd")
            urd = f"{u['right_language']}/{u['language_due']}" if u else "n/a"
            lines.append(
                f"| {label} | {mode.replace('_', ' ')} | {a['answered']}/{a['n']} | {urd} | "
                f"{a['hit_limit']} | {a['loops']} |"
            )
    return "\n".join(lines) + "\n"
