"""Report section: stage 4, restoring chat behavior, and serving the chat-tuned model.

Every measured number comes from the saved results: notebook 15's ``stage4_checks.json`` (the
success checks and the chat-check summaries of the base, Stage 3 and Stage 4 models through vLLM)
and ``tune_quality.json`` (``zubaan adapt eval`` on Stage 3 and Stage 4), notebook 16's export
summary, export check and chat check on the Q4_K_M file, and the local Ollama check. Direction
words are derived from the numbers. The training recipe and the reading of the replies (check 6,
done by Claude, see ``docs/decisions.md``, 2026-11-28) are written as prose.
"""

from __future__ import annotations

from zubaan.report.adaptation import gguf_parts
from zubaan.tune.evaluate import BASE_QUALITY, quantized_checks

MODELS = (("base", "base"), ("stage3", "Stage 3"), ("stage4", "Stage 4"))
MODES = (("think", "on"), ("no_think", "off"))


def _bpc(v: dict, s: str) -> float:
    return v["bpc"][s]["bits_per_character"]


def check_tune_results(checks: dict, quality: dict) -> None:
    """Raise if the results lack what the section compares."""
    missing = [k for k, _ in MODELS if k not in checks.get("chat", {})]
    if missing or not checks.get("checks"):
        raise ValueError(f"stage 4 checks have no checks or no chat results for {missing}")
    missing_q = [k for k in ("stage3", "stage4") if k not in quality.get("variants", {})]
    if missing_q:
        raise ValueError(f"stage 4 quality results have no {missing_q}")


def _answered(summary: dict, mode: str) -> tuple[int, int]:
    g = summary[mode]["all"]
    return g["answered"], g["n"]


def tuning_summary(checks: dict, quality: dict) -> str:
    """One summary bullet: thinking restored, Stage 3 gains kept, quality of the answers."""
    chat = checks["chat"]
    on = {k: _answered(chat[k], "think") for k, _ in MODELS}
    passed = sum(c["passed"] for c in checks["checks"])
    q4 = _bpc(quality["variants"]["stage4"], "urdu_flores")
    q3 = _bpc(quality["variants"]["stage3"], "urdu_flores")
    return (
        f"- Stage 4 (a short chat fine-tune) **restored thinking mode**: with thinking on it "
        f"answered {on['stage4'][0]} of {on['stage4'][1]} prompts, against {on['stage3'][0]} "
        f"for Stage 3 and {on['base'][0]} for the base. {passed} of {len(checks['checks'])} "
        f"success checks pass, and FLORES Urdu stays at {q4:.3f} bits per character (Stage 3 "
        f"{q3:.3f}). Its Urdu answers are better than the base's but not reliable (read by "
        "Claude, not a native speaker)."
    )


def _chat_table(chat: dict, q4: dict | None, gpu: str, vllm: str) -> str:
    lines = [
        "| Model | Served by | Thinking | Answered | Urdu prompts in Urdu | Ran to the limit "
        "| Loops |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: |",
    ]
    rows = [(label, chat[k], f"vLLM {vllm}, float16, {gpu}") for k, label in MODELS]
    if q4:
        rows.append(("Stage 4 Q4_K_M", q4["summary"], "llama-server, CPU"))
    for label, summary, runtime in rows:
        for mode, word in MODES:
            a, u = summary[mode]["all"], summary[mode]["urd"]
            lines.append(
                f"| {label} | {runtime} | {word} | {a['answered']}/{a['n']} "
                f"| {u['right_language']}/{u['language_due']} | {a['hit_limit']} | {a['loops']} |"
            )
    return "\n".join(lines)


def _quality_rows(quality: dict, base: dict | None) -> list[tuple[str, dict]]:
    """Base (from the Stage 3 results if given, else the recorded numbers), Stage 3, Stage 4."""
    if base is None:
        base = {
            "bpc": {
                "urdu_flores": {"bits_per_character": BASE_QUALITY["urdu_flores"]},
                "english_flores": {"bits_per_character": BASE_QUALITY["english_flores"]},
            },
            "belebele": {
                "urd_Arab": {
                    "acc_char": BASE_QUALITY["belebele_urd"],
                    "acc_char_ci": list(BASE_QUALITY["belebele_urd_ci"]),
                }
            },
        }
    v = quality["variants"]
    return [("base", base), ("Stage 3", v["stage3"]), ("Stage 4", v["stage4"])]


def _quality_table(rows: list[tuple[str, dict]]) -> str:
    def cell(m: dict, s: str) -> str:
        return f"{_bpc(m, s):.3f}" if s in m["bpc"] else "n/a"

    def bel(m: dict, lang: str) -> str:
        b = m["belebele"].get(lang)
        if not b:
            return "n/a"
        lo, hi = b["acc_char_ci"]
        return f"{100 * b['acc_char']:.1f}% ({100 * lo:.1f} to {100 * hi:.1f})"

    lines = [
        "| Model | Urdu bits/char (FLORES) | Urdu bits/char (held-out) | English bits/char "
        "| Belebele Urdu | Belebele English |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for label, m in rows:
        lines.append(
            f"| {label} | {cell(m, 'urdu_flores')} | {cell(m, 'urdu_heldout')} "
            f"| {cell(m, 'english_flores')} | {bel(m, 'urd_Arab')} | {bel(m, 'eng_Latn')} |"
        )
    return "\n".join(lines)


def _check_lines(rows: list[dict]) -> str:
    return "\n".join(
        f"- **{'PASS' if c['passed'] else 'FAIL'}**: {c['name']}. {c['detail']}." for c in rows
    )


def _reading(quality: dict) -> str:
    v = quality["variants"]
    s3, s4 = v["stage3"], v["stage4"]
    sets = ("urdu_flores", "urdu_heldout", "english_flores")
    worst = max(_bpc(s4, s) - _bpc(s3, s) for s in sets)
    moved = (
        f"FLORES Urdu went from {_bpc(s3, 'urdu_flores'):.3f} (Stage 3) to "
        f"{_bpc(s4, 'urdu_flores'):.3f} bits per character, held-out Urdu from "
        f"{_bpc(s3, 'urdu_heldout'):.3f} to {_bpc(s4, 'urdu_heldout'):.3f}, English from "
        f"{_bpc(s3, 'english_flores'):.3f} to {_bpc(s4, 'english_flores'):.3f}."
    )
    if worst <= 0.05:
        head = "- **Plain-text likelihood barely moved.** " + moved
        if worst > 0:
            head += (
                " Chat tuning traded a little likelihood of Wikipedia-style text for answering "
                "questions."
            )
    else:
        head = f"- **Plain-text likelihood got worse by up to {worst:.3f}.** " + moved
    out = [
        head,
        "- **The base is a weak bar for Urdu here.** Under greedy decoding the base Qwen3-1.7B "
        "loops on most open-ended Urdu prompts, so the token-limit check is easy to pass. The "
        "thinking-on answer rate and the reading below say more.",
        "- **Answered is not right.** A fluent wrong answer counts as answered.",
        "- **What a reader sees** (all 240 base and Stage 4 replies, read side by side by "
        "Claude, not a native speaker): Stage 4 writes readable, on-topic Urdu where the base "
        "mostly loops, and short facts and sums are right. Long answers still slide into "
        "repetition and facts are often wrong (the capital of Punjab given as Islamabad, Iqbal "
        "born in Multan). The reasoning is in English, as the teacher's was. With thinking off, "
        "English reasoning prompts got worse: 1 of 4 right, against 4 of 4 for the base, "
        "because the training answers without thinking were short and direct.",
    ]
    return "\n".join(out)


def _threads(q4: dict) -> str:
    cmd = q4.get("server", {}).get("command") or []
    return cmd[cmd.index("-t") + 1] if "-t" in cmd[:-1] else "?"


def _ollama_paragraph(ollama: dict) -> str:
    """Answered prompts and reasoning per mode, from the local Ollama check."""
    runs = ollama["runs"]
    parts = []
    for mode, label in (("default_thinking_on", "thinking on (Ollama's default)"),
                        ("think_false", "`think: false`")):  # fmt: skip
        rows = [r for r in runs if r["mode"] == mode]
        if not rows:
            continue
        ans = sum(r["answered"] for r in rows)
        thought = sum(r["thought"] for r in rows)
        limit = sum(r["done_reason"] == "length" for r in rows)
        text = f"with {label} it answered {ans} of {len(rows)} prompts"
        if mode == "default_thinking_on":
            text += f" (reasoning shown in {thought})"
        if limit:
            text += f", {limit} ran to the token limit"
        parts.append(text)
    speeds = [r["tokens_per_s"] for r in runs if r.get("tokens_per_s")]
    return (
        f"**Ollama** {ollama['ollama']} on a laptop CPU (Q4_K_M, {ollama['decoding']}): "
        + "; ".join(parts)
        + f". {min(speeds):.0f} to {max(speeds):.0f} tokens per second. Ollama uses the chat "
        "template embedded in the GGUF, so thinking is on by default and is turned off per "
        "request (`think: false`, or `ollama run --think=false`)."
    )


def _serving(
    checks: dict,
    export: dict | None,
    export_check: dict | None,
    q4: dict | None,
    ollama: dict | None,
) -> str:
    parts = gguf_parts(export, export_check)
    parts.append(
        f"**vLLM** {checks.get('vllm', '?')} served the merged model in float16 on a "
        f"{checks.get('gpu', '?')} for the chat check above, with the chat template from the "
        "model folder; thinking is switched per request with "
        '`"chat_template_kwargs": {"enable_thinking": false}`.'
    )
    if q4:
        rows = quantized_checks(q4["summary"])
        verdict = "all pass" if all(c["passed"] for c in rows) else "not all pass"
        parts.append(
            "**The Q4_K_M file through llama-server** (the same 60 prompts and limits, greedy, "
            f"{q4['seconds'] / 60:.0f} minutes on {_threads(q4)} CPU threads): "
            f"checks 1 to 4 against the base model's vLLM numbers, {verdict}. The runtimes "
            "differ, so small differences from the float16 rows are expected.\n\n"
            + _check_lines(rows)
        )
    if ollama:
        parts.append(_ollama_paragraph(ollama))
    return "### Serving the Stage 4 model\n\n" + "\n\n".join(parts) + "\n"


def tuning_section(
    checks: dict,
    quality: dict,
    export: dict | None = None,
    export_check: dict | None = None,
    q4: dict | None = None,
    ollama: dict | None = None,
    base: dict | None = None,
) -> str:
    """The stage 4 section as Markdown. ``base`` is the base model's quality row (Stage 3)."""
    check_tune_results(checks, quality)
    return f"""
## Stage 4: restoring chat behavior

Stage 3 trained a chat model on plain text, and with thinking on (Qwen3's default) it then
usually ended its turn after the reasoning, with no answer. Stage 4 fine-tuned the Stage 3 model
on 9,854 conversations (80% Urdu, 20% English, about half with thinking) written by Qwen3-14B-AWQ
and grounded in Urdu Wikipedia passages of the Stage 3 training split, with the loss on the reply
only: LoRA rank 64 on all attention and MLP projections, the new embedding rows frozen, 2 epochs
(140 steps of 16 rows of 4,096 packed tokens) on one T4 in fp16. A sample of 150 conversations was
read before training: 71% passed (Urdu 66%, English 90%), so the training data has wrong words and
facts in about one Urdu answer in three. Design and reasons: `docs/stage4-plan.md` and
`docs/decisions.md`.

**Chat behavior** on 60 hand-written prompts (40 Urdu, 20 English, never shown to the teacher),
greedy decoding, at most 2,048 new tokens with thinking and 768 without. "Answered" means any
reasoning block was closed and an answer followed. A loop is an 8-word run repeated more than
twice.

{_chat_table(checks["chat"], q4, checks.get("gpu", "?"), checks.get("vllm", "?"))}

**Quality** with the Stage 3 protocol (same texts, float32). The base row is from Stage 3.

{_quality_table(_quality_rows(quality, base))}

Success checks set before training (`docs/stage4-plan.md`):

{_check_lines(checks["checks"])}

### How to read it

{_reading(quality)}

{_serving(checks, export, export_check, q4, ollama)}"""
