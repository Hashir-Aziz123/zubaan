"""Report section: stage 3, training the new tokens, and serving the adapted model.

Every measured number comes from the saved results: ``zubaan adapt eval``
(``adapt_quality.json``), ``zubaan export all`` and ``zubaan export check`` on the adapted model
(notebook 09) and the vLLM check (notebook 10). Sentences that state a direction (better, worse,
overlapping) are derived from the numbers, so the text cannot disagree with the tables. Only the
training recipe (fixed design settings from ``docs/decisions.md``) is written as prose.
"""

from __future__ import annotations

from statistics import median

from zubaan.adapt.evaluate import comparison_rows, success_checks

BASE, UNTRAINED, ADAPTED = "base", "extended-untrained", "adapted"


def _pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def _ci(b: dict) -> str:
    lo, hi = b["acc_char_ci"]
    return f"{_pct(b['acc_char'])} ({_pct(lo)} to {_pct(hi)})"


def _bpc(v: dict, s: str) -> float:
    return v["bpc"][s]["bits_per_character"]


def _overlap(a: list[float], b: list[float]) -> bool:
    return a[0] <= b[1] and b[0] <= a[1]


def check_adapt_results(adapt: dict) -> None:
    """Raise if the results lack the three models the section compares."""
    missing = [k for k in (BASE, UNTRAINED, ADAPTED) if k not in adapt.get("variants", {})]
    if missing:
        raise ValueError(f"adapt results have no {missing}; expected base, untrained, adapted")


def adaptation_summary(adapt: dict) -> str:
    """One summary bullet: cost and quality of the adapted model against the base."""
    v = adapt["variants"]
    b, a = v[BASE], v[ADAPTED]
    tb = b["bpc"]["urdu_flores"]["tokens_per_char"]
    ta = a["bpc"]["urdu_flores"]["tokens_per_char"]
    d = _bpc(a, "urdu_flores") - _bpc(b, "urdu_flores")
    word = "fewer" if d < 0 else "more"
    return (
        f"- After stage 3 training, the extended model needs **{tb / ta:.2f}x fewer tokens** for "
        f"the same FLORES Urdu text and scores **{abs(d):.2f} {word} bits per character** than "
        f"the base ({_bpc(b, 'urdu_flores'):.3f} to {_bpc(a, 'urdu_flores'):.3f}). Belebele Urdu "
        f"is {_pct(a['belebele']['urd_Arab']['acc_char'])} against the base's "
        f"{_pct(b['belebele']['urd_Arab']['acc_char'])}. Whether the gain comes from the new "
        "tokens or from the extra Urdu training is not yet known (see the stage 3 section)."
    )


def _table(adapt: dict) -> str:
    lines = [
        "| Model | Vocabulary | Urdu tokens/char (FLORES) | Urdu bits/char (held-out) "
        "| Urdu bits/char (FLORES) | English bits/char | Belebele Urdu | Belebele English |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    v = adapt["variants"]
    for r in comparison_rows(adapt):
        m = v[r["model"]]
        lines.append(
            f"| `{r['model']}` | {r['vocab_size']:,} | {r['tokens_per_char_urdu_flores']:.3f} "
            f"| {r['bpc_urdu_heldout']:.3f} | {r['bpc_urdu_flores']:.3f} "
            f"| {r['bpc_english_flores']:.3f} | {_ci(m['belebele']['urd_Arab'])} "
            f"| {_ci(m['belebele']['eng_Latn'])} |"
        )
    return "\n".join(lines)


def _checks(adapt: dict) -> str:
    rows = success_checks(adapt, BASE, UNTRAINED, ADAPTED)

    def label(c: dict) -> str:
        if not c["passed"]:
            return "FAIL"
        # A higher Belebele score inside overlapping intervals passes the check as written,
        # but it is not a measurable gain, so the label says so.
        return "PASS, within noise" if "intervals overlap" in c["detail"] else "PASS"

    return "\n".join(f"- **{label(c)}**: {c['name']}. {c['detail']}." for c in rows)


def _reading(adapt: dict) -> str:
    v = adapt["variants"]
    b, u, a = v[BASE], v[UNTRAINED], v[ADAPTED]
    out = [
        f"- **Training repaired the damage of the untrained rows and went past the base.** "
        f"FLORES Urdu went {_bpc(b, 'urdu_flores'):.3f} (base), {_bpc(u, 'urdu_flores'):.3f} "
        f"(untrained), {_bpc(a, 'urdu_flores'):.3f} (adapted)."
        if _bpc(a, "urdu_flores") < _bpc(b, "urdu_flores")
        else f"- FLORES Urdu went {_bpc(b, 'urdu_flores'):.3f} (base), "
        f"{_bpc(u, 'urdu_flores'):.3f} (untrained), {_bpc(a, 'urdu_flores'):.3f} (adapted).",
        "- **FLORES Urdu is the cleaner Urdu number.** The held-out lines come from the same "
        "Wikipedia dump as the training text, so they are in-domain and flatter the result "
        f"({_bpc(a, 'urdu_heldout'):.3f} held-out against {_bpc(a, 'urdu_flores'):.3f} FLORES).",
    ]
    d_en = _bpc(a, "english_flores") - _bpc(b, "english_flores")
    if d_en < 0:
        out.append(
            f"- **The lower English number ({_bpc(b, 'english_flores'):.3f} to "
            f"{_bpc(a, 'english_flores'):.3f}) is not evidence that English got better.** The "
            "base is a chat model, whose plain-text likelihood is weaker than a model tuned on "
            "plain text, and the English replay is Wikipedia, close to FLORES in style. Neither "
            "cause was tested. What it does show is that English was not damaged on this metric."
        )
    else:
        out.append(f"- English moved by {d_en:+.3f} bits per character.")
    bu, au = b["belebele"]["urd_Arab"], a["belebele"]["urd_Arab"]
    if _overlap(bu["acc_char_ci"], au["acc_char_ci"]):
        out.append(
            f"- **Understanding did not measurably change.** Belebele Urdu is {_ci(au)} against "
            f"{_ci(bu)} for the base; the intervals overlap. Better likelihood from a small "
            "continued-pretraining run did not turn into better multiple-choice answers."
        )
    else:
        out.append(f"- Belebele Urdu moved from {_ci(bu)} to {_ci(au)}; the intervals separate.")
    out.append(
        "- **The gain is not attributed to the tokenizer.** The adapted model trained on 16M "
        "tokens of Urdu that the base never saw. The compute-matched control (the base model "
        "with its own tokenizer, the same recipe and the same number of tokens, notebook 11) "
        "answers whether the new tokens help or hurt at equal cost. It has not been run yet."
    )
    return "\n".join(out)


def _serving(
    export: dict | None, check: dict | None, vllm: dict | None, ollama: dict | None = None
) -> str:
    parts = []
    gguf = (export or {}).get("gguf")
    if gguf:
        p = gguf["parity"]
        quants = ", ".join(f"`{f}`" for f in gguf["quantized"].values())
        how = (
            "mapped to the base type with a converter patch"
            if gguf.get("converter_patched")
            else "recognized without a converter patch"
        )
        parts.append(
            f"**GGUF.** llama.cpp `{gguf['llama_cpp_tag']}`, pre-tokenizer "
            f"`{gguf['pre_tokenizer']}` {how}. Tokenizer parity: "
            f"{p['n_texts'] - p['n_mismatch']:,} of {p['n_texts']:,} texts give identical ids, "
            f"the other {p.get('n_nfc_only', 0):,} differ only because the text is not NFC, and "
            f"{p.get('n_real_mismatch', 0)} differ for real. Training does not touch the "
            f"tokenizer, so this matches stage 2. Files: {quants}, with an importance matrix from "
            f"{gguf['imatrix_calibration']}."
        )
    if check and check.get("quant_cost"):
        rows = [
            "| Text | Quantization | Mean KL divergence | Same top token | Perplexity ratio |",
            "| --- | --- | ---: | ---: | ---: |",
        ]
        for lang, label in (("urdu", "FLORES Urdu"), ("english", "FLORES English")):
            qc = check["quant_cost"].get(lang)
            if not qc:
                continue
            for q, r in sorted(qc["quants"].items(), key=lambda kv: kv[0], reverse=True):
                rows.append(
                    f"| {label} | {q} | {r['mean_kld']['value']:.4f} "
                    f"| {r['same_top_p_percent']['value']:.1f}% | {r['ppl_ratio']['value']:.4f} |"
                )
        ks = check["kld_settings"]
        parts.append(
            "**Quantization cost against the f16 file** (llama.cpp KL-divergence mode, "
            f"{ks['chunks']} chunks of {ks['ctx']} tokens). All files share one tokenizer, so "
            "these per-token numbers compare the files with each other; they are not a language "
            "comparison.\n\n" + "\n".join(rows)
        )
    if check and check.get("samples"):
        speeds = {
            q: median(s["tokens_per_s"] for s in samples if s.get("tokens_per_s"))
            for q, samples in check["samples"].items()
        }
        listed = ", ".join(f"{q} {t:.1f}" for q, t in speeds.items())
        parts.append(
            f"**CPU generation speed** (llama.cpp, {check['threads']} threads, "
            f"{check['decoding']}, median over the sample prompts, tokens per second): "
            f"{listed}. This compares the "
            "files with each other on one Kaggle CPU session; it is not a benchmark."
        )
    if vllm:
        think = vllm.get("samples", {}).get("chat_thinking", {})
        ended_in_think = (
            "<think>" in think.get("text", "")
            and "</think>" not in think.get("text", "")
            and think.get("finish_reason") == "stop"
        )
        text = (
            f"**vLLM** {vllm['vllm']} on a {vllm['gpu']} (`dtype {vllm['dtype']}`): "
            f"{'passed' if vllm['passed'] else 'failed'}; tokenization identical to "
            f"`tokenizer.json` on {vllm['tokenization_identical']} test texts, and an Urdu chat "
            "question with thinking off gets an answer in Urdu script."
        )
        if ended_in_think:
            text += (
                " **With thinking allowed, the model wrote an English reasoning block and ended "
                "the turn inside it, with no closing tag and no answer** (one greedy sample). "
                "Training the chat model on plain text taught it to end documents; serve it with "
                "thinking off."
            )
        parts.append(text)
    if ollama:
        parts.append(_ollama_paragraph(ollama))
    if not parts:
        return ""
    parts.append(
        "**Generated text, read by a person** (greedy decoding, details in "
        "`docs/decisions.md`): fluent Urdu that answers simple questions, but with repetition "
        "loops on open-ended prompts and invented facts (for example Islamabad called the "
        "largest city, and inconsistent population figures). This is a 1.7B model after a small "
        "continued-pretraining run, not a source of facts."
    )
    return "### Serving the adapted model\n\n" + "\n\n".join(parts) + "\n"


def _ollama_paragraph(ollama: dict) -> str:
    """Answered prompts with thinking on and off, from the local Ollama check."""
    runs = ollama["runs"]

    def answered(mode: str) -> tuple[int, int]:
        rows = [r for r in runs if r["mode"] == mode]
        return sum(r["answered"] for r in rows), len(rows)

    on, n_on = answered("default_thinking_on")
    off, n_off = answered("think_false")
    speeds = [r["tokens_per_s"] for r in runs if r.get("tokens_per_s")]
    loops = sum(1 for r in runs if r["mode"] == "think_false" and r["done_reason"] == "length")
    return (
        f"**Ollama** {ollama['ollama']} on a laptop CPU (Q4_K_M, {ollama['decoding']}): with "
        f"thinking on (Ollama's default) the model answered **{on} of {n_on}** prompts; the "
        "others ended after the reasoning with no answer. With `think: false` it answered "
        f"{off} of {n_off}"
        + (f", and {loops} of them ran to the token limit" if loops else "")
        + f". {min(speeds):.0f} to {max(speeds):.0f} tokens per second. Ollama used the chat "
        "template embedded in the GGUF, not the Modelfile's, so thinking has to be turned off "
        "per request (`think: false`, or `ollama run --think=false`)."
    )


def _reproduced(adapt: dict, stage2_base: dict | None) -> str:
    """Whether the base model scored the same as in stage 2 (same texts, same protocol)."""
    if not stage2_base:
        return ""
    now = adapt["variants"][BASE]
    sets = [s for s in now["bpc"] if s in stage2_base.get("bpc", {})]
    if not sets:
        return ""
    worst = max(abs(_bpc(now, s) - _bpc(stage2_base, s)) for s in sets)
    if worst < 0.0005:
        return " The base model reproduced its stage 2 bits per character to three decimals."
    if worst < 0.005:
        return f" The base model reproduced its stage 2 bits per character (within {worst:.3f})."
    return f" Note: the base model differs from its stage 2 numbers by up to {worst:.3f}."


def adaptation_section(
    adapt: dict,
    export: dict | None = None,
    check: dict | None = None,
    vllm: dict | None = None,
    stage2_base: dict | None = None,
    ollama: dict | None = None,
) -> str:
    """The stage 3 section as Markdown (the chart is ``adaptation_quality.png``)."""
    check_adapt_results(adapt)
    s = adapt.get("meta", {}).get("settings", {})
    n = s.get("n_texts", {})
    return f"""
## Stage 3: training the new tokens

The extended model (10,000 new tokens, mean-of-pieces initialization) was trained by continued
pretraining on a Kaggle T4 in fp16: 100 steps that train only the new embedding rows, then 700
steps of LoRA (rank 64 on all attention and MLP projections) plus the new rows. Each step is 32
blocks of 1,024 tokens; the data is Urdu Wikipedia (16.05M tokens with the extended tokenizer)
with 10% English Wikipedia mixed in, after removing every line that overlaps FLORES or Belebele.
The old embedding rows were never changed. Design and reasons: `docs/decisions.md`, 2026-11-18.

All three models were then scored on the same texts as the table above ({
        n.get("urdu_heldout", "?")
    } held-out Urdu, {n.get("urdu_flores", "?")} FLORES Urdu and {n.get("english_flores", "?")}
FLORES English texts, all {s.get("belebele_questions") or 900} Belebele questions per language,
{s.get("dtype", "float32")}).{_reproduced(adapt, stage2_base)}

![Bits per character of the three models on three text sets](adaptation_quality.png)

{_table(adapt)}

Success checks set before training (`docs/decisions.md`):

{_checks(adapt)}

### How to read it

{_reading(adapt)}

{_serving(export, check, vllm, ollama)}"""
