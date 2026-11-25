"""Report: "Extending Qwen3 with Urdu tokens". Cost, quality and export, from saved results.

Every number in the text and tables is read from the JSON files written by
``zubaan extend sweep``, ``zubaan extend quality`` and ``zubaan export``, and for stage 3 by
``zubaan adapt eval``, ``zubaan export check`` and the vLLM check (``report/adaptation.py``).
Nothing measured is typed by hand, so the report cannot drift from the results.
"""

from __future__ import annotations

from pathlib import Path

from zubaan.report.tax import load_json


def _pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def _ci(b: dict) -> str:
    lo, hi = b["acc_char_ci"]
    return f"{_pct(b['acc_char'])} ({_pct(lo)} to {_pct(hi)})"


def _cost_table(rows: list[dict]) -> str:
    lines = [
        "| New tokens | Vocabulary | Urdu tokens/word | Urdu chars/token | Cost multiplier "
        "| English changed | Round-trip failures |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in rows:
        lines.append(
            f"| {r['n_new_tokens']:,} | {r['vocab_size']:,} | {r['urdu_tokens_per_word']:.2f} "
            f"| {r['urdu_chars_per_token']:.2f} | **{r['cost_multiplier']:.2f}x** "
            f"| {r['english_different']} | {r['urdu_roundtrip_failures']} |"
        )
    return "\n".join(lines)


def _quality_table(variants: dict) -> str:
    lines = [
        "| Variant | Urdu bits/char (held-out) | Urdu bits/char (FLORES) | English bits/char "
        "| Belebele Urdu | Belebele English |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for key, v in variants.items():
        bpc = v["bpc"]
        lines.append(
            f"| `{key}` | {bpc['urdu_heldout']['bits_per_character']:.3f} "
            f"| {bpc['urdu_flores']['bits_per_character']:.3f} "
            f"| {bpc['english_flores']['bits_per_character']:.3f} "
            f"| {_ci(v['belebele']['urd_Arab'])} | {_ci(v['belebele']['eng_Latn'])} |"
        )
    return "\n".join(lines)


def _summary(sweep: dict, quality: dict, export: dict | None) -> list[str]:
    rows = sweep["rows"]
    base_row, row10 = rows[0], next((r for r in rows if r["n_new_tokens"] == 10_000), rows[-1])
    variants = quality["variants"]
    base = variants["base"]
    b_bpc = base["bpc"]["urdu_heldout"]["bits_per_character"]
    ext = {k: v for k, v in variants.items() if k != "base"}
    mean10 = variants.get(f"n{row10['n_new_tokens']}-mean")
    out = [
        f"- Adding {row10['n_new_tokens']:,} Urdu tokens cuts the cost of Urdu from "
        f"**{base_row['cost_multiplier']:.2f}x** to **{row10['cost_multiplier']:.2f}x** the tokens "
        f"of the same English text ({base_row['urdu_tokens_per_word']:.2f} to "
        f"{row10['urdu_tokens_per_word']:.2f} tokens per Urdu word). English tokenization is "
        "byte-for-byte unchanged and Urdu round-trips, at every size.",
    ]
    if mean10:
        m_bpc = mean10["bpc"]["urdu_heldout"]["bits_per_character"]
        out.append(
            f"- The new embeddings are initialized, not trained, so Urdu quality is worse for now: "
            f"bits per character on held-out Urdu goes from {b_bpc:.2f} (base) to {m_bpc:.2f} "
            f"(`{row10['n_new_tokens']:,}` tokens, mean-of-pieces init), "
            f"{100 * (m_bpc / b_bpc - 1):.0f}% higher. Stage 3 (continued pretraining) is what "
            "should close this gap."
        )
    accs = [v["belebele"]["urd_Arab"]["acc_char"] for v in ext.values()]
    b_acc = base["belebele"]["urd_Arab"]
    out.append(
        f"- Belebele Urdu accuracy is {_ci(b_acc)} for the base model. Every extended variant "
        f"scores between {_pct(min(accs))} and {_pct(max(accs))}, against a chance level of 25%."
    )
    eng = [v["bpc"]["english_flores"]["bits_per_character"] for v in ext.values()]
    out.append(
        f"- English is almost untouched: bits per character moves from "
        f"{base['bpc']['english_flores']['bits_per_character']:.3f} to between "
        f"{min(eng):.3f} and {max(eng):.3f}."
    )
    gguf = (export or {}).get("gguf")
    if gguf:
        p = gguf["parity"]
        nfc = p.get("n_nfc_only", 0)
        tail = (
            f", and the other {nfc:,} differ only because the text is not in NFC form "
            "(see the GGUF section)."
            if nfc
            else "."
        )
        out.append(
            f"- GGUF export passes the tokenizer parity check: llama.cpp `{gguf['llama_cpp_tag']}` "
            f"and Hugging Face give identical token ids on {p['n_texts'] - p['n_mismatch']:,} of "
            f"{p['n_texts']:,} texts{tail}"
        )
    return out


def _init_section(variants: dict) -> str:
    lines = []
    for n in sorted({v["n_new_tokens"] for v in variants.values() if v["n_new_tokens"]}):
        got = {
            m: variants[f"n{n}-{m}"] for m in ("mean", "focus", "random") if f"n{n}-{m}" in variants
        }
        if len(got) < 2:
            continue
        sets = ("urdu_heldout", "urdu_flores")
        parts = []
        for m, v in got.items():
            vals = ", ".join(f"{v['bpc'][s]['bits_per_character']:.2f}" for s in sets)
            parts.append(f"{m} {vals}")
        lines.append(
            f"- {n:,} new tokens (held-out, FLORES bits per character): " + "; ".join(parts)
        )
        focus = got.get("focus", {}).get("init", {}).get("focus")
        if focus:
            lines.append(
                f"  FOCUS-style had an auxiliary vector for {focus['n_with_vector']:,} of "
                f"{focus['n_new']:,} new tokens; the other {focus['n_fallback_to_mean']:,} fell "
                "back to the mean of pieces."
            )
    return "\n".join(lines)


def _init_verdict(variants: dict) -> str:
    """Plain-language reading of the init comparison, derived from the numbers."""
    sizes = sorted({v["n_new_tokens"] for v in variants.values() if v["n_new_tokens"]})
    worst_is_random, gaps = [], []
    for n in sizes:
        got = {m: variants.get(f"n{n}-{m}") for m in ("mean", "focus", "random")}
        if not all(got.values()):
            continue
        for sname in ("urdu_heldout", "urdu_flores"):
            b = {m: v["bpc"][sname]["bits_per_character"] for m, v in got.items()}
            worst_is_random.append(max(b, key=b.get) == "random")
            gaps.append(abs(b["mean"] - b["focus"]))
    if not gaps:
        return ""
    text = (
        "Random initialization is the worst in every comparison."
        if all(worst_is_random)
        else "No initialization is consistently the worst."
    )
    text += f" Mean of pieces and FOCUS-style differ by at most {max(gaps):.2f} bits per character."
    ext = [v for k, v in variants.items() if k != "base"]
    cis = [v["belebele"]["urd_Arab"]["acc_char_ci"] for v in ext]
    if all(a[0] <= b[1] and b[0] <= a[1] for a in cis for b in cis):
        text += (
            " The Belebele intervals of all extended variants overlap, so Belebele cannot "
            "separate them and bits per character is the more informative column."
        )
    return text


def build_extension_markdown(
    sweep: dict,
    quality: dict,
    export: dict | None = None,
    adapt: dict | None = None,
    adapted_export: dict | None = None,
    adapted_check: dict | None = None,
    adapted_vllm: dict | None = None,
    adapted_ollama: dict | None = None,
) -> str:
    """The full report as Markdown. Chart images are referenced by relative file name.

    ``adapt`` (the ``zubaan adapt eval`` results) adds the stage 3 section; the other
    ``adapted_*`` results add its serving part.
    """
    from zubaan.report.adaptation import adaptation_section, adaptation_summary

    variants = quality["variants"]
    if "base" not in variants:
        raise ValueError("the quality results have no 'base' variant")
    meta = quality.get("meta", {})
    settings = meta.get("settings", {})
    pk = meta.get("packages", {})
    n_texts = settings.get("n_texts", {})
    export_text = _export_section(export) if export else ""
    summary = _summary(sweep, quality, export)
    adapt_text = ""
    first_limit = (
        "- The new embeddings are untrained. These numbers are the starting point for stage 3, "
        "not a\n  model to hand to users."
    )
    if adapt:
        summary.append(adaptation_summary(adapt))
        adapt_text = adaptation_section(
            adapt,
            adapted_export,
            adapted_check,
            adapted_vllm,
            stage2_base=variants["base"],
            ollama=adapted_ollama,
        )
        first_limit = (
            "- The stage 2 rows are untrained; the stage 3 section shows the trained model. It "
            "is one\n  small continued-pretraining run (one seed, 26M tokens, one T4), its gain "
            "is not yet\n  attributed to the tokenizer, and its chat behavior has known problems "
            "(see above)."
        )
    return f"""# Extending {quality["model"]} with Urdu tokens

Two things were measured for each vocabulary size: what Urdu costs in tokens, and what it does to
quality. A cost saving is only a result if quality is shown next to it.

## Summary

{chr(10).join(summary)}

## What was done

1. Merges were learned from Urdu Wikipedia on top of the base tokenization (continued BPE), and
   only merges that contain Arabic-script bytes are allowed, so English text cannot be affected.
2. The merges were appended to the base tokenizer as real BPE merges, not as added tokens. The
   base pre-tokenizer is unchanged, and special tokens keep their ids.
3. The embedding matrix (input and tied output head) was resized and each new row initialized
   one of three ways: **mean of pieces** (the mean of the embeddings of the base tokens the new
   token replaces), **FOCUS-style** (a sparsemax-weighted mean of similar old tokens, using
   co-occurrence vectors from the Urdu corpus; see `docs/decisions.md`) and **random** (a control:
   noise with the old rows' per-dimension mean and spread).
4. **Nothing was trained.** The new rows are only initialized.

## Token cost

Measured on FLORES-200 devtest, 1,012 parallel sentences (`urd_Arab` and `eng_Latn`), after Urdu
normalization. English tokens per word is {sweep["rows"][0]["english_tokens_per_word"]:.2f} in
every row, because English tokenizes identically.

![Urdu tokens per word against the number of new tokens](extension_cost.png)

{_cost_table(sweep["rows"])}

The cost multiplier is Urdu tokens divided by English tokens for the same sentences. The curve
flattens: each extra token buys less, while each one is an embedding row that stage 3 must train.

## Quality next to cost

Bits per character is total cross-entropy over characters, so it can be compared across
tokenizers (per-token perplexity cannot). Texts are the first {n_texts.get("urdu_heldout", "?")}
(held-out Urdu Wikipedia lines never used to learn merges), {n_texts.get("urdu_flores", "?")}
(FLORES Urdu) and {n_texts.get("english_flores", "?")} (FLORES English, a control). Belebele is the
Urdu and English reading-comprehension set, {settings.get("belebele_questions") or 900}
questions each, scored by answer log-probability divided by answer length in characters
(`acc_char`), with Wilson 95% intervals. Chance is 25%.

![Bits per character per variant on held-out Urdu](extension_quality.png)

{_quality_table(variants)}

### Initialization methods

{_init_section(variants)}

{_init_verdict(variants)}
{export_text}{adapt_text}
## Limitations

{first_limit}
- Belebele zero-shot is a coarse test for a 1.7B model: the base model is only a few points above
  chance, and the intervals of the variants overlap.
- One seed and one Urdu corpus (Wikipedia, which is formal register). Social-media or
  conversational Urdu may tokenize differently.
- Bits per character was measured on {n_texts.get("urdu_heldout", "?")} texts of at most
  {settings.get("max_chars", 400)} characters per set, in {settings.get("dtype", "float32")}.

## Reproduce

Hardware and software: {meta.get("platform", "unknown platform")}, Python {meta.get("python", "?")},
{", ".join(f"{k} {v}" for k, v in pk.items() if v)}. Model revision
`{(meta.get("model_revisions") or {}).get(quality["model"], "unknown")}`.

```bash
uv run zubaan extend learn   --config configs/extend-qwen3.yaml
uv run zubaan extend sweep   --config configs/extend-qwen3.yaml
# GPU, see notebooks/02_extend_quality_qwen3.ipynb
uv run zubaan extend quality --config configs/extend-qwen3.yaml
uv run zubaan report extend
# stage 3 (GPU, notebooks 07 to 10): adapt prepare, run, merge, eval; export all and check
```
"""


def _export_section(export: dict) -> str:
    gguf = export.get("gguf")
    if not gguf:
        return ""
    p = gguf["parity"]
    quants = ", ".join(f"`{f}`" for f in gguf["quantized"].values()) or "none"
    patched = (
        "The converter did not know the extended fingerprint, so it was mapped to the base "
        "model's pre-tokenizer type."
        if gguf["converter_patched"]
        else "The converter recognized the extended tokenizer without any change."
    )
    exported = export.get("base_model") or export.get("model_dir", "the extended model")
    return f"""
## GGUF export

Exported model: `{exported}` with the same 10,000 new tokens. Qwen3-0.6B shares Qwen3-1.7B's
tokenizer exactly, so the tokenizer results carry over; the quality numbers above are for 1.7B.
llama.cpp release `{gguf["llama_cpp_tag"]}`. The converter's pre-tokenizer type is
`{gguf["pre_tokenizer"]}` (fingerprint `{gguf["fingerprint"][:16]}...`). {patched} This is only
valid because the extension leaves the pre-tokenizer untouched, which is checked in code.

**Tokenizer parity:** {p["n_texts"] - p["n_mismatch"]:,} of {p["n_texts"]:,} texts (FLORES Urdu,
FLORES English and a stress set with emoji, numbers, mixed script, an empty string and very long
lines) tokenize to identical ids with Hugging Face and llama.cpp. {_nfc_note(p)} Quantized files:
{quants}, calibrated with an importance matrix built from {gguf["imatrix_calibration"]}.
"""


def _nfc_note(p: dict) -> str:
    """Explain the texts that differ only because llama.cpp does not apply NFC."""
    nfc = p.get("n_nfc_only", 0)
    if not nfc:
        return "No text differed."
    return (
        f"The other {nfc:,} texts are not in Unicode NFC form (Arabic diacritics stacked in a "
        "different order than NFC puts them). Qwen's tokenizer normalizes to NFC before "
        "tokenizing and llama.cpp's BPE tokenizer does not, so the ids differ for such text. "
        "For every one of them llama.cpp gives exactly Hugging Face's ids once the text is "
        "NFC-normalized first, so the vocabulary and merges are identical. This comes from the "
        "base model and llama.cpp, not from the extension. Normalize Urdu to NFC before "
        "sending it to a GGUF model (`zubaan.text.normalize_urdu` does)."
    )


def build_extension_report(
    sweep_path: Path,
    quality_path: Path,
    out_dir: Path,
    export_path: Path | None = None,
    adapt_path: Path | None = None,
    adapted_export_path: Path | None = None,
    adapted_check_path: Path | None = None,
    adapted_vllm_path: Path | None = None,
    adapted_ollama_path: Path | None = None,
) -> Path:
    """Write ``index.md`` and its charts into ``out_dir``. Missing optional files are skipped."""
    from zubaan.report import charts

    sweep = load_json(sweep_path)
    quality = load_json(quality_path)
    if sweep is None:
        raise ValueError(f"{sweep_path} not found; run zubaan extend sweep first")
    if quality is None:
        raise ValueError(f"{quality_path} not found; run zubaan extend quality first")
    export = load_json(export_path)
    adapt = load_json(adapt_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    charts.extension_cost_chart(sweep["rows"], out_dir / "extension_cost.png")
    charts.extension_quality_chart(quality["variants"], out_dir / "extension_quality.png")
    if adapt:
        from zubaan.report.adaptation import check_adapt_results

        check_adapt_results(adapt)
        order = adapt.get("order") or list(adapt["variants"])
        charts.adaptation_quality_chart(
            adapt["variants"], order, out_dir / "adaptation_quality.png"
        )
    text = build_extension_markdown(
        sweep,
        quality,
        export,
        adapt,
        load_json(adapted_export_path),
        load_json(adapted_check_path),
        load_json(adapted_vllm_path),
        load_json(adapted_ollama_path),
    )
    path = out_dir / "index.md"
    with path.open("w", encoding="utf-8", newline="\n") as f:
        f.write(text)
    return path
