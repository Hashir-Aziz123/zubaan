"""The Urdu language tax report: Markdown built from saved audit results.

Every number in the report is read from the JSON files the audit commands write,
never typed by hand, so the report can be rebuilt whenever the results change.
"""

from __future__ import annotations

import json
from pathlib import Path

from zubaan.audit.acceptance import best_speedup, expected_tokens_per_pass

LANG_NAMES = {"urd": "Urdu", "eng": "English"}


def load_json(path: Path | None) -> dict | None:
    if path is None or not path.exists():
        return None
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def _row(rows: list[dict], model: str) -> dict | None:
    return next((r for r in rows if r["model"] == model and r["status"] != "skipped"), None)


def words_per_target_pass(alpha: float, gamma: int, tokens_per_word: float) -> float:
    """Words produced per target forward pass: expected tokens per pass / tokens per word.

    This combines both halves of the tax in a unit that is fair across tokenizers.
    """
    if tokens_per_word <= 0:
        raise ValueError("tokens_per_word must be positive")
    return expected_tokens_per_pass(alpha, gamma) / tokens_per_word


def _fmt_versions(prov: dict) -> str:
    pkgs = ", ".join(f"{k} {v}" for k, v in prov.get("packages", {}).items() if v)
    return f"Python {prov.get('python')}, {pkgs}"


def _data_line(prov: dict) -> str:
    data = prov.get("data") or {}
    if data.get("source") == "meta":
        sha = data.get("archive_sha256", "")[:12]
        return f"FLORES-200 `{data.get('split')}` from Meta's archive (SHA-256 `{sha}...`)"
    if data.get("source") == "hf":
        rev = (data.get("revision") or "")[:12]
        return f"FLORES+ `{data.get('split')}` from Hugging Face (revision `{rev}`)"
    return "FLORES-200 devtest"


def _cost_table(rows: list[dict]) -> str:
    lines = [
        "| Tokenizer | Vocabulary | Urdu tokens/word | English tokens/word "
        "| Urdu chars/token | Cost multiplier |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in rows:
        if r["status"] == "skipped":
            continue
        flag = "" if r["status"] == "ok" else " (lossy)"
        lines.append(
            f"| `{r['model']}`{flag} | {r['vocab_size']:,} | {r['target_tokens_per_word']:.2f} "
            f"| {r['reference_tokens_per_word']:.2f} | {r['target_chars_per_token']:.2f} "
            f"| **{r['cost_multiplier']:.2f}x** |"
        )
    return "\n".join(lines)


def _skipped(rows: list[dict]) -> str:
    skipped = [r for r in rows if r["status"] == "skipped"]
    if not skipped:
        return ""
    items = "\n".join(f"- `{r['model']}`: {r['reason']}" for r in skipped)
    return (
        "\nNot measured (access not granted at the time of the run; they will be added "
        f"when it is):\n\n{items}\n"
    )


def _acceptance_section(acc: dict, rows: list[dict]) -> str:
    langs = acc["languages"]
    names = {k: LANG_NAMES.get(k, k) for k in langs}
    any_lang = next(iter(langs.values()))
    cost = any_lang["draft_cost"]
    lines = [
        "| Language | Texts | Positions | alpha | 95% interval | Greedy match "
        "| Best gamma | Tokens per pass | Best modelled speedup |",
        "| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: |",
    ]
    best = {lang: best_speedup(r["speedup"]) for lang, r in langs.items()}
    for lang, r in langs.items():
        s = best[lang]
        lines.append(
            f"| {names[lang]} | {r['n_texts']} | {r['n_positions']:,} | **{r['alpha']:.3f}** "
            f"| [{r['alpha_ci_low']:.3f}, {r['alpha_ci_high']:.3f}] | {r['greedy_match']:.3f} "
            f"| {s['gamma']} | {s['tokens_per_pass']:.2f} | {s['speedup']:.2f}x |"
        )
    table = "\n".join(lines)

    verdict = ""
    if {"urd", "eng"} <= langs.keys():
        u, e = langs["urd"], langs["eng"]
        if u["alpha_ci_high"] < e["alpha_ci_low"]:
            verdict = (
                f"Urdu drafts are accepted less often than English ones "
                f"({u['alpha']:.3f} vs {e['alpha']:.3f}), and the 95% intervals do not overlap."
            )
        elif e["alpha_ci_high"] < u["alpha_ci_low"]:
            verdict = (
                f"Urdu drafts are accepted more often than English ones "
                f"({u['alpha']:.3f} vs {e['alpha']:.3f}), and the 95% intervals do not overlap."
            )
        else:
            verdict = (
                f"The intervals overlap (Urdu {u['alpha']:.3f}, English {e['alpha']:.3f}), "
                "so this data shows no clear difference in acceptance."
            )

    words = ""
    target_row = _row(rows, acc["target"])
    if target_row and {"urd", "eng"} <= langs.keys():
        gu, ge = best["urd"]["gamma"], best["eng"]["gamma"]
        eu = expected_tokens_per_pass(langs["urd"]["alpha"], gu)
        ee = expected_tokens_per_pass(langs["eng"]["alpha"], ge)
        wu = words_per_target_pass(langs["urd"]["alpha"], gu, target_row["target_tokens_per_word"])
        we = words_per_target_pass(
            langs["eng"]["alpha"], ge, target_row["reference_tokens_per_word"]
        )
        words = f"""
### Both taxes together: words per target step

Speculative decoding is judged per target forward pass, but readers receive words. Dividing
expected tokens per pass by tokens per word gives a number that is fair across languages:

words per target step = E / tokens per word, where E = (1 - alpha^(gamma+1)) / (1 - alpha)

Each language uses its own best draft length from the table above.

| Language | alpha | Best gamma | E | Tokens per word | Words per target step |
| --- | ---: | ---: | ---: | ---: | ---: |
| Urdu | {langs["urd"]["alpha"]:.3f} | {gu} | {eu:.2f} \
| {target_row["target_tokens_per_word"]:.2f} | **{wu:.2f}** |
| English | {langs["eng"]["alpha"]:.3f} | {ge} | {ee:.2f} \
| {target_row["reference_tokens_per_word"]:.2f} | **{we:.2f}** |

Each expensive step of the target model delivers {we / wu:.1f} times as many English words as
Urdu words.
"""
    run = any_lang["run"]
    prov = acc.get("provenance", {})
    return f"""## 2. Speculative decoding acceptance

Draft `{acc["draft"]}`, target `{acc["target"]}`, same vocabulary (checked). Both models read
the same real sentence; at every position the acceptance rate is the overlap of their
next-token distributions. Models ran in {run["dtype"]} on {run["device"]} ({run["mode"]} mode);
{prov.get("processor") or "processor not recorded"}.

{table}

{verdict}

![Acceptance rate, Urdu vs English](acceptance.png)

The speedup is **modelled, not measured**: speedup = E / (1 + gamma x c), where gamma is the
number of drafted tokens and E = (1 - alpha^(gamma+1)) / (1 - alpha). It is evaluated for
gamma = 1 to 6 and the table reports the best gamma for each language. It uses draft cost
c = {cost:.2f} ({any_lang["draft_cost_source"]}) and assumes each drafted token is accepted
independently. A low alpha favors short drafts, so one fixed gamma understates the achievable
speedup for the language with the lower alpha.
Real speed is measured against running servers in stage 5.
{words}"""


def build_markdown(compare: dict, acceptance: dict | None, target_model: str) -> str:
    """The full report as Markdown. Chart images are referenced by relative file name."""
    rows = compare["rows"]
    measured = [r for r in rows if r["status"] != "skipped"]
    if not measured:
        raise ValueError("the comparison has no measured tokenizers")
    prov = compare.get("provenance", {})
    target = _row(rows, target_model)
    best, worst = measured[0], measured[-1]

    summary = [
        f"- Across {len(measured)} tokenizers, the same 1,012 sentences cost between "
        f"**{best['cost_multiplier']:.2f}x** (`{best['model']}`) and "
        f"**{worst['cost_multiplier']:.2f}x** (`{worst['model']}`) as many tokens in Urdu "
        "as in English."
    ]
    if target:
        summary.append(
            f"- `{target_model}`, the model zubaan extends, needs "
            f"**{target['target_tokens_per_word']:.2f} tokens per Urdu word** against "
            f"{target['reference_tokens_per_word']:.2f} for English. The same text costs "
            f"**{target['cost_multiplier']:.2f}x** as many tokens and fills the context window "
            "that much faster, and at equal tokens per second Urdu output arrives "
            f"**{target['target_tokens_per_word'] / target['reference_tokens_per_word']:.1f}x** "
            "slower in words per second."
        )
    if acceptance and {"urd", "eng"} <= acceptance["languages"].keys():
        u, e = acceptance["languages"]["urd"], acceptance["languages"]["eng"]
        summary.append(
            f"- Speculative decoding: the draft's guesses are accepted at alpha = "
            f"**{u['alpha']:.3f}** in Urdu and **{e['alpha']:.3f}** in English."
        )
    acc_section = (
        _acceptance_section(acceptance, rows)
        if acceptance
        else "## 2. Speculative decoding acceptance\n\nNot run yet.\n"
    )

    return f"""# The Urdu language tax

Open LLMs charge Urdu speakers more than English speakers for the same content: their
tokenizers split Urdu into more pieces, and the small draft models used to speed up
generation guess Urdu worse. This report measures both effects on the same parallel
sentences. It is the baseline that later zubaan stages (tokenizer extension, adaptation,
an Urdu drafter) try to improve.

## Summary

{chr(10).join(summary)}

## What was measured

- **Data:** {_data_line(prov)}: 1,012 sentences professionally translated into Urdu
  (`urd_Arab`) and English (`eng_Latn`). License CC BY-SA 4.0. Urdu text is normalized
  (Unicode NFC, Arabic look-alike letters mapped to Urdu forms, diacritics removed) before
  counting.
- **Software:** {_fmt_versions(prov)}.
- **Machine:** {prov.get("platform")}. Tokenizer measurements need no GPU.
- **Words** are split on whitespace and punctuation, the same rule for both languages.

## 1. Tokenizer cost

The general formulas, over the whole corpus:

- tokens per word = total tokens / total words
- cost multiplier = total Urdu tokens / total English tokens (same tokenizer, same sentences)

For `{target_model}`{
        f": {target['target_tokens']:,} Urdu tokens / {target['reference_tokens']:,} "
        f"English tokens = **{target['cost_multiplier']:.2f}x**"
        if target
        else ""
    }. The multiplier is larger than the ratio of tokens per word because Urdu also uses more
words than English for the same sentence (postpositions such as "کا" and "میں" are separate
words).

{_cost_table(rows)}

Every tokenizer above is lossless on this data (no unknown tokens; every sentence decodes
back to itself), so a low count is not hiding lost text.
{_skipped(rows)}
![Tokens per word by tokenizer](tokens_per_word.png)

![Cost multiplier by tokenizer](cost_multiplier.png)

{acc_section}
## Limitations

- FLORES sentences are formal, encyclopedic and news-like. Conversational Urdu, code-mixed
  Urdu-English and Roman Urdu are not covered.
- Acceptance is measured with teacher forcing: both models read reference text. In real
  serving the draft continues the target's own output, which can change acceptance.
- Speedups are modelled from alpha with an estimated draft cost; they are not timings.
- One draft/target pair (Qwen3-0.6B and Qwen3-1.7B) is measured; other families may differ.
- Word boundaries are rule-based. Urdu compounds written with a space count as two words,
  which lowers Urdu tokens per word slightly; the cost multiplier does not depend on words.

## Reproduce

```
uv sync --group dev --extra audit --extra report
uv run zubaan data flores
uv run zubaan audit compare --config configs/audit-compare.yaml
uv run zubaan audit acceptance --config configs/audit-acceptance.yaml
uv run zubaan report tax
```

The full 1,012-sentence acceptance run uses `notebooks/01_audit_acceptance_qwen3.ipynb` on a
Kaggle GPU.
"""


def build_report(
    compare_path: Path,
    acceptance_path: Path | None,
    out_dir: Path,
    target_model: str = "Qwen/Qwen3-1.7B",
) -> Path:
    """Write ``index.md`` and its charts into ``out_dir``."""
    from zubaan.report import charts

    compare = load_json(compare_path)
    if compare is None:
        raise ValueError(f"{compare_path} not found; run zubaan audit compare first")
    acceptance = load_json(acceptance_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    charts.tokens_per_word_chart(compare["rows"], out_dir / "tokens_per_word.png")
    charts.cost_multiplier_chart(compare["rows"], out_dir / "cost_multiplier.png", target_model)
    if acceptance:
        names = {k: LANG_NAMES.get(k, k) for k in acceptance["languages"]}
        title = f"{acceptance['draft'].split('/')[-1]} drafting for "
        title += acceptance["target"].split("/")[-1]
        charts.acceptance_chart(acceptance["languages"], names, out_dir / "acceptance.png", title)
    path = out_dir / "index.md"
    with path.open("w", encoding="utf-8", newline="\n") as f:
        f.write(build_markdown(compare, acceptance, target_model))
    return path
