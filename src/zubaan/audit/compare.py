"""Compare tokenizer fertility across model families on the same parallel corpus.

Tokens per word and the cost multiplier are fair to compare across tokenizers: they
measure how much text each token carries. Speeds in tokens per second are not, and
never appear here.
"""

from __future__ import annotations

import csv
from collections.abc import Callable
from pathlib import Path

from zubaan.audit.fertility import fertility_report

COLUMNS = (
    "model",
    "status",
    "vocab_size",
    "target_tokens_per_word",
    "reference_tokens_per_word",
    "target_chars_per_token",
    "reference_chars_per_token",
    "target_bytes_per_token",
    "cost_multiplier",
    "target_tokens",
    "reference_tokens",
    "target_words",
    "reference_words",
    "lossless",
    "n_unk",
    "reason",
)


def _row_from_report(report: dict) -> dict:
    t, r = report["target"], report["reference"]
    tc, rc = report["target_checks"], report["reference_checks"]
    lossless = tc["lossless"] and rc["lossless"]
    return {
        "model": report["tokenizer"],
        "status": "ok" if lossless else "lossy",
        "vocab_size": report["vocab_size"],
        "target_tokens_per_word": t["tokens_per_word"],
        "reference_tokens_per_word": r["tokens_per_word"],
        "target_chars_per_token": t["chars_per_token"],
        "reference_chars_per_token": r["chars_per_token"],
        "target_bytes_per_token": tc["bytes_per_token"],
        "cost_multiplier": report["cost_multiplier"],
        "target_tokens": t["n_tokens"],
        "reference_tokens": r["n_tokens"],
        "target_words": t["n_words"],
        "reference_words": r["n_words"],
        "lossless": lossless,
        "n_unk": tc["n_unk"] + rc["n_unk"],
        "reason": "" if lossless else "unknown tokens or failed round trip; counts understate",
    }


def _skipped(name: str, error: Exception) -> dict:
    reason = f"{type(error).__name__}: {(str(error).splitlines() or [''])[0]}"
    return {**dict.fromkeys(COLUMNS), "model": name, "status": "skipped", "reason": reason[:200]}


def compare_tokenizers(
    names: list[str],
    target_texts: list[str],
    reference_texts: list[str],
    load_fn: Callable[[str], object],
    normalize: bool = True,
    on_row: Callable[[dict], None] | None = None,
) -> list[dict]:
    """Fertility for each tokenizer on the same texts, one row per model.

    A tokenizer that cannot be loaded (gated, missing, offline) becomes a ``skipped``
    row with the reason, and the other models still run. Rows that load are sorted by
    cost multiplier, lowest (fairest to the target language) first.
    """
    if not names:
        raise ValueError("no tokenizers given")
    if len(target_texts) != len(reference_texts):
        raise ValueError("target and reference must be parallel corpora of equal length")
    rows = []
    for name in names:
        try:
            tok = load_fn(name)
            row = _row_from_report(
                fertility_report(tok, name, target_texts, reference_texts, normalize)
            )
        except Exception as e:  # one bad model must not stop the comparison
            row = _skipped(name, e)
        rows.append(row)
        if on_row is not None:
            on_row(row)
    measured = sorted(
        (r for r in rows if r["status"] != "skipped"), key=lambda r: r["cost_multiplier"]
    )
    return measured + [r for r in rows if r["status"] == "skipped"]


def write_csv(rows: list[dict], path: Path) -> Path:
    """Write the comparison as UTF-8 CSV, one row per model."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    return path
