"""Learn merges from an Urdu corpus and measure what each vocabulary size buys.

Every row of a sweep answers one question: if we add the first N learned merges, how many
tokens does Urdu cost, and did anything else change? English is measured on the same
parallel sentences as a control, and the held-out Wikipedia lines (never seen in learning)
show whether the gain carries over to text the merges were not fitted on.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

from zubaan.audit.fertility import compute_fertility
from zubaan.extend.merge import (
    check_roundtrip,
    compare_encodings,
    extend_tokenizer_json,
    load_fast_tokenizer,
)
from zubaan.extend.selection import pair_is_safe, validate_merges
from zubaan.extend.tokenizer_info import inspect_tokenizer_json
from zubaan.extend.train_bpe import Merge, learn_merges, word_piece_counts


class FastTokenizerAdapter:
    """Gives a ``tokenizers.Tokenizer`` the encode/decode interface the audit code expects."""

    unk_token_id = None

    def __init__(self, tokenizer, total_vocab_size: int) -> None:
        self._tok = tokenizer
        self._n = total_vocab_size

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        return self._tok.encode(text, add_special_tokens=add_special_tokens).ids

    def decode(self, ids: list[int], skip_special_tokens: bool = False) -> str:
        return self._tok.decode(ids, skip_special_tokens=skip_special_tokens)

    def __len__(self) -> int:
        return self._n


def adapter_for(tokenizer_json: dict) -> FastTokenizerAdapter:
    info = inspect_tokenizer_json(tokenizer_json)
    return FastTokenizerAdapter(load_fast_tokenizer(tokenizer_json), info.total_vocab_size)


def learn_urdu_merges(
    lines: Sequence[str], base: dict, n_merges: int, min_count: int = 2
) -> list[Merge]:
    """Learn up to ``n_merges`` English-safe merges for ``lines`` on top of the ``base`` tokenizer.

    The lines should already be normalized (``zubaan data urdu`` does that).
    """
    counts = word_piece_counts(lines, base)
    merges = learn_merges(
        counts,
        n_merges,
        existing_tokens=base["model"]["vocab"].keys(),
        pair_filter=pair_is_safe,
        min_count=min_count,
    )
    validate_merges(merges, base["model"]["vocab"].keys())
    return merges


def save_merges(merges: Sequence[Merge], path: Path) -> Path:
    """Write merges as JSON ``[[left, right, count], ...]`` in learning order."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        json.dump([[m.left, m.right, m.count] for m in merges], f, ensure_ascii=False)
    return path


def load_merges(path: Path) -> list[Merge]:
    with path.open(encoding="utf-8") as f:
        return [Merge(a, b, c) for a, b, c in json.load(f)]


def measure_tokenizer(
    tokenizer_json: dict,
    urdu: Sequence[str],
    english: Sequence[str],
    heldout: Sequence[str] = (),
) -> dict:
    """Tokens per word, cost multiplier and round trips for one tokenizer.

    ``urdu`` and ``english`` must be parallel (FLORES). Urdu is measured twice: normalized
    (comparable with the stage 1 report) and as written, since real users do not normalize.
    """
    if len(urdu) != len(english):
        raise ValueError("urdu and english must be parallel corpora of equal length")
    tok = adapter_for(tokenizer_json)
    u = compute_fertility(tok, urdu, normalize=True)
    u_raw = compute_fertility(tok, urdu, normalize=False)
    e = compute_fertility(tok, english, normalize=False)
    row = {
        "vocab_size": len(tok),
        "urdu_tokens": u.n_tokens,
        "urdu_tokens_per_word": u.tokens_per_word,
        "urdu_chars_per_token": u.chars_per_token,
        "english_tokens": e.n_tokens,
        "english_tokens_per_word": e.tokens_per_word,
        "cost_multiplier": u.n_tokens / e.n_tokens,
        "raw_urdu_tokens_per_word": u_raw.tokens_per_word,
        "raw_cost_multiplier": u_raw.n_tokens / e.n_tokens,
    }
    if heldout:
        h = compute_fertility(tok, heldout, normalize=False)
        row["heldout_tokens"] = h.n_tokens
        row["heldout_tokens_per_word"] = h.tokens_per_word
        row["heldout_chars_per_token"] = h.chars_per_token
    return row


def sweep_vocab_sizes(
    base: dict,
    merges: Sequence[Merge],
    sizes: Sequence[int],
    urdu: Sequence[str],
    english: Sequence[str],
    heldout: Sequence[str] = (),
) -> list[dict]:
    """One row for the base tokenizer (0 new tokens) and one per size in ``sizes``.

    Sizes larger than the number of learned merges are reported with the merges available
    (``n_new_tokens`` shows the real number), never silently padded.
    Each row also records the safety checks: English encodings that differ from the base
    (must be 0) and Urdu round-trip failures (must be 0).
    """
    if any(s <= 0 for s in sizes):
        raise ValueError("sizes must be positive")
    rows = []
    for n in [0, *sorted(set(sizes))]:
        use = list(merges[:n])
        ext = extend_tokenizer_json(base, use) if use else base
        measured = measure_tokenizer(ext, urdu, english, heldout)
        row = {"n_new_tokens": len(use), "requested": n, **measured}
        row["english_different"] = compare_encodings(base, ext, english)["n_different"]
        row["urdu_roundtrip_failures"] = check_roundtrip(ext, urdu)["n_failures"]
        rows.append(row)
    return rows
