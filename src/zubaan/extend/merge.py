"""Append learned merges to a base tokenizer as real BPE merges.

The output is the base ``tokenizer.json`` with these changes and nothing else:

* new entries at the end of ``model.vocab`` (ids start after every id already in use),
* new pairs at the end of ``model.merges``,
* the strings of added tokens that are missing from ``model.vocab`` (Qwen's special tokens)
  are also listed in ``model.vocab`` at their existing ids, see below,
* nothing in ``added_tokens``, ``normalizer``, ``pre_tokenizer``, ``post_processor`` or
  ``decoder`` is touched. Keeping the pre-tokenizer unchanged is what lets the GGUF export
  reuse the base model's pre-tokenizer type later.

Why the added tokens are listed in the vocabulary: the ``tokenizers`` library ignores the
``id`` stored for an added token that is not in ``model.vocab`` and assigns it
``len(model.vocab) + position``. In Qwen3 that happens to equal the stored id (151643...), but
once the vocabulary grows, every special token (``<|im_end|>`` and the rest) would silently
move to a new id and no longer match the model's embeddings. Listing them in ``model.vocab``
at their own ids keeps every special token where it was.

``added_tokens`` is never used for new Urdu tokens: those are matched as literal strings before
BPE runs, which would change how text is split instead of extending the vocabulary.
"""

from __future__ import annotations

import copy
import json
import unicodedata
from collections.abc import Iterable, Sequence
from pathlib import Path

from zubaan.extend.selection import contains_arabic_block, validate_merges
from zubaan.extend.tokenizer_info import inspect_tokenizer_json
from zubaan.extend.train_bpe import Merge

# Everything in tokenizer.json except the BPE model must stay byte-for-byte as in the base.
_UNTOUCHED = ("normalizer", "pre_tokenizer", "post_processor", "decoder", "added_tokens")


def _merge_entry(base_merges: list, m: Merge):
    """A merge in the same format the base file uses (``[a, b]`` list or ``"a b"`` string)."""
    if base_merges and isinstance(base_merges[0], str):
        return f"{m.left} {m.right}"
    return [m.left, m.right]


def extend_tokenizer_json(base: dict, merges: Sequence[Merge]) -> dict:
    """Return a copy of ``base`` with ``merges`` appended. ``base`` is not modified.

    New token ids are assigned in merge order, starting at the first id that no model token
    and no added token uses (for Qwen3 that is 151669, not the padded embedding size).
    Raises ValueError if any merge is invalid (see ``validate_merges``).
    """
    info = inspect_tokenizer_json(base)
    vocab = base["model"]["vocab"]
    validate_merges(merges, vocab.keys())
    added_strings = {a["content"] for a in base.get("added_tokens", [])}
    clash = [m.token for m in merges if m.token in added_strings]
    if clash:
        raise ValueError(f"merge token equals an added token: {clash[0]!r}")

    out = copy.deepcopy(base)
    model = out["model"]
    for a in base.get("added_tokens", []):
        if a["content"] not in model["vocab"]:
            model["vocab"][a["content"]] = a["id"]  # keeps the special token at its id
    ids = list(model["vocab"].values())
    if len(ids) != len(set(ids)):
        raise ValueError("an added token shares its id with a vocabulary token")
    next_id = info.next_free_id
    for m in merges:
        model["vocab"][m.token] = next_id
        model["merges"].append(_merge_entry(base["model"]["merges"], m))
        next_id += 1
    return out


def assert_only_model_changed(base: dict, extended: dict) -> None:
    """Raise AssertionError if anything except ``model.vocab`` and ``model.merges`` differs."""
    for key in base:
        if key == "model":
            continue
        assert base[key] == extended[key], f"tokenizer.json field {key!r} was changed"
    assert set(base) == set(extended), "tokenizer.json gained or lost a top-level field"
    for key in base["model"]:
        if key in ("vocab", "merges"):
            continue
        assert base["model"][key] == extended["model"][key], f"model.{key} was changed"
    n_base, n_ext = len(base["model"]["merges"]), len(extended["model"]["merges"])
    assert extended["model"]["merges"][:n_base] == base["model"]["merges"], "base merges changed"
    for token, i in base["model"]["vocab"].items():
        assert extended["model"]["vocab"].get(token) == i, f"base token {token!r} changed id"
    for a in base.get("added_tokens", []):
        assert extended["model"]["vocab"].get(a["content"]) == a["id"], (
            f"added token {a['content']!r} is not kept at id {a['id']}"
        )
    listed = sum(
        1 for a in base.get("added_tokens", []) if a["content"] not in base["model"]["vocab"]
    )
    assert len(extended["model"]["vocab"]) - len(base["model"]["vocab"]) == n_ext - n_base + listed


def _load(tokenizer_json: dict):
    from tokenizers import Tokenizer

    return Tokenizer.from_str(json.dumps(tokenizer_json, ensure_ascii=False))


def load_fast_tokenizer(tokenizer_json: dict):
    """A ``tokenizers.Tokenizer`` for a parsed tokenizer.json."""
    return _load(tokenizer_json)


def compare_encodings(base: dict, extended: dict, texts: Iterable[str]) -> dict:
    """Encode every text with both tokenizers and report where the ids differ.

    Returns the number of texts, how many differ, and the first few differing texts.
    """
    tb, te = _load(base), _load(extended)
    n = n_diff = 0
    examples: list[str] = []
    for text in texts:
        n += 1
        if (
            tb.encode(text, add_special_tokens=False).ids
            != te.encode(text, add_special_tokens=False).ids
        ):
            n_diff += 1
            if len(examples) < 5:
                examples.append(text[:80])
    return {"n_texts": n, "n_different": n_diff, "examples": examples}


def check_roundtrip(tokenizer_json: dict, texts: Iterable[str]) -> dict:
    """Check ``decode(encode(x)) == x`` for every text.

    The tokenizer applies NFC first, so the expected output is the NFC form of the input; for
    text that is already NFC (all normalized Urdu) this is the input itself.
    """
    tok = _load(tokenizer_json)
    n = n_fail = 0
    examples: list[str] = []
    for text in texts:
        n += 1
        ids = tok.encode(text, add_special_tokens=False).ids
        if tok.decode(ids, skip_special_tokens=False) != unicodedata.normalize("NFC", text):
            n_fail += 1
            if len(examples) < 5:
                examples.append(text[:80])
    return {"n_texts": n, "n_failures": n_fail, "examples": examples}


def english_unchanged(base: dict, extended: dict, texts: Iterable[str]) -> dict:
    """Compare encodings on texts that must be unaffected (no Arabic-block characters).

    Raises ValueError if a text contains an Arabic-block character, since the guarantee
    does not cover it and the result would be misleading.
    """
    texts = list(texts)
    for t in texts:
        if contains_arabic_block(t):
            raise ValueError(f"text contains Arabic-block characters: {t[:40]!r}")
    return compare_encodings(base, extended, texts)


def merges_to_txt(tokenizer_json: dict) -> str:
    """``merges.txt`` content (with the version header) for a tokenizer.json."""
    lines = ["#version: 0.2"]
    for m in tokenizer_json["model"]["merges"]:
        lines.append(m if isinstance(m, str) else f"{m[0]} {m[1]}")
    return "\n".join(lines) + "\n"


def save_extended(extended: dict, base_dir: Path, out_dir: Path) -> Path:
    """Write the extended tokenizer next to the base tokenizer's other files.

    Writes ``tokenizer.json``, ``vocab.json`` and ``merges.txt`` (the last two are what
    the GGUF converter and slow tokenizers read) and copies the remaining small config files
    from ``base_dir`` unchanged. Model weights are never copied.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "tokenizer.json").open("w", encoding="utf-8", newline="\n") as f:
        json.dump(extended, f, ensure_ascii=False)
    with (out_dir / "vocab.json").open("w", encoding="utf-8", newline="\n") as f:
        json.dump(extended["model"]["vocab"], f, ensure_ascii=False)
    (out_dir / "merges.txt").write_text(merges_to_txt(extended), encoding="utf-8", newline="\n")
    for name in ("tokenizer_config.json", "special_tokens_map.json", "generation_config.json"):
        src = base_dir / name
        if src.exists():
            (out_dir / name).write_bytes(src.read_bytes())
    return out_dir
