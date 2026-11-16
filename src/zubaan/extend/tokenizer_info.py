"""Read the facts about a byte-level BPE tokenizer that extension depends on.

Extension appends real merges to ``tokenizer.json``. Before doing that we need to know
where the model vocabulary ends, which ids the added (special) tokens occupy, which
pre-tokenizer is in use, and how big the embedding matrix is. Reading these from the
file avoids assuming them.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class TokenizerInfo:
    """Facts about a tokenizer.json (and optionally the model config next to it)."""

    model_type: str
    model_vocab_size: int
    n_merges: int
    max_model_id: int
    n_added_tokens: int
    n_special_tokens: int
    first_added_id: int | None
    last_added_id: int | None
    next_free_id: int
    total_vocab_size: int
    pre_tokenizer_type: str
    normalizer_type: str | None
    byte_fallback: bool
    embedding_rows: int | None = None
    tie_word_embeddings: bool | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def inspect_tokenizer_json(data: dict, config: dict | None = None) -> TokenizerInfo:
    """Summarize a parsed tokenizer.json, plus ``vocab_size`` and tying from config.json.

    ``total_vocab_size`` counts distinct ids across the model vocabulary and added tokens
    (what ``len(tokenizer)`` reports; Qwen3: 151,669). ``next_free_id`` is the first id used
    by neither the model vocabulary nor an added token. New merge tokens must start there, not
    at the embedding matrix size, which is usually padded above it.
    """
    model = data.get("model") or {}
    if model.get("type") != "BPE":
        raise ValueError(f"only BPE tokenizers can be extended, got {model.get('type')!r}")
    vocab = model.get("vocab") or {}
    merges = model.get("merges") or []
    added = data.get("added_tokens") or []
    added_ids = [a["id"] for a in added]
    all_ids = list(vocab.values()) + added_ids
    pre = data.get("pre_tokenizer") or {}
    norm = data.get("normalizer") or {}
    return TokenizerInfo(
        model_type="BPE",
        model_vocab_size=len(vocab),
        n_merges=len(merges),
        max_model_id=max(vocab.values()) if vocab else -1,
        n_added_tokens=len(added),
        n_special_tokens=sum(1 for a in added if a.get("special")),
        first_added_id=min(added_ids) if added_ids else None,
        last_added_id=max(added_ids) if added_ids else None,
        next_free_id=max(all_ids) + 1 if all_ids else 0,
        total_vocab_size=len(set(all_ids)),
        pre_tokenizer_type=pre.get("type", "none"),
        normalizer_type=norm.get("type") if norm else None,
        byte_fallback=bool(model.get("byte_fallback", False)),
        embedding_rows=(config or {}).get("vocab_size"),
        tie_word_embeddings=(config or {}).get("tie_word_embeddings"),
    )


def same_tokenizer(a: dict, b: dict) -> bool:
    """True if two tokenizer.json files tokenize identically: same vocabulary, merges, added
    tokens, normalizer, pre-tokenizer and decoder.

    Used to confirm that a model's weights belong to the tokenizer the merges were learned on
    (Qwen3-0.6B and Qwen3-1.7B share one tokenizer, which is what lets one extension serve both).
    """
    keys = ("added_tokens", "normalizer", "pre_tokenizer", "post_processor", "decoder")
    if any(a.get(k) != b.get(k) for k in keys):
        return False
    return (
        a["model"]["vocab"] == b["model"]["vocab"] and a["model"]["merges"] == b["model"]["merges"]
    )


def find_local_files(name_or_path: str) -> Path:
    """Folder holding ``tokenizer.json``: a local path, or a Hub repo from the local cache.

    Downloads only the small tokenizer and config files, never model weights.
    """
    p = Path(name_or_path)
    if p.is_file():
        return p.parent
    if p.is_dir():
        return p
    from huggingface_hub import hf_hub_download

    folder = None
    for fname in ("config.json", "tokenizer.json"):
        try:
            folder = Path(hf_hub_download(name_or_path, fname)).parent
        except Exception:
            if fname == "tokenizer.json":
                raise
    assert folder is not None
    return folder


def load_tokenizer_json(name_or_path: str) -> dict:
    """Parsed ``tokenizer.json`` of a local folder, file or Hub repo."""
    folder = find_local_files(name_or_path)
    path = folder / "tokenizer.json"
    if not path.exists():
        raise ValueError(f"no tokenizer.json in {folder}")
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def inspect_tokenizer(name_or_path: str) -> TokenizerInfo:
    """Inspect a tokenizer by local path or Hub repo id (tokenizer and config files only)."""
    folder = find_local_files(name_or_path)
    data = load_tokenizer_json(str(folder))
    cfg_path = folder / "config.json"
    config = json.loads(cfg_path.read_text(encoding="utf-8")) if cfg_path.exists() else None
    return inspect_tokenizer_json(data, config)
