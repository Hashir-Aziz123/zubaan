"""Build one extended model and save it to disk, ready for export.

``zubaan extend quality`` builds each variant, scores it and throws it away. This module keeps
one variant: it extends the weights, writes the model and the extended tokenizer into one
folder (the layout ``transformers``, llama.cpp's converter and vLLM all read), and checks the
folder before anyone relies on it.

Nothing here trains the model. The new rows are only initialized, so quality is expected to be
worse than the base model until stage 3.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from pathlib import Path

from zubaan.extend.merge import (
    assert_only_model_changed,
    check_roundtrip,
    english_unchanged,
    save_extended,
)
from zubaan.extend.quality_run import build_variant
from zubaan.extend.tokenizer_info import (
    inspect_tokenizer_json,
    load_tokenizer_json,
)
from zubaan.extend.train_bpe import Merge

METADATA_FILE = "zubaan_extension.json"


def verify_extended_folder(
    folder: Path,
    base: dict,
    urdu_texts: Sequence[str],
    english_texts: Sequence[str],
    expected_rows: int | None = None,
) -> dict:
    """Check a saved extended model folder and return what was checked.

    Raises ``ValueError`` when any of these fail:

    * only ``model.vocab`` and ``model.merges`` differ from the base ``tokenizer.json``;
    * English text (no Arabic-block characters) tokenizes exactly as with the base tokenizer;
    * Urdu text round-trips, ``decode(encode(x)) == x``;
    * ``config.json`` says the model has as many embedding rows as the tokenizer has ids;
    * if ``expected_rows`` is given, the saved weights have exactly that many embedding rows.
    """
    ext = load_tokenizer_json(str(folder))
    assert_only_model_changed(base, ext)
    english = english_unchanged(base, ext, english_texts)
    if english["n_different"]:
        raise ValueError(f"English tokenization changed: {english['examples']}")
    roundtrip = check_roundtrip(ext, urdu_texts)
    if roundtrip["n_failures"]:
        raise ValueError(f"Urdu does not round-trip: {roundtrip['examples']}")
    vocab_size = inspect_tokenizer_json(ext).total_vocab_size
    config = json.loads((folder / "config.json").read_text(encoding="utf-8"))
    if config["vocab_size"] != vocab_size:
        raise ValueError(
            f"config.json vocab_size {config['vocab_size']} differs from the "
            f"tokenizer's {vocab_size}"
        )
    rows = None
    if expected_rows is not None:
        rows = saved_embedding_rows(folder)
        if rows != expected_rows:
            raise ValueError(f"saved embedding has {rows} rows, expected {expected_rows}")
    return {
        "vocab_size": vocab_size,
        "english_texts": english["n_texts"],
        "urdu_texts": roundtrip["n_texts"],
        "embedding_rows": rows,
    }


def saved_embedding_rows(folder: Path) -> int:
    """Rows of the input embedding in the saved ``.safetensors`` files (reads headers only)."""
    from safetensors import safe_open

    for path in sorted(folder.glob("*.safetensors")):
        with safe_open(str(path), framework="pt") as f:
            for name in f.keys():  # noqa: SIM118 - safe_open handles are not dict-like
                if name.endswith("embed_tokens.weight"):
                    return f.get_slice(name).get_shape()[0]
    raise ValueError(f"no embed_tokens.weight found in {folder}")


def build_extended_model(
    model_factory: Callable[[], object],
    base: dict,
    base_dir: Path,
    merges: Sequence[Merge],
    n_new: int,
    method: str,
    out_dir: Path,
    seed: int = 0,
    focus_lines: Sequence[str] = (),
    focus_dim: int = 100,
    base_model_name: str = "",
) -> dict:
    """Extend a model with ``n_new`` merges, save model and tokenizer to ``out_dir``.

    ``base_dir`` is the base model's folder (its small tokenizer config files are copied).
    Returns the metadata dict that is also written to ``out_dir / zubaan_extension.json``.
    """
    if n_new < 1:
        raise ValueError("n_new must be at least 1")
    model, ext, info = build_variant(
        model_factory, base, merges, n_new, method, seed, focus_lines, focus_dim
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir, safe_serialization=True)
    save_extended(ext, base_dir, out_dir)
    meta = {
        "base_model": base_model_name,
        "n_new_tokens": n_new,
        "init_method": method,
        "first_new_id": info["first_new_id"],
        "vocab_size": inspect_tokenizer_json(ext).total_vocab_size,
        "tied_embeddings": info["tied"],
        "trained": False,
        "note": "New embedding rows are initialized, not trained (stage 3 trains them).",
    }
    (out_dir / METADATA_FILE).write_text(
        json.dumps(meta, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    return meta
