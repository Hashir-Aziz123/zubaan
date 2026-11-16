"""Build extended model variants and measure their quality next to their token cost.

One *variant* is a base model, a number of new tokens and an initialization method. For each
variant this builds the extended tokenizer, resizes the model and initializes the new rows,
then measures:

* bits per character on held-out Urdu (Wikipedia lines never used to learn merges), on FLORES
  Urdu, and on FLORES English (a control: English tokenizes identically, so any change comes
  from the new rows stealing probability);
* Belebele multiple-choice accuracy in Urdu and English;
* the token cost on the same texts.

Results are written after every variant, and variants already in the output file are skipped,
so an interrupted run (a Kaggle session ending) resumes where it stopped.

Nothing here trains the model. The new rows are only initialized.
"""

from __future__ import annotations

import gc
import json
import time
from collections.abc import Callable, Sequence
from pathlib import Path

from zubaan.bench.quality import belebele_scores, text_bits_per_character
from zubaan.extend.init_emb import INIT_METHODS, merge_pieces, resize_and_init
from zubaan.extend.merge import extend_tokenizer_json, load_fast_tokenizer
from zubaan.extend.tokenizer_info import inspect_tokenizer_json, same_tokenizer
from zubaan.extend.train_bpe import Merge
from zubaan.text.normalize import normalize_urdu

CONTEXT_TOKEN = "<|endoftext|>"


def variant_key(n_new: int, method: str | None) -> str:
    """Name of a variant in the results file: ``base`` or ``n10000-mean``."""
    return "base" if not n_new else f"n{n_new}-{method}"


def select_bpc_texts(
    texts: Sequence[str], limit: int | None, max_chars: int, normalize: bool
) -> list[str]:
    """The first ``limit`` texts of at most ``max_chars`` characters, normalized if asked.

    The filter looks only at characters, so every tokenizer is scored on the same texts.
    """
    out = []
    for t in texts:
        t = normalize_urdu(t) if normalize else t
        if t and len(t) <= max_chars:
            out.append(t)
            if limit is not None and len(out) >= limit:
                break
    return out


def context_id(tokenizer_json: dict) -> int:
    """Id of the end-of-text token, used as the one-token context before every text."""
    for a in tokenizer_json.get("added_tokens", []):
        if a["content"] == CONTEXT_TOKEN:
            return a["id"]
    raise ValueError(f"tokenizer has no {CONTEXT_TOKEN} token")


def build_focus_rows(
    model,
    base: dict,
    ext: dict,
    pieces: list[list[int]],
    first_new_id: int,
    lines: Sequence[str],
    dim: int,
    seed: int,
):
    """FOCUS rows for a model: auxiliary vectors from ``lines`` tokenized with ``ext``."""
    from zubaan.extend.focus import build_aux_embeddings, focus_rows

    tok = load_fast_tokenizer(ext)
    streams = [tok.encode(line, add_special_tokens=False).ids for line in lines]
    aux = build_aux_embeddings(streams, dim=dim, window=5, min_count=5, seed=seed)
    weight = model.get_input_embeddings().weight.detach()[:first_new_id]
    return focus_rows(weight, aux, first_new_id, len(pieces), fallback_pieces=pieces)


def build_variant(
    model_factory: Callable[[], object],
    base: dict,
    merges: Sequence[Merge],
    n_new: int,
    method: str,
    seed: int = 0,
    focus_lines: Sequence[str] = (),
    focus_dim: int = 100,
):
    """Load a fresh model, extend it, and return ``(model, extended_tokenizer_json, info)``."""
    if method not in INIT_METHODS:
        raise ValueError(f"method must be one of {INIT_METHODS}")
    if n_new > len(merges):
        raise ValueError(f"asked for {n_new} new tokens but only {len(merges)} merges exist")
    chosen = list(merges[:n_new])
    ext = extend_tokenizer_json(base, chosen)
    first_new_id = inspect_tokenizer_json(base).next_free_id
    pieces = merge_pieces(base["model"]["vocab"], chosen)
    model = model_factory()
    rows, extra = None, {}
    if method == "focus":
        if not focus_lines:
            raise ValueError("focus needs focus_lines")
        rows, extra = build_focus_rows(
            model, base, ext, pieces, first_new_id, focus_lines, focus_dim, seed
        )
    info = resize_and_init(model, pieces, first_new_id, method, seed=seed, focus_rows=rows)
    info.update({"focus": extra} if extra else {})
    return model, ext, info


def evaluate_variant(
    model,
    tokenizer_json: dict,
    sets: dict[str, list[str]],
    belebele: dict[str, list[dict]],
    batch_size: int = 2,
    device: str = "cpu",
    log: Callable[[str], None] = lambda s: None,
) -> dict:
    """Bits per character on each text set and Belebele accuracy on each language."""
    tok = load_fast_tokenizer(tokenizer_json)
    ctx = context_id(tokenizer_json)
    out: dict = {"bpc": {}, "belebele": {}}
    for name, texts in sets.items():
        t0 = time.perf_counter()
        out["bpc"][name] = text_bits_per_character(
            model, tok, texts, ctx, batch_size=batch_size, device=device
        )
        out["bpc"][name]["seconds"] = round(time.perf_counter() - t0, 1)
        log(f"    bpc {name}: {out['bpc'][name]['bits_per_character']:.4f}")
    for lang, items in belebele.items():
        t0 = time.perf_counter()
        out["belebele"][lang] = belebele_scores(model, tok, items, ctx, device=device)
        out["belebele"][lang]["seconds"] = round(time.perf_counter() - t0, 1)
        log(
            f"    belebele {lang}: acc {out['belebele'][lang]['acc']:.3f} "
            f"acc_char {out['belebele'][lang]['acc_char']:.3f}"
        )
    return out


def load_results(path: Path) -> dict:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {"variants": {}}


def save_results(results: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
        f.write("\n")


def run_quality(
    model_name: str,
    model_factory: Callable[[], object],
    base: dict,
    merges: Sequence[Merge],
    sizes: Sequence[int],
    methods: Sequence[str],
    sets: dict[str, list[str]],
    belebele: dict[str, list[dict]],
    output: Path,
    seed: int = 0,
    focus_lines: Sequence[str] = (),
    focus_dim: int = 100,
    batch_size: int = 2,
    device: str = "cpu",
    meta: dict | None = None,
    log: Callable[[str], None] = print,
) -> dict:
    """Evaluate the base model and every (size, method) variant, saving after each one."""
    results = load_results(output)
    results.update({"model": model_name, "meta": meta or {}})
    plan = [(0, None)] + [(n, m) for n in sorted(set(sizes)) for m in methods]
    for n, method in plan:
        key = variant_key(n, method)
        if key in results["variants"]:
            log(f"{key}: already done, skipping")
            continue
        log(f"{key}: building")
        t0 = time.perf_counter()
        if n == 0:
            model, tokenizer_json, info = model_factory(), base, {"method": None, "n_new": 0}
        else:
            model, tokenizer_json, info = build_variant(
                model_factory, base, merges, n, method, seed, focus_lines, focus_dim
            )
        row = evaluate_variant(model, tokenizer_json, sets, belebele, batch_size, device, log)
        row.update(
            {
                "n_new_tokens": n,
                "method": method,
                "init": info,
                "vocab_size": inspect_tokenizer_json(tokenizer_json).total_vocab_size,
                "seconds": round(time.perf_counter() - t0, 1),
            }
        )
        results["variants"][key] = row
        save_results(results, output)
        del model
        gc.collect()
    return results


def check_weights_match_tokenizer(
    weights_tokenizer_json: dict, merges_tokenizer_json: dict
) -> None:
    """Raise unless the model's own tokenizer is identical to the one the merges extend."""
    if not same_tokenizer(weights_tokenizer_json, merges_tokenizer_json):
        raise ValueError(
            "the model's tokenizer differs from the tokenizer the merges were learned on; "
            "the merges would not apply to this model"
        )


def summary_rows(results: dict) -> list[dict]:
    """One flat row per variant: cost, bits per character, Belebele, relative to the base."""
    variants = results["variants"]
    base = variants.get("base")
    rows = []
    for key, v in variants.items():
        row = {
            "variant": key,
            "n_new_tokens": v["n_new_tokens"],
            "method": v["method"],
            "vocab_size": v["vocab_size"],
        }
        for name, b in v["bpc"].items():
            row[f"bpc_{name}"] = b["bits_per_character"]
            row[f"tokens_per_char_{name}"] = b["tokens_per_char"]
            if base and name in base["bpc"]:
                row[f"bpc_{name}_delta"] = (
                    b["bits_per_character"] - base["bpc"][name]["bits_per_character"]
                )
        for lang, b in v["belebele"].items():
            row[f"belebele_{lang}_acc_char"] = b["acc_char"]
            row[f"belebele_{lang}_acc"] = b["acc"]
        rows.append(row)
    return rows
