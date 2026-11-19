"""Stage 3 evaluation: score saved model folders on identical texts and check the success tests.

Each entry of ``models`` is a folder written by ``zubaan extend run`` or ``zubaan adapt merge``
(or a Hub id). Every model is scored with its own ``tokenizer.json`` on the same character
filtered texts, so rows are comparable: bits per character on held-out Urdu, FLORES Urdu and
FLORES English, and Belebele accuracy in both languages. Per-token numbers are never compared
across tokenizers (see ``metrics.py``).

Results use the same layout as ``zubaan extend quality`` (``variants`` keyed by name), so the
report code can read them. Finished models in the output file are skipped, so a session that
dies resumes.
"""

from __future__ import annotations

import gc
import time
import traceback
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

from zubaan.extend.quality_run import (
    evaluate_variant,
    load_results,
    memory_report,
    release_gpu_memory,
    save_results,
)
from zubaan.extend.tokenizer_info import inspect_tokenizer_json, load_tokenizer_json


def variant_row(
    name: str,
    path: str,
    tokenizer_json: dict,
    scores: dict,
    seconds: float,
    memory: dict,
) -> dict:
    """One results row: the scores plus what was scored."""
    info = inspect_tokenizer_json(tokenizer_json)
    row = dict(scores)
    row.update(
        {
            "name": name,
            "path": path,
            "vocab_size": info.total_vocab_size,
            "seconds": round(seconds, 1),
            "memory": memory,
        }
    )
    return row


def run_model_eval(
    models: Mapping[str, str],
    load_model: Callable[[str], object],
    sets: dict[str, list[str]],
    belebele: dict[str, list[dict]],
    output: Path,
    device: str = "cpu",
    batch_size: int = 2,
    max_batch_tokens: int = 2048,
    meta: dict | None = None,
    log: Callable[[str], None] = print,
) -> dict:
    """Score every model in order, saving after each one; finished models are skipped."""
    if not models:
        raise ValueError("models is empty")
    results = load_results(output)
    results["meta"] = meta or results.get("meta", {})
    results["order"] = list(models)
    failures: dict[str, str] = {}
    for name, path in models.items():
        if name in results["variants"]:
            log(f"{name}: already done, skipping")
            continue
        log(f"{name}: loading {path}")
        t0 = time.perf_counter()
        model = None
        try:
            tokenizer_json = load_tokenizer_json(path)
            model = load_model(path)
            scores = evaluate_variant(
                model, tokenizer_json, sets, belebele, batch_size, device, log, max_batch_tokens
            )
            mem = memory_report()
            log(f"    memory: {mem}")
            results["variants"][name] = variant_row(
                name, path, tokenizer_json, scores, time.perf_counter() - t0, mem
            )
            save_results(results, output)
        except Exception as exc:  # one bad model must not throw away the others
            log(f"    FAILED {name}: {type(exc).__name__}: {exc}")
            log(traceback.format_exc())
            failures[name] = f"{type(exc).__name__}: {exc}"
        finally:
            del model
            gc.collect()
            release_gpu_memory()
    if failures:
        raise RuntimeError(f"{len(failures)} model(s) failed: {failures}")
    return results


def _bpc(row: dict, name: str) -> float:
    return row["bpc"][name]["bits_per_character"]


def _fmt_ci(ci: Sequence[float]) -> str:
    return f"[{ci[0]:.3f}, {ci[1]:.3f}]"


def success_checks(
    results: dict,
    reference: str,
    untrained: str,
    adapted: str,
    english_tolerance: float = 0.02,
) -> list[dict]:
    """The stage 3 success tests (decision 9) as ``name, passed, detail`` rows.

    ``reference`` is the base model, ``untrained`` the extended model before training.

    * English: bits per character on FLORES English at most ``english_tolerance`` above the
      reference. Both use the same English tokenization, so this compares like with like.
    * Urdu (must): held-out Urdu bits per character below the untrained model.
    * Urdu (stretch): at or below the reference, although the adapted model uses far fewer
      tokens. Bits per character is per character, so it is comparable across tokenizers.
    * Belebele Urdu ``acc_char`` above the reference. It only counts as a real gain if the 95%
      intervals do not overlap, and the detail says which case it is.
    """
    v = results["variants"]
    for key in (reference, untrained, adapted):
        if key not in v:
            raise ValueError(f"no results for {key!r}; have {sorted(v)}")
    ref, unt, ada = v[reference], v[untrained], v[adapted]
    checks = []

    d_en = _bpc(ada, "english_flores") - _bpc(ref, "english_flores")
    checks.append(
        {
            "name": "English bits per character within tolerance of the base",
            "passed": d_en <= english_tolerance + 1e-9,  # float noise must not fail the edge
            "detail": f"{_bpc(ada, 'english_flores'):.3f} vs {_bpc(ref, 'english_flores'):.3f} "
            f"(difference {d_en:+.3f}, allowed {english_tolerance:+.3f})",
        }
    )
    checks.append(
        {
            "name": "Urdu held-out bits per character below the untrained model (must)",
            "passed": _bpc(ada, "urdu_heldout") < _bpc(unt, "urdu_heldout"),
            "detail": f"{_bpc(ada, 'urdu_heldout'):.3f} vs {_bpc(unt, 'urdu_heldout'):.3f}",
        }
    )
    checks.append(
        {
            "name": "Urdu held-out bits per character at or below the base (stretch)",
            "passed": _bpc(ada, "urdu_heldout") <= _bpc(ref, "urdu_heldout"),
            "detail": f"{_bpc(ada, 'urdu_heldout'):.3f} vs {_bpc(ref, 'urdu_heldout'):.3f}",
        }
    )
    b_ref, b_ada = ref["belebele"]["urd_Arab"], ada["belebele"]["urd_Arab"]
    above = b_ada["acc_char"] > b_ref["acc_char"]
    separated = b_ada["acc_char_ci"][0] > b_ref["acc_char_ci"][1]
    if above and separated:
        note = "intervals do not overlap: a real gain"
    elif above:
        note = "intervals overlap: not distinguishable from noise"
    else:
        note = "not above the base"
    checks.append(
        {
            "name": "Belebele Urdu acc_char above the base",
            "passed": above,
            "detail": f"{b_ada['acc_char']:.3f} {_fmt_ci(b_ada['acc_char_ci'])} vs "
            f"{b_ref['acc_char']:.3f} {_fmt_ci(b_ref['acc_char_ci'])}; {note}",
        }
    )
    return checks


def comparison_rows(results: dict) -> list[dict]:
    """One row per model in table order: cost, bits per character, Belebele, delta to the first."""
    order = results.get("order") or list(results["variants"])
    v = results["variants"]
    first = v.get(order[0]) if order else None
    rows = []
    for name in order:
        if name not in v:
            continue
        r = v[name]
        row: dict = {"model": name, "vocab_size": r["vocab_size"]}
        for s, b in r["bpc"].items():
            row[f"bpc_{s}"] = b["bits_per_character"]
            row[f"tokens_per_char_{s}"] = b["tokens_per_char"]
            if first is not None and s in first["bpc"]:
                row[f"bpc_{s}_delta"] = b["bits_per_character"] - _bpc(first, s)
        for lang, b in r["belebele"].items():
            row[f"belebele_{lang}_acc_char"] = b["acc_char"]
            row[f"belebele_{lang}_ci"] = b["acc_char_ci"]
        rows.append(row)
    return rows
