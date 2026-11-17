"""One call from a base model and an Urdu text file to a saved, verified extended model.

Steps: learn merges (skipped when the merges file exists), extend the weights and tokenizer,
save both to ``<output_dir>/model``, then verify the saved folder (English tokenization
unchanged, Urdu round-trips, embedding rows match the vocabulary).
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from zubaan.config import ExtendConfig
from zubaan.extend.build import build_extended_model, verify_extended_folder
from zubaan.extend.merge import assert_only_model_changed, extend_tokenizer_json, save_extended
from zubaan.extend.quality_run import check_weights_match_tokenizer
from zubaan.extend.sweep import learn_urdu_merges, load_merges, save_merges, sweep_vocab_sizes
from zubaan.extend.tokenizer_info import find_local_files, load_tokenizer_json
from zubaan.io import read_lines
from zubaan.text.normalize import normalize_urdu


def run_extend(
    cfg: ExtendConfig,
    n_new: int | None = None,
    method: str | None = None,
    relearn: bool = False,
    log: Callable[[str], None] = print,
    model_factory: Callable[[], object] | None = None,
) -> dict:
    """Learn merges if needed, build and save the extended model, verify it, return a summary.

    ``model_factory`` loads the base weights; it defaults to ``load_causal_lm`` with
    ``cfg.model_dtype`` on ``cfg.device`` and is replaced by a tiny model in tests.
    """
    n = n_new or cfg.n_new_tokens
    method = method or cfg.init_method
    base = load_tokenizer_json(cfg.base_model)
    base_dir = find_local_files(cfg.base_model)

    if relearn or not cfg.merges_file.exists():
        n_learn = max([n, *cfg.sizes])
        lines = read_lines(cfg.corpus, cfg.max_lines)
        log(f"Learning up to {n_learn:,} merges from {len(lines):,} lines...")
        merges = learn_urdu_merges(lines, base, n_learn, cfg.min_count)
        save_merges(merges, cfg.merges_file)
    else:
        merges = load_merges(cfg.merges_file)
        log(f"Using {len(merges):,} merges from {cfg.merges_file}")
    if n > len(merges):
        raise ValueError(f"asked for {n} new tokens but only {len(merges)} merges exist")

    # The tokenizer-only folder (also useful on its own, e.g. for 'audit fertility').
    ext = extend_tokenizer_json(base, merges[:n])
    assert_only_model_changed(base, ext)
    save_extended(ext, base_dir, cfg.output_dir / "tokenizer")

    urdu_raw = read_lines(cfg.eval_urdu)
    english_eval = read_lines(cfg.eval_english)
    cost_row = sweep_vocab_sizes(base, merges, [n], urdu_raw, english_eval, [])[-1]
    urdu_eval = [normalize_urdu(t) for t in urdu_raw]

    if model_factory is None:
        from zubaan.audit.acceptance_lm import load_causal_lm

        check_weights_match_tokenizer(load_tokenizer_json(cfg.base_model), base)

        def model_factory():
            return load_causal_lm(cfg.base_model, cfg.model_dtype, cfg.device)

    out_dir = cfg.output_dir / "model"
    log(f"Building {n:,} new tokens with '{method}' init into {out_dir} ...")
    focus_lines = read_lines(cfg.corpus, cfg.focus_lines) if method == "focus" else ()
    meta = build_extended_model(
        model_factory,
        base,
        base_dir,
        merges,
        n,
        method,
        out_dir,
        seed=cfg.seed,
        focus_lines=focus_lines,
        focus_dim=cfg.focus_dim,
        base_model_name=cfg.base_model,
    )
    checked = verify_extended_folder(
        out_dir, base, urdu_eval, english_eval, expected_rows=meta["vocab_size"]
    )
    log(f"Verified {out_dir}: {checked}")
    return {
        "model_dir": str(out_dir),
        "tokenizer_dir": str(Path(cfg.output_dir) / "tokenizer"),
        "metadata": meta,
        "verified": checked,
        "cost": {
            k: cost_row[k]
            for k in (
                "n_new_tokens",
                "urdu_tokens_per_word",
                "english_tokens_per_word",
                "cost_multiplier",
                "english_different",
                "urdu_roundtrip_failures",
            )
        },
    }
