"""Command-line interface. Kept thin: all logic lives in the library modules."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Annotated, Optional

import typer
from rich.console import Console
from rich.table import Table

from zubaan import __version__
from zubaan.audit.fertility import fertility_report
from zubaan.config import load_config
from zubaan.io import read_lines
from zubaan.provenance import run_metadata, write_json

app = typer.Typer(
    help="Measure and remove the inference tax non-English languages pay.",
    no_args_is_help=True,
)
audit_app = typer.Typer(help="Stage 1: measure the language tax.", no_args_is_help=True)
app.add_typer(audit_app, name="audit")
data_app = typer.Typer(help="Download evaluation data.", no_args_is_help=True)
app.add_typer(data_app, name="data")
report_app = typer.Typer(help="Build reports from saved results.", no_args_is_help=True)
app.add_typer(report_app, name="report")
extend_app = typer.Typer(help="Stage 2: extend a tokenizer with Urdu merges.", no_args_is_help=True)
app.add_typer(extend_app, name="extend")
export_app = typer.Typer(
    help="Stage 2: export an extended model to GGUF, Ollama and vLLM.", no_args_is_help=True
)
app.add_typer(export_app, name="export")
adapt_app = typer.Typer(
    help="Stage 3: continued pretraining so the model learns its new tokens.", no_args_is_help=True
)
app.add_typer(adapt_app, name="adapt")
tune_app = typer.Typer(help="Stage 4: restore chat behavior.", no_args_is_help=True)
app.add_typer(tune_app, name="tune")

console = Console()


def load_tokenizer(name: str):
    """Load a Hugging Face tokenizer. Imported lazily so the core CLI stays light."""
    try:
        from transformers import AutoTokenizer
    except ImportError as e:  # pragma: no cover - depends on installed extras
        raise typer.BadParameter(
            "This command needs the audit extra: pip install 'zubaan[audit]'"
        ) from e
    return AutoTokenizer.from_pretrained(name)


def _section(config: Path | None, name: str):
    """One section of a YAML config, or None when no config was given."""
    if config is None:
        return None
    section = getattr(load_config(config), name)
    if section is None:
        raise typer.BadParameter(f"{config} has no '{name}' section")
    return section


@app.command()
def version() -> None:
    """Print the installed version."""
    console.print(__version__)


@data_app.command("flores")
def data_flores(
    lang: Annotated[str, typer.Option(help="Target FLORES language code")] = "urd_Arab",
    ref: Annotated[str, typer.Option(help="Reference FLORES language code")] = "eng_Latn",
    split: Annotated[str, typer.Option(help="dev or devtest")] = "devtest",
    out: Annotated[Path, typer.Option(help="Output directory")] = Path("data"),
    source: Annotated[
        str, typer.Option(help="meta (FLORES-200), hf (FLORES+, gated) or auto")
    ] = "meta",
) -> None:
    """Download parallel FLORES sentences, one per line, aligned across languages."""
    from zubaan.data.flores import download_flores

    result = download_flores([lang, ref], split=split, out_dir=out, source=source)
    console.print(f"FLORES {split} from [bold]{result.meta['source']}[/]: {result.n_lines} lines")
    if "fallback_reason" in result.meta:
        console.print(f"[yellow]Hugging Face not used:[/] {result.meta['fallback_reason']}")
    for path in result.paths.values():
        console.print(f"  {path}")
    console.print(f"  {out / 'flores_meta.json'}")


@audit_app.command("fertility")
def audit_fertility(
    tokenizer: Annotated[
        Optional[str],  # noqa: UP045 - Typer needs Optional on Python 3.10
        typer.Option("--tokenizer", "-t", help="HF repo id or local path"),
    ] = None,
    corpus: Annotated[
        Optional[Path],  # noqa: UP045
        typer.Option("--corpus", "-c", exists=True, help="Target-language file"),
    ] = None,
    reference: Annotated[
        Optional[Path],  # noqa: UP045
        typer.Option("--reference", "-r", exists=True, help="Parallel reference-language file"),
    ] = None,
    max_texts: Annotated[
        Optional[int],  # noqa: UP045
        typer.Option(help="Use only the first N lines"),
    ] = None,
    no_normalize: Annotated[bool, typer.Option(help="Skip Urdu normalization")] = False,
    examples: Annotated[int, typer.Option(help="Show how the first N sentences split")] = 0,
    config: Annotated[
        Optional[Path],  # noqa: UP045
        typer.Option("--config", exists=True, help="YAML config with an audit section"),
    ] = None,
    out: Annotated[
        Optional[Path],  # noqa: UP045
        typer.Option("--out", help="Save the JSON result (with provenance) here"),
    ] = None,
    as_json: Annotated[bool, typer.Option("--json", help="Print machine-readable JSON")] = False,
) -> None:
    """Tokens per word and characters per token, plus cost versus a reference language.

    Flags override values from --config.
    """
    cfg = _section(config, "audit")
    tokenizer = tokenizer or (cfg.tokenizer if cfg else None)
    corpus = corpus or (cfg.corpus if cfg else None)
    reference = reference or (cfg.reference_corpus if cfg else None)
    max_texts = max_texts or (cfg.max_texts if cfg else None)
    normalize = not no_normalize and (cfg.normalize if cfg else True)
    if tokenizer is None or corpus is None:
        raise typer.BadParameter("give --tokenizer and --corpus, or a --config with them")

    tok = load_tokenizer(tokenizer)
    ref_texts = read_lines(reference, max_texts) if reference is not None else None
    result = fertility_report(
        tok, tokenizer, read_lines(corpus, max_texts), ref_texts, normalize, examples
    )
    result["provenance"] = run_metadata([tokenizer], corpus.parent / "flores_meta.json")
    if out is not None:
        write_json(result, out)

    if as_json:
        console.print_json(json.dumps(result, ensure_ascii=False))
        return

    table = Table(title=f"Fertility: {tokenizer}")
    table.add_column("Corpus")
    for col in ("Texts", "Words", "Tokens", "Tokens/word", "Chars/token", "Lossless"):
        table.add_column(col, justify="right")
    for name in ("target", "reference"):
        if name not in result:
            continue
        r, c = result[name], result[f"{name}_checks"]
        table.add_row(
            name,
            str(r["n_texts"]),
            str(r["n_words"]),
            str(r["n_tokens"]),
            f"{r['tokens_per_word']:.2f}",
            f"{r['chars_per_token']:.2f}",
            "yes" if c["lossless"] else f"[red]no[/] ({c['n_unk']} unk)",
        )
    console.print(table)
    if "cost_multiplier" in result:
        console.print(f"Cost multiplier vs reference: [bold]{result['cost_multiplier']:.2f}x[/]")
    for name, pieces_list in result.get("examples", {}).items():
        for pieces in pieces_list:
            console.print(f"{name}: " + " | ".join(repr(p) for p in pieces))
    if out is not None:
        console.print(f"Saved {out}")


@audit_app.command("compare")
def audit_compare(
    tokenizers: Annotated[
        Optional[list[str]],  # noqa: UP045
        typer.Option("--tokenizer", "-t", help="Tokenizer to include; repeat for several"),
    ] = None,
    corpus: Annotated[
        Optional[Path],  # noqa: UP045
        typer.Option("--corpus", "-c", exists=True, help="Target-language file"),
    ] = None,
    reference: Annotated[
        Optional[Path],  # noqa: UP045
        typer.Option("--reference", "-r", exists=True, help="Parallel reference-language file"),
    ] = None,
    max_texts: Annotated[
        Optional[int],  # noqa: UP045
        typer.Option(help="Use only the first N lines"),
    ] = None,
    out_dir: Annotated[
        Optional[Path],  # noqa: UP045
        typer.Option("--out-dir", help="Where compare.csv and compare.json go"),
    ] = None,
    config: Annotated[
        Optional[Path],  # noqa: UP045
        typer.Option("--config", exists=True, help="YAML config with a compare section"),
    ] = None,
) -> None:
    """Fertility for several tokenizers on the same parallel data, as one table.

    Tokenizers that cannot be loaded (for example gated ones) are skipped, not fatal.
    """
    from zubaan.audit.compare import compare_tokenizers, write_csv

    cfg = _section(config, "compare")
    tokenizers = tokenizers or (cfg.models if cfg else None)
    corpus = corpus or (cfg.corpus if cfg else None)
    reference = reference or (cfg.reference_corpus if cfg else None)
    max_texts = max_texts or (cfg.max_texts if cfg else None)
    out_dir = out_dir or (cfg.output_dir if cfg else Path("reports"))
    normalize = cfg.normalize if cfg else True
    if not tokenizers or corpus is None or reference is None:
        raise typer.BadParameter("give --tokenizer, --corpus and --reference, or a --config")

    def progress(row: dict) -> None:
        if row["status"] == "skipped":
            console.print(f"[yellow]skipped[/] {row['model']}: {row['reason']}")
        else:
            console.print(f"[green]done[/] {row['model']}")

    rows = compare_tokenizers(
        tokenizers,
        read_lines(corpus, max_texts),
        read_lines(reference, max_texts),
        load_tokenizer,
        normalize,
        on_row=progress,
    )
    measured = [r for r in rows if r["status"] != "skipped"]
    result = {
        "corpus": str(corpus),
        "reference_corpus": str(reference),
        "normalize": normalize,
        "rows": rows,
        "provenance": run_metadata(tokenizers, corpus.parent / "flores_meta.json"),
    }
    write_json(result, out_dir / "compare.json")
    write_csv(rows, out_dir / "compare.csv")

    table = Table(title="Tokenizer comparison, sorted by cost multiplier")
    table.add_column("Model", no_wrap=True)
    for col in ("Vocab", "Target tok/word", "Ref tok/word", "Cost"):
        table.add_column(col, justify="right")
    table.add_column("Status")
    for r in measured:
        table.add_row(
            r["model"],
            f"{r['vocab_size']:,}",
            f"{r['target_tokens_per_word']:.2f}",
            f"{r['reference_tokens_per_word']:.2f}",
            f"{r['cost_multiplier']:.2f}x",
            r["status"],
        )
    console.print(table)
    console.print(f"Saved {out_dir / 'compare.csv'} and {out_dir / 'compare.json'}")
    if not measured:
        console.print("[red]No tokenizer could be loaded.[/]")
        raise typer.Exit(code=1)


@report_app.command("tax")
def report_tax(
    compare: Annotated[Path, typer.Option(help="Output of audit compare")] = Path(
        "reports/compare.json"
    ),
    acceptance: Annotated[Path, typer.Option(help="Output of audit acceptance")] = Path(
        "reports/acceptance_qwen3.json"
    ),
    target_model: Annotated[str, typer.Option(help="Model highlighted in the report")] = (
        "Qwen/Qwen3-1.7B"
    ),
    out: Annotated[Path, typer.Option(help="Report directory")] = Path(
        "docs/reports/urdu-language-tax"
    ),
) -> None:
    """Write "The Urdu language tax" report (Markdown plus PNG charts)."""
    from zubaan.report.tax import build_report

    if not acceptance.exists():
        console.print(f"[yellow]{acceptance} not found; the acceptance section is left empty.[/]")
    path = build_report(compare, acceptance, out, target_model)
    console.print(f"Wrote {path}")


@report_app.command("extend")
def report_extend(
    sweep: Annotated[Path, typer.Option(help="Output of extend sweep")] = Path(
        "reports/extend_sweep.json"
    ),
    quality: Annotated[Path, typer.Option(help="Output of extend quality")] = Path(
        "reports/extend_quality_qwen3-1.7b.json"
    ),
    export_summary: Annotated[Path, typer.Option(help="export_summary.json from export")] = Path(
        "reports/export_summary.json"
    ),
    adapt_quality: Annotated[Path, typer.Option(help="Stage 3: output of adapt eval")] = Path(
        "reports/adapt_quality.json"
    ),
    adapted_export: Annotated[
        Path, typer.Option(help="Stage 3: export_summary.json of the adapted model")
    ] = Path("reports/export_summary_09.json"),
    adapted_check: Annotated[
        Path, typer.Option(help="Stage 3: export check results of the adapted model")
    ] = Path("reports/export_check_09.json"),
    adapted_vllm: Annotated[
        Path, typer.Option(help="Stage 3: vLLM check of the adapted model")
    ] = Path("reports/vllm_check_adapted.json"),
    adapted_ollama: Annotated[
        Path, typer.Option(help="Stage 3: local Ollama chat check of the adapted model")
    ] = Path("reports/ollama_check_adapted.json"),
    out: Annotated[Path, typer.Option(help="Report directory")] = Path(
        "docs/reports/tokenizer-extension"
    ),
) -> None:
    """Write the tokenizer extension report: cost, quality, GGUF parity and stage 3 results.

    Optional inputs that do not exist are left out with a note.
    """
    from zubaan.report.extension import build_extension_report

    for f, what in [
        (export_summary, "export section"),
        (adapt_quality, "stage 3 section"),
        (adapted_export, "stage 3 GGUF part"),
        (adapted_check, "stage 3 quantization part"),
        (adapted_vllm, "stage 3 vLLM part"),
        (adapted_ollama, "stage 3 Ollama part"),
    ]:
        if not f.exists():
            console.print(f"[yellow]{f} not found; the {what} is left out.[/]")
    try:
        path = build_extension_report(
            sweep, quality, out, export_summary, adapt_quality, adapted_export, adapted_check,
            adapted_vllm, adapted_ollama,
        )  # fmt: skip
    except ValueError as e:
        raise typer.BadParameter(str(e)) from e
    console.print(f"Wrote {path}")


def _planned(stage: int, what: str) -> None:
    console.print(f"[yellow]Not implemented yet.[/] Planned for stage {stage}: {what}")
    raise typer.Exit(code=1)


@audit_app.command("acceptance")
def audit_acceptance(
    draft: Annotated[
        Optional[str],  # noqa: UP045
        typer.Option("--draft", "-d", help="Draft model"),
    ] = None,
    target: Annotated[
        Optional[str],  # noqa: UP045
        typer.Option("--target", "-t", help="Target model (same vocabulary as the draft)"),
    ] = None,
    corpus: Annotated[
        Optional[Path],  # noqa: UP045
        typer.Option("--corpus", "-c", exists=True, help="Target-language file"),
    ] = None,
    reference: Annotated[
        Optional[Path],  # noqa: UP045
        typer.Option("--reference", "-r", exists=True, help="Parallel reference-language file"),
    ] = None,
    max_texts: Annotated[
        Optional[int],  # noqa: UP045
        typer.Option(help="Use only the first N lines of each file"),
    ] = None,
    dtype: Annotated[
        Optional[str],  # noqa: UP045
        typer.Option(help="float32, bfloat16 or float16"),
    ] = None,
    device: Annotated[Optional[str], typer.Option(help="cpu or cuda")] = None,  # noqa: UP045
    mode: Annotated[
        Optional[str],  # noqa: UP045
        typer.Option(help="joint (both models loaded) or sequential (one at a time)"),
    ] = None,
    out: Annotated[
        Optional[Path],  # noqa: UP045
        typer.Option("--out", help="Where to save the JSON result"),
    ] = None,
    config: Annotated[
        Optional[Path],  # noqa: UP045
        typer.Option("--config", exists=True, help="YAML config with an acceptance section"),
    ] = None,
) -> None:
    """Speculative decoding acceptance of a draft/target pair, target language vs reference.

    Flags override values from --config. The speedup shown is modelled, not measured.
    """
    from zubaan.config import AcceptanceConfig

    cfg = _section(config, "acceptance")
    values = cfg.model_dump() if cfg else {}
    flags = {
        "draft": draft,
        "target": target,
        "corpus": corpus,
        "reference_corpus": reference,
        "max_texts": max_texts,
        "dtype": dtype,
        "device": device,
        "mode": mode,
        "output": out,
    }
    values.update({k: v for k, v in flags.items() if v is not None})
    if not {"draft", "target", "corpus"} <= values.keys():
        raise typer.BadParameter("give --draft, --target and --corpus, or a --config")
    cfg = AcceptanceConfig.model_validate(values)

    from zubaan.audit.acceptance import best_speedup
    from zubaan.audit.acceptance_lm import acceptance_report

    texts = {cfg.target_lang: read_lines(cfg.corpus, cfg.max_texts)}
    if cfg.reference_corpus is not None:
        texts[cfg.reference_lang] = read_lines(cfg.reference_corpus, cfg.max_texts)

    def progress(lang: str, n: int, total: int) -> None:
        if n == total or n % 10 == 0:
            console.print(f"  {lang}: {n}/{total} texts")

    result = acceptance_report(
        cfg.draft,
        cfg.target,
        texts,
        normalize_lang=cfg.target_lang if cfg.normalize else None,
        dtype=cfg.dtype,
        device=cfg.device,
        mode=cfg.mode,
        chunk_size=cfg.chunk_size,
        draft_cost=cfg.draft_cost,
        on_text=progress,
    )
    result["provenance"] = run_metadata(
        [cfg.draft, cfg.target], cfg.corpus.parent / "flores_meta.json"
    )
    write_json(result, cfg.output)

    table = Table(title=f"Acceptance: draft {cfg.draft}, target {cfg.target}")
    table.add_column("Language")
    for col in ("Texts", "Positions", "alpha", "95% CI", "Greedy match", "Best gamma", "Speedup"):
        table.add_column(col, justify="right")
    for lang, r in result["languages"].items():
        best = best_speedup(r["speedup"])
        table.add_row(
            lang,
            str(r["n_texts"]),
            str(r["n_positions"]),
            f"{r['alpha']:.3f}",
            f"[{r['alpha_ci_low']:.3f}, {r['alpha_ci_high']:.3f}]",
            f"{r['greedy_match']:.3f}",
            str(best["gamma"]),
            f"{best['speedup']:.2f}x",
        )
    console.print(table)
    cost = next(iter(result["languages"].values()))["draft_cost"]
    console.print(f"Best of 1 to 6 drafted tokens, c = {cost:.2f}; modelled, not measured.")
    console.print(f"Saved {cfg.output}")


@data_app.command("urdu")
def data_urdu(
    config: Annotated[
        Optional[Path],  # noqa: UP045
        typer.Option("--config", exists=True, help="YAML config with a urdu_corpus section"),
    ] = None,
    out: Annotated[
        Optional[Path],  # noqa: UP045
        typer.Option(help="Output directory (default data/urdu)"),
    ] = None,
    max_mb: Annotated[
        Optional[float],  # noqa: UP045
        typer.Option(help="Stop after this many MB of training text"),
    ] = None,
    parquet: Annotated[
        Optional[Path],  # noqa: UP045
        typer.Option(help="Use this Wikipedia parquet file instead of downloading one"),
    ] = None,
) -> None:
    """Build the Urdu training text for tokenizer extension from Urdu Wikipedia.

    Cleans and normalizes lines, removes anything that overlaps the FLORES sentences, and
    writes urdu_train.txt, urdu_heldout.txt and urdu_meta.json.
    """
    from zubaan.config import UrduCorpusConfig
    from zubaan.data.urdu import (
        WIKIPEDIA_FILE,
        WIKIPEDIA_LICENSE,
        WIKIPEDIA_REPO,
        build_corpus,
        download_wikipedia,
        iter_parquet_text,
    )

    cfg = (_section(config, "urdu_corpus") if config else None) or UrduCorpusConfig()
    out_dir = out or cfg.out_dir
    max_mb = max_mb if max_mb is not None else cfg.max_mb
    if not cfg.flores.exists():
        raise typer.BadParameter(f"{cfg.flores} not found; run 'zubaan data flores' first")
    path = parquet or download_wikipedia(cfg.raw_dir / "wikipedia_ur")
    meta = build_corpus(
        iter_parquet_text(path),
        out_dir,
        read_lines(cfg.flores),
        max_mb=max_mb,
        heldout_percent=cfg.heldout_percent,
        min_words=cfg.min_words,
        min_urdu_fraction=cfg.min_urdu_fraction,
        source={
            "name": WIKIPEDIA_REPO,
            "file": WIKIPEDIA_FILE,
            "license": WIKIPEDIA_LICENSE,
            "local_file": str(path),
        },
    )
    st = meta["stats"]
    console.print(
        f"Train: {st['train_lines']:,} lines, {st['train_bytes'] / 1e6:.1f} MB. "
        f"Held-out: {st['heldout_lines']:,} lines. "
        f"Dropped: {st['dropped_filter']:,} filtered, {st['dropped_duplicate']:,} duplicate, "
        f"{st['dropped_flores']:,} overlapping FLORES."
    )
    console.print(f"Saved {out_dir}")


def _extend_config(config: Path, base_model: str | None):
    cfg = _section(config, "extend")
    if base_model:
        cfg = cfg.model_copy(update={"base_model": base_model})
    return cfg


@extend_app.command("learn")
def extend_learn(
    config: Annotated[Path, typer.Option("--config", exists=True, help="YAML with extend:")],
    base_model: Annotated[
        Optional[str],  # noqa: UP045
        typer.Option(help="Override extend.base_model (HF repo id or local folder)"),
    ] = None,
) -> None:
    """Learn Urdu merges and write the extended tokenizer (no model weights are touched)."""
    from zubaan.extend.merge import assert_only_model_changed, extend_tokenizer_json, save_extended
    from zubaan.extend.sweep import learn_urdu_merges, save_merges
    from zubaan.extend.tokenizer_info import (
        find_local_files,
        inspect_tokenizer_json,
        load_tokenizer_json,
    )

    cfg = _extend_config(config, base_model)
    base = load_tokenizer_json(cfg.base_model)
    n_max = max([cfg.n_new_tokens, *cfg.sizes])
    lines = read_lines(cfg.corpus, cfg.max_lines)
    console.print(f"Learning up to {n_max:,} merges from {len(lines):,} lines...")
    merges = learn_urdu_merges(lines, base, n_max, cfg.min_count)
    save_merges(merges, cfg.merges_file)
    console.print(f"Learned {len(merges):,} merges, saved {cfg.merges_file}")
    chosen = merges[: cfg.n_new_tokens]
    ext = extend_tokenizer_json(base, chosen)
    assert_only_model_changed(base, ext)
    out = save_extended(ext, find_local_files(cfg.base_model), cfg.output_dir / "tokenizer")
    info = inspect_tokenizer_json(ext)
    console.print(
        f"Extended tokenizer: {len(chosen):,} new tokens, vocabulary {info.total_vocab_size:,}, "
        f"saved to {out}"
    )


@extend_app.command("quality")
def extend_quality(
    config: Annotated[Path, typer.Option("--config", exists=True, help="YAML with extend:")],
    base_model: Annotated[
        Optional[str],  # noqa: UP045
        typer.Option(help="Override extend.base_model: the model whose weights are extended"),
    ] = None,
    tokenizer_model: Annotated[
        Optional[str],  # noqa: UP045
        typer.Option(help="Override extend.tokenizer_model: where the merges were learned"),
    ] = None,
    out: Annotated[
        Optional[Path],  # noqa: UP045
        typer.Option(help="Results JSON (default extend.quality_output); finished variants resume"),
    ] = None,
) -> None:
    """Extend a model with each size and init method, then measure quality and cost.

    Needs the model weights (use a GPU or the 0.6B model). Nothing is trained: the new rows
    are only initialized, so expect quality to be worse than the base model.
    """
    from zubaan.audit.acceptance_lm import load_causal_lm
    from zubaan.bench.quality import load_belebele
    from zubaan.extend.quality_run import (
        check_weights_match_tokenizer,
        run_quality,
        select_bpc_texts,
        summary_rows,
    )
    from zubaan.extend.sweep import load_merges
    from zubaan.extend.tokenizer_info import load_tokenizer_json

    cfg = _extend_config(config, base_model)
    weights_model = cfg.base_model
    tok_model = tokenizer_model or cfg.tokenizer_model or weights_model
    if not cfg.merges_file.exists():
        raise typer.BadParameter(f"{cfg.merges_file} not found; run 'zubaan extend learn' first")
    base = load_tokenizer_json(tok_model)
    try:
        check_weights_match_tokenizer(load_tokenizer_json(weights_model), base)
    except ValueError as e:
        raise typer.BadParameter(str(e)) from e
    merges = load_merges(cfg.merges_file)
    if max(cfg.eval_sizes) > len(merges):
        raise typer.BadParameter(
            f"eval_sizes needs {max(cfg.eval_sizes)} merges, have {len(merges)}"
        )
    sets = {
        "urdu_heldout": select_bpc_texts(
            read_lines(cfg.heldout) if cfg.heldout else [], cfg.quality_texts, cfg.max_chars, True
        ),
        "urdu_flores": select_bpc_texts(
            read_lines(cfg.eval_urdu), cfg.quality_texts, cfg.max_chars, True
        ),
        "english_flores": select_bpc_texts(
            read_lines(cfg.eval_english), cfg.quality_texts, cfg.max_chars, False
        ),
    }
    sets = {k: v for k, v in sets.items() if v}
    belebele = {
        lang: load_belebele(lang, cfg.belebele_questions) for lang in ("urd_Arab", "eng_Latn")
    }
    output = out or cfg.quality_output
    meta = run_metadata([weights_model], cfg.corpus.parent / "urdu_meta.json")
    meta["settings"] = {
        "dtype": cfg.dtype,
        "device": cfg.device,
        "seed": cfg.seed,
        "n_texts": {k: len(v) for k, v in sets.items()},
        "belebele_questions": cfg.belebele_questions,
        "tokenizer_model": tok_model,
        "init_methods": cfg.init_methods,
        "eval_sizes": cfg.eval_sizes,
    }
    results = run_quality(
        weights_model,
        lambda: load_causal_lm(weights_model, cfg.dtype, cfg.device),
        base,
        merges,
        cfg.eval_sizes,
        cfg.init_methods,
        sets,
        belebele,
        output,
        seed=cfg.seed,
        focus_lines=read_lines(cfg.corpus, cfg.focus_lines),
        focus_dim=cfg.focus_dim,
        batch_size=cfg.batch_size,
        device=cfg.device,
        meta=meta,
        log=console.print,
        max_batch_tokens=cfg.max_batch_tokens,
    )
    table = Table(title=f"Quality next to cost: {weights_model} (new rows untrained)")
    cols = ["Variant", "Vocab", "Tok/char Urdu", "BPC Urdu held-out", "BPC Urdu FLORES"]
    cols += ["BPC English", "Belebele urd", "Belebele eng"]
    for col in cols:
        table.add_column(col, justify="right")
    for r in summary_rows(results):
        table.add_row(
            r["variant"],
            f"{r['vocab_size']:,}",
            f"{r.get('tokens_per_char_urdu_heldout', float('nan')):.3f}",
            f"{r.get('bpc_urdu_heldout', float('nan')):.3f}",
            f"{r.get('bpc_urdu_flores', float('nan')):.3f}",
            f"{r.get('bpc_english_flores', float('nan')):.3f}",
            f"{r['belebele_urd_Arab_acc_char']:.3f}",
            f"{r['belebele_eng_Latn_acc_char']:.3f}",
        )
    console.print(table)
    console.print("Lower bits per character is better; Belebele chance is 0.25.")
    console.print(f"Saved {output}")


@extend_app.command("run")
def extend_run(
    config: Annotated[Path, typer.Option("--config", exists=True, help="YAML with extend:")],
    base_model: Annotated[
        Optional[str],  # noqa: UP045
        typer.Option(help="Override extend.base_model (HF repo id or local folder)"),
    ] = None,
    n_new: Annotated[
        Optional[int],  # noqa: UP045
        typer.Option(help="Override extend.n_new_tokens"),
    ] = None,
    method: Annotated[
        Optional[str],  # noqa: UP045
        typer.Option(help="Override extend.init_method: mean | focus | random"),
    ] = None,
    relearn: Annotated[
        bool, typer.Option(help="Learn merges again even if the file exists")
    ] = False,
    json_out: Annotated[bool, typer.Option("--json", help="Print the summary as JSON")] = False,
) -> None:
    """Learn merges, extend the model, save it to <output_dir>/model and verify it.

    Needs the model weights. The new rows are initialized, not trained (stage 3 trains them).
    """
    from zubaan.extend.pipeline import run_extend

    cfg = _extend_config(config, base_model)
    summary = run_extend(
        cfg, n_new, method, relearn, log=(lambda s: None) if json_out else console.print
    )
    if json_out:
        typer.echo(json.dumps(summary, indent=2, ensure_ascii=False))
        return
    cost = summary["cost"]
    console.print(
        f"Urdu cost {cost['cost_multiplier']:.2f}x English "
        f"({cost['urdu_tokens_per_word']:.2f} tokens per word). "
        f"English changed: {cost['english_different']}. Round-trip failures: "
        f"{cost['urdu_roundtrip_failures']}."
    )
    console.print(f"Saved the extended model to {summary['model_dir']}")
    console.print("The new rows are untrained: expect worse Urdu quality until stage 3.")


@extend_app.command("sweep")
def extend_sweep(
    config: Annotated[Path, typer.Option("--config", exists=True, help="YAML with extend:")],
    base_model: Annotated[
        Optional[str],  # noqa: UP045
        typer.Option(help="Override extend.base_model (HF repo id or local folder)"),
    ] = None,
    out: Annotated[Path, typer.Option(help="Save the JSON result here")] = Path(
        "reports/extend_sweep.json"
    ),
) -> None:
    """Measure token cost for each vocabulary size in extend.sizes, with English as control."""
    from zubaan.extend.sweep import load_merges, sweep_vocab_sizes
    from zubaan.extend.tokenizer_info import load_tokenizer_json

    cfg = _extend_config(config, base_model)
    if not cfg.merges_file.exists():
        raise typer.BadParameter(f"{cfg.merges_file} not found; run 'zubaan extend learn' first")
    base = load_tokenizer_json(cfg.base_model)
    merges = load_merges(cfg.merges_file)
    heldout = read_lines(cfg.heldout) if cfg.heldout else []
    rows = sweep_vocab_sizes(
        base, merges, cfg.sizes, read_lines(cfg.eval_urdu), read_lines(cfg.eval_english), heldout
    )
    result = {"base_model": cfg.base_model, "n_merges_learned": len(merges), "rows": rows}
    result["provenance"] = run_metadata([cfg.base_model], cfg.corpus.parent / "urdu_meta.json")
    write_json(result, out)

    table = Table(title=f"Urdu token cost by number of new tokens ({cfg.base_model})")
    for col in (
        "New tokens",
        "Vocab",
        "Urdu tok/word",
        "Cost x",
        "Raw cost x",
        "Held-out tok/word",
        "English changed",
        "Round-trip fails",
    ):
        table.add_column(col, justify="right")
    for r in rows:
        table.add_row(
            f"{r['n_new_tokens']:,}",
            f"{r['vocab_size']:,}",
            f"{r['urdu_tokens_per_word']:.2f}",
            f"{r['cost_multiplier']:.2f}x",
            f"{r['raw_cost_multiplier']:.2f}x",
            f"{r.get('heldout_tokens_per_word', float('nan')):.2f}",
            str(r["english_different"]),
            str(r["urdu_roundtrip_failures"]),
        )
    console.print(table)
    console.print("Quality is not measured here: new embeddings are untrained (see stage 3).")
    console.print(f"Saved {out}")


def _adapt_config(config: Path, model_path: Path | None):
    cfg = _section(config, "adapt")
    if model_path:
        cfg = cfg.model_copy(update={"model_path": model_path})
    return cfg


@adapt_app.command("prepare")
def adapt_prepare(
    config: Annotated[Path, typer.Option("--config", exists=True, help="YAML with adapt:")],
    model_path: Annotated[
        Optional[Path],  # noqa: UP045
        typer.Option(help="Override adapt.model_path (the extended model folder)"),
    ] = None,
    json_out: Annotated[bool, typer.Option("--json", help="Print the metadata as JSON")] = False,
) -> None:
    """Tokenize Urdu and English replay text into training blocks (no model weights needed).

    Training lines that overlap FLORES or Belebele text are dropped and counted.
    """
    from zubaan.adapt.pipeline import run_prepare

    cfg = _adapt_config(config, model_path)
    meta = run_prepare(cfg, log=(lambda s: None) if json_out else console.print)
    if json_out:
        typer.echo(json.dumps(meta, indent=2, ensure_ascii=False))
        return
    console.print(f"Saved blocks to {cfg.data_dir}")


@adapt_app.command("run")
def adapt_run(
    config: Annotated[Path, typer.Option("--config", exists=True, help="YAML with adapt:")],
    model_path: Annotated[
        Optional[Path],  # noqa: UP045
        typer.Option(help="Override adapt.model_path (the extended model folder)"),
    ] = None,
    steps: Annotated[
        Optional[int],  # noqa: UP045
        typer.Option(help="Stop after this many steps (smoke runs); default is the full schedule"),
    ] = None,
    stop_at: Annotated[
        Optional[int],  # noqa: UP045
        typer.Option(help="Stop when this global step is reached (a later run can continue)"),
    ] = None,
    json_out: Annotated[bool, typer.Option("--json", help="Print the summary as JSON")] = False,
) -> None:
    """Train the new embedding rows, then LoRA plus the new rows. Needs a GPU for real models.

    With adapt.control the base model trains LoRA only (the compute-matched control run).
    Writes checkpoints under <output_dir>/checkpoints, resumes from the newest one (or from the
    private Hub repo adapt.hub_repo) and logs to <output_dir>/train_log.jsonl.
    """
    from zubaan.adapt.pipeline import run_adapt

    cfg = _adapt_config(config, model_path)
    summary = run_adapt(
        cfg,
        log=(lambda s: None) if json_out else console.print,
        max_steps=steps,
        stop_at_step=stop_at,
    )
    if json_out:
        typer.echo(json.dumps(summary, indent=2, ensure_ascii=False))
        return
    done = "finished" if summary["finished"] else "stopped early"
    console.print(f"Training {done} at step {summary['final_step']} of {summary['total_steps']}.")
    for e in summary["evals"][-1:]:
        console.print(f"Last eval: {e['eval']}")
    console.print("Next: merge LoRA and the new rows, verify, then evaluate (stage 3, chunks 6-7).")


@adapt_app.command("merge")
def adapt_merge(
    config: Annotated[Path, typer.Option("--config", exists=True, help="YAML with adapt:")],
    model_path: Annotated[
        Optional[Path],  # noqa: UP045
        typer.Option(help="Override adapt.model_path (the extended model folder)"),
    ] = None,
    checkpoint: Annotated[
        Optional[Path],  # noqa: UP045
        typer.Option(help="Checkpoint folder (default: newest in <output_dir>/checkpoints)"),
    ] = None,
    out: Annotated[
        Optional[Path],  # noqa: UP045
        typer.Option(help="Where to save the merged model (default <output_dir>/model)"),
    ] = None,
    milestone: Annotated[
        Optional[int],  # noqa: UP045
        typer.Option(help="Merge the Hub milestone saved at this step instead of the newest"),
    ] = None,
    json_out: Annotated[bool, typer.Option("--json", help="Print the summary as JSON")] = False,
) -> None:
    """Merge LoRA and the trained rows into a plain model, save it and verify the folder."""
    from zubaan.adapt.finalize import merge_and_save

    cfg = _adapt_config(config, model_path)
    if milestone is not None:
        if checkpoint is not None or not cfg.hub_repo:
            raise typer.BadParameter("--milestone needs hub_repo in the config and no --checkpoint")
        from zubaan.adapt.pipeline import download_hub_checkpoint, milestone_path

        checkpoint = download_hub_checkpoint(
            cfg.hub_repo, cfg.output_dir / "checkpoints", milestone_path(milestone)
        )
        if checkpoint is None:
            raise typer.BadParameter(f"no milestone at step {milestone} in {cfg.hub_repo}")
    summary = merge_and_save(
        cfg, checkpoint, out, log=(lambda s: None) if json_out else console.print
    )
    if json_out:
        typer.echo(json.dumps(summary, indent=2, ensure_ascii=False))
        return
    console.print(f"Saved and verified: {summary['model_dir']} (trained at step {summary['step']})")
    console.print("Next: evaluate it, then 'zubaan export all' with model_dir pointing at it.")


@adapt_app.command("eval")
def adapt_eval(
    config: Annotated[Path, typer.Option("--config", exists=True, help="YAML with evaluate:")],
    out: Annotated[
        Optional[Path],  # noqa: UP045
        typer.Option(help="Results JSON (default evaluate.output); finished models resume"),
    ] = None,
    json_out: Annotated[bool, typer.Option("--json", help="Print the checks as JSON")] = False,
) -> None:
    """Score saved model folders (base, extended, adapted) on the same texts.

    Bits per character on held-out Urdu, FLORES Urdu and FLORES English, plus Belebele in both
    languages, and the stage 3 success checks when untrained and adapted are named.
    """
    from zubaan.adapt.evaluate import comparison_rows, run_model_eval, success_checks
    from zubaan.audit.acceptance_lm import load_causal_lm
    from zubaan.bench.quality import load_belebele
    from zubaan.extend.quality_run import select_bpc_texts

    cfg = _section(config, "evaluate")
    sets = {
        "urdu_heldout": select_bpc_texts(
            read_lines(cfg.heldout), cfg.quality_texts, cfg.max_chars, True
        ),
        "urdu_flores": select_bpc_texts(
            read_lines(cfg.eval_urdu), cfg.quality_texts, cfg.max_chars, True
        ),
        "english_flores": select_bpc_texts(
            read_lines(cfg.eval_english), cfg.quality_texts, cfg.max_chars, False
        ),
    }
    sets = {k: v for k, v in sets.items() if v}
    belebele = {
        lang: load_belebele(lang, cfg.belebele_questions) for lang in ("urd_Arab", "eng_Latn")
    }
    output = out or cfg.output
    meta = run_metadata(list(cfg.models.values()))
    meta["settings"] = {
        "dtype": cfg.dtype,
        "device": cfg.device,
        "n_texts": {k: len(v) for k, v in sets.items()},
        "belebele_questions": cfg.belebele_questions,
        "models": cfg.models,
    }
    results = run_model_eval(
        cfg.models,
        lambda path: load_causal_lm(path, cfg.dtype, cfg.device),
        sets,
        belebele,
        output,
        device=cfg.device,
        batch_size=cfg.batch_size,
        max_batch_tokens=cfg.max_batch_tokens,
        meta=meta,
        log=(lambda s: None) if json_out else console.print,
    )
    checks = []
    if cfg.untrained and cfg.adapted:
        reference = next(iter(cfg.models))
        checks = success_checks(
            results, reference, cfg.untrained, cfg.adapted, cfg.english_tolerance
        )
    if json_out:
        typer.echo(json.dumps({"checks": checks, "output": str(output)}, ensure_ascii=False))
        return
    table = Table(title="Stage 3 quality next to cost")
    cols = ["Model", "Vocab", "Tok/char Urdu", "BPC Urdu held-out", "BPC Urdu FLORES"]
    cols += ["BPC English", "Belebele urd", "Belebele eng"]
    for col in cols:
        table.add_column(col, justify="right")
    nan = float("nan")
    for r in comparison_rows(results):
        table.add_row(
            r["model"],
            f"{r['vocab_size']:,}",
            f"{r.get('tokens_per_char_urdu_heldout', nan):.3f}",
            f"{r.get('bpc_urdu_heldout', nan):.3f}",
            f"{r.get('bpc_urdu_flores', nan):.3f}",
            f"{r.get('bpc_english_flores', nan):.3f}",
            f"{r.get('belebele_urd_Arab_acc_char', nan):.3f}",
            f"{r.get('belebele_eng_Latn_acc_char', nan):.3f}",
        )
    console.print(table)
    for c in checks:
        console.print(f"{'PASS' if c['passed'] else 'FAIL'}  {c['name']}: {c['detail']}")
    console.print("Lower bits per character is better; Belebele chance is 0.25.")
    console.print(f"Saved {output}")


@app.command()
def draft() -> None:
    """Build drafters that share the extended vocabulary."""
    _planned(5, "draft model distillation and n-gram drafter")


def _export_config(config: Path, model_dir: Path | None):
    cfg = _section(config, "export")
    if model_dir:
        cfg = cfg.model_copy(update={"model_dir": model_dir})
    return cfg


def _run_export_steps(config: Path, model_dir: Path | None, steps: tuple[str, ...], json_out: bool):
    from zubaan.export.pipeline import run_export

    cfg = _export_config(config, model_dir)
    try:
        summary = run_export(cfg, steps, log=(lambda s: None) if json_out else console.print)
    except (FileNotFoundError, ValueError) as e:
        raise typer.BadParameter(str(e)) from e
    if json_out:
        typer.echo(json.dumps(summary, indent=2, ensure_ascii=False))
    else:
        console.print(f"Saved {cfg.output_dir / 'export_summary.json'}")


_EXPORT_CONFIG = Annotated[Path, typer.Option("--config", exists=True, help="YAML with export:")]
_EXPORT_MODEL = Annotated[
    Optional[Path],  # noqa: UP045
    typer.Option(help="Override export.model_dir"),
]
_EXPORT_JSON = Annotated[bool, typer.Option("--json", help="Print the summary as JSON")]


@export_app.command("gguf")
def export_gguf(
    config: _EXPORT_CONFIG, model_dir: _EXPORT_MODEL = None, json_out: _EXPORT_JSON = False
) -> None:
    """Convert to GGUF, prove tokenizer parity with Hugging Face, then quantize (Urdu imatrix)."""
    _run_export_steps(config, model_dir, ("gguf",), json_out)


@export_app.command("ollama")
def export_ollama(
    config: _EXPORT_CONFIG, model_dir: _EXPORT_MODEL = None, json_out: _EXPORT_JSON = False
) -> None:
    """Write an Ollama Modelfile for the exported GGUF (run 'export gguf' first)."""
    _run_export_steps(config, model_dir, ("ollama",), json_out)


@export_app.command("vllm")
def export_vllm(
    config: _EXPORT_CONFIG, model_dir: _EXPORT_MODEL = None, json_out: _EXPORT_JSON = False
) -> None:
    """Write a vLLM config and launch script (dtype half, for T4 and P100)."""
    _run_export_steps(config, model_dir, ("vllm",), json_out)


@export_app.command("all")
def export_all(
    config: _EXPORT_CONFIG, model_dir: _EXPORT_MODEL = None, json_out: _EXPORT_JSON = False
) -> None:
    """GGUF with parity check, Ollama Modelfile and vLLM config in one go."""
    _run_export_steps(config, model_dir, ("gguf", "ollama", "vllm"), json_out)


@export_app.command("check")
def export_check(
    config: _EXPORT_CONFIG, model_dir: _EXPORT_MODEL = None, json_out: _EXPORT_JSON = False
) -> None:
    """Urdu generation samples on every GGUF and quantization cost (KL divergence vs f16)."""
    from zubaan.export.check import run_check

    # The samples are Urdu; a Windows console with a legacy code page cannot print them. Show "?"
    # instead of stopping the run (export_check.json always holds the real text).
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    cfg = _export_config(config, model_dir)
    try:
        result = run_check(cfg, log=(lambda s: None) if json_out else console.print)
    except (FileNotFoundError, ValueError) as e:
        raise typer.BadParameter(str(e)) from e
    if json_out:
        typer.echo(json.dumps(result, indent=2, ensure_ascii=False))
        return
    for lang, res in result["quant_cost"].items():
        for q, k in res["quants"].items():
            console.print(
                f"{lang} {q}: mean KLD {k['mean_kld']['value']:.4f}, "
                f"same top token {k['same_top_p_percent']['value']:.1f}%"
            )
    console.print(f"Saved {cfg.output_dir / 'export_check.json'}")


@tune_app.command("check")
def tune_check(
    config: Annotated[Path, typer.Option("--config", exists=True, help="YAML with chat_check:")],
    label: Annotated[
        Optional[str],  # noqa: UP045
        typer.Option(help="Override chat_check.label"),
    ] = None,
    gguf: Annotated[
        Optional[Path],  # noqa: UP045
        typer.Option(help="Override chat_check.gguf (starts llama-server on it)"),
    ] = None,
    output: Annotated[
        Optional[Path],  # noqa: UP045
        typer.Option(help="Override chat_check.output"),
    ] = None,
    max_prompts: Annotated[
        Optional[int],  # noqa: UP045
        typer.Option(help="Use only the first N prompts (smoke run)"),
    ] = None,
    json_out: Annotated[bool, typer.Option("--json", help="Print the summary as JSON")] = False,
) -> None:
    """Chat-behavior check: answer rate with thinking on and off, reply language, loops."""
    from zubaan.tune.chat_check import run_from_config

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    cfg = _section(config, "chat_check")
    updates = {
        k: v
        for k, v in {
            "label": label,
            "gguf": gguf,
            "output": output,
            "max_prompts": max_prompts,
        }.items()
        if v is not None
    }
    cfg = cfg.model_copy(update=updates)
    try:
        result = run_from_config(cfg, log=(lambda s: None) if json_out else console.print)
    except (FileNotFoundError, ValueError, RuntimeError) as e:
        raise typer.BadParameter(str(e)) from e
    if json_out:
        typer.echo(json.dumps(result["summary"], indent=2, ensure_ascii=False))
        return
    for mode, groups in result["summary"].items():
        for lang, g in groups.items():
            console.print(
                f"{cfg.label} {mode} {lang}: answered {g['answered']}/{g['n']}, "
                f"right language {g['right_language']}/{g['language_due']}, "
                f"hit limit {g['hit_limit']}, loops {g['loops']}"
            )
    console.print(f"Saved {cfg.output} and {cfg.output.with_suffix('.md')}")


def _tune_config(config: Path, model_path: Path | None, train_file: Path | None):
    cfg = _section(config, "tune")
    updates = {k: v for k, v in {"model_path": model_path, "train_file": train_file}.items() if v}
    return cfg.model_copy(update=updates)


@tune_app.command("run")
def tune_run(
    config: Annotated[Path, typer.Option("--config", exists=True, help="YAML with tune:")],
    model_path: Annotated[
        Optional[Path],  # noqa: UP045
        typer.Option(help="Override tune.model_path (the merged Stage 3 model)"),
    ] = None,
    train_file: Annotated[
        Optional[Path],  # noqa: UP045
        typer.Option(help="Override tune.train_file (filtered conversations JSONL)"),
    ] = None,
    steps: Annotated[
        Optional[int],  # noqa: UP045
        typer.Option(help="Stop after this many steps (smoke runs); default is the full schedule"),
    ] = None,
    stop_at: Annotated[
        Optional[int],  # noqa: UP045
        typer.Option(help="Stop when this global step is reached (a later run can continue)"),
    ] = None,
    json_out: Annotated[bool, typer.Option("--json", help="Print the summary as JSON")] = False,
) -> None:
    """Chat SFT: LoRA on the Stage 3 model, loss on the assistant's reply only. Needs a GPU.

    Checkpoints go to <output_dir>/checkpoints (and the private Hub repo tune.hub_repo if set);
    a run resumes from the newest one. The log is <output_dir>/train_log.jsonl.
    """
    from zubaan.tune.sft import run_tune

    cfg = _tune_config(config, model_path, train_file)
    summary = run_tune(
        cfg,
        log=(lambda s: None) if json_out else console.print,
        max_steps=steps,
        stop_at_step=stop_at,
    )
    if json_out:
        typer.echo(json.dumps(summary, indent=2, ensure_ascii=False))
        return
    done = "finished" if summary["finished"] else "stopped early"
    console.print(f"Training {done} at step {summary['final_step']} of {summary['total_steps']}.")
    for e in summary["evals"][-1:]:
        console.print(f"Last eval: {e['eval']}")


@tune_app.command("merge")
def tune_merge(
    config: Annotated[Path, typer.Option("--config", exists=True, help="YAML with tune:")],
    model_path: Annotated[
        Optional[Path],  # noqa: UP045
        typer.Option(help="Override tune.model_path (the merged Stage 3 model)"),
    ] = None,
    checkpoint: Annotated[
        Optional[Path],  # noqa: UP045
        typer.Option(help="Checkpoint folder (default: newest in <output_dir>/checkpoints)"),
    ] = None,
    out: Annotated[
        Optional[Path],  # noqa: UP045
        typer.Option(help="Where to save the merged model (default <output_dir>/model)"),
    ] = None,
    json_out: Annotated[bool, typer.Option("--json", help="Print the summary as JSON")] = False,
) -> None:
    """Fold the chat LoRA into the Stage 3 model, check the logits and save the folder."""
    from zubaan.tune.sft import merge_tuned

    cfg = _tune_config(config, model_path, None)
    summary = merge_tuned(cfg, checkpoint, out, log=(lambda s: None) if json_out else console.print)
    if json_out:
        typer.echo(json.dumps(summary, indent=2, ensure_ascii=False))
        return
    console.print(f"Saved: {summary['model_dir']} (trained to step {summary['step']})")
    console.print("Next: the chat check and the Stage 3 quality checks on it (stage 4, chunk 7).")


@app.command()
def bench() -> None:
    """Benchmark running servers with tokenizer-fair metrics."""
    _planned(6, "end-to-end benchmarks")


if __name__ == "__main__":  # pragma: no cover
    app()
