"""Command-line interface. Kept thin: all logic lives in the library modules."""

from __future__ import annotations

import json
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


def _planned(stage: int, what: str) -> None:
    console.print(f"[yellow]Not implemented yet.[/] Planned for stage {stage}: {what}")
    raise typer.Exit(code=1)


@audit_app.command("acceptance")
def audit_acceptance() -> None:
    """Speculative decoding acceptance rate of a draft/target pair on a corpus."""
    _planned(1, "acceptance rates from model distributions")


@app.command()
def extend() -> None:
    """Add target-language merges to a tokenizer and initialize the new embeddings."""
    _planned(2, "tokenizer extension")


@app.command()
def adapt() -> None:
    """Continued pretraining so the model learns its new tokens."""
    _planned(3, "continued pretraining recipe")


@app.command()
def draft() -> None:
    """Build drafters that share the extended vocabulary."""
    _planned(4, "draft model distillation and n-gram drafter")


@app.command()
def export() -> None:
    """Export to GGUF, vLLM and Ollama with tokenizer parity checks."""
    _planned(2, "GGUF export with parity check")


@app.command()
def bench() -> None:
    """Benchmark running servers with tokenizer-fair metrics."""
    _planned(5, "end-to-end benchmarks")


if __name__ == "__main__":  # pragma: no cover
    app()
