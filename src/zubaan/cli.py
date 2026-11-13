"""Command-line interface. Kept thin: all logic lives in the library modules."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Optional

import typer
from rich.console import Console
from rich.table import Table

from zubaan import __version__
from zubaan.audit.fertility import FertilityResult, compute_fertility, cost_multiplier
from zubaan.io import read_lines

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
    tokenizer: Annotated[str, typer.Option("--tokenizer", "-t", help="HF repo id or local path")],
    corpus: Annotated[
        Path, typer.Option("--corpus", "-c", exists=True, help="Target-language file")
    ],
    reference: Annotated[
        Optional[Path],  # noqa: UP045 - Typer needs Optional on Python 3.10
        typer.Option("--reference", "-r", exists=True, help="Parallel reference-language file"),
    ] = None,
    max_texts: Annotated[
        Optional[int],  # noqa: UP045
        typer.Option(help="Use only the first N lines"),
    ] = None,
    no_normalize: Annotated[bool, typer.Option(help="Skip Urdu normalization")] = False,
    as_json: Annotated[bool, typer.Option("--json", help="Print machine-readable JSON")] = False,
) -> None:
    """Tokens per word and characters per token, plus cost versus a reference language."""
    tok = load_tokenizer(tokenizer)
    target = compute_fertility(tok, read_lines(corpus, max_texts), normalize=not no_normalize)
    ref: FertilityResult | None = None
    if reference is not None:
        ref = compute_fertility(tok, read_lines(reference, max_texts), normalize=False)

    if as_json:
        out = {"tokenizer": tokenizer, "target": target.to_dict()}
        if ref is not None:
            out["reference"] = ref.to_dict()
            out["cost_multiplier"] = cost_multiplier(target, ref)
        console.print_json(json.dumps(out))
        return

    table = Table(title=f"Fertility: {tokenizer}")
    table.add_column("Corpus")
    for col in ("Texts", "Words", "Tokens", "Tokens/word", "Chars/token"):
        table.add_column(col, justify="right")
    rows = [("target", target)] + ([("reference", ref)] if ref is not None else [])
    for name, r in rows:
        table.add_row(
            name,
            str(r.n_texts),
            str(r.n_words),
            str(r.n_tokens),
            f"{r.tokens_per_word:.2f}",
            f"{r.chars_per_token:.2f}",
        )
    console.print(table)
    if ref is not None:
        console.print(f"Cost multiplier vs reference: [bold]{cost_multiplier(target, ref):.2f}x[/]")


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
