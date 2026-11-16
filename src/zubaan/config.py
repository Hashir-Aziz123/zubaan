"""Run configuration. Every command can be driven from a YAML file so runs are reproducible."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field


class AuditConfig(BaseModel):
    tokenizer: str = Field(description="Hugging Face repo id or local path")
    corpus: Path = Field(description="UTF-8 text file, one text per line, target language")
    reference_corpus: Path | None = Field(
        default=None, description="Parallel corpus in the reference language (e.g. English)"
    )
    normalize: bool = True
    max_texts: int | None = None


class CompareConfig(BaseModel):
    models: list[str] = Field(description="Tokenizers to compare (HF repo ids or local paths)")
    corpus: Path = Field(description="Target-language file, one text per line")
    reference_corpus: Path = Field(description="Parallel reference-language file")
    normalize: bool = True
    max_texts: int | None = None
    output_dir: Path = Path("reports")


class AcceptanceConfig(BaseModel):
    draft: str = Field(description="Draft (small) model repo id or path")
    target: str = Field(description="Target (verifier) model; must share the draft's vocabulary")
    corpus: Path = Field(description="Target-language file, one text per line")
    reference_corpus: Path | None = Field(default=None, description="Parallel reference file")
    target_lang: str = "urd"
    reference_lang: str = "eng"
    normalize: bool = True
    max_texts: int | None = 50
    dtype: str = Field(default="bfloat16", description="float32 | bfloat16 | float16")
    device: str = "cpu"
    mode: str = Field(default="joint", description="joint (both loaded) | sequential")
    chunk_size: int = 16
    draft_cost: float | None = Field(
        default=None, description="Draft step time / target step time; None = parameter ratio"
    )
    output: Path = Path("reports/acceptance.json")


class UrduCorpusConfig(BaseModel):
    """Build the Urdu training text for tokenizer extension (zubaan data urdu)."""

    raw_dir: Path = Field(default=Path("data/raw"), description="Where the source is downloaded")
    out_dir: Path = Path("data/urdu")
    flores: Path = Field(
        default=Path("data/flores_urd.txt"),
        description="Evaluation sentences that must not appear in the training text",
    )
    max_mb: float | None = Field(
        default=100.0, description="Stop after this many MB of training text (None = all)"
    )
    heldout_percent: float = 1.0
    min_words: int = 5
    min_urdu_fraction: float = 0.7


class ExtendConfig(BaseModel):
    base_model: str
    corpus: Path
    heldout: Path | None = Field(default=None, description="Held-out Urdu lines for measuring")
    eval_urdu: Path = Path("data/flores_urd.txt")
    eval_english: Path = Path("data/flores_eng.txt")
    n_new_tokens: int = 10_000
    sizes: list[int] = Field(
        default=[5_000, 10_000, 20_000, 32_000], description="Vocabulary sizes to compare"
    )
    max_lines: int | None = Field(default=None, description="Use only the first N training lines")
    min_count: int = Field(default=2, description="A merge must save at least this many tokens")
    merges_file: Path = Path("outputs/extend/merges.json")
    # Quality evaluation (zubaan extend quality). Weights come from base_model; the tokenizer
    # the merges were learned on comes from tokenizer_model (default: base_model) and must be
    # identical to the weights' own tokenizer.
    tokenizer_model: str | None = None
    eval_sizes: list[int] = Field(default=[10_000, 20_000], description="New-token counts to test")
    init_methods: list[str] = Field(default=["mean", "random", "focus"])
    dtype: str = Field(default="float32", description="float32 | bfloat16 | float16")
    device: str = "cpu"
    quality_texts: int | None = Field(default=300, description="Texts per BPC set (None = all)")
    max_chars: int = Field(default=400, description="Skip BPC texts longer than this")
    belebele_questions: int | None = Field(default=100, description="Per language (None = 900)")
    batch_size: int = 2
    max_batch_tokens: int = Field(
        default=2048, description="Cap on padded tokens per forward pass (bounds GPU memory)"
    )
    focus_lines: int = Field(default=30_000, description="Corpus lines used for FOCUS vectors")
    focus_dim: int = 100
    seed: int = 0
    quality_output: Path = Path("reports/extend_quality.json")
    init_method: str = Field(default="mean", description="mean | focus | random")
    output_dir: Path = Path("outputs/extended")


class AdaptConfig(BaseModel):
    model_path: Path
    corpus: str = Field(description="Dataset name or path for continued pretraining")
    english_replay_ratio: float = 0.1
    lora_rank: int = 16
    learning_rate: float = 2e-4
    max_seq_length: int = 2048
    output_dir: Path = Path("outputs/adapted")


class ProjectConfig(BaseModel):
    """Top-level config file: any section may be omitted."""

    audit: AuditConfig | None = None
    compare: CompareConfig | None = None
    acceptance: AcceptanceConfig | None = None
    urdu_corpus: UrduCorpusConfig | None = None
    extend: ExtendConfig | None = None
    adapt: AdaptConfig | None = None


def load_config(path: str | Path) -> ProjectConfig:
    with Path(path).open(encoding="utf-8") as f:
        return ProjectConfig.model_validate(yaml.safe_load(f) or {})
