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


class ExtendConfig(BaseModel):
    base_model: str
    corpus: Path
    n_new_tokens: int = 10_000
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
    extend: ExtendConfig | None = None
    adapt: AdaptConfig | None = None


def load_config(path: str | Path) -> ProjectConfig:
    with Path(path).open(encoding="utf-8") as f:
        return ProjectConfig.model_validate(yaml.safe_load(f) or {})
