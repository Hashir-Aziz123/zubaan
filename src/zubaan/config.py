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
    output_dir: Path = Field(
        default=Path("outputs/extend"),
        description="Tokenizer goes to <output_dir>/tokenizer, the model to <output_dir>/model",
    )
    model_dtype: str = Field(
        default="bfloat16", description="dtype of the saved model (Qwen3's own dtype)"
    )


class ExportConfig(BaseModel):
    """Export an extended model: GGUF (with a tokenizer parity check), Ollama, vLLM."""

    model_dir: Path = Field(
        default=Path("outputs/extend/model"), description="Folder written by 'zubaan extend build'"
    )
    base_model: str = Field(
        default="Qwen/Qwen3-1.7B", description="The model that was extended (tokenizer reference)"
    )
    output_dir: Path = Path("outputs/export")
    name: str = Field(default="qwen3-urdu", description="Prefix of the exported file names")
    llama_cpp_dir: Path = Field(
        default=Path("tools/llama.cpp-b12477"), description="llama.cpp checkout (converter script)"
    )
    llama_cpp_bin: Path | None = Field(
        default=None, description="Folder with llama-quantize and friends (default: <dir>/bin)"
    )
    python: str | None = Field(default=None, description="Python used to run the converter")
    outtype: str = "f16"
    quants: list[str] = Field(default=["Q4_K_M", "Q8_0"])
    calibration: Path = Field(
        default=Path("data/urdu/urdu_train.txt"),
        description="Urdu text for the importance matrix (never generic English)",
    )
    calibration_chars: int = 200_000
    imatrix_chunks: int = 40
    imatrix_ctx: int = 512
    threads: int | None = None
    parity_urdu: Path = Path("data/flores_urd.txt")
    parity_english: Path = Path("data/flores_eng.txt")
    max_parity_texts: int | None = Field(default=None, description="Per language (None = all)")
    parity_workers: int = 4
    max_model_len: int = 4096
    force: bool = Field(default=False, description="Rebuild files that already exist")


class AdaptConfig(BaseModel):
    """Stage 3: continued pretraining of an extended model (zubaan adapt prepare | run)."""

    # Inputs
    model_path: Path = Field(
        default=Path("outputs/extend/model"), description="Folder written by 'zubaan extend run'"
    )
    urdu_corpus: Path = Path("data/urdu/urdu_train.txt")
    heldout: Path = Field(
        default=Path("data/urdu/urdu_heldout.txt"), description="Urdu lines for the eval curve"
    )
    english_corpus: Path = Field(
        default=Path("data/english/english_replay.txt"),
        description="English replay text; built from English Wikipedia if missing",
    )
    english_mb: float = Field(default=12.0, description="Size of the English replay text")
    guard_urdu: list[Path] = Field(
        default=[Path("data/flores_urd.txt")],
        description="Urdu evaluation texts no training line may copy (add the FLORES dev split)",
    )
    guard_english: list[Path] = [Path("data/flores_eng.txt")]
    guard_belebele: bool = Field(
        default=True, description="Also guard against Belebele passages (needs the datasets pkg)"
    )
    drop_overlaps: bool = Field(
        default=True, description="Drop training lines that overlap evaluation text (else raise)"
    )
    data_dir: Path = Field(default=Path("data/adapt"), description="Token blocks go here")
    block_size: int = 1024
    english_replay_ratio: float = Field(default=0.1, description="Share of English blocks")
    # Training
    output_dir: Path = Path("outputs/adapted")
    first_new_id: int | None = Field(
        default=None, description="First new token id (default: read from the model folder)"
    )
    phase_a_steps: int = Field(default=150, description="Optimizer steps training only new rows")
    phase_b_steps: int = Field(default=800, description="Steps training LoRA and the new rows")
    # 2 x 16 blocks of 1024 tokens = about 33k tokens per step, so about 470 steps per epoch
    micro_batch_size: int = 2
    grad_accum: int = 16
    lr_new_rows: float = 1e-3
    lr_lora: float = 2e-4
    warmup_steps: int = 50
    min_lr_ratio: float = 0.1
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    lora_rank: int = 64
    lora_alpha: int = 128
    lora_dropout: float = 0.05
    precision: str = Field(default="fp16", description="fp16 (autocast) | fp32")
    gradient_checkpointing: bool = True
    device: str = "cuda"
    seed: int = 0
    # Monitoring, checkpoints, resume
    log_every: int = 10
    eval_every: int = 200
    eval_texts: int = 200
    eval_max_chars: int = 400
    checkpoint_every: int = 100
    keep_checkpoints: int = 2
    milestone_every: int = Field(
        default=0,
        description="Also keep a permanent copy of every Nth checkpoint on the Hub (0 = off)",
    )
    resume: bool = True
    hub_repo: str | None = Field(
        default=None, description="Private Hub repo for checkpoints, e.g. user/zubaan-qwen3-ckpt"
    )
    hub_private: bool = True
    # Merge and save (zubaan adapt merge)
    merged_dtype: str = Field(
        default="bfloat16", description="dtype of the saved model (Qwen3's own dtype)"
    )


class EvaluateConfig(BaseModel):
    """Stage 3 evaluation: score saved model folders on the same texts (zubaan adapt eval).

    Defaults reproduce the Stage 2 protocol (1000 texts of at most 400 characters per set, all
    900 Belebele questions, float32), so the rows can sit next to the Stage 2 table.
    """

    models: dict[str, str] = Field(
        description="Name -> model folder or Hub id, in table order. The first is the reference",
    )
    untrained: str | None = Field(
        default=None, description="Name of the extended model before training (for the checks)"
    )
    adapted: str | None = Field(default=None, description="Name of the adapted model")
    heldout: Path = Path("data/urdu/urdu_heldout.txt")
    eval_urdu: Path = Path("data/flores_urd.txt")
    eval_english: Path = Path("data/flores_eng.txt")
    quality_texts: int | None = Field(default=1000, description="Texts per BPC set (None = all)")
    max_chars: int = 400
    belebele_questions: int | None = Field(default=None, description="Per language (None = 900)")
    dtype: str = Field(default="float32", description="float32 | bfloat16 | float16")
    device: str = "cuda"
    batch_size: int = 4
    max_batch_tokens: int = 2048
    output: Path = Path("reports/adapt_quality.json")
    english_tolerance: float = Field(
        default=0.02, description="Allowed English bits per character above the reference"
    )


class ProjectConfig(BaseModel):
    """Top-level config file: any section may be omitted."""

    audit: AuditConfig | None = None
    compare: CompareConfig | None = None
    acceptance: AcceptanceConfig | None = None
    urdu_corpus: UrduCorpusConfig | None = None
    extend: ExtendConfig | None = None
    export: ExportConfig | None = None
    adapt: AdaptConfig | None = None
    evaluate: EvaluateConfig | None = None


def load_config(path: str | Path) -> ProjectConfig:
    with Path(path).open(encoding="utf-8") as f:
        return ProjectConfig.model_validate(yaml.safe_load(f) or {})
