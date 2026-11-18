"""Stage 3 commands as library functions: prepare the token blocks, run continued pretraining.

``zubaan adapt prepare`` and ``zubaan adapt run`` are thin wrappers over ``run_prepare`` and
``run_adapt``. Heavy imports (torch, transformers, datasets, huggingface_hub) happen inside.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from pathlib import Path

from zubaan.adapt.data import (
    BlockSampler,
    build_english_replay,
    hf_batch_encoder,
    iter_english_wikipedia,
    load_blocks,
    prepare_blocks,
)
from zubaan.config import AdaptConfig
from zubaan.extend.build import METADATA_FILE
from zubaan.io import read_lines


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def belebele_passages(lang: str) -> list[str]:
    """Distinct reading passages of Belebele (they are FLORES-200 text, so they must stay out)."""
    from zubaan.bench.quality import load_belebele

    return sorted({item["flores_passage"] for item in load_belebele(lang)})


def _guard_texts(files: list[Path], belebele_lang: str | None) -> list[str]:
    texts: list[str] = []
    for f in files:
        if f.exists():
            texts += read_lines(f)
    if belebele_lang:
        texts += belebele_passages(belebele_lang)
    return texts


def run_prepare(cfg: AdaptConfig, log: Callable[[str], None] = print) -> dict:
    """Tokenize the Urdu and English text with the extended tokenizer and write token blocks."""
    from tokenizers import Tokenizer

    from zubaan.extend.quality_run import context_id

    tok_path = cfg.model_path / "tokenizer.json"
    if not tok_path.exists():
        raise FileNotFoundError(f"{tok_path} not found; run 'zubaan extend run' first")
    tokenizer = Tokenizer.from_file(str(tok_path))
    eos_id = context_id(json.loads(tok_path.read_text(encoding="utf-8")))

    urdu_eval = _guard_texts(cfg.guard_urdu, "urd_Arab" if cfg.guard_belebele else None)
    english_eval = _guard_texts(cfg.guard_english, "eng_Latn" if cfg.guard_belebele else None)
    if not urdu_eval or not english_eval:
        raise ValueError(
            "no evaluation texts to guard against; check guard_urdu and guard_english "
            "(run 'zubaan data flores' first)"
        )

    if not cfg.english_corpus.exists():
        log(f"Building English replay text at {cfg.english_corpus} ...")
        meta = build_english_replay(
            iter_english_wikipedia(), cfg.english_corpus, english_eval, cfg.english_mb
        )
        meta_path = cfg.english_corpus.with_suffix(".meta.json")
        meta_path.write_text(
            json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n"
        )
    english_lines = read_lines(cfg.english_corpus)
    urdu_lines = read_lines(cfg.urdu_corpus)
    log(f"{len(urdu_lines):,} Urdu lines, {len(english_lines):,} English lines")

    info = json.loads((cfg.model_path / METADATA_FILE).read_text(encoding="utf-8"))
    meta = prepare_blocks(
        urdu_lines,
        english_lines,
        hf_batch_encoder(tokenizer),
        eos_id,
        cfg.data_dir,
        cfg.block_size,
        urdu_eval_texts=urdu_eval,
        english_eval_texts=english_eval,
        extra_meta={
            "tokenizer_sha256": _sha256_file(tok_path),
            "first_new_id": info["first_new_id"],
            "urdu_corpus": str(cfg.urdu_corpus),
            "english_corpus": str(cfg.english_corpus),
        },
        drop_overlaps=cfg.drop_overlaps,
    )
    log(
        f"Urdu: {meta['urdu']['n_blocks']:,} blocks ({meta['urdu']['n_tokens']:,} tokens). "
        f"English: {meta['english']['n_blocks']:,} blocks. "
        f"Guard dropped {meta['overlap_guard']['urdu_lines_dropped']} Urdu and "
        f"{meta['overlap_guard']['english_lines_dropped']} English lines."
    )
    return meta


MILESTONE_DIR = "milestones"


def milestone_path(step: int) -> str:
    """Folder name inside the Hub repo for the permanent copy of a checkpoint."""
    return f"{MILESTONE_DIR}/step-{step:07d}"


def make_hub_uploader(
    repo_id: str,
    private: bool = True,
    milestone_every: int = 0,
    log: Callable[[str], None] = print,
) -> Callable[[Path, int], None]:
    """Upload a checkpoint folder to ``<repo>/last`` in a private Hub repo.

    Every ``milestone_every`` steps the same folder is also copied to
    ``<repo>/milestones/step-NNNNNNN`` and kept, so any of them can be merged later.
    A failed upload is reported loudly but does not stop the run: the local checkpoint is
    still there and the next upload retries.

    Uses the token from the environment (``HF_TOKEN``) or a cached login. Only called when
    ``hub_repo`` is configured; nothing is ever pushed otherwise.
    """
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(repo_id, private=private, exist_ok=True)

    def upload(path: Path, step: int) -> None:
        try:
            api.upload_folder(
                folder_path=str(path),
                repo_id=repo_id,
                path_in_repo="last",
                delete_patterns="*",
                commit_message=f"checkpoint step {step}",
            )
            if milestone_every and step % milestone_every == 0:
                api.upload_folder(
                    folder_path=str(path),
                    repo_id=repo_id,
                    path_in_repo=milestone_path(step),
                    commit_message=f"milestone step {step}",
                )
        except Exception as e:  # keep training; the local checkpoint exists
            log(f"WARNING: Hub upload at step {step} failed ({type(e).__name__}: {e}). Continuing.")

    return upload


def download_hub_checkpoint(repo_id: str, directory: Path, folder: str = "last") -> Path | None:
    """Fetch ``<repo>/<folder>`` into ``directory`` as a ``ckpt-<step>`` folder, or None if absent.

    ``folder`` is ``last`` (the newest checkpoint) or ``milestone_path(step)``.
    """
    import shutil

    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import HfHubHTTPError

    from zubaan.adapt.cpt import CHECKPOINT_PREFIX

    tmp = directory / ".hub-download"
    try:
        snapshot_download(repo_id, allow_patterns=f"{folder}/*", local_dir=str(tmp))
    except HfHubHTTPError:
        return None
    last = tmp / folder
    if not (last / "meta.json").exists():
        return None
    step = json.loads((last / "meta.json").read_text(encoding="utf-8"))["step"]
    final = directory / f"{CHECKPOINT_PREFIX}{step:07d}"
    directory.mkdir(parents=True, exist_ok=True)
    if final.exists():
        shutil.rmtree(final)
    shutil.move(str(last), str(final))
    shutil.rmtree(tmp, ignore_errors=True)
    return final


RESUME_FIELDS = (
    "phase_a_steps",
    "phase_b_steps",
    "micro_batch_size",
    "grad_accum",
    "block_size",
    "lora_rank",
    "lora_alpha",
    "seed",
    "english_replay_ratio",
    "lr_new_rows",
    "lr_lora",
    "precision",
)


def check_resume_compatible(ckpt: Path, cfg: AdaptConfig) -> None:
    """Refuse to resume a checkpoint that was written under a different schedule or data order.

    The step number selects the data and the learning rate, so changing any of these fields
    would continue a different run than the one that was saved.
    """
    saved = json.loads((ckpt / "meta.json").read_text(encoding="utf-8"))["config"]
    now = json.loads(cfg.model_dump_json())
    diffs = [
        f"{k}: saved {saved.get(k)!r}, now {now[k]!r}"
        for k in RESUME_FIELDS
        if saved.get(k) != now[k]
    ]
    if diffs:
        raise ValueError(
            f"checkpoint {ckpt.name} was made with different settings ({'; '.join(diffs)}); "
            "use the same config, or point output_dir and hub_repo at a fresh run"
        )


def make_eval_fn(cfg: AdaptConfig, tokenizer, ctx_id: int, device: str) -> Callable | None:
    """Held-out Urdu bits per character, plus an English control, for the training curve."""
    from zubaan.bench.quality import text_bits_per_character

    def pick(path: Path, n: int) -> list[str]:
        if not path.exists():
            return []
        lines = [t for t in read_lines(path) if 0 < len(t) <= cfg.eval_max_chars]
        return lines[:n]

    urdu = pick(cfg.heldout, cfg.eval_texts)
    english = pick(cfg.guard_english[0], 100) if cfg.guard_english else []
    if not urdu:
        return None

    def evaluate(model) -> dict:
        out = {}
        # Texts are capped by characters (eval_max_chars), so 1024 tokens is never reached in
        # practice; the harness raises instead of cutting a text if it ever is.
        kw = {"context_id": ctx_id, "batch_size": 4, "max_tokens": 1024, "device": device}
        r = text_bits_per_character(model, tokenizer, urdu, **kw)
        out["bpc_urdu_heldout"] = r["bits_per_character"]
        out["tokens_per_char_urdu"] = r["tokens_per_char"]
        if english:
            out["bpc_english_flores"] = text_bits_per_character(model, tokenizer, english, **kw)[
                "bits_per_character"
            ]
        return out

    return evaluate


def run_adapt(
    cfg: AdaptConfig,
    log: Callable[[str], None] = print,
    max_steps: int | None = None,
    model_factory: Callable[[], object] | None = None,
    uploader: Callable[[Path, int], None] | None = None,
    downloader: Callable[[Path], Path | None] | None = None,
) -> dict:
    """Train: load the extended model, attach the new rows and LoRA, run, checkpoint, resume.

    ``model_factory`` replaces loading from ``cfg.model_path`` (used by tests). ``max_steps``
    stops after that many steps (smoke runs). If ``hub_repo`` is set, checkpoints are pushed to
    that private repo and a checkpoint found there is used to resume when none exists locally.
    """
    import torch
    from tokenizers import Tokenizer

    from zubaan.adapt.cpt import (
        build_optimizer,
        latest_checkpoint,
        load_checkpoint,
        prepare_for_training,
        train,
    )
    from zubaan.extend.quality_run import context_id

    info = json.loads((cfg.model_path / METADATA_FILE).read_text(encoding="utf-8"))
    first_new_id = cfg.first_new_id or info["first_new_id"]
    if info.get("trained"):
        log("Warning: this model folder is already marked trained; continuing from it.")

    urdu, english, data_meta = load_blocks(cfg.data_dir)
    tok_path = cfg.model_path / "tokenizer.json"
    if data_meta.get("tokenizer_sha256") != _sha256_file(tok_path):
        raise ValueError(
            "the token blocks were made with a different tokenizer than the model folder; "
            "run 'zubaan adapt prepare' again"
        )
    if data_meta["block_size"] != cfg.block_size:
        raise ValueError(
            f"block_size {cfg.block_size} differs from the prepared blocks "
            f"({data_meta['block_size']})"
        )
    sampler = BlockSampler(urdu, english, cfg.english_replay_ratio, cfg.seed)

    device = cfg.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("device is cuda but no GPU is available (Kaggle: turn on the T4)")
    torch.manual_seed(cfg.seed)
    if model_factory is None:
        from transformers import AutoModelForCausalLM

        dtype = torch.float16 if cfg.precision == "fp16" else torch.float32
        model = AutoModelForCausalLM.from_pretrained(cfg.model_path, dtype=dtype).to(device)
    else:
        model = model_factory().to(device)
    peft_model, handle = prepare_for_training(model, first_new_id, cfg)
    optimizer = build_optimizer(peft_model, handle, cfg)
    scaler = (
        torch.amp.GradScaler("cuda")
        if cfg.precision == "fp16" and device.startswith("cuda")
        else None
    )

    ckpt_dir = cfg.output_dir / "checkpoints"
    if uploader is None and cfg.hub_repo:
        uploader = make_hub_uploader(cfg.hub_repo, cfg.hub_private, cfg.milestone_every, log)
    if downloader is None and cfg.hub_repo:

        def downloader(directory: Path) -> Path | None:
            return download_hub_checkpoint(cfg.hub_repo, directory)

    start = 0
    if cfg.resume:
        ckpt = latest_checkpoint(ckpt_dir)
        if ckpt is None and downloader is not None:
            ckpt = downloader(ckpt_dir)
        if ckpt is not None:
            check_resume_compatible(ckpt, cfg)
            start = load_checkpoint(ckpt, peft_model, handle, optimizer, scaler)
            log(f"Resumed from {ckpt.name} (step {start})")

    tokenizer = Tokenizer.from_file(str(tok_path))
    ctx = context_id(json.loads(tok_path.read_text(encoding="utf-8")))
    eval_fn = make_eval_fn(cfg, tokenizer, ctx, device)
    result = train(
        peft_model,
        handle,
        sampler,
        cfg,
        device=device,
        start_step=start,
        optimizer=optimizer,
        scaler=scaler,
        eval_fn=eval_fn,
        on_checkpoint=uploader,
        log=log,
        log_path=cfg.output_dir / "train_log.jsonl",
        max_steps=max_steps,
    )
    summary = {
        "model_path": str(cfg.model_path),
        "first_new_id": first_new_id,
        "resumed_from_step": start,
        "final_step": result["final_step"],
        "total_steps": result["total_steps"],
        "finished": result["finished"],
        "tokens_per_step": cfg.micro_batch_size * cfg.grad_accum * cfg.block_size,
        "data": {"urdu_blocks": len(urdu), "english_blocks": len(english)},
        "evals": [h for h in result["history"] if "eval" in h],
    }
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    (cfg.output_dir / "adapt_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n"
    )
    return summary
