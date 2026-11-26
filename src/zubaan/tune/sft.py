"""Chat SFT for stage 4: LoRA on the merged Stage 3 model, loss on the assistant's reply only.

The loop follows Stage 3's (``adapt/cpt.py``) and reuses its checkpoints, Hub upload and resume,
learning-rate schedule and NaN guard. A step is ``micro_batch_size x grad_accum`` packed rows from
a deterministic ``RowSampler``, so the step number alone says which data comes next.

Differences from Stage 3:

* **Loss is a mean over reply tokens of the whole step.** Each micro batch adds
  ``sum of token losses / reply tokens in the step``, so a long reply and a short one weigh the
  same per token however the rows fall into micro batches. Logits are computed only at positions
  whose next token is trained (``logits_to_keep``), which saves memory on a 160k vocabulary.
* **One phase.** LoRA trains from step 0; the Stage 2 new rows stay frozen unless
  ``train_new_rows`` (then they train in their own small tensor, as in Stage 3).
* **Packing is checked before training.** ``check_packing`` compares a conversation's logits
  alone and packed behind another one; if attention leaks across conversations the run stops.
* **A held-out loss** on conversations about passages never seen in training replaces bits per
  character as the curve to watch.

``merge_tuned`` folds the adapters into the weights, checks that this does not change the
logits, that the tokenizer files are the Stage 3 ones, and (rows frozen) that every embedding row
is unchanged, and writes the folder with the Stage 3 metadata plus a ``stage4`` entry, so the
Stage 2 export tools work on it unchanged.

Torch, transformers and peft are imported inside the functions that need them.
"""

from __future__ import annotations

import contextlib
import json
import shutil
import time
from collections.abc import Callable, Sequence
from pathlib import Path

from zubaan.adapt.cpt import (
    TrainingDiverged,
    latest_checkpoint,
    load_checkpoint,
    lr_at,
    prune_checkpoints,
    save_checkpoint,
)
from zubaan.adapt.params import (
    AdaptHandle,
    apply_lora,
    attach_new_rows,
    lora_parameters,
    merge_lora,
    merge_new_rows,
    no_new_rows,
)
from zubaan.config import TuneConfig
from zubaan.extend.build import METADATA_FILE
from zubaan.tune.data import (
    IGNORE,
    Example,
    RowSampler,
    encode_records,
    file_sha256,
    hf_renderer,
    is_validation,
    load_conversations,
    pack,
    pad_batch,
    row_arrays,
    total_steps,
)

RESUME_FIELDS = (
    "max_len",
    "val_share",
    "epochs",
    "micro_batch_size",
    "grad_accum",
    "lr_lora",
    "train_new_rows",
    "lr_new_rows",
    "warmup_steps",
    "min_lr_ratio",
    "lora_rank",
    "lora_alpha",
    "precision",
    "seed",
)
TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json")


def first_new_id_of(model_path: Path) -> int | None:
    """First Stage 2 token id from the model folder's zubaan metadata, or None if absent."""
    meta = Path(model_path) / METADATA_FILE
    if not meta.exists():
        return None
    return int(json.loads(meta.read_text(encoding="utf-8"))["first_new_id"])


def prepare_model(model, cfg: TuneConfig, first_new_id: int | None) -> tuple[object, AdaptHandle]:
    """Attach LoRA (and the new-row tensor if ``train_new_rows``) and freeze everything else."""
    if cfg.gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.config.use_cache = False
    if cfg.train_new_rows:
        if first_new_id is None:
            raise ValueError(
                f"train_new_rows needs {METADATA_FILE} with first_new_id in the model folder"
            )
        handle = attach_new_rows(model, first_new_id)
    else:
        handle = no_new_rows(model)
    peft_model = apply_lora(model, cfg.lora_rank, cfg.lora_alpha, cfg.lora_dropout)
    for p in peft_model.parameters():
        p.requires_grad_(False)
    for p in lora_parameters(peft_model):
        p.requires_grad_(True)
    if handle.new_rows is not None:
        handle.new_rows.requires_grad_(True)
    return peft_model, handle


def build_optimizer(model, handle: AdaptHandle, cfg: TuneConfig):
    """AdamW with one group for LoRA and, if trained, one for the new rows."""
    import torch

    groups = [
        {
            "params": lora_parameters(model),
            "lr": cfg.lr_lora,
            "weight_decay": cfg.weight_decay,
            "name": "lora",
        }
    ]
    if handle.new_rows is not None:
        groups.append(
            {"params": [handle.new_rows], "lr": cfg.lr_new_rows, "weight_decay": 0.0, "name": "new"}
        )
    return torch.optim.AdamW(groups, betas=(0.9, 0.999))


def _autocast(precision: str, device: str):
    import torch

    if precision == "fp16" and device.startswith("cuda"):
        return torch.autocast("cuda", dtype=torch.float16)
    return contextlib.nullcontext()


def trained_targets(labels: Sequence[Sequence[int]]) -> int:
    """Reply tokens that are predicted in a batch: labels after the first position not IGNORE."""
    return sum(sum(1 for t in row[1:] if t != IGNORE) for row in labels)


def batch_loss(model, batch, device: str, precision: str):
    """Sum of next-token losses over reply tokens of one padded batch, and their count.

    ``batch`` is ``(ids, labels, positions)`` from ``pad_batch``. Position ``i`` predicts
    token ``i + 1``; only positions whose target is a reply token get logits. Position ids are
    passed without an attention mask, which Transformers reads as packed conversations.
    """
    import torch
    import torch.nn.functional as F

    ids, labels, positions = batch
    x = torch.tensor(ids, dtype=torch.long, device=device)
    y = torch.tensor(labels, dtype=torch.long, device=device)
    pos = torch.tensor(positions, dtype=torch.long, device=device)
    targets = y[:, 1:]
    keep = (targets != IGNORE).any(dim=0).nonzero().squeeze(1)
    if keep.numel() == 0:
        return None, 0
    with _autocast(precision, device):
        logits = model(input_ids=x, position_ids=pos, logits_to_keep=keep).logits
    t = targets[:, keep]
    loss = F.cross_entropy(
        logits.float().reshape(-1, logits.shape[-1]),
        t.reshape(-1),
        ignore_index=IGNORE,
        reduction="sum",
    )
    return loss, int((t != IGNORE).sum())


def check_packing(model, examples: Sequence[Example], device: str, precision: str) -> float:
    """Largest logit difference for a conversation alone versus packed behind another one.

    Uses the two shortest examples. Raises ValueError above the tolerance (1e-3 in float32,
    5e-2 under fp16), which means attention crosses conversation boundaries in this
    Transformers version and packing would train on the wrong context.
    """
    import torch

    if len(examples) < 2:
        return 0.0
    a, b = sorted(examples, key=len)[:2]
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad(), _autocast(precision, device):
            alone_ids = torch.tensor([list(b.ids)], device=device)
            alone = model(input_ids=alone_ids).logits[0].float()
            ids, _, pos = row_arrays([a, b])
            packed = (
                model(
                    input_ids=torch.tensor([ids], device=device),
                    position_ids=torch.tensor([pos], device=device),
                )
                .logits[0, len(a) :]
                .float()
            )
    finally:
        model.train(was_training)
    diff = float((alone - packed).abs().max())
    tol = 5e-2 if precision == "fp16" and device.startswith("cuda") else 1e-3
    if diff > tol:
        raise ValueError(
            f"packed conversations change each other's logits (max difference {diff:.3g}): this "
            "Transformers version does not isolate packed sequences; upgrade it or set max_len "
            "to fit one conversation per row"
        )
    return diff


def eval_loss(model, batches: Sequence, device: str, precision: str) -> dict:
    """Mean loss per reply token over held-out batches."""
    import torch

    total, n = 0.0, 0
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            for batch in batches:
                loss, k = batch_loss(model, batch, device, precision)
                if k:
                    total += float(loss)
                    n += k
    finally:
        model.train(was_training)
    return {"val_loss": total / n if n else None, "val_tokens": n}


def _batches(
    examples: Sequence[Example], rows: Sequence[Sequence[int]], size: int, pad_id: int
) -> list:
    out = []
    for i in range(0, len(rows), size):
        arrays = [row_arrays([examples[j] for j in row]) for row in rows[i : i + size]]
        out.append(pad_batch(arrays, pad_id))
    return out


def train(
    model,
    handle: AdaptHandle,
    sampler: RowSampler,
    examples: Sequence[Example],
    cfg: TuneConfig,
    pad_id: int,
    device: str = "cpu",
    start_step: int = 0,
    optimizer=None,
    scaler=None,
    val_batches: Sequence | None = None,
    on_checkpoint: Callable[[Path, int], None] | None = None,
    checkpoint_meta: dict | None = None,
    log: Callable[[str], None] = print,
    log_path: Path | None = None,
    max_steps: int | None = None,
) -> dict:
    """Run optimizer steps ``start_step .. total`` and return a summary with the history.

    ``checkpoint_meta`` is added to each checkpoint's ``meta.json`` (the data fingerprint the
    resume check compares). Raises ``TrainingDiverged`` on a non-finite loss.
    """
    import torch

    per_step = cfg.micro_batch_size * cfg.grad_accum
    total = total_steps(len(sampler), per_step, cfg.epochs)
    stop = total if max_steps is None else min(total, start_step + max_steps)
    if optimizer is None:
        optimizer = build_optimizer(model, handle, cfg)
    if scaler is None and cfg.precision == "fp16" and device.startswith("cuda"):
        scaler = torch.amp.GradScaler("cuda")
    ckpt_dir = cfg.output_dir / "checkpoints"
    warmup = min(cfg.warmup_steps, total)
    history: list[dict] = []
    t0, tokens_since = time.time(), 0
    model.train()

    def write(row: dict) -> None:
        history.append(row)
        if log_path is not None:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with log_path.open("a", encoding="utf-8", newline="\n") as f:
                f.write(json.dumps(row) + "\n")

    for step in range(start_step, stop):
        for group in optimizer.param_groups:
            peak = cfg.lr_new_rows if group["name"] == "new" else cfg.lr_lora
            group["lr"] = lr_at(step, total, peak, warmup, cfg.min_lr_ratio)
        rows = sampler.rows_at(step * per_step, per_step)
        batches = _batches(examples, rows, cfg.micro_batch_size, pad_id)
        n_step = sum(trained_targets(b[1]) for b in batches)
        loss_sum = 0.0
        for batch in batches:
            loss, k = batch_loss(model, batch, device, cfg.precision)
            if not k:
                continue
            if not torch.isfinite(loss):
                raise TrainingDiverged(
                    f"loss is {loss.item()} at step {step}. With fp16 this is usually an "
                    "overflow: set precision: fp32 and rerun the smoke test."
                )
            loss_sum += loss.item()
            scaled = loss / n_step
            (scaler.scale(scaled) if scaler is not None else scaled).backward()
            tokens_since += sum(len(r) for r in batch[0])

        params = [p for g in optimizer.param_groups for p in g["params"] if p.grad is not None]
        if scaler is not None:
            scaler.unscale_(optimizer)
        grad_norm = float(torch.nn.utils.clip_grad_norm_(params, cfg.max_grad_norm))
        if scaler is not None:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        done = step + 1

        if done % cfg.log_every == 0 or done == stop:
            dt = max(time.time() - t0, 1e-9)
            row = {
                "step": done,
                "loss": loss_sum / max(n_step, 1),
                "lr_lora": optimizer.param_groups[0]["lr"],
                "grad_norm": grad_norm,
                "reply_tokens": n_step,
                "epoch": sampler.epoch_of(done * per_step),
                "tokens_per_s": tokens_since / dt,
            }
            if device.startswith("cuda"):
                row["gpu_mem_gb"] = torch.cuda.max_memory_allocated() / 1e9
            write(row)
            log(
                f"step {done}/{total} loss {row['loss']:.4f} grad {grad_norm:.2f} "
                f"epoch {row['epoch']:.2f} {row['tokens_per_s']:.0f} tok/s"
            )
            t0, tokens_since = time.time(), 0
        if val_batches and (done % cfg.eval_every == 0 or done == stop):
            metrics = eval_loss(model, val_batches, device, cfg.precision)
            write({"step": done, "eval": metrics})
            log(f"step {done} eval {metrics}")
        if done % cfg.checkpoint_every == 0 or done == stop:
            path = save_checkpoint(ckpt_dir, done, model, handle, optimizer, scaler, cfg)
            if checkpoint_meta:
                meta_file = path / "meta.json"
                meta = json.loads(meta_file.read_text(encoding="utf-8"))
                meta.update(checkpoint_meta)
                meta_file.write_text(
                    json.dumps(meta, indent=2) + "\n", encoding="utf-8", newline="\n"
                )
            prune_checkpoints(ckpt_dir, cfg.keep_checkpoints)
            if on_checkpoint is not None:
                on_checkpoint(path, done)

    return {"final_step": stop, "total_steps": total, "finished": stop == total, "history": history}


def check_resume_compatible(ckpt: Path, cfg: TuneConfig, data: dict) -> None:
    """Refuse a checkpoint written with other settings or other data: the step would then point
    at different rows and a different learning rate than the run that was saved."""
    saved = json.loads((ckpt / "meta.json").read_text(encoding="utf-8"))
    now = json.loads(cfg.model_dump_json())
    diffs = [
        f"{k}: saved {saved['config'].get(k)!r}, now {now[k]!r}"
        for k in RESUME_FIELDS
        if saved["config"].get(k) != now[k]
    ]
    for k, v in data.items():
        if saved.get("data", {}).get(k) != v:
            diffs.append(f"data {k}: saved {saved.get('data', {}).get(k)!r}, now {v!r}")
    if diffs:
        raise ValueError(
            f"checkpoint {ckpt.name} was made with different settings ({'; '.join(diffs)}); "
            "use the same config and data, or point output_dir and hub_repo at a fresh run"
        )


def load_data(cfg: TuneConfig, tokenizer) -> dict:
    """Read, split, tokenize and pack the conversations. Deterministic for a given file."""
    render, encode = hf_renderer(tokenizer)
    records = load_conversations(cfg.train_file)
    train_recs = [r for r in records if not is_validation(r, cfg.val_share)]
    val_recs = [r for r in records if is_validation(r, cfg.val_share)]
    train_ex, train_stats = encode_records(train_recs, render, encode, cfg.max_len)
    val_ex, val_stats = encode_records(val_recs, render, encode, cfg.max_len)
    train_rows = pack([len(e) for e in train_ex], cfg.max_len)
    val_rows = pack([len(e) for e in val_ex], cfg.max_len) if val_ex else []
    fill = train_stats["tokens"] / (len(train_rows) * cfg.max_len) if train_rows else 0.0
    return {
        "train": train_ex,
        "val": val_ex,
        "train_rows": train_rows,
        "val_rows": val_rows,
        "stats": {
            "file": str(cfg.train_file),
            "train": {**train_stats, "rows": len(train_rows), "row_fill": round(fill, 4)},
            "val": {**val_stats, "rows": len(val_rows)},
            "thinking_share": round(
                sum(bool(r["thinking"]) for r in train_recs) / max(len(train_recs), 1), 4
            ),
        },
        "fingerprint": {
            "train_file_sha256": file_sha256(cfg.train_file),
            "train_rows": len(train_rows),
        },
    }


def _load_tokenizer(model_path: Path):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(model_path)


def run_tune(
    cfg: TuneConfig,
    log: Callable[[str], None] = print,
    max_steps: int | None = None,
    stop_at_step: int | None = None,
    model_factory: Callable[[], object] | None = None,
    uploader: Callable[[Path, int], None] | None = None,
    downloader: Callable[[Path], Path | None] | None = None,
) -> dict:
    """Train: load the Stage 3 model and the conversations, check packing, run, checkpoint, resume.

    ``max_steps`` stops after that many steps (smoke runs); ``stop_at_step`` stops at that global
    step wherever the run resumed. With ``hub_repo`` set, checkpoints go to that private repo and
    a checkpoint found there is used to resume when none exists locally.
    """
    import torch

    from zubaan.adapt.pipeline import download_hub_checkpoint, make_hub_uploader

    if max_steps is not None and stop_at_step is not None:
        raise ValueError("give max_steps or stop_at_step, not both")
    device = cfg.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("device is cuda but no GPU is available (Kaggle: turn on the T4)")

    tokenizer = _load_tokenizer(cfg.model_path)
    data = load_data(cfg, tokenizer)
    log(f"Data: {json.dumps(data['stats'])}")
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

    torch.manual_seed(cfg.seed)
    if model_factory is None:
        from transformers import AutoModelForCausalLM

        dtype = torch.float16 if cfg.precision == "fp16" else torch.float32
        model = AutoModelForCausalLM.from_pretrained(cfg.model_path, dtype=dtype).to(device)
    else:
        model = model_factory().to(device)
    peft_model, handle = prepare_model(model, cfg, first_new_id_of(cfg.model_path))
    packing_diff = check_packing(peft_model, data["train"], device, cfg.precision)
    log(f"Packing check: max logit difference {packing_diff:.2e}")
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
            check_resume_compatible(ckpt, cfg, data["fingerprint"])
            start = load_checkpoint(ckpt, peft_model, handle, optimizer, scaler)
            log(f"Resumed from {ckpt.name} (step {start})")
    if stop_at_step is not None:
        max_steps = max(0, stop_at_step - start)

    sampler = RowSampler(data["train_rows"], cfg.seed)
    val_batches = _batches(data["val"], data["val_rows"], cfg.micro_batch_size, pad_id)
    result = train(
        peft_model,
        handle,
        sampler,
        data["train"],
        cfg,
        pad_id,
        device=device,
        start_step=start,
        optimizer=optimizer,
        scaler=scaler,
        val_batches=val_batches,
        on_checkpoint=uploader,
        checkpoint_meta={"data": data["fingerprint"]},
        log=log,
        log_path=cfg.output_dir / "train_log.jsonl",
        max_steps=max_steps,
    )
    summary = {
        "model_path": str(cfg.model_path),
        "data": data["stats"],
        "packing_max_logit_diff": packing_diff,
        "train_new_rows": cfg.train_new_rows,
        "resumed_from_step": start,
        "final_step": result["final_step"],
        "total_steps": result["total_steps"],
        "finished": result["finished"],
        "evals": [h for h in result["history"] if "eval" in h],
    }
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    (cfg.output_dir / "tune_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n"
    )
    return summary


def merge_tuned(
    cfg: TuneConfig,
    checkpoint: Path | None = None,
    out_dir: Path | None = None,
    log: Callable[[str], None] = print,
    downloader: Callable[[Path], Path | None] | None = None,
) -> dict:
    """Merge the newest checkpoint (or ``checkpoint``) into the Stage 3 model and save it."""
    import torch
    from transformers import AutoModelForCausalLM

    from zubaan.adapt.finalize import copy_support_files

    ckpt_root = cfg.output_dir / "checkpoints"
    ckpt = checkpoint or latest_checkpoint(ckpt_root)
    if ckpt is None and downloader is None and cfg.hub_repo:
        from zubaan.adapt.pipeline import download_hub_checkpoint

        def downloader(directory: Path) -> Path | None:
            return download_hub_checkpoint(cfg.hub_repo, directory)

    if ckpt is None and downloader is not None:
        ckpt = downloader(ckpt_root)
    if ckpt is None:
        raise FileNotFoundError(f"no checkpoint in {ckpt_root}; run 'zubaan tune run' first")
    out = out_dir or cfg.output_dir / "model"

    log(f"Merging {ckpt.name} into {cfg.model_path} (float32)")
    model = AutoModelForCausalLM.from_pretrained(cfg.model_path, dtype=torch.float32)
    original_rows = model.get_input_embeddings().weight.detach().clone()
    merge_cfg = cfg.model_copy(update={"gradient_checkpointing": False})
    peft_model, handle = prepare_model(model, merge_cfg, first_new_id_of(cfg.model_path))
    step = load_checkpoint(ckpt, peft_model, handle, restore_state=False)
    peft_model.eval()

    rows = original_rows.shape[0]
    probe = torch.cat([torch.arange(0, 16), torch.arange(rows - 16, rows)])[None]
    with torch.no_grad():
        before = peft_model(input_ids=probe).logits
    plain = merge_lora(peft_model)
    merge_new_rows(plain, handle)
    with torch.no_grad():
        after = plain(input_ids=probe).logits
    max_diff = float((before - after).abs().max())
    if max_diff > 1e-3:
        raise ValueError(f"merging changed the logits by {max_diff:.2e}; refusing to save")
    emb = plain.get_input_embeddings().weight
    n_fixed = handle.first_new_id if handle.new_rows is not None else rows
    if not torch.equal(emb[:n_fixed], original_rows[:n_fixed]):
        raise ValueError("embedding rows that were meant to stay frozen changed; refusing to save")

    plain.config.use_cache = True
    plain = plain.to(getattr(torch, cfg.merged_dtype))
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    plain.save_pretrained(out, safe_serialization=True)
    copied = copy_support_files(cfg.model_path, out)
    for name in TOKENIZER_FILES:
        src, dst = cfg.model_path / name, out / name
        if src.exists() and src.read_bytes() != dst.read_bytes():
            raise ValueError(f"{name} differs from the Stage 3 model's; refusing to save")

    src_meta = cfg.model_path / METADATA_FILE
    info = json.loads(src_meta.read_text(encoding="utf-8")) if src_meta.exists() else {}
    meta = {
        **info,
        "stage4": {
            "note": "Chat SFT (LoRA) on teacher-written conversations, merged into the weights.",
            "checkpoint": ckpt.name,
            "step": step,
            "train_new_rows": cfg.train_new_rows,
            "lora_rank": cfg.lora_rank,
            "lora_alpha": cfg.lora_alpha,
            "precision": cfg.precision,
            "merge_max_logit_diff": max_diff,
            "saved_dtype": cfg.merged_dtype,
            "support_files_copied": copied,
        },
    }
    (out / METADATA_FILE).write_text(
        json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n"
    )
    log(f"Saved the merged model to {out} (step {step}, max logit difference {max_diff:.1e})")
    return {"model_dir": str(out), "step": step, "merge_max_logit_diff": max_diff, "metadata": meta}
