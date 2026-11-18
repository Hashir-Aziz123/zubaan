"""Continued pretraining loop for stage 3 (new rows, then LoRA).

The loop is small on purpose: raw-text pretraining needs no trainer framework, and owning the loop
makes resume exact. A run is a list of optimizer *steps*. Each step consumes
``micro_batch_size x grad_accum`` consecutive blocks from a deterministic ``BlockSampler``, so the
step number alone says which data comes next. A checkpoint stores the step, the new rows, the
LoRA weights, the optimizer, the gradient scaler and the random state; resuming from it continues
the same run bit for bit on CPU.

Phases (decision of 2026-11-18): steps ``0 .. phase_a_steps - 1`` train only the new rows; the
following ``phase_b_steps`` train the LoRA adapters and the new rows. Each phase has its own
learning rate schedule: linear warmup, then cosine decay to ``min_lr_ratio`` of the peak.

Torch is imported inside the functions that need it.
"""

from __future__ import annotations

import json
import math
import shutil
import time
from collections.abc import Callable
from pathlib import Path

from zubaan.adapt.data import BlockSampler
from zubaan.adapt.params import (
    AdaptHandle,
    apply_lora,
    attach_new_rows,
    lora_parameters,
    set_phase,
)
from zubaan.config import AdaptConfig

CHECKPOINT_PREFIX = "ckpt-"


class TrainingDiverged(RuntimeError):
    """The loss stopped being a finite number. Training stops instead of burning GPU hours."""


def lr_at(step: int, total: int, peak: float, warmup: int, min_ratio: float = 0.1) -> float:
    """Learning rate at ``step`` (0-based) of a phase that lasts ``total`` steps.

    Warmup: lr = peak x (step + 1) / warmup while step < warmup.
    Then cosine: lr = peak x (min_ratio + (1 - min_ratio) x 0.5 x (1 + cos(pi x p))), where
    p = (step - warmup) / (total - warmup) is how far through the decay the phase is.
    Example: peak 1e-3, warmup 10, total 110, step 60: p = 0.5, so lr = 1e-3 x (0.1 + 0.9 x 0.5)
    = 5.5e-4.
    """
    if total < 1 or warmup < 0 or not 0 <= min_ratio <= 1:
        raise ValueError("need total >= 1, warmup >= 0 and min_ratio in [0, 1]")
    if step < warmup:
        return peak * (step + 1) / warmup
    p = (step - warmup) / max(1, total - warmup)
    p = min(p, 1.0)
    return peak * (min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * p)))


def phase_at(step: int, cfg: AdaptConfig) -> tuple[str, int, int]:
    """Phase name, step inside the phase, and the phase length for a global step."""
    if step < cfg.phase_a_steps:
        return "A", step, cfg.phase_a_steps
    return "B", step - cfg.phase_a_steps, cfg.phase_b_steps


def prepare_for_training(model, first_new_id: int, cfg: AdaptConfig):
    """Attach the new-row parameter and LoRA to a loaded model. Returns ``(model, handle)``."""
    if cfg.gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.config.use_cache = False
    handle = attach_new_rows(model, first_new_id)
    peft_model = apply_lora(model, cfg.lora_rank, cfg.lora_alpha, cfg.lora_dropout)
    set_phase(peft_model, handle, "A")
    return peft_model, handle


def build_optimizer(model, handle: AdaptHandle, cfg: AdaptConfig):
    """One AdamW with a group for the new rows and one for LoRA.

    LoRA tensors have no gradient in phase A, and AdamW skips tensors without one, so a single
    optimizer serves both phases and keeps the new rows' moments across the switch.
    """
    import torch

    groups = [
        {"params": [handle.new_rows], "lr": cfg.lr_new_rows, "weight_decay": 0.0, "name": "new"},
    ]
    lora = lora_parameters(model)
    if lora:
        groups.append(
            {"params": lora, "lr": cfg.lr_lora, "weight_decay": cfg.weight_decay, "name": "lora"}
        )
    return torch.optim.AdamW(groups, betas=(0.9, 0.999))


# checkpoints -------------------------------------------------------------------------------------


def _rng_state() -> dict:
    import torch

    state = {"cpu": torch.get_rng_state()}
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng(state: dict) -> None:
    import torch

    torch.set_rng_state(state["cpu"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def save_checkpoint(
    directory: Path, step: int, model, handle: AdaptHandle, optimizer, scaler, cfg: AdaptConfig
) -> Path:
    """Write ``ckpt-<step>/`` atomically (temp folder, then rename) and return its path."""
    import torch
    from peft import get_peft_model_state_dict

    directory.mkdir(parents=True, exist_ok=True)
    final = directory / f"{CHECKPOINT_PREFIX}{step:07d}"
    tmp = directory / f".tmp-{step}"
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir()
    torch.save(handle.new_rows.detach().cpu(), tmp / "new_rows.pt")
    torch.save(
        {k: v.detach().cpu() for k, v in get_peft_model_state_dict(model).items()},
        tmp / "lora.pt",
    )
    torch.save(
        {
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict() if scaler is not None else None,
            "rng": _rng_state(),
        },
        tmp / "trainer.pt",
    )
    meta = {"step": step, "config": json.loads(cfg.model_dump_json())}
    (tmp / "meta.json").write_text(
        json.dumps(meta, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    if final.exists():
        shutil.rmtree(final)
    tmp.rename(final)
    return final


def latest_checkpoint(directory: Path) -> Path | None:
    """Newest complete ``ckpt-*`` folder, or None."""
    if not directory.exists():
        return None
    found = sorted(p for p in directory.glob(f"{CHECKPOINT_PREFIX}*") if (p / "meta.json").exists())
    return found[-1] if found else None


def prune_checkpoints(directory: Path, keep: int) -> None:
    """Delete all but the newest ``keep`` checkpoints."""
    found = sorted(directory.glob(f"{CHECKPOINT_PREFIX}*"))
    for old in found[: max(0, len(found) - keep)]:
        shutil.rmtree(old, ignore_errors=True)


def load_checkpoint(path: Path, model, handle: AdaptHandle, optimizer, scaler) -> int:
    """Restore a checkpoint into a freshly prepared model. Returns the step it was saved at."""
    import torch
    from peft import set_peft_model_state_dict

    meta = json.loads((path / "meta.json").read_text(encoding="utf-8"))
    with torch.no_grad():
        handle.new_rows.copy_(torch.load(path / "new_rows.pt"))
    set_peft_model_state_dict(model, torch.load(path / "lora.pt"))
    trainer = torch.load(path / "trainer.pt", weights_only=False)
    optimizer.load_state_dict(trainer["optimizer"])
    if scaler is not None and trainer["scaler"] is not None:
        scaler.load_state_dict(trainer["scaler"])
    _restore_rng(trainer["rng"])
    return int(meta["step"])


# the loop ----------------------------------------------------------------------------------------


def _autocast(cfg: AdaptConfig, device: str):
    import contextlib

    import torch

    if cfg.precision == "fp16" and device.startswith("cuda"):
        return torch.autocast("cuda", dtype=torch.float16)
    return contextlib.nullcontext()


def train(
    model,
    handle: AdaptHandle,
    sampler: BlockSampler,
    cfg: AdaptConfig,
    device: str = "cpu",
    start_step: int = 0,
    optimizer=None,
    scaler=None,
    eval_fn: Callable[[object], dict] | None = None,
    on_checkpoint: Callable[[Path, int], None] | None = None,
    log: Callable[[str], None] = print,
    log_path: Path | None = None,
    max_steps: int | None = None,
) -> dict:
    """Run optimizer steps ``start_step .. total`` and return a summary.

    ``eval_fn(model)`` returns a dict of metrics (for example bits per character) and runs every
    ``eval_every`` steps and at the end. ``on_checkpoint(path, step)`` is called after each saved
    checkpoint (the caller pushes it to the Hub). ``max_steps`` stops early without finishing
    the schedule (used by the smoke run and by tests). Raises ``TrainingDiverged`` on a
    non-finite loss.
    """
    import torch

    total = cfg.phase_a_steps + cfg.phase_b_steps
    stop = total if max_steps is None else min(total, start_step + max_steps)
    if optimizer is None:
        optimizer = build_optimizer(model, handle, cfg)
    use_scaler = cfg.precision == "fp16" and device.startswith("cuda")
    if scaler is None and use_scaler:
        scaler = torch.amp.GradScaler("cuda")
    ckpt_dir = cfg.output_dir / "checkpoints"
    batch_blocks = cfg.micro_batch_size * cfg.grad_accum
    block = sampler.block_size
    history: list[dict] = []
    current_phase = None
    t0, tokens_since = time.time(), 0
    model.train()

    def write(row: dict) -> None:
        history.append(row)
        if log_path is not None:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with log_path.open("a", encoding="utf-8", newline="\n") as f:
                f.write(json.dumps(row) + "\n")

    for step in range(start_step, stop):
        phase, local, length = phase_at(step, cfg)
        if phase != current_phase:
            set_phase(model, handle, phase)
            current_phase = phase
        for group in optimizer.param_groups:
            peak = cfg.lr_new_rows if group["name"] == "new" else cfg.lr_lora
            group["lr"] = lr_at(
                local, length, peak, min(cfg.warmup_steps, length), cfg.min_lr_ratio
            )

        loss_sum = 0.0
        position = step * batch_blocks
        for micro in range(cfg.grad_accum):
            blocks = sampler.blocks_at(
                position + micro * cfg.micro_batch_size, cfg.micro_batch_size
            )
            x = torch.tensor(blocks, dtype=torch.long, device=device)
            with _autocast(cfg, device):
                loss = model(input_ids=x, labels=x).loss
            if not torch.isfinite(loss):
                raise TrainingDiverged(
                    f"loss is {loss.item()} at step {step} (phase {phase}). With fp16 this is "
                    "usually an overflow: set precision: fp32 and rerun the smoke test."
                )
            loss_sum += loss.item() / cfg.grad_accum
            scaled = loss / cfg.grad_accum
            (scaler.scale(scaled) if scaler is not None else scaled).backward()
        tokens_since += batch_blocks * block

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
                "phase": phase,
                "loss": loss_sum,
                "lr_new": optimizer.param_groups[0]["lr"],
                "grad_norm": grad_norm,
                "epoch": sampler.epoch_of(done * batch_blocks),
                "tokens_per_s": tokens_since / dt,
            }
            if device.startswith("cuda"):
                row["gpu_mem_gb"] = torch.cuda.max_memory_allocated() / 1e9
            write(row)
            log(
                f"step {done}/{total} {phase} loss {loss_sum:.4f} grad {grad_norm:.2f} "
                f"{row['tokens_per_s']:.0f} tok/s"
            )
            t0, tokens_since = time.time(), 0
        if eval_fn is not None and (done % cfg.eval_every == 0 or done == stop):
            model.eval()
            with _autocast(cfg, device):
                metrics = eval_fn(model)
            model.train()
            write({"step": done, "eval": metrics})
            log(f"step {done} eval {metrics}")
        if done % cfg.checkpoint_every == 0 or done == stop:
            path = save_checkpoint(ckpt_dir, done, model, handle, optimizer, scaler, cfg)
            prune_checkpoints(ckpt_dir, cfg.keep_checkpoints)
            if on_checkpoint is not None:
                on_checkpoint(path, done)

    return {"final_step": stop, "total_steps": total, "finished": stop == total, "history": history}
