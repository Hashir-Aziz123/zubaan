"""Merge a stage 3 checkpoint into a plain Hugging Face model, save it and verify it.

Training keeps the new rows and the LoRA adapters outside the model's own weights. This module
puts them back: LoRA is folded into the weights (in float32, so small updates are not lost to
fp16 rounding), the trained rows are written into the tied embedding matrix, the model is cast to
the saved dtype and written next to the extended tokenizer files. The folder then looks exactly
like a Stage 2 extended model, with ``trained: true``, so the Stage 2 tools (GGUF export with the
parity check, Ollama, vLLM) work on it unchanged.

Checks before it reports success:

* the merged model gives the same logits as the model with adapters and hooks (a float32 merge
  must not change the function);
* the Stage 2 folder verification passes (only vocabulary and merges changed in the tokenizer,
  English tokenizes as before, Urdu round-trips, embedding rows match the vocabulary);
* the old embedding rows (every id below ``first_new_id``) are identical to the extended model.

A control run (base model, LoRA only) has no new rows: every embedding row must be identical to
the base, the tokenizer files are copied unchanged and checked, and the metadata goes to
``zubaan_adapt.json`` instead, so tools that look for an extended model never pick it up.
"""

from __future__ import annotations

import json
import shutil
from collections.abc import Callable
from pathlib import Path

from zubaan.adapt.cpt import latest_checkpoint, load_checkpoint, prepare_for_training
from zubaan.adapt.params import merge_lora, merge_new_rows
from zubaan.config import AdaptConfig
from zubaan.extend.build import METADATA_FILE, saved_embedding_rows, verify_extended_folder
from zubaan.extend.tokenizer_info import load_tokenizer_json
from zubaan.io import read_lines

WEIGHT_SUFFIXES = (".safetensors", ".bin", ".pt")
SKIP_COPY = {"config.json", "model.safetensors.index.json", METADATA_FILE}
CHAT_FILES = ("tokenizer_config.json", "special_tokens_map.json", "generation_config.json")
CONTROL_METADATA_FILE = "zubaan_adapt.json"


def copy_support_files(src: Path, dst: Path) -> list[str]:
    """Copy tokenizer and config files (never weights) that ``dst`` does not have yet."""
    copied = []
    for f in sorted(src.iterdir()):
        if not f.is_file() or f.suffix in WEIGHT_SUFFIXES or f.name in SKIP_COPY:
            continue
        if not (dst / f.name).exists():
            shutil.copy(f, dst / f.name)
            copied.append(f.name)
    return copied


def ensure_chat_files(folder: Path, base_model: str, log: Callable[[str], None]) -> list[str]:
    """Fetch missing chat template files from the base model (older Stage 2 exports lack them)."""
    fetched = []
    missing = [n for n in CHAT_FILES if not (folder / n).exists()]
    if not missing:
        return fetched
    from huggingface_hub import hf_hub_download

    for name in missing:
        try:
            shutil.copy(hf_hub_download(base_model, name), folder / name)
            fetched.append(name)
        except Exception as e:  # not every model has every file
            log(f"could not fetch {name} from {base_model}: {type(e).__name__}")
    return fetched


def merge_and_save(
    cfg: AdaptConfig,
    checkpoint: Path | None = None,
    out_dir: Path | None = None,
    base_tokenizer_json: dict | None = None,
    log: Callable[[str], None] = print,
    downloader: Callable[[Path], Path | None] | None = None,
) -> dict:
    """Merge the newest checkpoint (or ``checkpoint``) and save a verified model folder."""
    import torch
    from transformers import AutoModelForCausalLM

    ckpt_root = cfg.output_dir / "checkpoints"
    ckpt = checkpoint or latest_checkpoint(ckpt_root)
    if ckpt is None and downloader is None and cfg.hub_repo:
        from zubaan.adapt.pipeline import download_hub_checkpoint

        def downloader(directory: Path) -> Path | None:
            return download_hub_checkpoint(cfg.hub_repo, directory)

    if ckpt is None and downloader is not None:
        ckpt = downloader(ckpt_root)
    if ckpt is None:
        raise FileNotFoundError(f"no checkpoint in {ckpt_root}; run 'zubaan adapt run' first")
    out = out_dir or cfg.output_dir / "model"
    from zubaan.adapt.pipeline import model_info

    info, first_new_id = model_info(cfg)

    log(f"Merging {ckpt.name} into {cfg.model_path} (float32)")
    model = AutoModelForCausalLM.from_pretrained(cfg.model_path, dtype=torch.float32)
    peft_model, handle = prepare_for_training(model, first_new_id, cfg)
    n_old = handle.first_new_id  # every row in a control run
    original_rows = model.get_input_embeddings().weight[:n_old].detach().clone()
    step = load_checkpoint(ckpt, peft_model, handle, restore_state=False)
    peft_model.eval()

    if cfg.control:
        probe = torch.arange(1000, 1032).clamp(max=n_old - 1)[None]
    else:
        probe = torch.arange(first_new_id - 8, first_new_id + 24).clamp(
            max=first_new_id + handle.n_new - 1
        )[None]
    with torch.no_grad():
        before = peft_model(input_ids=probe).logits
    plain = merge_lora(peft_model)
    merge_new_rows(plain, handle)
    with torch.no_grad():
        after = plain(input_ids=probe).logits
    max_diff = float((before - after).abs().max())
    if max_diff > 1e-3:
        raise ValueError(f"merging changed the logits by {max_diff:.2e}; refusing to save")
    old_identical = bool(torch.equal(plain.get_input_embeddings().weight[:n_old], original_rows))
    if not old_identical:
        raise ValueError("old embedding rows changed during training; refusing to save")

    plain.config.use_cache = True
    plain = plain.to(getattr(torch, cfg.merged_dtype))
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    plain.save_pretrained(out, safe_serialization=True)
    copied = copy_support_files(cfg.model_path, out)
    if cfg.control:
        return _finish_control(cfg, out, ckpt, step, max_diff, copied, n_old, log)
    fetched = (
        ensure_chat_files(out, info.get("base_model", ""), log) if info.get("base_model") else []
    )

    base = base_tokenizer_json or load_tokenizer_json(info["base_model"])
    urdu = read_lines(cfg.guard_urdu[0])
    english = read_lines(cfg.guard_english[0])
    expected_rows = first_new_id + handle.n_new
    verified = verify_extended_folder(out, base, urdu, english, expected_rows=expected_rows)
    if saved_embedding_rows(out) != expected_rows:
        raise ValueError("saved embedding has the wrong number of rows")

    meta = {
        **info,
        "trained": True,
        "note": "New rows and LoRA trained in stage 3 and merged into the weights.",
        "stage3": {
            "checkpoint": ckpt.name,
            "step": step,
            "lora_rank": cfg.lora_rank,
            "lora_alpha": cfg.lora_alpha,
            "precision": cfg.precision,
            "english_replay_ratio": cfg.english_replay_ratio,
            "merge_max_logit_diff": max_diff,
            "saved_dtype": cfg.merged_dtype,
            "support_files_copied": copied,
            "chat_files_fetched": fetched,
        },
    }
    (out / METADATA_FILE).write_text(
        json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n"
    )
    log(f"Saved the merged model to {out} (step {step}, max logit difference {max_diff:.1e})")
    return {"model_dir": str(out), "step": step, "verified": verified, "metadata": meta}


def _finish_control(
    cfg: AdaptConfig,
    out: Path,
    ckpt: Path,
    step: int,
    max_diff: float,
    copied: list[str],
    rows: int,
    log: Callable[[str], None],
) -> dict:
    """Check and describe a merged control model: same tokenizer and row count as the base."""
    for name in ("tokenizer.json", "tokenizer_config.json"):
        src, dst = cfg.model_path / name, out / name
        if src.exists() and src.read_bytes() != dst.read_bytes():
            raise ValueError(f"{name} differs from the base model's; refusing to report success")
    if saved_embedding_rows(out) != rows:
        raise ValueError("saved embedding has a different number of rows than the base model")
    meta = {
        "control": True,
        "trained": True,
        "base_model_path": str(cfg.model_path),
        "note": "Control run (decision 6): base model and tokenizer, LoRA trained in stage 3 and "
        "merged into the weights. No new tokens.",
        "stage3": {
            "checkpoint": ckpt.name,
            "step": step,
            "lora_rank": cfg.lora_rank,
            "lora_alpha": cfg.lora_alpha,
            "precision": cfg.precision,
            "english_replay_ratio": cfg.english_replay_ratio,
            "merge_max_logit_diff": max_diff,
            "saved_dtype": cfg.merged_dtype,
            "support_files_copied": copied,
        },
    }
    (out / CONTROL_METADATA_FILE).write_text(
        json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n"
    )
    log(f"Saved the merged control model to {out} (step {step}, max logit diff {max_diff:.1e})")
    verified = {"tokenizer_identical_to_base": True, "embedding_rows": rows}
    return {"model_dir": str(out), "step": step, "verified": verified, "metadata": meta}
