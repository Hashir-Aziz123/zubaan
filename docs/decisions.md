# Decisions

Append new decisions at the bottom with the date and the reason.

## 2026-11-12: Project scope and name

zubaan is a developer tool, not an app: measure the language tax, then produce a deployable model with a matching drafter. Urdu is the flagship language; code stays language-agnostic where it is cheap to do so.

## 2026-11-12: Python package with optional extras

Core install is light (typer, rich, pydantic, pyyaml) so the CLI runs on CPU-only machines. Heavy libraries live in extras (`audit`, `train`, `bench`, `docs`) and are imported lazily.

## 2026-11-12: Tokenizer-fair metrics

Speed is reported as words per second and quality as bits per character. Tokens per second and per-token perplexity are not comparable across tokenizers.

## 2026-11-12: Default models

Qwen3-1.7B as the target and Qwen3-0.6B as the drafter. Sized for free Kaggle GPUs (T4/P100, 16 GB, no BF16) and an 8 GB RAM local machine.

## 2026-11-12: Extension via real merges

New tokens are added as BPE merges with the base pre-tokenizer unchanged, not as `added_tokens`. This keeps tokenization natural and makes GGUF export possible by mapping to the base pre-tokenizer type.

## 2026-11-13: Evaluation data is FLORES-200 devtest, from Meta's archive by default

All Stage 1 measurements use the 1,012 `devtest` sentences in `urd_Arab` and `eng_Latn`. The `dev` split stays unused so it can serve later as held-out data (for example imatrix calibration) without touching the test set. The default source is the original FLORES-200 archive from Meta: it needs no login, never changes (its SHA-256 is recorded in `data/flores_meta.json`), and is the version used by the papers we compare against. FLORES+ (`openlanguagedata/flores_plus`, gated) is available with `--source hf`; mixing sources silently would make results depend on who ran them, so `auto` is opt-in.
