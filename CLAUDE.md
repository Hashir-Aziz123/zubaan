# CLAUDE.md

Guidance for Claude Code when working in this repository. Read this first, then `docs/roadmap.md` and `docs/decisions.md`.

## What this project is

**zubaan** is an open-source CLI and Python library that measures and removes the inference tax non-English languages pay on LLMs. Urdu is the flagship language; the code must stay language-agnostic where possible.

The tax has two parts:
1. **Tokenizer fertility.** English-centric tokenizers split Urdu into several times more tokens per word, so Urdu costs more, generates slower, and fills the context faster.
2. **Speculative decoding disparity.** Small draft models are weak in low-resource languages, so acceptance rates and speedups are much lower than for English.

The pipeline, one CLI command per stage:

| Command | Purpose | Stage |
| --- | --- | --- |
| `zubaan data flores` | Download FLORES-200 parallel sentences into `data/` | 1 |
| `zubaan audit fertility` | Tokens per word, chars per token, cost vs a parallel reference corpus | 1 (done) |
| `zubaan audit compare` | Fertility for several tokenizers on the same data, CSV and JSON table | 1 |
| `zubaan audit acceptance` | Draft/target acceptance rate on a corpus | 1 |
| `zubaan report tax` / `report extend` | "The Urdu language tax" and "Tokenizer extension" reports: Markdown plus charts in `docs/reports/` | 1 to 4 |
| `zubaan extend` | `learn`, `sweep`, `quality`, `run`: add target-language merges, resize and initialize embeddings, save and verify the model | 2 (done; rows untrained) |
| `zubaan export` | `gguf` (with tokenizer parity check), `ollama`, `vllm`, `all` | 2 (done; 1.7B parity passed) |
| `zubaan adapt` | Continued pretraining so the model learns new tokens: new rows, then LoRA, English replay (Kaggle) | 3 (done; optional compute-matched control run pending) |
| `zubaan tune` | Restore chat behavior: chat check, generated conversations with thinking traces from a larger Qwen3, short SFT (Kaggle) | 4 (done; `docs/stage4-usage.md`) |
| `zubaan draft` | Drafters sharing the extended vocabulary: distilled small model, n-gram | 5 |
| `zubaan bench` | Benchmarks against running servers with tokenizer-fair metrics | 6 |

Goal of the project for its author: a public, credible demonstration of skill in model inference, serving, and local deployment. Correct measurements and honest reporting matter more than impressive numbers.

## Commands

```bash
uv sync --group dev                 # core + dev tools
uv sync --group dev --extra audit   # add transformers/tokenizers/datasets/torch for real audits
uv sync --group dev --extra audit --extra report   # plus matplotlib for zubaan report tax
uv sync --group dev --extra audit --extra report --extra export   # plus what llama.cpp's converter imports
uv run pytest                       # fast tests (slow and gpu markers skipped by default)
uv run pytest -m slow               # tests that download models or data
uv run ruff check . && uv run ruff format .
uv run zubaan --help
uv run zubaan audit fertility -t Qwen/Qwen3-1.7B -c data/flores_urd.txt -r data/flores_eng.txt
uv run mkdocs serve                 # needs --extra docs
```

## Layout

```
src/zubaan/
  cli.py        thin Typer CLI; no logic here beyond parsing, calling, printing
  config.py     Pydantic configs; every command must be drivable from configs/*.yaml
  metrics.py    words_per_second, bits_per_character (tokenizer-fair metrics)
  io.py         file helpers
  text/         Urdu normalization and word splitting
  audit/        stage 1: fertility.py, acceptance.py
  extend/       stage 2: tokenizer_info.py, train_bpe.py, selection.py, merge.py, sweep.py,
                init_emb.py, focus.py, quality_run.py, build.py (save and verify the model),
                pipeline.py (extend run)
  export/       stage 2: gguf.py (llama.cpp commands, fingerprint), parity.py (HF vs llama.cpp ids),
                ollama.py, vllm.py, pipeline.py (export all)
  adapt/        stage 3: data.py (token blocks), params.py (new rows + LoRA), cpt.py (loop),
                pipeline.py, finalize.py (merge, save, verify), evaluate.py (score model folders, checks)
  tune/         stage 4: client.py (OpenAI-compatible server client), generate.py, filters.py,
                chat_check.py (+ prompts/chat_check_v1.yaml), curate.py (filter, balance, review
                sample), data.py (render, mask, pack), evaluate.py (success checks),
                sft.py (LoRA SFT loop, merge)
  draft/        stage 5: distill.py, ngram.py (planned)
  bench/        stage 6: runner.py, quality.py (planned)
  report/       charts.py, tax.py, extension.py, adaptation.py (stage 3), tuning.py (stage 4):
                Markdown reports with PNG charts
  data/         flores.py: FLORES download; urdu.py: Urdu Wikipedia training corpus
tests/          pytest; fixtures/ holds tiny text samples; conftest.py has a fake llama.cpp folder
tools/          llama.cpp release b12477 (converter and binaries), gitignored
configs/        example YAML run configs
notebooks/      Kaggle notebooks for GPU steps; they pip install this package and call it
docs/           MkDocs site; roadmap.md and decisions.md are the source of truth for plans
data/, outputs/, reports/, tools/   local artifacts, gitignored
```

## Architecture rules

- **Library first, CLI thin.** All logic goes in importable functions with type hints. Notebooks and the CLI only call them. Never duplicate logic inside a notebook.
- **Heavy dependencies are optional and imported lazily** inside the functions that need them (`transformers`, `torch`, `datasets`, `openai`). The core install (`typer`, `rich`, `pydantic`, `pyyaml`) must work on a CPU-only machine with no extras. Add new heavy deps to the right extra in `pyproject.toml`, never to core.
- **Pure math stays dependency-free** (see `audit/acceptance.py`, `metrics.py`) so it is trivially testable.
- **Config-driven.** New commands get a Pydantic config model in `config.py` and accept `--config path.yaml`.
- **Results are data.** Commands return dataclasses or dicts with a `--json` output option; rendering is separate.

## Domain rules (do not violate)

- **Normalize Urdu before counting anything** (`text.normalize_urdu`). Unnormalized text skews every metric.
- **Never compare tokens/s or per-token perplexity across different tokenizers.** Use words per second and bits per character (`metrics.py`). This is a core claim of the project.
- **Cost comparisons need parallel corpora** (same sentences in both languages, e.g. FLORES-200 `urd_Arab` / `eng_Latn`).
- **Extend tokenizers with real BPE merges, not `added_tokens`.** Keep the base model's pre-tokenizer unchanged.
- **After extension, English tokenization must be byte-for-byte identical to the base** and Urdu must round-trip (`decode(encode(x)) == x`). Both are tested.
- **Drafter and target must share the exact same vocabulary.** A drafter for an extended model must use the extended tokenizer.
- **GGUF export must prove tokenizer parity**: tokenize a test set with Hugging Face and llama.cpp and assert identical ids; the export stops before quantizing if one differs. The converter picks the pre-tokenizer type from a fingerprint of the tokenizer. On llama.cpp `b12477` the extended Qwen3 tokenizer has the same fingerprint as the base (the converter's test string has no Arabic-script characters), so it is recognized as `qwen2` without a patch. If a tokenizer or release is not recognized, `export/gguf.py` maps the fingerprint to the base model's type, which is only valid because the pre-tokenizer is unchanged (it asserts that first). The parity test, not the absence of a warning, is the proof.
- **Calibrate quantization (imatrix) on target-language text**, not generic English.
- **Report quality next to speed** (bits per character, Belebele Urdu). A speedup with a hidden quality drop is not a result.
- **Check chat behavior after any training of a chat model** (answer rate with thinking on and off, reply language), against the base model. Stage 3 passed every likelihood check and still broke thinking mode.

## Hardware and environment constraints

- **Local machine: 8 GB RAM, no GPU.** Locally run only tokenizer-level work, tests, tiny models (under about 1B params), and llama.cpp CPU benchmarks of quantized models. Never write tests that load real models by default.
- **GPU work runs on Kaggle**: P100 or 2x T4, 16 GB each, roughly 30 GPU hours per week, 12-hour sessions.
  - T4 and P100 have **no BF16, no FP8, no FlashAttention 2**: use `fp16=True` and `dtype=half` in vLLM.
  - Use one GPU per training job unless multi-GPU is set up deliberately; the two T4s are linked over PCIe.
  - Push checkpoints to the Hugging Face Hub during training (`hub_strategy="checkpoint"`) and resume with `resume_from_checkpoint=True`.
  - Do quick 20 to 50 step test runs before long jobs; GPU hours are scarce.
- Target models: Qwen3 family (0.6B drafter, 1.7B target as default; 4B only if budget allows).

## Testing conventions

- Default tests run offline on CPU in seconds. Use the fake `CharTokenizer` in `tests/conftest.py` or tiny fixtures.
- Mark tests that download models or data `@pytest.mark.slow`; tests needing CUDA `@pytest.mark.gpu`.
- Every formula in the code has a test reproducing a worked example from the docs.
- Run `uv run pytest` and `uv run ruff check .` before declaring any task done.

## Code style

- Python 3.10+, type hints everywhere, `from __future__ import annotations`.
- Ruff for lint and format (line length 100). Small, single-purpose functions with docstrings that state what a formula means.
- Raise `ValueError` with a clear message on invalid inputs; do not silently return zeros.

## Writing style (README, docs, reports, docstrings)

- No em dashes. Plain, direct, technically precise language; avoid AI-sounding filler.
- For math: state the general formula first, then substitute numbers, with a short note on what each step does.
- Say what was measured, on what hardware, with what versions. Note limitations honestly.

## Git and workflow

- Conventional commits (`feat:`, `fix:`, `test:`, `docs:`, `refactor:`, `chore:`). Small, focused commits.
- Do not commit anything in `data/`, `outputs/`, `reports/`, model weights, or tokens/secrets.
- Do not push to the Hugging Face Hub, publish to PyPI, or open upstream PRs without asking first.
- When a design decision is made, append it to `docs/decisions.md` with date and reason.
- When a stage's scope changes, update `docs/roadmap.md` and the stage table above.

## Key references

- Sandler et al. 2025, *The Disparate Impacts of Speculative Decoding* (arXiv 2510.02128): low-resource languages get lower acceptance and speedup.
- Paudel et al. 2026, *Speculative Decoding Across Languages* (arXiv 2605.30580): language-specific drafter distillation; n-gram drafters give consistent speedups.
- Yamaguchi et al. 2025, *How can we effectively expand the vocabulary of LLMs with 0.01GB of target language text?* (Computational Linguistics): heuristic embedding init beats random in low-resource settings. Code: github.com/gucci-j/lowres-cve
- `langsfer` (PyPI): embedding initialization methods (FOCUS, WECHSEL, CLP).
- llama.cpp `convert_hf_to_gguf.py` / `convert_hf_to_gguf_update.py`: pre-tokenizer fingerprinting.
