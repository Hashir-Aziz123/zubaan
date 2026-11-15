# Stage 1 usage: measure the language tax

Stage 1 answers one question before you commit to a model: **what will my language cost me on this model, and will speculative decoding help?** It only measures the problem. Fixing it is Stage 2 onward.

Currently Urdu only. Examples use Qwen3, but any Hugging Face model works.

## The commands at a glance

| Command | What it does | Needs a model download? | Needs a GPU? |
| --- | --- | --- | --- |
| `zubaan data flores` | Downloads the same 1,012 sentences in Urdu and English | No | No |
| `zubaan audit fertility` | Counts tokens per Urdu word and the cost versus English for one tokenizer | Tokenizer files only (small) | No |
| `zubaan audit compare` | Same as above for several tokenizers, as one table | Tokenizer files only (small) | No |
| `zubaan audit acceptance` | Measures how often a small draft model's guesses are accepted by the big model | **Yes, both full models** | Recommended (Kaggle for the full run) |
| `zubaan report tax` | Turns the saved results into a Markdown report with charts | No | No |

## Setup

```bash
uv sync --group dev --extra audit --extra report
```

- The `audit` extra installs `transformers`, `tokenizers`, `datasets` and `torch`. Fertility, compare and acceptance need it.
- The `report` extra installs `matplotlib`. Only `report tax` needs it.
- The core install (no extras) is enough for `data flores`.

## 1. `zubaan data flores`: get test data

```bash
uv run zubaan data flores
```

**Function:** downloads FLORES-200 (devtest) and writes `data/flores_urd.txt` and `data/flores_eng.txt`, one sentence per line, line N in both files being the same sentence. Cost comparisons are only valid on parallel text like this.

**Requirements:**
- Internet. It downloads one archive of about 25 MB from Meta (no login needed).
- `--source hf` uses the Hugging Face copy instead. That one is gated: accept the dataset terms and log in first.

## 2. `zubaan audit fertility`: cost of one tokenizer

```bash
uv run zubaan audit fertility -t Qwen/Qwen3-1.7B -c data/flores_urd.txt -r data/flores_eng.txt
```

**Function:** for one tokenizer, reports tokens per word, characters per token, and the **cost multiplier** (Urdu tokens divided by English tokens for the same sentences). It also checks that the tokenizer is lossless, so a low count cannot hide dropped text.

**Result on Qwen3-1.7B:** 3.12 tokens per Urdu word against 1.24 for English, a 3.14x multiplier.

**Requirements:**
- Step 1 done (the two corpus files).
- Downloads **only the tokenizer files**, not the model weights. Runs on CPU in seconds.
- Useful flags: `--examples 3` shows how sentences split, `--out result.json` saves the result, `--json` prints it.

## 3. `zubaan audit compare`: shortlist models

```bash
uv run zubaan audit compare \
  -t Qwen/Qwen3-1.7B -t bigscience/bloom-560m -t mistralai/Mistral-7B-Instruct-v0.3 \
  -c data/flores_urd.txt -r data/flores_eng.txt
```

Or run the preset list: `uv run zubaan audit compare --config configs/audit-compare.yaml`.

**Function:** runs the fertility measurement for many tokenizers on identical data and prints one table sorted by cost multiplier. Writes `reports/compare.csv` and `reports/compare.json`.

**What it showed:** the same Urdu text costs between 1.35x (BLOOM) and 4.29x (Mistral 7B v0.3) as many tokens as English.

**Requirements:**
- Same as step 2: tokenizer files only, CPU, no GPU.
- **Gated models** (Gemma 3, Llama 3.2) need a Hugging Face login with the model's terms accepted. Without that they appear as `skipped` with the reason, and the other models still run.

## 4. `zubaan audit acceptance`: will a draft model help?

```bash
uv run zubaan audit acceptance -d Qwen/Qwen3-0.6B -t Qwen/Qwen3-1.7B \
  -c data/flores_urd.txt -r data/flores_eng.txt --max-texts 50
```

**Function:** speculative decoding uses a small draft model to guess tokens that the big model then verifies. This command feeds both models the same sentence and measures **alpha**, the probability that the draft's guess is accepted, for Urdu and English. It then models the speedup from alpha.

**Result on Qwen3-0.6B drafting for 1.7B:** alpha 0.573 in Urdu against 0.632 in English. Modelled speedup at 4 drafted tokens is 0.92x for Urdu (slower than not drafting) and 1.02x for English.

**Requirements:**
- **Downloads both full models.** Roughly 1.2 GB for the 0.6B and 3.4 GB for the 1.7B in bfloat16.
- The draft and target must have the **same vocabulary**. The command stops if they do not.
- **Local machine (8 GB RAM, no GPU):** works with `--dtype bfloat16 --device cpu` on a small sample (`--max-texts 50`, the default). `--mode sequential` loads one model at a time to save memory. Expect it to be slow.
- **Full 1,012-sentence run: use Kaggle.** We ran it on a T4 GPU in about 8 minutes using `notebooks/01_audit_acceptance_qwen3.ipynb`. On T4/P100 use `float32` or `float16`, never bfloat16 (those GPUs lack it).
- The speedup shown is **modelled, not measured**. Real measured speedup comes in Stage 5.

## 5. `zubaan report tax`: shareable report

```bash
uv run zubaan report tax
```

**Function:** reads the saved `reports/compare.json` and `reports/acceptance_qwen3.json` and writes a Markdown report with charts to `docs/reports/urdu-language-tax/`. Every number comes from the saved files, nothing is typed by hand.

**Requirements:**
- Run steps 3 and 4 first, so the JSON files exist. Without the acceptance file the acceptance section is left empty.
- Needs the `report` extra (matplotlib). No downloads, no GPU.

## Using your own models and data

Nothing is hard-coded to the models in the examples above. They are examples, and `configs/audit-compare.yaml` is only a preset list. You choose both the models and the data.

### Models

| Command | What you can pass | Limit |
| --- | --- | --- |
| `audit fertility`, `audit compare` | `-t` with any Hugging Face repo id or a local tokenizer folder. `compare` takes `-t` repeatedly. | Must load with `AutoTokenizer`. Gated repos need a login, otherwise they show as `skipped` and the others still run. |
| `audit acceptance` | `-d` for the draft and `-t` for the target, any causal language model. | The two must have identical vocabularies, and the command stops if they differ. In practice that means the same model family. |
| `report tax` | `--compare`, `--acceptance` and `--target-model` point at your own result files. | The model named in `--target-model` must appear in your compare results. |

Only the models in our own comparison (Qwen3, BLOOM, Mistral, Phi-4-mini, gpt-oss) have been run so far. Other models use the same code path, but an unusual tokenizer could surface an error.

### Data

- **Your own files.** `-c` takes any UTF-8 text file with one text per line, and `-r` takes the reference file. FLORES is not required.
- **Parallel data for cost.** The cost multiplier needs `-c` and `-r` to have the same number of lines, with line N being the same sentence in both. The command stops if the counts differ. It cannot check that the translations match, so that is your responsibility.
- **No parallel data.** `audit fertility` works without `-r`, but you get only tokens per word and characters per token, not the cost multiplier.
- **Other FLORES options.** `zubaan data flores` also accepts `--lang` and `--ref` codes, `--split dev` instead of `devtest`, and `--source hf` for the gated Hugging Face copy.
- **Config files.** Every audit command can be driven from a YAML file with `--config`, and flags override it. See `configs/` for examples.

Example with your own tokenizer and corpus:

```bash
uv run zubaan audit compare -t my-org/my-model -t Qwen/Qwen3-1.7B \
  -c my_urdu.txt -r my_english.txt --out-dir reports/mine
```

### What is still Urdu-specific

The commands take any models and any Urdu data, but not another language yet.

- Text normalization is Urdu only. Feeding it another language gives distorted counts.
- The `report tax` text and the `urd` and `eng` keys in the acceptance results are Urdu-specific.

Supporting another language needs a per-language normalizer and a generalized report. Neither is built.

## Recommended order for a developer

1. `data flores`, once.
2. `audit compare` on the models you are considering, to pick a tokenizer by cost.
3. `audit fertility --examples` on your pick, to see where words get split.
4. `audit acceptance` only if you plan to use speculative decoding. Do the full run on Kaggle.
5. `report tax` to share the result.

## Limits

- Stage 1 reports the problem and does not fix it. The fix starts in Stage 2 (`extend`, `export`).
- FLORES is formal, news-like text. Chat, Roman Urdu and code-mixed text are not covered.
- Word boundaries are rule-based, so tokens per word is approximate. The cost multiplier does not depend on word counts.
