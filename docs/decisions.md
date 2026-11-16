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

## 2026-11-13: Every audit result carries sanity checks and provenance

A low token count is only good news if no text was lost, so fertility results include unknown-token counts, a decode(encode(x)) round-trip check, and UTF-8 bytes per token. Saved results (`--out`) include the tokenizer commit hash, package versions, platform and the FLORES source, so any number in a report can be traced and reproduced.

## 2026-11-13: Tokenizers in the comparison

`configs/audit-compare.yaml` compares Qwen3 (our target), Gemma 3, Llama 3.2, Mistral 7B v0.3 (32k SentencePiece), Mistral Nemo (Tekken), Phi-4-mini, gpt-oss and BLOOM (trained with Urdu, used as an "Urdu-aware" reference). Only tokenizer files are downloaded, and the tokenizer is shared across sizes within a family, so the smallest repo of each family is used. A model that cannot be loaded (gated, missing) becomes a `skipped` row with the reason; the run continues. A tokenizer with unknown tokens or failed round trips is marked `lossy`, because its low count would hide lost text. transformers warns that the Mistral Nemo tokenizer has an incorrect regex; loading it with `fix_mistral_regex=True` gave identical counts on FLORES devtest, so the default loader is kept.

## 2026-11-13: How acceptance is measured

`zubaan audit acceptance` feeds both models the same FLORES sentence (teacher forcing) and computes alpha = sum(min(p_target, p_draft)) at every position, which is the exact acceptance probability of speculative sampling. The pure-Python `acceptance_rate()` stays the reference; model code uses a vectorized `torch.minimum(p, q).sum(-1)` that a test checks against it. We also report the greedy top-1 match rate (acceptance under greedy decoding). The mean is pooled over all positions, with a 95% bootstrap interval that resamples whole sentences, because positions inside one sentence are not independent. If Urdu and English intervals overlap, the report says "no clear difference".

Models used: `Qwen/Qwen3-0.6B` drafting for `Qwen/Qwen3-1.7B` (the post-trained models people serve), raw normalized text, no chat template. The two tokenizers must have identical vocabularies or the command stops. Softmax always runs in float32. Locally (8 GB RAM, no GPU) models run in bfloat16, since float32 for 1.7B needs about 8 GB; `--mode sequential` holds one model at a time. The full 1,012-sentence run uses float32 on a Kaggle T4 (`notebooks/01_audit_acceptance_qwen3.ipynb`), since T4 has no bfloat16 and float16 risks overflow.

Speedup is modelled, not measured: speedup = E / (1 + gamma x c), with c estimated as draft parameters / target parameters (token generation mostly reads every weight once). torch is added to the `audit` extra (CPU build on Windows by default).

## 2026-11-16: Report the best draft length, not a fixed gamma = 4

The first report used gamma = 4 drafted tokens for every language. With a draft costing about 35% of the target, that understates the achievable speedup, especially for the language with the lower alpha, where shorter drafts win. The acceptance command now sweeps gamma = 1 to 6 and the report shows the best gamma and its modelled speedup per language (`best_speedup` in `audit/acceptance.py`). The speedup is still modelled, not measured.

## 2026-11-13: Reports live in docs/

`reports/` is gitignored scratch output. `zubaan report tax` writes the published report and its PNG charts to `docs/reports/urdu-language-tax/`, which is committed and shown on the MkDocs site. Every number in it is read from the saved JSON results, never typed by hand. Charts use a validated categorical palette with Urdu always in slot 1 (blue) and English in slot 2 (orange), bars starting at zero, and direct value labels. matplotlib lives in a new `report` extra.

## 2026-11-16: Stage 2 tokenizer extension design

**Corpus.** Urdu Wikipedia (`wikimedia/wikipedia`, `20231101.ur`, CC BY-SA 3.0 and GFDL), cleaned with `normalize_urdu`, lines under 5 words or under 70% Arabic-block characters dropped, exact duplicates dropped, and any line that shares 8 consecutive words with a FLORES devtest sentence dropped (16 lines in the full file, 2 within the 100 MB cap). 1% of lines go to a held-out file by a hash of the line. Training text is capped at 100 MB. CC-100 is only added if the cost-versus-size curve is still dropping, which it is not (see below).

**Merges are learned on top of the base tokenization ("continued BPE").** A merge appended to the base merge list only sees text the base has already split as far as it can. Merges learned from raw bytes can name pairs that never occur, so they would never fire. Instead every word is first split by the base BPE, then the most frequent adjacent pair of those pieces is merged, repeatedly. Each merge is therefore applied exactly as learned, and a test replays the merges by hand and checks the tokenizer gives the same pieces. The Hugging Face `BpeTrainer` is not used. One long run serves every vocabulary size, because the first k merges of a run equal a run asked for k (tested).

**English is unchanged by construction, not by luck.** A new merge is only allowed if the merged token contains a UTF-8 lead byte 0xD8 to 0xDB. Those four bytes occur only in characters U+0600 to U+06FF (checked for every Unicode code point in a test), so a new token can only appear in text that has an Arabic-block character. Any text without one tokenizes exactly as before. Tests cover FLORES English, 4,500 random strings over Latin, Cyrillic, Hebrew, Devanagari, CJK and emoji, and a mutation check that sneaks in an English merge and confirms the test fails. The rule is stricter than needed (Arabic Supplement, presentation forms and ZWNJ-only merges are not allowed) and can be relaxed later.

**Special token ids must be pinned in `model.vocab`.** The `tokenizers` library ignores the stored id of an added token that is not in `model.vocab` and assigns it `len(vocab) + position`. Appending 100 merges to Qwen3 moved `<|im_end|>` from 151645 to 151745 with a naive append. The extended `tokenizer.json` therefore also lists the 26 added token strings in `model.vocab` at their original ids. New Urdu ids start at 151669, the first id used by neither the vocabulary nor an added token, not at the padded embedding size 151,936.

**Training text is normalized, evaluation is reported both ways.** Merges are learned from normalized text, matching the audit. The sweep reports cost on normalized FLORES Urdu and on raw FLORES Urdu, and the two agree within 0.02x, so skipping normalization at inference costs almost nothing.

**Vocabulary sizes.** Sweep of 5k, 10k, 20k and 32k new tokens. See the Stage 2 results in the roadmap. Choice for Stage 3 is a separate decision, recorded when made.

