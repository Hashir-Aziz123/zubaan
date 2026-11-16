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

## 2026-11-16: Stage 2 embeddings and quality evaluation design (chunks 6 to 8)

**Sizes and methods.** Evaluate 10,000 and 20,000 new tokens, each initialized three ways: mean of pieces (required), FOCUS-style, and random (control). 10k is the recommended size; 20k is included because it buys a lower cost (1.22x against 1.29x) and quality may or may not justify it.

**Where it runs.** Everything is validated locally on Qwen3-0.6B (fp32 on CPU). The 0.6B model is also the Stage 4 drafter, and it shares Qwen3-1.7B's tokenizer exactly (`same_tokenizer` checks vocabulary, merges, added tokens, normalizer, pre-tokenizer and decoder), so one set of merges serves both. The 1.7B numbers come from `notebooks/02_extend_quality_qwen3.ipynb` on a Kaggle T4, which saves after every variant and resumes if the session stops.

**Resizing.** New rows start at id 151,669, not at the padded matrix size 151,936. Rows below 151,669 are kept bit for bit (tested), and the rows between 151,669 and 151,936 that already existed as padding are overwritten. Qwen3-0.6B and 1.7B tie the input embedding and the output head, so one matrix is resized; the code also handles untied models by initializing the head from the head's own rows. The matrix is sized exactly to the new vocabulary, not rounded up.

**Mean of pieces** uses the base tokens each new token was merged from, found by expanding the merge tree, so it is exactly what the extended tokenizer replaces. Computed in float32.

**FOCUS is our own implementation, not `langsfer`.** Installing `langsfer` would downgrade `transformers` from 5.22 to 4.57, `numpy` from 2.4 to 1.26 and `tokenizers`, so it was not added. FOCUS here follows Dobler and de Melo (2023): new rows are a sparsemax-weighted sum of the old embeddings whose auxiliary vectors are most similar. The auxiliary vectors come from a positive-PMI co-occurrence matrix with a truncated SVD over the extended tokenization of the Urdu corpus, not from fastText. Results are labelled FOCUS-style. A token with no auxiliary vector falls back to mean of pieces and is counted.

**Random control** draws each new row from a normal with the per-dimension mean and standard deviation of the old rows, so the scale is right and the content is arbitrary.

**Quality metrics.** Bits per character (total cross-entropy over characters x ln 2) on three sets: held-out Urdu Wikipedia lines, FLORES Urdu, and FLORES English as a control. Every text gets the end-of-text token as one token of context so every real token is predicted. Texts are filtered by character count (at most 400), never by token count, so each tokenizer scores the same texts, and an over-long text raises instead of being silently cut. Belebele (`urd_Arab` and `eng_Latn`, 900 questions each) is scored zero-shot: the log-probability of each answer as a continuation of "P: passage Q: question A:". It is reported two ways: the plain sum (`acc`) and the sum divided by the answer's character count (`acc_char`). The sum favors tokenizers that need fewer tokens for an answer, so `acc_char` is the number to compare across tokenizers. Intervals are Wilson 95%; chance is 0.25.

**Expected outcome, stated before measuring.** The new rows are untrained, so Urdu quality should be no better than the base and may be worse. English bits per character can only get worse (the new rows add probability mass to the softmax and English hidden states are unchanged). A cost saving is never reported without these numbers next to it.

**Smoke run on Qwen3-0.6B (2026-11-16, fp32, CPU, 25 texts per set, 5 Belebele questions).** Not a result: samples are tiny and Belebele with 5 questions says nothing. It confirmed the pipeline on real weights and the expected direction. Urdu bits per character, held-out lines: base 2.80, mean of pieces 3.68, FOCUS-style 3.67, random 5.01. English FLORES: 1.273 base against 1.274 (mean), 1.275 (random), 1.277 (FOCUS-style), so English is almost untouched. Tokens per character on Urdu fell from 0.640 to 0.279. FOCUS-style had vector support for 8,952 of 10,000 new tokens (the rest fell back to mean) from only 20,000 corpus lines. Mean and FOCUS-style are indistinguishable at this sample size; both are far better than random and worse than the base, as expected for untrained rows. The 1.7B run on Kaggle with the full 1,000 texts per set and all 900 Belebele questions decides this. Local CPU cost: about 3 minutes per variant at this size, which is why the full run is on Kaggle.

