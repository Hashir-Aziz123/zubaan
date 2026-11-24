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


## 2026-11-17: Stage 2 results and export design (chunks 9 to 12)

**Qwen3-1.7B quality results (Kaggle T4, float32, 1,000 texts per set, 900 Belebele questions per language).** Bits per character on held-out Urdu: base 2.31; 10k new tokens 3.57 (mean of pieces), 3.63 (FOCUS-style), 5.27 (random); 20k new tokens 3.63, 3.60, 5.15. English FLORES moves from 1.208 to at most 1.214. Belebele Urdu (`acc_char`) is 33.4% for the base and 24.9% to 27.0% for every extended variant, against 25% chance. Mean of pieces and FOCUS-style are within 0.1 bits per character of each other and neither is consistently better; random is clearly worst. The numbers and charts are in `docs/reports/tokenizer-extension/`.

**Default for stage 3: 10,000 new tokens, mean of pieces.** 20k buys 1.22x against 1.29x cost but adds 10,000 more rows to train, and quality is no better. Mean of pieces is the simpler method and FOCUS-style gave no consistent gain here. FOCUS stays available (`init_method: focus`). Revisit after stage 3: untrained rows are a poor guide to how well training will go.

**The converter fingerprint did not change.** `CLAUDE.md` and the stage 2 plan expected "BPE pre-tokenizer was not recognized". On llama.cpp `b12477` the converter's test string (`chktxt`) has no Arabic-script characters, and every new merge must contain an Arabic-block byte, so the extended tokenizer produces the same ids and the same hash as the base: `e636dc30a262dcc0d8c323492e32ae2b70728f4df7dfe9737d9f920a282b8aea`, which the converter maps to `qwen2`. Checked on Qwen3-1.7B's tokenizer against the 10k-token extension. The consequence cuts both ways: no patch is needed, but the converter's own check says nothing about the Urdu merges, so the HF vs llama.cpp parity test is the only proof. The code still handles the unknown-hash case (`plan_pretokenizer`, `patch_converter_source`) because a different base model or a future release can differ. Mapping the hash to the base type is only allowed after `assert_same_pretokenizer` passes.

**llama.cpp is pinned and not a dependency.** Release `b12477` (2026-11-17), source tree for `convert_hf_to_gguf.py` (in this release the fingerprint code lives in `conversion/base.py`, so `find_pretokenizer_source` searches both) plus the prebuilt `llama-quantize`, `llama-imatrix` and `llama-tokenize`. It lives in `tools/` (gitignored). The converter's Python imports go into a new `export` extra.

**Parity method.** `llama-tokenize -f <file> --ids --no-bos --no-escape --log-disable`, once per text, with the text written byte for byte (the tool's `-p` processes escape sequences, `-f` with `--no-escape` does not). Compared with `AutoTokenizer.encode(add_special_tokens=False)` on the saved model folder. Texts: all FLORES Urdu and English lines plus a stress set. The export stops before quantizing if one id differs.

**Importance matrix** is built from Urdu Wikipedia training text (200,000 characters, 40 chunks of 512 by default), per the project rule. Quantization runs with it for every type.

**Ollama and vLLM.** The Modelfile uses Qwen's ChatML template (no tool calls) and stops on `<|im_end|>` and `<|im_start|>`; `check_stop_tokens` requires both in the extended vocabulary at their base ids. The vLLM config is `dtype: half` plus `--tokenizer` pointing at the model folder; a check refuses any text containing `bfloat16` or `bf16`. Neither has been run against the real servers yet (see the roadmap).

**Commands.** `zubaan extend run` (learn if needed, build, save, verify) and `zubaan export gguf|ollama|vllm|all` are thin wrappers over `extend/pipeline.py` and `export/pipeline.py`, config driven, with `--json`. Models are saved in bfloat16 (Qwen3's dtype); embedding math is float32. `zubaan_extension.json` in the saved folder records `trained: false`.

**Local runs use Qwen3-0.6B.** It shares Qwen3-1.7B's tokenizer exactly, so the merges file is the same. The 8 GB laptop built and verified the 0.6B model in about 40 seconds.

## 2026-11-17: First real parity run found an NFC gap in llama.cpp

Converting the extended Qwen3-0.6B (f16) and comparing 2,045 texts (FLORES Urdu and English plus the stress set) gave identical ids on 2,041. The other 4 are FLORES Urdu lines that contain stacked diacritics in non-NFC order (shadda U+0651, combining class 33, before damma U+064F, class 31). In each, the same two ids are swapped, 64604 and 73771, and the lengths are equal. Qwen's tokenizer.json has an `NFC` normalizer, which reorders the marks; llama.cpp's BPE tokenizer has no normalizer. Checked without running llama.cpp: Hugging Face with the normalizer removed reproduces llama.cpp's ids at the mismatch, and FLORES Urdu has exactly 4 non-NFC lines.

**Decision.** A parity mismatch is excused only if the text is not NFC and llama.cpp, given the NFC form, gives Hugging Face's exact ids. It is recorded as `n_nfc_only`, never as a pass for the raw text, and every other mismatch still stops the export (`check_parity`, tested including the cases that must still fail). Not a bug in the extension: the vocabulary and merges agree, and the same gap would exist for the base model's GGUF. That last point is not tested on a base GGUF yet. Practical rule: normalize Urdu to NFC before sending it to a GGUF model, which `normalize_urdu` already does.

Also fixed on the way: the converter imports `sentencepiece` for Qwen before it checks for `tokenizer.model`, so `sentencepiece` joined the `export` extra.

## 2026-11-17: Stage 2 checks on the exported 0.6B, and the 1.7B notebook

**Generation.** `llama-completion` on the Q4_K_M (4 CPU threads): the prompt "the capital of Pakistan" in Urdu is 3 tokens with the extended tokenizer and 13 with the base one; generation ran at 34 tokens per second. The text is nonsense and drifts into Bengali, which is what untrained rows should give. `llama-cli` in this release opens an interactive chat, so the docs use `llama-completion`.

**Ollama.** Ollama 0.40.16 (portable build, model store inside `tools/`) accepted the Modelfile with `ollama create` and answered a chat request through the ChatML template: 14 prompt tokens, both stop parameters registered. The reply was an English `<think>` block that misread the Urdu question, so the quality loss is visible in practice too. The Modelfile does not suppress `<think>`; revisit when stage 3 gives a model worth serving.

**1.7B is built by a notebook, on CPU.** `notebooks/03_extend_export_qwen3.ipynb` runs `extend learn`, `extend run` and `export gguf|ollama|vllm` for Qwen3-1.7B. It needs no GPU (about 12 GB RAM, 20 GB disk, about 1 hour) and the 8 GB laptop cannot hold the 1.7B. It installs zubaan from one pinned commit, checks the installed commit, the machine and the llama.cpp tools before the long steps, asserts that parity passed, deletes the f16 intermediate to stay under Kaggle's 20 GB output limit, and keeps the extended Hugging Face model folder because stage 3 starts from it. The vLLM check is an optional cell that is off by default. Not yet run on Kaggle: only its commands were exercised locally, on the 0.6B.

## 2026-11-18: Extended model folders must carry the chat template

The first vLLM run on a Kaggle T4 (vLLM 0.34.0, `dtype half`) started, tokenized 5 of 5 sentences exactly like the extended `tokenizer.json`, and completed text, but `/v1/chat/completions` returned HTTP 400. Cause: `find_local_files` downloaded only `config.json` and `tokenizer.json` from the Hub, so `tokenizer_config.json` (which holds Qwen's `chat_template`) was never copied into the saved model. It now also fetches `tokenizer_config.json`, `generation_config.json` and `special_tokens_map.json` when they exist. Notebook 04 repairs folders exported before the fix by linking the weights and fetching the two tokenizer config files from the base model, so the 1.7B export from notebook 03 does not need to be rerun. The GGUF files are unaffected (the Modelfile has its own template).

Result of the rerun (2026-11-18): `vllm_check.json` passed on a Tesla T4 (compute capability 7.5), vLLM 0.34.0, `dtype half`, 5 of 5 tokenizations identical, completion and chat both returned text. vLLM serving of the extended model is verified.

## 2026-11-18: Stage 3 (adapt) design decisions

All defaults from the stage 3 plan in `continue.md` were accepted. Values marked "start" are starting points to be checked in the smoke run, not tuned numbers.

**Starting model.** The existing extended `Qwen/Qwen3-1.7B` (the post-trained chat model), not `Qwen3-1.7B-Base`. Reason: no rework, and the Stage 2 exports, Ollama and vLLM setups, and the Stage 4 and 5 plans all target it. Cost: training on raw Urdu text will weaken instruction following and `<think>` behavior. The report must say so. Our metrics (bits per character, Belebele log-likelihood scoring, speed, acceptance) do not need chat ability.

**What is trainable.** Two phases. Phase A trains only the 10,000 new embedding rows (the head rows are the same tensor, since the model ties them). Phase B adds LoRA and keeps training the new rows. Old embedding rows stay frozen, implemented as a separate trainable tensor for the new rows rather than a gradient mask, so optimizer state is not allocated for the full matrix. Reason: fits a 16 GB T4 (about 20.5M embedding parameters), and English cannot drift through the embeddings.

**LoRA (start).** Rank 64, alpha 128, dropout 0.05, all attention and MLP linear layers. Learning rate about 2e-4 for LoRA and about 1e-3 for the new rows.

**Data.** Urdu: Wikipedia only (`data/urdu_train.txt`), about 14M tokens (estimate, to be measured). Add a larger source only if held-out Urdu bits per character is still falling when the data runs out. English replay: English Wikipedia, 10% of blocks. Documents are joined with `<|endoftext|>`, matching how the quality harness frames texts.

**Control experiment.** Base Qwen3-1.7B with the base tokenizer, same recipe, matched on compute (same number of tokens), run after the main run if GPU hours allow. Matching on text would cost 8 to 17 GPU hours; matching on compute about 4 to 7. It shows whether gains come from the new tokens or from training on more Urdu.

**Precision.** fp16 autocast with a gradient scaler. The T4 has no BF16. Fall back to fp32 base weights only if the smoke run shows NaN or inf.

**Hugging Face Hub.** A private repo for training checkpoints, token stored as a Kaggle secret. Approved by the user on 2026-11-18 for checkpoints only. Nothing is made public without a separate decision.

**Success checks.** English FLORES bits per character at most 0.02 above the base 1.208. Held-out Urdu bits per character below the untrained 3.57 (must) and at or below the base 2.31 (stretch). Belebele Urdu `acc_char` above the base 33.4%, reported with its Wilson interval (about plus or minus 3 points at 900 questions, so a gain under about 4 points is not distinguishable from noise). GPU budget for the main run: about 10 hours.

**Defaults taken without a question.** Block size 1024, one T4, gradient checkpointing on, plain `transformers` loop instead of `trl`, notebook number 05.

## 2026-11-18: Stage 3 implementation choices (chunks 1 to 3)

**New rows live outside the model.** `adapt/params.py` keeps the trained rows in their own float32 parameter (`AdaptHandle.new_rows`, 10,000 x 2048 for Qwen3-1.7B) and applies them with two forward hooks: the input embedding output is replaced at ids at or above `first_new_id`, and the last `N` columns of the head's logits are replaced by `hidden @ new_rows^T`. The model's own tensors and state_dict are never touched in training, so old rows stay bit for bit identical (tested), no optimizer state is allocated for the other 310M embedding values, and the hooks sidestep peft's handling of tied weights. `merge_new_rows` writes the rows back into the tied matrix and removes the hooks, so the saved folder is an ordinary model. Models with a separate output head are refused (Qwen3-0.6B and 1.7B tie it).

**One optimizer, two phases.** A single AdamW holds a group for the new rows and one for LoRA. LoRA tensors have no gradient in phase A and AdamW skips them, so the new rows keep their moments across the switch. Each phase has its own warmup and cosine schedule (floor 10% of the peak). Default length: 150 steps of phase A and 800 of phase B; each step is 2 x 16 blocks of 1,024 tokens, about 33k tokens, so 950 steps is about 2 epochs if the corpus is about 14M tokens. To be set from the measured token count and speed.

**Data layout.** Each corpus line is one document, written as `<|endoftext|>` plus its tokens, packed end to end into blocks of 1,024 ids; the partial last block is dropped and counted. Urdu and English blocks are separate files (raw little-endian uint32). Each epoch contains every Urdu block once plus English blocks drawn without replacement so that English is 10% of all blocks; epoch `e` shuffles with seed + `e`. The position in the stream is a pure function of the step number, which is what makes resume exact. `blocks_meta.json` records the tokenizer's sha256, and training refuses blocks made with a different tokenizer.

**Contamination guard widened.** The Stage 2 corpus builder checked only FLORES devtest. Belebele passages are FLORES-200 text, so `zubaan adapt prepare` also checks the FLORES dev split and the Belebele passages (Urdu for the Urdu lines, English for the English replay), by the same 8-consecutive-word test. Offending lines are dropped and counted in the metadata (`drop_overlaps`), and the remainder is checked again; a surviving overlap raises.

**Resume.** A checkpoint holds the step, the new rows, the LoRA weights, the optimizer, the gradient scaler and the random state, and is written to a temporary folder and renamed. A test runs 10 steps straight and as 6 plus a resumed 4 and requires identical new rows and LoRA weights (bit for bit on CPU). The newest checkpoint is pushed to `<repo>/last` in the private Hub repo, and a run with no local checkpoint resumes from there. Nothing is pushed unless `hub_repo` is set.

**Guards.** The loop stops with `TrainingDiverged` as soon as a loss is not finite, with a message to switch to fp32. In-training evaluation reports held-out Urdu bits per character and an English FLORES control with the same harness as the Stage 2 quality run (same one-token end-of-text context).

**Dependencies.** `trl` was removed from the `train` extra (own loop). `peft` 0.21 and `accelerate` 1.15 installed next to the pinned `transformers` 5.22 without changing it.

**Measured corpus size (2026-11-18, extended tokenizer, block size 1,024).** Urdu: 295,858 documents, 16,053,605 tokens, 15,677 blocks (357 tokens dropped). English replay: 21,674 paragraphs (12 MB), 2,676,343 tokens, 2,613 blocks. The wider guard (FLORES dev and devtest, Belebele passages) dropped 47 Urdu lines that the Stage 2 devtest-only check had let through, and no English lines. One epoch is 15,677 Urdu blocks plus 1,742 English blocks (10%), 17,419 blocks, 544 steps of 32 blocks. The default 950 steps is therefore about 1.75 epochs. The 14M-token estimate in the plan was low by about 15%.

## 2026-11-18: Merge, resume safety and the notebook (chunks 0, 4 and 6)

**Merge in float32, then cast.** `zubaan adapt merge` loads the extended model in float32, restores the checkpoint, folds LoRA into the weights, writes the trained rows into the tied matrix, and only then casts to bfloat16 (Qwen3's dtype). It refuses to save if the merged logits differ from the adapter-and-hook model by more than 1e-3 on a probe input, or if any old embedding row differs from the extended model. The saved folder then passes the Stage 2 verification (tokenizer diff, English identical, Urdu round-trip, embedding rows) and carries the chat template files, fetched from the base model if the extended export lacked them. `zubaan_extension.json` gets `trained: true` and the stage 3 details, so the Stage 2 export tools work on it unchanged.

**Resume refuses a different run.** A checkpoint stores its config. Resuming with a different phase length, batch shape, seed, block size, replay ratio, LoRA shape, learning rate or precision raises, because the step number selects the data and the learning rate and a changed field would silently continue a different run. Logging and checkpoint cadence may change.

**Notebook 05 has three modes and keeps smoke and main apart.** `check` (versions, GPU, five fp16 steps over both phases), `smoke` (a 30-step schedule stopped at step 20, local checkpoints deleted, resumed from the Hub, then speed, peak memory and a projection of the main run's hours), `main` (950 steps, then merge). Smoke runs use `<repo>-smoke` as the Hub repo, so a main run can never resume from a smoke checkpoint. The notebook rebuilds the Urdu corpus, FLORES and the blocks on Kaggle (about 10 minutes) instead of depending on files from notebook 03.

## Stage 3: checkpoint-first main run and speed trial (2026-11-18)

The smoke run on a T4 gave a median 582 tokens/s at micro batch 2 with gradient checkpointing, which projects to 14.9 h for the planned 950 steps: over the 10 h cap and the 12 h session limit. Peak memory was 9.29 GB of 15, so there is room to trade memory for speed.

- **Speed trial before the main run.** Notebook mode `speed` runs 8 real steps under four settings (micro batch 2, 4 with checkpointing; 1, 2 without), each in its own process so an out-of-memory error cannot affect the others, and names the fastest that fits. Each step always sees 32 blocks (`grad_accum = 32 / micro_batch`), so the optimization is the same whichever setting wins.
- **Train and watch instead of a fixed hour budget.** The main run gets a generous schedule (default 100 + 700 steps, set from the speed trial), a Hub checkpoint every 50 steps, an eval every 100, and a permanent copy of every 200th checkpoint under `<repo>/milestones/step-NNNNNNN` (`milestone_every`). Any milestone can be merged and evaluated with `zubaan adapt merge --milestone STEP`.
- **Why not "stop whenever".** The cosine learning rate is tied to the total step count, and resume refuses a changed schedule. A run stopped early has a learning rate that has not decayed, so its weights are usable but noisier than an annealed run. Milestones are the fallback, not the plan.
- **A failed Hub upload warns and training continues.** The local checkpoint exists and the next upload retries; losing a 10 h run to a network error is worse.

## Stage 3: speed trial and main run completed (2026-11-19)

- **Speed trial result (notebook 06, T4).** Micro batch 2 with gradient checkpointing: 592 tokens/s, 9.28 GB. Micro batch 4 with checkpointing, and micro batch 2 without: out of memory. Micro batch 1 without checkpointing: 850 tokens/s, 13.26 GB, 44% faster, so it became the main-run setting (`MICRO_BATCH = 1`, `GRAD_CHECKPOINT = False`). Each step still sees 32 blocks.
- **Schedule cut from 950 to 800 steps** (100 new-rows-only plus 700 LoRA and rows), about 1.47 passes over the 16.05M Urdu tokens, to fit a 10 h budget.
- **Main run finished (notebook 07, T4).** All 800 steps ran and the notebook wrote `adapted/checkpoints/ckpt-0000800` and the merged `adapted/model/model.safetensors` (3.48 GB). Wall time of the whole notebook was about 33,800 s (9.4 h), inside the 10 h cap and the 12 h session limit.
- **Training curve (in-run eval: 200 held-out Urdu texts and only the first 100 FLORES English lines, same texts every time).** Held-out Urdu bits per character: 2.413 (step 100, end of phase A), 1.799, 1.679, 1.616, 1.569, 1.547, 1.534, 1.523 (step 800). FLORES English: 1.213, 1.099, 1.094, 1.085, 1.086, 1.084, 1.082, 1.080. Urdu was still falling slowly at the end (gain of 0.011 over the last 100 steps, 0.014 over the 100 before). Training loss went from 6.94 (step 10) to about 2.6 to 2.8 (steps 700 to 800). Gradient norms settled at about 0.5. No NaN, no loss-scaler trouble, no Hub upload warnings. Throughput 817 to 840 tokens/s (850 in the trial), about 9.4 h wall time for the whole notebook.
- **Do not compare this curve with the Stage 2 numbers yet.** The in-run eval uses 200 texts; the Stage 2 table (Urdu 2.31 base, 3.57 untrained, English 1.208) used a different sample. English falling from 1.213 to 1.080 while training mostly on Urdu is larger than expected and is probably a sample effect or domain adaptation to English Wikipedia, so it is not evidence that English improved. The chunk 7 evaluation scores base, untrained and adapted on the same texts.
- **Merge and verify passed.** Merged in float32 into `adapted/model` (step 800): max logit difference between merged and unmerged model 6.5e-05, "Saved and verified". The merged weights are saved as bfloat16 (`merged_dtype`), which is fine for storage, but T4 serving and evaluation must load them as float16. A harmless warning about `special_tokens_map.json` from `Qwen/Qwen3-1.7B` appeared during save.

## Stage 3: evaluation design (2026-11-19)

- **One protocol, three models.** `zubaan adapt eval` (`adapt/evaluate.py`, config section `evaluate`, notebook 08) scores the base model, the extended model before training and the adapted model on the same texts with their own tokenizers: 1000 texts of at most 400 characters per set (held-out Urdu, FLORES Urdu, FLORES English), all 900 Belebele questions per language, float32. The base model is rerun instead of copied from Stage 2 and must reproduce the Stage 2 numbers within 0.01 bits per character, or the notebook stops: that proves the protocol did not drift. The held-out file's hash is checked against the one Stage 2 used.
- **Success checks are code, not prose.** `success_checks` returns PASS or FAIL for the four tests of decision 9, and says when a Belebele gain has overlapping 95% intervals (then it is reported as noise, not a gain). The English tolerance edge is inclusive with a 1e-9 float margin (a test found 1.228 - 1.208 evaluating to slightly above 0.02).

## Stage 3: evaluation results (2026-11-19, notebook 08, Kaggle T4, float32, commit 8d16b47)

Same texts for all three models (1000 per set, at most 400 characters; Belebele 900 questions per language). The base model reproduced its Stage 2 numbers exactly (2.310, 2.339, 1.208), and the held-out file matched Stage 2's hash.

| Model | Tokens per char (Urdu) | BPC Urdu held-out | BPC Urdu FLORES | BPC English FLORES | Belebele Urdu acc_char | Belebele English acc_char |
| --- | --- | --- | --- | --- | --- | --- |
| Base Qwen3-1.7B | 0.643 | 2.310 | 2.339 | 1.208 | 0.334 [0.304, 0.366] | 0.447 [0.414, 0.479] |
| Extended, untrained | 0.281 | 3.570 | 3.562 | 1.210 | 0.253 [0.226, 0.283] | 0.447 [0.414, 0.479] |
| Extended, adapted (800 steps) | 0.281 | 1.486 | 1.718 | 1.066 | 0.340 [0.310, 0.372] | 0.459 [0.427, 0.492] |

Checks (decision 9): English within tolerance PASS (1.066 vs 1.208); Urdu below untrained PASS; Urdu at or below base PASS (stretch); Belebele Urdu above base is numerically above (0.340 vs 0.334) but **the intervals overlap, so it is no detectable change, not a gain**. Belebele English is also unchanged.

How to read it, and what is not shown:

- **Cost and quality together.** The same Urdu text needs 2.29x fewer tokens (0.281 vs 0.643 per character) and the adapted model scores 0.62 fewer bits per character than the base on FLORES Urdu (2.339 to 1.718). FLORES Urdu is the cleaner Urdu number: the held-out Wikipedia lines come from the same dump as the training text, so they are in-domain and flatter the result.
- **The English gain (1.208 to 1.066) is not evidence that English got better.** The likely causes are that the base is a chat model, whose raw-text likelihood is worse than a model tuned on plain text, and that the replay text is English Wikipedia, close to FLORES in style. Neither cause is tested here. What the result does show is that English was not damaged by this metric.
- **Not attributable to the tokenizer.** The gain comes from adding 16M tokens of Urdu training, which the base never had. Whether the extended tokenizer helps or hurts quality against the same training on the base tokenizer is exactly what the compute-matched control (decision 6) answers, and it has not been run.
- **Understanding did not improve.** Belebele Urdu is at the base level, about 0.34, with an interval of about 6 points. This is a small continued-pretraining run, so better likelihood has not turned into better multiple-choice answers. Chat and instruction behavior was not measured and may have degraded.

## Stage 3: re-export of the adapted model (2026-11-24, notebook 09, Kaggle CPU, commit 59f2708)

`zubaan export all` and `zubaan export check` on the merged adapted model (notebook 07 output, step 800), llama.cpp `b12477`, 4 CPUs, no GPU, about 94 min wall. Results: `reports/export_summary_09.json` and `reports/export_check_09.json` (gitignored, local).

- **Tokenizer parity passed.** 2,041 of 2,045 texts give identical ids in Hugging Face and llama.cpp; the 4 others differ only because the input is not NFC (two combining marks swap order), 0 real mismatches. Same counts as Stage 2, as expected: training does not touch the tokenizer. Pre-tokenizer recognized as `qwen2` from the fingerprint, converter not patched.
- **Files.** Q4_K_M 1.12 GB and Q8_0 1.86 GB, both quantized with the Urdu importance matrix (200,056 characters of Urdu Wikipedia, 40 chunks of 512). The f16 file (3.5 GB) was only used for the checks and deleted.
- **Quantization cost against f16** (llama.cpp KL-divergence mode, FLORES devtest, 20 chunks of 512 tokens). All three files share one tokenizer, so per-token numbers compare between them; they must not be compared with the base model.

| Text | Quant | Mean KLD | Same top token | PPL ratio |
| --- | --- | --- | --- | --- |
| FLORES Urdu (f16 PPL 55.5) | Q8_0 | 0.0017 | 96.7% | 1.0036 |
| FLORES Urdu | Q4_K_M | 0.0341 | 85.9% | 1.0247 |
| FLORES English (f16 PPL 20.3) | Q8_0 | 0.0018 | 97.2% | 1.0033 |
| FLORES English | Q4_K_M | 0.0320 | 89.8% | 1.0271 |

  Mean KLD is about the same for both languages, so quantization does not hurt Urdu more than English. Q4_K_M keeps the top token less often on Urdu (85.9% vs 89.8%). The two perplexities are per token on different text and tokenizations, so they are not a language comparison.
- **Generation speed** (llama-completion, greedy, 80 new tokens, 4 Kaggle CPUs): f16 about 6 tokens/s, Q8_0 about 10.5, Q4_K_M about 15. This compares the quantizations with each other on one machine only.
- **Generated text (greedy, read by a person).** Every file answers Urdu prompts in fluent Urdu script and names Islamabad as the capital; English completion still works. Problems: repetition loops on open-ended prompts in all three files (worst in Q4_K_M, which repeats "علامہ اقبال" until the token limit) and invented facts (Islamabad's population as 21,000,000 and "largest city", 1.25 billion native Urdu speakers). With thinking allowed, the model reasons in English inside `<think>`, as base Qwen3 does. Greedy decoding on a 1.7B model makes loops more likely; sampling settings were not tested.
- **Bug found and fixed.** The first run failed its Urdu-script check (0.769 on an answer that was entirely Urdu) because `urdu_letter_share` counted the letters of the `<|endoftext|>` text llama.cpp prints. Special-token text and `<think>` tags are now removed before counting (commit 59f2708, with a test using that exact answer). The measurements of the first run were identical to this one.
