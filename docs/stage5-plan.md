# Stage 5 plan: drafters for the Urdu model (speculative decoding)

Draft, 2026-11-28, not yet approved. Decisions and results will be recorded in `docs/decisions.md`. Where this plan and a later decision differ, the decision holds.

## Why this stage exists

Stages 2 to 4 cut the first part of the Urdu tax: the Stage 4 model reads and writes Urdu with 2.43x fewer tokens than the base Qwen3-1.7B, answers in chat with thinking on and off, and runs in vLLM, llama.cpp and Ollama. This stage goes after the second part: **speculative decoding works worse for Urdu.**

In speculative decoding a small, cheap *drafter* proposes the next few tokens and the large *target* model checks them all in one forward pass. Every accepted token is a step the target did not have to take alone. With greedy decoding the output is exactly what the target would have written without a drafter, so quality cannot change; only speed does. Speed depends on how often the target accepts the drafter's guesses.

Stage 1 measured this for the stock pair, Qwen3-0.6B drafting for Qwen3-1.7B, on FLORES (`reports/acceptance_qwen3.json`):

| Language | Acceptance per token (alpha) | Greedy top-1 match | Best modelled speedup |
| --- | ---: | ---: | ---: |
| Urdu | 0.573 (0.570 to 0.575) | 0.520 | 1.17x (1 drafted token) |
| English | 0.632 (0.628 to 0.636) | 0.616 | 1.21x (1 drafted token) |

Two problems follow for our model:

1. **No existing drafter can be used.** A drafter must share the target's vocabulary exactly. The Stage 4 model has 10,000 new Urdu tokens, so the stock Qwen3-0.6B cannot draft for it.
2. **Small drafters are weak in Urdu,** the finding of Sandler et al. 2025, and Stage 1 confirms it for Qwen3.

## Goal and limits

**Goal.** Build drafters that share the Stage 4 model's vocabulary, and measure, Urdu against English:

1. acceptance per token, for each drafter against the Stage 4 model;
2. real generation speed in **words per second**, with and without each drafter, on a GPU (vLLM) and on a CPU (llama.cpp);
3. whether the Urdu speed gap to English is smaller than with today's stock setup (Qwen3-1.7B with Qwen3-0.6B).

**Not goals.** The full benchmark suite (time to first token, many runtimes and settings, server-reported acceptance at scale) is Stage 6. Sampling-based decoding is measured only if time allows; greedy is the main setting because it makes the lossless check exact. Other model families stay out of scope.

**An honest expectation.** A 0.6B drafter for a 1.7B target is an expensive ratio: each draft step costs about a third of a target step, so even a good drafter gives modest speedups (Stage 1 modelled at most 1.17x to 1.21x for the stock pair). The n-gram drafter costs almost nothing per step, so it may give more speed even with lower acceptance. Both results get reported, whichever wins.

## Decisions proposed (to confirm before starting)

1. **The target is the Stage 4 model** (`qwen3-1.7b-urdu-chat`: the merged folder from notebook 14 for vLLM, the Q4_K_M GGUF for llama.cpp). Its tokenizer is the Stage 3 extended one.
2. **The model drafter is Qwen3-0.6B, extended with exactly the same merges** (Stage 2 already did this once; `same_tokenizer` is the gate). Qwen3 has no smaller model, and a drafter from another family cannot share the vocabulary.
3. **Two training phases for the drafter, measured after each:**
   - **(a) Continued pretraining with the Stage 3 recipe** (new rows first, then LoRA plus the new rows, Urdu Wikipedia with English replay), so the drafter learns the new tokens. This reuses `zubaan adapt` unchanged on a smaller model.
   - **(b) Distillation on the target's own replies.** The Stage 4 model answers the user prompts of the Stage 4 training data (9,854 conversations; thinking on and off as in the original), and the drafter learns to predict those replies.
   Measuring acceptance after (a) and after (b) shows what each phase buys.
4. **Distillation loss: match the target's next-token distribution,** not only its chosen tokens. Acceptance at a position is the overlap of the two distributions, sum of min(p_target, p_draft), so training the drafter's distribution toward the target's targets acceptance directly. The target runs in fp16 without gradients next to the drafter on the same T4. Plain cross-entropy on the target's tokens is the simpler baseline and the fallback if memory or time runs short. Decided in the smoke run.
5. **The n-gram drafter uses the runtimes' built-in n-gram (prompt lookup) methods,** not new serving code. zubaan adds an offline simulation that computes n-gram acceptance from text, so it can be measured on the same texts as the model drafter.
6. **Two kinds of measurement, for two questions.**
   - **Teacher-forced acceptance** (`zubaan audit acceptance`, Stage 1 method) compares drafters for the *same* target and tokenizer. Per-token acceptance is not compared across tokenizers, because a token covers more Urdu text after extension (the same rule as tokens per second).
   - **Measured words per second** compares setups across tokenizers: stock Qwen3-1.7B with stock Qwen3-0.6B against the Stage 4 model with each new drafter, on the same prompts and hardware.
7. **Texts for measurement are never trained on:** FLORES devtest (Urdu and English, as Stage 1) and the 60 hand-written chat prompts of `chat_check_v1.yaml` (the target's replies to them, for acceptance on chat text). The distillation prompts are checked for near-copies of both.
8. **The chat check rule does not apply to the drafter in the usual way.** CLAUDE.md asks for a chat check after any training of a chat model. The drafter is never served alone, and with greedy decoding the target's output is unchanged. The check that replaces it is the lossless check (success check 1). The Stage 4 model is not retrained in this stage.

## Success checks (to implement in `draft/evaluate.py`)

1. **Lossless.** With greedy decoding, the output with each drafter is identical to the output without one, on every measured prompt on llama.cpp (CPU). On vLLM, fp16 batching can change a few greedy paths; the share of identical outputs is reported, and any difference is inspected.
2. **Vocabulary.** The drafter's tokenizer is identical to the target's (`same_tokenizer`), and its GGUF passes the parity check. Hard gate: nothing else runs without it.
3. **Distillation helps.** Urdu acceptance of the distilled drafter on the target's chat replies is above the continued-pretraining-only drafter's, and that one is above the untrained extended drafter's, with non-overlapping 95% intervals.
4. **Real speedup.** At least one drafter gives a measured Urdu speedup above 1.0x, in words per second, on at least one runtime. If none does, that is the result and it is reported.
5. **The Urdu gap.** The Urdu-to-English ratio of words per second, best Stage 5 setup against today's stock setup (both measured in this stage, same prompts, same hardware). This is reported with its numbers, not as pass or fail, because the size of the gain is the finding.

## Chunks

Each chunk has one deliverable and a done criterion.

0. **Runtime trial** (Kaggle 2x T4 about 1 GPU hour, plus the laptop). Check that vLLM 0.34 runs speculative decoding with a separate draft model and with n-gram, and that llama-server runs with a draft model (`-md`) and with n-gram lookup, all with greedy decoding and our extended vocabulary. Use the untrained extended 0.6B as a stand-in drafter. Measure the stock pair's words per second, Urdu and English, as the baseline. Check whether Ollama supports a drafter at all. **Done when** the runtimes and settings are recorded and the stock baseline is measured.
1. **Drafter build** (laptop). Extend Qwen3-0.6B with the Stage 3 merges, mean-of-pieces initialization, chat template carried over (the Stage 2 0.6B GGUF lacked it), `same_tokenizer` against the Stage 4 model. **Done when** the drafter folder passes the checks.
2. **Acceptance on chat text** (laptop code). `zubaan audit acceptance` gains chat-formatted texts (the target's replies rendered with the chat template, acceptance counted on the reply tokens only) and local model folders. Plus the offline n-gram acceptance simulation (`draft/ngram.py`). **Done when** tests pass on tiny models.
3. **Drafter continued pretraining** (Kaggle T4, about 3 to 4 GPU hours). The Stage 3 recipe on the extended 0.6B, smoke run first, Hub checkpoints, merge and verify. **Done when** a merged drafter is saved and its acceptance against the Stage 4 model is measured.
4. **Distillation data** (Kaggle 2x T4, about 1 to 2 GPU hours). The Stage 4 model answers the Stage 4 training prompts through vLLM, thinking on and off as in the original data, with a fixed seed per request. Failed requests are dropped; everything else is kept as written, loops included, because the drafter should predict what the target actually does. **Done when** the reply file and its counts are recorded.
5. **Distillation code** (laptop, tiny models). `draft/distill.py`: chat rendering and reply-only loss from `tune/data.py`, the distribution loss computed in chunks of positions (a 161,669-token vocabulary makes full logits for both models large), LoRA plus the new rows, resume, merge. **Done when** the tests pass on CPU.
6. **Distillation smoke and main run** (Kaggle T4, about 2 to 3 GPU hours). The smoke run picks the loss (decision 4) and checks memory and speed. **Done when** a merged distilled drafter is saved and verified.
7. **Evaluation** (Kaggle, about 2 to 3 GPU hours; CPU parts on Kaggle CPU or the laptop). Teacher-forced acceptance for every drafter (untrained, after continued pretraining, distilled, n-gram) on FLORES and on chat replies, Urdu and English. Measured words per second with and without each drafter, through vLLM on a T4 and llama-server on CPU, against the stock pair. The lossless check on every run. **Done when** every success check is in `docs/decisions.md`.
8. **Export and write-up.** Drafter GGUF with the parity check (Q8_0 and Q4_K_M), the command to serve target and drafter together in llama-server and vLLM, `docs/stage5-usage.md`, a report section, roadmap and the `CLAUDE.md` stage table.

## Budget

About 10 to 12 Kaggle GPU hours, planned with a ceiling of 15 (Stage 4 used about 15 against 10 budgeted):

| Chunk | GPU hours |
| --- | ---: |
| 0. Runtime trial and stock baseline | about 1 |
| 3. Drafter continued pretraining (smoke and main) | 3 to 4 |
| 4. Distillation data | 1 to 2 |
| 6. Distillation (smoke and main) | 2 to 3 |
| 7. Evaluation | 2 to 3 |

Chunks 1, 2 and 5 run on the laptop; the n-gram drafter, GGUF export and the CPU speed runs need no GPU. Kaggle gives about 30 GPU hours a week.

## Risks

- **The size ratio limits the gain.** A 0.6B drafter costs about a third of a 1.7B step. Even with much better acceptance, measured speedups may be small, or below 1.0x on some runtimes. The n-gram drafter and honest reporting are the answer, not a bigger claim.
- **Bigger tokens are harder to guess.** After extension one Urdu token covers about 2.4 times as much text, so per-token acceptance may fall even as each accepted token is worth more. That is why the cross-setup comparison is in words per second.
- **Runtime support.** Separate-draft-model speculative decoding has changed across vLLM versions, and support for a custom vocabulary size or an extended tokenizer is untested. Chunk 0 checks this before any training. If vLLM cannot do it, llama.cpp carries the measurement and the gap is stated.
- **Memory in distillation.** Two models and two sets of logits over a 161,669-token vocabulary on one 16 GB T4. The loss is computed in chunks; shorter rows (2,048 tokens) are the fallback.
- **The target's loops become training data.** The drafter learns to predict them too. That is correct for acceptance, but the reply counts and loop counts are recorded so it is visible.
- **fp16 nondeterminism on vLLM** can make greedy outputs differ slightly with and without a drafter. The lossless check is exact on llama.cpp CPU and reported as a share on vLLM.

## Code layout (proposed)

`src/zubaan/draft/`: `build.py` (extend the drafter with the target's merges, reusing `extend/`), `distill.py` (data, distribution loss, training loop, merge), `ngram.py` (offline n-gram acceptance), `speed.py` (words per second with and without a drafter against a running server, the seed of the Stage 6 benchmark runner), `evaluate.py` (success checks). Acceptance on chat text extends `audit/acceptance.py`. CLI: `zubaan draft build | distill | merge | speed`, plus the extended `zubaan audit acceptance`. Configs in `config.py` and `configs/`. Notebooks continue from 17. Heavy dependencies stay lazy.

## Status (2026-11-28)

- [ ] Plan approved by the author
- [ ] Chunk 0: runtime trial and stock baseline
- [ ] Chunk 1: drafter build
- [ ] Chunk 2: acceptance on chat text, offline n-gram simulation
- [ ] Chunk 3: drafter continued pretraining
- [ ] Chunk 4: distillation data
- [ ] Chunk 5: distillation code
- [ ] Chunk 6: distillation smoke and main run
- [ ] Chunk 7: evaluation
- [ ] Chunk 8: export and write-up
