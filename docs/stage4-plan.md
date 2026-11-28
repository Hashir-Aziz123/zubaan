# Stage 4 plan: restore chat behavior (instruction tuning with generated data)

Draft, 2026-11-25; status updated 2026-11-28 (see "Status" at the end). Decisions and results are recorded in `docs/decisions.md`, from "Stage 4: restore chat behavior" (2026-11-25) onwards. Where this plan and a later decision differ, the decision holds. Stage numbers changed on that date: the drafter stage is now Stage 5, and benchmarks and release are Stage 6.

## Why this stage exists

Stage 3 trained the extended Qwen3-1.7B on plain Urdu and English Wikipedia text. It met its goals: Urdu bits per character better than the base (FLORES 2.339 to 1.718) with 2.43x fewer tokens, English not damaged, and the model runs in llama.cpp, Ollama and vLLM. But training a chat model on plain text damaged its chat behavior:

- **With thinking on** (Qwen3's default), the model usually ends its turn after its reasoning with no answer: 3 of 4 prompts in the Ollama check, and the one vLLM sample.
- **With thinking off**, it answers, but it invents facts and falls into repetition loops on open-ended prompts.
- **None of Stage 3's checks caught this.** It was found by hand at the end.

This affects every chat model, not only Qwen3: anyone who runs `zubaan adapt` on a chat model with today's recipe gets a model that passes every check and then fails as a chat model. The project promises a model people can deploy, so this is fixed before building drafters on top of it. The drafter stage (now Stage 5) distills from the target model and measures acceptance against it. Fixing the target first means the drafter work runs once, against the final model.

## Goal and limits

**Goal.** Starting from the Stage 3 adapted model, a short instruction-tuning step (supervised fine-tuning, SFT) on generated Urdu and English conversations, so that the model:

1. answers with thinking on, closing its `<think>` block and then replying;
2. answers with thinking off;
3. replies to Urdu prompts in Urdu;
4. keeps the Stage 3 gains (Urdu bits per character, English not damaged, token saving unchanged).

**Not goals.** Facts are limited by what a 1.7B model can hold; instruction tuning changes how it answers, not what it knows. Qwen's own post-training (preference tuning, safety, tool use) is not reproduced. Safety behavior is not measured in this stage, and the docs must say so. Support for other model families (SentencePiece tokenizers, models with separate input and output matrices, non-ChatML chat formats) is a separate later item, not this stage.

## Decisions taken (2026-11-25)

1. **Repair after training, not prevention during it.** A short SFT step on the existing adapted model, not a full retrain with chat replay. It keeps everything Stage 3 learned, costs about 10 GPU hours in total instead of about 15, and teaches Urdu *replies*, which Wikipedia training never did. Replay during training stays possible later if SFT is not enough.
2. **Thinking mode is supported,** not switched off. That needs training examples with reasoning traces, which no Urdu dataset has, so the traces are generated.
3. **Generated data from a larger open Qwen3 teacher,** not Aya or translated datasets. Aya has no reasoning traces. A larger Qwen3 writes traces natively in Qwen3's format, and its Apache 2.0 license allows training on its outputs. This follows the "strong-to-weak distillation" Qwen describes for its own small models. Commercial API models are not used to generate training data (their terms generally restrict training on outputs), only possibly to judge quality.
4. **The Stage 3 control run is unaffected.** The continued-pretraining recipe does not change, so the control (notebook 11) still answers its question and can run whenever there is quota. It is not part of this stage.

## The data plan

- **Teacher.** Qwen3-8B (fp16, vLLM split over Kaggle's two T4s) or a 4-bit Qwen3-14B on one T4. Picked in chunk 0 by reading its Urdu, not by reputation. **Picked: Qwen3-14B-AWQ** over both T4s (2026-11-26).
- **The teacher needs no extended tokenizer.** It produces text; the text is tokenized with the extended tokenizer at training time.
- **Prompts grounded in Urdu Wikipedia passages** from the Stage 3 training split. The teacher writes questions about a passage (factual, explain, summarize, list, compare, open-ended) and then answers them. Grounding reduces invented facts in the training answers. The existing contamination guard keeps FLORES and Belebele text out.
- **A mix of modes and languages.** Starting point, to confirm in chunk 0: about 10,000 conversations, 40% Urdu with thinking, 40% Urdu with thinking off (empty think block, as Qwen3's template writes it), and 20% English, half with thinking. Mostly single turn, with some two-turn conversations.
- **Expected shape of the traces.** Qwen3 reasons in English even for Urdu questions, so traces will mostly be English, followed by an Urdu answer. That matches the base model's behavior; the docs will say so.
- **Filters, all automatic and counted:**
  - with thinking on, the `<think>` block is closed and an answer follows;
  - answers to Urdu prompts are in Urdu script (`urdu_letter_share`);
  - no repetition loops;
  - length within bounds;
  - no near-duplicates.
- **A person reads a sample.** 100 to 200 conversations, with the pass rate recorded. This catches stiff phrasing and subtle errors that filters miss. The author is not an Urdu speaker, so the 150-conversation sample was read by Claude, and the record says so (71% pass, Urdu 66%; 2026-11-27).
- **The evaluation prompts are separate from the generator.** The chat-behavior test set is written by hand (or comes from a different source) and is never generated by the teacher, so the model is not graded on the teacher's own homework.

## Training plan

- **Method.** LoRA SFT on the merged Stage 3 model, fp16 on one T4, the same infrastructure as Stage 3: checkpoints to a private Hub repo, exact resume, and a NaN guard.
- **Loss only on the assistant's reply** (thinking plus answer); the system and user text are masked out.
- **Conversations are rendered with the model's own chat template** (`tokenizer_config.json`), with and without thinking, so the training format is exactly the serving format.
- **Open question for chunk 5:** whether the 10,000 new embedding rows also train in SFT (at a low learning rate) or stay frozen. Start frozen; test both in the smoke run if Urdu replies look weak.
- **Size.** About 10,000 conversations of about 600 tokens is about 6M tokens: roughly 1 to 2 hours on a T4 at Stage 3's measured speed, for 1 to 2 epochs.

## Success checks (implemented in `tune/evaluate.py`)

Measured on a held-out, hand-written set of chat prompts (Urdu and English), greedy decoding, through vLLM or llama.cpp, for the base model, the Stage 3 adapted model and the Stage 4 model:

1. **Thinking on:** the share of prompts that get an answer is within 10 points of the base model's (Stage 3 adapted: 1 of 4 in the Ollama check).
2. **Thinking off:** at least 95% of prompts get an answer.
3. **Language:** at least 90% of Urdu prompts are answered in Urdu script.
4. **Loops:** no more replies that run to the token limit than the base model.
5. **Stage 3 gains kept:** FLORES Urdu bits per character at most 0.05 above the Stage 3 model (1.718), and still below the base (2.339); FLORES English at most 0.02 above the base (1.208); Belebele Urdu not below the base's interval.
6. **Read by a person:** a sample of replies, judged against the base model's replies to the same prompts, recorded honestly even if it is worse.

## Chunks

Each chunk has one deliverable and a done criterion.

0. **Teacher trial** (Kaggle, about 1 to 1.5 GPU hours, notebook 12). Qwen3-8B and Qwen3-14B-AWQ, both split over the two T4s, answer the same questions about 50 Urdu passages, with and without thinking, in two formats (grounded and standalone, see `docs/decisions.md`, 2026-11-26). **Done when** the author has read the outputs and picked the teacher, and the data mix and size are confirmed.
1. **Chat-behavior check** (laptop code, then a short run). A command that runs a fixed prompt set through a model server and reports, per mode, the answer rate, Urdu-script share, closed think blocks and loops. It also writes the hand-written prompt set. Run it on the base and Stage 3 models to get a measured baseline (base Q4_K_M from Qwen's official GGUF, through Ollama or llama.cpp on the laptop). **Done when** the baseline numbers are in `docs/decisions.md`.
2. **Prompt generation** (code plus Kaggle). Passages are sampled from the training split; the teacher writes questions; the contamination guard is applied. **Done when** there is a prompt file with counts.
3. **Response generation** (Kaggle, a few GPU hours). The teacher answers with thinking on and off, batched in vLLM, and saves JSONL with metadata (teacher, mode, passage id). **Done when** the target count is reached.
4. **Filtering and human review** (laptop). Apply the filters with counts, then the author reads 100 to 200 conversations. **Done when** the filtered dataset and its review pass rate are recorded.
5. **SFT code** (laptop, tiny random model). Chat-template rendering, assistant-only loss masking, packing, LoRA, resume; config-driven; tests. **Done when** the tests pass on CPU.
6. **Smoke and main SFT run** (Kaggle T4). A 30-step smoke run, then the main run, then merge and verify. **Done when** a merged model is saved and verified.
7. **Evaluation.** The chat-behavior check, plus the Stage 3 quality protocol (bits per character, Belebele), on base, Stage 3 and Stage 4. **Done when** every success check reads PASS or FAIL in `docs/decisions.md`.
8. **Re-export.** GGUF with the parity check, Q4_K_M and Q8_0, Ollama (does thinking now work with Ollama's default?), vLLM. **Done when** all three serve the model and the chat check passes on the quantized file.
9. **Write-up.** `docs/stage4-usage.md`, a report section, the scope note (tested on Qwen3 only), roadmap, and the `CLAUDE.md` stage table.

## Budget

About 10 Kaggle GPU hours: teacher trial about 1, generation about 3 to 5, SFT smoke and main about 2, evaluation about 2, vLLM check about 0.5. The re-export runs on CPU. Kaggle gives about 30 GPU hours a week.

## Risks

- **The teacher's Urdu is the ceiling.** Its mistakes and style become the student's. Chunk 0 exists to check this before spending hours.
- **Uniform synthetic data.** Everything sounds like one model; the human-read sample is the guard.
- **SFT can cost some of the Stage 3 gain.** It is measured (check 5), not assumed.
- **Hardware.** Qwen3-8B in fp16 (about 16 GB) does not fit one T4. It needs vLLM tensor parallelism over two T4s (PCIe), or a 4-bit model; both are untested here.
- **Chat quality is hard to measure.** Answer rate and language are measurable; helpfulness is judged by a person on a sample, and that judgment is reported as such.
- **Safety is not measured** in this stage.

## Code layout

`src/zubaan/tune/`: `client.py` (OpenAI-compatible server client), `generate.py` (prompts and teacher responses), `filters.py`, `curate.py` (filter, balance, review sample), `data.py` (chat rendering, masking, packing), `sft.py` (training loop, packing check, merge, probe), `chat_check.py` (the behavior check, prompt set in `prompts/chat_check_v1.yaml`), `evaluate.py` (success checks). CLI: `zubaan tune check | filter | run | merge`; generation runs in notebook 13. Config: a `tune:` section in `config.py`. Heavy dependencies stay lazy, and vLLM runs only in its own notebook environment, as in notebooks 04 and 10.

## Status (2026-11-28)

- [x] Chunk 0: teacher trial (notebook 12). Qwen3-14B-AWQ; standalone questions only with thinking off; mix fixed
- [x] Chunk 1: `zubaan tune check` and prompt set v1. The baselines run in notebook 15 (the laptop ran short of memory)
- [x] Chunks 2 and 3: notebook 13, 2 shards, 11,502 answers
- [x] Chunk 4: `zubaan tune filter` kept 9,854 conversations; the 150-sample read (by Claude) passed 71%
- [x] Chunk 5: SFT code, tested on a tiny model
- [x] Chunk 6: smoke run (658 tokens/s, 12.4 GB) and main run (140 steps, held-out loss 0.970, merged; 8 of 8 probe replies answered)
- [x] Chunk 7: notebook 15, checks 1 to 5 PASS; check 6 read by Claude (Urdu better than the base but still weak; English reasoning with thinking off worse)
- [ ] Chunk 8: re-export
- [ ] Chunk 9: write-up

GPU hours so far: teacher trial about 1 to 1.5, generation about 7.2 (2 shards of about 3.6 h, above the 3 to 5 planned), smoke about 0.5, main about 4 when done. With evaluation (about 2) the stage comes to about 15 GPU hours, above the 10 budgeted, mainly because generation used longer reply limits.
