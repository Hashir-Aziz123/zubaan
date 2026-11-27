# Stage 3 plan: adapt (continued pretraining)

Archived 2026-11-28 from `continue.md`, where it was written. Stage 3 is closed (2026-11-25); the results are in `docs/stage3-usage.md` and `docs/decisions.md`. Stage numbers in the last section are the new ones (draft is Stage 5, bench Stage 6).

Draft, 2026-11-18, decisions settled the same day. See the status list at the end of this section for what is built. The `train` extra lists `peft` and `accelerate` (`trl` was dropped, see decision 10).

**Goal.** Take the extended Qwen3-1.7B (10,000 new tokens, untrained rows) and train it on Urdu text until it uses the new tokens well, without hurting English. The result is the first model that is both cheaper per Urdu word and not worse at Urdu than the base.

**Limit to state everywhere.** This is a small continued-pretraining run (tens of millions of tokens, one T4), not a new foundation model. Gains will be modest, Belebele on a 1.7B model is noisy, and chat and instruction behavior may degrade because we train on plain text. Quality is always reported next to cost, with confidence intervals.

## Why this stage is needed (the idea)

A transformer reads a token through its embedding row and writes a prediction through the output head row. After Stage 2, 10,000 tokens have rows that are only an average of the old pieces they replace. The model has never seen those tokens in context, so its attention and MLP layers do not yet expect them. That is why Urdu bits per character went from 2.31 to 3.57 and Belebele fell to chance.

Training has to do two things:
1. **Move the new rows to the right place** (they are the only parameters that know about the new tokens).
2. **Let the rest of the network adapt a little** so that "one new token for a whole Urdu word piece" is handled like the old multi-token spelling was.

We cannot full fine-tune 1.7B parameters on a 16 GB T4 (AdamW needs about 16 bytes per parameter, roughly 27 GB). So we train a small number of parameters: the new embedding rows (10,000 x 2048 = 20.5M values) plus LoRA adapters on the transformer layers. English replay (a slice of English text mixed in) guards against forgetting.

## Recipe in plain steps

1. **Data.** Urdu: `data/urdu_train.txt` (Urdu Wikipedia, cleaned, capped at 100 MB, 295,905 lines). Tokenize with the extended tokenizer, join documents with `<|endoftext|>`, pack into fixed blocks (for example 1024 tokens). Rough size, to be measured not assumed: about 50M characters, so about 14M tokens at the 0.28 tokens per character seen in the 0.6B smoke run. English replay: a small share of English text, tokenized the same way (English tokenization is identical in both tokenizers).
2. **Phase A, new rows only.** Freeze everything except the 10,000 new embedding rows (and the matching head rows, which are the same tensor because Qwen3-1.7B ties them). Short run, higher learning rate. This stops the new rows from being noise that the LoRA layers would otherwise have to work around.
3. **Phase B, LoRA plus new rows.** Add LoRA (rank and targets in the decisions below) on attention and MLP projections and keep training the new rows. Cosine schedule with warmup, gradient accumulation, gradient checkpointing, mixed precision.
4. **Checkpoint and resume.** Save to the Hugging Face Hub during training (`hub_strategy="checkpoint"`) and resume with `resume_from_checkpoint=True`, because Kaggle sessions end after 12 hours and can die.
5. **Merge and save.** Merge LoRA into the weights so the output is a plain Hugging Face folder. Then the existing Stage 2 tools (verify, GGUF export, Ollama, vLLM) work unchanged. Mark `zubaan_extension.json` with `trained: true` and the training details.
6. **Evaluate with the Stage 2 harness** (`extend/quality_run.py`, `bench/quality.py`): bits per character on held-out Urdu, FLORES Urdu, FLORES English; Belebele `urd_Arab` and `eng_Latn` with `acc_char` and Wilson 95% intervals. Never compare per-token loss across tokenizers.
7. **Re-export and recheck:** GGUF parity test again, Q4_K_M with the Urdu imatrix, a real Urdu generation sample, a vLLM check.

## Success checks (decision 9)

1. English tokenization still byte-identical and Urdu still round-trips (the tokenizer is not touched, so this must stay true; the verify step re-checks).
2. Held-out Urdu bits per character clearly below the untrained 3.57. Stretch goal: at or below the base model's 2.31 even though the extended model needs 2.4x fewer tokens.
3. English FLORES bits per character within a stated tolerance of the base (1.208). Proposed: at most +0.02.
4. Belebele Urdu `acc_char` above the base 33.4%, reported with its interval. With 900 questions the 95% interval is about plus or minus 3 points, so a gain under about 4 points is not distinguishable from noise and must be described that way.
5. Parity still passes on the final GGUF, and a Q4_K_M answers a simple Urdu prompt in readable Urdu.

## Decisions (settled 2026-11-18, full reasoning in `docs/decisions.md`)

All recommended defaults were accepted.

| # | Decision | Choice |
| --- | --- | --- |
| 1 | Starting model | The existing extended `Qwen3-1.7B` (chat model), not `-Base`. Chat and `<think>` ability will degrade and the report says so |
| 2 | Trainable | Phase A: only the 10,000 new rows. Phase B: add LoRA. Old embedding rows frozen via a separate trainable tensor, not a gradient mask |
| 3 | Hub | Private repo for checkpoints, token as a Kaggle secret. Still to do by you: create the repo and token. Nothing public |
| 4 | LoRA (start values) | Rank 64, alpha 128, dropout 0.05, all attention and MLP linears, LR about 2e-4 (LoRA) and 1e-3 (new rows) |
| 5 | Urdu data | Wikipedia only (about 14M tokens, estimate) |
| 6 | English replay | English Wikipedia, 10% of blocks |
| 7 | Control | Base tokenizer, same recipe, matched on compute, after the main run if hours allow |
| 8 | Precision | fp16 autocast with a gradient scaler, fp32 only if NaN |
| 9 | Success checks | English bpc at most +0.02 over 1.208; Urdu bpc below 3.57 (must), 2.31 (stretch); Belebele Urdu above 33.4% with interval; cap about 10 GPU hours |
| 10 | Defaults | Block size 1024, one T4, gradient checkpointing, own training loop (no `trl`), notebook 05 |

## Chunks

Each chunk has one deliverable, a test that can fail, and a done criterion. "Laptop" means CPU, 8 GB RAM, tiny random models only. Default tests never load real weights.

### Chunk 0: Environment check (laptop plus a short Kaggle cell)
- Confirm versions of `torch`, `transformers`, `peft`, `accelerate` that install together on Kaggle and on the laptop. Pin them in the notebook.
- **Done when:** a Kaggle cell prints the versions and a 5-step LoRA forward and backward in fp16 on the T4 does not give NaN.

### Chunk 1: Training data, `adapt/data.py` (laptop)
- Pure functions: pack token streams into blocks with `<|endoftext|>` separators, interleave Urdu and English at the configured ratio, deterministic for a seed, write token blocks to disk (numpy memmap or Arrow) with a metadata JSON (sources, token counts, hash).
- Guard: no training line shares 8 consecutive words with FLORES. The Stage 2 guard checked only devtest. Belebele passages come from FLORES-200 (verify which splits), so also check the dev split and any Belebele passage text.
- Count real tokens and replace the 14M estimate above in this file.
- **Tests:** packing and ratio on a tiny fixture; same seed gives identical blocks; an injected FLORES line is rejected.
- **Done when:** blocks exist for the real corpus and the metadata records exact token counts and zero overlap hits.

### Chunk 2: Trainable parameters, `adapt/params.py` (laptop, tiny random model)
- Load the extended folder, attach LoRA, give the new rows their own trainable tensor, keep input embedding and head tied.
- **Tests:** only the intended tensors get gradients; old embedding rows are bit-identical after several steps; head and embedding remain one tensor; merging LoRA gives the same logits as the unmerged model within tolerance; parameter counts match the hand calculation (10,000 x 2048 for the new rows).

### Chunk 3: Training loop and config, `adapt/cpt.py`, `AdaptConfig` (laptop)
- Config driven (`--config path.yaml`): phases, learning rates, schedule, batch and accumulation, precision, eval interval, checkpoint interval, Hub repo.
- Features: fp16 autocast with scaler, gradient checkpointing, NaN guard that stops and says why, JSONL log, periodic held-out bits per character, checkpoint and resume.
- **Tests (tiny model, CPU, seconds):** loss falls on a small repeated corpus; resume from a checkpoint reproduces the next step's loss exactly; the NaN guard fires on a poisoned batch.
- **Done when:** `zubaan adapt run --config configs/adapt-qwen3-smoke.yaml` works end to end on the tiny fixture.

### Chunk 4: Smoke run on Kaggle (T4, 20 to 50 steps)
- `notebooks/make_05_notebook.py` generates `notebooks/05_adapt_qwen3.ipynb` (same pattern as 03 and 04: pin the pushed commit, check versions and machine first). Kaggle needs the pinned commit pushed to GitHub first, so ask before pushing.
- Record: tokens per second, peak GPU memory, loss finite, a kill-and-resume test through the Hub.
- **Done when:** the numbers are in `docs/decisions.md` and decisions 3, 7 and 9 are settled with data.

### Chunk 5: Main run (Kaggle, one or more sessions)
- Phase A then phase B, held-out bits per character logged, Hub checkpoints.
- **Done when:** the schedule completes or the held-out curve flattens, and the final checkpoint is on the Hub (private).

### Chunk 6: Merge, save and verify (laptop or Kaggle CPU)
- Merge LoRA, save with the chat template files (the Stage 2 fix), run the existing folder verification (tokenizer identical, English unchanged, Urdu round-trip, embedding row count), write `trained: true` and the recipe into `zubaan_extension.json`.

### Chunk 7: Quality evaluation (Kaggle T4)
- Reuse `evaluate_variant`; add a small entry point if needed to score a saved folder instead of building a variant. Table: base, extended untrained, extended adapted, plus the control from decision 6 if done. Chart and a new "Adaptation" section in `docs/reports/tokenizer-extension/`.
- **Done when:** cost and quality for every row sit in one table with intervals, and the checks above are marked pass or fail honestly.

### Chunk 8: Re-export and recheck (laptop plus Kaggle)
- `zubaan export all` on the adapted model: GGUF parity, Q4_K_M and Q8_0 with the Urdu imatrix (now meaningful because the model produces Urdu), Ollama chat sample, vLLM check on a T4.
- **Done when:** parity passes and the Q4_K_M produces readable Urdu on CPU.

### Chunk 9: Write-up
- `docs/stage3-usage.md`, decisions appended to `docs/decisions.md` with date and reason, roadmap and the stage table in `CLAUDE.md` updated, limitations stated (small run, noisy Belebele, chat ability).

## Status

- [x] Decisions settled and baked into `docs/decisions.md`, `docs/roadmap.md`, `CLAUDE.md` (2026-11-18)
- [x] Chunk 1: `adapt/data.py` (blocks, mixing, contamination guard, English replay), `zubaan adapt prepare`
- [x] Chunk 2: `adapt/params.py` (new rows as their own tensor, LoRA, phases, merge)
- [x] Chunk 3: `adapt/cpt.py`, `adapt/pipeline.py`, `AdaptConfig`, `zubaan adapt run`, `configs/adapt-qwen3-1.7b.yaml`. 40 new tests, all CPU, tiny random model; exact-resume test passes
- [x] Real corpus prepared locally in 2 minutes: 16.05M Urdu tokens, 15,677 blocks, 2,613 English blocks, 47 Urdu lines dropped by the wider guard. One epoch is 544 steps, so the default 950 steps is about 1.75 epochs. `data/adapt/` is gitignored; Kaggle needs to rerun `adapt prepare` or take it as a dataset
- [x] Chunk 6: `adapt/finalize.py` and `zubaan adapt merge` (merge, save, verify) on a tiny model; resume refuses a checkpoint from a different schedule
- [x] Chunks 0 and 4: notebook 05 (`check` and `smoke` modes) ran on Kaggle. Two notebook bugs were found and fixed (Kaggle's torchao 0.10 breaks peft: removed in the install cell; a `history` key `run_adapt` does not return; and a stale step-30 smoke checkpoint on the Hub: smoke now clears it). Notebook 05 is left as is
- [x] Speed trial and checkpoint plan: notebook 06 (speed trial, optional Hub round-trip test) and notebook 07 (main run, Hub required) built from `notebooks/adapt_cells.py`, which reuses 05's cells. Checkpoint to the Hub every 50 steps, permanent copy every 200 (`<repo>/milestones/step-NNNNNNN`), upload failure warns without stopping training, `merge --milestone`
- [x] Hub repo `hashiraziz21/zubaan-qwen3-ckpt` created (keep it private); token made. The secret is not yet attached to the Kaggle notebook (see Tomorrow)
- [x] Chunk 5: main run in notebook 07 finished (800 steps, about 9.4 h wall). Chunk 6 merge ran at the end of the notebook. Merge verified (max logit diff 6.5e-05). In-run held-out Urdu bpc 2.41 (step 100) to 1.52 (step 800), English 1.21 to 1.08 on 200 texts each; must be re-measured on the Stage 2 protocol before comparing with base 2.31 / untrained 3.57 / English 1.208. Log: `reports/notebook7_logs.txt`
- [ ] Optional compute-matched control run (decision 6): code and notebook 11 done (2026-11-25); Kaggle run pending
- [x] Chunk 7 evaluation: `zubaan adapt eval`, notebook 08 run on Kaggle (2026-11-19), results in `docs/decisions.md`
- [x] Chunk 7 report section and chart in `docs/reports/tokenizer-extension/` (2026-11-25)
- [x] Chunk 8 part 1: notebook 09 re-export on Kaggle CPU (2026-11-24): parity 2041/2045 (4 NFC-only, 0 real), Q4_K_M and Q8_0, quantization cost and samples recorded in `docs/decisions.md`. Files: `reports/export_summary_09.json`, `reports/export_check_09.json`
- [x] Chunk 8 part 2a: notebook 10 vLLM check on a T4 (2026-11-25): tokenization 5/5 identical, Urdu completion and chat (thinking off) work; with thinking allowed the model ends inside `<think>` with no answer. File: `reports/vllm_check_adapted.json`
- [x] Chunk 8 part 2b: local Ollama chat check (2026-11-25): with thinking on 1 of 4 prompts answered, with `think: false` 4 of 4. GGUF and Modelfile in `outputs/adapted-export/`, Ollama model `zubaan-urdu-adapted` in `tools/ollama-models`
- [x] Chunk 9 write-up (`docs/stage3-usage.md`, roadmap and CLAUDE.md stage table), 2026-11-25

## Order

```
0 -> 1 -> 2 -> 3 -> 4 -> 5 -> 6 -> 7 -> 8 -> 9
                         (control run for decision 6 goes after 5, before 7)
```

Chunks 0 to 3 need no GPU and can be finished and tested on the laptop first. Only chunks 4, 5, 7 and 8 use Kaggle hours.

## Risks

- **fp16 overflow or NaN** on T4. Mitigation: NaN guard, smoke run first, fp32 fallback (decision 7).
- **Tied embedding bug.** Training only one of the two tied tensors, or breaking the tie on save, silently ruins the model. Covered by the chunk 2 tests and the verify step in chunk 6.
- **English or chat regression.** Mitigation: English replay, old rows frozen, English bits per character checked every eval, caveat documented (decision 1).
- **Overfitting a small corpus.** 14M tokens is small. Mitigation: watch held-out bits per character, stop when it flattens, add data only if justified (decision 4).
- **Evaluation contamination.** FLORES and Belebele share source passages; the overlap guard must cover the dev split and Belebele text (chunk 1).
- **Kaggle limits.** 12-hour sessions, about 30 GPU hours a week, 20 GB output. Mitigation: Hub checkpoints, test on 20 to 50 steps first, delete intermediates.
- **Noisy headline numbers.** Belebele Urdu on a 1.7B model sits near chance. Report intervals; lean on bits per character as the primary quality measure.
- **Merging LoRA in fp16** can lose small updates to rounding. Mitigation: merge in fp32, then save, then re-run the verify and quality checks on the merged folder, not the adapter.

## Later: stage 5 (draft) and 6 (bench), numbered 4 and 5 before 2026-11-25

Stage 5 (drafters sharing the extended vocabulary, distilled from the Stage 4 target) and stage 6 (benchmarks); see `docs/roadmap.md`. Stage 5 uses Qwen3-0.6B with the same merges, which is why the 0.6B was extended with the identical tokenizer.
