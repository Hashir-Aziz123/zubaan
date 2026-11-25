# Where we left off (end of 2026-11-19)

Stages 1 and 2 are done. **Stage 3 training and quality evaluation are done**; what is left is the report section, the re-export and recheck (chunk 8) and the write-up (chunk 9), plus an optional control run next week. To resume, tell Claude: "read continue.md and pick up from there".

## Stage 3 result (details and table: `docs/decisions.md`, "evaluation results", 2026-11-19)

The adapted Qwen3-1.7B (10,000 new Urdu tokens, 800 steps of new rows then LoRA, about 9.4 h on a T4, notebook 07) was scored by notebook 08 on the same texts as Stage 2 (the base model reproduced its Stage 2 numbers exactly):

| | Tokens per Urdu char | BPC Urdu FLORES | BPC Urdu held-out | BPC English | Belebele Urdu acc_char |
| --- | --- | --- | --- | --- | --- |
| Base | 0.643 | 2.339 | 2.310 | 1.208 | 0.334 [0.304, 0.366] |
| Extended, untrained | 0.281 | 3.562 | 3.570 | 1.210 | 0.253 |
| Adapted | 0.281 | 1.718 | 1.486 | 1.066 | 0.340 [0.310, 0.372] |

- Same Urdu text needs 2.29x fewer tokens and scores 0.62 fewer bits per character than the base on FLORES Urdu (the cleaner number; held-out is in-domain Wikipedia). English not damaged.
- **Belebele shows no detectable change** (intervals overlap with the base; the check printed PASS only because 0.340 > 0.334). Say "no change", not "gain".
- The English gain (1.208 to 1.066) is not evidence English improved: likely a chat model scored on plain text plus Wikipedia-like replay. Untested.
- **The gain cannot be attributed to the tokenizer** until the compute-matched control (decision 6) runs: base tokenizer, same recipe. Not run; planned for next week's Kaggle quota. Chat and instruction behavior was not measured.

## What to do next (in order)

1. **Report section and chart** (last part of chunk 7, laptop only): add an "Adaptation" section and chart to `docs/reports/tokenizer-extension/` from `reports/adapt_quality.json` (gitignored, local; it holds the full results with intervals). Look at `report/extension.py` and `report/charts.py`; the results file has `variants` keyed `base`, `extended-untrained`, `adapted` and an `order` list. `adapt/evaluate.py` has `comparison_rows` and `success_checks` to build the table. State the caveats above in the report.
2. **Chunk 8, re-export and recheck.** Notebooks 09 (2026-11-24) and 10 (2026-11-25, vLLM on a T4: passed, but thinking mode ends without an answer) are done, results in `docs/decisions.md`; left: the local Ollama check (serve with thinking off). Original plan: `zubaan export all` on the adapted model (`adapted/model/` in the Kaggle output of notebook 07; not downloaded locally, 3.48 GB, saved as bfloat16, load as float16 on a T4): GGUF parity (HF vs llama.cpp), Q4_K_M and Q8_0 with the Urdu imatrix, a real Urdu generation on CPU (the metrics cannot say whether the text is readable), Ollama chat sample (watch `<think>` behavior), vLLM check on a T4 (notebook 04 pattern, `dtype half`). A new notebook is needed (`make_09_notebook.py`, reuse `adapt_cells.py` and notebook 03 and 04 cells). Kaggle needs the pinned commit pushed first.
3. **Chunk 9, write-up:** `docs/stage3-usage.md`, decisions, roadmap, the stage table in `CLAUDE.md`; state the limits (small run, noisy Belebele, chat ability untested, gain not attributed to the tokenizer).
4. **Optional control run**: notebook 11 is written (2026-11-25, `make_11_notebook.py`, design in `docs/decisions.md`); needs a push, a new private Hub repo, then a Kaggle T4 run. Original note: (about 9 GPU hours for a full match; a shortened 300 to 400 step control of about 4 h compared at the same step is possible). Matching on text instead of tokens would need over 20 h, so do not do that. Only if the quota allows.

Kaggle quota: about 7 GPU hours were left on 2026-11-19 before notebook 08 used about 2 h (so about 5 h). Check before planning GPU work.

## Stage 3 run facts

- **Main run (notebook 07, T4):** micro batch 1, no gradient checkpointing, 817 to 840 tokens/s, 13.3 GB, 800 steps, about 9.4 h wall. In-run eval (200 Urdu and only 100 English texts) fell from Urdu 2.41 at step 100 to 1.52 at step 800; do not compare it with Stage 2 numbers (use the notebook 08 table). Merge verified, max logit difference 6.5e-05. Log: `reports/notebook7_logs.txt` (gitignored).
- **Speed trial (notebook 06):** micro batch 1 without checkpointing 850 tok/s (13.26 GB) beat micro batch 2 with checkpointing (592 tok/s); micro batch 4 and micro batch 2 without checkpointing ran out of memory.
- **Smoke run (notebook 05):** 30 steps, fp16 stable, kill and resume through the Hub worked.
- **Evaluation (notebook 08):** about 1h55m on a T4 (base 38 min, each extended model about 22 min, mostly Belebele). Pinned to commit `8d16b47`.
- Hub repo `hashiraziz21/zubaan-qwen3-ckpt` (private) holds `last` and milestones at steps 200, 400, 600, 800.
- Kaggle output of notebook 07 (`adapted/model/`) must stay saved (Save Version) to be used as an input later. Paths seen: `/kaggle/input/notebooks/hashiraziz1234/personal-project-notebook7/adapted/model` and `.../personal-project-notebook3/extended/model`.

## New in this session (code)

`zubaan adapt eval --config configs/evaluate-qwen3-1.7b.yaml` (`adapt/evaluate.py`, config section `evaluate`): scores several model folders on identical texts with the Stage 2 protocol, resumes, and prints PASS or FAIL for the decision 9 checks. Notebook 08 (`notebooks/make_08_notebook.py`) wraps it with a held-out hash check, a rehearsal and a base-model reproduction check. 488 tests pass, ruff clean.

## State in one paragraph

On Qwen3-1.7B, 10,000 new Urdu tokens cut the Urdu token cost from 3.14x to 1.29x the English cost, with English tokenization byte-identical and Urdu round-tripping. Stage 3 trained the new rows and LoRA, and the adapted model beats the base on Urdu bits per character (FLORES 2.339 to 1.718) while using 2.29x fewer tokens, with English not damaged and Belebele unchanged.

## What stage 2 did (short)

Learned Urdu BPE merges on cleaned Wikipedia (Arabic-block filter so English cannot change), appended them to Qwen3's `tokenizer.json`, added and initialized embedding rows, saved and verified the folder, exported GGUF (llama.cpp b12477) with a Hugging Face vs llama.cpp parity test, Urdu imatrix, Q4_K_M and Q8_0, Ollama Modelfile and vLLM config, and measured quality next to cost. Full write-up with numbers: `docs/stage2-usage.md` ("What Stage 2 did, and how"). Report: `docs/reports/tokenizer-extension/index.md`.

## Verified

- 0.6B and 1.7B: extend, export, parity (2,041 of 2,045 identical, 4 NFC-only, 0 real mismatches), no converter patch.
- 0.6B: Q4_K_M generates at 34 tokens/s on 4 CPU threads; Ollama 0.40.16 accepts the Modelfile and chats.
- 1.7B: vLLM 0.34.0 on a Kaggle T4 (`dtype half`) served it; tokenization identical; completion and chat work. Needs a GPU with compute capability 7.0 or more, so not a P100.
- Bug found and fixed on 2026-11-18: saved models lacked `tokenizer_config.json` (chat template). Notebook 04 repairs old folders.

## Git state

Everything is pushed up to the commit that carries this file. Notebooks 06 and 07 install zubaan from commit `98fb1a5`, notebook 08 from `8d16b47`. Stage 3 code: `src/zubaan/adapt/` (data, params, cpt, pipeline, finalize), CLI `zubaan adapt prepare | run | merge [--milestone STEP]`, config `AdaptConfig` (new field `milestone_every`). 488 tests pass, ruff clean. Ask before pushing.

## Files you have locally (gitignored)

- `reports/`: `export_summary.json` (1.7B), `vllm_check.json`, `extend_quality_qwen3-1.7b.json`, `extend_sweep.json`, fertility and acceptance results.
- `outputs/qwen3-0.6b/`: extended model and GGUFs. The 1.7B extended model and GGUFs are in the Kaggle output of notebook 03 (`extended/model/`); Stage 3 starts from that folder. Not downloaded locally.

---

# Stage 3 plan: adapt (continued pretraining)

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
- [ ] Chunk 7 report section and chart in `docs/reports/tokenizer-extension/`
- [x] Chunk 8 part 1: notebook 09 re-export on Kaggle CPU (2026-11-24): parity 2041/2045 (4 NFC-only, 0 real), Q4_K_M and Q8_0, quantization cost and samples recorded in `docs/decisions.md`. Files: `reports/export_summary_09.json`, `reports/export_check_09.json`
- [x] Chunk 8 part 2a: notebook 10 vLLM check on a T4 (2026-11-25): tokenization 5/5 identical, Urdu completion and chat (thinking off) work; with thinking allowed the model ends inside `<think>` with no answer. File: `reports/vllm_check_adapted.json`
- [ ] Chunk 8 part 2b: local Ollama chat check with the Q4_K_M file (download `export/qwen3-1.7b-urdu-adapted-Q4_K_M.gguf` and `export/Modelfile` from notebook 09's output)
- [ ] Chunk 9 write-up (`docs/stage3-usage.md`, roadmap and CLAUDE.md stage table)

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

## Next: stage 4 and 5

Stage 4 (drafters sharing the extended vocabulary, distilled from the adapted target) and stage 5 (benchmarks); see `docs/roadmap.md`. Stage 4 uses Qwen3-0.6B with the same merges, which is why the 0.6B was extended with the identical tokenizer.

## Notes for Claude

- Decisions are settled (table above). Continue from the first unchecked item in the status list.
- Read `CLAUDE.md`, `docs/roadmap.md` and `docs/decisions.md` first. Record each design decision in `docs/decisions.md` with date and reason.
- `uv sync --group dev --extra audit --extra report --extra export` before working (add `--extra train` for Stage 3 code); `uv run pytest` and `uv run ruff check .` before saying a task is done. Do not run `ruff format .` on notebooks.
- Never load real models in default tests. Never push to Hugging Face or PyPI without asking.
- Gitignored: `reports/`, `data/`, `outputs/`, `tools/`. Use `newline="\n"` when writing from Python on Windows. No em dashes.
- Kaggle CLI: `uvx --with six kaggle ...`, after `kaggle auth login`. Kaggle notebooks pin a commit that must be pushed first.
- Notebook numbers: 05 Stage 3 smoke (do not rewrite), 06 speed trial, 07 main run, 08 evaluation (04 is the vLLM check). The user asked for new notebooks rather than edits to old ones. Generators: `notebooks/make_0N_notebook.py`; 06 and 07 take `[COMMIT [HUB_REPO]]` and share `notebooks/adapt_cells.py`. Regenerate only after the pinned commit is pushed.
- Run `ruff format` only on files you changed (it reformats old notebooks). Do not chain `git push` after other steps with `&&` and `tail`: a failed lint hid behind it once. GitHub pushes timed out several times; retrying worked.
- Throughput note: `tokens_per_s` in the log covers the time since the last log line, so the line after an eval or upload reads low.
