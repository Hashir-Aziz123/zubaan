# Where we left off (2026-11-18)

Stage 1 and Stage 2 are done. Stage 3 is planned below and needs a few decisions from you before any code is written. To resume, tell Claude: "read continue.md and pick up from there".

## State in one paragraph

On Qwen3-1.7B, 10,000 new Urdu tokens cut the Urdu token cost from 3.14x to 1.29x the English cost, with English tokenization byte-identical and Urdu round-tripping. The new embedding rows are untrained, so Urdu quality is worse for now (held-out Urdu bits per character 2.31 base, 3.57 with mean init; Belebele Urdu back to chance). Stage 3 fixes that.

## What stage 2 did (short)

Learned Urdu BPE merges on cleaned Wikipedia (Arabic-block filter so English cannot change), appended them to Qwen3's `tokenizer.json`, added and initialized embedding rows, saved and verified the folder, exported GGUF (llama.cpp b12477) with a Hugging Face vs llama.cpp parity test, Urdu imatrix, Q4_K_M and Q8_0, Ollama Modelfile and vLLM config, and measured quality next to cost. Full write-up with numbers: `docs/stage2-usage.md` ("What Stage 2 did, and how"). Report: `docs/reports/tokenizer-extension/index.md`.

## Verified

- 0.6B and 1.7B: extend, export, parity (2,041 of 2,045 identical, 4 NFC-only, 0 real mismatches), no converter patch.
- 0.6B: Q4_K_M generates at 34 tokens/s on 4 CPU threads; Ollama 0.40.16 accepts the Modelfile and chats.
- 1.7B: vLLM 0.34.0 on a Kaggle T4 (`dtype half`) served it; tokenization identical; completion and chat work. Needs a GPU with compute capability 7.0 or more, so not a P100.
- Bug found and fixed on 2026-11-18: saved models lacked `tokenizer_config.json` (chat template). Notebook 04 repairs old folders.

## Git state

Local `main` is ahead of GitHub. Pushed through `dbff2e8`. Not pushed: `b74aa9e` (notebook 04), `26ec9bc` (chat template fix), `f913880`, `40f02ba`, and the final docs commit. Ask before pushing.

## Files you have locally (gitignored)

- `reports/`: `export_summary.json` (1.7B), `vllm_check.json`, `extend_quality_qwen3-1.7b.json`, `extend_sweep.json`, fertility and acceptance results.
- `outputs/qwen3-0.6b/`: extended model and GGUFs. The 1.7B extended model and GGUFs are in the Kaggle output of notebook 03 (`extended/model/`); Stage 3 starts from that folder. Not downloaded locally.

---

# Stage 3 plan: adapt (continued pretraining)

Draft, 2026-11-18. Nothing is built yet. `src/zubaan/adapt/` is empty and `AdaptConfig` in `config.py` only has three fields (`model_path`, `corpus`, `english_replay_ratio`). The `train` extra already lists `peft`, `trl` and `accelerate`.

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

## Success checks (proposed, confirm in decision 9)

1. English tokenization still byte-identical and Urdu still round-trips (the tokenizer is not touched, so this must stay true; the verify step re-checks).
2. Held-out Urdu bits per character clearly below the untrained 3.57. Stretch goal: at or below the base model's 2.31 even though the extended model needs 2.4x fewer tokens.
3. English FLORES bits per character within a stated tolerance of the base (1.208). Proposed: at most +0.02.
4. Belebele Urdu `acc_char` above the base 33.4%, reported with its interval. With 900 questions the 95% interval is about plus or minus 3 points, so a gain under about 4 points is not distinguishable from noise and must be described that way.
5. Parity still passes on the final GGUF, and a Q4_K_M answers a simple Urdu prompt in readable Urdu.

## Decisions you need to make

Each one has my recommendation. I will use the recommendation if you just say "go with the defaults".

1. **Which model do we continue from?** `Qwen/Qwen3-1.7B` is the post-trained chat and thinking model, not `Qwen3-1.7B-Base`. Training it on raw Urdu text will blunt its instruction following and `<think>` behavior.
   - Option A: continue from the existing extended model (no rework, Stage 2 artifacts, Ollama and vLLM setups all target it).
   - Option B: rebuild the extension on `Qwen3-1.7B-Base` (about 1 hour CPU notebook, assumes the tokenizer is identical, which we must verify with `same_tokenizer`) and train that.
   - **Recommend A.** Our metrics (bits per character, Belebele log-likelihood scoring, speed, acceptance) do not need chat ability, and Stage 4 and 5 reuse the current artifacts. State the chat caveat in the report. Revisit B only if instruction following matters for the final demo.
2. **What is trainable?** Choices: (a) new rows only, (b) new rows plus LoRA, (c) the whole embedding matrix (331M parameters, so Adam state about 4 GB extra) plus LoRA.
   - **Recommend two phases, A then B as described above**, with old embedding rows frozen. Unfreezing old rows risks English drift and costs memory. The code should freeze old rows by construction (a separate trainable tensor for the new rows) instead of a gradient mask, because a mask still allocates optimizer state for the full matrix and peft's tied-embedding handling is easy to get wrong.
3. **LoRA shape.** Starting point: rank 64, alpha 128, dropout 0.05, all attention and MLP linear layers; learning rate about 2e-4 for LoRA and higher for the new rows (about 1e-3). These are starting values to check in the smoke run, not tuned numbers. **Recommend these defaults**, adjusted only if the smoke run shows instability.
4. **Urdu data.** Wikipedia only (about 14M tokens, already clean, FLORES-guarded, licence known) or add a larger source such as CC-100 `ur` (more text, licence terms to check, needs cleaning and the same overlap guard).
   - **Recommend Wikipedia only first.** One to three epochs over 14M tokens risks mild overfitting, so we watch held-out Urdu bits per character every N steps. Add a bigger source only if that curve is still falling when the data runs out.
5. **English replay.** Source and share. Default share in `AdaptConfig` is 10%. **Recommend English Wikipedia (same licence family as the Urdu text), 10% of blocks.** Say if you prefer another source.
6. **Control experiment.** Without it we cannot tell whether gains come from the new tokens or from simply training on more Urdu. The control: base Qwen3-1.7B with the **base** tokenizer, same LoRA recipe, same Urdu lines (it needs about 2.4x more tokens for the same text, so more GPU time).
   - **Recommend yes, but after the main run and only if GPU hours allow.** It is what makes the result credible. Decide whether to match text (same lines) or compute (same number of tokens); recommend matching text.
7. **Precision.** T4 has no BF16. Options: fp16 autocast with a gradient scaler (fast, fp16 overflow is a known risk for some Qwen layers) or fp32 base weights (about 7 GB for the model, safer, slower). **Recommend trying fp16 in the smoke run and switching to fp32 only if the loss goes NaN or inf.** Stage 2 evaluation already ran in float32 on a T4 without issue.
8. **Hugging Face Hub.** Checkpoint resume needs a place to push. `CLAUDE.md` says never push to the Hub without asking. **Please approve, or refuse:** a **private** repo (name of your choice) for checkpoints, with the token stored as a Kaggle secret. Nothing is made public until you decide in Stage 5. If you refuse, resume works only inside one session and a crash loses the run.
9. **Success thresholds and GPU budget.** Confirm or change the checks above (English at most +0.02 bits per character; Urdu below 3.57 as the must, 2.31 as the stretch). GPU budget: **recommend a cap of about 10 GPU hours for the main run**, set after the smoke run gives real tokens per second. Kaggle gives about 30 hours a week.
10. **Defaults I will pick unless you object:** block size 1024, one T4, gradient checkpointing on, plain `transformers` training loop instead of `trl` (raw-text pretraining does not need it, and it avoids version conflicts like the `langsfer` one), seed fixed and logged, notebook number 05.

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

- First action next session: ask the user the decisions above (or get "go with the defaults"), then start at chunk 0. Do not write training code before decisions 1, 2 and 8 are answered.
- Read `CLAUDE.md`, `docs/roadmap.md` and `docs/decisions.md` first. Record each design decision in `docs/decisions.md` with date and reason.
- `uv sync --group dev --extra audit --extra report --extra export` before working (add `--extra train` for Stage 3 code); `uv run pytest` and `uv run ruff check .` before saying a task is done. Do not run `ruff format .` on notebooks.
- Never load real models in default tests. Never push to Hugging Face or PyPI without asking.
- Gitignored: `reports/`, `data/`, `outputs/`, `tools/`. Use `newline="\n"` when writing from Python on Windows. No em dashes.
- Kaggle CLI: `uvx --with six kaggle ...`, after `kaggle auth login`. Kaggle notebooks pin a commit that must be pushed first.
- Notebook number 05 is for Stage 3 (04 is the vLLM check).
