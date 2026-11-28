# Where we left off (2026-11-28)

Stages 1, 2 and 3 are done. **Stage 4 (restore chat behavior) is in progress:** the training data is made and checked, the SFT main run (notebook 14) finished on 2026-11-28, and the evaluation (notebook 15) passed checks 1 to 5 the same day. Next is the re-export (chunk 8). Plan: `docs/stage4-plan.md`. Every Stage 4 decision and result so far: `docs/decisions.md`, entries from "Stage 4: restore chat behavior" (2026-11-25) onwards. To resume, tell Claude: "read continue.md and pick up from there".

## Next steps

1. Done 2026-11-28: notebook 14 main run (held-out loss 0.970, merged into `tuned/model/`, 8 of 8 probe replies answered with think blocks closed; one wrong fact). Its committed output is the Stage 4 input to notebook 15.
2. Done 2026-11-28, results in `docs/decisions.md` ("Stage 4: evaluation results"). **Notebook 15 (evaluation)** on Kaggle: GPU T4 x2, internet on, no secret. Inputs: the outputs of notebook 07 (Stage 3 model) and notebook 14 (Stage 4 model). About 1.5 to 2 hours. It chat-checks base, Stage 3 and Stage 4 through vLLM (this is also the baseline the chunk 1 laptop runs never finished), scores quality on the Stage 3 protocol, and prints PASS or FAIL for checks 1 to 5. Send back `tune_eval.zip`. Record the results in `docs/decisions.md`. Notebook 15 is pinned to `2e0634e`; the only code change since then is in `tune/sft.py`, which notebook 15 does not use, so it does not need regenerating.
3. Done 2026-11-28: **check 6, the person read**, by Claude. Urdu is better than the base but loops on long answers and gets facts wrong; English reasoning with thinking off got worse (1 of 4 right, base 4).
4. **Chunk 8, re-export: notebook 16 is ready** (pinned to `aebd864`; Kaggle CPU, about 3 to 3.5 h; input: notebook 14's output only). It exports GGUF with parity, Q4_K_M and Q8_0, and runs the chat check on Q4_K_M with checks 1 to 4. The author sends back `export_results.zip` and downloads `export/qwen3-1.7b-urdu-chat-Q4_K_M.gguf` into `outputs/`; then Claude runs the local Ollama check (does thinking work with Ollama's default?). vLLM is covered by notebook 15.
5. **Chunk 9, write-up:** `docs/stage4-usage.md` is drafted (its Serving section waits for notebook 16). Still to do: the Stage 4 section of the extension report (`report/`), final roadmap and `CLAUDE.md` stage table, close the stage in `docs/decisions.md`.

If a check fails in step 2, stop and decide with the author before changing anything (options in `docs/stage4-plan.md`: train the new rows, more epochs, a second round of data).

Optional, any time: the Stage 3 compute-matched control run (notebook 11, guide `docs/stage3-control-run.md`, about 10 Kaggle GPU hours, needs a new private Hub repo).

## Stage 4 so far in short

| Chunk | What | State |
| --- | --- | --- |
| 0 | Teacher trial (notebook 12) | Done. Teacher Qwen3-14B-AWQ; standalone questions only with thinking off; mix of 10,000 conversations |
| 1 | Chat check `zubaan tune check`, prompt set v1 (40 Urdu, 20 English, approved by the author) | Code done. Baselines run in notebook 15 (the laptop ran out of memory twice) |
| 2, 3 | Questions and teacher answers (notebook 13, 2 shards, about 3.6 h each) | Done. 11,502 answers |
| 4 | `zubaan tune filter` and the read sample | Done. 9,854 kept; the 150-sample read passed 71% (see below) |
| 5 | SFT code `zubaan tune run \| merge` | Done |
| 6 | Notebook 14: smoke, then main | Done (held-out loss 0.970, merged) |
| 7 | Notebook 15: evaluation | Done: checks 1 to 5 PASS, check 6 read by Claude |
| 8 | Re-export (notebook 16) | **Next**: ready to run |
| 9 | Write-up | `docs/stage4-usage.md` drafted |

**Training data.** 11,502 generated, 10,429 passed the filters, 9,854 kept after balancing. Drops: loop 504, mixed-script word 214, failed request 203 (all client timeouts with thinking on), wrong language 114, token limit 19, think block in the wrong mode 10, source mention 9. Every slot full except Urdu grounded with thinking (3,854 of 4,000). Train 9,669 conversations, 4.57M tokens (3.31M in replies), median 422, longest 2,707; held out 185 (2% of passages). 49% with thinking on. Packed into 1,116 rows of 4,096, so 140 steps for 2 epochs at 16 rows per step.

**The read sample (done by Claude, not a native speaker).** 106 of 150 pass (71%): English 28 of 31, Urdu 78 of 119 (66%). Urdu fails are mostly wrong or invented words, misread passages, verb errors and some wrong facts. Two faults are findable by pattern (the Prophet's honorific after a companion's name, letters from other scripts): about 2% of the Urdu data. The run was not stopped for them; this is a stated limitation, and any regeneration should add those two filters and consider a stronger Urdu teacher.

**Smoke run (notebook 14, T4, commit `c5c1309`).** Packing check in float32 1.06e-4. Held-out loss 1.060 at step 20, 1.032 at step 30. Resume from the Hub worked. 658 tokens/s, peak 12.4 GB, so gradient checkpointing stays on.

**Two bugs found on the way, both fixed in `c5c1309`:** the packing check failed in fp16 on rounding alone (0.082), so it now runs in float32 before the cast; and Transformers 5.22 only isolates packed conversations when no cache is passed, so every forward call passes `use_cache=False`.

## Stage 3 result in short

| | Urdu tokens/char (FLORES) | BPC Urdu FLORES | BPC Urdu held-out | BPC English | Belebele Urdu acc_char |
| --- | --- | --- | --- | --- | --- |
| Base | 0.668 | 2.339 | 2.310 | 1.208 | 0.334 [0.304, 0.366] |
| Extended, untrained | 0.275 | 3.562 | 3.570 | 1.210 | 0.253 |
| Adapted | 0.275 | 1.718 | 1.486 | 1.066 | 0.340 [0.310, 0.372] |

2.43x fewer tokens on FLORES Urdu and 0.62 fewer bits per character than the base, English not damaged, Belebele no change. Not attributed to the tokenizer until the control run. With thinking on, the Stage 3 model usually ends its turn without an answer; that is what Stage 4 repairs. Full write-up: `docs/stage3-usage.md`; the original plan: `docs/stage3-plan.md`.

## Kaggle and Hub facts

- Hub repos (private): `hashiraziz21/zubaan-qwen3-ckpt` (Stage 3), `hashiraziz21/zubaan-qwen3-tune-ckpt` (Stage 4) and its `-smoke` twin. Token only as a Kaggle secret.
- Notebook pins: 06 and 07 `98fb1a5`, 08 `8d16b47`, 09 and 10 `59f2708`, 11 `ad7692d`, 12 `751ae11`, 13 `4c73860`, 14 `c5c1309` (Hub repo filled in), 15 `2e0634e`.
- Notebook 07's output (`adapted/model/`) must stay saved; notebooks 14 and 15 take it as an input. Path seen: `/kaggle/input/notebooks/hashiraziz1234/personal-project-notebook7/adapted/model`.
- GPU notebooks need the accelerator set in Session options (notebook 13 once stopped with "No GPU in this session"). vLLM notebooks need GPU T4 x2 and internet on.

## Files you have locally (gitignored)

- `data/tune/`: `conversations.jsonl`, `curate_report.json`, `review.md`, `review.csv` (marked).
- `reports/teacher_trial/`, `reports/tune_data_shard0/`, `reports/tune_data_shard1/`: notebook 12 and 13 outputs. `reports/tune_eval/`: notebook 15 (chat replies of all three models, quality, checks). `reports/smoke_report.json`, `reports/tune_summary.json`, `reports/probe.json`: notebook 14 smoke and main. `reports/chat_check_base.partial.jsonl`: 15 base replies from the abandoned laptop baseline.
- Stage 3: `outputs/adapted-export/` (Q4_K_M and Modelfile), Ollama model `zubaan-urdu-adapted` in `tools/ollama-models`, `reports/notebook7_logs.txt`, `reports/export_*_09.json`, `reports/vllm_check_adapted.json`.

## Notes for Claude

- Read `CLAUDE.md`, `docs/roadmap.md` and `docs/decisions.md` first. Record each design decision and each result in `docs/decisions.md` with date and reason.
- The author is not an Urdu speaker. Claude reads Urdu output and says so wherever that read is recorded.
- When the author asks for a plan or an explanation, give only that and take no action. Explain in short, plain terms and say clearly when the author's help is needed.
- `uv sync --group dev --extra audit --extra report --extra export --extra train` before working; `uv run pytest` (656 pass) and `uv run ruff check .` before saying a task is done. Run `ruff format` only on files you changed (it reformats old notebooks).
- Never load real models in default tests. Never push to Hugging Face or PyPI without asking. Do not restart a run the host stopped for memory without asking.
- Gitignored: `reports/`, `data/`, `outputs/`, `tools/`. Use `newline="\n"` when writing from Python on Windows. No em dashes.
- Notebooks: generators `notebooks/make_NN_notebook.py`; the user wants new notebooks rather than edits to old ones. Kaggle installs zubaan from a pinned commit, so push first and regenerate after. `make_14_notebook.py` takes `COMMIT HUB_REPO`; leaving out the second argument blanks the Hub repo.
- Kaggle CLI: `uvx --with six kaggle ...`, after `kaggle auth login`. GitHub pushes sometimes time out; retrying works. Do not chain `git push` after other steps with `&&` and `tail`.
- Throughput note: `tokens_per_s` in a training log covers the time since the last log line, so the line after an eval or upload reads low.
