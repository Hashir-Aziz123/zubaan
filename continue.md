# Where we left off (2026-11-28)

Stages 1, 2, 3 and 4 are done. **Stage 4 (restore chat behavior) closed on 2026-11-28:** the chat-tuned model answers again with thinking on (60 of 60 prompts, Stage 3 13), replies to Urdu in Urdu, keeps the Stage 3 gains, and keeps that behavior as a 4-bit GGUF in llama.cpp and Ollama. Write-up: `docs/stage4-usage.md`. Report: `docs/reports/tokenizer-extension/` (Stage 4 section). Every decision and result: `docs/decisions.md`. To resume, tell Claude: "read continue.md and pick up from there".

## Next steps

1. **Stage 5 (draft) needs a plan first**, like `docs/stage4-plan.md` was for Stage 4: a drafter that shares the extended tokenizer (Qwen3-0.6B extended the same way and distilled on the Stage 4 model's outputs), an n-gram drafter, and acceptance before and after, Urdu against English (`docs/roadmap.md`, Stage 5). The target is the Stage 4 model (`qwen3-1.7b-urdu-chat`); its tokenizer is the Stage 3 extended one.
2. Optional, any time: the Stage 3 compute-matched control run (notebook 11, guide `docs/stage3-control-run.md`, about 10 Kaggle GPU hours, needs a new private Hub repo).
3. Optional later Stage 4 data round (not a blocker): worked thinking-off answers for reasoning prompts, the two pattern filters from the data review, a stronger Urdu teacher or a second checking pass, multi-turn data.

## Stage 4 in short

| | Thinking on: answered | Thinking off: answered | Urdu prompts in Urdu (on / off) | FLORES Urdu bpc |
| --- | --- | --- | --- | --- |
| Base | 58/60 | 60/60 | 32/39, 37/39 | 2.339 |
| Stage 3 | 13/60 | 60/60 | 8/39, 37/39 | 1.718 |
| Stage 4 (float16, vLLM) | 60/60 | 60/60 | 38/39, 38/39 | 1.735 |
| Stage 4 Q4_K_M (llama-server) | 60/60 | 60/60 | 38/39, 39/39 | |

- Checks 1 to 5 pass; check 6 (reading the replies, by Claude, not a native speaker): Urdu clearly better than the base but repeats on long answers and gets facts wrong; English reasoning with thinking off got worse (1 of 4, base 4).
- Ollama on the laptop: thinking works with Ollama's default (4 of 4 prompts, Stage 3 1 of 4), 10 to 15 tokens per second.
- GGUF parity 2,041/2,045 (4 NFC-only, 0 real). Q4_K_M costs Urdu about what it costs English (KLD 0.029 and 0.030).
- About 15 Kaggle GPU hours against 10 budgeted.

## Stage 3 result in short

2.43x fewer tokens on FLORES Urdu and 0.62 fewer bits per character than the base (2.339 to 1.718), English not damaged, Belebele no change. Not attributed to the tokenizer until the control run. Full write-up: `docs/stage3-usage.md`.

## Kaggle and Hub facts

- Hub repos (private): `hashiraziz21/zubaan-qwen3-ckpt` (Stage 3), `hashiraziz21/zubaan-qwen3-tune-ckpt` (Stage 4) and its `-smoke` twin. Token only as a Kaggle secret.
- Notebook pins: 06 and 07 `98fb1a5`, 08 `8d16b47`, 09 and 10 `59f2708`, 11 `ad7692d`, 12 `751ae11`, 13 `4c73860`, 14 `c5c1309`, 15 `2e0634e`, 16 `aebd864`.
- Notebook 07's output (`adapted/model/`) and notebook 14's output (`tuned/model/`, the Stage 4 model) must stay saved; later notebooks take them as inputs.
- GPU notebooks need the accelerator set in Session options. vLLM notebooks need GPU T4 x2 and internet on.

## Files you have locally (gitignored)

- `outputs/adapted-export/`: Stage 3 Q4_K_M and Q8_0 with `Modelfile`; Stage 4 `qwen3-1.7b-urdu-chat-Q4_K_M.gguf` with `Modelfile.chat`. Ollama models `zubaan-urdu-adapted` and `zubaan-urdu-chat` in `tools/ollama-models` (start the portable server with `OLLAMA_MODELS` set to that folder).
- `reports/tune_eval/` (notebook 15), `reports/export_16/` (notebook 16), `reports/ollama_check_chat.json` (Stage 4 Ollama check); `reports/teacher_trial/`, `reports/tune_data_shard0/`, `reports/tune_data_shard1/` (notebooks 12 and 13); `reports/smoke_report.json`, `reports/tune_summary.json`, `reports/probe.json` (notebook 14).
- `data/tune/`: `conversations.jsonl`, `curate_report.json`, `review.md`, `review.csv` (marked).
- Stage 3: `reports/notebook7_logs.txt`, `reports/export_*_09.json`, `reports/vllm_check_adapted.json`, `reports/ollama_check_adapted.json`.
- `uv run zubaan report extend` rebuilds the extension report from these files; missing ones are left out with a note.

## Notes for Claude

- Read `CLAUDE.md`, `docs/roadmap.md` and `docs/decisions.md` first. Record each design decision and each result in `docs/decisions.md` with date and reason.
- The author is not an Urdu speaker. Claude reads Urdu output and says so wherever that read is recorded.
- When the author asks for a plan or an explanation, give only that and take no action. Explain in short, plain terms and say clearly when the author's help is needed.
- `uv sync --group dev --extra audit --extra report --extra export --extra train` before working; `uv run pytest` (666 pass) and `uv run ruff check .` before saying a task is done. Run `ruff format` only on files you changed (it reformats old notebooks and Markdown code blocks).
- Never load real models in default tests. Never push to Hugging Face or PyPI without asking. Do not restart a run the host stopped for memory without asking.
- Gitignored: `reports/`, `data/`, `outputs/`, `tools/`. Use `newline="\n"` when writing from Python on Windows. No em dashes.
- Notebooks: generators `notebooks/make_NN_notebook.py`; the user wants new notebooks rather than edits to old ones. Kaggle installs zubaan from a pinned commit, so push first and regenerate after. `make_14_notebook.py` takes `COMMIT HUB_REPO`; leaving out the second argument blanks the Hub repo.
- Kaggle CLI: `uvx --with six kaggle ...`, after `kaggle auth login`. GitHub pushes sometimes time out; retrying works. Do not chain `git push` after other steps with `&&` and `tail`.
- Throughput note: `tokens_per_s` in a training log covers the time since the last log line, so the line after an eval or upload reads low.
