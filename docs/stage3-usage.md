# Stage 3 usage: train the new tokens and serve the result

Stage 2 made Urdu cheaper: 10,000 new Urdu tokens cut the token cost of Urdu on Qwen3-1.7B from 3.14x to 1.29x that of the same English text. But those tokens' embedding rows were only initialized, never trained, so Urdu quality got worse. Stage 3 trains them with continued pretraining, checks quality against the base model, and exports the trained model again for llama.cpp, Ollama and vLLM.

**Read this first.** The adapted model scores better than the base on Urdu bits per character while using far fewer tokens, and English was not damaged. But this is one small training run (26M tokens on one T4), not a new model to hand to users:

- Belebele Urdu (reading comprehension) did not measurably change.
- The gain has not yet been attributed to the tokenizer. The compute-matched control (notebook 11) answers that, and it has not been run.
- Chat behavior degraded. With thinking on, the model often ends its turn after its reasoning without answering (3 of 4 prompts in Ollama). Serve it with thinking off, which has to be set per request.
- It invents facts and falls into repetition loops on open-ended prompts.

Status (2026-11-25): **Stage 3 is done.** Training, evaluation, re-export, the vLLM check, the local Ollama check and the report are complete. The optional compute-matched control run (notebook 11) is written but not run.

## What Stage 3 did, and how

**Why training is needed.** A transformer reads a token through its embedding row and predicts through the output head row; Qwen3-1.7B uses one tied matrix for both. After Stage 2, the 10,000 new rows were the mean of the embeddings of the pieces each token replaces. The layers above had never seen these tokens in context, so held-out Urdu bits per character rose from 2.31 to 3.57 and Belebele Urdu fell to chance.

**What was trained.** Full fine-tuning of 1.7B parameters does not fit on a 16 GB T4 (AdamW needs about 16 bytes per parameter), so two small sets of parameters were trained:

1. **The new embedding rows** (10,000 x 2,048 = 20.5M values), kept in their own float32 tensor and applied with forward hooks, so the 151,669 old rows cannot change. The old rows were checked to be bit-identical after training.
2. **LoRA adapters** of rank 64 on all seven attention and MLP projections.

Training ran in two phases: 100 steps that train only the new rows (so the adapters do not learn to work around bad rows), then 700 steps that train both. Each step is 32 blocks of 1,024 tokens, about 33k tokens, so the run saw 26.2M tokens in total.

**Data.** Urdu Wikipedia, cleaned and normalized: 16.05M tokens with the extended tokenizer, in 15,677 blocks. English Wikipedia makes up 10% of blocks as replay, to guard against forgetting. Every training line that shares 8 consecutive words with FLORES (dev or devtest) or a Belebele passage was dropped (47 Urdu lines), so the evaluation text was never trained on.

**How it ran.** fp16 with a loss scaler on one Kaggle T4 (no BF16 on a T4), micro batch 1 without gradient checkpointing (the fastest setting that fit, from a speed trial), 817 to 850 tokens/s, 13.3 GB peak, about 9.4 hours. A checkpoint went to a private Hugging Face repo every 50 steps, and a run that stops resumes with exactly the same data and learning rate.

**Merge and verify.** LoRA was folded into the weights in float32, and the trained rows were written into the tied matrix. The merge refuses to save if the logits change by more than 1e-3 (they changed by 6.5e-5) or if any old row differs. The saved folder passes the Stage 2 checks: English tokenizes identically and Urdu round-trips.

**What we found.** All three models were scored on the same texts with the Stage 2 protocol (1,000 texts per set, 900 Belebele questions per language, float32). The base model reproduced its Stage 2 numbers to three decimals.

| | Base | Extended, untrained | Adapted |
| --- | ---: | ---: | ---: |
| Urdu tokens per character (FLORES) | 0.668 | 0.275 | 0.275 |
| Bits per character, FLORES Urdu | 2.339 | 3.562 | **1.718** |
| Bits per character, held-out Urdu | 2.310 | 3.570 | 1.486 |
| Bits per character, FLORES English | 1.208 | 1.210 | 1.066 |
| Belebele Urdu `acc_char` (95% interval) | 33.4% (30.4 to 36.6) | 25.3% | 34.0% (31.0 to 37.2) |

On the same FLORES Urdu text, the adapted model uses 2.43x fewer tokens than the base and scores 0.62 fewer bits per character. Full table, chart and reading: [the extension report](reports/tokenizer-extension/index.md), section "Stage 3".

## The commands at a glance

| Command | What it does | Needs a GPU | Downloads |
| --- | --- | --- | --- |
| `zubaan adapt prepare` | Tokenize Urdu and English replay into blocks, drop lines that overlap the evaluation text | No | English Wikipedia, Belebele |
| `zubaan adapt run` | Train (new rows, then LoRA), checkpoint, resume; `--steps N`, `--stop-at STEP` | Yes | No |
| `zubaan adapt merge` | Merge LoRA and the rows into a plain model, save and verify; `--milestone STEP` | No, but the float32 merge does not fit in 8 GB RAM (notebook 07 runs it) | the Hub checkpoint |
| `zubaan adapt eval` | Score model folders on the same texts, print the success checks | Yes in practice | Belebele, base model |
| `zubaan export all` / `export check` | GGUF with parity, quantize, Modelfile, vLLM config; samples and quantization cost | No | No |
| `zubaan report extend` | Rebuild the extension report, now with the Stage 3 section | No | No |

Every command takes `--config path.yaml` (`configs/adapt-qwen3-1.7b.yaml`, `configs/adapt-control-qwen3-1.7b.yaml`, `configs/evaluate-qwen3-1.7b.yaml`).

## Setup

```bash
uv sync --group dev --extra audit --extra train      # torch, transformers, peft, accelerate
uv sync --group dev --extra audit --extra report --extra export --extra train   # everything
```

The training code was tested on a tiny random model on CPU (`uv run pytest`). Real runs need a GPU with about 16 GB.

## Running it on Kaggle

The GPU steps are notebooks generated by `notebooks/make_NN_notebook.py`. Each one installs zubaan from a pinned commit, so push that commit before running it.

| Notebook | Purpose | Machine | Time | Input |
| --- | --- | --- | --- | --- |
| 05 | Version check, 5 fp16 steps, 30-step smoke run with a kill and resume | T4 | about 1 h | output of 03 |
| 06 | Speed trial: four batch settings, picks the fastest that fits | T4 | about 1 h | output of 03 |
| 07 | Main run (800 steps), merge, save | T4 | about 9.5 h | output of 03; `HF_TOKEN` secret |
| 08 | Evaluation of base, untrained and adapted | T4 | about 2 h | outputs of 03 and 07 |
| 09 | Re-export: GGUF, parity, Q4_K_M and Q8_0, samples, quantization cost | CPU | about 1.5 h | output of 07 |
| 10 | vLLM check of the adapted model | T4 | about 25 min | output of 07 |
| 11 | Compute-matched control (optional, not run yet) | T4 | about 10 h | none; `HF_TOKEN` secret |

Notebook 11 has its own guide: [Stage 3 control run](stage3-control-run.md).

Locally, the same steps without notebooks:

```bash
uv run zubaan adapt prepare --config configs/adapt-qwen3-1.7b.yaml
uv run zubaan adapt run     --config configs/adapt-qwen3-1.7b.yaml --steps 30   # smoke test
uv run zubaan adapt run     --config configs/adapt-qwen3-1.7b.yaml              # full run
uv run zubaan adapt merge   --config configs/adapt-qwen3-1.7b.yaml
uv run zubaan adapt eval    --config configs/evaluate-qwen3-1.7b.yaml
uv run zubaan report extend
```

## Serving the adapted model

Use **thinking off** everywhere. With thinking on, the model writes an English reasoning block and then often ends its turn with no answer: once in the vLLM check, and on 3 of 4 prompts in the Ollama check, including an English one. Training a chat model on plain text taught it to end documents. With thinking off, every prompt got an answer.

**llama.cpp.** Notebook 09 wrote `qwen3-1.7b-urdu-adapted-Q4_K_M.gguf` (1.12 GB) and `Q8_0` (1.86 GB), both quantized with an importance matrix from Urdu text. Tokenizer parity passed: 2,041 of 2,045 texts give identical ids, the other 4 differ only because the text is not in NFC form, and none differ for real. Send NFC-normalized text (`zubaan.text.normalize_urdu` does this). On 4 Kaggle CPU threads, Q4_K_M generated about 15 tokens/s.

**Ollama.** Put the GGUF next to the Modelfile that `zubaan export ollama` writes, then turn thinking off **per request**:

```bash
ollama create zubaan-urdu -f Modelfile
ollama run zubaan-urdu --think=false "پاکستان کا دارالحکومت کیا ہے؟"
```

```bash
curl http://localhost:11434/api/chat -d '{"model": "zubaan-urdu", "think": false,
  "messages": [{"role": "user", "content": "پاکستان کا دارالحکومت کیا ہے؟"}]}'
```

A Modelfile cannot do this. Ollama 0.40.16 ignored the Modelfile's `TEMPLATE` for this GGUF and used the Qwen3 chat template embedded in the file (`/api/show` lists it, with the `thinking` capability). It also rejects `PARAMETER think`. With thinking on, Ollama returns the reasoning in a separate `thinking` field, so a reply that ends after thinking looks like an empty answer.

Measured on the 8 GB laptop's CPU (Ollama 0.40.16, Q4_K_M, greedy, 300 tokens at most; replies in `reports/ollama_check_adapted.json`), with `think: false`:

| Prompt | Reply |
| --- | --- |
| "What is the capital of Pakistan?" (Urdu) | "The capital of Pakistan is Islamabad." in Urdu, 8 tokens |
| "Which province is Lahore in?" (Urdu) | Punjab, and that Lahore is its capital, in Urdu: correct |
| "Write three sentences about the Urdu language." (Urdu) | Fluent Urdu with false claims (the most spoken language in the world), then one sentence repeated until the 300-token limit |
| "What is the capital of France?" (English) | "The capital of France is Paris." |

Speed was 9 to 13 tokens per second. With thinking on, the capital, Lahore and France prompts ended after the thinking with no answer, and the thinking for Lahore placed it in "the state of Punjab in northern India". The base model was not run through Ollama, so this test does not show how much of the thinking failure is new.

**vLLM.** `zubaan export vllm` writes a config with `dtype: half` (T4 has no BF16). vLLM 0.34.0 served the adapted model on a T4, and its token ids matched `tokenizer.json`. Turn thinking off per request:

```json
{
  "model": "qwen3-urdu-adapted",
  "messages": [{"role": "user", "content": "پاکستان کا دارالحکومت کیا ہے؟"}],
  "chat_template_kwargs": {"enable_thinking": false}
}
```

## How to read the results

- **Compare bits per character, never per-token numbers, across tokenizers.** The adapted model predicts fewer, longer tokens, so loss per token and tokens per second are not comparable with the base.
- **FLORES Urdu is the cleaner Urdu number.** The held-out lines come from the same Wikipedia dump as the training text, so they flatter the result (1.486 against 1.718).
- **The English drop (1.208 to 1.066) is not evidence that English improved.** The base is a chat model, and the English replay is Wikipedia, close to FLORES in style. It does show English was not damaged on this metric.
- **Belebele did not move.** 34.0% against 33.4%, with intervals of about 6 points. Better likelihood did not turn into better reading comprehension.

## Limitations

- One run, one seed, 26M tokens, one T4. A small continued-pretraining run, not a new foundation model.
- The gain is not attributed to the tokenizer. The adapted model saw 16M tokens of Urdu that the base never saw; notebook 11 trains the base on the same budget.
- Chat and instruction behavior were not measured systematically. What was seen: thinking mode can end without an answer, repetition loops on open-ended prompts, and invented facts (for example Islamabad called the largest city, and inconsistent population figures). All of it comes from greedy samples read by a person.
- Belebele zero-shot is coarse for a 1.7B model, and only Wikipedia Urdu (formal register) was used for training.
- Thinking mode is broken in practice: it must be turned off per request in Ollama and vLLM. The base model was not put through the same chat test, so how much of this is new is not measured.
