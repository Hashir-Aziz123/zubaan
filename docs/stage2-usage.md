# Stage 2 usage: extend a model and export it

Stage 1 measured the problem. Stage 2 fixes the token cost: it adds Urdu tokens to a model's vocabulary, then packages the result as GGUF (llama.cpp, Ollama) and a vLLM config.

**Read this first.** After Stage 2 the model has **not learned** its new tokens. The new embedding rows are only initialized, so Urdu quality is worse than the base model until Stage 3 trains them (see the results in [the extension report](reports/tokenizer-extension/index.md)). Stage 2 lowers the token cost; it is not a model to hand to users yet.

Status: **Stage 2 is done** (2026-11-18). Qwen3-0.6B and 1.7B are extended and exported, GGUF parity passed on both, and vLLM served the 1.7B.

Currently Urdu only, on the Qwen3 family. The code is language-agnostic except for the Urdu-specific normalization and the Arabic-script merge filter.

## What Stage 2 did, and how

**The problem.** Qwen3's tokenizer was built mostly from English. On FLORES-200 (the same sentences in both languages) Urdu needs 3.14 times as many tokens as English. Every token costs money, takes a forward pass, and uses context.

**The fix, in five steps.**

1. **Learn Urdu merges.** Byte-pair encoding (BPE) builds a vocabulary by repeatedly merging the most frequent adjacent pair. We continued Qwen3's own BPE on cleaned, normalized Urdu Wikipedia text (FLORES removed so the test sentences are never seen) and kept only merges whose UTF-8 bytes lead with the Arabic block (`0xD8` to `0xDB`). That filter is why English tokenization cannot change. The new merges are appended to the end of `tokenizer.json` as real merges, not `added_tokens`, and the special-token ids stay pinned.
2. **Give the new tokens embedding rows.** The embedding matrix and LM head get one new row per new token. Rows start from the mean of the embeddings of the pieces the token used to split into (default), a FOCUS-style similarity average, or random. Nothing is trained, so these are guesses.
3. **Save and verify the model folder** (`extend run`). The checks: only `model.vocab` and `model.merges` differ from the base tokenizer, English tokenizes identically, Urdu round-trips, and the config and weights have one row per id.
4. **Export** (`export all`). Convert to GGUF with llama.cpp `b12477`, **prove tokenizer parity** (Hugging Face against `llama-tokenize` on 2,045 texts), build an importance matrix from Urdu text, and quantize to Q4_K_M and Q8_0. Write an Ollama Modelfile and a vLLM config with `dtype half`.
5. **Measure quality next to cost.** Bits per character and Belebele, so a cheaper tokenizer cannot hide a worse model.

**What we found.**

| Result | Value |
| --- | --- |
| Urdu cost multiplier, base | 3.14x |
| With 5k / 10k / 20k / 32k new tokens | 1.40x / 1.29x / 1.22x / 1.20x |
| English tokenization after extension | byte-identical to the base |
| Held-out Urdu bits per character (1.7B) | 2.31 base, 3.57 with 10k mean init, 3.63 focus, 5.27 random |
| Belebele Urdu `acc_char` | 33.4% base, 24.9% to 27.0% extended (chance 25%) |
| Parity, Qwen3-1.7B | 2,041 of 2,045 identical, 4 differ only by NFC, 0 real mismatches |
| GGUF sizes, Qwen3-0.6B | Q4_K_M 405 MB, Q8_0 651 MB |
| vLLM 0.34.0 on a Kaggle T4 | tokenization identical, completion and chat answer |

So the tokenizer is much cheaper, but the model is worse at Urdu until it learns the new rows. That is Stage 3. We chose 10,000 tokens with mean-of-pieces init as the default because it keeps most of the saving (1.29x) with the least quality loss; random init is clearly worst.

**Things found along the way** (all in `docs/decisions.md`):

- The converter's tokenizer fingerprint uses a test string with no Arabic characters, so the extended tokenizer hashes the same as Qwen's and needs no patch. The parity test, not that hash, is the proof.
- llama.cpp does not apply NFC and Qwen's Hugging Face tokenizer does, so 4 FLORES lines with stacked diacritics tokenize differently. This is the base model against llama.cpp, not the extension. The parity report counts them separately.
- vLLM needs compute capability 7.0, so a Kaggle P100 cannot run it and a T4 can.
- The saved model lacked `tokenizer_config.json`, so vLLM chat answered HTTP 400. Fixed on 2026-11-18.

**Where each piece ran.**

| Step | Where |
| --- | --- |
| Tokenizer learn, sweep, 0.6B extend and export, Ollama test | the 8 GB laptop |
| 1.7B quality run (`02_extend_quality_qwen3.ipynb`) | Kaggle T4, about 2 hours |
| 1.7B extend and export (`03_extend_export_qwen3.ipynb`) | Kaggle CPU, about 1 hour |
| vLLM check (`04_vllm_check_qwen3.ipynb`) | Kaggle GPU T4 |

## The commands at a glance

| Command | What it does | Needs model weights? | Needs a GPU? |
| --- | --- | --- | --- |
| `zubaan data urdu` | Builds the Urdu training text (Wikipedia, cleaned, FLORES removed, held-out split) | No | No |
| `zubaan extend learn` | Learns Urdu BPE merges and writes the extended tokenizer | No (tokenizer files) | No |
| `zubaan extend sweep` | Token cost for each vocabulary size, English as control | No | No |
| `zubaan extend quality` | Bits per character and Belebele for each size and init method | **Yes** | Kaggle for 1.7B |
| `zubaan extend run` | Learn (if needed), extend the weights, save and verify the model folder | **Yes** | No (about 40 s for 0.6B) |
| `zubaan export gguf` | Convert to GGUF, prove tokenizer parity, quantize with an Urdu imatrix | The saved model | No |
| `zubaan export ollama` | Write an Ollama `Modelfile` | No | No |
| `zubaan export vllm` | Write a vLLM config and launch script (`dtype half`) | No | No |
| `zubaan export all` | The three export steps in order | The saved model | No |
| `zubaan report extend` | Markdown report with charts from the saved results | No | No |

Every command takes `--config path.yaml`. Examples: `configs/export-qwen3-0.6b.yaml` and `configs/export-qwen3-1.7b.yaml` (both sections), `configs/extend-qwen3.yaml` (merges and sweep).

## Setup

```bash
uv sync --group dev --extra audit --extra report --extra export
```

The `export` extra adds `requests`, `tqdm`, `safetensors` and `sentencepiece`, which llama.cpp's converter imports. (For Qwen it imports `sentencepiece` before it notices there is no `tokenizer.model`, so without the package it stops with `No module named 'sentencepiece'`.)

### llama.cpp (only for `export gguf`)

zubaan pins llama.cpp release **`b12477`** (`LLAMA_CPP_TAG` in `export/gguf.py`). You need the source tree for `convert_hf_to_gguf.py` and the built tools `llama-quantize`, `llama-imatrix` and `llama-tokenize`. The default location is `tools/llama.cpp-b12477/` (gitignored); override it with `export.llama_cpp_dir` and `export.llama_cpp_bin`.

```bash
mkdir -p tools/llama.cpp-b12477/bin && cd tools/llama.cpp-b12477
# source tree (converter script and gguf-py)
curl -L https://github.com/ggml-org/llama.cpp/archive/refs/tags/b12477.tar.gz | tar xz --strip-components=1
# prebuilt tools; pick the build for your machine from the release page
curl -L -o bin.zip https://github.com/ggml-org/llama.cpp/releases/download/b12477/llama-b12477-bin-win-cpu-x64.zip
unzip bin.zip -d bin
```

On Linux use the `llama-b12477-bin-ubuntu-x64.tar.gz` asset, or build from the source tree with CMake. Using a newer release works only if the parser in `export/gguf.py` still finds `get_vocab_base_pre` and `chktxt`; the tests in `tests/test_gguf.py` check that against the checkout when it is present.

## 1. Build the extended model

```bash
uv run zubaan data flores
uv run zubaan data urdu --config configs/extend-qwen3.yaml
uv run zubaan extend learn --config configs/extend-qwen3.yaml     # 32,000 merges, a few minutes
uv run zubaan extend run --config configs/export-qwen3-0.6b.yaml
```

`extend run` reuses `extend.merges_file` if it exists (`--relearn` forces a new learn), extends the weights, writes the model and tokenizer to `<output_dir>/model`, and verifies the saved folder before it reports success:

- only `model.vocab` and `model.merges` differ from the base `tokenizer.json`;
- English text tokenizes exactly as with the base tokenizer;
- Urdu round-trips, `decode(encode(x)) == x`;
- `config.json` and the saved weights have one embedding row per token id.

Options: `--n-new`, `--method mean|focus|random`, `--json`. The model is saved in `bfloat16` by default (`extend.model_dtype`, Qwen3's own dtype). Embedding math runs in float32 first.

**Memory.** Qwen3-0.6B ran in about 40 seconds on the 8 GB laptop. Qwen3-1.7B needs about 8 GB free; use a machine with more RAM, or Kaggle with `notebooks/03_extend_export_qwen3.ipynb` (see the notebook table below).

The folder also holds `zubaan_extension.json` (base model, size, init method, `trained: false`).

## 2. Export

```bash
uv run zubaan export all --config configs/export-qwen3-0.6b.yaml
```

`export gguf` runs these steps, in this order, and stops at the first failure:

1. **Fingerprint.** The converter chooses `tokenizer.ggml.pre` by hashing the ids of a fixed test string. zubaan computes that hash for the base and the extended tokenizer, resolves the base's type from the converter source (`qwen2` for Qwen3), and checks that the pre-tokenizer is unchanged. If the converter does not know the extended hash, zubaan adds it to the converter (a `.orig` backup is kept). On the pinned release the test string contains no Arabic-script characters, so Urdu-only merges leave the hash unchanged and no patch is needed. The summary records which case happened.
2. **Convert** to an f16 GGUF.
3. **Parity.** Tokenize FLORES Urdu, FLORES English and a stress set (emoji, numbers, mixed script, an empty string, `\r\n`, special tokens in text, very long lines) with Hugging Face and with `llama-tokenize`, one text per call, and compare every id. A text whose ids differ is excused only if it is not in NFC form and llama.cpp gives Hugging Face's exact ids once the text is NFC-normalized (see below). **If any other id differs, the export stops here** and the mismatches are saved to `export_summary.json`.
4. **Importance matrix** from Urdu text (`export.calibration`, `calibration_chars` characters), never generic English.
5. **Quantize** each type in `export.quants` (default `Q4_K_M`, `Q8_0`) with that matrix.

Files that already exist are skipped unless `export.force: true`.

Outputs in `<export.output_dir>`:

| File | What it is |
| --- | --- |
| `<name>-F16.gguf`, `<name>-Q4_K_M.gguf`, `<name>-Q8_0.gguf` | The model |
| `<name>.imatrix.dat`, `calibration_urdu.txt` | Importance matrix and its Urdu text |
| `Modelfile` | Ollama: `ollama create <name> -f Modelfile` |
| `vllm.yaml`, `serve_vllm.sh` | vLLM: `dtype: half` (T4 and P100 have no bfloat16) |
| `export_summary.json` | Tool versions, fingerprint, parity result, file names |

Run the model: `llama-completion -m <name>-Q4_K_M.gguf -p "..." -n 40` (the non-interactive tool in recent llama.cpp releases; `llama-cli` opens a chat), or `ollama create` then `ollama run`. On the 0.6B, `llama-completion` generated at 34 tokens/s on 4 CPU threads; the text is nonsense because the new rows are untrained, which is expected.

**Qwen3 thinking.** The Modelfile uses Qwen's ChatML template and stops on `<|im_end|>` and `<|im_start|>`, both checked to exist in the extended vocabulary at their original ids. It does not suppress the `<think>` block that Qwen3 can start a reply with.

**NFC and llama.cpp.** Qwen's Hugging Face tokenizer normalizes text to Unicode NFC first; llama.cpp's BPE tokenizer does not. For Urdu this shows up with stacked diacritics: FLORES has 4 lines where shadda comes before damma, NFC reorders them, and llama.cpp keeps the original order, so two ids swap. The parity summary counts these as `n_nfc_only`, separately from `n_real_mismatch`. It is a property of the base model and llama.cpp, not of the extension. In practice, send NFC text to GGUF models: `zubaan.text.normalize_urdu` does this.

## 3. Measure quality next to cost

```bash
uv run zubaan extend sweep   --config configs/extend-qwen3.yaml        # tokenizer only
uv run zubaan extend quality --config configs/extend-qwen3.yaml        # needs a GPU for 1.7B
uv run zubaan report extend
```

The 1.7B quality run is `notebooks/02_extend_quality_qwen3.ipynb` on Kaggle (T4, about 2 hours, float32). `report extend` reads `reports/extend_sweep.json`, `reports/extend_quality_qwen3-1.7b.json` and the export summary, and writes `docs/reports/tokenizer-extension/`. Every number in it comes from those files.

## How to read the results

- **Cost multiplier** is Urdu tokens divided by English tokens for the same sentences. 1.0x would be parity.
- **Bits per character** is the quality metric. Never compare per-token perplexity across tokenizers.
- **Belebele** is scored with `acc_char` (answer log-probability divided by answer characters), which is fair across tokenizers. Chance is 25%.
- **Parity passed** means llama.cpp tokenizes exactly like Hugging Face on the test set. It is evidence, not a proof for every possible string; extend the stress set if you add scripts.

## Kaggle notebooks

| Notebook | Session | Does | Output to keep |
| --- | --- | --- | --- |
| `02_extend_quality_qwen3.ipynb` | GPU T4 | Quality sweep for the 1.7B (float32) | `extend_quality_qwen3-1.7b.json`, `extend_sweep.json` |
| `03_extend_export_qwen3.ipynb` | CPU | Every step of this guide for the 1.7B | `export/export_summary.json`, GGUF files, `extended/model/` |
| `04_vllm_check_qwen3.ipynb` | GPU T4 (not P100) | Starts vLLM on the 03 model, checks tokenization, completion and chat | `vllm_check.json` |

Each installs zubaan at a pinned commit that must be on GitHub (`python notebooks/make_0N_notebook.py [COMMIT]` regenerates it). 04 takes the 03 output through Add Input > Notebooks and repairs a model folder that lacks the chat template.

## Limitations

- New rows are untrained (Stage 3).
- One corpus (Urdu Wikipedia) and one seed.
- The Ollama Modelfile was accepted by `ollama create` (0.40.16) and answered a chat request on the 0.6B. Qwen3 starts replies with a `<think>` block, which the Modelfile does not suppress.
- vLLM 0.34.0 served the extended Qwen3-1.7B on a Kaggle T4 (`dtype half`, `notebooks/04_vllm_check_qwen3.ipynb`): vLLM's tokenization of 5 Urdu and English sentences equals the extended `tokenizer.json`, and completion and chat both answered (40 tokens in 0.9 s including the first request). The text is meaningless because the rows are untrained. A P100 cannot run vLLM (compute capability 6.0).
