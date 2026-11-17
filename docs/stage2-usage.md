# Stage 2 usage: extend a model and export it

Stage 1 measured the problem. Stage 2 fixes the token cost: it adds Urdu tokens to a model's vocabulary, then packages the result as GGUF (llama.cpp, Ollama) and a vLLM config.

**Read this first.** After Stage 2 the model has **not learned** its new tokens. The new embedding rows are only initialized, so Urdu quality is worse than the base model until Stage 3 trains them (see the results in [the extension report](reports/tokenizer-extension/index.md)). Stage 2 lowers the token cost; it is not a model to hand to users yet.

Currently Urdu only, on the Qwen3 family. The code is language-agnostic except for the Urdu-specific normalization and the Arabic-script merge filter.

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

**Memory.** Qwen3-0.6B ran in about 40 seconds on the 8 GB laptop. Qwen3-1.7B needs about 8 GB free; use a machine with more RAM, or Kaggle with `notebooks/03_extend_export_qwen3.ipynb`, which runs every step of this guide for the 1.7B on a CPU session (about 1 hour; push the pinned commit to GitHub first).

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

## Limitations

- New rows are untrained (Stage 3).
- One corpus (Urdu Wikipedia) and one seed.
- The Ollama Modelfile was accepted by `ollama create` (0.40.16) and answered a chat request on the 0.6B. Qwen3 starts replies with a `<think>` block, which the Modelfile does not suppress.
- vLLM has not been started on a GPU yet. `notebooks/04_vllm_check_qwen3.ipynb` does it on a Kaggle T4 (not a P100: vLLM needs compute capability 7.0 or higher) from the 03 output, and also checks that vLLM tokenizes Urdu exactly like the extended `tokenizer.json`.
