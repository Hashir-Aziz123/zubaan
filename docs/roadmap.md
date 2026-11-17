# Roadmap

Each stage ends with something usable and public. Estimates assume part-time work.

## Stage 1: audit (about 2 weeks)

- [x] Urdu normalization and word splitting
- [x] Fertility: tokens per word, chars per token, cost multiplier on parallel corpora
- [x] Acceptance-rate and speedup math (dependency-free)
- [x] Data script: `zubaan data flores` downloads FLORES-200 devtest `urd_Arab` and `eng_Latn` into `data/`
- [x] First real audit: Qwen3-1.7B fertility on FLORES, with sanity checks and provenance (Urdu 3.12 vs English 1.24 tokens per word, 3.14x cost)
- [x] Multi-model audit: `zubaan audit compare` across popular tokenizers, CSV and JSON table (Qwen3 3.14x is among the worst; BLOOM 1.35x is the best measured)
- [x] `audit acceptance`: draft/target acceptance on Urdu and English. Qwen3-0.6B vs 1.7B run on Kaggle: alpha 0.573 (Urdu) vs 0.632 (English)
- [x] Report: "The Urdu language tax" (Markdown plus charts) in `docs/reports/urdu-language-tax/`. Tokenizer and acceptance sections done

## Stage 2: extend and export (2 to 3 weeks)

- [x] Urdu training corpus: Urdu Wikipedia, cleaned, FLORES-overlap removed, held-out split (`zubaan data urdu`)
- [x] Learn Urdu merges on top of the base tokenization; select merges that cannot touch English (`zubaan extend learn`)
- [x] Merge into base tokenizer as real merges; English-unchanged and Urdu round-trip tests pass on Qwen3 (special token ids pinned)
- [x] Token cost sweep (`zubaan extend sweep`), tokenizer only, no quality yet. Qwen3-1.7B Urdu cost multiplier 3.14x -> 1.40x (5k new tokens), 1.29x (10k), 1.22x (20k), 1.20x (32k), English unchanged
- [x] Resize embeddings and LM head; mean-of-pieces, random and FOCUS-style initialization (own FOCUS implementation: `langsfer` conflicts with the pinned transformers and numpy). Validated on Qwen3-0.6B (CPU smoke run, 25 texts per set)
- [x] Quality next to cost: bits per character (held-out Urdu, FLORES Urdu, FLORES English) and Belebele `urd_Arab`/`eng_Latn` (`zubaan extend quality`)
- [x] Full quality run on Qwen3-1.7B (Kaggle T4, `notebooks/02_extend_quality_qwen3.ipynb`): bits per character on held-out Urdu 2.31 (base) -> 3.57 (10k, mean of pieces); Belebele Urdu back to chance. English unchanged. Untrained rows cost quality, as expected; stage 3 trains them
- [x] Save the extended model and verify the folder (`zubaan extend run`); built and verified on Qwen3-0.6B locally (10,000 new tokens, 161,669 rows, English identical and Urdu round-trip on 1,012 FLORES texts)
- [x] GGUF export code: pinned llama.cpp `b12477`, pre-tokenizer fingerprint handling, HF vs llama.cpp parity test, Urdu importance matrix, Q4_K_M and Q8_0 (`zubaan export gguf`). Unit-tested with a fake llama.cpp
- [x] Ollama Modelfile and vLLM config generation (`zubaan export ollama`, `export vllm`); snapshot tests, no `bfloat16`, stop tokens checked against the vocabulary
- [x] One command each way: `zubaan extend run` and `zubaan export all`, config driven, `--json`
- [x] Docs: `docs/stage2-usage.md`, report `docs/reports/tokenizer-extension/`, decisions
- [x] **Real GGUF run on Qwen3-0.6B with 10,000 new tokens** (llama.cpp `b12477`): converter recognized the tokenizer as `qwen2` with no patch; HF vs llama.cpp parity on 2,045 texts: 2,041 identical, 4 differ only because the text is not NFC (llama.cpp has no NFC normalizer), 0 real mismatches. F16 1.2 GB, Q8_0 651 MB, Q4_K_M 405 MB (Urdu imatrix)
- [x] Q4_K_M loads and generates on CPU with `llama-completion` (34 tokens/s, 4 threads, 0.6B). The Urdu prompt "the capital of Pakistan" is 3 tokens (13 with the base tokenizer); the text itself is nonsense and drifts into Bengali, as expected for untrained rows
- [x] Kaggle notebook for the 1.7B: `notebooks/03_extend_export_qwen3.ipynb` (CPU session, extend, verify, GGUF with parity, Q4_K_M and Q8_0, Modelfile, vLLM config). Needs the pinned commit pushed first; run on Kaggle: 1.7B parity passed
- [x] `ollama create` accepts the generated Modelfile and chat works through its ChatML template (Ollama 0.40.16, Q4_K_M of the 0.6B; both stop tokens registered). The reply is a `<think>` block in English that misreads the Urdu question, as expected for untrained rows
- [x] Kaggle notebook for the vLLM check: `notebooks/04_vllm_check_qwen3.ipynb` (GPU T4, takes the 03 output as an input; starts vLLM with the generated config, compares vLLM tokenization of Urdu with the extended `tokenizer.json`, requests a completion). Run on a Kaggle T4: vLLM 0.34.0 passed (tokenization identical, completion and chat work)
- [x] 04 notebook run on a Kaggle T4: vLLM serves the extended 1.7B (see above)
- [x] 03 notebook run on Kaggle: Qwen3-1.7B extended and exported, parity passed (2,041 of 2,045 identical, 4 NFC-only, 0 real mismatches), no converter patch, Q4_K_M and Q8_0 built
- [x] Chat template fix (2026-11-18): saved models now carry `tokenizer_config.json`

## Stage 3: adapt (2 to 3 weeks)

- [ ] Continued pretraining recipe: LoRA plus trainable embeddings and LM head, English replay
- [ ] Kaggle notebook with Hub checkpointing and resume
- [ ] Quality: bits per character on held-out Urdu, Belebele `urd_Arab`

## Stage 4: draft (2 to 3 weeks)

- [ ] Extend Qwen3-0.6B with the same tokenizer; distill on adapted target outputs
- [ ] n-gram drafter experiment
- [ ] Acceptance before and after, Urdu versus English

## Stage 5: bench and release (2 weeks)

- [ ] Benchmark runner: TTFT, tokens/s, words/s, server-reported acceptance
- [ ] llama.cpp on CPU (laptop) and vLLM on Kaggle
- [ ] Final report, docs site, PyPI release, blog post
