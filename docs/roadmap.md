# Roadmap

Each stage ends with something usable and public. Estimates assume part-time work.

## Stage 1: audit (about 2 weeks)

- [x] Urdu normalization and word splitting
- [x] Fertility: tokens per word, chars per token, cost multiplier on parallel corpora
- [x] Acceptance-rate and speedup math (dependency-free)
- [x] Data script: `zubaan data flores` downloads FLORES-200 devtest `urd_Arab` and `eng_Latn` into `data/`
- [x] First real audit: Qwen3-1.7B fertility on FLORES, with sanity checks and provenance (Urdu 3.12 vs English 1.24 tokens per word, 3.14x cost)
- [x] Multi-model audit: `zubaan audit compare` across popular tokenizers, CSV and JSON table (Qwen3 3.14x is among the worst; BLOOM 1.35x is the best measured)
- [ ] `audit acceptance`: draft/target acceptance on Urdu and English. Code, tests and Kaggle notebook done; the real Qwen3 run is pending (local run stopped for lack of RAM)
- [ ] Report: "The Urdu language tax" (Markdown plus charts) in `docs/reports/urdu-language-tax/`. Tokenizer sections done; acceptance section waits for the run above

## Stage 2: extend and export (2 to 3 weeks)

- [ ] Train Urdu BPE with the base pre-tokenizer; select new merges
- [ ] Merge into base tokenizer as real merges; English-unchanged and Urdu round-trip tests
- [ ] Resize embeddings and LM head; mean-of-pieces initialization (compare with langsfer FOCUS)
- [ ] GGUF export with pre-tokenizer fingerprint handling and HF vs llama.cpp parity test
- [ ] Ollama Modelfile and vLLM config generation

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
