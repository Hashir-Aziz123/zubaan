# zubaan

**Urdu users pay more and wait longer for AI than English users. zubaan measures that gap and produces a faster, cheaper model you can actually deploy.**

> Status: pre-alpha. Stage 1 (`audit fertility`) works; later stages are planned. See [docs/roadmap.md](docs/roadmap.md).

## The problem

LLM tokenizers were built mostly from English. An Urdu word is often split into several tokens where an English word needs one, so the same message in Urdu:

- costs more (you pay per token),
- generates more slowly (one forward pass per token),
- fills the context window faster.

Speculative decoding, the standard way to speed up generation, also works worse for low-resource languages: small draft models predict them poorly, so fewer drafted tokens are accepted.

Research has shown fixes for both: extend the tokenizer with target-language tokens and adapt the model, and train a drafter for the language. What is missing is a practical path from those papers to a model a developer can deploy in Ollama, llama.cpp, or vLLM. zubaan is that path.

## Pipeline

| Command | What it does | Status |
| --- | --- | --- |
| `zubaan audit fertility` | Tokens per word and cost multiplier versus a parallel English corpus | Working |
| `zubaan audit acceptance` | Speculative decoding acceptance rate for a draft/target pair | Planned |
| `zubaan extend` | Adds target-language merges to a tokenizer and initializes new embeddings | Planned |
| `zubaan adapt` | Short continued pretraining recipe sized for a free Kaggle GPU | Planned |
| `zubaan draft` | Drafters that share the extended vocabulary | Planned |
| `zubaan export` | GGUF (with tokenizer parity checks), vLLM and Ollama configs | Planned |
| `zubaan bench` | End-to-end benchmarks with tokenizer-fair metrics | Planned |

## Fair metrics

Tokens per second cannot be compared across tokenizers, because a better tokenizer packs more text into each token. zubaan reports words per second instead:

**Words per second = tokens per second ÷ tokens per word**

For example, 40 tokens/s at 4.0 tokens per word is 40 ÷ 4.0 = 10 words/s, while 38 tokens/s at 1.8 tokens per word is 38 ÷ 1.8 ≈ 21 words/s. The second model is slower per token but twice as fast for the reader.

Quality is compared with bits per character for the same reason.

## Install

```bash
pip install "zubaan[audit]"        # once published; for now, install from source:
git clone <repo-url> && cd zubaan
uv sync --group dev --extra audit
```

## Quick start

```bash
zubaan audit fertility \
  --tokenizer Qwen/Qwen3-1.7B \
  --corpus data/flores_urd.txt \
  --reference data/flores_eng.txt
```

Use a parallel corpus such as FLORES-200 (`urd_Arab` and `eng_Latn`) so the cost multiplier compares the same content.

## Development

```bash
uv sync --group dev
uv run pytest
uv run ruff check .
```

See [CLAUDE.md](CLAUDE.md) for architecture rules and conventions.

## License

Apache License 2.0.
