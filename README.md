# zubaan

**Urdu users pay more and wait longer for AI than English users. zubaan measures that gap and produces a faster, cheaper model you can actually deploy.**

> Status: pre-alpha. Stages 1 (audit), 2 (extend and export) and 3 (adapt) are done; draft and bench are planned. The adapted Qwen3-1.7B reads Urdu with 2.4x fewer tokens than the base and scores better on Urdu bits per character, but it is a research artifact, not a chat assistant (see [docs/stage3-usage.md](docs/stage3-usage.md)). See [docs/roadmap.md](docs/roadmap.md).

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
| `zubaan audit compare` | Fertility for several tokenizers on the same data | Working |
| `zubaan audit acceptance` | Speculative decoding acceptance rate for a draft/target pair | Working |
| `zubaan extend` | Adds target-language merges to a tokenizer, initializes new embeddings, saves and verifies the model | Working (new rows untrained) |
| `zubaan adapt` | Short continued pretraining recipe sized for a free Kaggle GPU | Done for Qwen3-1.7B: Urdu quality above the base, English not damaged |
| `zubaan draft` | Drafters that share the extended vocabulary | Planned |
| `zubaan export` | GGUF (with tokenizer parity checks), vLLM and Ollama configs | Done for Qwen3-0.6B and 1.7B; parity passed, vLLM and Ollama checked |
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
# add --extra report for charts and --extra export for GGUF export
```

## Quick start

```bash
zubaan audit fertility \
  --tokenizer Qwen/Qwen3-1.7B \
  --corpus data/flores_urd.txt \
  --reference data/flores_eng.txt
```

Extend a model and export it (see [docs/stage2-usage.md](docs/stage2-usage.md)):

```bash
uv run zubaan extend run --config configs/export-qwen3-0.6b.yaml
uv run zubaan export all --config configs/export-qwen3-0.6b.yaml   # needs llama.cpp, see the guide
```

On Qwen3-1.7B, 10,000 new Urdu tokens cut the Urdu token cost from 3.14x to 1.29x the English cost
of the same text. Stage 3 then trained the new tokens on a free Kaggle T4: on FLORES Urdu the
adapted model scores 1.72 bits per character against the base's 2.34, while English was not
damaged. See the [extension report](docs/reports/tokenizer-extension/index.md) and
[docs/stage3-usage.md](docs/stage3-usage.md), including its limitations.

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
