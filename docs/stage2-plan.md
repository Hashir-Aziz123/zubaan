# Stage 2 plan: extend and export

Draft, 2026-11-16, updated 2026-11-17. All chunks (0 to 12) are built and the Kaggle runs are done: the 1.7B export passed parity and vLLM served the model (see "Status" at the end). Based on `docs/roadmap.md` and `CLAUDE.md`.

**Goal.** Take a base model (Qwen3-1.7B) and an Urdu text file, and produce an extended model whose tokenizer costs far fewer tokens on Urdu, plus the GGUF, Ollama and vLLM files to run it.

**Limit to state everywhere.** After Stage 2 the new embeddings are only initialized, not trained. Urdu quality may drop until Stage 3. A cost saving is never reported without quality next to it.

**Success checks (from `continue.md`).**
1. English tokenization is byte-for-byte identical to the base.
2. Urdu round-trips: `decode(encode(x)) == x`.
3. `audit fertility` on the extended tokenizer shows a large drop from 3.14x (BLOOM at 1.35x is the yardstick, a goal and not a result).
4. GGUF token ids from llama.cpp equal the Hugging Face ids.

## Facts about the Qwen3 tokenizer that shape the design

These are things to verify in chunk 0, not assume.

- **New ids start after the specials.** The vocabulary size 151,669 includes special tokens (ids from about 151,643). New ids therefore start at 151,669. The embedding matrix is padded (151,936 rows), so rows already exist past 151,669. Resize to `151669 + N`, do not append to 151,936.
- **Appended merges have the lowest priority.** A new merge only fires on pairs left over after every old merge. It still fires on English text if the pair of old tokens it joins can occur in English. The English-unchanged test is therefore not automatic: candidate merges must be restricted to pairs that cannot occur in English output (for example both sides contain Urdu-range UTF-8 bytes). Chunk 3 enforces this.
- **Training text vs. user text.** Stage 1 normalizes Urdu before counting. Users will not normalize their input. Decide whether to train merges on normalized text (cleaner, matches the audit) and check round-trip on raw text too.

## Chunks

Each chunk has one deliverable, a test that can fail, and a done criterion. Order matters; later chunks list their dependencies. "Laptop" means CPU, 8 GB RAM, no model weights.

### Chunk 0: Inspect the base tokenizer (laptop, tokenizer files only)

- Write `extend/inspect.py` (or a throwaway script) that loads Qwen3's `tokenizer.json` and reports: vocab size, number of merges, special token id range, pre-tokenizer type and regex, whether `added_tokens` hold anything besides specials, and the embedding matrix row count from `config.json`.
- **Test:** a unit test on a tiny fixture `tokenizer.json` that the inspector reads the fields correctly. One `slow` test on the real Qwen3 tokenizer asserting the numbers above.
- **Done when:** the facts in the section above are confirmed or corrected in `docs/decisions.md`.

### Chunk 1: Urdu training corpus (laptop)

- Decision needed: which corpus (see Decisions below). Check licence first.
- `data/` script (`zubaan data urdu`, config driven) that downloads, normalizes with `text.normalize_urdu`, deduplicates, and writes `data/urdu_train.txt` plus a metadata JSON (source, licence, size, line count, sha256).
- Hold out FLORES: assert no FLORES devtest sentence appears in the training file.
- **Tests (default, offline):** a fixture of ten lines checks normalization, dedup and the FLORES overlap guard (an injected FLORES line must be rejected). A `slow` test downloads a small slice.
- **Done when:** the file exists, the metadata records the licence, and the overlap guard has been run on the real file with zero hits.

### Chunk 2: Train Urdu BPE with the base pre-tokenizer (laptop)

- `extend/train_bpe.py`: train a BPE with the `tokenizers` library, using the base tokenizer's own pre-tokenizer object (loaded from its `tokenizer.json`, never rebuilt by hand), starting from the byte-level alphabet. Output: an ordered list of Urdu merges, each as a pair of byte-level token strings, with frequency.
- `ExtendConfig` in `config.py` (it exists with a 10,000 default); add `--config` support.
- **Tests:** train on the tiny fixture with a fake base and assert (a) the pre-tokenizer in the trained model equals the base's serialized pre-tokenizer, (b) merges are deterministic for a fixed seed and input, (c) a hand-made 4-sentence corpus yields the merge order worked out by hand in the test.
- **Done when:** a merge list is produced for the real corpus in minutes on the laptop.

### Chunk 3: Select merges that are safe for English (laptop, pure Python)

- `extend/select.py`: from the trained merges keep the top N by frequency, drop merges already present in the base, and drop any merge that could change English tokenization. Rule to start with: both parts must contain at least one byte in the UTF-8 lead range used by Arabic-script Urdu, and no part may be a pure ASCII string. Keep merge dependencies in order (a merge's parts must exist earlier in the list or in the base).
- Dependency-free where possible, per the architecture rules.
- **Tests:** property-style tests on small inputs: no selected merge has an ASCII-only side; every part exists earlier or in the base; N is respected; selection is deterministic.
- **Done when:** selecting for N = 10,000 on the real merges returns exactly N or reports honestly how many safe merges exist.

### Chunk 4: Merge into the base tokenizer as real merges (laptop)

- `extend/merge.py`: write a new `tokenizer.json` by appending the selected merges to `model.merges` and the new tokens to `model.vocab` with ids starting at the current vocab size. Leave `pre_tokenizer`, `normalizer`, `decoder`, `post_processor` and `added_tokens` untouched. Never use `add_tokens`.
- **Tests (the two required ones, plus a guard):**
  1. **English identical.** On an English fixture and on FLORES English, `encode(x).ids` from the extended tokenizer equals the base's for every sentence. A mutation test: temporarily add one English-capable merge and assert this test fails, so the test is proven able to fail.
  2. **Urdu round-trips.** `decode(encode(x)) == x` for the Urdu fixture, FLORES Urdu, and raw un-normalized Urdu with mixed scripts, digits and emoji.
  3. **Specials unchanged.** Ids of all special tokens are identical to the base.
- **Done when:** the tests pass on the real Qwen3 tokenizer (`slow`) and the saved folder reloads with `AutoTokenizer.from_pretrained`.

### Chunk 5: Measure the token cost drop (laptop)

- Run the existing `audit fertility` and `audit compare` on the extended tokenizer, for several N (for example 2,000, 5,000, 10,000, 20,000), using `configs/*.yaml`.
- Output a small table: N versus Urdu tokens per word versus cost multiplier, with English unchanged shown as a control.
- **Tests:** a default test on the fixture checking the extended tokenizer never produces more Urdu tokens than the base on held-out lines (extra merges can only reduce or keep the count). The real numbers are a `slow` check recorded in the report, not asserted against a target.
- **Done when:** the table exists and the vocabulary size choice is made and written to `docs/decisions.md`.

### Chunk 6: Resize embeddings and LM head, initialize new rows (Kaggle, or laptop with a memory check)

- `extend/init_emb.py` with the initialization as a pure function over arrays: for each new token, mean of the old-token embeddings of the pieces it used to split into (found by encoding the new token's string with the **base** tokenizer). Handle tied embeddings: Qwen3 small models tie input embeddings and the LM head; check `tie_word_embeddings` and resize both consistently.
- Wrapper that loads the model, resizes to `old_vocab + N`, writes the new rows, saves.
- **Tests (default, tiny):** use a randomly initialized tiny model config (a few layers, hidden size 8, no download) and a fake tokenizer pair. Assert: (a) old rows unchanged bit for bit, (b) each new row equals the hand-computed mean of its pieces, (c) shapes of embedding and head equal the new vocab size, (d) with tying, the head and embedding stay the same tensor. A `gpu`/`slow` test does this on the real Qwen3-0.6B.
- **Sanity check, real model:** on English text the extended model's logits restricted to old tokens match the base's within tolerance (new rows must not disturb English; this works because the softmax over the larger vocab only changes mass on new tokens, so also report the English bits per character before and after).
- **Done when:** the extended Qwen3-1.7B saves and loads, and English bits per character is within a small stated tolerance of the base.

### Chunk 7: Compare initialization methods (Kaggle, after chunk 6)

- Add FOCUS from `langsfer` as a second init method behind the same interface, plus random init as a control.
- Compare on held-out Urdu with **bits per character** (not per token perplexity, the tokenizers differ) for: base model, mean-of-pieces, FOCUS, random.
- **Tests:** the interface test from chunk 6 runs against every method (shape, old rows untouched). Quality numbers are a recorded result, not a pass/fail assertion.
- **Done when:** a table of init method versus Urdu bits per character is in the docs, with the honest statement that none is trained yet. If `langsfer` is heavy or conflicts with the pinned `transformers`, say so and keep it as an optional extra.

### Chunk 8: Quality baseline next to the cost saving (Kaggle)

- Bits per character on held-out Urdu (use `metrics.bits_per_character`) and Belebele `urd_Arab` accuracy for the base and extended models. Belebele is a multiple-choice reading task, so it needs a small evaluation harness; keep it in `bench/quality.py` so Stage 5 reuses it.
- **Tests:** the metric functions already follow worked examples; add a Belebele scoring test on three hand-made questions with a fake model that returns known logits.
- **Done when:** the cost multiplier and the quality numbers sit in one table. Expect quality to be worse or equal; write that down.

### Chunk 9: GGUF export with parity check (laptop, needs llama.cpp)

- Vendor or document how to fetch `convert_hf_to_gguf.py` at a pinned llama.cpp commit; record the commit.
- `export/gguf.py`: run the converter on the extended model. Expect `BPE pre-tokenizer was not recognized`. Fix by mapping the extended tokenizer's fingerprint (the hash the converter computes over a test string) to the base model's pre-tokenizer type (`qwen2` for Qwen3, verify). Only valid because the pre-tokenizer is unchanged, so also assert that in code by comparing the serialized pre-tokenizers.
- **Parity test:** tokenize a test set (FLORES Urdu and English, plus a stress set with emoji, numbers, mixed script, empty string, very long line) with Hugging Face and with llama.cpp (`llama-tokenize` or `llama-cpp-python` vocab-only load) and assert identical ids for every line.
- Quantize (Q4_K_M, Q8_0) using an imatrix calibrated on **Urdu** text, per the project rule.
- **Tests:** default tests cover the fingerprint mapping logic and the parity comparer with fake ids. The real parity test is `slow` and skipped if llama.cpp is not installed.
- **Done when:** parity passes on the whole test set, and a Q4_K_M file loads and generates Urdu text locally on CPU.

### Chunk 10: Ollama Modelfile and vLLM config (laptop)

- `export/ollama.py`: generate a `Modelfile` (FROM path, template matching Qwen3's chat template, stop tokens, parameters).
- `export/vllm.py`: generate a config / launch command with `dtype=half` (T4 and P100 have no BF16), the extended tokenizer path, and a sensible `max_model_len`.
- **Tests:** snapshot tests on the generated text; a test that the vLLM config never contains `bfloat16`; a test that the Modelfile's stop tokens exist in the extended vocabulary.
- **Done when:** `ollama create` accepts the Modelfile (manual check) and the vLLM command line is valid (`--help` style dry run on Kaggle).

### Chunk 11: One command, end to end (laptop plus Kaggle)

- Wire `zubaan extend` (and `zubaan export`) in `cli.py` as thin wrappers over chunks 2 to 10, config driven, with `--json` output. Replace the "not implemented" placeholders.
- **Tests:** a CLI test with the fake tokenizer and tiny model that runs the whole pipeline in seconds and checks the output folder contents and the JSON summary.
- **Done when:** one command on the tiny fixture works in CI-style default tests, and one real run (base model plus Urdu file) produces the extended model, the GGUF, the Modelfile and the vLLM config.

### Chunk 12: Write-up (laptop)

- Update `docs/roadmap.md` (tick Stage 2 items), the stage table in `CLAUDE.md` if scope changed, add a `docs/stage2-usage.md` in the style of `stage1-usage.md`, and extend the report with a "Tokenizer extension" section: cost multiplier versus N, init comparison, quality next to cost, limitations.
- Record every design decision in `docs/decisions.md` with date and reason.

## Dependencies and order

```
0 -> 2 -> 3 -> 4 -> 5 -> 6 -> 7
1 -> 2                  4 -> 9 -> 10
              6 -> 8 -> 9
all -> 11 -> 12
```

Chunks 0 to 5 need no model weights and can all be finished on the laptop, and they already deliver the headline cost number. Chunks 6 to 8 need the weights and a GPU for the quality runs. Start the Qwen3-1.7B download early (the earlier download took 51 minutes).

## Decisions needed from you before coding

1. **Training corpus.** Options: Urdu Wikipedia (CC BY-SA, clean but small), CC-100 `ur` (large, check terms), OSCAR (gated, check terms). Recommendation: start with Urdu Wikipedia for a clean licence and a fast loop, and test whether about 10 MB is enough, as Yamaguchi et al. suggest. Add a larger source only if the cost curve is still dropping.
2. **Vocabulary size N.** Recommendation: sweep 2,000, 5,000, 10,000 and 20,000 in chunk 5, then choose using the cost-versus-N curve, remembering each new token is an embedding row to train in Stage 3.
3. **Where to resize (chunk 6).** Check free RAM first. Recommendation: Kaggle for the 1.7B, laptop for the 0.6B and the tiny test models.
4. **Normalize before training merges?** Recommendation: train on normalized text, then verify round-trip on raw text and report the token cost on both.

## Risks

- **English changes silently.** Mitigated by the Urdu-only merge filter in chunk 3 and the mutation-checked test in chunk 4.
- **Wrong id range.** Appending after the padded embedding size instead of the vocabulary size would leave gaps. Chunk 0 and chunk 6 tests cover it.
- **Tied embeddings.** Resizing only one of the two tensors breaks the model without an error. Chunk 6 test (d).
- **GGUF fingerprint.** The converter may also want the new tokens' types and scores; the parity test is the real proof, not the absence of warnings.
- **Quality drop mistaken for a bug.** It is expected before Stage 3; chunk 8 records it so nobody has to guess.
- **Time.** Chunks 0 to 5 are about a week part-time; 6 to 10 are the bulk. If time runs short, cut chunk 7 (FOCUS) first, since mean-of-pieces is the required method.

## Status (2026-11-16): chunks 0 to 5 built

Differences from the plan above, all recorded in `docs/decisions.md`:

- **Chunk 2** uses continued BPE (merges learned on top of the base tokenization) instead of the Hugging Face `BpeTrainer`, because merges learned from raw bytes can name pairs that never occur after base tokenization. Merge counts are pair occurrences, not frequencies from a trainer.
- **Chunk 3** is a proof, not a heuristic: a merge is allowed only if its token contains a UTF-8 lead byte 0xD8 to 0xDB, which occurs only for characters U+0600 to U+06FF. The learner applies the filter while learning, so no later merge depends on a dropped one.
- **Chunk 4** found that the `tokenizers` library moves special tokens that are only in `added_tokens` when the vocabulary grows (a naive append moved `<|im_end|>` from 151645 to 151745). The extended `tokenizer.json` lists them in `model.vocab` at their original ids. Chunk 6 should still resize the embedding matrix to `151669 + N`.
- **Chunk 5** vocabulary sizes are 5k, 10k, 20k and 32k (your choice). Results are in the roadmap. Stage 1's `audit fertility` run on the saved extended tokenizer gives the same 1.29x as the sweep for 10k, with English tokens identical to the base (27,621).
- The corpus is capped at 100 MB (295,905 lines); the full file had 785,348 training lines.

## Status (2026-11-17): chunks 6 to 12 built

- **Chunks 6 to 8** (`extend/init_emb.py`, `focus.py`, `quality_run.py`, `bench/quality.py`) ran on Kaggle for Qwen3-1.7B. Results and the default for stage 3 (10k new tokens, mean of pieces) are in `docs/decisions.md` and `docs/reports/tokenizer-extension/`. The Belebele harness lives in `bench/quality.py` as planned; the run orchestration is `extend/quality_run.py`.
- **A gap in the plan:** nothing saved the extended model. `extend/build.py` and `zubaan extend run` now do, and verify the saved folder (tokenizer diff, English unchanged, Urdu round-trip, embedding rows).
- **Chunk 9** (`export/gguf.py`, `export/parity.py`, `export/pipeline.py`) is built and unit-tested with a fake llama.cpp. The expected "pre-tokenizer not recognized" error does **not** occur on release `b12477`: the converter's test string has no Arabic-script characters, so the fingerprint equals the base's. The patch path exists for other cases. The real run on Qwen3-0.6B (10,000 new tokens) passed: parity 2,041 of 2,045 identical, the other 4 differ only because llama.cpp does not apply NFC (see `docs/decisions.md`, 2026-11-17), 0 real mismatches, and Q4_K_M and Q8_0 were produced with an Urdu imatrix. The Q4_K_M loads and generates on CPU (34 tokens/s; nonsense text, as expected for untrained rows).
- **Chunk 10** (`export/ollama.py`, `export/vllm.py`) is built and snapshot-tested. `ollama create` accepted the Modelfile and a chat request worked (Ollama 0.40.16, 0.6B Q4_K_M). A real vLLM start is still to do (`notebooks/04_vllm_check_qwen3.ipynb`, Kaggle T4).
- **Chunk 11** (`zubaan extend run`, `zubaan export gguf|ollama|vllm|all`) replaces the "not implemented" placeholder. The tiny-fixture end-to-end tests are `tests/test_extend_run.py` and `tests/test_export_pipeline.py`.
- **Chunk 12:** `docs/stage2-usage.md`, the extension report (`zubaan report extend`), roadmap and decisions are updated.

## Status (2026-11-18): stage 2 done

- `notebooks/03_extend_export_qwen3.ipynb` ran on Kaggle (CPU): Qwen3-1.7B extended with 10,000 tokens, parity 2,041 of 2,045 identical, 4 NFC-only, 0 real mismatches, `qwen2` pre-tokenizer, no converter patch, Q4_K_M and Q8_0 built.
- `notebooks/04_vllm_check_qwen3.ipynb` ran on a Kaggle T4: vLLM 0.34.0 served the extended model with `dtype half`; tokenization identical to `tokenizer.json`; completion and chat answered. The run found that saved models lacked the chat template, which is fixed.
- Remaining work belongs to stage 3 (the new rows are still untrained).
