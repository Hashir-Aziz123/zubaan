"""Stage 2: tokenizer extension.

Planned modules:
- train_bpe.py: learn target-language merges with the base model's pre-tokenizer.
- merge.py: append new merges to the base tokenizer (as real merges, not added_tokens)
  and verify English tokenization is unchanged and Urdu round-trips.
- init_emb.py: resize embeddings and LM head; initialize new rows (mean of old pieces,
  with langsfer methods as alternatives).
"""
