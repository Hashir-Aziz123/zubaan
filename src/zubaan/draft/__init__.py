"""Stage 5: drafters that share the extended vocabulary.

Planned modules:
- distill.py: extend a small model with the same tokenizer, then distill it on the
  adapted target's outputs (top-k logprob KL).
- ngram.py: a cheap n-gram drafter built from target-language text.
"""
