"""Stage 2 onward: deployable exports.

Planned modules:
- gguf.py: run llama.cpp's converter, handle the pre-tokenizer fingerprint for an
  extended tokenizer, quantize with an Urdu imatrix, and assert token-for-token parity
  between Hugging Face and llama.cpp tokenization on a test set.
- vllm.py: emit serve commands including speculative decoding settings.
- ollama.py: emit a Modelfile.
"""
