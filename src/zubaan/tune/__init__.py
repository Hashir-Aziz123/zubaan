"""Stage 4: restore chat behavior after continued pretraining (docs/stage4-plan.md).

Modules:
- client.py: a standard-library client for OpenAI-compatible chat servers (vLLM, llama.cpp).
- generate.py: Wikipedia passages, teacher-written questions, teacher answers (chunks 0, 2, 3).
- filters.py: split a reply into reasoning and answer; loop and source-mention checks.

Planned: data.py (chat rendering, masking, packing), sft.py (training loop), chat_check.py.
"""
