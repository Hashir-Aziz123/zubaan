"""Stage 4: restore chat behavior after continued pretraining (docs/stage4-plan.md).

Modules:
- client.py: a standard-library client for OpenAI-compatible chat servers (vLLM, llama.cpp).
- generate.py: Wikipedia passages, teacher-written questions, teacher answers (chunks 0, 2, 3).
- filters.py: split a reply into reasoning and answer; loop and source-mention checks.
- chat_check.py: answer rate with thinking on and off, reply language and loops on a fixed
  hand-written prompt set (prompts/chat_check_v1.yaml), against any served model (chunk 1).
- data.py: render conversations with the model's chat template, mask prompts, pack rows.
- sft.py: LoRA SFT on the Stage 3 model (reply-only loss, resume, Hub checkpoints), merge.
"""
