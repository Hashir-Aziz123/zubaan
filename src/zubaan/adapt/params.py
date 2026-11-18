"""What is trained in stage 3: the new embedding rows, then LoRA adapters.

Qwen3-1.7B ties its input embedding and its output head: one matrix of ``first_new_id + N`` rows
serves both. Stage 3 may only change the ``N`` new rows. Two obvious ways to do that go wrong:

* Training the whole matrix would also change the old rows (English could drift) and would keep
  AdamW state for 331M values, about 4 GB extra.
* Masking gradients on the old rows still allocates that optimizer state.

So the new rows live in their own small parameter (``AdaptHandle.new_rows``, N x hidden, float32).
Two forward hooks overlay it on the frozen model:

* a hook on the input embedding replaces the output at positions whose id is >= ``first_new_id``;
* a hook on the output head replaces the last ``N`` logit columns with ``hidden @ new_rows^T``.

The model's own weights and state_dict are never touched during training. ``merge_new_rows``
writes the trained rows back into the tied matrix and removes the hooks, so the saved folder is
an ordinary Hugging Face model. Logits for the old tokens are unchanged by construction.

Torch and peft are imported inside the functions that need them.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

LORA_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


@dataclass
class AdaptHandle:
    """The trainable new rows plus the hooks that apply them to a model."""

    new_rows: object  # torch.nn.Parameter, shape (n_new, hidden), float32
    first_new_id: int
    n_new: int
    hooks: list = field(default_factory=list)

    def remove_hooks(self) -> None:
        for h in self.hooks:
            h.remove()
        self.hooks = []


def attach_new_rows(model, first_new_id: int) -> AdaptHandle:
    """Make the rows ``first_new_id`` and above of a tied model trainable, in their own tensor.

    The new parameter starts as a copy of the current rows (the Stage 2 initialization), so
    the model computes exactly what it did before until training changes them. Returns a handle;
    the model itself is not modified apart from the two hooks.
    """
    import torch

    emb = model.get_input_embeddings()
    head = model.get_output_embeddings()
    if head is None or head.weight.data_ptr() != emb.weight.data_ptr():
        raise ValueError(
            "stage 3 supports models that tie the input embedding and the output head "
            "(Qwen3-0.6B and 1.7B do); this model has separate matrices"
        )
    rows, hidden = emb.weight.shape
    if not 0 < first_new_id < rows:
        raise ValueError(f"first_new_id {first_new_id} must be inside the {rows} embedding rows")
    n_new = rows - first_new_id
    new_rows = torch.nn.Parameter(emb.weight[first_new_id:].detach().clone().float())
    handle = AdaptHandle(new_rows, first_new_id, n_new)

    def embed_hook(module, args, output):
        ids = args[0]
        is_new = ids >= first_new_id
        if not bool(is_new.any()):
            # Keep the graph connected so gradient checkpointing and the optimizer see the rows.
            return output + 0.0 * handle.new_rows.sum().to(output.dtype)
        idx = (ids - first_new_id).clamp(0, n_new - 1)
        rows_here = handle.new_rows[idx].to(output.dtype)
        return torch.where(is_new.unsqueeze(-1), rows_here, output)

    def head_hook(module, args, output):
        x = args[0]
        new_logits = torch.nn.functional.linear(x, handle.new_rows.to(x.dtype))
        return torch.cat([output[..., :first_new_id], new_logits.to(output.dtype)], dim=-1)

    handle.hooks = [
        emb.register_forward_hook(embed_hook),
        head.register_forward_hook(head_hook),
    ]
    return handle


def apply_lora(
    model,
    rank: int = 64,
    alpha: int = 128,
    dropout: float = 0.05,
    targets: Sequence[str] = LORA_TARGETS,
):
    """Wrap ``model`` with LoRA adapters on the attention and MLP projections (peft)."""
    from peft import LoraConfig, get_peft_model

    cfg = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=list(targets),
        bias="none",
        task_type="CAUSAL_LM",
    )
    return get_peft_model(model, cfg)


def lora_parameters(model) -> list:
    """The LoRA adapter tensors of a peft model."""
    return [p for n, p in model.named_parameters() if "lora_" in n]


def set_phase(model, handle: AdaptHandle, phase: str) -> dict:
    """Choose what trains. Phase ``A``: only the new rows. Phase ``B``: new rows and LoRA.

    Everything else is frozen. Returns the number of trainable values per group so callers (and
    tests) can check it against the hand calculation ``n_new x hidden``.
    """
    if phase not in ("A", "B"):
        raise ValueError("phase must be 'A' or 'B'")
    for p in model.parameters():
        p.requires_grad_(False)
    handle.new_rows.requires_grad_(True)
    lora = lora_parameters(model)
    for p in lora:
        p.requires_grad_(phase == "B")
    return {
        "phase": phase,
        "new_rows": handle.new_rows.numel(),
        "lora": sum(p.numel() for p in lora) if phase == "B" else 0,
    }


def merge_new_rows(model, handle: AdaptHandle) -> None:
    """Write the trained rows into the tied matrix and remove the hooks.

    ``model`` must be the plain Hugging Face model (after ``merge_lora``). Because the head is the
    same tensor as the embedding, one copy updates both.
    """
    import torch

    handle.remove_hooks()
    emb = model.get_input_embeddings()
    head = model.get_output_embeddings()
    if head is None or head.weight.data_ptr() != emb.weight.data_ptr():
        raise ValueError("the model is no longer tied; refusing to merge")
    with torch.no_grad():
        emb.weight[handle.first_new_id :] = handle.new_rows.detach().to(emb.weight.dtype)


def merge_lora(peft_model):
    """Fold the LoRA adapters into the base weights and return the plain model."""
    return peft_model.merge_and_unload()
