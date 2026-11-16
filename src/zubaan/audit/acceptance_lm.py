"""Acceptance rates from real draft and target models (torch and transformers, lazy).

Both models read the same real sentence (teacher forcing). At every position each model
gives a distribution over the next token, and the acceptance rate is their overlap,
sum(min(p_target, p_draft)), exactly as ``acceptance.acceptance_rate`` defines it.

Memory: ``joint`` keeps both models loaded. ``sequential`` holds one model at a time: it
runs the draft over a chunk of texts, keeps those distributions in float16, unloads it,
then loads the target to compare. That trades reload time for a lower memory peak.
"""

from __future__ import annotations

import gc
import time
from collections.abc import Callable, Iterator, Sequence
from functools import partial

from zubaan.audit.acceptance import measure_acceptance, speedup_table, summarize_acceptance
from zubaan.text.normalize import normalize_urdu

DTYPES = ("float32", "bfloat16", "float16")
MODES = ("joint", "sequential")


def overlap_batched(p, q):
    """sum(min(p, q)) over the last axis: one acceptance rate per position."""
    import torch

    return torch.minimum(p, q).sum(dim=-1)


def load_causal_lm(name: str, dtype: str = "bfloat16", device: str = "cpu"):
    """Load a causal language model for inference only."""
    import torch
    from transformers import AutoModelForCausalLM

    if dtype not in DTYPES:
        raise ValueError(f"dtype must be one of {DTYPES}")
    model = AutoModelForCausalLM.from_pretrained(name, dtype=getattr(torch, dtype))
    return model.to(device).eval()


def next_token_probs(model, ids: Sequence[int], device: str = "cpu"):
    """Distributions for positions 1..n-1 of a text, as float32 on the CPU.

    The forward pass runs in the model's dtype; softmax runs in float32 so the
    probabilities stay accurate even for a bfloat16 model.
    """
    import torch

    with torch.inference_mode():
        x = torch.tensor([list(ids)], device=device)
        logits = model(input_ids=x).logits[0, :-1]
        return torch.softmax(logits.float(), dim=-1).cpu()


def _compare(pt, pd) -> tuple[list[float], list[bool]]:
    if pt.shape != pd.shape:
        raise ValueError(f"target and draft outputs differ in shape: {pt.shape} vs {pd.shape}")
    # sum(min(p, q)) <= 1 mathematically; float32 rounding can overshoot to 1.0000001.
    alphas = overlap_batched(pt, pd.float()).clamp(max=1.0).tolist()
    greedy = (pt.argmax(dim=-1) == pd.argmax(dim=-1)).tolist()
    return alphas, greedy


def check_same_vocabulary(draft_tok, target_tok) -> None:
    """Speculative decoding needs one shared vocabulary: same tokens, same ids."""
    if draft_tok.get_vocab() != target_tok.get_vocab():
        raise ValueError("draft and target tokenizers differ; they must share one vocabulary")


def _chunks(items: list, size: int) -> Iterator[list]:
    for i in range(0, len(items), size):
        yield items[i : i + size]


def acceptance_from_models(
    draft: str,
    target: str,
    token_ids: list[list[int]],
    dtype: str = "bfloat16",
    device: str = "cpu",
    mode: str = "joint",
    chunk_size: int = 16,
    on_text: Callable[[int], None] | None = None,
) -> tuple[list[list[float]], list[list[bool]], dict]:
    """Per-position acceptance and greedy agreement for every text, plus run info.

    Run info holds parameter counts (for the draft cost estimate) and timings.
    """
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    if chunk_size < 1:
        raise ValueError("chunk_size must be at least 1")
    info: dict = {"dtype": dtype, "device": device, "mode": mode}
    start = time.perf_counter()
    done = 0

    def tick() -> None:
        nonlocal done
        done += 1
        if on_text is not None:
            on_text(done)

    if mode == "joint":
        d_model = load_causal_lm(draft, dtype, device)
        t_model = d_model if draft == target else load_causal_lm(target, dtype, device)
        info["draft_params"] = d_model.num_parameters()
        info["target_params"] = t_model.num_parameters()

        def pair(ids: list[int]) -> tuple[list[float], list[bool]]:
            out = _compare(
                next_token_probs(t_model, ids, device), next_token_probs(d_model, ids, device)
            )
            tick()
            return out

        alphas, greedy = measure_acceptance(token_ids, pair)
    else:
        alphas, greedy = [], []
        usable = [ids for ids in token_ids if len(ids) >= 2]
        for chunk in _chunks(usable, chunk_size):
            d_model = load_causal_lm(draft, dtype, device)
            info["draft_params"] = d_model.num_parameters()
            cached = [next_token_probs(d_model, ids, device).half() for ids in chunk]
            del d_model
            gc.collect()
            t_model = load_causal_lm(target, dtype, device)
            info["target_params"] = t_model.num_parameters()
            for ids, pd in zip(chunk, cached, strict=True):
                a, g = _compare(next_token_probs(t_model, ids, device), pd)
                alphas.append(a)
                greedy.append(g)
                tick()
            del t_model, cached
            gc.collect()
    info["seconds"] = round(time.perf_counter() - start, 1)
    return alphas, greedy, info


def acceptance_report(
    draft: str,
    target: str,
    texts_by_lang: dict[str, list[str]],
    normalize_lang: str | None = None,
    dtype: str = "bfloat16",
    device: str = "cpu",
    mode: str = "joint",
    chunk_size: int = 16,
    draft_cost: float | None = None,
    gammas: Sequence[int] = range(1, 7),
    on_text: Callable[[str, int, int], None] | None = None,
) -> dict:
    """Acceptance per language for one draft/target pair, with modelled speedups.

    ``normalize_lang`` names the language whose texts go through ``normalize_urdu``.
    If ``draft_cost`` is None it is estimated as draft parameters / target parameters:
    generating one token mostly means reading every weight once, so time scales with size.
    """
    from transformers import AutoTokenizer

    target_tok = AutoTokenizer.from_pretrained(target)
    if draft != target:
        check_same_vocabulary(AutoTokenizer.from_pretrained(draft), target_tok)

    out: dict = {"draft": draft, "target": target, "languages": {}}
    for lang, texts in texts_by_lang.items():
        if lang == normalize_lang:
            texts = [normalize_urdu(t) for t in texts]
        ids = [target_tok.encode(t, add_special_tokens=False) for t in texts]
        progress = partial(on_text, lang, total=len(ids)) if on_text else None
        alphas, greedy, info = acceptance_from_models(
            draft, target, ids, dtype, device, mode, chunk_size, on_text=progress
        )
        cost = draft_cost
        if cost is None:
            cost = info["draft_params"] / info["target_params"]
        summary = summarize_acceptance(alphas, greedy)
        out["languages"][lang] = {
            **summary.to_dict(),
            "draft_cost": cost,
            "draft_cost_source": "given" if draft_cost is not None else "parameter ratio",
            "speedup": speedup_table(summary.alpha, cost, gammas),
            "per_text_alpha": [sum(a) / len(a) for a in alphas],
            "run": info,
        }
    return out
