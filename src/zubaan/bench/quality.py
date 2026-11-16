"""Quality metrics that stay fair when the tokenizer changes.

A model with an extended tokenizer reads and writes different tokens than the base, so
per-token perplexity and per-token accuracy cannot be compared. Two measures are used instead:

* **Bits per character.** Total cross-entropy of a text divided by its number of characters
  and ln 2. The same text has the same characters whatever the tokenizer.
* **Belebele multiple choice.** The model scores each of four answers as a continuation of
  the passage and question. The answer with the highest score wins. The score is reported two
  ways: the plain sum of log-probabilities (``acc``) and the sum divided by the answer's
  character count (``acc_char``, tokenizer-fair). An answer written with fewer tokens collects
  fewer negative terms, which favors the extended tokenizer under ``acc``, so ``acc_char`` is
  the number to compare.

The model-facing functions import torch lazily. The statistics at the top need nothing.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence

from zubaan.metrics import bits_per_character

CHANCE_4_CHOICES = 0.25


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for a proportion.

    centre = (p + z^2 / 2n) / (1 + z^2 / n)
    half   = z x sqrt(p (1 - p) / n + z^2 / 4n^2) / (1 + z^2 / n)

    Better than the plain normal interval for accuracies near 0.25 or 1 with a few hundred
    questions. Example: 225 correct of 900 (p = 0.25) gives about [0.223, 0.279].
    """
    if n <= 0:
        raise ValueError("n must be positive")
    if not 0 <= successes <= n:
        raise ValueError("successes must be between 0 and n")
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def pick_answer(scores: Sequence[float]) -> int:
    """Index of the highest score; the first one wins ties."""
    if not scores:
        raise ValueError("no scores")
    return max(range(len(scores)), key=scores.__getitem__)


def multiple_choice_accuracy(
    logprobs: Sequence[Sequence[float]],
    n_chars: Sequence[Sequence[int]],
    correct: Sequence[int],
) -> dict:
    """Accuracy from per-answer log-probabilities.

    ``logprobs[q][a]`` is the summed log-probability of answer ``a`` of question ``q``;
    ``n_chars[q][a]`` its character count; ``correct[q]`` the index of the right answer.
    ``acc`` picks by the sum; ``acc_char`` picks by sum / characters.
    """
    if not (len(logprobs) == len(n_chars) == len(correct)):
        raise ValueError("logprobs, n_chars and correct must have one entry per question")
    if not logprobs:
        raise ValueError("no questions")
    hit = hit_char = 0
    for lp, nc, c in zip(logprobs, n_chars, correct, strict=True):
        if len(lp) != len(nc):
            raise ValueError("each question needs one character count per answer")
        if any(k <= 0 for k in nc):
            raise ValueError("answers must have at least one character")
        hit += pick_answer(lp) == c
        hit_char += pick_answer([x / k for x, k in zip(lp, nc, strict=True)]) == c
    n = len(logprobs)
    return {
        "n": n,
        "acc": hit / n,
        "acc_ci": list(wilson_interval(hit, n)),
        "acc_char": hit_char / n,
        "acc_char_ci": list(wilson_interval(hit_char, n)),
        "chance": CHANCE_4_CHOICES,
    }


def belebele_prompt(item: dict) -> str:
    """Zero-shot prompt in the style of the lm-evaluation-harness Belebele task."""
    return f"P: {item['flores_passage'].strip()}\nQ: {item['question'].strip()}\nA:"


def belebele_choices(item: dict) -> list[str]:
    """The four answer strings of a Belebele item, in order."""
    return [item[f"mc_answer{i}"].strip() for i in (1, 2, 3, 4)]


def belebele_correct(item: dict) -> int:
    """Zero-based index of the right answer (the dataset stores it as '1' to '4')."""
    k = int(item["correct_answer_num"]) - 1
    if not 0 <= k < 4:
        raise ValueError(f"correct_answer_num out of range: {item['correct_answer_num']!r}")
    return k


# model-facing ------------------------------------------------------------------------------------


def _pad_batch(seqs: list[list[int]], pad_id: int, device: str):
    import torch

    width = max(len(s) for s in seqs)
    ids = torch.full((len(seqs), width), pad_id, dtype=torch.long)
    mask = torch.zeros((len(seqs), width), dtype=torch.long)
    for i, s in enumerate(seqs):
        ids[i, : len(s)] = torch.tensor(s)
        mask[i, : len(s)] = 1
    return ids.to(device), mask.to(device)


def token_logprobs(model, seqs: list[list[int]], pad_id: int, device: str = "cpu"):
    """Log-probability of every token after the first, for each sequence in the batch.

    Returns a list of float32 CPU tensors, one per sequence, of length ``len(seq) - 1``.
    The normalizer runs in float32 whatever the model's dtype: log p = logit - logsumexp(logits).
    This avoids building a second full (batch, length, vocabulary) log-softmax tensor, which
    for a 150,000-token vocabulary is the largest memory cost of scoring.
    """
    import torch

    ids, mask = _pad_batch(seqs, pad_id, device)
    with torch.inference_mode():
        logits = model(input_ids=ids, attention_mask=mask).logits[:, :-1]
        target = logits.gather(-1, ids[:, 1:].unsqueeze(-1)).squeeze(-1).float()
        logp = target - torch.logsumexp(logits.float(), dim=-1)
        del logits
    return [logp[i, : len(s) - 1].cpu() for i, s in enumerate(seqs)]


def token_budget_batches(
    lengths: Sequence[int], max_items: int, max_tokens: int
) -> list[list[int]]:
    """Group positions of ``lengths`` (already sorted ascending) into batches.

    A batch holds at most ``max_items`` sequences and at most ``max_tokens`` padded tokens
    (count x longest length), so memory per forward pass stays bounded even when a few texts
    are long. A single sequence longer than the budget still gets its own batch.
    """
    if max_items < 1 or max_tokens < 1:
        raise ValueError("max_items and max_tokens must be at least 1")
    batches: list[list[int]] = []
    current: list[int] = []
    for i, n in enumerate(lengths):
        if current and (len(current) + 1 > max_items or (len(current) + 1) * n > max_tokens):
            batches.append(current)
            current = []
        current.append(i)
    if current:
        batches.append(current)
    return batches


def text_bits_per_character(
    model,
    tokenizer,
    texts: Sequence[str],
    context_id: int,
    batch_size: int = 4,
    max_tokens: int = 1024,
    device: str = "cpu",
    on_progress: Callable[[int, int], None] | None = None,
    max_batch_tokens: int = 2048,
) -> dict:
    """Bits per character of ``model`` on ``texts``.

    Every text is tokenized and given ``context_id`` (an end-of-text token) as a one-token
    context, so every real token is predicted and none is free. ``tokenizer`` is a
    ``tokenizers.Tokenizer``. Raises if a text is longer than ``max_tokens``: cutting it would
    change which characters are counted, differently for each tokenizer.
    """
    if not texts:
        raise ValueError("no texts to score")
    encoded = []
    for t in texts:
        ids = tokenizer.encode(t, add_special_tokens=False).ids
        if not ids:
            raise ValueError("a text has no tokens")
        if len(ids) + 1 > max_tokens:
            raise ValueError(f"a text has {len(ids)} tokens, more than max_tokens={max_tokens}")
        encoded.append((len(t), [context_id, *ids]))
    order = sorted(range(len(texts)), key=lambda i: len(encoded[i][1]))
    total_nll = 0.0
    n_tokens = 0
    lengths = [len(encoded[i][1]) for i in order]
    done = 0
    for group in token_budget_batches(lengths, batch_size, max_batch_tokens):
        batch = [encoded[order[k]][1] for k in group]
        for lp in token_logprobs(model, batch, context_id, device):
            total_nll -= float(lp.double().sum())
            n_tokens += lp.numel()
        done += len(group)
        if on_progress is not None:
            on_progress(done, len(order))
    n_chars = sum(c for c, _ in encoded)
    return {
        "n_texts": len(texts),
        "n_chars": n_chars,
        "n_tokens": n_tokens,
        "total_nll_nats": total_nll,
        "bits_per_character": bits_per_character(total_nll, n_chars),
        "bits_per_token": total_nll / (n_tokens * math.log(2)),
        "tokens_per_char": n_tokens / n_chars,
    }


def belebele_scores(
    model,
    tokenizer,
    items: Sequence[dict],
    context_id: int,
    device: str = "cpu",
    max_tokens: int = 2048,
    on_progress: Callable[[int, int], None] | None = None,
    max_batch_tokens: int = 2048,
) -> dict:
    """Score every answer of every Belebele item and return accuracy.

    For each answer the prompt and the answer (with a leading space) are tokenized
    separately and joined, and the answer tokens' log-probabilities are summed. The four
    answers of an item run as one batch.
    """
    logprobs: list[list[float]] = []
    chars: list[list[int]] = []
    correct: list[int] = []
    for n_done, item in enumerate(items, 1):
        prompt_ids = [
            context_id,
            *tokenizer.encode(belebele_prompt(item), add_special_tokens=False).ids,
        ]
        choices = belebele_choices(item)
        seqs, spans = [], []
        for ch in choices:
            cont = tokenizer.encode(" " + ch, add_special_tokens=False).ids
            if not cont:
                raise ValueError("an answer has no tokens")
            seq = prompt_ids + cont
            if len(seq) > max_tokens:
                raise ValueError(f"an item has {len(seq)} tokens, more than max_tokens")
            seqs.append(seq)
            spans.append(len(cont))
        lps = []
        for group in token_budget_batches([len(x) for x in seqs], 4, max_batch_tokens):
            lps += token_logprobs(model, [seqs[k] for k in group], context_id, device)
        logprobs.append([float(lp[-k:].double().sum()) for lp, k in zip(lps, spans, strict=True)])
        chars.append([len(c) + 1 for c in choices])  # +1 for the leading space that was scored
        correct.append(belebele_correct(item))
        if on_progress is not None:
            on_progress(n_done, len(items))
    return multiple_choice_accuracy(logprobs, chars, correct)


def load_belebele(lang: str, n: int | None = None, split: str = "test") -> list[dict]:
    """Belebele items for a language code such as ``urd_Arab`` (needs the ``datasets`` package).

    The 900 test questions are returned in dataset order; ``n`` keeps the first n, so a
    subset is the same on every machine.
    """
    from datasets import load_dataset

    data = load_dataset("facebook/belebele", lang, split=split)
    rows = [dict(r) for r in data]
    return rows if n is None else rows[:n]
