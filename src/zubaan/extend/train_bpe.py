"""Learn new BPE merges on top of a base tokenizer, from target-language text.

Why this is "continued" BPE and not a fresh BPE run
---------------------------------------------------
A new merge is appended after all the base merges, so it only ever sees text that the base
tokenizer has already split as far as it can. A merge learned from raw bytes can name a pair
that never occurs after base tokenization, and would never fire. Instead we first tokenize
every word with the base model, then repeatedly merge the most frequent adjacent pair of
those pieces. Each learned merge is then guaranteed to be applied by the extended tokenizer
in the same order, and the frequency counts are exactly the tokens it saves.

The base pre-tokenizer is used as it is (loaded from tokenizer.json), so word boundaries
cannot change. The learning loop itself is pure Python and needs no model or extra package.
"""

from __future__ import annotations

import heapq
import json
from collections import Counter
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass


@dataclass(frozen=True)
class Merge:
    """One learned merge: ``left`` and ``right`` join into the token ``left + right``.

    ``count`` is how many times the pair occurred (over all words, weighted by word
    frequency) when it was chosen, which is also the number of tokens it saved.
    """

    left: str
    right: str
    count: int

    @property
    def token(self) -> str:
        return self.left + self.right


PairFilter = Callable[[str, str], bool]


def word_piece_counts(lines: Iterable[str], tokenizer_json: dict) -> Counter:
    """Count words as tuples of base-tokenizer pieces.

    Each line is normalized and pre-tokenized exactly as the base tokenizer does, then every
    distinct word is split into pieces by the base BPE model (cached per word). Returns a
    Counter mapping ``(piece, piece, ...)`` to how often that word occurred.
    """
    from tokenizers import Tokenizer

    tok = Tokenizer.from_str(json.dumps(tokenizer_json, ensure_ascii=False))
    if tok.pre_tokenizer is None:
        raise ValueError("the base tokenizer has no pre-tokenizer")
    normalize = tok.normalizer.normalize_str if tok.normalizer is not None else (lambda s: s)
    pre = tok.pre_tokenizer.pre_tokenize_str
    model = tok.model
    words: Counter = Counter()
    for line in lines:
        for word, _ in pre(normalize(line)):
            words[word] += 1
    out: Counter = Counter()
    for word, n in words.items():
        out[tuple(t.value for t in model.tokenize(word))] += n
    return out


def _pairs(seq: list[int]) -> Iterable[tuple[int, int]]:
    return zip(seq, seq[1:], strict=False)


def _apply_merge(seq: list[int], a: int, b: int, new: int) -> list[int]:
    """Replace each non-overlapping ``a b`` with ``new``, scanning left to right."""
    out: list[int] = []
    i, n = 0, len(seq)
    while i < n:
        if i + 1 < n and seq[i] == a and seq[i + 1] == b:
            out.append(new)
            i += 2
        else:
            out.append(seq[i])
            i += 1
    return out


def learn_merges(
    word_counts: Mapping[tuple[str, ...], int],
    n_merges: int,
    existing_tokens: Iterable[str] = (),
    pair_filter: PairFilter | None = None,
    min_count: int = 2,
) -> list[Merge]:
    """Greedy BPE: repeatedly merge the most frequent adjacent pair, up to ``n_merges`` times.

    ``word_counts`` maps each word (a tuple of current pieces) to its frequency.
    A pair is skipped for good if ``pair_filter(left, right)`` is False, or if the merged
    token already exists in ``existing_tokens`` or was produced earlier, because a merge must
    create a new token. Stops early if no eligible pair occurs at least ``min_count`` times.
    Ties are broken by the pair's strings, so the result does not depend on dict order.

    Because each step only depends on the steps before it, the first ``k`` merges of a long
    run equal the result of a run asked for ``k`` merges.
    """
    if n_merges < 0:
        raise ValueError("n_merges must be non-negative")
    if min_count < 1:
        raise ValueError("min_count must be at least 1")

    ids: dict[str, int] = {}
    strs: list[str] = []

    def tid(s: str) -> int:
        if s not in ids:
            ids[s] = len(strs)
            strs.append(s)
        return ids[s]

    seqs: list[list[int]] = []
    freqs: list[int] = []
    for word, n in word_counts.items():
        if n <= 0:
            raise ValueError("word counts must be positive")
        seqs.append([tid(p) for p in word])
        freqs.append(n)

    vocab = set(existing_tokens) | set(strs)
    pair_count: dict[tuple[int, int], int] = {}
    pair_words: dict[tuple[int, int], set[int]] = {}
    for w, seq in enumerate(seqs):
        for p in _pairs(seq):
            pair_count[p] = pair_count.get(p, 0) + freqs[w]
            pair_words.setdefault(p, set()).add(w)

    heap = [(-c, strs[p[0]], strs[p[1]], p) for p, c in pair_count.items()]
    heapq.heapify(heap)
    banned: set[tuple[int, int]] = set()
    merges: list[Merge] = []

    while len(merges) < n_merges and heap:
        neg, ls, rs, pair = heapq.heappop(heap)
        count = -neg
        if pair in banned or pair_count.get(pair, 0) != count:
            continue  # stale entry: the count changed after it was pushed
        if count < min_count:
            break
        token = ls + rs
        if token in vocab or (pair_filter is not None and not pair_filter(ls, rs)):
            banned.add(pair)
            continue

        new = tid(token)
        vocab.add(token)
        merges.append(Merge(ls, rs, count))
        changed: set[tuple[int, int]] = set()
        for w in list(pair_words.get(pair, ())):
            seq, f = seqs[w], freqs[w]
            for p in _pairs(seq):
                c = pair_count[p] - f
                if c:
                    pair_count[p] = c
                else:
                    del pair_count[p]
                changed.add(p)
            seq = _apply_merge(seq, pair[0], pair[1], new)
            seqs[w] = seq
            for p in _pairs(seq):
                pair_count[p] = pair_count.get(p, 0) + f
                pair_words.setdefault(p, set()).add(w)
                changed.add(p)
        pair_count.pop(pair, None)
        for p in changed:
            c = pair_count.get(p, 0)
            if c and p not in banned:
                heapq.heappush(heap, (-c, strs[p[0]], strs[p[1]], p))
    return merges
