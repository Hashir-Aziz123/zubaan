import json
import random
from collections import Counter

import pytest

from zubaan.extend.train_bpe import Merge, learn_merges, word_piece_counts

# Classic worked example (Sennrich et al.): words as sequences of single characters.
WORDS = {
    tuple("low"): 5,
    tuple("lower"): 2,
    tuple("newest"): 6,
    tuple("widest"): 3,
}


def test_worked_example_by_hand():
    """Each step is derived by hand from the pair counts, with ties broken alphabetically.

    Start: es=9 (newest 6 + widest 3), st=9, lo=7, ow=7, we=8 (lower 2 + newest 6), ...
    1. es and st tie at 9; "e" < "s" so merge (e, s) -> es.      newest = n e w es t
    2. (es, t) = 9 is now the top pair -> est.                  newest = n e w est
    3. lo and ow tie at 7; (l, o) < (o, w) -> lo.               low = lo w
    4. (lo, w) = 7 -> low.                                      lower = low e r
    5. ew, ne and w-est tie at 6; (e, w) sorts first -> ew.     newest = n ew est
    """
    merges = learn_merges(WORDS, 5)
    assert [(m.left, m.right, m.count) for m in merges] == [
        ("e", "s", 9),
        ("es", "t", 9),
        ("l", "o", 7),
        ("lo", "w", 7),
        ("e", "w", 6),
    ]
    assert [m.token for m in merges] == ["es", "est", "lo", "low", "ew"]


def test_prefix_property_longer_run_starts_with_shorter_run():
    long = learn_merges(WORDS, 12, min_count=1)
    for k in range(0, len(long) + 1):
        assert learn_merges(WORDS, k, min_count=1) == long[:k]


def test_min_count_stops_early():
    merges = learn_merges(WORDS, 100, min_count=4)
    assert merges and all(m.count >= 4 for m in merges)
    assert len(merges) < len(learn_merges(WORDS, 100, min_count=1))
    assert learn_merges(WORDS, 100, min_count=1000) == []


def test_zero_merges_and_empty_input():
    assert learn_merges(WORDS, 0) == []
    assert learn_merges({}, 5) == []
    assert learn_merges({("a",): 10}, 5) == []  # one piece: no pairs


def test_invalid_arguments():
    with pytest.raises(ValueError, match="non-negative"):
        learn_merges(WORDS, -1)
    with pytest.raises(ValueError, match="min_count"):
        learn_merges(WORDS, 1, min_count=0)
    with pytest.raises(ValueError, match="positive"):
        learn_merges({("a", "b"): 0}, 1)


def test_existing_tokens_are_never_recreated():
    # "es" already exists in the base vocabulary, so (e, s) must be skipped for good.
    merges = learn_merges(WORDS, 5, existing_tokens=["es"])
    assert "es" not in [m.token for m in merges]
    assert ("e", "s") not in [(m.left, m.right) for m in merges]
    # The words are still single characters, so the next best pairs are derived by hand:
    # st=9 -> "st"; then (e, st)=9 -> "est"; then lo and low as before.
    assert [m.token for m in merges] == ["st", "est", "lo", "low", "ew"]


def test_pair_filter_blocks_pair_and_everything_built_on_it():
    merges = learn_merges(WORDS, 10, pair_filter=lambda a, b: (a, b) != ("e", "s"), min_count=1)
    pairs = [(m.left, m.right) for m in merges]
    assert ("e", "s") not in pairs and "es" not in [m.token for m in merges]
    # "est" is still reachable by another route: s + t first, then e + st.
    assert ("s", "t") in pairs and ("e", "st") in pairs
    assert "low" in [m.token for m in merges]
    # Blocking every pair that involves "s" removes st, est and anything built on them.
    none = learn_merges(WORDS, 10, pair_filter=lambda a, b: "s" not in a + b, min_count=1)
    assert not any("s" in m.token for m in none)


def test_each_token_is_created_once():
    # (a, bc) and (ab, c) would both make "abc"; only the first may be merged.
    words = {("a", "b", "c"): 3, ("ab", "c"): 2, ("a", "bc"): 4}
    tokens = [m.token for m in learn_merges(words, 10, min_count=1)]
    assert len(tokens) == len(set(tokens))
    assert tokens.count("abc") == 1


def test_run_of_identical_pieces_merges_left_to_right():
    # "aaa" -> [aa, a]. The pair (a, a) occurs twice, which is what the count reports: counts
    # are pair occurrences, so a run of identical pieces is counted more than it can be merged.
    merges = learn_merges({("a", "a", "a"): 1}, 5, min_count=1)
    assert merges == [Merge("a", "a", 2), Merge("aa", "a", 1)]


def test_result_does_not_depend_on_dict_order():
    items = list(WORDS.items())
    base = learn_merges(dict(items), 8, min_count=1)
    for seed in range(5):
        random.Random(seed).shuffle(items)
        assert learn_merges(dict(items), 8, min_count=1) == base


def naive_learn(words, n, existing=(), pair_filter=None, min_count=2):
    """Slow reference: recount every pair from scratch at each step."""
    seqs = {w: list(w) for w in words}
    vocab = set(existing) | {p for w in words for p in w}
    out = []
    banned = set()
    while len(out) < n:
        counts: Counter = Counter()
        for w, seq in seqs.items():
            for a, b in zip(seq, seq[1:], strict=False):
                counts[(a, b)] += words[w]
        cands = [(-c, a, b) for (a, b), c in counts.items() if (a, b) not in banned]
        if not cands:
            break
        c, a, b = min(cands)
        c = -c
        if c < min_count:
            break
        if a + b in vocab or (pair_filter and not pair_filter(a, b)):
            banned.add((a, b))
            continue
        out.append(Merge(a, b, c))
        vocab.add(a + b)
        for w, seq in seqs.items():
            new, i = [], 0
            while i < len(seq):
                if i + 1 < len(seq) and seq[i] == a and seq[i + 1] == b:
                    new.append(a + b)
                    i += 2
                else:
                    new.append(seq[i])
                    i += 1
            seqs[w] = new
    return out


@pytest.mark.parametrize("seed", range(40))
def test_matches_slow_reference_on_random_corpora(seed):
    rng = random.Random(seed)
    alphabet = list("abcdexyz")
    words = {}
    for _ in range(rng.randint(3, 25)):
        w = tuple(rng.choice(alphabet) for _ in range(rng.randint(1, 8)))
        words[w] = rng.randint(1, 20)
    existing = {a + b for a in alphabet for b in alphabet if rng.random() < 0.1}
    block = rng.choice(alphabet)

    def flt(a, b):
        return block not in a + b[:1]

    for min_count in (1, 2):
        fast = learn_merges(words, 30, existing, flt, min_count)
        slow = naive_learn(words, 30, existing, flt, min_count)
        assert fast == slow


# word_piece_counts: needs the tokenizers library and the tiny base tokenizer -------------------


def test_word_pieces_match_the_tokenizers_own_segmentation(tiny_base_json, urdu_lines):
    from tokenizers import Tokenizer

    tok = Tokenizer.from_str(json.dumps(tiny_base_json))
    lines = ["The government announced new projects.", urdu_lines[0], "Mixed " + urdu_lines[1]]
    counts = word_piece_counts(lines, tiny_base_json)
    expected: Counter = Counter()
    for line in lines:
        # Encode word by word is not possible, so compare the flattened pieces of all lines.
        for piece in tok.encode(line, add_special_tokens=False).tokens:
            expected[piece] += 1
    got: Counter = Counter()
    for word, n in counts.items():
        for piece in word:
            got[piece] += n
    assert got == expected


def test_word_pieces_concatenate_to_the_pre_tokenized_word(tiny_base_json, urdu_lines):
    from tokenizers import Tokenizer

    tok = Tokenizer.from_str(json.dumps(tiny_base_json))
    counts = word_piece_counts(urdu_lines[:5], tiny_base_json)
    words = {
        w
        for line in urdu_lines[:5]
        for w, _ in tok.pre_tokenizer.pre_tokenize_str(tok.normalizer.normalize_str(line))
    }
    assert {"".join(pieces) for pieces in counts} == words
    # Urdu is absent from the tiny tokenizer's training text, so words stay as single bytes.
    longest = max(len(p) for pieces in counts for p in pieces)
    assert longest <= 3


def test_word_pieces_count_repeated_words(tiny_base_json):
    # Words: "the", " the", " the", "the" (a leading space belongs to the next word).
    counts = word_piece_counts(["the the the", "the"], tiny_base_json)
    assert sum(counts.values()) == 4
    assert sorted(counts.values()) == [2, 2]


def test_word_pieces_need_a_pre_tokenizer(tiny_base_json):
    data = json.loads(json.dumps(tiny_base_json))
    data["pre_tokenizer"] = None
    with pytest.raises(ValueError, match="pre-tokenizer"):
        word_piece_counts(["abc"], data)
