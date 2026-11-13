import random

import pytest

from zubaan.audit.acceptance import acceptance_rate

torch = pytest.importorskip("torch")

from zubaan.audit.acceptance_lm import _compare, overlap_batched  # noqa: E402


def _random_dists(n, vocab, seed):
    g = torch.Generator().manual_seed(seed)
    return torch.softmax(torch.randn(n, vocab, generator=g) * 3, dim=-1)


def test_overlap_batched_matches_reference_formula():
    p, q = _random_dists(5, 50, 0), _random_dists(5, 50, 1)
    fast = overlap_batched(p, q).tolist()
    slow = [acceptance_rate(p[i].tolist(), q[i].tolist()) for i in range(5)]
    assert fast == pytest.approx(slow, abs=1e-6)


def test_compare_self_is_perfect():
    p = _random_dists(4, 20, random.randint(0, 100))
    alphas, greedy = _compare(p, p.clone())
    assert alphas == pytest.approx([1.0] * 4, abs=1e-6)
    assert all(greedy)


def test_compare_rejects_vocabulary_mismatch():
    with pytest.raises(ValueError, match="shape"):
        _compare(_random_dists(2, 10, 0), _random_dists(2, 11, 0))


@pytest.mark.slow
@pytest.mark.parametrize("mode", ["joint", "sequential"])
def test_tiny_model_against_itself(mode):
    from zubaan.audit.acceptance_lm import acceptance_from_models

    name = "hf-internal-testing/tiny-random-LlamaForCausalLM"
    alphas, greedy, info = acceptance_from_models(
        name, name, [[1, 5, 9, 13], [2, 6]], dtype="float32", mode=mode, chunk_size=1
    )
    assert [len(a) for a in alphas] == [3, 1]
    # float16 caching in sequential mode costs a little precision.
    assert all(x == pytest.approx(1.0, abs=2e-3) for a in alphas for x in a)
    assert info["draft_params"] == info["target_params"]
