import pytest

torch = pytest.importorskip("torch")

from zubaan.extend.focus import (  # noqa: E402
    AuxEmbeddings,
    build_aux_embeddings,
    cooccurrence,
    focus_rows,
    ppmi,
    sparsemax,
)
from zubaan.extend.init_emb import mean_of_pieces, resize_and_init  # noqa: E402


def test_sparsemax_worked_examples():
    # z = [0.5, 0.3, 0.1]: all three are in the support, tau = (0.9 - 1) / 3.
    out = sparsemax(torch.tensor([0.5, 0.3, 0.1]))
    assert torch.allclose(out, torch.tensor([0.5 + 1 / 30, 0.3 + 1 / 30, 0.1 + 1 / 30]))
    # z = [2, 0, 0]: only the first survives, the others get exactly zero.
    assert sparsemax(torch.tensor([2.0, 0.0, 0.0])).tolist() == [1.0, 0.0, 0.0]
    # Equal scores stay equal.
    assert torch.allclose(sparsemax(torch.tensor([1.0, 1.0])), torch.tensor([0.5, 0.5]))


def test_sparsemax_is_a_probability_vector_and_sparse_on_random_input():
    g = torch.Generator().manual_seed(0)
    z = torch.randn(200, 50, generator=g)
    p = sparsemax(z, dim=-1)
    assert torch.allclose(p.sum(-1), torch.ones(200), atol=1e-5)
    assert (p >= 0).all()
    assert (p == 0).any()  # unlike softmax it produces exact zeros
    assert torch.equal(p.argmax(-1), z.argmax(-1))  # order is preserved


def test_sparsemax_ignores_a_constant_shift_and_works_on_other_dims():
    z = torch.randn(4, 6, generator=torch.Generator().manual_seed(1))
    assert torch.allclose(sparsemax(z), sparsemax(z + 5.0), atol=1e-5)
    assert torch.allclose(sparsemax(z, dim=0), sparsemax(z.T, dim=-1).T, atol=1e-6)


def test_cooccurrence_by_hand():
    # One line [0, 1, 2], window 2. Pairs: (0,1) d1, (1,2) d1, (0,2) d2 -> weight 1/2.
    c = cooccurrence(torch.tensor([0, 1, 2]), torch.tensor([0, 0, 0]), window=2, n_tokens=3)
    dense = c.to_dense()
    expected = torch.tensor([[0, 1, 0.5], [1, 0, 1], [0.5, 1, 0]])
    assert torch.allclose(dense, expected)


def test_cooccurrence_never_crosses_lines():
    c = cooccurrence(torch.tensor([0, 1, 2, 3]), torch.tensor([0, 0, 1, 1]), window=3, n_tokens=4)
    d = c.to_dense()
    assert d[0, 1] == 1 and d[2, 3] == 1
    assert d[1, 2] == 0 and d[0, 2] == 0 and d[0, 3] == 0
    with pytest.raises(ValueError, match="window"):
        cooccurrence(torch.tensor([0, 1]), torch.tensor([0, 0]), window=0)


def test_ppmi_worked_example():
    # Two tokens that only ever appear together: counts [[0, 4], [4, 0]].
    c = torch.sparse_coo_tensor([[0, 1], [1, 0]], [4.0, 4.0], (2, 2)).coalesce()
    out = ppmi(c, smoothing=1.0).to_dense()
    # p(0,1) = 4/8 = 0.5; p(0) = 0.5; p_s(1) = 0.5; pmi = log(0.5 / (0.5 x 0.5)) = log 2.
    assert torch.allclose(out, torch.tensor([[0, 0.6931472], [0.6931472, 0]]))


def test_ppmi_drops_negative_associations():
    # Token 2 pairs with everything equally often, so it has no positive association.
    idx = torch.tensor([[0, 1, 0, 2, 1, 2], [1, 0, 2, 0, 2, 1]])
    c = torch.sparse_coo_tensor(idx, torch.ones(6), (3, 3)).coalesce()
    out = ppmi(c, smoothing=1.0)
    assert (out.values() > 0).all()


def clustered_streams(n_lines=300, seed=0):
    """Tokens 0..9 co-occur with each other; tokens 10..19 with each other; never mixed."""
    g = torch.Generator().manual_seed(seed)
    streams = []
    for i in range(n_lines):
        base = 0 if i % 2 == 0 else 10
        streams.append((base + torch.randint(0, 10, (12,), generator=g)).tolist())
    return streams


def test_aux_embeddings_place_cooccurring_tokens_together():
    aux = build_aux_embeddings(clustered_streams(), dim=4, window=3, min_count=1)
    assert aux.ids.tolist() == list(range(20)) and aux.vectors.shape == (20, 4)
    v = torch.nn.functional.normalize(aux.vectors, dim=1)
    sim = v @ v.T
    same = (sim[:10, :10].sum() - 10) / 90  # mean over distinct pairs inside group A
    cross = sim[:10, 10:].mean()
    assert same > cross + 0.5


def test_aux_embeddings_are_deterministic_and_respect_min_count():
    streams = clustered_streams() + [[99, 98]]
    a = build_aux_embeddings(streams, dim=4, window=3, min_count=2, seed=1)
    b = build_aux_embeddings(streams, dim=4, window=3, min_count=2, seed=1)
    assert torch.equal(a.vectors, b.vectors) and 99 not in a.ids.tolist()
    with pytest.raises(ValueError, match="need more than"):
        build_aux_embeddings([[0, 1, 2, 3]], dim=10, min_count=1)
    with pytest.raises(ValueError, match="no token streams"):
        build_aux_embeddings([])


def test_lookup_returns_minus_one_for_unknown_ids():
    aux = AuxEmbeddings(ids=torch.tensor([2, 5, 9]), vectors=torch.zeros(3, 2))
    assert aux.lookup(torch.tensor([5, 4, 9, 100, 0])).tolist() == [1, -1, 2, -1, -1]


# focus_rows: weights and fallbacks, hand-built so the answer is known ---------------------------


def hand_setup():
    # Old tokens 0..3 with 2-d embeddings; aux vectors in 2-d. Two new tokens, ids 4 and 5.
    weight = torch.tensor([[1.0, 0.0], [0.0, 1.0], [2.0, 2.0], [-1.0, 3.0]])
    aux_vec = torch.tensor(
        [
            [1.0, 0.0],  # old 0
            [0.0, 1.0],  # old 1
            [-1.0, 0.0],  # old 2
            [0.0, -1.0],  # old 3
            [1.0, 0.0],  # new 4: identical direction to old 0
            [0.7071, 0.7071],  # new 5: halfway between old 0 and old 1
        ]
    )
    aux = AuxEmbeddings(ids=torch.arange(6), vectors=aux_vec)
    return weight, aux


def test_focus_copies_the_embedding_of_an_identical_neighbour():
    weight, aux = hand_setup()
    rows, info = focus_rows(weight, aux, first_new_id=4, n_new=2)
    # New token 4 has cosine 1 with old 0 and <= 0 with the others, so sparsemax gives it
    # weight 1 on old 0 and its row is exactly old 0's embedding.
    assert torch.allclose(rows[0], weight[0])
    assert info["n_with_vector"] == 2 and info["n_fallback_to_mean"] == 0
    assert info["n_old_candidates"] == 4


def test_focus_blends_two_equally_close_neighbours():
    weight, aux = hand_setup()
    rows, _ = focus_rows(weight, aux, first_new_id=4, n_new=2)
    # cos(new5, old0) = cos(new5, old1) = 0.7071; cos with old2 = -0.7071, old3 = -0.7071.
    # sparsemax([.7071, .7071, -.7071, -.7071]) = [0.5, 0.5, 0, 0] -> mean of old 0 and old 1.
    assert torch.allclose(rows[1], (weight[0] + weight[1]) / 2, atol=1e-3)


def test_focus_falls_back_to_the_mean_of_pieces_when_a_token_has_no_vector():
    weight, aux = hand_setup()
    aux = AuxEmbeddings(ids=aux.ids[:5], vectors=aux.vectors[:5])  # new token 5 has no vector
    pieces = [[0], [1, 2]]
    rows, info = focus_rows(weight, aux, 4, 2, fallback_pieces=pieces)
    assert info["n_fallback_to_mean"] == 1
    assert torch.allclose(rows[1], mean_of_pieces(weight, [[1, 2]])[0])
    with pytest.raises(ValueError, match="no auxiliary vector"):
        focus_rows(weight, aux, 4, 2)


def test_focus_needs_at_least_one_old_vector():
    weight = torch.eye(4)
    aux = AuxEmbeddings(ids=torch.tensor([4, 5]), vectors=torch.ones(2, 3))
    with pytest.raises(ValueError, match="no old token"):
        focus_rows(weight, aux, 4, 2)


def test_focus_chunking_does_not_change_the_result():
    g = torch.Generator().manual_seed(0)
    weight = torch.randn(30, 6, generator=g)
    aux = AuxEmbeddings(ids=torch.arange(50), vectors=torch.randn(50, 5, generator=g))
    a, _ = focus_rows(weight, aux, 30, 20, chunk=3)
    b, _ = focus_rows(weight, aux, 30, 20, chunk=1000)
    assert torch.allclose(a, b, atol=1e-6)


def test_focus_rows_plug_into_resize_and_init(tiny_qwen3):
    model = tiny_qwen3(48)
    w = model.get_input_embeddings().weight.detach().clone()
    g = torch.Generator().manual_seed(2)
    aux = AuxEmbeddings(ids=torch.arange(52), vectors=torch.randn(52, 8, generator=g))
    rows, _ = focus_rows(w, aux, 40, 12)
    resize_and_init(model, [[0]] * 12, 40, "focus", focus_rows=rows)
    new = model.get_input_embeddings().weight.detach()
    assert torch.equal(new[:40], w[:40]) and torch.allclose(new[40:], rows, atol=1e-6)
