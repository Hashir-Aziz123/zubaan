import pytest

torch = pytest.importorskip("torch")

from zubaan.extend.init_emb import (  # noqa: E402
    INIT_METHODS,
    mean_of_pieces,
    merge_pieces,
    random_rows,
    resize_and_init,
)
from zubaan.extend.merge import extend_tokenizer_json, load_fast_tokenizer  # noqa: E402
from zubaan.extend.sweep import learn_urdu_merges  # noqa: E402
from zubaan.extend.tokenizer_info import inspect_tokenizer_json  # noqa: E402
from zubaan.extend.train_bpe import Merge  # noqa: E402

# merge_pieces ------------------------------------------------------------------------------------


def test_merge_pieces_worked_example():
    vocab = {"A": 10, "B": 11, "C": 12}
    merges = [Merge("A", "B", 5), Merge("AB", "C", 4), Merge("C", "AB", 3), Merge("ABC", "ABC", 2)]
    assert merge_pieces(vocab, merges) == [
        [10, 11],  # AB
        [10, 11, 12],  # ABC = AB + C
        [12, 10, 11],  # CAB = C + AB
        [10, 11, 12, 10, 11, 12],  # ABCABC
    ]


def test_merge_pieces_rejects_unknown_parts():
    with pytest.raises(ValueError, match="neither a base token"):
        merge_pieces({"A": 0}, [Merge("A", "Z", 1)])
    with pytest.raises(ValueError, match="neither a base token"):
        merge_pieces({"A": 0, "B": 1}, [Merge("AB", "A", 1)])  # AB not created yet


def test_merge_pieces_agree_with_the_base_tokenizer(tiny_base_json, urdu_lines):
    """A new token's pieces are exactly what the base BPE gives for the token's own bytes."""
    merges = learn_urdu_merges(urdu_lines, tiny_base_json, 120, min_count=2)
    vocab = tiny_base_json["model"]["vocab"]
    pieces = merge_pieces(vocab, merges)
    base = load_fast_tokenizer(tiny_base_json)
    inv = {i: t for t, i in vocab.items()}
    for m, p in zip(merges, pieces, strict=True):
        assert "".join(inv[i] for i in p) == m.token  # pieces spell the token
        assert all(i in inv for i in p)
    # For tokens that are whole words, the base tokenizer splits them into the same pieces.
    same = sum(
        [t.value for t in base.model.tokenize(m.token)] == [inv[i] for i in p]
        for m, p in zip(merges, pieces, strict=True)
    )
    assert same >= len(merges) * 0.5


# mean_of_pieces and random_rows ------------------------------------------------------------------


def test_mean_of_pieces_worked_example():
    # Rows: id0 = [1, 2], id1 = [3, 4], id2 = [5, 6]. Token (0,1) -> [2, 3]; token (2) -> [5, 6].
    w = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    out = mean_of_pieces(w, [[0, 1], [2], [0, 0, 2]])
    assert torch.allclose(out, torch.tensor([[2.0, 3.0], [5.0, 6.0], [7 / 3, 10 / 3]]))


def test_mean_of_pieces_edge_cases():
    w = torch.eye(3)
    assert mean_of_pieces(w, []).shape == (0, 3)
    with pytest.raises(ValueError, match="no pieces"):
        mean_of_pieces(w, [[]])
    half = torch.tensor([[1.0, 3.0], [3.0, 5.0]], dtype=torch.bfloat16)
    assert mean_of_pieces(half, [[0, 1]]).dtype == torch.float32  # computed in float32


def test_random_rows_have_the_scale_of_the_old_rows_and_are_seeded():
    g = torch.Generator().manual_seed(3)
    w = torch.randn(5000, 8, generator=g) * torch.arange(1, 9) + 2.0
    r = random_rows(w, 20000, seed=1)
    assert r.shape == (20000, 8)
    assert torch.allclose(r.mean(0), w.mean(0), atol=0.15)
    assert torch.allclose(r.std(0), w.std(0), rtol=0.05)
    assert torch.equal(r, random_rows(w, 20000, seed=1))
    assert not torch.equal(r, random_rows(w, 20000, seed=2))


# resize_and_init on a tiny model -----------------------------------------------------------------

OLD, PAD, NEW = 40, 48, 12  # real ids 0..39, matrix padded to 48 rows, 12 new tokens


def fake_pieces(n=NEW):
    return [[i % OLD, (3 * i + 1) % OLD, (7 * i + 2) % OLD][: 1 + i % 3] for i in range(n)]


@pytest.mark.parametrize("method", ["mean", "random"])
def test_old_rows_are_kept_bit_for_bit_and_shapes_are_right(tiny_qwen3, method):
    model = tiny_qwen3(PAD)
    before = model.get_input_embeddings().weight.detach().clone()
    info = resize_and_init(model, fake_pieces(), OLD, method)
    emb = model.get_input_embeddings().weight
    assert emb.shape == (OLD + NEW, 16) and model.config.vocab_size == OLD + NEW
    assert torch.equal(emb[:OLD], before[:OLD])
    assert model.get_output_embeddings().weight.shape == (OLD + NEW, 16)
    assert info["embedding_rows_before"] == PAD and info["embedding_rows_after"] == OLD + NEW
    assert info["tied"] is True


def test_tied_embeddings_stay_one_tensor(tiny_qwen3):
    model = tiny_qwen3(PAD, tie=True)
    resize_and_init(model, fake_pieces(), OLD, "mean")
    emb, head = model.get_input_embeddings().weight, model.get_output_embeddings().weight
    assert head.data_ptr() == emb.data_ptr()
    emb.data[OLD] += 1.0
    assert torch.equal(head.data[OLD], emb.data[OLD])  # same storage, so the head changed too


def test_mean_rows_equal_the_hand_computed_mean(tiny_qwen3):
    model = tiny_qwen3(PAD)
    w = model.get_input_embeddings().weight.detach().clone()
    pieces = fake_pieces()
    resize_and_init(model, pieces, OLD, "mean")
    new = model.get_input_embeddings().weight.detach()
    for i, p in enumerate(pieces):
        expected = sum(w[j] for j in p) / len(p)
        assert torch.allclose(new[OLD + i], expected, atol=1e-6)


def test_rows_inside_the_padding_are_overwritten_not_left_as_junk(tiny_qwen3):
    # Ids OLD..PAD-1 already exist in the padded matrix; they must become the new tokens.
    model = tiny_qwen3(PAD)
    junk = model.get_input_embeddings().weight[OLD:PAD].detach().clone()
    resize_and_init(model, fake_pieces(), OLD, "mean")
    new = model.get_input_embeddings().weight[OLD:PAD].detach()
    assert not torch.allclose(new, junk)


def test_growing_beyond_the_padding_and_shrinking_below_it(tiny_qwen3):
    big = tiny_qwen3(PAD)
    resize_and_init(big, fake_pieces(30), OLD, "mean")
    assert big.get_input_embeddings().weight.shape[0] == OLD + 30 > PAD
    small = tiny_qwen3(PAD)
    resize_and_init(small, fake_pieces(3), OLD, "mean")
    assert small.get_input_embeddings().weight.shape[0] == OLD + 3 < PAD


def test_the_old_tokens_get_identical_logits_after_extension(tiny_qwen3):
    """Text with no new tokens must behave the same: logits over the old ids do not change."""
    model = tiny_qwen3(PAD)
    ids = torch.randint(0, OLD, (3, 20), generator=torch.Generator().manual_seed(0))
    with torch.no_grad():
        before = model(input_ids=ids).logits[..., :OLD].clone()
        resize_and_init(model, fake_pieces(), OLD, "mean")
        after = model(input_ids=ids).logits
    assert after.shape[-1] == OLD + NEW
    assert torch.allclose(before, after[..., :OLD], atol=1e-5)
    assert torch.equal(before.argmax(-1), after[..., :OLD].argmax(-1))


def test_untied_model_gets_head_rows_from_the_head(tiny_qwen3):
    model = tiny_qwen3(PAD, tie=False)
    head_before = model.get_output_embeddings().weight.detach().clone()
    emb_before = model.get_input_embeddings().weight.detach().clone()
    pieces = fake_pieces()
    info = resize_and_init(model, pieces, OLD, "mean")
    assert info["tied"] is False
    head = model.get_output_embeddings().weight.detach()
    emb = model.get_input_embeddings().weight.detach()
    assert head.data_ptr() != emb.data_ptr()
    assert torch.equal(head[:OLD], head_before[:OLD]) and torch.equal(emb[:OLD], emb_before[:OLD])
    for i, p in enumerate(pieces):
        assert torch.allclose(head[OLD + i], sum(head_before[j] for j in p) / len(p), atol=1e-6)
        assert torch.allclose(emb[OLD + i], sum(emb_before[j] for j in p) / len(p), atol=1e-6)


def test_half_precision_models_keep_their_dtype(tiny_qwen3):
    model = tiny_qwen3(PAD, dtype="bfloat16")
    resize_and_init(model, fake_pieces(), OLD, "mean")
    assert model.get_input_embeddings().weight.dtype == torch.bfloat16


def test_bad_arguments(tiny_qwen3):
    model = tiny_qwen3(PAD)
    with pytest.raises(ValueError, match="method must be"):
        resize_and_init(model, fake_pieces(), OLD, "bogus")
    with pytest.raises(ValueError, match="above the"):
        resize_and_init(model, fake_pieces(), PAD + 1, "mean")
    with pytest.raises(ValueError, match="needs focus_rows"):
        resize_and_init(model, fake_pieces(), OLD, "focus")
    wrong = torch.zeros(NEW + 1, 16)
    with pytest.raises(ValueError, match="shape"):
        resize_and_init(model, fake_pieces(), OLD, "focus", focus_rows=wrong)
    assert INIT_METHODS == ("mean", "random", "focus")


def test_random_is_seeded_and_differs_from_mean(tiny_qwen3):
    a, b, c = tiny_qwen3(PAD), tiny_qwen3(PAD), tiny_qwen3(PAD)
    resize_and_init(a, fake_pieces(), OLD, "random", seed=5)
    resize_and_init(b, fake_pieces(), OLD, "random", seed=5)
    resize_and_init(c, fake_pieces(), OLD, "mean")
    ea, eb, ec = (m.get_input_embeddings().weight for m in (a, b, c))
    assert torch.equal(ea, eb) and not torch.allclose(ea[OLD:], ec[OLD:])


def test_end_to_end_with_a_real_extended_tokenizer(tiny_base_json, urdu_lines, tiny_qwen3):
    """Tokenizer, merges and model agree: ids stay inside the matrix and new rows are means."""
    merges = learn_urdu_merges(urdu_lines, tiny_base_json, 80, min_count=2)
    ext = extend_tokenizer_json(tiny_base_json, merges)
    info = inspect_tokenizer_json(tiny_base_json)
    model = tiny_qwen3(info.total_vocab_size + 10)  # padded like Qwen3 (rows above the vocab)
    w = model.get_input_embeddings().weight.detach().clone()
    pieces = merge_pieces(tiny_base_json["model"]["vocab"], merges)
    resize_and_init(model, pieces, info.next_free_id, "mean")
    total = inspect_tokenizer_json(ext).total_vocab_size
    assert model.get_input_embeddings().weight.shape[0] == total
    tok = load_fast_tokenizer(ext)
    ids = tok.encode(" ".join(urdu_lines[:3]), add_special_tokens=False).ids
    assert max(ids) < total and max(ids) >= info.next_free_id  # uses new tokens
    with torch.no_grad():
        logits = model(input_ids=torch.tensor([ids])).logits
    assert logits.shape == (1, len(ids), total) and torch.isfinite(logits).all()
    new_id = info.next_free_id
    expected = sum(w[j] for j in pieces[0]) / len(pieces[0])
    assert torch.allclose(model.get_input_embeddings().weight[new_id], expected, atol=1e-6)
