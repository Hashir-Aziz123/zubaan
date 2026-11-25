import pytest

torch = pytest.importorskip("torch")

from zubaan.adapt.params import (  # noqa: E402
    attach_new_rows,
    merge_new_rows,
    set_phase,
)

ROWS, FIRST_NEW, HIDDEN = 60, 50, 16


def ids(*rows):
    return torch.tensor(rows)


@pytest.fixture
def model(tiny_qwen3):
    return tiny_qwen3(ROWS, seed=1)


def logits(m, x):
    with torch.no_grad():
        return m(input_ids=x).logits


def test_attached_model_computes_exactly_what_it_did_before(model):
    x = ids([1, 5, 52, 59, 7, 50])  # mixes old and new ids
    before = logits(model, x)
    handle = attach_new_rows(model, FIRST_NEW)
    after = logits(model, x)
    assert torch.allclose(before, after, atol=1e-6)
    handle.remove_hooks()
    assert torch.equal(logits(model, x), before)  # hooks leave no trace


def test_new_rows_parameter_has_the_hand_computed_size(model):
    handle = attach_new_rows(model, FIRST_NEW)
    assert handle.n_new == ROWS - FIRST_NEW == 10
    assert handle.new_rows.shape == (10, HIDDEN)
    assert handle.new_rows.numel() == 10 * HIDDEN
    assert handle.new_rows.dtype == torch.float32
    assert torch.equal(handle.new_rows, model.get_input_embeddings().weight[FIRST_NEW:])


def test_embedding_output_uses_the_new_rows_for_new_ids_only(model):
    handle = attach_new_rows(model, FIRST_NEW)
    emb = model.get_input_embeddings()
    plain = emb.weight.detach().clone()
    with torch.no_grad():
        handle.new_rows += 1.0
    out = emb(ids([3, 50, 59, 49]))[0]
    assert torch.equal(out[0], plain[3]) and torch.equal(out[3], plain[49])
    assert torch.allclose(out[1], plain[50] + 1.0) and torch.allclose(out[2], plain[59] + 1.0)
    # An input with no new id at all still works and is unchanged.
    assert torch.equal(emb(ids([1, 2]))[0], plain[[1, 2]])


def test_old_token_logits_ignore_the_new_rows_when_no_new_id_is_in_the_input(model):
    handle = attach_new_rows(model, FIRST_NEW)
    x = ids([1, 5, 7, 9, 30])
    base = logits(model, x)
    with torch.no_grad():
        handle.new_rows.add_(torch.randn_like(handle.new_rows))
    changed = logits(model, x)
    assert torch.equal(changed[..., :FIRST_NEW], base[..., :FIRST_NEW])
    assert not torch.allclose(changed[..., FIRST_NEW:], base[..., FIRST_NEW:])


def test_phase_a_trains_only_the_new_rows_and_leaves_everything_else_bit_identical(model):
    snapshot = {k: v.detach().clone() for k, v in model.state_dict().items()}
    handle = attach_new_rows(model, FIRST_NEW)
    counts = set_phase(model, handle, "A")
    assert counts == {"phase": "A", "new_rows": 10 * HIDDEN, "lora": 0}
    trainable = [n for n, p in model.named_parameters() if p.requires_grad]
    assert trainable == []  # the new rows live outside the model
    assert handle.new_rows.requires_grad

    opt = torch.optim.AdamW([handle.new_rows], lr=1e-2)
    x = ids([1, 51, 55, 4, 58, 50, 2, 59])
    start = handle.new_rows.detach().clone()
    for _ in range(5):
        loss = model(input_ids=x, labels=x).loss
        loss.backward()
        opt.step()
        opt.zero_grad()
    assert not torch.equal(handle.new_rows, start)  # the rows learned
    for k, v in model.state_dict().items():
        assert torch.equal(v, snapshot[k]), k  # the model itself did not change at all
    assert all(p.grad is None for p in model.parameters())


def test_training_lowers_the_loss_on_a_small_repeated_batch(model):
    handle = attach_new_rows(model, FIRST_NEW)
    set_phase(model, handle, "A")
    opt = torch.optim.AdamW([handle.new_rows], lr=5e-2)
    x = ids([50, 51, 52, 53, 54, 55, 56, 57] * 2)
    first = model(input_ids=x, labels=x).loss.item()
    for _ in range(40):
        loss = model(input_ids=x, labels=x).loss
        loss.backward()
        opt.step()
        opt.zero_grad()
    assert loss.item() < first * 0.8


def test_merge_writes_rows_back_into_the_tied_matrix(model):
    handle = attach_new_rows(model, FIRST_NEW)
    with torch.no_grad():
        handle.new_rows.add_(0.5)
    x = ids([3, 52, 58, 9])
    hooked = logits(model, x)
    old_part = model.get_input_embeddings().weight[:FIRST_NEW].detach().clone()
    merge_new_rows(model, handle)
    emb, head = model.get_input_embeddings(), model.get_output_embeddings()
    assert emb.weight.data_ptr() == head.weight.data_ptr()  # still tied
    assert torch.equal(emb.weight[:FIRST_NEW], old_part)
    assert torch.equal(emb.weight[FIRST_NEW:], handle.new_rows.detach())
    assert handle.hooks == []
    assert torch.allclose(logits(model, x), hooked, atol=1e-5)  # plain model == hooked model


def test_merge_converts_to_the_model_dtype(tiny_qwen3):
    m = tiny_qwen3(ROWS, seed=2, dtype="float16")
    handle = attach_new_rows(m, FIRST_NEW)
    assert handle.new_rows.dtype == torch.float32  # master copy stays float32
    merge_new_rows(m, handle)
    assert m.get_input_embeddings().weight.dtype == torch.float16


def test_untied_models_and_bad_ids_are_refused(tiny_qwen3):
    with pytest.raises(ValueError, match="tie"):
        attach_new_rows(tiny_qwen3(ROWS, tie=False), FIRST_NEW)
    m = tiny_qwen3(ROWS)
    for bad in (0, ROWS, ROWS + 5):
        with pytest.raises(ValueError, match="inside"):
            attach_new_rows(m, bad)


def test_set_phase_rejects_unknown_phase(model):
    handle = attach_new_rows(model, FIRST_NEW)
    with pytest.raises(ValueError, match="phase"):
        set_phase(model, handle, "C")


# LoRA --------------------------------------------------------------------------------------------


@pytest.fixture
def peft_model(model):
    pytest.importorskip("peft")
    from zubaan.adapt.params import apply_lora

    handle = attach_new_rows(model, FIRST_NEW)
    return apply_lora(model, rank=4, alpha=8, dropout=0.0), handle


def test_lora_phases_toggle_exactly_the_adapter_tensors(peft_model):
    from zubaan.adapt.params import lora_parameters

    pm, handle = peft_model
    lora = lora_parameters(pm)
    assert lora  # adapters were attached
    a = set_phase(pm, handle, "A")
    assert a["lora"] == 0 and not any(p.requires_grad for p in lora)
    b = set_phase(pm, handle, "B")
    assert b["lora"] == sum(p.numel() for p in lora) > 0
    assert all(p.requires_grad for p in lora)
    # Everything that is not an adapter stays frozen in phase B.
    assert all(not p.requires_grad for n, p in pm.named_parameters() if "lora_" not in n)


def test_phase_b_updates_adapters_and_new_rows_but_not_the_base(peft_model):
    pm, handle = peft_model
    set_phase(pm, handle, "B")
    base = {n: p.detach().clone() for n, p in pm.named_parameters() if "lora_" not in n}
    lora = {n: p.detach().clone() for n, p in pm.named_parameters() if "lora_" in n}
    params = [handle.new_rows] + [p for p in pm.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=1e-2)
    x = ids([1, 51, 55, 4, 58, 50, 2, 59])
    for _ in range(3):
        pm(input_ids=x, labels=x).loss.backward()
        opt.step()
        opt.zero_grad()
    for n, p in pm.named_parameters():
        if "lora_" in n:
            continue
        assert torch.equal(p, base[n]), n
    assert any(not torch.equal(p, lora[n]) for n, p in pm.named_parameters() if "lora_" in n)


def test_merging_lora_keeps_the_logits(peft_model):
    from zubaan.adapt.params import lora_parameters, merge_lora

    pm, handle = peft_model
    with torch.no_grad():
        for p in lora_parameters(pm):  # lora_B starts at zero; make the adapter do something
            p.add_(0.05 * torch.randn_like(p))
        handle.new_rows.add_(0.1)
    x = ids([1, 51, 55, 4, 58, 50])
    before = logits(pm, x)
    plain = merge_lora(pm)
    assert torch.allclose(logits(plain, x), before, atol=1e-4)
    merge_new_rows(plain, handle)
    assert torch.allclose(logits(plain, x), before, atol=1e-4)
    names = [n for n, _ in plain.named_parameters()]
    assert not any("lora_" in n for n in names)  # a plain model again


# control run (decision 6): base model, no new rows -----------------------------------------------


def test_control_handle_has_no_rows_no_hooks_and_no_phase_a(model):
    pytest.importorskip("peft")
    from zubaan.adapt.params import apply_lora, lora_parameters, no_new_rows

    x = ids([1, 5, 52, 59, 7, 50])
    before = logits(model, x)
    handle = no_new_rows(model)
    assert handle.new_rows is None and handle.n_new == 0 and handle.first_new_id == ROWS
    assert handle.hooks == []
    pm = apply_lora(model, rank=4, alpha=8, dropout=0.0)
    with pytest.raises(ValueError, match="phase_a_steps: 0"):
        set_phase(pm, handle, "A")
    b = set_phase(pm, handle, "B")
    assert b["new_rows"] == 0 and b["lora"] == sum(p.numel() for p in lora_parameters(pm)) > 0
    assert torch.equal(logits(pm, x), before)  # LoRA B starts at zero: same function
    merge_new_rows(pm, handle)  # nothing to write back, must not fail
