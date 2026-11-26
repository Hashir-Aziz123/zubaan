"""Stage 4 chat SFT on a tiny random Qwen3 model: masking, packing, resume, merge. CPU, seconds."""

from __future__ import annotations

import json

import pytest
import yaml
from typer.testing import CliRunner

torch = pytest.importorskip("torch")
pytest.importorskip("peft")

from test_tune_data import make_chat_tokenizer, record  # noqa: E402

from zubaan import cli  # noqa: E402
from zubaan.config import TuneConfig  # noqa: E402
from zubaan.extend.build import METADATA_FILE  # noqa: E402
from zubaan.tune.data import IGNORE, pad_batch, row_arrays  # noqa: E402
from zubaan.tune.sft import (  # noqa: E402
    batch_loss,
    check_packing,
    load_data,
    merge_tuned,
    prepare_model,
    run_tune,
)

runner = CliRunner()


@pytest.fixture
def world(tmp_path, tiny_base_json, tiny_qwen3):
    """A model folder (tiny Qwen3, chat tokenizer, zubaan metadata), conversations, a config."""
    tok = make_chat_tokenizer(tiny_base_json)
    model_dir = tmp_path / "stage3"
    tok.save_pretrained(model_dir)
    rows = len(tok)
    tiny_qwen3(rows, seed=1).save_pretrained(model_dir)
    first_new = rows - 60
    (model_dir / METADATA_FILE).write_text(
        json.dumps({"first_new_id": first_new, "trained": True}), encoding="utf-8"
    )
    recs = [record(i, i % 3 != 0, passage=f"urd-{i // 2}") for i in range(40)]
    data = tmp_path / "conversations.jsonl"
    data.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in recs), encoding="utf-8")
    cfg = TuneConfig(
        model_path=model_dir,
        train_file=data,
        max_len=256,
        val_share=0.2,
        output_dir=tmp_path / "tuned",
        epochs=1.0,
        micro_batch_size=1,
        grad_accum=2,
        lr_lora=1e-2,
        warmup_steps=1,
        lora_rank=4,
        lora_alpha=8,
        precision="fp32",
        gradient_checkpointing=False,
        device="cpu",
        log_every=1,
        eval_every=2,
        checkpoint_every=2,
    )
    return {"tok": tok, "cfg": cfg, "rows": rows, "first_new": first_new}


def _model(world):
    from transformers import AutoModelForCausalLM

    return AutoModelForCausalLM.from_pretrained(world["cfg"].model_path, dtype=torch.float32)


def test_packed_loss_equals_each_conversation_alone(world):
    """The loss of a packed row is the sum of each conversation's own reply loss.

    This is the check that masking (prompt tokens never trained), position restarts (no attention
    across conversations) and logits_to_keep (only reply positions) are all right.
    """
    data = load_data(world["cfg"], world["tok"])
    model = _model(world).eval()
    a, b, c = data["train"][:3]
    with torch.no_grad():
        loss, n = batch_loss(model, pad_batch([row_arrays([a, b, c])], 0), "cpu", "fp32")
        expected, count = 0.0, 0
        for ex in (a, b, c):
            ids, labels, _ = row_arrays([ex])
            out = model(input_ids=torch.tensor([ids]), labels=torch.tensor([labels]))
            k = sum(1 for t in labels[1:] if t != IGNORE)
            expected += float(out.loss) * k
            count += k
    assert n == count == a.n_trained + b.n_trained + c.n_trained
    assert float(loss) == pytest.approx(expected, rel=1e-4)


def test_check_packing_passes_on_this_transformers_version(world):
    data = load_data(world["cfg"], world["tok"])
    model, _ = prepare_model(_model(world), world["cfg"], None)
    assert check_packing(model, data["train"], "cpu", "fp32") < 1e-4


def test_load_data_splits_by_passage_and_reports_counts(world):
    data = load_data(world["cfg"], world["tok"])
    stats = data["stats"]
    assert stats["train"]["conversations"] + stats["val"]["conversations"] == 40
    assert stats["val"]["conversations"] > 0
    assert stats["train"]["rows"] == len(data["train_rows"])
    assert 0 < stats["train"]["row_fill"] <= 1
    assert all(len(e) <= world["cfg"].max_len for e in data["train"])


def test_run_tune_trains_and_resumes_exactly(world):
    cfg = world["cfg"]
    full = run_tune(cfg, log=lambda s: None)
    assert full["finished"]
    total = full["total_steps"]
    assert total >= 3
    losses = full["evals"]
    assert losses and losses[-1]["eval"]["val_tokens"] > 0
    final = torch.load(cfg.output_dir / "checkpoints" / f"ckpt-{total:07d}" / "lora.pt")

    split = cfg.model_copy(update={"output_dir": cfg.output_dir.parent / "split"})
    first = run_tune(split, log=lambda s: None, stop_at_step=2)
    assert not first["finished"] and first["final_step"] == 2
    second = run_tune(split, log=lambda s: None)
    assert second["resumed_from_step"] == 2 and second["finished"]
    resumed = torch.load(split.output_dir / "checkpoints" / f"ckpt-{total:07d}" / "lora.pt")
    for k, v in final.items():
        assert torch.equal(v, resumed[k]), k


def test_resume_refuses_changed_data(world):
    cfg = world["cfg"]
    run_tune(cfg, log=lambda s: None, max_steps=2)
    with cfg.train_file.open("a", encoding="utf-8") as f:
        f.write("\n" + json.dumps(record(99, True, passage="urd-99"), ensure_ascii=False))
    with pytest.raises(ValueError, match="train_file_sha256"):
        run_tune(cfg, log=lambda s: None)


def test_merge_keeps_the_function_and_the_frozen_rows(world):
    cfg = world["cfg"]
    run_tune(cfg, log=lambda s: None)
    out = merge_tuned(cfg, log=lambda s: None)
    assert out["merge_max_logit_diff"] < 1e-3
    folder = cfg.output_dir / "model"
    meta = json.loads((folder / METADATA_FILE).read_text(encoding="utf-8"))
    assert meta["first_new_id"] == world["first_new"] and meta["stage4"]["step"] == out["step"]
    for name in ("tokenizer.json", "tokenizer_config.json"):
        assert (folder / name).read_bytes() == (cfg.model_path / name).read_bytes()
    from transformers import AutoModelForCausalLM

    merged = AutoModelForCausalLM.from_pretrained(folder, dtype=torch.float32)
    original = _model(world)
    assert torch.equal(
        merged.get_input_embeddings().weight,
        original.get_input_embeddings().weight.bfloat16().float(),
    )
    probe = torch.tensor([[1, 2, 3, 4, 5]])
    assert not torch.allclose(merged(input_ids=probe).logits, original(input_ids=probe).logits)


def test_train_new_rows_changes_only_the_new_rows(world):
    cfg = world["cfg"].model_copy(update={"train_new_rows": True, "lr_new_rows": 1e-1})
    run_tune(cfg, log=lambda s: None)
    merge_tuned(cfg.model_copy(update={"merged_dtype": "float32"}), log=lambda s: None)
    from transformers import AutoModelForCausalLM

    merged = AutoModelForCausalLM.from_pretrained(cfg.output_dir / "model", dtype=torch.float32)
    before = _model(world).get_input_embeddings().weight
    after = merged.get_input_embeddings().weight
    n = world["first_new"]
    assert torch.equal(after[:n], before[:n])
    assert not torch.equal(after[n:], before[n:])


def test_cli_tune_run_smoke(world, tmp_path):
    cfg = world["cfg"]
    path = tmp_path / "tune.yaml"
    path.write_text(yaml.safe_dump({"tune": json.loads(cfg.model_dump_json())}), encoding="utf-8")
    result = runner.invoke(cli.app, ["tune", "run", "--config", str(path), "--steps", "1"])
    assert result.exit_code == 0, result.output
    assert "stopped early at step 1" in result.output
