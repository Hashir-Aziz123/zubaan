"""Stage 3 training loop and commands on a tiny random model: seconds, CPU, no downloads."""

import json
import shutil
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

torch = pytest.importorskip("torch")
pytest.importorskip("peft")

from zubaan import cli  # noqa: E402
from zubaan.adapt.cpt import (  # noqa: E402
    TrainingDiverged,
    latest_checkpoint,
    lr_at,
    phase_at,
    prepare_for_training,
    train,
)
from zubaan.adapt.data import BlockSampler, load_blocks  # noqa: E402
from zubaan.adapt.pipeline import run_adapt, run_prepare  # noqa: E402
from zubaan.config import AdaptConfig, ExtendConfig  # noqa: E402
from zubaan.extend.pipeline import run_extend  # noqa: E402
from zubaan.extend.tokenizer_info import inspect_tokenizer_json  # noqa: E402

runner = CliRunner()


def test_lr_schedule_worked_example():
    # peak 1e-3, warmup 10, total 110 steps, floor 10% of the peak.
    f = lambda step: lr_at(step, 110, 1e-3, 10, 0.1)  # noqa: E731
    assert f(0) == pytest.approx(1e-4)  # warmup: peak x (0 + 1) / 10
    assert f(9) == pytest.approx(1e-3)  # last warmup step reaches the peak
    assert f(10) == pytest.approx(1e-3)  # decay starts at p = 0
    assert f(60) == pytest.approx(5.5e-4)  # p = 0.5: 1e-3 x (0.1 + 0.9 x 0.5)
    assert f(109) == pytest.approx(1e-4, rel=1e-2)  # p = 0.99: just above the floor
    assert lr_at(0, 1, 1e-3, 0, 0.1) == pytest.approx(1e-3)
    with pytest.raises(ValueError):
        lr_at(0, 0, 1e-3, 0)


def test_phase_at():
    cfg = AdaptConfig(phase_a_steps=3, phase_b_steps=5)
    assert phase_at(0, cfg) == ("A", 0, 3)
    assert phase_at(2, cfg) == ("A", 2, 3)
    assert phase_at(3, cfg) == ("B", 0, 5)
    assert phase_at(7, cfg) == ("B", 4, 5)


@pytest.fixture
def world(tmp_path, tiny_base_json, urdu_lines, english_lines, tiny_qwen3):
    """A tiny extended model folder, raw text files, and a config that fits them."""
    base = tmp_path / "base"
    base.mkdir()
    (base / "tokenizer.json").write_text(json.dumps(tiny_base_json), encoding="utf-8")
    (base / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    data = tmp_path / "data"
    data.mkdir()
    n = min(len(urdu_lines), len(english_lines))
    (data / "urd.txt").write_text("\n".join(urdu_lines[:n]) + "\n", encoding="utf-8")
    (data / "eng.txt").write_text("\n".join(english_lines[:n]) + "\n", encoding="utf-8")
    (data / "train.txt").write_text("\n".join(urdu_lines) + "\n", encoding="utf-8")
    ext_cfg = ExtendConfig(
        base_model=str(base),
        corpus=data / "train.txt",
        eval_urdu=data / "urd.txt",
        eval_english=data / "eng.txt",
        n_new_tokens=50,
        sizes=[20, 80],
        merges_file=tmp_path / "ext" / "merges.json",
        output_dir=tmp_path / "ext",
    )
    rows = inspect_tokenizer_json(tiny_base_json).total_vocab_size + 8
    run_extend(ext_cfg, log=lambda s: None, model_factory=lambda: tiny_qwen3(rows, seed=2))

    # Training text repeats the fixture lines; evaluation text shares no 8 consecutive words.
    (data / "urdu_adapt.txt").write_text("\n".join(urdu_lines * 6) + "\n", encoding="utf-8")
    (data / "eng_adapt.txt").write_text("\n".join(english_lines * 12) + "\n", encoding="utf-8")
    (data / "heldout.txt").write_text("\n".join(urdu_lines[:6]) + "\n", encoding="utf-8")
    (data / "guard_urd.txt").write_text(
        "یہ ایک بالکل الگ جملہ ہے جس میں کوئی بھی لفظ پہلے نہیں آیا تھا\n", encoding="utf-8"
    )
    (data / "guard_eng.txt").write_text(
        "this evaluation sentence shares nothing with any training text at all today\n",
        encoding="utf-8",
    )
    cfg = AdaptConfig(
        model_path=tmp_path / "ext" / "model",
        urdu_corpus=data / "urdu_adapt.txt",
        heldout=data / "heldout.txt",
        english_corpus=data / "eng_adapt.txt",
        guard_urdu=[data / "guard_urd.txt"],
        guard_english=[data / "guard_eng.txt"],
        guard_belebele=False,
        data_dir=tmp_path / "blocks",
        block_size=16,
        output_dir=tmp_path / "out",
        phase_a_steps=4,
        phase_b_steps=6,
        micro_batch_size=2,
        grad_accum=2,
        lr_new_rows=1e-2,
        lr_lora=1e-3,
        warmup_steps=2,
        lora_rank=4,
        lora_alpha=8,
        lora_dropout=0.0,
        precision="fp32",
        gradient_checkpointing=True,
        device="cpu",
        log_every=2,
        eval_every=5,
        eval_texts=4,
        eval_max_chars=100,
        checkpoint_every=2,
        keep_checkpoints=3,
    )
    return cfg, tmp_path


def test_prepare_writes_blocks_that_match_the_tokenizer(world):
    cfg, _ = world
    meta = run_prepare(cfg, log=lambda s: None)
    assert meta["urdu"]["n_blocks"] >= 10 and meta["english"]["n_blocks"] >= 3
    assert meta["overlap_guard"]["hits"] == 0
    assert len(meta["tokenizer_sha256"]) == 64
    assert meta["first_new_id"] > 0


def test_run_trains_logs_evals_and_checkpoints(world):
    cfg, tmp = world
    run_prepare(cfg, log=lambda s: None)
    summary = run_adapt(cfg, log=lambda s: None)
    assert summary["finished"] and summary["final_step"] == 10 == summary["total_steps"]
    assert summary["resumed_from_step"] == 0
    rows = [
        json.loads(line) for line in (cfg.output_dir / "train_log.jsonl").read_text().splitlines()
    ]
    losses = [r["loss"] for r in rows if "loss" in r]
    assert losses and all(x == x and x < 1e3 for x in losses)  # finite
    phases = [r["phase"] for r in rows if "phase" in r]
    assert phases[0] == "A" and phases[-1] == "B"
    evals = [r["eval"] for r in rows if "eval" in r]
    assert evals and "bpc_urdu_heldout" in evals[-1] and "bpc_english_flores" in evals[-1]
    ckpts = sorted((cfg.output_dir / "checkpoints").glob("ckpt-*"))
    assert len(ckpts) == 3  # pruned to keep_checkpoints
    assert ckpts[-1].name == "ckpt-0000010"
    assert (cfg.output_dir / "adapt_summary.json").exists()


def test_training_changes_only_the_new_rows_in_phase_a(world, tiny_qwen3):
    cfg, _ = world
    run_prepare(cfg, log=lambda s: None)
    sampler = BlockSampler(load_blocks(cfg.data_dir)[0], None, 0.0, 0)
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(cfg.model_path, dtype=torch.float32)
    before = {k: v.clone() for k, v in model.state_dict().items()}
    info = json.loads((cfg.model_path / "zubaan_extension.json").read_text())
    pm, handle = prepare_for_training(model, info["first_new_id"], cfg)
    start = handle.new_rows.detach().clone()
    cfg_a = cfg.model_copy(update={"phase_a_steps": 3, "phase_b_steps": 0 + 1})
    train(pm, handle, sampler, cfg_a, max_steps=3, log=lambda s: None)
    assert not torch.equal(handle.new_rows, start)
    lora_names = {n for n, _ in pm.named_parameters() if "lora_" in n}
    for k, v in pm.state_dict().items():
        clean = k.replace("base_model.model.", "").replace(".base_layer", "")
        if "lora_" in k or clean not in before:
            continue
        assert torch.equal(v, before[clean]), k  # phase A leaves every base weight alone
    lora_b = [p for n, p in pm.named_parameters() if n in lora_names and "lora_B" in n]
    assert all(torch.count_nonzero(p) == 0 for p in lora_b)  # adapters untouched in phase A


def test_resume_reproduces_the_uninterrupted_run_exactly(world):
    cfg, tmp = world
    run_prepare(cfg, log=lambda s: None)
    full_cfg = cfg.model_copy(update={"output_dir": tmp / "full"})
    run_adapt(full_cfg, log=lambda s: None)

    split_cfg = cfg.model_copy(update={"output_dir": tmp / "split"})
    first = run_adapt(split_cfg, log=lambda s: None, max_steps=6)  # stops inside phase B
    assert first["final_step"] == 6 and not first["finished"]
    second = run_adapt(split_cfg, log=lambda s: None)  # resumes from the step-6 checkpoint
    assert second["resumed_from_step"] == 6 and second["finished"]

    a = latest_checkpoint(full_cfg.output_dir / "checkpoints")
    b = latest_checkpoint(split_cfg.output_dir / "checkpoints")
    assert a.name == b.name == "ckpt-0000010"
    for name in ("new_rows.pt", "lora.pt"):
        x, y = torch.load(a / name), torch.load(b / name)
        if isinstance(x, dict):
            assert x.keys() == y.keys()
            assert all(torch.equal(x[k], y[k]) for k in x), name
        else:
            assert torch.equal(x, y), name


def test_non_finite_loss_stops_the_run_with_a_clear_message(world):
    cfg, _ = world
    run_prepare(cfg, log=lambda s: None)
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(cfg.model_path, dtype=torch.float32)
    info = json.loads((cfg.model_path / "zubaan_extension.json").read_text())
    pm, handle = prepare_for_training(model, info["first_new_id"], cfg)
    with torch.no_grad():
        handle.new_rows.fill_(float("nan"))
    sampler = BlockSampler(load_blocks(cfg.data_dir)[0], None, 0.0, 0)
    with pytest.raises(TrainingDiverged, match="fp32"):
        train(pm, handle, sampler, cfg, log=lambda s: None)


def test_hub_hooks_are_called_and_a_remote_checkpoint_resumes(world):
    cfg, tmp = world
    run_prepare(cfg, log=lambda s: None)
    pushed = []
    first_cfg = cfg.model_copy(update={"output_dir": tmp / "one"})
    run_adapt(
        first_cfg,
        log=lambda s: None,
        max_steps=4,
        uploader=lambda path, step: pushed.append((path.name, step)),
    )
    assert pushed == [("ckpt-0000002", 2), ("ckpt-0000004", 4)]

    remote = latest_checkpoint(first_cfg.output_dir / "checkpoints")
    second_cfg = cfg.model_copy(update={"output_dir": tmp / "two"})

    def downloader(directory):
        dest = directory / remote.name
        shutil.copytree(remote, dest)
        return dest

    summary = run_adapt(second_cfg, log=lambda s: None, max_steps=2, downloader=downloader)
    assert summary["resumed_from_step"] == 4 and summary["final_step"] == 6


def test_tokenizer_or_block_size_mismatch_is_refused(world):
    cfg, tmp = world
    run_prepare(cfg, log=lambda s: None)
    with pytest.raises(ValueError, match="block_size"):
        run_adapt(cfg.model_copy(update={"block_size": 32}), log=lambda s: None)
    tok = cfg.model_path / "tokenizer.json"
    data = json.loads(tok.read_text(encoding="utf-8"))
    data["model"]["merges"] = data["model"]["merges"][:-1]
    tok.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="different tokenizer"):
        run_adapt(cfg, log=lambda s: None)


def test_adapt_cli_prepare_and_run(world):
    cfg, tmp = world
    path = tmp / "cfg.yaml"
    path.write_text(yaml.safe_dump({"adapt": json.loads(cfg.model_dump_json())}), encoding="utf-8")
    result = runner.invoke(cli.app, ["adapt", "prepare", "--config", str(path)])
    assert result.exit_code == 0, result.output
    assert "Saved blocks" in result.output
    result = runner.invoke(
        cli.app, ["adapt", "run", "--config", str(path), "--steps", "3", "--json"]
    )
    assert result.exit_code == 0, result.output
    out = json.loads(result.stdout)
    assert out["final_step"] == 3 and out["finished"] is False


def test_example_config_loads_and_matches_the_decisions():
    from zubaan.config import load_config

    cfg = load_config("configs/adapt-qwen3-1.7b.yaml").adapt
    assert (cfg.lora_rank, cfg.lora_alpha, cfg.precision) == (64, 128, "fp16")
    assert cfg.english_replay_ratio == 0.1 and cfg.block_size == 1024
    assert cfg.hub_repo is None  # never pushes unless the user sets a private repo


# merge and save (chunk 6) ------------------------------------------------------------------------


def test_merge_saves_a_verified_plain_model_with_the_old_rows_untouched(world, tiny_base_json):
    from transformers import AutoModelForCausalLM

    from zubaan.adapt.finalize import merge_and_save

    cfg, tmp = world
    cfg = cfg.model_copy(update={"merged_dtype": "float32"})
    run_prepare(cfg, log=lambda s: None)
    run_adapt(cfg, log=lambda s: None, max_steps=6)  # into phase B, so LoRA has learned something
    out = tmp / "merged"
    summary = merge_and_save(
        cfg, out_dir=out, base_tokenizer_json=tiny_base_json, log=lambda s: None
    )
    assert summary["step"] == 6 and summary["metadata"]["trained"] is True
    assert summary["metadata"]["stage3"]["merge_max_logit_diff"] < 1e-3
    meta = json.loads((out / "zubaan_extension.json").read_text())
    assert meta["trained"] is True and meta["first_new_id"] == summary["metadata"]["first_new_id"]
    for name in ("config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json"):
        assert (out / name).exists(), name

    merged = AutoModelForCausalLM.from_pretrained(out, dtype=torch.float32)
    original = AutoModelForCausalLM.from_pretrained(cfg.model_path, dtype=torch.float32)
    n_old = meta["first_new_id"]
    w_new, w_old = merged.get_input_embeddings().weight, original.get_input_embeddings().weight
    assert w_new.shape == w_old.shape
    assert torch.equal(w_new[:n_old], w_old[:n_old])  # old rows are the extended model's rows
    assert not torch.equal(w_new[n_old:], w_old[n_old:])  # the new rows trained
    assert merged.get_output_embeddings().weight.data_ptr() == w_new.data_ptr()  # still tied
    assert not any("lora_" in n for n, _ in merged.named_parameters())  # a plain model


def test_merge_needs_a_checkpoint_and_the_cli_runs_it(world, tiny_base_json, monkeypatch):
    from zubaan.adapt.finalize import merge_and_save

    cfg, tmp = world
    cfg = cfg.model_copy(update={"merged_dtype": "float32"})
    with pytest.raises(FileNotFoundError, match="no checkpoint"):
        merge_and_save(cfg, base_tokenizer_json=tiny_base_json, log=lambda s: None)

    run_prepare(cfg, log=lambda s: None)
    run_adapt(cfg, log=lambda s: None, max_steps=4)
    monkeypatch.setattr("zubaan.adapt.finalize.load_tokenizer_json", lambda name: tiny_base_json)
    path = tmp / "cfg.yaml"
    path.write_text(yaml.safe_dump({"adapt": json.loads(cfg.model_dump_json())}), encoding="utf-8")
    result = runner.invoke(cli.app, ["adapt", "merge", "--config", str(path), "--json"])
    assert result.exit_code == 0, result.output
    out = json.loads(result.stdout)
    assert out["step"] == 4 and (tmp / "out" / "model" / "model.safetensors").exists()


def test_resume_refuses_a_checkpoint_from_a_different_schedule(world):
    cfg, _ = world
    run_prepare(cfg, log=lambda s: None)
    run_adapt(cfg, log=lambda s: None, max_steps=4)
    changed = cfg.model_copy(update={"phase_b_steps": 9, "lr_lora": 5e-4})
    with pytest.raises(ValueError, match="different settings.*phase_b_steps.*lr_lora"):
        run_adapt(changed, log=lambda s: None)
    # Fields that do not affect the run (logging, checkpoint cadence) may change.
    again = cfg.model_copy(update={"log_every": 5, "checkpoint_every": 3})
    assert run_adapt(again, log=lambda s: None, max_steps=2)["resumed_from_step"] == 4


def test_milestone_path_and_hub_uploader_keep_permanent_copies(monkeypatch):
    import sys
    import types

    from zubaan.adapt.pipeline import make_hub_uploader, milestone_path

    assert milestone_path(200) == "milestones/step-0000200"
    calls = []

    class FakeApi:
        def create_repo(self, *a, **k):
            pass

        def upload_folder(self, **k):
            calls.append(k["path_in_repo"])
            if k.get("fail"):
                raise RuntimeError

    monkeypatch.setitem(sys.modules, "huggingface_hub", types.SimpleNamespace(HfApi=FakeApi))
    up = make_hub_uploader("me/r", True, 200)
    up(Path("x"), 150)
    up(Path("x"), 200)
    assert calls == ["last", "last", "milestones/step-0000200"]


def test_a_failed_hub_upload_warns_and_does_not_stop_training(monkeypatch):
    import sys
    import types

    from zubaan.adapt.pipeline import make_hub_uploader

    class BrokenApi:
        def create_repo(self, *a, **k):
            pass

        def upload_folder(self, **k):
            raise ConnectionError("offline")

    monkeypatch.setitem(sys.modules, "huggingface_hub", types.SimpleNamespace(HfApi=BrokenApi))
    messages = []
    make_hub_uploader("me/r", True, 0, messages.append)(Path("x"), 50)
    assert len(messages) == 1 and "WARNING" in messages[0] and "step 50" in messages[0]


# control run (decision 6): the base model and its own tokenizer, LoRA only ------------------------


@pytest.fixture
def control(world, tiny_base_json, tiny_qwen3):
    """The world's base folder with weights added, and a control config on it."""
    cfg, tmp = world
    base = tmp / "base"
    rows = inspect_tokenizer_json(tiny_base_json).total_vocab_size + 8
    tiny_qwen3(rows, seed=2).save_pretrained(base)
    ctl = cfg.model_copy(
        update={
            "model_path": base,
            "control": True,
            "phase_a_steps": 0,
            "data_dir": tmp / "blocks_control",
            "output_dir": tmp / "control",
        }
    )
    return ctl, cfg, tmp


def test_control_prepare_uses_the_base_tokenizer(control):
    import hashlib

    ctl, cfg, _ = control
    main = run_prepare(cfg, log=lambda s: None)
    meta = run_prepare(ctl, log=lambda s: None)
    assert meta["control"] is True and meta["first_new_id"] is None
    tok = (ctl.model_path / "tokenizer.json").read_bytes()
    assert meta["tokenizer_sha256"] == hashlib.sha256(tok).hexdigest()
    # The same Urdu text needs more tokens without the new merges: that is the cost being matched.
    assert meta["urdu"]["n_tokens"] > main["urdu"]["n_tokens"]
    assert main["control"] is False


def test_control_run_trains_lora_only_and_saves_no_rows(control):
    ctl, _, _ = control
    run_prepare(ctl, log=lambda s: None)
    summary = run_adapt(ctl, log=lambda s: None)
    assert summary["finished"] and summary["final_step"] == 6 and summary["control"] is True
    assert summary["first_new_id"] is None
    rows = [json.loads(x) for x in (ctl.output_dir / "train_log.jsonl").read_text().splitlines()]
    steps = [r for r in rows if "loss" in r]
    assert {r["phase"] for r in steps} == {"B"}
    assert all(r["lr_new"] is None and r["lr_lora"] > 0 for r in steps)
    ckpt = latest_checkpoint(ctl.output_dir / "checkpoints")
    assert (ckpt / "lora.pt").exists() and not (ckpt / "new_rows.pt").exists()


def test_control_and_main_folders_and_settings_cannot_be_swapped(control):
    ctl, cfg, tmp = control
    with pytest.raises(ValueError, match="phase_a_steps: 0"):
        run_prepare(ctl.model_copy(update={"phase_a_steps": 2}), log=lambda s: None)
    with pytest.raises(ValueError, match="extended model"):
        run_prepare(ctl.model_copy(update={"model_path": cfg.model_path}), log=lambda s: None)
    with pytest.raises(FileNotFoundError, match="control: true"):
        run_prepare(cfg.model_copy(update={"model_path": ctl.model_path}), log=lambda s: None)


def test_control_never_resumes_a_main_run_checkpoint(control):
    ctl, cfg, tmp = control
    run_prepare(cfg, log=lambda s: None)
    run_prepare(ctl, log=lambda s: None)
    shared = tmp / "shared"
    run_adapt(cfg.model_copy(update={"output_dir": shared}), log=lambda s: None, max_steps=2)
    with pytest.raises(ValueError, match="control"):
        run_adapt(ctl.model_copy(update={"output_dir": shared}), log=lambda s: None)


def test_stop_at_step_stops_there_and_a_later_run_finishes(control):
    ctl, _, _ = control
    run_prepare(ctl, log=lambda s: None)
    first = run_adapt(ctl, log=lambda s: None, stop_at_step=4)
    assert first["final_step"] == 4 and not first["finished"]
    again = run_adapt(ctl, log=lambda s: None, stop_at_step=4)  # already there: no step taken
    assert again["resumed_from_step"] == 4 and again["final_step"] == 4
    rest = run_adapt(ctl, log=lambda s: None)
    assert rest["resumed_from_step"] == 4 and rest["finished"]
    with pytest.raises(ValueError, match="not both"):
        run_adapt(ctl, log=lambda s: None, max_steps=1, stop_at_step=2)


def test_control_merge_keeps_every_row_and_the_tokenizer(control):
    from transformers import AutoModelForCausalLM

    from zubaan.adapt.finalize import CONTROL_METADATA_FILE, merge_and_save

    ctl, _, tmp = control
    ctl = ctl.model_copy(update={"merged_dtype": "float32"})
    run_prepare(ctl, log=lambda s: None)
    run_adapt(ctl, log=lambda s: None)
    out = tmp / "control_merged"
    summary = merge_and_save(ctl, out_dir=out, log=lambda s: None)
    assert summary["step"] == 6 and summary["metadata"]["control"] is True
    assert summary["metadata"]["stage3"]["merge_max_logit_diff"] < 1e-3
    assert (out / CONTROL_METADATA_FILE).exists()
    assert not (out / "zubaan_extension.json").exists()  # never mistaken for an extended model
    assert (out / "tokenizer.json").read_bytes() == (ctl.model_path / "tokenizer.json").read_bytes()

    merged = AutoModelForCausalLM.from_pretrained(out, dtype=torch.float32)
    base = AutoModelForCausalLM.from_pretrained(ctl.model_path, dtype=torch.float32)
    assert torch.equal(merged.get_input_embeddings().weight, base.get_input_embeddings().weight)
    assert merged.get_output_embeddings().weight.data_ptr() == (
        merged.get_input_embeddings().weight.data_ptr()
    )
    assert not any("lora_" in n for n, _ in merged.named_parameters())
    x = torch.arange(3, 19)[None]
    with torch.no_grad():
        assert not torch.allclose(merged(input_ids=x).logits, base(input_ids=x).logits)


def test_example_control_config_loads():
    from zubaan.config import load_config

    cfg = load_config("configs/adapt-control-qwen3-1.7b.yaml").adapt
    assert cfg.control and cfg.phase_a_steps == 0 and cfg.phase_b_steps == 800
    assert cfg.micro_batch_size * cfg.grad_accum == 32 and cfg.hub_repo is None
