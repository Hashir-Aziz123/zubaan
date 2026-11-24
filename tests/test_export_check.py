"""zubaan export check: generation samples and quantization cost, with a fake llama.cpp."""

import json
from pathlib import Path

import pytest
from conftest import make_llama_dir
from typer.testing import CliRunner

from zubaan import cli
from zubaan.config import ExportConfig
from zubaan.export.check import (
    PROMPTS,
    Prompt,
    chatml_prompt,
    completion_command,
    kld_base_command,
    kld_command,
    parse_eval_speed,
    parse_kld,
    parse_ppl,
    run_check,
    strip_think,
    urdu_letter_share,
    write_eval_text,
)
from zubaan.export.gguf import find_llama_cpp

# Summary lines printed by llama-perplexity b12477 --kl-divergence (Qwen3-0.6B extended,
# Q4_K_M against f16, 2 chunks of FLORES Urdu), trimmed.
KLD_OUTPUT = """
====== Perplexity statistics ======
Mean PPL(Q)                   : 3959.522593 ±  532.806832
Mean PPL(base)                : 3994.991944 ±  533.078712
Cor(ln(PPL(Q)), ln(PPL(base))):  99.52%
Mean ln(PPL(Q)/PPL(base))     :  -0.008918 ±   0.013192
Mean PPL(Q)/PPL(base)         :   0.991122 ±   0.013075
Mean PPL(Q)-PPL(base)         : -35.469351 ±  52.278323

====== KL divergence statistics ======
Mean    KLD:   0.020981 ±   0.001223
Maximum KLD:   0.371473

====== Token probability statistics ======
Mean    Δp:  0.157 ± 0.143 %
RMS Δp    :  3.228 ± 0.598 %
Same top p: 73.529 ± 1.955 %
"""

TIMINGS = """
common_perf_print: prompt eval time =     177.74 ms /    14 tokens (   12.70 ms per token,    78.77 tokens per second)
common_perf_print:        eval time =    1402.83 ms /    29 runs   (   48.37 ms per token,    20.67 tokens per second)
"""  # noqa: E501


def test_chatml_prompt_no_think_adds_empty_block():
    p = chatml_prompt("سلام")
    assert p == "<|im_start|>user\nسلام<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    assert chatml_prompt("سلام", thinking=True).endswith("<|im_start|>assistant\n")
    with pytest.raises(ValueError):
        chatml_prompt("  ")


def test_strip_think():
    assert strip_think("<think>\nhmm\n</think>\n\nجواب") == "جواب"
    assert strip_think("<think>\nnever closed") == ""
    assert strip_think("plain") == "plain"


def test_urdu_letter_share_worked_example():
    # Letters: "اردو" (4 Arabic-block) + "ab" (2 Latin); digits, spaces and "۔" are not letters.
    # share = 4 / (4 + 2) = 0.6667
    assert urdu_letter_share("اردو ab 12۔") == pytest.approx(4 / 6)
    assert urdu_letter_share("123 ۔") is None


def test_parse_kld_and_ppl():
    k = parse_kld(KLD_OUTPUT)
    assert k["mean_kld"] == {"value": 0.020981, "uncertainty": 0.001223}
    assert k["same_top_p_percent"]["value"] == pytest.approx(73.529)
    assert k["ppl_ratio"]["value"] == pytest.approx(0.991122)
    assert k["ln_ppl_ratio"]["value"] == pytest.approx(-0.008918)
    assert k["rms_delta_p_percent"]["value"] == pytest.approx(3.228)
    assert parse_ppl("Final estimate: PPL = 3994.9966 +/- 533.07923") == {
        "value": 3994.9966,
        "uncertainty": 533.07923,
    }
    with pytest.raises(ValueError):
        parse_kld("nothing here")
    with pytest.raises(ValueError):
        parse_ppl("nothing here")


def test_parse_eval_speed_skips_prompt_eval():
    assert parse_eval_speed(TIMINGS) == pytest.approx(20.67)
    assert parse_eval_speed("no timings") is None


def _tools(tmp_path):
    root = make_llama_dir(tmp_path)
    for tool in ("llama-completion", "llama-perplexity"):
        (root / "bin" / f"{tool}.exe").write_text("", encoding="utf-8")
    return find_llama_cpp(root)


def test_commands(tmp_path):
    tools = _tools(tmp_path)
    cmd = completion_command(tools, Path("m.gguf"), "x", 40, threads=4)
    assert cmd[1:3] == ["-m", "m.gguf"] and cmd[cmd.index("-n") + 1] == "40"
    assert {"-no-cnv", "-sp", "--no-display-prompt"} <= set(cmd) and cmd[-2:] == ["-t", "4"]
    assert cmd[cmd.index("--temp") + 1] == "0"
    base = kld_base_command(tools, Path("f.gguf"), Path("t.txt"), Path("l.kld"), 512, 20)
    assert base[base.index("--kl-divergence-base") + 1] == "l.kld" and "-f" in base
    q = kld_command(tools, Path("q.gguf"), Path("l.kld"), 512, 20)
    assert q[-1] == "--kl-divergence" and "-f" not in q
    with pytest.raises(ValueError):
        completion_command(tools, Path("m.gguf"), "x", 0)
    with pytest.raises(ValueError):
        kld_command(tools, Path("q.gguf"), Path("l.kld"), 512, 0)


def test_write_eval_text_normalizes_urdu(tmp_path):
    out = tmp_path / "t.txt"
    # Arabic yeh (U+064A) maps to Farsi yeh (U+06CC); blank lines are dropped.
    n = write_eval_text(["علي  ", "", "دو"], out, urdu=True)
    assert out.read_text(encoding="utf-8") == "علی\nدو\n" and n == 7
    with pytest.raises(ValueError):
        write_eval_text(["", " "], out, urdu=False)


class FakeProc:
    def __init__(self, stdout="", stderr="", returncode=0):
        self.stdout, self.stderr, self.returncode = stdout, stderr, returncode


class FakeLlama:
    """Plays llama-completion and llama-perplexity; records commands."""

    def __init__(self):
        self.calls = []

    def __call__(self, cmd, **kwargs):
        self.calls.append(list(cmd))
        name = Path(cmd[0]).name
        if name.startswith("llama-completion"):
            prompt = cmd[cmd.index("-p") + 1]
            text = (
                "<think>\nسوچ\n</think>\n\nاسلام آباد"
                if prompt.endswith("assistant\n")
                else "اسلام آباد"
            )
            return FakeProc(text + " [end of text]\n", TIMINGS)
        if "--kl-divergence" in cmd:
            assert Path(cmd[cmd.index("--kl-divergence-base") + 1]).exists()
            return FakeProc(KLD_OUTPUT)
        Path(cmd[cmd.index("--kl-divergence-base") + 1]).write_bytes(b"logits")
        return FakeProc("Final estimate: PPL = 12.5 +/- 0.3\n")


def _export_folder(tmp_path, passed=True):
    out = tmp_path / "export"
    out.mkdir()
    for f in ("m-F16.gguf", "m-Q4_K_M.gguf", "m-Q8_0.gguf"):
        (out / f).write_bytes(b"gguf")
    summary = {
        "gguf": {
            "f16": "m-F16.gguf",
            "quantized": {"Q4_K_M": "m-Q4_K_M.gguf", "Q8_0": "m-Q8_0.gguf"},
            "parity": {"passed": passed},
        }
    }
    (out / "export_summary.json").write_text(json.dumps(summary), encoding="utf-8")
    urd, eng = tmp_path / "urd.txt", tmp_path / "eng.txt"
    urd.write_text("پاکستان ایک ملک ہے۔\n", encoding="utf-8")
    eng.write_text("Pakistan is a country.\n", encoding="utf-8")
    return out, urd, eng


def _cfg(tmp_path, out, urd, eng):
    root = _tools(tmp_path).root
    return ExportConfig(
        output_dir=out, llama_cpp_dir=root, check_urdu=urd, check_english=eng, threads=2
    )


def test_run_check_end_to_end(tmp_path):
    out, urd, eng = _export_folder(tmp_path)
    fake = FakeLlama()
    result = run_check(_cfg(tmp_path, out, urd, eng), log=lambda s: None, run=fake)

    assert set(result["samples"]) == {"F16", "Q4_K_M", "Q8_0"}
    samples = {s["prompt"]: s for s in result["samples"]["Q4_K_M"]}
    assert len(samples) == len(PROMPTS)
    think = samples["urdu_chat_thinking"]
    assert think["has_think_block"] and think["answer_urdu_letter_share"] == 1.0
    assert not samples["urdu_chat_capital"]["has_think_block"]
    assert samples["urdu_completion_capital"]["text"] == "اسلام آباد"
    assert samples["english_completion"]["tokens_per_s"] == pytest.approx(20.67)

    for lang in ("urdu", "english"):
        cost = result["quant_cost"][lang]
        assert cost["f16_ppl"]["value"] == 12.5
        assert set(cost["quants"]) == {"Q4_K_M", "Q8_0"}
    # One f16 base run and two comparisons per language; the logits file is removed afterwards.
    kld_calls = [c for c in fake.calls if Path(c[0]).name.startswith("llama-perplexity")]
    assert len(kld_calls) == 6
    assert not list(out.rglob("*.kld")) and not (out / "check_tmp").exists()
    saved = json.loads((out / "export_check.json").read_text(encoding="utf-8"))
    assert saved["quant_cost"]["urdu"]["quants"]["Q8_0"]["mean_kld"]["value"] == 0.020981


def test_run_check_refuses_without_parity(tmp_path):
    out, urd, eng = _export_folder(tmp_path, passed=False)
    with pytest.raises(ValueError, match="parity"):
        run_check(_cfg(tmp_path, out, urd, eng), log=lambda s: None, run=FakeLlama())


def test_run_check_needs_f16_kept(tmp_path):
    out, urd, eng = _export_folder(tmp_path)
    (out / "m-F16.gguf").unlink()
    with pytest.raises(FileNotFoundError, match="f16"):
        run_check(_cfg(tmp_path, out, urd, eng), log=lambda s: None, run=FakeLlama())


def test_cli_export_check(tmp_path, monkeypatch):
    import yaml

    out, urd, eng = _export_folder(tmp_path)
    cfg = _cfg(tmp_path, out, urd, eng)
    path = tmp_path / "c.yaml"
    path.write_text(yaml.safe_dump({"export": json.loads(cfg.model_dump_json())}), encoding="utf-8")
    import zubaan.export.check as check

    fake = FakeLlama()
    real = check.run_check
    monkeypatch.setattr(check, "run_check", lambda c, log: real(c, log=log, run=fake))
    r = CliRunner().invoke(cli.app, ["export", "check", "--config", str(path)])
    assert r.exit_code == 0, r.output
    assert "same top token" in r.output and (out / "export_check.json").exists()


def test_prompts_are_unique():
    assert len({p.name for p in PROMPTS}) == len(PROMPTS)
    assert all(isinstance(p, Prompt) for p in PROMPTS)
