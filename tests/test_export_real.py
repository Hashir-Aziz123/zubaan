"""Real GGUF conversion and tokenizer parity. Needs llama.cpp in tools/ and a built model.

uv run zubaan extend run --config configs/export-qwen3-0.6b.yaml
uv run pytest -m slow tests/test_export_real.py
"""

from pathlib import Path

import pytest

from zubaan.config import ExportConfig
from zubaan.export.gguf import LLAMA_CPP_TAG
from zubaan.export.pipeline import run_export

ROOT = Path(__file__).parent.parent
LLAMA = ROOT / "tools" / f"llama.cpp-{LLAMA_CPP_TAG}"
MODEL = ROOT / "outputs" / "qwen3-0.6b" / "model"


@pytest.mark.slow
def test_real_gguf_conversion_has_tokenizer_parity(tmp_path):
    if not (LLAMA / "convert_hf_to_gguf.py").exists() or not (MODEL / "model.safetensors").exists():
        pytest.skip("needs llama.cpp in tools/ and a model built by 'zubaan extend run'")
    pytest.importorskip("transformers")
    cfg = ExportConfig(
        model_dir=MODEL,
        base_model="Qwen/Qwen3-0.6B",
        output_dir=tmp_path,
        name="test",
        llama_cpp_dir=LLAMA,
        quants=[],  # parity is what this test proves; quantizing is slow
        max_parity_texts=100,
    )
    summary = run_export(cfg, ("gguf",), log=print)
    parity = summary["gguf"]["parity"]
    assert parity["passed"], parity["mismatches"]
    assert parity["n_texts"] > 100  # Urdu and English lines plus the stress set
