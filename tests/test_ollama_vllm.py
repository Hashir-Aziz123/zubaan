import copy

import pytest

from zubaan.export.ollama import check_stop_tokens, modelfile_text, write_modelfile
from zubaan.export.vllm import (
    check_no_bfloat16,
    config_yaml,
    serve_command,
    vllm_options,
    write_vllm_files,
)


@pytest.fixture
def qwen_like():
    """A minimal tokenizer.json with the ChatML special tokens at their Qwen3 ids."""
    return {
        "added_tokens": [
            {"id": 151644, "content": "<|im_start|>", "special": True},
            {"id": 151645, "content": "<|im_end|>", "special": True},
        ],
        "model": {"vocab": {"<|im_start|>": 151644, "<|im_end|>": 151645, "a": 0}},
    }


def test_modelfile_snapshot():
    text = modelfile_text("m-Q4_K_M.gguf", ("<|im_end|>", "<|im_start|>"), num_ctx=2048)
    assert text.startswith("FROM ./m-Q4_K_M.gguf\n")
    assert 'TEMPLATE """{{- range $i, $_ := .Messages }}' in text
    assert "<|im_start|>{{ .Role }}" in text and "<|im_start|>assistant" in text
    assert 'PARAMETER stop "<|im_end|>"\nPARAMETER stop "<|im_start|>"\n' in text
    assert "PARAMETER num_ctx 2048\n" in text
    assert "PARAMETER temperature 0.7\nPARAMETER top_p 0.8\nPARAMETER top_k 20\n" in text
    assert text.endswith("\n") and "SYSTEM" not in text


def test_modelfile_system_prompt_and_validation():
    text = modelfile_text("m.gguf", system="تم ایک مددگار معاون ہو۔")
    assert 'SYSTEM """تم ایک مددگار معاون ہو۔"""' in text
    with pytest.raises(ValueError, match="triple quotes"):
        modelfile_text("m.gguf", system='bad """ system')
    with pytest.raises(ValueError, match=".gguf"):
        modelfile_text("model.bin")
    with pytest.raises(ValueError, match="num_ctx"):
        modelfile_text("m.gguf", num_ctx=10)


def test_stop_tokens_must_exist_and_keep_their_ids(qwen_like):
    ext = copy.deepcopy(qwen_like)
    assert check_stop_tokens(ext, qwen_like, ("<|im_end|>", "<|im_start|>")) == {
        "<|im_end|>": 151645,
        "<|im_start|>": 151644,
    }
    with pytest.raises(ValueError, match="not in the extended vocabulary"):
        check_stop_tokens(ext, qwen_like, ("<|nope|>",))
    ext["added_tokens"][1]["id"] = 151745  # the id shift the merge code must prevent
    with pytest.raises(ValueError, match="moved from id 151645 to 151745"):
        check_stop_tokens(ext, qwen_like, ("<|im_end|>",))


def test_write_modelfile(tmp_path):
    path = write_modelfile(tmp_path / "sub" / "Modelfile", "FROM ./x.gguf\n")
    assert path.read_bytes() == b"FROM ./x.gguf\n"  # LF line endings


def test_vllm_options_use_half_precision():
    opts = vllm_options("qwen3-urdu", 2048)
    assert opts["dtype"] == "half" and opts["max-model-len"] == 2048
    yaml_text = config_yaml(opts)
    assert "dtype: half\n" in yaml_text and "served-model-name: qwen3-urdu\n" in yaml_text
    assert "bfloat16" not in yaml_text.lower()


def test_vllm_validation():
    with pytest.raises(ValueError, match="max_model_len"):
        vllm_options("n", 10)
    with pytest.raises(ValueError, match="gpu_memory_utilization"):
        vllm_options("n", gpu_memory_utilization=1.5)
    with pytest.raises(ValueError, match="bfloat16"):
        check_no_bfloat16("dtype: bfloat16")
    with pytest.raises(ValueError, match="bfloat16"):
        check_no_bfloat16("--dtype bf16")


def test_serve_command_quotes_paths():
    cmd = serve_command("/kaggle/working/my model", "vllm.yaml")
    assert cmd == (
        "vllm serve '/kaggle/working/my model' --tokenizer '/kaggle/working/my model' "
        "--config vllm.yaml"
    )


def test_write_vllm_files(tmp_path):
    paths = write_vllm_files(tmp_path, "outputs/model", "qwen3-urdu")
    assert paths["config"].read_text(encoding="utf-8").startswith("dtype: half\n")
    script = paths["script"].read_text(encoding="utf-8")
    assert script.startswith("#!/bin/sh\n") and "--tokenizer outputs/model" in script
    assert b"\r" not in paths["script"].read_bytes()
