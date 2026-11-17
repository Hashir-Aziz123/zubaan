from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent / "fixtures"


class CharTokenizer:
    """Fake tokenizer: one token per non-space character. No downloads needed."""

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        ids = [ord(c) for c in text if not c.isspace()]
        return ([0] + ids) if add_special_tokens else ids

    def decode(self, ids: list[int], skip_special_tokens: bool = False) -> str:
        return "".join(chr(i) for i in ids if i)


class SpaceTokenizer:
    """Fake lossless tokenizer: one token per character, spaces included."""

    unk_token_id = None

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        return [ord(c) for c in text]

    def decode(self, ids: list[int], skip_special_tokens: bool = False) -> str:
        return "".join(chr(i) for i in ids)

    def __len__(self) -> int:
        return 0x110000


@pytest.fixture
def char_tokenizer() -> CharTokenizer:
    return CharTokenizer()


@pytest.fixture
def space_tokenizer() -> SpaceTokenizer:
    return SpaceTokenizer()


@pytest.fixture(autouse=True)
def _offline_provenance(monkeypatch):
    """Keep default tests offline: never ask the Hub for model revisions."""
    monkeypatch.setattr("zubaan.provenance.model_revision", lambda name: None)


@pytest.fixture
def fixtures_dir() -> Path:
    return FIXTURES


QWEN_STYLE_REGEX = (
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*"
    r"|\s*[\r\n]+|\s+(?!\S)|\s+"
)


def make_tiny_base_tokenizer_json(vocab_size: int = 400) -> dict:
    """A small byte-level BPE trained on English only, shaped like Qwen3's tokenizer.json.

    Same pre-tokenizer (regex split then byte-level), NFC normalizer, byte-level decoder, and
    a few special tokens as ``added_tokens``. Urdu is not in its training text, so Urdu text
    falls back to single bytes, which is the situation extension is meant to fix.
    """
    import json

    from tokenizers import Regex, Tokenizer, decoders, models, normalizers, pre_tokenizers
    from tokenizers import trainers as tr

    tok = Tokenizer(models.BPE())
    tok.normalizer = normalizers.NFC()
    tok.pre_tokenizer = pre_tokenizers.Sequence(
        [
            pre_tokenizers.Split(Regex(QWEN_STYLE_REGEX), behavior="isolated", invert=False),
            pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
        ]
    )
    tok.decoder = decoders.ByteLevel()
    trainer = tr.BpeTrainer(
        vocab_size=vocab_size,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        special_tokens=["<|endoftext|>", "<|im_start|>"],
        show_progress=False,
    )
    path = FIXTURES / "english_train_small.txt"
    tok.train_from_iterator(path.read_text(encoding="utf-8").splitlines() * 3, trainer)
    return json.loads(tok.to_str())


@pytest.fixture(scope="session")
def tiny_base_json() -> dict:
    pytest.importorskip("tokenizers")
    return make_tiny_base_tokenizer_json()


@pytest.fixture
def urdu_lines() -> list[str]:
    path = FIXTURES / "urdu_train_small.txt"
    return path.read_text(encoding="utf-8").splitlines()


@pytest.fixture
def english_lines() -> list[str]:
    path = FIXTURES / "english_train_small.txt"
    return path.read_text(encoding="utf-8").splitlines()


def make_tiny_qwen3(rows: int, tie: bool = True, seed: int = 0, dtype: str = "float32"):
    """A random 2-layer Qwen3-style model with ``rows`` embedding rows. No download."""
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    cfg = transformers.Qwen3Config(
        vocab_size=rows,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        max_position_embeddings=256,
        tie_word_embeddings=tie,
    )
    torch.manual_seed(seed)
    model = transformers.AutoModelForCausalLM.from_config(cfg)
    return model.to(getattr(torch, dtype)).eval()


@pytest.fixture
def tiny_qwen3():
    return make_tiny_qwen3


# Fake llama.cpp pieces shared by the export tests ------------------------------
H_BASE = "a" * 64
H_OTHER = "b" * 64
H_EXT = "c" * 64

FAKE_SOURCE = f'''
class Model:
    def get_vocab_base_pre(self, tokenizer) -> str:
        chktxt = 'abc\\n \\'quoted\\' ق'
        chktok = tokenizer.encode(chktxt)
        res = None
        if chkhsh == "{H_BASE}":
            # ref: https://example.com/base
            res = "qwen2"
        if chkhsh == "{H_OTHER}":
            # ref: https://example.com/other
            # second comment line
            res = "llama-bpe"

        if res is None:
            raise NotImplementedError("BPE pre-tokenizer was not recognized")
        return res
'''


def make_llama_dir(tmp_path: Path, source: str = FAKE_SOURCE) -> Path:
    """A fake llama.cpp folder: converter source and empty tool files."""
    root = tmp_path / "llama.cpp"
    (root / "bin").mkdir(parents=True)
    (root / "convert_hf_to_gguf.py").write_text(source, encoding="utf-8")
    for tool in ("llama-quantize", "llama-imatrix", "llama-tokenize"):
        (root / "bin" / f"{tool}.exe").write_text("", encoding="utf-8")
    return root
