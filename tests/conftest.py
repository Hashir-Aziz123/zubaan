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
