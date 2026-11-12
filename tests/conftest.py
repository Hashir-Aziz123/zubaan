from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent / "fixtures"


class CharTokenizer:
    """Fake tokenizer: one token per non-space character. No downloads needed."""

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        ids = [ord(c) for c in text if not c.isspace()]
        return ([0] + ids) if add_special_tokens else ids


@pytest.fixture
def char_tokenizer() -> CharTokenizer:
    return CharTokenizer()


@pytest.fixture
def fixtures_dir() -> Path:
    return FIXTURES
