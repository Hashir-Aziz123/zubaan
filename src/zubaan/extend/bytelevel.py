"""The GPT-2 byte-to-character table used by byte-level BPE vocabularies.

Byte-level tokenizers (Qwen, GPT-2, Llama 3) store every token as a string in which each
byte of the UTF-8 text is shown as one printable character, for example the space byte
0x20 is written as ``Ġ``. These helpers convert between that display form and real bytes,
which is how we can tell what text a token stands for without loading a model.
"""

from __future__ import annotations

from functools import lru_cache


@lru_cache(maxsize=1)
def bytes_to_unicode() -> dict[int, str]:
    """Map each byte value 0..255 to the printable character GPT-2 style vocabularies use.

    Printable ASCII and Latin-1 bytes map to themselves; the rest are shifted to
    U+0100 and up so no token contains a control character or a space.
    """
    keep = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    chars = keep[:]
    shift = 0
    for b in range(256):
        if b not in keep:
            keep.append(b)
            chars.append(256 + shift)
            shift += 1
    return {b: chr(c) for b, c in zip(keep, chars, strict=True)}


@lru_cache(maxsize=1)
def unicode_to_bytes() -> dict[str, int]:
    """Inverse of :func:`bytes_to_unicode`."""
    return {c: b for b, c in bytes_to_unicode().items()}


def token_to_bytes(token: str) -> bytes:
    """The UTF-8 bytes a byte-level token string stands for.

    Raises ValueError if the token contains a character outside the byte table (for example a
    literal space, which byte-level tokens write as U+0120), since it has no byte meaning.
    Special tokens such as ``<|endoftext|>`` happen to be printable ASCII and convert fine.
    """
    table = unicode_to_bytes()
    try:
        return bytes(table[c] for c in token)
    except KeyError as e:
        raise ValueError(f"token {token!r} is not a byte-level token") from e


def bytes_to_token(data: bytes) -> str:
    """Display form of a byte string, the inverse of :func:`token_to_bytes`."""
    table = bytes_to_unicode()
    return "".join(table[b] for b in data)
