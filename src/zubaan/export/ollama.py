"""Generate an Ollama ``Modelfile`` for an exported GGUF.

The template is Qwen's ChatML format (``<|im_start|>role ... <|im_end|>``). The stop tokens
must be real tokens of the extended vocabulary at their original ids, so they are checked
against the tokenizer before the file is written: a stop string that is not a single token
would never end generation.

Qwen3 can start a reply with a ``<think>`` block. By default this Modelfile does not suppress it.
With ``no_think`` the template starts every reply with an empty ``<think>\\n\\n</think>\\n\\n``
block, which is what Qwen3's own chat template does with ``enable_thinking=False``. Use it for
the stage 3 adapted model: with thinking allowed it ended its turn inside ``<think>`` without an
answer (vLLM check, notebook 10).
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

DEFAULT_STOP_TOKENS = ("<|im_end|>", "<|im_start|>")

CHATML_TEMPLATE = """{{- range $i, $_ := .Messages }}
{{- $last := eq (len (slice $.Messages $i)) 1 -}}
<|im_start|>{{ .Role }}
{{ .Content }}{{ if not $last }}<|im_end|>
{{ end }}
{{- if and (ne .Role "assistant") $last }}<|im_end|>
<|im_start|>assistant
{{ end }}
{{- end }}"""

# The assistant prefix Qwen3's template writes when thinking is turned off.
EMPTY_THINK = "<think>\n\n</think>\n\n"


def chatml_template(no_think: bool = False) -> str:
    """The ChatML template, optionally opening every reply with an empty think block."""
    if not no_think:
        return CHATML_TEMPLATE
    prefix = "<|im_start|>assistant\n"
    if CHATML_TEMPLATE.count(prefix) != 1:  # pragma: no cover - guards a future template edit
        raise ValueError("the ChatML template must contain the assistant prefix exactly once")
    return CHATML_TEMPLATE.replace(prefix, prefix + EMPTY_THINK)


def check_stop_tokens(tokenizer_json: dict, base: dict, stops: Sequence[str]) -> dict[str, int]:
    """Return ``{token: id}``; raise unless each is a token of the extended vocabulary.

    Also raises if a stop token has a different id in the base tokenizer, since that would
    mean the extension moved a special token.
    """
    ids: dict[str, int] = {}
    ext_tokens = {t: i for t, i in tokenizer_json["model"]["vocab"].items() if t in stops}
    ext_tokens.update({a["content"]: a["id"] for a in tokenizer_json.get("added_tokens", [])})
    base_tokens = {a["content"]: a["id"] for a in base.get("added_tokens", [])}
    for s in stops:
        if s not in ext_tokens:
            raise ValueError(f"stop token {s!r} is not in the extended vocabulary")
        if s in base_tokens and base_tokens[s] != ext_tokens[s]:
            raise ValueError(f"stop token {s!r} moved from id {base_tokens[s]} to {ext_tokens[s]}")
        ids[s] = ext_tokens[s]
    return ids


def modelfile_text(
    gguf_name: str,
    stop_tokens: Sequence[str] = DEFAULT_STOP_TOKENS,
    num_ctx: int = 4096,
    temperature: float = 0.7,
    top_p: float = 0.8,
    top_k: int = 20,
    system: str | None = None,
    no_think: bool = False,
) -> str:
    """Text of a Modelfile for the GGUF file ``gguf_name`` (a path relative to the Modelfile)."""
    if not gguf_name.endswith(".gguf"):
        raise ValueError("gguf_name must end with .gguf")
    if num_ctx < 256:
        raise ValueError("num_ctx must be at least 256")
    lines = [f"FROM ./{gguf_name}", "", f'TEMPLATE """{chatml_template(no_think)}"""', ""]
    if system:
        if '"""' in system:
            raise ValueError("system prompt must not contain triple quotes")
        lines += [f'SYSTEM """{system}"""', ""]
    lines += [f'PARAMETER stop "{s}"' for s in stop_tokens]
    lines += [
        f"PARAMETER num_ctx {num_ctx}",
        f"PARAMETER temperature {temperature}",
        f"PARAMETER top_p {top_p}",
        f"PARAMETER top_k {top_k}",
    ]
    return "\n".join(lines) + "\n"


def write_modelfile(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")
    return path
