"""Resize a model's embeddings for an extended tokenizer and initialize the new rows.

The extended tokenizer adds ``N`` tokens with ids ``first_new_id .. first_new_id + N - 1``.
The model needs that many more rows in its input embedding matrix (and in the output head
when the two are not tied). The old rows must stay exactly as they are, so everything the
model could already do on text without new tokens is untouched.

Initialization methods for the new rows (none of them is trained yet):

* ``mean``: each new token starts as the mean of the embeddings of the base tokens it was
  built from. A new token "means" what its pieces meant together, so the model sees a familiar
  vector instead of noise.
* ``random``: a control. Rows are drawn from a normal distribution with the per-dimension mean
  and standard deviation of the old rows, so the scale is right but the content is arbitrary.
* ``focus``: a FOCUS-style weighted mean (see ``focus.py``).

Torch is imported inside the functions that need it, so the module imports without it.
"""

from __future__ import annotations

from collections.abc import Sequence

from zubaan.extend.train_bpe import Merge

INIT_METHODS = ("mean", "random", "focus")


def merge_pieces(base_vocab: dict[str, int], merges: Sequence[Merge]) -> list[list[int]]:
    """Base-tokenizer ids that each new token is made of, in order, one list per merge.

    A merge joins two tokens that are either base tokens or earlier new tokens, so the pieces
    of a new token are the pieces of its left part followed by the pieces of its right part,
    down to base tokens. Example: a new token built as ("A", "B") then ("AB", "C") has pieces
    [id(A), id(B), id(C)].
    """
    expanded: dict[str, list[int]] = {}

    def pieces_of(token: str) -> list[int]:
        if token in base_vocab:
            return [base_vocab[token]]
        try:
            return expanded[token]
        except KeyError:
            raise ValueError(
                f"token {token!r} is neither a base token nor an earlier merge"
            ) from None

    out = []
    for m in merges:
        p = pieces_of(m.left) + pieces_of(m.right)
        expanded[m.token] = p
        out.append(p)
    return out


def mean_of_pieces(weight, pieces: Sequence[Sequence[int]]):
    """Mean of the rows of ``weight`` listed in each entry of ``pieces``.

    mean_i = (1 / |pieces_i|) x sum over j in pieces_i of weight[j], computed in float32.
    Returns a float32 tensor of shape (len(pieces), hidden).
    """
    import torch

    if not len(pieces):
        return torch.zeros((0, weight.shape[1]), dtype=torch.float32)
    rows = []
    for p in pieces:
        if not len(p):
            raise ValueError("a new token has no pieces")
        rows.append(weight[list(p)].float().mean(dim=0))
    return torch.stack(rows)


def random_rows(weight, n: int, seed: int = 0):
    """``n`` rows drawn from a normal with the per-dimension mean and std of ``weight``."""
    import torch

    w = weight.float()
    gen = torch.Generator().manual_seed(seed)
    # Noise is drawn on the CPU so the same seed gives the same rows on any device.
    noise = torch.randn((n, w.shape[1]), generator=gen).to(w.device)
    return w.mean(dim=0) + noise * w.std(dim=0)


def resize_and_init(
    model,
    pieces: Sequence[Sequence[int]],
    first_new_id: int,
    method: str = "mean",
    seed: int = 0,
    focus_rows=None,
    focus_head_rows=None,
) -> dict:
    """Resize ``model`` in place to ``first_new_id + len(pieces)`` rows and fill the new rows.

    ``first_new_id`` is the first id used by neither the old vocabulary nor an added token
    (151,669 for Qwen3), which can be lower than the padded matrix size (151,936). Rows below
    it are kept bit for bit. For ``focus`` the caller passes the precomputed rows.
    Returns a small dict describing what was done.
    """
    import torch

    if method not in INIT_METHODS:
        raise ValueError(f"method must be one of {INIT_METHODS}")
    n_new = len(pieces)
    emb = model.get_input_embeddings()
    head = model.get_output_embeddings()
    tied = head is None or head.weight.data_ptr() == emb.weight.data_ptr()
    old_rows = emb.weight.shape[0]
    if first_new_id > old_rows:
        raise ValueError(f"first_new_id {first_new_id} is above the {old_rows} embedding rows")
    new_total = first_new_id + n_new

    with torch.no_grad():
        in_old = emb.weight[:first_new_id].detach().clone()
        out_old = None if tied else head.weight[:first_new_id].detach().clone()
        if method == "mean":
            in_new = mean_of_pieces(in_old, pieces)
            out_new = None if tied else mean_of_pieces(out_old, pieces)
        elif method == "random":
            in_new = random_rows(in_old, n_new, seed)
            out_new = None if tied else random_rows(out_old, n_new, seed + 1)
        else:
            if focus_rows is None:
                raise ValueError("method 'focus' needs focus_rows (see extend.focus)")
            in_new = focus_rows
            out_new = None if tied else focus_head_rows
            if not tied and out_new is None:
                raise ValueError("an untied model also needs focus_head_rows")
        if in_new.shape != (n_new, in_old.shape[1]):
            raise ValueError(f"new rows have shape {tuple(in_new.shape)}, expected {n_new} rows")

        model.resize_token_embeddings(new_total, mean_resizing=False)
        emb = model.get_input_embeddings()
        dtype = emb.weight.dtype
        emb.weight[:first_new_id] = in_old.to(dtype)
        emb.weight[first_new_id:new_total] = in_new.to(dtype)
        if not tied:
            head = model.get_output_embeddings()
            head.weight[:first_new_id] = out_old.to(head.weight.dtype)
            head.weight[first_new_id:new_total] = out_new.to(head.weight.dtype)
        model.config.vocab_size = new_total
    return {
        "method": method,
        "n_new": n_new,
        "first_new_id": first_new_id,
        "embedding_rows_before": old_rows,
        "embedding_rows_after": new_total,
        "tied": tied,
    }
