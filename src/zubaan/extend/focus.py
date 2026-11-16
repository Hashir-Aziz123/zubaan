"""A FOCUS-style initialization for new token embeddings.

FOCUS (Dobler and de Melo, 2023) starts a new token not from its pieces but from the old
tokens that behave most like it in the target language:

1. Learn a small auxiliary vector for every token that occurs in target-language text
   tokenized with the *new* tokenizer. Tokens used in similar contexts get similar vectors.
2. For a new token, compute its cosine similarity to every old token that has an auxiliary
   vector, turn the similarities into weights with *sparsemax* (a softmax that gives exactly
   zero weight to dissimilar tokens), and set its embedding to the weighted sum of those old
   tokens' real embeddings.

Differences from the reference implementation, stated so nobody mistakes this for it:
the published code trains fastText vectors; here the auxiliary vectors come from a
positive-PMI co-occurrence matrix reduced with a truncated SVD (the classic count-based
alternative to word2vec). The ``langsfer`` package that ships FOCUS could not be installed next
to the transformers and numpy versions this project uses, so this is our own implementation
following the paper. Results should be called "FOCUS-style".

Torch is imported inside the functions that need it.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass


@dataclass
class AuxEmbeddings:
    """Auxiliary vectors for the tokens that occur in the corpus.

    ``ids[i]`` is the token id whose vector is ``vectors[i]``; ``ids`` is sorted.
    """

    ids: object  # LongTensor (n,)
    vectors: object  # FloatTensor (n, dim)

    def lookup(self, token_ids):
        """Row index for each id, or -1 where the token has no vector."""
        import torch

        ids = self.ids
        pos = torch.searchsorted(ids, token_ids).clamp(max=len(ids) - 1)
        found = ids[pos] == token_ids
        return torch.where(found, pos, torch.full_like(pos, -1))


def sparsemax(z, dim: int = -1):
    """Sparsemax (Martins and Astudillo, 2016): the closest probability vector to ``z``.

    Sort z in decreasing order, find the largest k with 1 + k z_(k) > sum of the top k values,
    set tau = (sum of the top k - 1) / k, and return max(z - tau, 0). Unlike softmax it can
    return exact zeros. Worked example: z = [0.5, 0.3, 0.1] gives [0.533, 0.333, 0.133];
    z = [2, 0, 0] gives [1, 0, 0].
    """
    import torch

    z = z.transpose(dim, -1)
    zs, _ = torch.sort(z, dim=-1, descending=True)
    cssv = zs.cumsum(dim=-1)
    k = torch.arange(1, z.shape[-1] + 1, device=z.device, dtype=z.dtype)
    support = (1 + k * zs) > cssv
    k_z = support.sum(dim=-1, keepdim=True)
    tau = (cssv.gather(-1, k_z - 1) - 1) / k_z.to(z.dtype)
    return torch.clamp(z - tau, min=0).transpose(dim, -1)


def cooccurrence(token_ids, line_ids, window: int = 5, n_tokens: int | None = None):
    """Symmetric, distance-weighted co-occurrence counts as a coalesced sparse tensor.

    ``token_ids`` is the flat token stream and ``line_ids`` says which line each token came
    from, so pairs never straddle two lines. A pair at distance d counts 1 / d, in both
    directions.
    """
    import torch

    if window < 1:
        raise ValueError("window must be at least 1")
    n = int(token_ids.max()) + 1 if n_tokens is None else n_tokens
    parts_i, parts_j, parts_w = [], [], []
    for d in range(1, window + 1):
        same = line_ids[:-d] == line_ids[d:]
        a, b = token_ids[:-d][same], token_ids[d:][same]
        w = torch.full((a.numel(),), 1.0 / d)
        parts_i += [a, b]
        parts_j += [b, a]
        parts_w += [w, w]
    idx = torch.stack([torch.cat(parts_i), torch.cat(parts_j)])
    return torch.sparse_coo_tensor(idx, torch.cat(parts_w), (n, n)).coalesce()


def ppmi(counts, smoothing: float = 0.75):
    """Positive PMI of a coalesced sparse count matrix, with context-distribution smoothing.

    pmi(i, j) = log( p(i, j) / ( p(i) x p_s(j) ) ), where p_s(j) is proportional to count(j)^0.75
    (which stops rare contexts from getting huge scores). Negative values are set to 0.
    """
    import torch

    idx, val = counts.indices(), counts.values()
    total = val.sum()
    row = torch.zeros(counts.shape[0]).index_add_(0, idx[0], val)
    col = torch.zeros(counts.shape[1]).index_add_(0, idx[1], val)
    col_s = col.pow(smoothing)
    col_s = col_s / col_s.sum()
    pmi = torch.log((val / total) / ((row[idx[0]] / total) * col_s[idx[1]]))
    keep = pmi > 0
    return torch.sparse_coo_tensor(idx[:, keep], pmi[keep], counts.shape).coalesce()


def build_aux_embeddings(
    streams: Sequence[Sequence[int]],
    dim: int = 100,
    window: int = 5,
    min_count: int = 5,
    seed: int = 0,
) -> AuxEmbeddings:
    """Auxiliary vectors from token streams (one list of token ids per line).

    Tokens seen fewer than ``min_count`` times get no vector. Vectors are the left singular
    vectors of the PPMI matrix scaled by the square root of the singular values.
    """
    import torch

    if not streams:
        raise ValueError("no token streams given")
    flat = torch.cat([torch.as_tensor(s, dtype=torch.long) for s in streams if len(s)])
    lines = torch.cat(
        [torch.full((len(s),), i, dtype=torch.long) for i, s in enumerate(streams) if len(s)]
    )
    freq = torch.bincount(flat)
    keep_ids = torch.nonzero(freq >= min_count).squeeze(1)
    if keep_ids.numel() <= dim:
        raise ValueError(f"only {keep_ids.numel()} tokens reach min_count; need more than {dim}")
    remap = torch.full((freq.numel(),), -1, dtype=torch.long)
    remap[keep_ids] = torch.arange(keep_ids.numel())
    compact = remap[flat]
    ok = compact >= 0
    compact, lines = compact[ok], lines[ok]
    counts = cooccurrence(compact, lines, window, n_tokens=keep_ids.numel())
    torch.manual_seed(seed)
    u, s, _ = torch.svd_lowrank(ppmi(counts), q=dim + 10, niter=3)
    vectors = (u[:, :dim] * s[:dim].sqrt()).contiguous()
    return AuxEmbeddings(ids=keep_ids, vectors=vectors)


def focus_rows(
    weight,
    aux: AuxEmbeddings,
    first_new_id: int,
    n_new: int,
    fallback_pieces: Sequence[Sequence[int]] | None = None,
    chunk: int = 512,
):
    """FOCUS rows for the new ids ``first_new_id .. first_new_id + n_new - 1``.

    ``weight`` holds the old embeddings (rows below ``first_new_id`` are used). A new token
    without an auxiliary vector cannot be placed, so it falls back to the mean of its pieces
    when ``fallback_pieces`` is given, otherwise this raises. Returns ``(rows, info)``.
    """
    import torch

    from zubaan.extend.init_emb import mean_of_pieces

    old_mask = aux.ids < first_new_id
    old_ids = aux.ids[old_mask]
    if old_ids.numel() == 0:
        raise ValueError("no old token has an auxiliary vector")
    old_vecs = torch.nn.functional.normalize(aux.vectors[old_mask], dim=1)
    old_emb = weight[old_ids].float()

    new_ids = torch.arange(first_new_id, first_new_id + n_new)
    pos = aux.lookup(new_ids)
    rows = torch.zeros((n_new, weight.shape[1]), dtype=torch.float32)
    support = torch.zeros(n_new)
    have = pos >= 0
    have_idx = torch.nonzero(have).squeeze(1)
    for start in range(0, have_idx.numel(), chunk):
        sel = have_idx[start : start + chunk]
        q = torch.nn.functional.normalize(aux.vectors[pos[sel]], dim=1)
        w = sparsemax(q @ old_vecs.T, dim=-1)
        rows[sel] = w @ old_emb
        support[sel] = (w > 0).sum(dim=1).float()
    missing = torch.nonzero(~have).squeeze(1)
    if missing.numel():
        if fallback_pieces is None:
            raise ValueError(f"{missing.numel()} new tokens have no auxiliary vector")
        rows[missing] = mean_of_pieces(weight, [fallback_pieces[i] for i in missing.tolist()])
    info = {
        "n_new": n_new,
        "n_with_vector": int(have.sum()),
        "n_fallback_to_mean": int(missing.numel()),
        "n_old_candidates": int(old_ids.numel()),
        "mean_support": float(support[have].mean()) if have.any() else 0.0,
    }
    return rows, info
