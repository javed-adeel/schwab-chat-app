"""Rank fusion for combining lexical and semantic result lists."""

from __future__ import annotations

from collections.abc import Sequence


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[int]],
    k: int = 60,
    weights: Sequence[float] | None = None,
) -> dict[int, float]:
    """Reciprocal Rank Fusion (Cormack et al., 2009).

    Score-free, so it needs no calibration between BM25 scores and cosine similarities;
    an item ranked well by *either* retriever surfaces, and by both surfaces higher.
    """
    if weights is not None and len(weights) != len(rankings):
        raise ValueError("weights must match the number of rankings")
    fused: dict[int, float] = {}
    for i, ranking in enumerate(rankings):
        weight = 1.0 if weights is None else weights[i]
        for rank, item in enumerate(ranking, start=1):
            fused[item] = fused.get(item, 0.0) + weight / (k + rank)
    return fused
