"""Ranking metrics over article ids (best-first lists)."""

from __future__ import annotations

from collections.abc import Collection, Sequence


def hit_at_k(ranked: Sequence[str], relevant: Collection[str], k: int) -> float:
    return 1.0 if any(a in relevant for a in ranked[:k]) else 0.0


def recall_at_k(ranked: Sequence[str], relevant: Collection[str], k: int) -> float:
    if not relevant:
        raise ValueError("recall is undefined without relevant items")
    return len({a for a in ranked[:k] if a in relevant}) / len(relevant)


def reciprocal_rank(ranked: Sequence[str], relevant: Collection[str]) -> float:
    for position, article in enumerate(ranked, start=1):
        if article in relevant:
            return 1.0 / position
    return 0.0
