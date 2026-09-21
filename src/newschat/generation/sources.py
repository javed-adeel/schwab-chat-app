"""Groups retrieved passages into numbered, citable sources (one per article)."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from newschat.domain import Article
from newschat.retrieval.retriever import RetrievedPassage


@dataclass(frozen=True, slots=True)
class Source:
    number: int  # 1-based citation number shown to the model and the user
    article: Article
    passages: tuple[RetrievedPassage, ...]

    @property
    def text(self) -> str:
        return " ".join(p.chunk.text for p in self.passages)


def build_sources(passages: Sequence[RetrievedPassage]) -> tuple[Source, ...]:
    """One numbered source per article, in order of first (best) appearance."""
    grouped: dict[str, list[RetrievedPassage]] = {}
    for passage in passages:
        grouped.setdefault(passage.article.id, []).append(passage)
    return tuple(
        Source(number=i, article=group[0].article, passages=tuple(group))
        for i, group in enumerate(grouped.values(), start=1)
    )
