"""Core domain types shared across the ingestion, retrieval and generation layers.

They are plain frozen dataclasses: immutable, hashable where possible, and free of any
framework dependency so every layer can be tested in isolation.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Literal

Role = Literal["user", "assistant"]


class Intent(str, Enum):
    """What kind of question the user asked; drives the retrieval strategy."""

    COMPANY = "company"  # about one company
    COMPARISON = "comparison"  # about two or more companies
    MARKET = "market"  # no specific company: broad / thematic


@dataclass(frozen=True, slots=True)
class ChatTurn:
    role: Role
    content: str


@dataclass(frozen=True, slots=True)
class Article:
    id: str
    title: str
    link: str
    text: str  # cleaned body
    source_tickers: tuple[str, ...]  # feeds the article was listed under (weak label)
    tickers: tuple[str, ...]  # companies the article is actually about
    headline_only: bool  # paywalled / too short: only the headline is reliable


@dataclass(frozen=True, slots=True)
class Chunk:
    id: str
    article_id: str
    ordinal: int
    text: str  # what is shown to the model / user
    index_text: str  # title + text: what is searched


@dataclass(frozen=True, slots=True)
class IngestionReport:
    """What the ingestion pipeline did; exposed by the API for transparency."""

    raw_records: int
    invalid_records: int
    duplicates_merged: int
    headline_only: int
    articles: int
    chunks: int
    boilerplate_removed: dict[str, int] = field(default_factory=dict)
    articles_per_ticker: dict[str, int] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class Corpus:
    articles: dict[str, Article]
    chunks: tuple[Chunk, ...]
    report: IngestionReport

    @property
    def tickers(self) -> tuple[str, ...]:
        return tuple(sorted({t for a in self.articles.values() for t in a.tickers}))


@dataclass(frozen=True, slots=True)
class QueryPlan:
    """The analysed form of a user question."""

    question: str
    standalone_query: str  # question + context inherited from earlier turns
    intent: Intent
    tickers: tuple[str, ...]
    is_follow_up: bool
