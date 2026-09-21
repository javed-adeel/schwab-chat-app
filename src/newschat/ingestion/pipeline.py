"""Ingestion pipeline: raw JSON -> cleaned, deduplicated, entity-tagged, chunked corpus."""

from __future__ import annotations

import hashlib
import logging
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlsplit

from newschat.domain import Article, Chunk, Corpus, IngestionReport
from newschat.ingestion.chunking import chunk_text
from newschat.ingestion.cleaning import clean_text, is_headline_only
from newschat.ingestion.entities import EntityRegistry
from newschat.ingestion.loader import RawArticle, load_raw_articles

logger = logging.getLogger(__name__)

_NON_ALNUM = re.compile(r"[^a-z0-9]+")


def canonical_link(link: str) -> str:
    """Normalise a URL for identity comparison (scheme, host case, query, trailing slash)."""
    parts = urlsplit(link.strip())
    return f"https://{parts.netloc.lower()}{parts.path.rstrip('/')}"


def normalize_title(title: str) -> str:
    return _NON_ALNUM.sub(" ", title.lower()).strip()


def article_id(link: str) -> str:
    return hashlib.sha1(canonical_link(link).encode()).hexdigest()[:12]


@dataclass(slots=True)
class _Candidate:
    raw: RawArticle
    source_tickers: list[str] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class IngestionConfig:
    chunk_max_chars: int = 1100
    chunk_overlap_sentences: int = 1
    min_content_chars: int = 400


class IngestionPipeline:
    def __init__(self, registry: EntityRegistry, config: IngestionConfig | None = None) -> None:
        self._registry = registry
        self._config = config or IngestionConfig()

    def run(self, path: Path) -> Corpus:
        raw, invalid = load_raw_articles(path)
        return self.build(raw, invalid_records=invalid)

    def build(self, raw: Mapping[str, Sequence[RawArticle]], invalid_records: int = 0) -> Corpus:
        raw_count = sum(len(v) for v in raw.values())
        candidates = self._deduplicate(raw)
        duplicates_merged = raw_count - len(candidates)

        articles: dict[str, Article] = {}
        chunks: list[Chunk] = []
        boilerplate: Counter[str] = Counter()
        headline_only = 0

        for cand in candidates:
            cleaned = clean_text(cand.raw.full_text)
            boilerplate.update(cleaned.rules_applied)
            stub = is_headline_only(cleaned, self._config.min_content_chars)
            headline_only += stub
            source = tuple(dict.fromkeys(cand.source_tickers))
            article = Article(
                id=article_id(cand.raw.link),
                title=cand.raw.title,
                link=cand.raw.link,
                text=cleaned.text,
                source_tickers=source,
                tickers=self._registry.resolve_article_tickers(
                    cand.raw.title, cleaned.text, source
                ),
                headline_only=stub,
            )
            articles[article.id] = article
            chunks.extend(self._chunk(article))

        per_ticker = Counter(t for a in articles.values() for t in a.tickers)
        report = IngestionReport(
            raw_records=raw_count,
            invalid_records=invalid_records,
            duplicates_merged=duplicates_merged,
            headline_only=headline_only,
            articles=len(articles),
            chunks=len(chunks),
            boilerplate_removed=dict(boilerplate),
            articles_per_ticker=dict(sorted(per_ticker.items())),
        )
        logger.info("ingestion complete", extra={"report": report.as_dict()})
        return Corpus(articles=articles, chunks=tuple(chunks), report=report)

    # ------------------------------------------------------------------ internals

    @staticmethod
    def _deduplicate(raw: Mapping[str, Sequence[RawArticle]]) -> list[_Candidate]:
        """Merge records that share a canonical link or a normalised title.

        The dataset lists the same article under several tickers; we keep the record with
        the most text and remember every ticker feed it appeared in.
        """
        by_key: dict[str, _Candidate] = {}
        order: list[_Candidate] = []
        for ticker, records in raw.items():
            for record in records:
                keys = (
                    f"link:{canonical_link(record.link)}",
                    f"title:{normalize_title(record.title)}",
                )
                existing = next((by_key[k] for k in keys if k in by_key), None)
                if existing is None:
                    existing = _Candidate(raw=record)
                    order.append(existing)
                elif len(record.full_text) > len(existing.raw.full_text):
                    existing.raw = record
                existing.source_tickers.append(ticker)
                for k in keys:
                    by_key.setdefault(k, existing)
        return order

    def _chunk(self, article: Article) -> list[Chunk]:
        body = article.text or article.title
        pieces = (
            [body]
            if article.headline_only
            else chunk_text(
                body, self._config.chunk_max_chars, self._config.chunk_overlap_sentences
            )
        )
        return [
            Chunk(
                id=f"{article.id}:{i}",
                article_id=article.id,
                ordinal=i,
                text=piece,
                index_text=f"{article.title}\n{piece}",
            )
            for i, piece in enumerate(pieces or [article.title])
        ]
