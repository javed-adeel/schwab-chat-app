"""Hybrid retriever: BM25 + dense search fused with RRF, made entity- and intent-aware.

Beyond plain hybrid search it adds three behaviours that matter for this domain:

* **Company filtering** for single-company questions (with automatic relaxation when the
  filter would leave too little material).
* **Balanced retrieval** for comparisons: each company is retrieved separately and the
  results are interleaved, so one company cannot crowd the other out.
* **Query coverage**: the share of the question's informative terms (idf-weighted) that
  exist in the corpus at all. Questions naming a known company need little evidence;
  unanchored questions need much more. Anything below the bar is reported as out of
  scope instead of returning confident-looking noise.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Collection, Sequence
from dataclasses import dataclass

from newschat.config import Settings
from newschat.domain import Article, Chunk, Corpus, Intent, QueryPlan
from newschat.retrieval.bm25 import BM25Index
from newschat.retrieval.dense import DenseIndex, Embedder, LsaEmbedder
from newschat.retrieval.fusion import reciprocal_rank_fusion
from newschat.retrieval.tokenize import Term, terms, tokenize

_MAX_COMPARISON_PASSAGES = 12


@dataclass(frozen=True, slots=True)
class RetrievalConfig:
    top_k: int = 8
    candidate_pool: int = 40
    per_ticker_k: int = 4
    max_chunks_per_article: int = 2
    rrf_k: int = 60
    min_query_coverage: float = 0.60  # required when no company is named
    min_anchored_coverage: float = 0.20  # required when a known company is named
    min_articles_for_filter: int = 2
    lexical_weight: float = 1.0  # 0 disables BM25 (ablation / tuning)
    semantic_weight: float = 1.0  # 0 disables the dense index

    @classmethod
    def from_settings(cls, s: Settings) -> RetrievalConfig:
        return cls(
            top_k=s.top_k,
            candidate_pool=s.candidate_pool,
            per_ticker_k=s.per_ticker_k,
            max_chunks_per_article=s.max_chunks_per_article,
            rrf_k=s.rrf_k,
            min_query_coverage=s.min_query_coverage,
            min_anchored_coverage=s.min_anchored_coverage,
            min_articles_for_filter=s.min_articles_for_filter,
        )


@dataclass(frozen=True, slots=True)
class RetrievedPassage:
    chunk: Chunk
    article: Article
    score: float


@dataclass(frozen=True, slots=True)
class RetrievalResult:
    plan: QueryPlan
    passages: tuple[RetrievedPassage, ...]
    coverage: float
    uncovered_terms: tuple[str, ...]
    sufficient: bool  # False => the corpus cannot meaningfully answer this question


class HybridRetriever:
    def __init__(
        self,
        corpus: Corpus,
        embedder: Embedder | None = None,
        config: RetrievalConfig | None = None,
        lsa_components: int = 128,
    ) -> None:
        self._corpus = corpus
        self._config = config or RetrievalConfig()
        self._chunks = corpus.chunks
        self._bm25 = BM25Index([tokenize(c.index_text) for c in self._chunks])
        self._dense = DenseIndex(
            embedder or LsaEmbedder(n_components=lsa_components),
            [c.index_text for c in self._chunks],
        )
        self._chunk_article = [c.article_id for c in self._chunks]
        by_ticker: dict[str, set[int]] = {}
        for idx, chunk in enumerate(self._chunks):
            for ticker in corpus.articles[chunk.article_id].tickers:
                by_ticker.setdefault(ticker, set()).add(idx)
        self._by_ticker = {t: frozenset(v) for t, v in by_ticker.items()}

    # ------------------------------------------------------------------ public API

    def retrieve(self, plan: QueryPlan) -> RetrievalResult:
        cfg = self._config
        query_terms = terms(plan.standalone_query)
        coverage, uncovered = self._coverage(query_terms)

        if plan.intent is Intent.COMPARISON:
            selected = self._balanced(plan)
        elif plan.intent is Intent.COMPANY:
            selected = self._filtered(plan)
        else:
            selected = self._select(self._rank(plan.standalone_query, None), cfg.top_k)

        # A named, known company anchors the question in this corpus's domain, so it needs
        # far less lexical evidence than an unanchored one ("weather in Toronto").
        threshold = cfg.min_anchored_coverage if plan.tickers else cfg.min_query_coverage
        passages = tuple(
            RetrievedPassage(self._chunks[i], self._corpus.articles[self._chunks[i].article_id], s)
            for i, s in selected
        )
        return RetrievalResult(
            plan=plan,
            passages=passages,
            coverage=coverage,
            uncovered_terms=uncovered,
            sufficient=bool(passages) and coverage >= threshold,
        )

    # ------------------------------------------------------------------ strategies

    def _filtered(self, plan: QueryPlan) -> list[tuple[int, float]]:
        cfg = self._config
        allowed = self._by_ticker.get(plan.tickers[0], frozenset())
        ranked = self._rank(plan.standalone_query, allowed)
        selected = self._select(ranked, cfg.top_k)
        if len({self._chunk_article[i] for i, _ in selected}) >= cfg.min_articles_for_filter:
            return selected
        # Too little company-specific material: fall back to everything, company first.
        seen = {i for i, _ in ranked}
        widened = ranked + [r for r in self._rank(plan.standalone_query, None) if r[0] not in seen]
        return self._select(widened, cfg.top_k)

    def _balanced(self, plan: QueryPlan) -> list[tuple[int, float]]:
        cfg = self._config
        per_ticker = [
            self._select(
                self._rank(plan.standalone_query, self._by_ticker.get(t, frozenset())),
                cfg.per_ticker_k,
            )
            for t in plan.tickers
        ]
        merged: list[tuple[int, float]] = []
        seen: set[int] = set()
        for rank in range(max((len(p) for p in per_ticker), default=0)):
            for results in per_ticker:
                if rank < len(results) and results[rank][0] not in seen:
                    seen.add(results[rank][0])
                    merged.append(results[rank])
        return merged[:_MAX_COMPARISON_PASSAGES]

    # ------------------------------------------------------------------ primitives

    def _rank(self, query: str, allowed: Collection[int] | None) -> list[tuple[int, float]]:
        """Hybrid ranking over the (optionally restricted) chunk set, best first."""
        cfg = self._config
        rankings: list[list[int]] = []
        weights: list[float] = []
        if cfg.lexical_weight > 0:
            rankings.append(
                [i for i, _ in self._bm25.search(tokenize(query), cfg.candidate_pool, allowed)]
            )
            weights.append(cfg.lexical_weight)
        if cfg.semantic_weight > 0:
            rankings.append([i for i, _ in self._dense.search(query, cfg.candidate_pool, allowed)])
            weights.append(cfg.semantic_weight)
        fused = reciprocal_rank_fusion(rankings, k=cfg.rrf_k, weights=weights)
        return sorted(fused.items(), key=lambda kv: (-kv[1], kv[0]))

    def _select(self, ranked: Sequence[tuple[int, float]], limit: int) -> list[tuple[int, float]]:
        """Take the best ``limit`` chunks, capping how many may come from one article."""
        per_article: Counter[str] = Counter()
        out: list[tuple[int, float]] = []
        for idx, score in ranked:
            article = self._chunk_article[idx]
            if per_article[article] >= self._config.max_chunks_per_article:
                continue
            per_article[article] += 1
            out.append((idx, score))
            if len(out) >= limit:
                break
        return out

    def _coverage(self, query_terms: Sequence[Term]) -> tuple[float, tuple[str, ...]]:
        """idf-weighted share of query terms present in the corpus, plus the missing ones."""
        unique = {t.stem: t.surface for t in query_terms}
        if not unique:
            return 0.0, ()
        total = matched = 0.0
        missing: list[str] = []
        for stem, surface in unique.items():
            weight = self._bm25.idf(stem)
            total += weight
            if self._bm25.document_frequency(stem) > 0:
                matched += weight
            else:
                missing.append(surface)
        return (matched / total if total else 0.0), tuple(missing)
