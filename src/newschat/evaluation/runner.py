"""Runs the golden set through query analysis + retrieval and aggregates metrics.

This evaluates the *retrieval* stack, which is deterministic and free, so it can gate CI.
Answer quality on top of retrieval depends on the LLM and is checked structurally
(citation validity, refusals) in the API/service tests rather than by string matching.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from statistics import mean

from newschat.domain import Corpus
from newschat.evaluation.golden import GoldenCase, resolve_relevant
from newschat.evaluation.metrics import hit_at_k, recall_at_k, reciprocal_rank
from newschat.retrieval.query_analysis import QueryAnalyzer
from newschat.retrieval.retriever import HybridRetriever


@dataclass(frozen=True, slots=True)
class CaseResult:
    case: GoldenCase
    ranked_titles: tuple[str, ...]
    hit: float | None  # None when the case has no relevant articles
    recall: float | None
    rr: float | None
    ticker_coverage: float | None  # share of requested companies represented in results
    gate_correct: bool
    coverage: float


@dataclass(frozen=True, slots=True)
class EvalReport:
    k: int
    results: tuple[CaseResult, ...]

    def _mean(self, values: Sequence[float | None]) -> float:
        present = [v for v in values if v is not None]
        return mean(present) if present else float("nan")

    @property
    def hit_rate(self) -> float:
        return self._mean([r.hit for r in self.results])

    @property
    def mean_recall(self) -> float:
        return self._mean([r.recall for r in self.results])

    @property
    def mrr(self) -> float:
        return self._mean([r.rr for r in self.results])

    @property
    def ticker_coverage(self) -> float:
        return self._mean([r.ticker_coverage for r in self.results])

    @property
    def gate_accuracy(self) -> float:
        return mean(1.0 if r.gate_correct else 0.0 for r in self.results)

    def by_category(self) -> dict[str, tuple[int, float, float]]:
        """category -> (cases, hit rate, mrr) over the cases that have relevant articles."""
        out: dict[str, tuple[int, float, float]] = {}
        for cat in sorted({r.case.category for r in self.results}):
            rows = [r for r in self.results if r.case.category == cat]
            out[cat] = (
                len(rows),
                self._mean([r.hit for r in rows]),
                self._mean([r.rr for r in rows]),
            )
        return out


def evaluate(
    cases: Sequence[GoldenCase],
    corpus: Corpus,
    analyzer: QueryAnalyzer,
    retriever: HybridRetriever,
    k: int = 5,
) -> EvalReport:
    results: list[CaseResult] = []
    for case in cases:
        plan = analyzer.analyze(case.question, case.history)
        retrieved = retriever.retrieve(plan)
        ranked_ids = list(dict.fromkeys(p.article.id for p in retrieved.passages))
        relevant = resolve_relevant(case, corpus)

        ticker_coverage = None
        if case.must_cover_tickers:
            top = {t for p in retrieved.passages[: k * 2] for t in p.article.tickers}
            ticker_coverage = mean(1.0 if t in top else 0.0 for t in case.must_cover_tickers)

        results.append(
            CaseResult(
                case=case,
                ranked_titles=tuple(corpus.articles[a].title for a in ranked_ids),
                hit=hit_at_k(ranked_ids, relevant, k) if relevant else None,
                recall=recall_at_k(ranked_ids, relevant, k) if relevant else None,
                rr=reciprocal_rank(ranked_ids, relevant) if relevant else None,
                ticker_coverage=ticker_coverage,
                gate_correct=retrieved.sufficient == case.expect_answered,
                coverage=retrieved.coverage,
            )
        )
    return EvalReport(k=k, results=tuple(results))


def format_report(report: EvalReport) -> str:
    lines = [
        f"Retrieval evaluation ({len(report.results)} cases, k={report.k} articles)",
        f"  hit@{report.k}:            {report.hit_rate:.3f}",
        f"  recall@{report.k}:         {report.mean_recall:.3f}",
        f"  MRR:               {report.mrr:.3f}",
        f"  ticker coverage:   {report.ticker_coverage:.3f}   (comparison questions)",
        f"  gate accuracy:     {report.gate_accuracy:.3f}   (answer vs. decline)",
        "",
        "By category:",
    ]
    for cat, (n, hit, rr) in report.by_category().items():
        lines.append(f"  {cat:<13} n={n:<3} hit={hit:.2f} mrr={rr:.2f}")
    misses = [
        r
        for r in report.results
        if (r.hit == 0.0)
        or not r.gate_correct
        or (r.ticker_coverage is not None and r.ticker_coverage < 1.0)
    ]
    if misses:
        lines += ["", "Failures:"]
        for r in misses:
            lines.append(f"  - {r.case.id}: {r.case.question!r} (coverage={r.coverage:.2f})")
            lines += [f"      got: {t[:80]}" for t in r.ranked_titles[:3]]
    return "\n".join(lines)
