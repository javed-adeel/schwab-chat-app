"""Retrieval-quality regression gate.

The golden set was hand-written from the dataset by the author, so treat these numbers as a
regression guard and a sanity check, not as an unbiased benchmark. Thresholds sit slightly
below the measured values to allow harmless drift while catching real regressions.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from newschat.domain import Corpus
from newschat.evaluation.golden import GoldenCase, GoldenSetError, load_golden, resolve_relevant
from newschat.evaluation.metrics import hit_at_k, recall_at_k, reciprocal_rank
from newschat.evaluation.runner import EvalReport, evaluate, format_report
from newschat.retrieval.query_analysis import QueryAnalyzer
from newschat.retrieval.retriever import HybridRetriever

GOLDEN = Path(__file__).resolve().parents[2] / "evals" / "golden_set.json"

pytestmark = pytest.mark.eval


@pytest.fixture(scope="module")
def cases() -> list[GoldenCase]:
    return load_golden(GOLDEN)


@pytest.fixture(scope="module")
def report(
    cases: list[GoldenCase], corpus: Corpus, analyzer: QueryAnalyzer, retriever: HybridRetriever
) -> EvalReport:
    return evaluate(cases, corpus, analyzer, retriever, k=5)


class TestGoldenSetIntegrity:
    def test_ids_are_unique_and_categories_are_covered(self, cases: list[GoldenCase]) -> None:
        ids = [c.id for c in cases]
        assert len(ids) == len(set(ids))
        assert {c.category for c in cases} >= {
            "company",
            "comparison",
            "follow_up",
            "market",
            "out_of_scope",
            "paraphrase",
        }

    def test_every_title_prefix_resolves_to_exactly_one_article(
        self, cases: list[GoldenCase], corpus: Corpus
    ) -> None:
        for case in cases:
            assert len(resolve_relevant(case, corpus)) == len(case.relevant_titles)

    def test_a_dangling_title_prefix_is_reported(
        self, cases: list[GoldenCase], corpus: Corpus
    ) -> None:
        from dataclasses import replace

        bad = replace(cases[0], relevant_titles=("This title does not exist",))
        with pytest.raises(GoldenSetError, match="matched 0"):
            resolve_relevant(bad, corpus)

    def test_missing_file_is_reported(self, tmp_path: Path) -> None:
        with pytest.raises(GoldenSetError):
            load_golden(tmp_path / "nope.json")


class TestQualityGate:
    def test_overall_hit_rate_and_mrr(self, report: EvalReport) -> None:
        assert report.hit_rate >= 0.95, format_report(report)
        assert report.mrr >= 0.85, format_report(report)
        assert report.mean_recall >= 0.90, format_report(report)

    def test_out_of_scope_questions_are_declined_and_in_scope_ones_answered(
        self, report: EvalReport
    ) -> None:
        wrong = [r.case.id for r in report.results if not r.gate_correct]
        assert wrong == [], wrong

    def test_comparisons_cover_every_requested_company(self, report: EvalReport) -> None:
        assert report.ticker_coverage == pytest.approx(1.0)

    def test_follow_ups_resolve_through_history(self, report: EvalReport) -> None:
        n, hit, mrr = report.by_category()["follow_up"]
        assert n >= 3 and hit == 1.0 and mrr >= 0.8

    def test_paraphrased_questions_still_retrieve_the_right_article(
        self, report: EvalReport
    ) -> None:
        n, hit, mrr = report.by_category()["paraphrase"]
        assert n >= 10 and hit >= 0.9 and mrr >= 0.75

    def test_report_formatting_lists_categories(self, report: EvalReport) -> None:
        text = format_report(report)
        assert "hit@5" in text and "paraphrase" in text and "out_of_scope" in text


class TestMetrics:
    def test_hit_recall_rr(self) -> None:
        ranked = ["a", "b", "c", "d"]
        assert hit_at_k(ranked, {"c"}, 2) == 0.0 and hit_at_k(ranked, {"c"}, 3) == 1.0
        assert recall_at_k(ranked, {"a", "c", "z"}, 3) == pytest.approx(2 / 3)
        assert reciprocal_rank(ranked, {"c", "d"}) == pytest.approx(1 / 3)
        assert reciprocal_rank(ranked, {"zzz"}) == 0.0

    def test_recall_needs_relevant_items(self) -> None:
        with pytest.raises(ValueError, match="undefined"):
            recall_at_k(["a"], set(), 1)
