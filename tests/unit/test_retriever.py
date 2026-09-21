from __future__ import annotations

from dataclasses import replace

import pytest

from newschat.domain import Corpus, Intent, QueryPlan
from newschat.retrieval.query_analysis import QueryAnalyzer
from newschat.retrieval.retriever import HybridRetriever, RetrievalConfig


def plan(question: str, intent: Intent, *tickers: str) -> QueryPlan:
    return QueryPlan(question, question, intent, tuple(tickers), False)


class TestOnMiniCorpus:
    def test_company_questions_only_return_that_companys_articles(
        self, mini_retriever: HybridRetriever
    ) -> None:
        result = mini_retriever.retrieve(plan("chip breakup", Intent.COMPANY, "INTC"))
        assert result.passages
        assert all(
            "INTC" in p.article.tickers
            for p in result.passages
            if p.article.title != "Stocks slip as yields climb"
        )

    def test_comparisons_include_every_requested_company(
        self, mini_retriever: HybridRetriever
    ) -> None:
        result = mini_retriever.retrieve(plan("revenue outlook", Intent.COMPARISON, "NVDA", "AAPL"))
        covered = {t for p in result.passages for t in p.article.tickers}
        assert {"NVDA", "AAPL"} <= covered

    def test_comparison_results_interleave_rather_than_cluster(
        self, mini_retriever: HybridRetriever
    ) -> None:
        result = mini_retriever.retrieve(plan("revenue outlook", Intent.COMPARISON, "NVDA", "AAPL"))
        first_two = [p.article.tickers for p in result.passages[:2]]
        assert first_two[0] != first_two[1]

    def test_per_article_cap_is_enforced(self, mini_corpus: Corpus) -> None:
        r = HybridRetriever(
            mini_corpus,
            config=RetrievalConfig(top_k=10, max_chunks_per_article=1),
            lsa_components=8,
        )
        result = r.retrieve(plan("nvidia data center revenue", Intent.MARKET))
        ids = [p.article.id for p in result.passages]
        assert len(ids) == len(set(ids))

    def test_filter_widens_when_a_company_has_too_little_material(
        self, mini_corpus: Corpus
    ) -> None:
        # INTC has 2 articles; demand 3 so the filter must relax to the whole corpus.
        cfg = RetrievalConfig(top_k=6, min_articles_for_filter=3, min_anchored_coverage=0.0)
        result = HybridRetriever(mini_corpus, config=cfg, lsa_components=8).retrieve(
            plan("intel revenue results", Intent.COMPANY, "INTC")
        )
        titles = [p.article.title for p in result.passages]
        assert any("Intel" in t for t in titles[:2])  # company material still comes first
        assert len({p.article.id for p in result.passages}) >= 3  # ...but the search widened

    def test_unknown_ticker_yields_no_company_material_but_does_not_crash(
        self, mini_retriever: HybridRetriever
    ) -> None:
        result = mini_retriever.retrieve(plan("anything", Intent.COMPANY, "TSLA"))
        assert isinstance(result.passages, tuple)

    @pytest.mark.parametrize(("lexical", "semantic"), [(0.0, 1.0), (1.0, 0.0), (1.0, 1.0)])
    def test_each_component_can_be_disabled_for_ablation(
        self, mini_corpus: Corpus, lexical: float, semantic: float
    ) -> None:
        cfg = RetrievalConfig(lexical_weight=lexical, semantic_weight=semantic)
        result = HybridRetriever(mini_corpus, config=cfg, lsa_components=8).retrieve(
            plan("nvidia blackwell revenue", Intent.MARKET)
        )
        assert result.passages and "Nvidia" in result.passages[0].article.title


class TestOnRealCorpus:
    def test_price_target_question_surfaces_the_stub_articles_that_hold_the_answer(
        self, analyzer: QueryAnalyzer, retriever: HybridRetriever
    ) -> None:
        result = retriever.retrieve(
            analyzer.analyze("What price target changes were reported for Intel?")
        )
        titles = " | ".join(p.article.title for p in result.passages[:4])
        assert "Evercore" in titles and "Cantor" in titles and "Citic" in titles

    def test_retrieval_is_deterministic(
        self, analyzer: QueryAnalyzer, retriever: HybridRetriever
    ) -> None:
        q = analyzer.analyze("Compare Nvidia and Intel")
        a = [p.chunk.id for p in retriever.retrieve(q).passages]
        b = [p.chunk.id for p in retriever.retrieve(q).passages]
        assert a == b

    @pytest.mark.parametrize(
        ("question", "sufficient"),
        [
            ("What is the weather in Toronto today?", False),
            ("Who won the Super Bowl?", False),
            ("Write me a poem about the ocean", False),
            ("hi", False),
            ("What price target changes were reported for Intel?", True),
            ("Which companies crushed quarterly expectations?", True),
            ("Intel rumours", True),  # anchored on a known company: a missing word is tolerated
        ],
    )
    def test_relevance_gate(
        self, analyzer: QueryAnalyzer, retriever: HybridRetriever, question: str, sufficient: bool
    ) -> None:
        assert retriever.retrieve(analyzer.analyze(question)).sufficient is sufficient

    def test_uncovered_terms_are_reported_with_their_surface_form(
        self, analyzer: QueryAnalyzer, retriever: HybridRetriever
    ) -> None:
        result = retriever.retrieve(analyzer.analyze("What did Intel say about bitcoin?"))
        assert "bitcoin" in result.uncovered_terms and 0 < result.coverage < 1

    def test_fully_covered_question_has_no_uncovered_terms(
        self, analyzer: QueryAnalyzer, retriever: HybridRetriever
    ) -> None:
        result = retriever.retrieve(analyzer.analyze("What did Jim Cramer say about Netflix?"))
        assert result.uncovered_terms == () and result.coverage == pytest.approx(1.0)

    def test_company_filter_holds_on_real_data(
        self, analyzer: QueryAnalyzer, retriever: HybridRetriever
    ) -> None:
        result = retriever.retrieve(analyzer.analyze("What is IBM doing with Penn State?"))
        assert all("IBM" in p.article.tickers for p in result.passages)

    def test_thresholds_are_configurable(
        self, corpus: Corpus, analyzer: QueryAnalyzer, retriever: HybridRetriever
    ) -> None:
        strict = HybridRetriever(
            corpus, config=replace(RetrievalConfig(), min_anchored_coverage=1.01), lsa_components=64
        )
        assert not strict.retrieve(
            analyzer.analyze("What is IBM doing with Penn State?")
        ).sufficient
