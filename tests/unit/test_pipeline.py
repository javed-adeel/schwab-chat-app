from __future__ import annotations

import pytest

from newschat.config import Settings
from newschat.domain import Corpus
from newschat.ingestion.entities import EntityRegistry
from newschat.ingestion.loader import RawArticle
from newschat.ingestion.pipeline import (
    IngestionConfig,
    IngestionPipeline,
    article_id,
    canonical_link,
    normalize_title,
)


@pytest.mark.parametrize(
    ("a", "b"),
    [
        ("https://Finance.Yahoo.com/news/x.html", "https://finance.yahoo.com/news/x.html"),
        ("https://finance.yahoo.com/news/x.html/", "https://finance.yahoo.com/news/x.html"),
        (
            "http://finance.yahoo.com/news/x.html?src=rss#top",
            "https://finance.yahoo.com/news/x.html",
        ),
    ],
)
def test_canonical_link_ignores_case_scheme_query_fragment_and_trailing_slash(
    a: str, b: str
) -> None:
    assert canonical_link(a) == canonical_link(b)
    assert article_id(a) == article_id(b)


def test_article_ids_are_deterministic_and_distinct() -> None:
    assert article_id("https://x.test/a") == article_id("https://x.test/a")
    assert article_id("https://x.test/a") != article_id("https://x.test/b")


def test_normalize_title() -> None:
    assert normalize_title("Intel: Breakup?! Mulled") == "intel breakup mulled"


class TestDeduplication:
    def test_same_link_under_two_tickers_is_merged_and_keeps_the_longest_text(
        self, mini_corpus: Corpus
    ) -> None:
        intel = [a for a in mini_corpus.articles.values() if a.title == "Intel breakup mulled"]
        assert len(intel) == 1
        assert set(intel[0].source_tickers) == {"INTC", "NVDA"}
        assert "chipmaker" in intel[0].text  # the long version won, not "short"

    def test_same_title_with_a_different_link_is_also_merged(
        self, registry: EntityRegistry
    ) -> None:
        raw = {
            "AAPL": [RawArticle(title="Big News!", link="https://a.test/1", full_text="x" * 500)],
            "MSFT": [RawArticle(title="big news", link="https://b.test/2", full_text="y" * 500)],
        }
        corpus = IngestionPipeline(registry).build(raw)
        assert len(corpus.articles) == 1 and corpus.report.duplicates_merged == 1

    def test_distinct_articles_are_not_merged(self, registry: EntityRegistry) -> None:
        raw = {
            "AAPL": [
                RawArticle(title="One", link="https://a.test/1", full_text="a"),
                RawArticle(title="Two", link="https://a.test/2", full_text="b"),
            ]
        }
        assert len(IngestionPipeline(registry).build(raw).articles) == 2

    def test_a_ticker_listed_twice_is_recorded_once(self, registry: EntityRegistry) -> None:
        raw = {
            "AAPL": [
                RawArticle(title="One", link="https://a.test/1", full_text="a"),
                RawArticle(title="One", link="https://a.test/1", full_text="a"),
            ]
        }
        article = next(iter(IngestionPipeline(registry).build(raw).articles.values()))
        assert article.source_tickers == ("AAPL",)


class TestBuild:
    def test_report_counts(self, mini_corpus: Corpus) -> None:
        r = mini_corpus.report
        assert r.raw_records == 6 and r.duplicates_merged == 1 and r.articles == 5
        assert r.headline_only == 1  # the paywalled Evercore item
        assert r.chunks == len(mini_corpus.chunks)
        assert r.boilerplate_removed["paywall"] == 1
        assert r.boilerplate_removed["view_comments"] == 1

    def test_paywalled_article_is_headline_only_with_a_single_chunk(
        self, mini_corpus: Corpus
    ) -> None:
        stub = next(a for a in mini_corpus.articles.values() if a.headline_only)
        chunks = [c for c in mini_corpus.chunks if c.article_id == stub.id]
        assert len(chunks) == 1 and stub.title in chunks[0].index_text

    def test_long_articles_are_split_into_multiple_bounded_chunks(
        self, mini_corpus: Corpus
    ) -> None:
        assert max(len(c.text) for c in mini_corpus.chunks) <= 300
        nvda = next(a for a in mini_corpus.articles.values() if "Nvidia" in a.title)
        assert len([c for c in mini_corpus.chunks if c.article_id == nvda.id]) > 1

    def test_every_chunk_is_searchable_by_its_title(self, mini_corpus: Corpus) -> None:
        for chunk in mini_corpus.chunks:
            title = mini_corpus.articles[chunk.article_id].title
            assert chunk.index_text.startswith(title)

    def test_generic_article_keeps_its_feed_label(self, mini_corpus: Corpus) -> None:
        wrap = next(a for a in mini_corpus.articles.values() if a.title.startswith("Stocks slip"))
        assert wrap.tickers == ("AAPL",)

    def test_chunk_ids_are_unique_and_reference_real_articles(self, mini_corpus: Corpus) -> None:
        ids = [c.id for c in mini_corpus.chunks]
        assert len(ids) == len(set(ids))
        assert all(c.article_id in mini_corpus.articles for c in mini_corpus.chunks)

    def test_ingestion_is_deterministic(
        self, registry: EntityRegistry, mini_raw: dict[str, list[RawArticle]]
    ) -> None:
        cfg = IngestionConfig(chunk_max_chars=300, min_content_chars=100)
        a = IngestionPipeline(registry, cfg).build(mini_raw)
        b = IngestionPipeline(registry, cfg).build(mini_raw)
        assert [c.id for c in a.chunks] == [c.id for c in b.chunks]
        assert a.report == b.report

    def test_empty_body_falls_back_to_the_title(self, registry: EntityRegistry) -> None:
        raw = {"IBM": [RawArticle(title="IBM wins deal", link="https://a.test/1", full_text="")]}
        corpus = IngestionPipeline(registry).build(raw)
        assert corpus.chunks[0].text == "IBM wins deal"

    def test_run_reads_from_disk(self, registry: EntityRegistry, settings: Settings) -> None:
        corpus = IngestionPipeline(registry).run(settings.data_path)
        assert corpus.report.raw_records > 0


class TestRealDataset:
    """Invariants of the actual scraped dataset (guards against cleaning regressions)."""

    def test_headline_counts(self, corpus: Corpus) -> None:
        r = corpus.report
        assert r.raw_records == 138 and r.invalid_records == 0
        assert r.articles == 118 and r.duplicates_merged == 20
        assert set(r.articles_per_ticker) == {"AAPL", "MSFT", "AMZN", "NFLX", "NVDA", "INTC", "IBM"}

    def test_no_publisher_boilerplate_survives(self, corpus: Corpus) -> None:
        markers = (
            "View Comments",
            "PREMIUM Upgrade",
            "hedge funds pile into",
            "Story Continues",
            "READ NEXT",
        )
        offenders = [a.title for a in corpus.articles.values() if any(m in a.text for m in markers)]
        assert offenders == []

    def test_paywalled_price_target_stubs_are_kept_and_flagged(self, corpus: Corpus) -> None:
        stub = next(
            a
            for a in corpus.articles.values()
            if a.title.startswith("Evercore ISI Adjusts Price Target")
        )
        assert stub.headline_only and stub.tickers == ("INTC",)

    def test_feed_label_is_not_treated_as_topic(self, corpus: Corpus) -> None:
        breakup = next(
            a for a in corpus.articles.values() if a.title.startswith("Intel Breakup Mulled")
        )
        assert breakup.source_tickers == ("AAPL",) and breakup.tickers == ("INTC",)

    def test_chunk_size_bound_holds_on_real_text(self, corpus: Corpus, settings: Settings) -> None:
        assert (
            max(
                len(c.text)
                for c in corpus.chunks
                if not corpus.articles[c.article_id].headline_only
            )
            <= settings.chunk_max_chars
        )
