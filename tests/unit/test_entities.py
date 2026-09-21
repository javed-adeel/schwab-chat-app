from __future__ import annotations

import pytest

from newschat.ingestion.entities import Company, EntityRegistry


def test_finds_tickers_in_order_of_first_appearance(registry: EntityRegistry) -> None:
    assert registry.find_tickers("Microsoft, Apple and NVDA vs microsoft again") == (
        "MSFT",
        "AAPL",
        "NVDA",
    )


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("The new iPhone launches", ("AAPL",)),
        ("AWS expands healthcare AI", ("AMZN",)),
        ("Satya Nadella spoke about Azure", ("MSFT",)),
        ("International Business Machines reported", ("IBM",)),
        ("Amazon Web Services outage", ("AMZN",)),
    ],
)
def test_aliases_map_to_their_company(
    registry: EntityRegistry, text: str, expected: tuple[str, ...]
) -> None:
    assert registry.find_tickers(text) == expected


@pytest.mark.parametrize(
    "text", ["artificial intelligence", "Applebee's earnings", "an intelligent move"]
)
def test_word_boundaries_prevent_false_positives(registry: EntityRegistry, text: str) -> None:
    assert registry.find_tickers(text) == ()


def test_mention_counts_and_strip(registry: EntityRegistry) -> None:
    text = "Nvidia beat estimates. NVDA rose. Intel lagged."
    counts = registry.mention_counts(text)
    assert counts["NVDA"] == 2 and counts["INTC"] == 1
    assert registry.strip(text) == "beat estimates. rose. lagged."


def test_name_and_tickers(registry: EntityRegistry) -> None:
    assert registry.name("AMZN") == "Amazon"
    assert set(registry.tickers) == {"AAPL", "MSFT", "AMZN", "NFLX", "NVDA", "INTC", "IBM"}


def test_custom_registry_is_supported() -> None:
    reg = EntityRegistry([Company("TSLA", "Tesla", ("model 3",))])
    assert reg.find_tickers("Tesla's Model 3") == ("TSLA",)


class TestResolveArticleTickers:
    def test_title_mention_is_enough(self, registry: EntityRegistry) -> None:
        assert registry.resolve_article_tickers(
            "Intel Breakup Mulled", "generic text", ["AAPL"]
        ) == ("INTC",)

    def test_feed_label_is_not_trusted_when_text_is_about_another_company(
        self, registry: EntityRegistry
    ) -> None:
        about = registry.resolve_article_tickers("Chipmaker news", "Intel " * 6 + "Apple", ["AAPL"])
        assert about == ("INTC",)

    def test_repeated_body_mentions_count(self, registry: EntityRegistry) -> None:
        body = "Nvidia. " * 4 + "Intel. " * 4
        assert registry.resolve_article_tickers("Market wrap", body, ["AAPL"]) == ("INTC", "NVDA")

    def test_a_single_passing_mention_does_not_make_an_article_about_a_company(
        self, registry: EntityRegistry
    ) -> None:
        text = "A long market report. " * 5 + "IBM was listed once."
        assert registry.resolve_article_tickers("Market report", text, ["IBM"]) == ("IBM",)

    def test_falls_back_to_feed_labels_when_no_company_is_named(
        self, registry: EntityRegistry
    ) -> None:
        assert registry.resolve_article_tickers(
            "Stocks slip", "Yields climbed.", ["MSFT", "AAPL"]
        ) == ("AAPL", "MSFT")
