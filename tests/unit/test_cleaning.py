from __future__ import annotations

import pytest

from newschat.ingestion.cleaning import clean_text, is_headline_only


def test_strips_trailing_view_comments_and_reports_the_rule() -> None:
    result = clean_text("Apple rose 3% today. View Comments")
    assert result.text == "Apple rose 3% today."
    assert "view_comments" in result.rules_applied


def test_paywall_teaser_is_flagged_and_boilerplate_removed() -> None:
    raw = (
        "Apple's launch of the iPhone SE this week has most likely been taken "
        "PREMIUM Upgrade to read this MT Newswires article and get so much more. "
        "A Silver or Gold subscription plan is required to access premium news articles. "
        "Upgrade Already have a subscription? Sign in"
    )
    result = clean_text(raw)
    assert result.paywalled
    assert "PREMIUM" not in result.text and "Sign in" not in result.text
    assert result.text.startswith("Apple's launch of the iPhone SE")
    assert is_headline_only(result, min_content_chars=400)


@pytest.mark.parametrize(
    "opening",
    [
        "Why are we interested in the stocks that hedge funds pile into?",
        "At Insider Monkey we are obsessed with the stocks that hedge funds pile into.",
    ],
)
def test_both_newsletter_pitch_variants_are_removed_but_content_after_survives(
    opening: str,
) -> None:
    raw = (
        f"Before. {opening} The reason is simple: our research shows we can outperform. "
        "Our newsletter has returned 275% (see more details here). Story Continues "
        "Apple is a technology company."
    )
    text = clean_text(raw).text
    assert "hedge funds" not in text and "Story Continues" not in text
    assert text.startswith("Before.") and text.endswith("Apple is a technology company.")


def test_zacks_promo_is_removed_through_to_the_end() -> None:
    raw = (
        "Apple beat estimates. Want the latest recommendations from Zacks Investment Research? "
        "Today, you can download 7 Best Stocks. Apple Inc. (AAPL) : Free Stock Analysis Report "
        "To read this article on Zacks.com click here. Zacks Investment Research View Comments"
    )
    result = clean_text(raw)
    assert result.text == "Apple beat estimates."
    assert "zacks_promo" in result.rules_applied


def test_yahoo_video_roll_markers_are_removed() -> None:
    raw = "Markets rally. Here's why. Yahoo Finance Video • 26 minutes ago 00:57 Amazon Inspire View Comments"
    text = clean_text(raw).text
    assert "Yahoo Finance Video" not in text and "26 minutes" not in text
    assert "Amazon Inspire" in text


def test_normalises_unicode_and_whitespace() -> None:
    assert clean_text("Nvidia\xa0rose \n\n  5%").text == "Nvidia rose 5%"


def test_clean_text_is_idempotent() -> None:
    raw = "Text here. Disclosure: None. View Comments"
    once = clean_text(raw).text
    assert clean_text(once).text == once


def test_plain_text_passes_through_untouched() -> None:
    result = clean_text("Intel shares rose on Tuesday.")
    assert result.text == "Intel shares rose on Tuesday."
    assert result.rules_applied == () and not result.paywalled


def test_short_bodies_are_headline_only_even_without_a_paywall() -> None:
    assert is_headline_only(clean_text("Tiny."), min_content_chars=50)
    assert not is_headline_only(clean_text("x " * 100), min_content_chars=50)
