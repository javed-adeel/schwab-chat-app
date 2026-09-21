"""Text cleaning for scraped news articles.

Every rule below was derived from patterns actually present in the dataset (see the
README's data-quality section). Rules are named so the ingestion report can say exactly
how many articles each one touched.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Rule:
    name: str
    pattern: re.Pattern[str]
    replacement: str = " "


def _rule(name: str, pattern: str, flags: int = re.IGNORECASE) -> Rule:
    return Rule(name, re.compile(pattern, flags))


# Order matters: bulky promotional blocks are removed before short trailing markers.
RULES: tuple[Rule, ...] = (
    _rule("paywall", r"PREMIUM\s+Upgrade to read.*?(?:Sign in|$)", re.IGNORECASE | re.DOTALL),
    _rule(
        "hedge_fund_pitch",
        r"(?:Why are we interested in|At Insider Monkey we are obsessed with) "
        r"the stocks that hedge funds pile into[.?].*?\(see more details here\)\.?",
        re.IGNORECASE | re.DOTALL,
    ),
    _rule("read_next_links", r"READ NEXT:.*$", re.DOTALL),
    _rule("story_continues", r"\bStory Continues\b"),
    _rule(
        "zacks_promo",
        r"Want the latest recommendations from Zacks Investment Research\?.*$",
        re.IGNORECASE | re.DOTALL,
    ),
    _rule(
        "yahoo_video_roll",
        r"Yahoo Finance Video\s*•\s*\d+\s+\w+\s+ago(?:\s+\d{1,2}:\d{2}(?::\d{2})?)?",
    ),
    _rule("view_comments", r"\bView Comments\b"),
    _rule("insider_monkey_footer", r"Disclosure:\s*None\.?"),
    _rule("insider_monkey_footer", r"This article is originally published at\s*Insider Monkey\.?"),
    _rule("motley_fool_attribution", r"was originally published by The Motley Fool"),
    _rule("byline", r"This post was written by [A-Z][\w.'\-]*(?: [A-Z][\w.'\-]*){0,3}\.?"),
    _rule(
        "benzinga_disclaimer",
        r"Benzinga does not provide investment advice\.(?:\s*All rights reserved\.)?",
    ),
    _rule("yahoo_footer", r"Read the latest financial and business news from Yahoo Finance"),
    _rule("continue_reading", r"\bContinue Reading\s*$"),
    _rule("promo_free_link", r"[^.]*available to you FREE via this link\.?"),
)

_WS = re.compile(r"\s+")


@dataclass(frozen=True, slots=True)
class CleanedText:
    text: str
    rules_applied: tuple[str, ...]
    paywalled: bool


def clean_text(raw: str, rules: tuple[Rule, ...] = RULES) -> CleanedText:
    """Normalise unicode/whitespace and strip publisher boilerplate."""
    text = unicodedata.normalize("NFKC", raw)
    applied: list[str] = []
    for rule in rules:
        text, n = rule.pattern.subn(rule.replacement, text)
        if n and rule.name not in applied:
            applied.append(rule.name)
    return CleanedText(
        text=_WS.sub(" ", text).strip(),
        rules_applied=tuple(applied),
        paywalled="paywall" in applied,
    )


def is_headline_only(cleaned: CleanedText, min_content_chars: int) -> bool:
    """True when only the headline is trustworthy (paywalled or near-empty body)."""
    return cleaned.paywalled or len(cleaned.text) < min_content_chars
