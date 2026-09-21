"""Company entity registry.

The dataset's ``ticker`` field records which *feed* an article came from, not what it is
about (37 articles mention several of the 7 tickers; many feed items are unrelated). This
module detects the companies an article or a question is genuinely about, and is shared by
ingestion and query analysis so both sides agree on what "Apple" means.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Company:
    ticker: str
    name: str
    aliases: tuple[str, ...] = ()


DEFAULT_COMPANIES: tuple[Company, ...] = (
    Company("AAPL", "Apple", ("iphone", "ipad", "macbook", "tim cook", "apple intelligence")),
    Company("MSFT", "Microsoft", ("azure", "satya nadella", "copilot")),
    Company("AMZN", "Amazon", ("aws", "amazon web services", "andy jassy", "prime video")),
    Company("NFLX", "Netflix"),
    Company("NVDA", "Nvidia", ("jensen huang", "geforce", "blackwell")),
    Company("INTC", "Intel", ("pat gelsinger",)),
    Company("IBM", "IBM", ("international business machines", "watsonx")),
)

# An article is "about" a company if it is named in the title, or mentioned repeatedly in
# the body relative to the most-mentioned company.
_TITLE_WEIGHT = 5
_MIN_BODY_MENTIONS = 3
_RELATIVE_THRESHOLD = 0.25

_WS = re.compile(r"\s+")


class EntityRegistry:
    def __init__(self, companies: Iterable[Company] = DEFAULT_COMPANIES) -> None:
        self._companies = {c.ticker: c for c in companies}
        self._alias_to_ticker: dict[str, str] = {}
        for company in self._companies.values():
            for alias in (company.ticker, company.name, *company.aliases):
                self._alias_to_ticker[alias.lower()] = company.ticker
        aliases = sorted(self._alias_to_ticker, key=len, reverse=True)
        self._regex = re.compile(
            r"\b(?:" + "|".join(re.escape(a) for a in aliases) + r")\b", re.IGNORECASE
        )

    @property
    def tickers(self) -> tuple[str, ...]:
        return tuple(self._companies)

    def name(self, ticker: str) -> str:
        return self._companies[ticker].name

    def _ticker_of(self, match: re.Match[str]) -> str:
        return self._alias_to_ticker[_WS.sub(" ", match.group(0).lower())]

    def mention_counts(self, text: str) -> Counter[str]:
        return Counter(self._ticker_of(m) for m in self._regex.finditer(text))

    def find_tickers(self, text: str) -> tuple[str, ...]:
        """Tickers mentioned in ``text``, in order of first appearance."""
        return tuple(dict.fromkeys(self._ticker_of(m) for m in self._regex.finditer(text)))

    def strip(self, text: str) -> str:
        """Remove every company mention (used to isolate the *topic* of a question)."""
        return _WS.sub(" ", self._regex.sub(" ", text)).strip()

    def resolve_article_tickers(
        self, title: str, text: str, source_tickers: Sequence[str]
    ) -> tuple[str, ...]:
        """Which companies an article is about.

        Falls back to the feed label when the text names no known company, so generic
        market wrap-ups listed under a ticker stay reachable through that ticker.
        """
        title_counts = self.mention_counts(title)
        body_counts = self.mention_counts(text)
        score = {
            t: _TITLE_WEIGHT * title_counts[t] + body_counts[t]
            for t in title_counts.keys() | body_counts.keys()
        }
        if not score:
            return tuple(sorted(source_tickers))
        threshold = max(_MIN_BODY_MENTIONS, _RELATIVE_THRESHOLD * max(score.values()))
        about = sorted(t for t, s in score.items() if title_counts[t] > 0 or s >= threshold)
        return tuple(about) if about else tuple(sorted(source_tickers))
