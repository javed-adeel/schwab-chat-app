"""Turns a raw question (plus chat history) into a ``QueryPlan``.

Deterministic and dependency-free by design: it is fast, free, unit-testable, and its
decisions are inspectable in the API response. It handles three things an embedding
search alone gets wrong in a chat setting:

* which companies the question is about (so retrieval can be filtered or balanced),
* what *kind* of question it is (one company, a comparison, or market-wide), and
* follow-ups such as "What about Amazon?" or "What are analysts saying about its price
  target?", which are meaningless without the earlier turns.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from newschat.domain import ChatTurn, Intent, QueryPlan
from newschat.ingestion.entities import EntityRegistry
from newschat.retrieval.tokenize import tokenize

_FOLLOW_UP_LEAD = re.compile(
    r"^\s*(?:(?:and|so|ok|okay|now|then|also)[\s,]+)*"
    r"(?:(?:what|how)\s+about|same\s+for|and\s+for|and|also)\b[\s,:]*",
    re.IGNORECASE,
)
_ANAPHORA = re.compile(
    r"\b(?:it|its|it's|they|their|them|this|that|those|these|he|his|she|her"
    r"|the\s+(?:company|stock|firm|shares))\b",
    re.IGNORECASE,
)
_ANAPHORA_WORDS = frozenset(
    "it its they their them this that those these he his she her company stock firm shares".split()
)
_MIN_TOPIC_TERMS = 2  # fewer than this and the question needs its topic from earlier turns


class QueryAnalyzer:
    def __init__(self, registry: EntityRegistry) -> None:
        self._registry = registry

    def analyze(self, question: str, history: Sequence[ChatTurn] = ()) -> QueryPlan:
        question = question.strip()
        tickers = self._registry.find_tickers(question)
        prior_user_turns = [t.content for t in history if t.role == "user"]
        follow_up = bool(prior_user_turns) and self._is_follow_up(question, bool(tickers))

        standalone = question
        if follow_up:
            inherited = self._inherited_tickers(prior_user_turns)
            if not tickers:
                tickers = inherited
                names = " ".join(self._registry.name(t) for t in inherited)
                standalone = f"{standalone} {names}".strip()
            if self._needs_topic(question):
                topic = self._inherited_topic(prior_user_turns)
                standalone = f"{standalone} {topic}".strip()

        return QueryPlan(
            question=question,
            standalone_query=standalone,
            intent=self._intent(tickers),
            tickers=tickers,
            is_follow_up=follow_up,
        )

    # ------------------------------------------------------------------ internals

    @staticmethod
    def _intent(tickers: Sequence[str]) -> Intent:
        if len(tickers) >= 2:
            return Intent.COMPARISON
        return Intent.COMPANY if tickers else Intent.MARKET

    @staticmethod
    def _is_follow_up(question: str, has_tickers: bool) -> bool:
        if _FOLLOW_UP_LEAD.match(question):
            return True
        return not has_tickers and bool(_ANAPHORA.search(question))

    def _own_topic_terms(self, text: str) -> list[str]:
        """Content terms a question brings itself, ignoring lead-ins, companies, pronouns."""
        bare = self._registry.strip(_FOLLOW_UP_LEAD.sub("", text))
        return [t for t in tokenize(bare) if t not in _ANAPHORA_WORDS]

    def _needs_topic(self, question: str) -> bool:
        return len(self._own_topic_terms(question)) < _MIN_TOPIC_TERMS

    def _inherited_tickers(self, prior_user_turns: Sequence[str]) -> tuple[str, ...]:
        for turn in reversed(prior_user_turns):
            found = self._registry.find_tickers(turn)
            if found:
                return found
        return ()

    def _inherited_topic(self, prior_user_turns: Sequence[str]) -> str:
        """The most recent earlier question that actually states a topic, sans companies."""
        for turn in reversed(prior_user_turns):
            bare = self._registry.strip(_FOLLOW_UP_LEAD.sub("", turn))
            if len(self._own_topic_terms(turn)) >= _MIN_TOPIC_TERMS:
                return bare
        return ""
