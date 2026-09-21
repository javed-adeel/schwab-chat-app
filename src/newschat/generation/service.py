"""ChatService: orchestrates analysis -> retrieval -> grounded generation -> citation check.

It is the only place that knows the whole flow, and it depends on abstractions (an
``Answerer``, a retriever, an analyzer), so each stage can be swapped or faked in tests.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field

from newschat.domain import ChatTurn, QueryPlan
from newschat.generation.answerers import Answerer, GenerationRequest
from newschat.generation.grounding import check_citations
from newschat.generation.sources import Source, build_sources
from newschat.retrieval.query_analysis import QueryAnalyzer
from newschat.retrieval.retriever import HybridRetriever, RetrievalResult

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class SourceRef:
    number: int
    title: str
    link: str
    tickers: tuple[str, ...]
    headline_only: bool

    @classmethod
    def from_source(cls, s: Source) -> SourceRef:
        a = s.article
        return cls(s.number, a.title, a.link, a.tickers, a.headline_only)


@dataclass(frozen=True, slots=True)
class ChatResult:
    answer: str
    answered: bool  # False => the corpus could not support an answer; no model was called
    plan: QueryPlan
    coverage: float
    uncovered_terms: tuple[str, ...]
    sources: tuple[SourceRef, ...]  # everything the model was shown
    citations: tuple[SourceRef, ...]  # the subset the answer actually cites
    invalid_citations: tuple[int, ...] = field(default=())
    answerer: str = ""


@dataclass(frozen=True, slots=True)
class MetaEvent:
    plan: QueryPlan
    coverage: float
    uncovered_terms: tuple[str, ...]
    sources: tuple[SourceRef, ...]
    answerer: str


@dataclass(frozen=True, slots=True)
class DeltaEvent:
    text: str


@dataclass(frozen=True, slots=True)
class FinalEvent:
    result: ChatResult


ChatEvent = MetaEvent | DeltaEvent | FinalEvent


class ChatService:
    def __init__(
        self,
        analyzer: QueryAnalyzer,
        retriever: HybridRetriever,
        answerer: Answerer,
        supported_tickers: Sequence[str],
    ) -> None:
        self._analyzer = analyzer
        self._retriever = retriever
        self._answerer = answerer
        self._supported = tuple(supported_tickers)

    @property
    def answerer_name(self) -> str:
        return self._answerer.name

    def chat(self, message: str, history: Sequence[ChatTurn] = ()) -> ChatResult:
        final: ChatResult | None = None
        for event in self.stream(message, history):
            if isinstance(event, FinalEvent):
                final = event.result
        assert final is not None  # stream() always ends with a FinalEvent
        return final

    def stream(self, message: str, history: Sequence[ChatTurn] = ()) -> Iterator[ChatEvent]:
        started = time.perf_counter()
        history = tuple(history)
        plan = self._analyzer.analyze(message, history)
        retrieval = self._retriever.retrieve(plan)

        if not retrieval.sufficient:
            yield from self._decline(plan, retrieval)
            return

        sources = build_sources(retrieval.passages)
        refs = tuple(SourceRef.from_source(s) for s in sources)
        yield MetaEvent(
            plan, retrieval.coverage, retrieval.uncovered_terms, refs, self._answerer.name
        )

        request = GenerationRequest(
            question=message,
            plan=plan,
            history=history,
            sources=sources,
            uncovered_terms=retrieval.uncovered_terms,
        )
        parts: list[str] = []
        for piece in self._answerer.stream(request):
            parts.append(piece)
            yield DeltaEvent(piece)

        answer = "".join(parts).strip()
        check = check_citations(answer, {s.number for s in sources})
        by_number = {r.number: r for r in refs}
        if check.invalid:
            logger.warning("model cited unknown sources", extra={"invalid": list(check.invalid)})
        logger.info(
            "chat answered",
            extra={
                "intent": plan.intent.value,
                "tickers": list(plan.tickers),
                "follow_up": plan.is_follow_up,
                "coverage": round(retrieval.coverage, 3),
                "sources": len(sources),
                "cited": len(check.cited),
                "answerer": self._answerer.name,
                "latency_ms": round((time.perf_counter() - started) * 1000),
            },
        )
        yield FinalEvent(
            ChatResult(
                answer=answer,
                answered=True,
                plan=plan,
                coverage=retrieval.coverage,
                uncovered_terms=retrieval.uncovered_terms,
                sources=refs,
                citations=tuple(by_number[n] for n in check.cited),
                invalid_citations=check.invalid,
                answerer=self._answerer.name,
            )
        )

    # ------------------------------------------------------------------ internals

    def _decline(self, plan: QueryPlan, retrieval: RetrievalResult) -> Iterator[ChatEvent]:
        """Out-of-scope: answer honestly without spending (or risking) a model call."""
        logger.info(
            "chat declined",
            extra={
                "coverage": round(retrieval.coverage, 3),
                "uncovered": retrieval.uncovered_terms,
            },
        )
        text = (
            "I couldn't find anything relevant to that in the news articles I have. "
            f"They cover {', '.join(self._supported)}. Try asking about a company's recent "
            "news, analyst ratings, earnings, or a comparison between two companies."
        )
        yield MetaEvent(
            plan, retrieval.coverage, retrieval.uncovered_terms, (), self._answerer.name
        )
        yield DeltaEvent(text)
        yield FinalEvent(
            ChatResult(
                answer=text,
                answered=False,
                plan=plan,
                coverage=retrieval.coverage,
                uncovered_terms=retrieval.uncovered_terms,
                sources=(),
                citations=(),
                answerer=self._answerer.name,
            )
        )
