"""Answerers turn retrieved sources into an answer stream.

Two implementations share one interface:

* ``LLMAnswerer``       - synthesises a cited summary with a language model.
* ``ExtractiveAnswerer`` - deterministic, offline: quotes the most relevant sentences with
  citations. It keeps the app fully functional without an API key and gives the test suite
  a real (non-mock) answerer to exercise the whole pipeline end to end.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Protocol

from newschat.domain import ChatTurn, Intent, QueryPlan
from newschat.generation.llm import LLMClient
from newschat.generation.prompts import SYSTEM_PROMPT, build_messages
from newschat.generation.sources import Source
from newschat.ingestion.chunking import split_sentences
from newschat.retrieval.tokenize import tokenize


@dataclass(frozen=True, slots=True)
class GenerationRequest:
    question: str
    plan: QueryPlan
    history: tuple[ChatTurn, ...]
    sources: tuple[Source, ...]
    uncovered_terms: tuple[str, ...]


class Answerer(Protocol):
    name: str

    def stream(self, request: GenerationRequest) -> Iterator[str]: ...


class LLMAnswerer:
    def __init__(self, client: LLMClient, max_tokens: int = 900, max_history: int = 8) -> None:
        self._client = client
        self._max_tokens = max_tokens
        self._max_history = max_history
        self.name = client.name

    def stream(self, request: GenerationRequest) -> Iterator[str]:
        messages = build_messages(
            request.question,
            request.history,
            request.sources,
            request.uncovered_terms,
            self._max_history,
        )
        yield from self._client.stream(
            system=SYSTEM_PROMPT, messages=messages, max_tokens=self._max_tokens
        )


_MIN_SENTENCE_CHARS = 40
_SENTENCES_PER_SOURCE = 2
_MIN_TERM_OVERLAP = 2  # a body sentence must share this many question terms to be quoted


class ExtractiveAnswerer:
    name = "extractive"

    def stream(self, request: GenerationRequest) -> Iterator[str]:
        yield (
            "Most relevant excerpts from the provided articles "
            "(extractive mode: set ANTHROPIC_API_KEY for a synthesised summary):\n"
        )
        query_terms = set(tokenize(request.plan.standalone_query))
        lines = {s.number: self._excerpts(s, query_terms) for s in request.sources}
        if not any(lines.values()):  # nothing overlaps strongly: fall back to the headlines
            lines = {s.number: [s.article.title] for s in request.sources}

        if request.plan.intent is Intent.COMPARISON:
            for ticker in request.plan.tickers:
                yield f"\n{ticker}:\n"
                relevant = [
                    s for s in request.sources if ticker in s.article.tickers and lines[s.number]
                ]
                if not relevant:
                    yield "- No articles about this company were found.\n"
                for source in relevant:
                    yield from self._bullets(source, lines[source.number])
        else:
            yield "\n"
            for source in request.sources:
                yield from self._bullets(source, lines[source.number])

    @staticmethod
    def _bullets(source: Source, sentences: list[str]) -> Iterator[str]:
        tag = " (headline only)" if source.article.headline_only else ""
        for sentence in sentences:
            yield f"- {sentence} [{source.number}]{tag}\n"

    @staticmethod
    def _excerpts(source: Source, query_terms: set[str]) -> list[str]:
        """Sentences that genuinely overlap the question (empty if none do)."""
        if source.article.headline_only:
            return [source.article.title]
        needed = min(_MIN_TERM_OVERLAP, len(query_terms))
        scored = [
            (len(query_terms & set(tokenize(sentence))), i, sentence)
            for i, sentence in enumerate(split_sentences(source.text))
            if len(sentence) >= _MIN_SENTENCE_CHARS
        ]
        relevant = [item for item in scored if item[0] >= max(needed, 1)]
        best = sorted(
            sorted(relevant, key=lambda item: (-item[0], item[1]))[:_SENTENCES_PER_SOURCE],
            key=lambda item: item[1],
        )
        return [_tidy(sentence) for _, _, sentence in best]


def _tidy(sentence: str) -> str:
    return re.sub(r"\s+", " ", sentence).strip()
