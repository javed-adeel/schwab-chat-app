"""Test doubles."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass

from newschat.domain import ChatTurn


@dataclass(frozen=True)
class LLMCall:
    system: str
    messages: list[ChatTurn]
    max_tokens: int


class FakeLLM:
    """Scripted LLMClient that records what it was asked."""

    name = "fake-llm"

    def __init__(self, pieces: Sequence[str] = ("Answer [1].",), error: Exception | None = None):
        self.pieces = list(pieces)
        self.error = error
        self.calls: list[LLMCall] = []

    def stream(
        self, *, system: str, messages: Sequence[ChatTurn], max_tokens: int
    ) -> Iterator[str]:
        self.calls.append(LLMCall(system, list(messages), max_tokens))
        if self.error is not None:
            raise self.error
        yield from self.pieces
