"""HTTP request/response models (the public contract of the API)."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator

from newschat.domain import ChatTurn
from newschat.generation.service import ChatResult, MetaEvent, SourceRef


class HistoryMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=4000)


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=2000)
    history: list[HistoryMessage] = Field(default_factory=list, max_length=40)

    @field_validator("message")
    @classmethod
    def _not_blank(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("message must not be blank")
        return v

    def turns(self) -> list[ChatTurn]:
        return [ChatTurn(m.role, m.content) for m in self.history]


class SourceOut(BaseModel):
    number: int
    title: str
    link: str
    tickers: list[str]
    headline_only: bool

    @classmethod
    def from_ref(cls, ref: SourceRef) -> SourceOut:
        return cls(
            number=ref.number,
            title=ref.title,
            link=ref.link,
            tickers=list(ref.tickers),
            headline_only=ref.headline_only,
        )


class PlanOut(BaseModel):
    intent: str
    tickers: list[str]
    standalone_query: str
    is_follow_up: bool


class RetrievalOut(BaseModel):
    coverage: float
    uncovered_terms: list[str]


class ChatResponse(BaseModel):
    answer: str
    answered: bool
    answerer: str
    plan: PlanOut
    retrieval: RetrievalOut
    sources: list[SourceOut]
    citations: list[SourceOut]
    invalid_citations: list[int]

    @classmethod
    def from_result(cls, r: ChatResult) -> ChatResponse:
        return cls(
            answer=r.answer,
            answered=r.answered,
            answerer=r.answerer,
            plan=PlanOut(
                intent=r.plan.intent.value,
                tickers=list(r.plan.tickers),
                standalone_query=r.plan.standalone_query,
                is_follow_up=r.plan.is_follow_up,
            ),
            retrieval=RetrievalOut(
                coverage=round(r.coverage, 3), uncovered_terms=list(r.uncovered_terms)
            ),
            sources=[SourceOut.from_ref(s) for s in r.sources],
            citations=[SourceOut.from_ref(s) for s in r.citations],
            invalid_citations=list(r.invalid_citations),
        )


class MetaOut(BaseModel):
    """First event of a streamed response: how the question was understood + sources."""

    answerer: str
    plan: PlanOut
    retrieval: RetrievalOut
    sources: list[SourceOut]

    @classmethod
    def from_event(cls, e: MetaEvent) -> MetaOut:
        return cls(
            answerer=e.answerer,
            plan=PlanOut(
                intent=e.plan.intent.value,
                tickers=list(e.plan.tickers),
                standalone_query=e.plan.standalone_query,
                is_follow_up=e.plan.is_follow_up,
            ),
            retrieval=RetrievalOut(
                coverage=round(e.coverage, 3), uncovered_terms=list(e.uncovered_terms)
            ),
            sources=[SourceOut.from_ref(s) for s in e.sources],
        )


class HealthResponse(BaseModel):
    status: Literal["ok"]
    answerer: str
    articles: int
    chunks: int
