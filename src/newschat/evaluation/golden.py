"""Golden-set loading. Relevant articles are named by title prefix, so the file stays
human-readable and survives changes to article ids / link formats."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from newschat.domain import ChatTurn, Corpus


class GoldenSetError(ValueError):
    """The golden set is malformed or refers to articles that don't exist."""


@dataclass(frozen=True, slots=True)
class GoldenCase:
    id: str
    category: str
    question: str
    history: tuple[ChatTurn, ...] = ()
    relevant_titles: tuple[str, ...] = ()
    must_cover_tickers: tuple[str, ...] = ()
    expect_answered: bool = True


def load_golden(path: Path) -> list[GoldenCase]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return [
            GoldenCase(
                id=c["id"],
                category=c["category"],
                question=c["question"],
                history=tuple(ChatTurn(h["role"], h["content"]) for h in c.get("history", [])),
                relevant_titles=tuple(c.get("relevant_titles", ())),
                must_cover_tickers=tuple(c.get("must_cover_tickers", ())),
                expect_answered=c.get("expect_answered", True),
            )
            for c in payload["cases"]
        ]
    except (OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
        raise GoldenSetError(f"cannot load golden set {path}: {exc!r}") from exc


def resolve_relevant(case: GoldenCase, corpus: Corpus) -> frozenset[str]:
    """Map a case's title prefixes to article ids; each prefix must match exactly one."""
    ids: set[str] = set()
    for prefix in case.relevant_titles:
        matches = [
            a.id
            for a in corpus.articles.values()
            if a.title.casefold().startswith(prefix.casefold())
        ]
        if len(matches) != 1:
            raise GoldenSetError(
                f"case {case.id!r}: title prefix {prefix!r} "
                f"matched {len(matches)} articles (need 1)"
            )
        ids.add(matches[0])
    return frozenset(ids)
