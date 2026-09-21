"""Citation parsing and validation: the guard against fabricated references."""

from __future__ import annotations

import re
from collections.abc import Collection
from dataclasses import dataclass

_CITATION = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")


@dataclass(frozen=True, slots=True)
class CitationCheck:
    cited: tuple[int, ...]  # valid source numbers the answer cites, in order of appearance
    invalid: tuple[int, ...]  # cited numbers that match no provided source


def extract_citation_numbers(text: str) -> list[int]:
    numbers: list[int] = []
    for match in _CITATION.finditer(text):
        numbers.extend(int(n) for n in match.group(1).replace(" ", "").split(","))
    return numbers


def check_citations(text: str, valid_numbers: Collection[int]) -> CitationCheck:
    valid = set(valid_numbers)
    ordered = list(dict.fromkeys(extract_citation_numbers(text)))
    return CitationCheck(
        cited=tuple(n for n in ordered if n in valid),
        invalid=tuple(n for n in ordered if n not in valid),
    )


def strip_citations(text: str) -> str:
    """Remove ``[n]`` markers (used on earlier assistant turns, whose numbering is stale)."""
    return re.sub(r"\s*\[\d+(?:\s*,\s*\d+)*\]", "", text).strip()
