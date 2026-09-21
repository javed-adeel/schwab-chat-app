"""Tokenisation shared by the lexical index, the dense index and query analysis."""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache

import snowballstemmer

_TOKEN = re.compile(r"[a-z0-9]+")
_STEMMER = snowballstemmer.stemmer("english")

# Function words plus conversational filler that carries no retrieval signal. Question
# scaffolding ("tell me about", "compare") is included so it neither dilutes BM25 scores
# nor counts against query coverage.
STOPWORDS = frozenset(
    """
    a about above after again all also am an and any are as at be because been before being
    below between both but by can could did do does doing down during each few for from
    further had has have having he her here hers him his how i if in into is it its just me
    more most my no nor not of off on once only or other our out over own s same she should
    so some such t than that the their them then there these they this those through to too
    under until up very was we were what whats when where which while who whom why will with
    would you your
    tell give show explain describe summarize summarise summary overview please
    compare compares compared comparison versus vs difference differences
    latest recent recently news lately today currently current happening going
    """.split()
)


@dataclass(frozen=True, slots=True)
class Term:
    stem: str
    surface: str


@lru_cache(maxsize=65536)
def _stem(word: str) -> str:
    return str(_STEMMER.stemWord(word))


def terms(text: str) -> list[Term]:
    """Lower-cased, stop-word-free, stemmed terms (keeping the surface form for display)."""
    out: list[Term] = []
    for word in _TOKEN.findall(text.lower()):
        if word in STOPWORDS or (len(word) < 2 and not word.isdigit()):
            continue
        out.append(Term(_stem(word), word))
    return out


def tokenize(text: str) -> list[str]:
    return [t.stem for t in terms(text)]
