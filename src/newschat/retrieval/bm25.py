"""A small, dependency-free Okapi BM25 index."""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from collections.abc import Collection, Sequence


class BM25Index:
    def __init__(self, docs: Sequence[Sequence[str]], k1: float = 1.5, b: float = 0.75) -> None:
        self._k1 = k1
        self._b = b
        self._n = len(docs)
        self._lengths = [len(d) for d in docs]
        self._avgdl = (sum(self._lengths) / self._n) if self._n else 0.0
        self._postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for doc_id, doc in enumerate(docs):
            for term, tf in Counter(doc).items():
                self._postings[term].append((doc_id, tf))

    def __len__(self) -> int:
        return self._n

    def document_frequency(self, term: str) -> int:
        return len(self._postings.get(term, ()))

    def idf(self, term: str) -> float:
        """Robertson-Sparck Jones idf, floored at 0. Unseen terms get the maximum idf."""
        df = self.document_frequency(term)
        return math.log(1.0 + (self._n - df + 0.5) / (df + 0.5))

    def search(
        self,
        query_terms: Sequence[str],
        top_k: int,
        allowed: Collection[int] | None = None,
    ) -> list[tuple[int, float]]:
        """Top ``top_k`` ``(doc_id, score)`` pairs, best first, optionally restricted."""
        scores: dict[int, float] = defaultdict(float)
        for term in set(query_terms):
            postings = self._postings.get(term)
            if not postings:
                continue
            idf = self.idf(term)
            for doc_id, tf in postings:
                if allowed is not None and doc_id not in allowed:
                    continue
                norm = 1.0 - self._b + self._b * self._lengths[doc_id] / self._avgdl
                scores[doc_id] += idf * tf * (self._k1 + 1.0) / (tf + self._k1 * norm)
        ranked = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
        return ranked[:top_k]
