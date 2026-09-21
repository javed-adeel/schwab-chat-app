"""Dense (semantic) retrieval behind a swappable ``Embedder`` interface.

The default embedder is latent semantic analysis (TF-IDF + truncated SVD): it captures
term co-occurrence ("chipmaker" ~ "semiconductor"), is fully deterministic, needs no model
download and no network, so the service and its tests run anywhere. A neural embedder can
be dropped in by implementing the two-method ``Embedder`` protocol.
"""

from __future__ import annotations

from collections.abc import Collection, Sequence
from typing import Protocol

import numpy as np
import numpy.typing as npt
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import Normalizer

from newschat.retrieval.tokenize import tokenize

Matrix = npt.NDArray[np.float64]


class Embedder(Protocol):
    def fit(self, texts: Sequence[str]) -> None: ...

    def embed(self, texts: Sequence[str]) -> Matrix:
        """Return an ``(n, d)`` matrix of L2-normalised row vectors."""
        ...


class LsaEmbedder:
    def __init__(self, n_components: int = 128, random_state: int = 0) -> None:
        self._n_components = n_components
        self._random_state = random_state
        self._vectorizer: TfidfVectorizer | None = None
        self._svd: TruncatedSVD | None = None
        self._normalizer = Normalizer(copy=False)

    def fit(self, texts: Sequence[str]) -> None:
        vectorizer = TfidfVectorizer(
            tokenizer=tokenize,
            lowercase=False,
            token_pattern=None,
            ngram_range=(1, 2),
            sublinear_tf=True,
            min_df=2 if len(texts) >= 20 else 1,
        )
        tfidf = vectorizer.fit_transform(texts)
        n_components = max(1, min(self._n_components, tfidf.shape[0] - 1, tfidf.shape[1] - 1))
        self._svd = TruncatedSVD(n_components=n_components, random_state=self._random_state)
        self._svd.fit(tfidf)
        self._vectorizer = vectorizer

    def embed(self, texts: Sequence[str]) -> Matrix:
        if self._vectorizer is None or self._svd is None:
            raise RuntimeError("embedder must be fitted before use")
        reduced = self._svd.transform(self._vectorizer.transform(texts))
        return np.asarray(self._normalizer.transform(reduced), dtype=np.float64)


class DenseIndex:
    def __init__(self, embedder: Embedder, texts: Sequence[str]) -> None:
        self._embedder = embedder
        embedder.fit(texts)
        self._matrix = embedder.embed(texts)

    def search(
        self, query: str, top_k: int, allowed: Collection[int] | None = None
    ) -> list[tuple[int, float]]:
        similarities = self._matrix @ self._embedder.embed([query])[0]
        if allowed is not None:
            mask = np.full(similarities.shape, -np.inf)
            mask[list(allowed)] = 0.0
            similarities = similarities + mask
        order = np.argsort(-similarities, kind="stable")[:top_k]
        return [(int(i), float(similarities[i])) for i in order if similarities[i] > 0.0]
