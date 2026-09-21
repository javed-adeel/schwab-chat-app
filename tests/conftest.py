"""Shared fixtures.

Two corpora are used on purpose:

* ``mini_*``  - a tiny hand-built corpus where every property under test is obvious.
* the real dataset (session-scoped, built once) - for behaviour that only shows up on real
  scraped text and for end-to-end/API tests.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from newschat.bootstrap import Application, build_corpus, build_retriever
from newschat.config import Settings
from newschat.domain import Corpus
from newschat.generation.answerers import ExtractiveAnswerer
from newschat.generation.llm import LLMError
from newschat.generation.service import ChatService
from newschat.ingestion.entities import EntityRegistry
from newschat.ingestion.loader import RawArticle
from newschat.ingestion.pipeline import IngestionConfig, IngestionPipeline
from newschat.retrieval.query_analysis import QueryAnalyzer
from newschat.retrieval.retriever import HybridRetriever, RetrievalConfig

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def fake_llm_error() -> LLMError:
    return LLMError("boom")


@pytest.fixture(scope="session")
def settings() -> Settings:
    return Settings(
        _env_file=None, llm_provider="extractive", data_path=ROOT / "data" / "stock_news.json"
    )


@pytest.fixture(scope="session")
def registry() -> EntityRegistry:
    return EntityRegistry()


@pytest.fixture(scope="session")
def corpus(settings: Settings, registry: EntityRegistry) -> Corpus:
    return build_corpus(settings, registry)


@pytest.fixture(scope="session")
def retriever(corpus: Corpus, settings: Settings) -> HybridRetriever:
    return build_retriever(corpus, settings)


@pytest.fixture(scope="session")
def analyzer(registry: EntityRegistry) -> QueryAnalyzer:
    return QueryAnalyzer(registry)


@pytest.fixture(scope="session")
def application(
    settings: Settings, corpus: Corpus, retriever: HybridRetriever, analyzer: QueryAnalyzer
) -> Application:
    service = ChatService(analyzer, retriever, ExtractiveAnswerer(), corpus.tickers)
    return Application(settings=settings, corpus=corpus, service=service)


# ---------------------------------------------------------------- mini corpus


def _raw(title: str, link: str, text: str) -> RawArticle:
    return RawArticle(title=title, link=link, full_text=text)


LONG_INTEL = (
    "Intel shares jumped after a report that rivals may split the chipmaker. "
    "Analysts said a breakup could unlock value for Intel investors. "
    "The foundry business is widely seen as the hardest part to separate. "
) * 4
LONG_NVDA = (
    "Nvidia reported record data center revenue driven by demand for AI accelerators. "
    "Nvidia expects the Blackwell ramp to continue through the year. "
    "Management said Nvidia supply constraints are easing gradually. "
) * 4
LONG_APPLE = (
    "Apple unveiled a new iPhone with on-device AI features and a faster modem. "
    "Apple said the launch would arrive in stores next week. "
    "Apple services revenue also grew during the quarter. "
) * 4


@pytest.fixture
def mini_raw() -> dict[str, list[RawArticle]]:
    return {
        "INTC": [
            _raw("Intel breakup mulled", "https://finance.yahoo.com/news/intel-1.html", LONG_INTEL),
            _raw(
                "Evercore trims Intel target",
                "https://finance.yahoo.com/news/intel-2.html",
                "Evercore cut its Intel target. PREMIUM Upgrade to read this article and get more. "
                "A plan is required. Sign in",
            ),
        ],
        "NVDA": [
            _raw(
                "Nvidia posts record quarter",
                "https://finance.yahoo.com/news/nvda-1.html",
                LONG_NVDA + " View Comments",
            ),
            # same article as the INTC one, different URL decoration + shorter text
            _raw(
                "Intel breakup mulled",
                "https://finance.yahoo.com/news/intel-1.html?src=nv",
                "short",
            ),
        ],
        "AAPL": [
            _raw(
                "Apple teases new iPhone", "https://finance.yahoo.com/news/aapl-1.html", LONG_APPLE
            ),
            # generic wrap-up that names no tracked company: keeps its feed label
            _raw(
                "Stocks slip as yields climb",
                "https://finance.yahoo.com/news/mkt-1.html",
                "Equities slipped on Tuesday as Treasury yields climbed to a fresh high. " * 8,
            ),
        ],
    }


@pytest.fixture
def mini_corpus(registry: EntityRegistry, mini_raw: dict[str, list[RawArticle]]) -> Corpus:
    pipeline = IngestionPipeline(
        registry, IngestionConfig(chunk_max_chars=300, min_content_chars=100)
    )
    return pipeline.build(mini_raw)


@pytest.fixture
def mini_retriever(mini_corpus: Corpus) -> HybridRetriever:
    return HybridRetriever(
        mini_corpus,
        config=RetrievalConfig(top_k=4, per_ticker_k=2, min_anchored_coverage=0.1),
        lsa_components=8,
    )
