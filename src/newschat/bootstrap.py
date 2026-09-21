"""Composition root: builds the object graph from ``Settings``.

Keeping construction in one place means the rest of the code takes its dependencies as
arguments and never reads global state, which is what makes it easy to test.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from newschat.config import LLMProvider, Settings
from newschat.domain import Corpus
from newschat.generation.answerers import Answerer, ExtractiveAnswerer, LLMAnswerer
from newschat.generation.llm import AnthropicClient, GeminiClient, LLMClient, OpenRouterClient
from newschat.generation.service import ChatService
from newschat.ingestion.entities import EntityRegistry
from newschat.ingestion.pipeline import IngestionConfig, IngestionPipeline
from newschat.retrieval.query_analysis import QueryAnalyzer
from newschat.retrieval.retriever import HybridRetriever, RetrievalConfig

logger = logging.getLogger(__name__)


def build_corpus(settings: Settings, registry: EntityRegistry | None = None) -> Corpus:
    pipeline = IngestionPipeline(
        registry or EntityRegistry(),
        IngestionConfig(
            chunk_max_chars=settings.chunk_max_chars,
            chunk_overlap_sentences=settings.chunk_overlap_sentences,
            min_content_chars=settings.min_content_chars,
        ),
    )
    return pipeline.run(settings.data_path)


def build_retriever(corpus: Corpus, settings: Settings) -> HybridRetriever:
    return HybridRetriever(
        corpus,
        config=RetrievalConfig.from_settings(settings),
        lsa_components=settings.lsa_components,
    )


def build_llm_client(settings: Settings, provider: LLMProvider) -> LLMClient:
    key, timeout = settings.api_key_for(provider), settings.llm_timeout_seconds
    if provider == "anthropic":
        return AnthropicClient(api_key=key, model=settings.anthropic_model, timeout=timeout)
    if provider == "gemini":
        return GeminiClient(api_key=key, model=settings.gemini_model, timeout=timeout)
    return OpenRouterClient(
        api_key=key,
        model=settings.openrouter_model,
        timeout=timeout,
        base_url=settings.openrouter_base_url,
    )


def build_answerer(settings: Settings) -> Answerer:
    provider = settings.resolved_provider
    if provider == "extractive":
        return ExtractiveAnswerer()
    client = build_llm_client(settings, provider)
    return LLMAnswerer(client, settings.llm_max_tokens, settings.history_max_messages)


@dataclass(frozen=True, slots=True)
class Application:
    """Everything the HTTP layer needs, built once at startup."""

    settings: Settings
    corpus: Corpus
    service: ChatService


def build_application(settings: Settings) -> Application:
    registry = EntityRegistry()
    corpus = build_corpus(settings, registry)
    retriever = build_retriever(corpus, settings)
    answerer = build_answerer(settings)
    logger.info(
        "chat service ready",
        extra={
            "articles": len(corpus.articles),
            "chunks": len(corpus.chunks),
            "answerer": answerer.name,
        },
    )
    service = ChatService(
        QueryAnalyzer(registry), retriever, answerer, supported_tickers=corpus.tickers
    )
    return Application(settings=settings, corpus=corpus, service=service)
