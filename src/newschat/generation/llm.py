"""LLM client abstraction and its provider implementations.

Every provider adapts its own SDK to the same ``LLMClient`` protocol: a system prompt plus
user/assistant turns in, text deltas out, and any provider failure surfaced as ``LLMError``.
Each SDK is imported lazily so it stays an optional dependency, and each client accepts an
injected ``client`` so tests can pass a fake.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator, Sequence
from typing import Any, Protocol

from newschat.domain import ChatTurn

logger = logging.getLogger(__name__)

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


class LLMError(RuntimeError):
    """The language model call failed (network, auth, rate limit, provider error)."""


class LLMClient(Protocol):
    name: str

    def stream(
        self, *, system: str, messages: Sequence[ChatTurn], max_tokens: int
    ) -> Iterator[str]:
        """Yield the response text incrementally."""
        ...


def _missing_extra(extra: str, provider: str) -> LLMError:
    return LLMError(f"install the '{extra}' extra to use the {provider} provider")


class AnthropicClient:
    """Streams completions from the Anthropic Messages API."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "claude-sonnet-5",
        timeout: float = 60.0,
        client: Any | None = None,
    ) -> None:
        self.name = f"anthropic:{model}"
        self._model = model
        if client is not None:
            self._client = client
            return
        try:
            import anthropic
        except ImportError as exc:  # pragma: no cover - exercised only without the extra
            raise _missing_extra("anthropic", "Anthropic") from exc
        self._client = anthropic.Anthropic(api_key=api_key, timeout=timeout, max_retries=2)

    def stream(
        self, *, system: str, messages: Sequence[ChatTurn], max_tokens: int
    ) -> Iterator[str]:
        payload = [{"role": m.role, "content": m.content} for m in messages]
        try:
            with self._client.messages.stream(
                model=self._model, max_tokens=max_tokens, system=system, messages=payload
            ) as stream:
                yield from stream.text_stream
        except Exception as exc:
            logger.exception("anthropic request failed")
            raise LLMError(f"{type(exc).__name__}: {exc}") from exc


class GeminiClient:
    """Streams completions from the Google Gemini API (``google-genai`` SDK).

    Gemini calls the assistant role ``model`` and takes the system prompt as config.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gemini-3.6-flash",
        timeout: float = 60.0,
        client: Any | None = None,
    ) -> None:
        self.name = f"gemini:{model}"
        self._model = model
        if client is not None:
            self._client = client
            return
        try:
            from google import genai
        except ImportError as exc:  # pragma: no cover - exercised only without the extra
            raise _missing_extra("gemini", "Gemini") from exc
        # The SDK's timeout is in milliseconds.
        self._client = genai.Client(api_key=api_key, http_options={"timeout": int(timeout * 1000)})

    def stream(
        self, *, system: str, messages: Sequence[ChatTurn], max_tokens: int
    ) -> Iterator[str]:
        contents = [
            {"role": "model" if m.role == "assistant" else "user", "parts": [{"text": m.content}]}
            for m in messages
        ]
        try:
            for chunk in self._client.models.generate_content_stream(
                model=self._model,
                contents=contents,
                config={
                    "system_instruction": system,
                    "max_output_tokens": max_tokens,
                    # no tools are declared, so skip the SDK's function-calling loop
                    "automatic_function_calling": {"disable": True},
                },
            ):
                if chunk.text:
                    yield chunk.text
        except Exception as exc:
            logger.exception("gemini request failed")
            raise LLMError(f"{type(exc).__name__}: {exc}") from exc


class OpenRouterClient:
    """Streams chat completions from OpenRouter's OpenAI-compatible API.

    One key reaches many vendors' models; ``model`` is an OpenRouter slug such as
    ``anthropic/claude-sonnet-5`` or ``google/gemini-3.6-flash``.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "anthropic/claude-sonnet-5",
        timeout: float = 60.0,
        base_url: str = OPENROUTER_BASE_URL,
        client: Any | None = None,
    ) -> None:
        self.name = f"openrouter:{model}"
        self._model = model
        if client is not None:
            self._client = client
            return
        try:
            import openai
        except ImportError as exc:  # pragma: no cover - exercised only without the extra
            raise _missing_extra("openrouter", "OpenRouter") from exc
        self._client = openai.OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
            max_retries=2,
            default_headers={"X-Title": "newschat"},  # app attribution on openrouter.ai
        )

    def stream(
        self, *, system: str, messages: Sequence[ChatTurn], max_tokens: int
    ) -> Iterator[str]:
        payload = [{"role": "system", "content": system}]
        payload += [{"role": m.role, "content": m.content} for m in messages]
        try:
            response = self._client.chat.completions.create(
                model=self._model, max_tokens=max_tokens, messages=payload, stream=True
            )
            for chunk in response:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        except Exception as exc:
            logger.exception("openrouter request failed")
            raise LLMError(f"{type(exc).__name__}: {exc}") from exc
