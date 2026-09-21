"""Application settings, loaded from environment variables (prefix ``NEWSCHAT_``).

All tunables that affect behaviour live here so they can be changed without touching
code, and so tests can construct isolated ``Settings`` instances.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import AliasChoices, Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

LLMProvider = Literal["anthropic", "gemini", "openrouter"]
ResolvedProvider = Literal["anthropic", "gemini", "openrouter", "extractive"]
Provider = Literal["auto", "anthropic", "gemini", "openrouter", "extractive"]

# Precedence for ``auto`` when several keys are configured.
LLM_PROVIDERS: tuple[LLMProvider, ...] = ("anthropic", "gemini", "openrouter")
_KEY_ENV_VARS = {
    "anthropic": "ANTHROPIC_API_KEY",
    "gemini": "GEMINI_API_KEY or GOOGLE_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
}


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="NEWSCHAT_",
        env_file=".env",
        extra="ignore",
        populate_by_name=True,
    )

    # --- data -------------------------------------------------------------
    data_path: Path = Path("data/stock_news.json")

    # --- ingestion --------------------------------------------------------
    chunk_max_chars: int = Field(default=1100, ge=200)
    chunk_overlap_sentences: int = Field(default=1, ge=0, le=3)
    min_content_chars: int = Field(default=400, ge=0)

    # --- retrieval --------------------------------------------------------
    top_k: int = Field(default=8, ge=1, le=30)
    candidate_pool: int = Field(default=40, ge=5)
    per_ticker_k: int = Field(default=4, ge=1, le=10)
    max_chunks_per_article: int = Field(default=2, ge=1, le=5)
    rrf_k: int = Field(default=60, ge=1)
    min_query_coverage: float = Field(default=0.60, ge=0.0, le=1.0)
    min_anchored_coverage: float = Field(default=0.20, ge=0.0, le=1.0)
    min_articles_for_filter: int = Field(default=2, ge=1)
    lsa_components: int = Field(default=128, ge=8)

    # --- generation -------------------------------------------------------
    llm_provider: Provider = "auto"
    anthropic_api_key: SecretStr | None = Field(
        default=None,
        validation_alias=AliasChoices("ANTHROPIC_API_KEY", "NEWSCHAT_ANTHROPIC_API_KEY"),
    )
    anthropic_model: str = "claude-sonnet-5"
    gemini_api_key: SecretStr | None = Field(
        default=None,
        validation_alias=AliasChoices(
            "GEMINI_API_KEY", "GOOGLE_API_KEY", "NEWSCHAT_GEMINI_API_KEY"
        ),
    )
    gemini_model: str = "gemini-3.6-flash"
    openrouter_api_key: SecretStr | None = Field(
        default=None,
        validation_alias=AliasChoices("OPENROUTER_API_KEY", "NEWSCHAT_OPENROUTER_API_KEY"),
    )
    openrouter_model: str = "anthropic/claude-sonnet-5"
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    llm_max_tokens: int = Field(default=900, ge=100, le=4000)
    llm_timeout_seconds: float = Field(default=60.0, gt=0)
    history_max_messages: int = Field(default=8, ge=0, le=30)

    # --- serving ----------------------------------------------------------
    host: str = "127.0.0.1"
    port: int = Field(default=8000, ge=1, le=65535)
    log_level: str = "INFO"
    log_json: bool = False

    @model_validator(mode="after")
    def _explicit_provider_needs_a_key(self) -> Settings:
        """Fail at startup, not on the first chat, when a chosen provider has no key."""
        if self.llm_provider in LLM_PROVIDERS and self.api_key_for(self.llm_provider) is None:
            raise ValueError(
                f"NEWSCHAT_LLM_PROVIDER={self.llm_provider} but no API key is set "
                f"(expected {_KEY_ENV_VARS[self.llm_provider]})"
            )
        return self

    def api_key_for(self, provider: LLMProvider) -> str | None:
        secret = {
            "anthropic": self.anthropic_api_key,
            "gemini": self.gemini_api_key,
            "openrouter": self.openrouter_api_key,
        }[provider]
        return secret.get_secret_value() if secret else None

    @property
    def resolved_provider(self) -> ResolvedProvider:
        """``auto`` picks the first provider with a key (in ``LLM_PROVIDERS`` order)."""
        if self.llm_provider != "auto":
            return self.llm_provider
        return next((p for p in LLM_PROVIDERS if self.api_key_for(p)), "extractive")
