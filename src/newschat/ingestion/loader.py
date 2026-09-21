"""Reads and validates the raw news JSON (``{ticker: [article, ...]}``)."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, ValidationError, field_validator

logger = logging.getLogger(__name__)


class DataLoadError(RuntimeError):
    """The dataset is missing or structurally unusable."""


class RawArticle(BaseModel):
    model_config = ConfigDict(extra="ignore", str_strip_whitespace=True)

    title: str
    link: str
    full_text: str = ""

    @field_validator("title")
    @classmethod
    def _title_not_empty(cls, v: str) -> str:
        if not v:
            raise ValueError("title must not be empty")
        return v

    @field_validator("link")
    @classmethod
    def _link_is_http(cls, v: str) -> str:
        if not v.lower().startswith(("http://", "https://")):
            raise ValueError("link must be an http(s) URL")
        return v


def parse_raw_articles(payload: Any) -> tuple[dict[str, list[RawArticle]], int]:
    """Validate a decoded payload. Returns ``(articles_by_ticker, invalid_count)``.

    Malformed *records* are skipped and counted (one bad article must not take the whole
    service down); a malformed *top level* is a hard error.
    """
    if not isinstance(payload, dict):
        raise DataLoadError("expected a JSON object mapping ticker -> list of articles")
    result: dict[str, list[RawArticle]] = {}
    invalid = 0
    for ticker, records in payload.items():
        if not isinstance(records, list):
            raise DataLoadError(f"value for {ticker!r} must be a list of articles")
        parsed: list[RawArticle] = []
        for record in records:
            try:
                parsed.append(RawArticle.model_validate(record))
            except ValidationError as exc:
                invalid += 1
                logger.warning(
                    "skipping invalid record under %s: %s", ticker, exc.errors()[0]["msg"]
                )
        result[str(ticker).upper()] = parsed
    return result, invalid


def load_raw_articles(path: Path) -> tuple[dict[str, list[RawArticle]], int]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise DataLoadError(f"dataset not found: {path.resolve()}") from exc
    except json.JSONDecodeError as exc:
        raise DataLoadError(f"dataset is not valid JSON: {exc}") from exc
    return parse_raw_articles(payload)
