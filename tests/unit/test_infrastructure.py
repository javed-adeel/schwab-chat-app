"""Config, logging, bootstrap, CLI and entry point."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any
from unittest import mock

import pytest
from pydantic import ValidationError

from newschat import __main__ as entry
from newschat.bootstrap import build_answerer, build_application
from newschat.config import Settings
from newschat.evaluation import cli
from newschat.generation.answerers import ExtractiveAnswerer, LLMAnswerer
from newschat.logging_config import JsonFormatter, configure_logging

ROOT = Path(__file__).resolve().parents[2]

KEY_ENV_VARS = ("ANTHROPIC_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY", "OPENROUTER_API_KEY")


@pytest.fixture(autouse=True)
def _no_ambient_api_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    """A developer's own keys must not change which provider these tests see."""
    for var in KEY_ENV_VARS:
        monkeypatch.delenv(var, raising=False)


class TestSettings:
    def test_auto_provider_is_extractive_without_a_key(self) -> None:
        assert Settings(_env_file=None).resolved_provider == "extractive"

    def test_auto_provider_switches_to_anthropic_when_a_key_is_present(self) -> None:
        assert (
            Settings(_env_file=None, anthropic_api_key="sk-test").resolved_provider == "anthropic"
        )

    def test_explicit_provider_wins(self) -> None:
        assert (
            Settings(
                _env_file=None, llm_provider="extractive", anthropic_api_key="k"
            ).resolved_provider
            == "extractive"
        )

    @pytest.mark.parametrize(
        ("keys", "expected"),
        [
            ({"gemini_api_key": "g"}, "gemini"),
            ({"openrouter_api_key": "o"}, "openrouter"),
            ({"gemini_api_key": "g", "openrouter_api_key": "o"}, "gemini"),
            ({"anthropic_api_key": "a", "gemini_api_key": "g"}, "anthropic"),
        ],
    )
    def test_auto_provider_picks_the_first_configured_key(
        self, keys: dict[str, Any], expected: str
    ) -> None:
        assert Settings(_env_file=None, **keys).resolved_provider == expected

    def test_explicit_provider_overrides_auto_precedence(self) -> None:
        s = Settings(
            _env_file=None, llm_provider="openrouter", anthropic_api_key="a", openrouter_api_key="o"
        )
        assert s.resolved_provider == "openrouter"

    @pytest.mark.parametrize("provider", ["anthropic", "gemini", "openrouter"])
    def test_explicit_provider_without_its_key_fails_fast(self, provider: str) -> None:
        with pytest.raises(ValidationError, match="no API key"):
            Settings(_env_file=None, llm_provider=provider)

    @pytest.mark.parametrize(
        ("env_var", "field"),
        [
            ("GEMINI_API_KEY", "gemini_api_key"),
            ("GOOGLE_API_KEY", "gemini_api_key"),
            ("OPENROUTER_API_KEY", "openrouter_api_key"),
        ],
    )
    def test_reads_the_standard_provider_env_vars(
        self, monkeypatch: pytest.MonkeyPatch, env_var: str, field: str
    ) -> None:
        monkeypatch.setenv(env_var, "k-env")
        s = Settings(_env_file=None)
        secret = getattr(s, field)
        assert secret is not None and secret.get_secret_value() == "k-env"
        assert "k-env" not in repr(s)

    def test_reads_the_standard_anthropic_env_var(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-env")
        s = Settings(_env_file=None)
        assert (
            s.anthropic_api_key is not None and s.anthropic_api_key.get_secret_value() == "sk-env"
        )
        assert "sk-env" not in repr(s)  # secrets never appear in logs / reprs

    def test_prefixed_env_overrides(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("NEWSCHAT_TOP_K", "3")
        assert Settings(_env_file=None).top_k == 3

    @pytest.mark.parametrize(
        "field", [{"top_k": 0}, {"min_query_coverage": 1.5}, {"port": 0}, {"chunk_max_chars": 10}]
    )
    def test_out_of_range_values_are_rejected(self, field: dict[str, Any]) -> None:
        with pytest.raises(ValidationError):
            Settings(_env_file=None, **field)


class TestBootstrap:
    def test_default_answerer_is_extractive(self) -> None:
        assert isinstance(build_answerer(Settings(_env_file=None)), ExtractiveAnswerer)

    def test_anthropic_answerer_when_configured(self) -> None:
        answerer = build_answerer(
            Settings(_env_file=None, anthropic_api_key="sk-test", anthropic_model="m-9")
        )
        assert isinstance(answerer, LLMAnswerer) and answerer.name == "anthropic:m-9"

    def test_gemini_answerer_when_configured(self) -> None:
        answerer = build_answerer(
            Settings(_env_file=None, gemini_api_key="g-test", gemini_model="gem-1")
        )
        assert isinstance(answerer, LLMAnswerer) and answerer.name == "gemini:gem-1"

    def test_openrouter_answerer_when_configured(self) -> None:
        answerer = build_answerer(
            Settings(
                _env_file=None,
                openrouter_api_key="o-test",
                openrouter_model="vendor/m-2",
                openrouter_base_url="http://localhost:9/v1",
            )
        )
        assert isinstance(answerer, LLMAnswerer) and answerer.name == "openrouter:vendor/m-2"

    def test_build_application_wires_a_working_service(self, settings: Settings) -> None:
        app = build_application(settings)
        assert len(app.corpus.articles) == 118
        assert app.service.chat("Tell me about Nvidia").answered


class TestLogging:
    def test_json_formatter_includes_extra_fields(self) -> None:
        record = logging.LogRecord(
            "newschat.x", logging.INFO, __file__, 1, "hello %s", ("world",), None
        )
        record.request_id = "abc"
        out = json.loads(JsonFormatter().format(record))
        assert (
            out["message"] == "hello world"
            and out["request_id"] == "abc"
            and out["level"] == "INFO"
        )

    def test_json_formatter_includes_exceptions(self) -> None:
        try:
            raise ValueError("bad")
        except ValueError:
            import sys

            record = logging.LogRecord("n", logging.ERROR, __file__, 1, "x", (), sys.exc_info())
        assert "ValueError" in json.loads(JsonFormatter().format(record))["exc_info"]

    @pytest.mark.parametrize("as_json", [True, False])
    def test_configure_logging_is_idempotent(self, as_json: bool) -> None:
        configure_logging("DEBUG", as_json)
        configure_logging("INFO", as_json)
        root = logging.getLogger("newschat")
        assert len(root.handlers) == 1 and root.level == logging.INFO


class TestEvalCli:
    def test_passes_with_reasonable_thresholds(
        self, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("NEWSCHAT_DATA_PATH", str(ROOT / "data" / "stock_news.json"))
        code = cli.main(
            [
                "--golden",
                str(ROOT / "evals" / "golden_set.json"),
                "--min-hit",
                "0.9",
                "--min-mrr",
                "0.8",
                "--min-gate",
                "1.0",
            ]
        )
        assert code == 0 and "hit@5" in capsys.readouterr().out

    def test_fails_when_a_threshold_is_not_met(
        self, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("NEWSCHAT_DATA_PATH", str(ROOT / "data" / "stock_news.json"))
        assert (
            cli.main(["--golden", str(ROOT / "evals" / "golden_set.json"), "--min-mrr", "1.01"])
            == 1
        )
        assert "FAILED thresholds: MRR" in capsys.readouterr().err

    def test_ablation_table(
        self, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("NEWSCHAT_DATA_PATH", str(ROOT / "data" / "stock_news.json"))
        cli.main(["--golden", str(ROOT / "evals" / "golden_set.json"), "--ablation"])
        out = capsys.readouterr().out
        assert "BM25 only" in out and "dense (LSA) only" in out and "hybrid" in out
        assert "no entity routing" in out


def test_main_starts_uvicorn_with_the_app_factory() -> None:
    with mock.patch("uvicorn.run") as run:
        entry.main()
    args, kwargs = run.call_args
    assert args[0] == "newschat.api.app:app_factory" and kwargs["factory"] is True


def test_app_factory_builds_an_app(monkeypatch: pytest.MonkeyPatch) -> None:
    from newschat.api.app import app_factory

    monkeypatch.setenv("NEWSCHAT_DATA_PATH", str(ROOT / "data" / "stock_news.json"))
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert app_factory().title == "News Chat"
