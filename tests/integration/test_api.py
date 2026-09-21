"""HTTP-level tests against the real dataset (extractive answerer unless stated)."""

from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Any

import pytest
from fastapi.testclient import TestClient
from tests.fakes import FakeLLM

from newschat.api.app import create_app
from newschat.bootstrap import Application
from newschat.generation.answerers import LLMAnswerer
from newschat.generation.llm import LLMError
from newschat.generation.service import ChatService
from newschat.retrieval.query_analysis import QueryAnalyzer
from newschat.retrieval.retriever import HybridRetriever


@pytest.fixture(scope="module")
def client(application: Application) -> Iterator[TestClient]:
    with TestClient(create_app(application=application)) as c:
        yield c


def parse_sse(body: str) -> list[tuple[str, dict[str, Any]]]:
    events = []
    for block in body.strip().split("\n\n"):
        lines = dict(line.split(": ", 1) for line in block.split("\n"))
        events.append((lines["event"], json.loads(lines["data"])))
    return events


class TestReadEndpoints:
    def test_health(self, client: TestClient) -> None:
        body = client.get("/api/health").json()
        assert body == {
            "status": "ok",
            "answerer": "extractive",
            "articles": 118,
            "chunks": body["chunks"],
        }

    def test_corpus_report_exposes_data_quality_findings(self, client: TestClient) -> None:
        report = client.get("/api/corpus").json()
        assert report["raw_records"] == 138 and report["duplicates_merged"] == 20
        assert report["boilerplate_removed"]["paywall"] == 13

    def test_ui_and_static_assets_are_served(self, client: TestClient) -> None:
        assert "News Chat" in client.get("/").text
        assert client.get("/static/app.js").status_code == 200
        assert client.get("/static/styles.css").status_code == 200

    def test_openapi_docs_available(self, client: TestClient) -> None:
        assert client.get("/api/docs").status_code == 200


class TestChat:
    def test_answers_with_citations_and_explains_how_it_read_the_question(
        self, client: TestClient
    ) -> None:
        r = client.post(
            "/api/chat", json={"message": "What price target changes were reported for Intel?"}
        )
        assert r.status_code == 200
        body = r.json()
        assert (
            body["answered"]
            and body["plan"]["tickers"] == ["INTC"]
            and body["plan"]["intent"] == "company"
        )
        assert body["citations"] and body["invalid_citations"] == []
        assert all(c["link"].startswith("http") for c in body["citations"])
        assert any(s["headline_only"] for s in body["sources"])

    def test_comparison(self, client: TestClient) -> None:
        body = client.post("/api/chat", json={"message": "Compare Nvidia and Intel"}).json()
        assert body["plan"]["intent"] == "comparison" and set(body["plan"]["tickers"]) == {
            "NVDA",
            "INTC",
        }

    def test_follow_up_uses_history(self, client: TestClient) -> None:
        body = client.post(
            "/api/chat",
            json={
                "message": "What about Amazon?",
                "history": [
                    {"role": "user", "content": "What did Jim Cramer say about Netflix?"},
                    {"role": "assistant", "content": "He is bullish [1]."},
                ],
            },
        ).json()
        assert body["plan"]["is_follow_up"] and body["plan"]["tickers"] == ["AMZN"]
        assert any("Cramer" in s["title"] for s in body["sources"])

    def test_out_of_scope_is_declined(self, client: TestClient) -> None:
        body = client.post(
            "/api/chat", json={"message": "What is the weather in Toronto today?"}
        ).json()
        assert body["answered"] is False and body["sources"] == []

    def test_security_and_tracing_headers(self, client: TestClient) -> None:
        r = client.get("/api/health", headers={"X-Request-ID": "trace-123"})
        assert r.headers["X-Request-ID"] == "trace-123"
        assert r.headers["X-Content-Type-Options"] == "nosniff"
        assert "script-src 'self'" in r.headers["Content-Security-Policy"]

    @pytest.mark.parametrize("bad_id", ["bad id with spaces", "x" * 200, "semi;colon"])
    def test_untrusted_request_ids_are_replaced(self, client: TestClient, bad_id: str) -> None:
        r = client.get("/api/health", headers={"X-Request-ID": bad_id})
        assert r.headers["X-Request-ID"] != bad_id and len(r.headers["X-Request-ID"]) == 12

    @pytest.mark.parametrize(
        "payload",
        [
            {},
            {"message": ""},
            {"message": "   "},
            {"message": "x" * 2001},
            {"message": "hi", "history": [{"role": "system", "content": "x"}]},
            {"message": "hi", "history": [{"role": "user", "content": ""}]},
            {"message": "hi", "history": [{"role": "user", "content": "x"}] * 41},
        ],
    )
    def test_invalid_requests_are_rejected_with_422(
        self, client: TestClient, payload: dict[str, Any]
    ) -> None:
        assert client.post("/api/chat", json=payload).status_code == 422


class TestStreaming:
    def test_event_sequence_and_content_type(self, client: TestClient) -> None:
        r = client.post("/api/chat/stream", json={"message": "Compare Nvidia and Intel"})
        assert r.headers["content-type"].startswith("text/event-stream")
        events = parse_sse(r.text)
        names = [n for n, _ in events]
        assert names[0] == "meta" and names[-1] == "final" and set(names[1:-1]) == {"delta"}
        streamed = "".join(d["text"] for n, d in events if n == "delta")
        assert streamed.strip() == events[-1][1]["answer"]

    def test_declined_questions_still_stream_cleanly(self, client: TestClient) -> None:
        events = parse_sse(client.post("/api/chat/stream", json={"message": "hi"}).text)
        assert [n for n, _ in events] == ["meta", "delta", "final"]
        assert events[-1][1]["answered"] is False

    def test_validation_errors_apply_to_streaming_too(self, client: TestClient) -> None:
        assert client.post("/api/chat/stream", json={"message": ""}).status_code == 422


class TestLLMFailures:
    @pytest.fixture
    def failing_client(
        self, application: Application, analyzer: QueryAnalyzer, retriever: HybridRetriever
    ) -> Iterator[TestClient]:
        llm = FakeLLM(error=LLMError("secret internal detail"))
        service = ChatService(analyzer, retriever, LLMAnswerer(llm), application.corpus.tickers)
        app = Application(application.settings, application.corpus, service)
        with TestClient(create_app(application=app)) as c:
            yield c

    def test_json_endpoint_returns_502_without_leaking_internals(
        self, failing_client: TestClient
    ) -> None:
        r = failing_client.post("/api/chat", json={"message": "Tell me about Nvidia"})
        assert r.status_code == 502
        assert (
            "secret internal detail" not in r.text
            and "temporarily unavailable" in r.json()["detail"]
        )

    def test_stream_reports_the_failure_in_band(self, failing_client: TestClient) -> None:
        events = parse_sse(
            failing_client.post("/api/chat/stream", json={"message": "Tell me about Nvidia"}).text
        )
        assert events[0][0] == "meta" and events[-1][0] == "error"
        assert "secret internal detail" not in json.dumps(events)
