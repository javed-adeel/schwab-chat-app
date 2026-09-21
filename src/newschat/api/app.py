"""FastAPI application factory."""

from __future__ import annotations

import json
import logging
import re
import time
import uuid
from collections.abc import Awaitable, Callable, Iterator
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles

from newschat import __version__
from newschat.api.schemas import (
    ChatRequest,
    ChatResponse,
    HealthResponse,
    MetaOut,
)
from newschat.bootstrap import Application, build_application
from newschat.config import Settings
from newschat.generation.llm import LLMError
from newschat.generation.service import DeltaEvent, FinalEvent, MetaEvent

logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).parent / "static"
_REQUEST_ID = re.compile(r"^[A-Za-z0-9_\-]{1,64}$")
_CSP = (
    "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; "
    "connect-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'self'"
)
_LLM_UNAVAILABLE = "The language model is temporarily unavailable. Please try again."


def _sse(event: str, data: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


def create_app(settings: Settings | None = None, application: Application | None = None) -> FastAPI:
    """Build the app. Pass ``application`` to inject a prebuilt object graph (tests)."""
    application = application or build_application(settings or Settings())
    app = FastAPI(title="News Chat", version=__version__, docs_url="/api/docs", redoc_url=None)
    app.state.application = application

    @app.middleware("http")
    async def request_context(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        incoming = request.headers.get("x-request-id", "")
        request_id = incoming if _REQUEST_ID.match(incoming) else uuid.uuid4().hex[:12]
        started = time.perf_counter()
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Content-Security-Policy"] = _CSP
        logger.info(
            "request",
            extra={
                "request_id": request_id,
                "method": request.method,
                "path": request.url.path,
                "status": response.status_code,
                "duration_ms": round((time.perf_counter() - started) * 1000),
            },
        )
        return response

    @app.exception_handler(LLMError)
    async def llm_error_handler(_: Request, exc: LLMError) -> JSONResponse:
        logger.error("llm error: %s", exc)
        return JSONResponse(status_code=502, content={"detail": _LLM_UNAVAILABLE})

    @app.get("/api/health", response_model=HealthResponse)
    def health(request: Request) -> HealthResponse:
        app_state: Application = request.app.state.application
        return HealthResponse(
            status="ok",
            answerer=app_state.service.answerer_name,
            articles=len(app_state.corpus.articles),
            chunks=len(app_state.corpus.chunks),
        )

    @app.get("/api/corpus")
    def corpus_report(request: Request) -> dict[str, Any]:
        """What the ingestion pipeline found and cleaned (data-quality transparency)."""
        return dict(request.app.state.application.corpus.report.as_dict())

    @app.post("/api/chat", response_model=ChatResponse)
    def chat(body: ChatRequest, request: Request) -> ChatResponse:
        service = request.app.state.application.service
        return ChatResponse.from_result(service.chat(body.message, body.turns()))

    @app.post("/api/chat/stream")
    def chat_stream(body: ChatRequest, request: Request) -> StreamingResponse:
        service = request.app.state.application.service

        def events() -> Iterator[str]:
            try:
                for event in service.stream(body.message, body.turns()):
                    if isinstance(event, MetaEvent):
                        yield _sse("meta", MetaOut.from_event(event).model_dump())
                    elif isinstance(event, DeltaEvent):
                        yield _sse("delta", {"text": event.text})
                    elif isinstance(event, FinalEvent):
                        yield _sse("final", ChatResponse.from_result(event.result).model_dump())
            except LLMError as exc:
                # Headers are already sent, so failures must travel in-band.
                logger.error("llm error during stream: %s", exc)
                yield _sse("error", {"detail": _LLM_UNAVAILABLE})

        return StreamingResponse(
            events(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
    return app


def app_factory() -> FastAPI:
    """Zero-argument factory for ``uvicorn --factory newschat.api.app:app_factory``."""
    return create_app()
