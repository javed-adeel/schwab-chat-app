FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    NEWSCHAT_HOST=0.0.0.0 \
    NEWSCHAT_LOG_JSON=true

WORKDIR /app

# Install dependencies + package first so this layer caches across data changes.
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install ".[llm]"

COPY data ./data

RUN useradd --create-home --uid 10001 app && chown -R app /app
USER app

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
  CMD python -c "import urllib.request as u; u.urlopen('http://127.0.0.1:8000/api/health', timeout=3)"

# Set ANTHROPIC_API_KEY, GEMINI_API_KEY or OPENROUTER_API_KEY at runtime for LLM summaries;
# without one the app runs extractively.
CMD ["python", "-m", "newschat"]
