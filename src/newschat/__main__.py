"""``python -m newschat`` / ``newschat``: run the web server."""

from __future__ import annotations

import uvicorn

from newschat.config import Settings
from newschat.logging_config import configure_logging


def main() -> None:
    settings = Settings()
    configure_logging(settings.log_level, settings.log_json)
    uvicorn.run(
        "newschat.api.app:app_factory",
        factory=True,
        host=settings.host,
        port=settings.port,
        log_config=None,
    )


if __name__ == "__main__":
    main()
