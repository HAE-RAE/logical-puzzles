"""Stdlib logging setup helper."""

import logging


def setup_logger(name: str | None = None, *, level: int = logging.INFO) -> logging.Logger:
    """Configure root logging with the project's standard format and return a logger.

    Idempotent: re-calling won't add duplicate handlers.
    """
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    return logging.getLogger(name)
