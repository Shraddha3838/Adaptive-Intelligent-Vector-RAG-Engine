"""Shared utilities: structured logging, timing helpers, and common helpers."""

from __future__ import annotations

import logging
import sys
import time
from contextlib import contextmanager
from typing import Any, Generator

import certifi
from pymongo import MongoClient

from src.constants import APP_NAME, APP_VERSION

_CONFIGURED = False


def get_mongo_client(uri: str) -> MongoClient:
    """Return a MongoDB client with Atlas-friendly TLS and timeout settings."""
    return MongoClient(
        uri,
        serverSelectionTimeoutMS=30_000,
        connectTimeoutMS=30_000,
        socketTimeoutMS=60_000,
        tlsCAFile=certifi.where(),
        retryWrites=True,
    )


class _AppFormatter(logging.Formatter):
    """Structured log formatter with consistent timestamp and level coloring hints."""

    FORMAT = (
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    )
    DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

    def __init__(self) -> None:
        super().__init__(fmt=self.FORMAT, datefmt=self.DATE_FORMAT)


def setup_logging(
    level: str = "INFO",
    *,
    logger_name: str | None = None,
) -> logging.Logger:
    """
    Configure application-wide structured logging.

    Args:
        level: Log level string (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        logger_name: Optional logger name. Defaults to the root app logger.

    Returns:
        Configured logger instance.
    """
    global _CONFIGURED

    numeric_level = getattr(logging, level.upper(), logging.INFO)
    root_logger = logging.getLogger(APP_NAME)
    root_logger.setLevel(numeric_level)

    if not _CONFIGURED:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(numeric_level)
        handler.setFormatter(_AppFormatter())
        root_logger.addHandler(handler)

        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("httpcore").setLevel(logging.WARNING)
        logging.getLogger("openai").setLevel(logging.WARNING)
        logging.getLogger("pymongo").setLevel(logging.WARNING)

        _CONFIGURED = True

    if logger_name:
        return logging.getLogger(f"{APP_NAME}.{logger_name}")
    return root_logger


def get_logger(name: str) -> logging.Logger:
    """
    Return a child logger under the application namespace.

    Args:
        name: Module or component name (e.g. 'ingest', 'retrieval').

    Returns:
        Logger instance.
    """
    return logging.getLogger(f"{APP_NAME}.{name}")


@contextmanager
def log_duration(
    logger: logging.Logger,
    operation: str,
    *,
    level: int = logging.INFO,
    extra: dict[str, Any] | None = None,
) -> Generator[None, None, None]:
    """
    Context manager that logs the duration of an operation.

    Args:
        logger: Logger to write to.
        operation: Human-readable operation name.
        level: Log level for the completion message.
        extra: Optional extra fields to include in the log message.
    """
    start = time.perf_counter()
    logger.debug("Starting: %s", operation, extra=extra)
    try:
        yield
    except Exception:
        elapsed_ms = (time.perf_counter() - start) * 1000
        logger.exception(
            "Failed: %s (%.1f ms)",
            operation,
            elapsed_ms,
            extra=extra,
        )
        raise
    else:
        elapsed_ms = (time.perf_counter() - start) * 1000
        logger.log(
            level,
            "Completed: %s (%.1f ms)",
            operation,
            elapsed_ms,
            extra=extra,
        )


def mask_secret(value: str, visible_chars: int = 4) -> str:
    """
    Mask a secret string for safe logging.

    Args:
        value: Secret value to mask.
        visible_chars: Number of trailing characters to reveal.

    Returns:
        Masked string like '****abcd'.
    """
    if not value:
        return "<empty>"
    if len(value) <= visible_chars:
        return "*" * len(value)
    return "*" * (len(value) - visible_chars) + value[-visible_chars:]


def app_info() -> dict[str, str]:
    """Return application metadata for health checks and UI."""
    return {
        "name": APP_NAME,
        "version": APP_VERSION,
    }
