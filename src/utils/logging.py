# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Structured logging configuration using structlog.

This module provides structured logging setup for EduSynapseOS.
Logs are formatted as JSON in production and as colored console output
in development for better readability.

Example:
    >>> from src.utils.logging import setup_logging, get_logger
    >>> from src.core.config import get_settings
    >>> setup_logging(get_settings())
    >>> logger = get_logger(__name__)
    >>> logger.info("User logged in", user_id="123", ip="192.168.1.1")
"""

import logging
import sys
from typing import TYPE_CHECKING

import structlog
from structlog.types import Processor

if TYPE_CHECKING:
    from src.core.config.settings import Settings


def setup_logging(settings: "Settings") -> None:
    """Configure structured logging for the application.

    Sets up structlog with appropriate processors based on environment:
    - Development: Colored console output with pretty formatting
    - Production: JSON output for log aggregation

    Args:
        settings: Application settings containing log_level and debug flag.
    """
    # Determine log level
    log_level = getattr(logging, settings.log_level.upper(), logging.INFO)

    # Common processors used in all environments
    shared_processors: list[Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.UnicodeDecoder(),
    ]

    if settings.is_development or settings.debug:
        # Development: colored console output
        processors: list[Processor] = [
            *shared_processors,
            structlog.dev.ConsoleRenderer(colors=True),
        ]
    else:
        # Production: JSON output
        processors = [
            *shared_processors,
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ]

    # Configure structlog
    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(log_level),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(file=sys.stdout),
        cache_logger_on_first_use=True,
    )

    # Also configure standard library logging
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=log_level,
    )

    # Set third-party loggers to WARNING to reduce noise
    for logger_name in [
        "uvicorn",
        "uvicorn.access",
        "uvicorn.error",
        "httpx",
        "httpcore",
        "sqlalchemy",
        "asyncio",
        "urllib3",
    ]:
        logging.getLogger(logger_name).setLevel(logging.WARNING)

    # Keep our loggers at configured level
    logging.getLogger("src").setLevel(log_level)


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """Get a structured logger for the given module name.

    Args:
        name: Usually __name__ of the calling module.

    Returns:
        A bound structlog logger instance.

    Example:
        >>> logger = get_logger(__name__)
        >>> logger.info("Processing started", item_count=42)
    """
    return structlog.get_logger(name)


def bind_context(**kwargs: object) -> None:
    """Bind context variables to all subsequent log calls in this context.

    Useful for adding request-scoped information like request_id or user_id.

    Args:
        **kwargs: Key-value pairs to bind to the logging context.

    Example:
        >>> bind_context(request_id="abc-123", user_id="user-456")
        >>> logger.info("Processing request")  # Will include request_id and user_id
    """
    structlog.contextvars.bind_contextvars(**kwargs)


def clear_context() -> None:
    """Clear all bound context variables.

    Should be called at the end of request processing to prevent
    context leakage between requests.
    """
    structlog.contextvars.clear_contextvars()
