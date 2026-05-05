import sys
import time
import logging
from pathlib import Path
from contextvars import ContextVar
from typing import Optional, Dict, Any

import structlog

from app.core.config import settings


#  CONTEXT VARS 
request_id_ctx: ContextVar[str] = ContextVar("request_id", default="-")
session_id_ctx: ContextVar[str] = ContextVar("session_id", default="-")
trace_id_ctx: ContextVar[str] = ContextVar("trace_id", default="-")


#  INTERNAL STATE 
_LOGGER_INITIALIZED: bool = False


#  CONTEXT BINDER 
def bind_request_context(
    request_id: str = "-",
    session_id: str = "-",
    trace_id: Optional[str] = "-"
) -> None:
    """
    Bind request-scoped context for structured logging.
    """
    request_id_ctx.set(request_id or "-")
    session_id_ctx.set(session_id or "-")
    trace_id_ctx.set(trace_id or "-")


def _add_context(logger, method_name, event_dict):
    """
    Inject contextvars into every log record.
    """
    event_dict["request_id"] = request_id_ctx.get()
    event_dict["session_id"] = session_id_ctx.get()
    event_dict["trace_id"] = trace_id_ctx.get()
    return event_dict


#  LOGGER CONFIG 
def _get_log_level() -> int:
    level = str(settings.LOG_LEVEL).upper()
    return getattr(logging, level, logging.INFO)


def _setup_logging() -> None:
    global _LOGGER_INITIALIZED

    if _LOGGER_INITIALIZED:
        return

    log_level = _get_log_level()

    #  STANDARD LOGGING 
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=log_level,
    )

    #  PROCESSORS 
    shared_processors = [
        structlog.contextvars.merge_contextvars,
        _add_context,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.add_log_level,
        structlog.processors.StackInfoRenderer(),
    ]

    if settings.LOG_JSON:
        renderer = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer()

    structlog.configure(
        processors=[
            *shared_processors,
            structlog.processors.format_exc_info,
            renderer,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(log_level),
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    #  FILE LOGGING 
    if settings.ENABLE_FILE_LOGGING:
        log_dir = Path(settings.LOG_DIR)
        log_dir.mkdir(parents=True, exist_ok=True)

        file_handler = logging.handlers.RotatingFileHandler(
            log_dir / settings.LOG_FILE_NAME,
            maxBytes=settings.LOG_MAX_BYTES,
            backupCount=settings.LOG_BACKUP_COUNT,
            encoding="utf-8",
        )

        file_handler.setLevel(log_level)

        file_formatter = logging.Formatter("%(message)s")
        file_handler.setFormatter(file_formatter)

        root_logger = logging.getLogger()
        root_logger.addHandler(file_handler)

    _LOGGER_INITIALIZED = True


#  PUBLIC LOGGER 
def get_logger(name: str):
    """
    Get structured logger instance.
    """
    _setup_logging()
    return structlog.get_logger(name)


#  LATENCY HELPER 
def log_latency(
    logger,
    event: str,
    start_time: float,
    extra: Optional[Dict[str, Any]] = None
) -> None:
    """
    Log latency with structured metadata.
    """
    try:
        latency = round(time.time() - start_time, 4)

        payload = {
            "event": event,
            "latency_sec": latency,
        }

        if extra:
            payload.update(extra)

        logger.info(**payload)

    except Exception:
        logger.warning(event="latency_logging_failed")