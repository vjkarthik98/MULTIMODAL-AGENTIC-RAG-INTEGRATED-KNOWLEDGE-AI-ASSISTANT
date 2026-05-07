import json
import logging
import os
import sys
import time
import traceback
import uuid
from contextvars import ContextVar
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Dict, Optional

from app.core.config import settings


# CONTEXT VARS

request_id_ctx: ContextVar[str] = ContextVar("request_id", default="-")
session_id_ctx: ContextVar[str] = ContextVar("session_id", default="-")


# INTERNAL STATE

_LOGGER_INITIALIZED = False


# REQUEST CONTEXT

def bind_request_context(
    request_id: str = "-",
    session_id: str = "-"
) -> None:
    request_id_ctx.set(request_id or str(uuid.uuid4()))
    session_id_ctx.set(session_id or "-")


# CLEAN FORMATTER

class CleanFormatter(logging.Formatter):

    def format(self, record: logging.LogRecord) -> str:

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        level     = record.levelname.upper()
        event     = getattr(record, "event", record.getMessage())

        parts = []

        if settings.LOG_SHOW_TIMESTAMP:
            parts.append(timestamp)

        parts.extend([level, record.name, event])

        skip_keys = {
            "name", "msg", "args", "levelname", "levelno",
            "pathname", "filename", "module", "exc_info",
            "exc_text", "stack_info", "lineno", "funcName",
            "created", "msecs", "relativeCreated", "thread",
            "threadName", "processName", "process", "event",
            "message", "taskName",
        }

        extras = []
        for key, value in record.__dict__.items():
            if key in skip_keys:
                continue
            if value in (None, "", "-", [], {}, ()):
                continue
            extras.append(f"{key}={value}")

        if extras:
            parts.extend(extras)

        message = " | ".join(parts)

        if record.exc_info:
            message += "\n" + "".join(traceback.format_exception(*record.exc_info))

        return message


# JSON FORMATTER

class JsonFormatter(logging.Formatter):

    def format(self, record: logging.LogRecord) -> str:

        skip_keys = {
            "name", "msg", "args", "levelname", "levelno",
            "pathname", "filename", "module", "exc_info",
            "exc_text", "stack_info", "lineno", "funcName",
            "created", "msecs", "relativeCreated", "thread",
            "threadName", "processName", "process", "message",
            "taskName",
        }

        payload: Dict[str, Any] = {
            "timestamp": datetime.utcnow().isoformat(),
            "level":     record.levelname,
            "logger":    record.name,
            "event":     getattr(record, "event", record.getMessage()),
            "request_id": request_id_ctx.get("-"),
            "session_id": session_id_ctx.get("-"),
        }

        for key, value in record.__dict__.items():
            if key in skip_keys or key == "event":
                continue
            if value in (None, "", "-", [], {}, ()):
                continue
            try:
                json.dumps(value)
                payload[key] = value
            except (TypeError, ValueError):
                payload[key] = str(value)

        if record.exc_info:
            payload["traceback"] = "".join(
                traceback.format_exception(*record.exc_info)
            )

        return json.dumps(payload, ensure_ascii=False)


# STRUCTURED LOGGER

class StructuredLogger:

    def __init__(self, logger: logging.Logger) -> None:
        self._logger = logger

    def _log(self, level: str, event: Optional[str] = None, **kwargs: Any) -> None:
        extra = {"event": event, **kwargs}
        getattr(self._logger, level)(event or "log", extra=extra)

    def info(self, event: Optional[str] = None, **kwargs: Any) -> None:
        self._log("info", event, **kwargs)

    def warning(self, event: Optional[str] = None, **kwargs: Any) -> None:
        self._log("warning", event, **kwargs)

    def error(self, event: Optional[str] = None, **kwargs: Any) -> None:
        self._log("error", event, **kwargs)

    def debug(self, event: Optional[str] = None, **kwargs: Any) -> None:
        self._log("debug", event, **kwargs)

    def critical(self, event: Optional[str] = None, **kwargs: Any) -> None:
        self._log("critical", event, **kwargs)

    def exception(self, event: Optional[str] = None, **kwargs: Any) -> None:
        self._logger.exception(event or "exception", extra=kwargs)


# LOGGER SETUP

def _get_log_level() -> int:
    return getattr(logging, str(settings.LOG_LEVEL).upper(), logging.INFO)


def _build_formatter() -> logging.Formatter:
    if settings.LOG_JSON:
        return JsonFormatter()
    return CleanFormatter()


def _setup_logging() -> None:

    global _LOGGER_INITIALIZED

    if _LOGGER_INITIALIZED:
        return

    level     = _get_log_level()
    formatter = _build_formatter()

    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(level)

    # CONSOLE
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(level)
    console.setFormatter(formatter)
    root.addHandler(console)

    # FILE
    if settings.ENABLE_FILE_LOGGING:
        log_dir = Path(settings.LOG_DIR)
        log_dir.mkdir(parents=True, exist_ok=True)

        file_handler = RotatingFileHandler(
            filename=log_dir / settings.LOG_FILE_NAME,
            maxBytes=settings.LOG_MAX_BYTES,
            backupCount=settings.LOG_BACKUP_COUNT,
            encoding="utf-8",
        )
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)

    # UVICORN
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access", "fastapi"):
        uv = logging.getLogger(name)
        uv.handlers.clear()
        uv.propagate = True

    _LOGGER_INITIALIZED = True


# PUBLIC LOGGER

def get_logger(name: str) -> StructuredLogger:
    _setup_logging()
    return StructuredLogger(logging.getLogger(name))


# HELPERS

def log_latency(
    logger: StructuredLogger,
    event: str,
    start_time: float,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    try:
        payload: Dict[str, Any] = {"latency_sec": round(time.time() - start_time, 4)}
        if extra:
            payload.update(extra)
        logger.info(event=event, **payload)
    except Exception:
        logger.warning(event="latency_logging_failed")


def log_exception(
    logger: StructuredLogger,
    event: str,
    error: Exception,
    **kwargs: Any,
) -> None:
    logger.error(
        event=event,
        error=str(error),
        error_type=type(error).__name__,
        **kwargs,
    )


def log_slow_request(
    logger: StructuredLogger,
    path: str,
    latency: float,
    threshold: float = None,
) -> None:
    threshold = threshold or settings.SLOW_REQUEST_THRESHOLD
    if latency > threshold:
        logger.warning(
            event="slow_request",
            path=path,
            latency=latency,
            threshold=threshold,
        )