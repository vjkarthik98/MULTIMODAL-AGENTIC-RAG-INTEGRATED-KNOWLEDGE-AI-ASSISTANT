from __future__ import annotations

import json
import logging
import os
import sys
import time
import traceback
import uuid
from contextvars import ContextVar
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Dict, Optional

from app.core.config import settings


# CONTEXT VARS — BOUND PER REQUEST

request_id_ctx: ContextVar[str] = ContextVar("request_id", default="-")
session_id_ctx: ContextVar[str] = ContextVar("session_id", default="-")
trace_id_ctx:   ContextVar[str] = ContextVar("trace_id",   default="-")
span_id_ctx:    ContextVar[str] = ContextVar("span_id",    default="-")
user_id_ctx:    ContextVar[str] = ContextVar("user_id",    default="-")


# INTERNAL STATE

_LOGGER_INITIALIZED: bool = False


# CONTEXT BINDING

def bind_request_context(
    request_id: str = "-",
    session_id: str = "-",
    trace_id:   str = "-",
    span_id:    str = "-",
    user_id:    str = "-",
) -> None:
    request_id_ctx.set(request_id or str(uuid.uuid4()))
    session_id_ctx.set(session_id or "-")
    trace_id_ctx.set(trace_id or "-")
    span_id_ctx.set(span_id or "-")
    user_id_ctx.set(user_id or "-")


def clear_request_context() -> None:
    request_id_ctx.set("-")
    session_id_ctx.set("-")
    trace_id_ctx.set("-")
    span_id_ctx.set("-")
    user_id_ctx.set("-")


def get_current_request_id() -> str:
    return request_id_ctx.get("-")


def get_current_session_id() -> str:
    return session_id_ctx.get("-")


# SKIP KEYS FOR FORMATTERS

_SKIP_KEYS = frozenset({
    "name", "msg", "args", "levelname", "levelno",
    "pathname", "filename", "module", "exc_info",
    "exc_text", "stack_info", "lineno", "funcName",
    "created", "msecs", "relativeCreated", "thread",
    "threadName", "processName", "process", "message",
    "taskName", "event",
})

# VALUES TO SUPPRESS IN OUTPUT
_SUPPRESS_VALUES = frozenset({None, "", "-", [], {}, ()})


# CLEAN CONSOLE FORMATTER

class CleanFormatter(logging.Formatter):

    LEVEL_COLORS = {
        "DEBUG":    "\033[36m",
        "INFO":     "\033[32m",
        "WARNING":  "\033[33m",
        "ERROR":    "\033[31m",
        "CRITICAL": "\033[35m",
    }
    RESET = "\033[0m"

    def __init__(self, use_color: bool = True) -> None:
        super().__init__()
        self._use_color = use_color and sys.stdout.isatty()

    def format(self, record: logging.LogRecord) -> str:
        timestamp = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        level = record.levelname.upper()
        event = getattr(record, "event", None) or record.getMessage()

        # CORRELATION IDs
        req_id = request_id_ctx.get("-")
        ses_id = session_id_ctx.get("-")
        trc_id = trace_id_ctx.get("-")

        parts: list[str] = []

        if settings.LOG_SHOW_TIMESTAMP:
            parts.append(timestamp)

        if self._use_color:
            color = self.LEVEL_COLORS.get(level, "")
            parts.append(f"{color}{level}{self.RESET}")
        else:
            parts.append(level)

        parts.append(record.name)
        parts.append(str(event))

        # CORRELATION CONTEXT
        if req_id != "-":
            parts.append(f"req={req_id[:8]}")
        if ses_id != "-":
            parts.append(f"ses={ses_id[:12]}")
        if trc_id != "-":
            parts.append(f"trace={trc_id[:16]}")

        # EXTRA FIELDS
        extras: list[str] = []
        for key, value in record.__dict__.items():
            if key in _SKIP_KEYS:
                continue
            if value in _SUPPRESS_VALUES:
                continue
            if isinstance(value, float):
                extras.append(f"{key}={round(value, 4)}")
            else:
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
        payload: Dict[str, Any] = {
            "timestamp":  datetime.now(tz=timezone.utc).isoformat(),
            "level":      record.levelname,
            "logger":     record.name,
            "event":      getattr(record, "event", None) or record.getMessage(),
            "request_id": request_id_ctx.get("-"),
            "session_id": session_id_ctx.get("-"),
            "trace_id":   trace_id_ctx.get("-"),
            "span_id":    span_id_ctx.get("-"),
            "user_id":    user_id_ctx.get("-"),
            "service":    settings.APP_NAME,
            "version":    settings.APP_VERSION,
            "env":        settings.ENV,
        }

        # EXTRA FIELDS
        for key, value in record.__dict__.items():
            if key in _SKIP_KEYS or key in payload:
                continue
            if value in _SUPPRESS_VALUES:
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

        if record.stack_info:
            payload["stack_info"] = record.stack_info

        try:
            return json.dumps(payload, ensure_ascii=False, default=str)
        except Exception:
            return json.dumps({
                "timestamp": datetime.now(tz=timezone.utc).isoformat(),
                "level":     record.levelname,
                "event":     "LOG_SERIALIZE_FAILED",
                "error":     str(record.getMessage()),
            })


# STRUCTURED LOGGER WRAPPER

class StructuredLogger:

    def __init__(self, logger: logging.Logger) -> None:
        self._logger = logger

    def _build_extra(self, event: Optional[str], **kwargs: Any) -> Dict[str, Any]:
        return {"event": event, **kwargs}

    def _log(
        self,
        level: str,
        event: Optional[str] = None,
        exc_info: bool = False,
        **kwargs: Any,
    ) -> None:
        extra = self._build_extra(event, **kwargs)
        getattr(self._logger, level)(
            event or "log",
            extra=extra,
            exc_info=exc_info,
        )

    def debug(self, event: Optional[str] = None, **kwargs: Any) -> None:
        self._log("debug", event, **kwargs)

    def info(self, event: Optional[str] = None, **kwargs: Any) -> None:
        self._log("info", event, **kwargs)

    def warning(self, event: Optional[str] = None, **kwargs: Any) -> None:
        self._log("warning", event, **kwargs)

    def error(self, event: Optional[str] = None, **kwargs: Any) -> None:
        self._log("error", event, **kwargs)

    def critical(self, event: Optional[str] = None, **kwargs: Any) -> None:
        self._log("critical", event, **kwargs)

    def exception(self, event: Optional[str] = None, **kwargs: Any) -> None:
        extra = self._build_extra(event, **kwargs)
        self._logger.exception(event or "exception", extra=extra)

    def bind(self, **kwargs: Any) -> "BoundLogger":
        return BoundLogger(self, **kwargs)


# BOUND LOGGER — CARRIES FIXED FIELDS

class BoundLogger:

    def __init__(self, parent: StructuredLogger, **bound: Any) -> None:
        self._parent = parent
        self._bound = bound

    def _merge(self, **kwargs: Any) -> Dict[str, Any]:
        return {**self._bound, **kwargs}

    def debug(self, event: Optional[str] = None, **kwargs: Any) -> None:
        self._parent.debug(event, **self._merge(**kwargs))

    def info(self, event: Optional[str] = None, **kwargs: Any) -> None:
        self._parent.info(event, **self._merge(**kwargs))

    def warning(self, event: Optional[str] = None, **kwargs: Any) -> None:
        self._parent.warning(event, **self._merge(**kwargs))

    def error(self, event: Optional[str] = None, **kwargs: Any) -> None:
        self._parent.error(event, **self._merge(**kwargs))

    def critical(self, event: Optional[str] = None, **kwargs: Any) -> None:
        self._parent.critical(event, **self._merge(**kwargs))

    def exception(self, event: Optional[str] = None, **kwargs: Any) -> None:
        self._parent.exception(event, **self._merge(**kwargs))


# LOG LEVEL RESOLUTION

def _get_log_level() -> int:
    level_str = str(getattr(settings, "LOG_LEVEL", "INFO")).upper()
    return getattr(logging, level_str, logging.INFO)


# FORMATTER FACTORY

def _build_formatter() -> logging.Formatter:
    use_json = getattr(settings, "LOG_JSON", False)
    if use_json:
        return JsonFormatter()
    use_color = not os.environ.get("NO_COLOR") and not os.environ.get("CI")
    return CleanFormatter(use_color=use_color)


# CONSOLE HANDLER

def _build_console_handler(
    level: int,
    formatter: logging.Formatter,
) -> logging.StreamHandler:
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)
    handler.setFormatter(formatter)
    return handler


# ROTATING FILE HANDLER

def _build_file_handler(
    level: int,
    formatter: logging.Formatter,
) -> Optional[RotatingFileHandler]:
    if not getattr(settings, "ENABLE_FILE_LOGGING", True):
        return None

    try:
        log_dir = Path(getattr(settings, "LOG_DIR", "./logs"))
        log_dir.mkdir(parents=True, exist_ok=True)

        log_file = log_dir / getattr(settings, "LOG_FILE_NAME", "app.log")

        handler = RotatingFileHandler(
            filename=str(log_file),
            maxBytes=getattr(settings, "LOG_MAX_BYTES", 10 * 1024 * 1024),
            backupCount=getattr(settings, "LOG_BACKUP_COUNT", 5),
            encoding="utf-8",
        )
        handler.setLevel(level)
        handler.setFormatter(formatter)
        return handler

    except Exception as exc:
        print(f"[LOGGER] Failed to create file handler: {exc}", file=sys.stderr)
        return None


# SUPPRESS NOISY THIRD-PARTY LOGGERS

_SUPPRESS_LOGGERS = (
    "httpx", "httpcore", "urllib3", "requests",
    "openai", "anthropic", "cohere",
    "sentence_transformers", "transformers",
    "faster_whisper", "torch",
    "PIL", "pdfplumber", "pydub",
    "pymongo", "motor",
    "qdrant_client", "grpc",
)


def _suppress_noisy_loggers() -> None:
    for name in _SUPPRESS_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)


# UVICORN INTEGRATION

def _configure_uvicorn_loggers(
    level: int,
    formatter: logging.Formatter,
) -> None:
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access", "fastapi"):
        uv_logger = logging.getLogger(name)
        uv_logger.handlers.clear()
        uv_logger.propagate = True
        uv_logger.setLevel(level)


# OPENTELEMETRY CONTEXT INJECTION

def _inject_otel_context() -> None:
    try:
        from opentelemetry import trace
        span = trace.get_current_span()
        ctx = span.get_span_context()
        if ctx and ctx.is_valid:
            trace_id_ctx.set(format(ctx.trace_id, "032x"))
            span_id_ctx.set(format(ctx.span_id, "016x"))
    except Exception:
        pass


# SETUP

def _setup_logging() -> None:
    global _LOGGER_INITIALIZED

    if _LOGGER_INITIALIZED:
        return

    level = _get_log_level()
    formatter = _build_formatter()

    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(level)

    # CONSOLE HANDLER
    console = _build_console_handler(level, formatter)
    root.addHandler(console)

    # FILE HANDLER
    file_handler = _build_file_handler(level, formatter)
    if file_handler:
        root.addHandler(file_handler)

    # UVICORN
    _configure_uvicorn_loggers(level, formatter)

    # SUPPRESS NOISY LIBRARIES
    _suppress_noisy_loggers()

    _LOGGER_INITIALIZED = True

    # STARTUP LOG
    root_logger = StructuredLogger(logging.getLogger("app.logger"))
    root_logger.info(
        event="logging_initialized",
        level=logging.getLevelName(level),
        json_mode=getattr(settings, "LOG_JSON", False),
        file_logging=getattr(settings, "ENABLE_FILE_LOGGING", True),
        service=settings.APP_NAME,
        version=settings.APP_VERSION,
        env=settings.ENV,
    )


# PUBLIC API

def get_logger(name: str) -> StructuredLogger:
    _setup_logging()
    return StructuredLogger(logging.getLogger(name))


def get_bound_logger(name: str, **bound: Any) -> BoundLogger:
    return get_logger(name).bind(**bound)


# HELPER UTILITIES

def log_latency(
    logger: StructuredLogger,
    event: str,
    start_time: float,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    try:
        payload: Dict[str, Any] = {
            "latency_sec": round(time.time() - start_time, 4),
        }
        if extra:
            payload.update(extra)
        logger.info(event=event, **payload)
    except Exception:
        pass


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
    threshold: Optional[float] = None,
) -> None:
    threshold = threshold or getattr(settings, "SLOW_REQUEST_THRESHOLD", 3.0)
    if latency > threshold:
        logger.warning(
            event="slow_request",
            path=path,
            latency_sec=round(latency, 3),
            threshold=threshold,
        )


def log_model_inference(
    logger: StructuredLogger,
    model_name: str,
    latency: float,
    tokens: int = 0,
    session_id: str = "",
) -> None:
    tps = round(tokens / max(latency, 1e-6), 1) if tokens else 0
    logger.info(
        event="model_inference",
        model=model_name,
        latency_sec=round(latency, 3),
        tokens=tokens,
        tokens_per_sec=tps,
        session_id=session_id,
    )


def log_embedding_batch(
    logger: StructuredLogger,
    model_name: str,
    batch_size: int,
    latency: float,
    session_id: str = "",
) -> None:
    throughput = round(batch_size / max(latency, 1e-6), 1)
    logger.info(
        event="embedding_batch",
        model=model_name,
        batch_size=batch_size,
        latency_sec=round(latency, 3),
        throughput_per_sec=throughput,
        session_id=session_id,
    )


def log_retrieval(
    logger: StructuredLogger,
    retriever: str,
    query_len: int,
    results: int,
    latency: float,
    session_id: str = "",
) -> None:
    logger.info(
        event="retrieval",
        retriever=retriever,
        query_len=query_len,
        results=results,
        latency_sec=round(latency, 3),
        session_id=session_id,
    )


def log_ingestion(
    logger: StructuredLogger,
    filename: str,
    modality: str,
    chunks: int,
    latency: float,
    session_id: str = "",
) -> None:
    logger.info(
        event="ingestion",
        filename=filename,
        modality=modality,
        chunks=chunks,
        latency_sec=round(latency, 3),
        session_id=session_id,
    )


def log_circuit_breaker(
    logger: StructuredLogger,
    service: str,
    state: str,
    failures: int = 0,
) -> None:
    level = "warning" if state == "open" else "info"
    getattr(logger, level)(
        event="circuit_breaker_state_change",
        service=service,
        state=state,
        failures=failures,
    )