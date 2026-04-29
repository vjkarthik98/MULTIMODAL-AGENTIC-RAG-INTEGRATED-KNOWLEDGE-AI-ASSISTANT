import logging
import sys
import time
from pathlib import Path
from logging.handlers import RotatingFileHandler
from contextvars import ContextVar

from app.core.config import settings


# CONTEXT VARIABLES
request_id_ctx = ContextVar("request_id", default="-")
session_id_ctx = ContextVar("session_id", default="-")


_LOGGER_INITIALIZED = False


# CONTEXT FILTER
class ContextFilter(logging.Filter):
    def filter(self, record):
        record.request_id = request_id_ctx.get()
        record.session_id = session_id_ctx.get()
        return True


def _get_log_level():
    level = str(settings.LOG_LEVEL).upper()
    return getattr(logging, level, logging.INFO)


# SAFE FORMATTER
class SafeFormatter(logging.Formatter):
    def format(self, record):
        if not hasattr(record, "request_id"):
            record.request_id = "-"
        if not hasattr(record, "session_id"):
            record.session_id = "-"
        return super().format(record)


def _get_formatter():
    if settings.LOG_JSON:
        return SafeFormatter()

    return SafeFormatter(
        "[%(asctime)s] | %(levelname)s | %(name)s | %(message)s | req=%(request_id)s | session=%(session_id)s",
        "%Y-%m-%d %H:%M:%S"
    )


def _setup_root_logger():
    global _LOGGER_INITIALIZED

    if _LOGGER_INITIALIZED:
        return

    log_level = _get_log_level()
    formatter = _get_formatter()

    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    root_logger.handlers.clear()

    context_filter = ContextFilter()

    # CONSOLE HANDLER
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    console_handler.addFilter(context_filter)

    root_logger.addHandler(console_handler)

    # FILE HANDLER
    if settings.ENABLE_FILE_LOGGING:
        log_dir = Path(settings.LOG_DIR)
        log_dir.mkdir(parents=True, exist_ok=True)

        file_handler = RotatingFileHandler(
            log_dir / settings.LOG_FILE_NAME,
            maxBytes=settings.LOG_MAX_BYTES,
            backupCount=settings.LOG_BACKUP_COUNT,
            encoding="utf-8"
        )

        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        file_handler.addFilter(context_filter)

        root_logger.addHandler(file_handler)

    _LOGGER_INITIALIZED = True


def get_logger(name: str) -> logging.Logger:
    _setup_root_logger()
    return logging.getLogger(name)


# SET REQUEST CONTEXT 
def set_request_context(request_id: str = "-", session_id: str = "-"):
    request_id_ctx.set(request_id)
    session_id_ctx.set(session_id)


# LATENCY HELPER
def log_latency(logger, label: str, start_time: float, extra: dict = None):
    try:
        latency = round(time.time() - start_time, 3)
        payload = {"latency": latency}

        if extra:
            payload.update(extra)

        logger.info(f"{label} | latency={latency}s | extra={payload}")

    except Exception:
        logger.warning("LATENCY LOGGING FAILED")