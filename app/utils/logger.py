import logging
import sys
from pathlib import Path
from logging.handlers import RotatingFileHandler

from app.core.config import settings


_LOGGER_INITIALIZED = False


def _get_log_level():
    level = str(settings.LOG_LEVEL).upper()
    return getattr(logging, level, logging.INFO)


def _get_formatter():
    if getattr(settings, "LOG_JSON", False):
        # Simple JSON-like format
        return logging.Formatter(
            '{"time":"%(asctime)s","level":"%(levelname)s","name":"%(name)s","msg":"%(message)s"}',
            "%Y-%m-%d %H:%M:%S"
        )

    return logging.Formatter(
        "[%(asctime)s] | %(levelname)s | %(name)s | %(message)s",
        "%y-%m-%d %H:%M:%S"
    )


def _setup_root_logger():
    global _LOGGER_INITIALIZED

    if _LOGGER_INITIALIZED:
        return

    log_level = _get_log_level()
    formatter = _get_formatter()

    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    # Clear existing handlers (important in reloads)
    root_logger.handlers.clear()

    # CONSOLE 
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)

    root_logger.addHandler(console_handler)

    # FILE  
    if getattr(settings, "ENABLE_FILE_LOGGING", True):

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

        root_logger.addHandler(file_handler)

    _LOGGER_INITIALIZED = True


def get_logger(name: str) -> logging.Logger:
    _setup_root_logger()
    return logging.getLogger(name)