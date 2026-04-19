import logging
import sys
from pathlib import Path
from logging.handlers import RotatingFileHandler

# GLOBAL SETTINGS 
LOG_LEVEL = logging.INFO
LOG_DIR = Path("logs")
LOG_FILE = LOG_DIR / "app.log"

# Ensure log directory exists
LOG_DIR.mkdir(parents=True, exist_ok=True)

# FORMATTER 
FORMAT = "[%(asctime)s] | %(levelname)s | %(name)s | %(message)s"
DATE_FORMAT = "%y-%m-%d %H:%M:%S"


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)

    if logger.handlers:
        return logger

    logger.setLevel(LOG_LEVEL)
    logger.propagate = False

    formatter = logging.Formatter(FORMAT, DATE_FORMAT)

    # CONSOLE HANDLER
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(LOG_LEVEL)
    console_handler.setFormatter(formatter)

    # FILE HANDLER 
    file_handler = RotatingFileHandler(
        LOG_FILE,
        maxBytes=5 * 1024 * 1024,
        backupCount=3,
        encoding="utf-8"
    )
    file_handler.setLevel(LOG_LEVEL)
    file_handler.setFormatter(formatter)

    # ADD HANDLERS
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger
