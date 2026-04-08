import logging
import os

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()


def get_logger(name: str):
    logger = logging.getLogger(name)

    # Prevent Duplicate Handlers
    if not logger.handlers:
        logger.setLevel(LOG_LEVEL)

        handler = logging.StreamHandler()
        handler.setLevel(LOG_LEVEL)

        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
        )
    
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    # Prevent propagation to root logger
    logger.propagate = False

    return logger
