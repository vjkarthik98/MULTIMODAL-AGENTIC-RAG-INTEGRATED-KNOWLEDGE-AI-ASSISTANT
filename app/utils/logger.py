import logging
logging.basicConfig(level=logging.INFO)

def get_logger(name: str):
    logger = logging.getLogger(name)

    logger.setLevel(logging.INFO)


    # FORCE handler reset 
    if logger.hasHandlers():
        logger.handlers.clear()

    handler = logging.StreamHandler()
    handler.setLevel(logging.INFO)

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )
    
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    return logger
