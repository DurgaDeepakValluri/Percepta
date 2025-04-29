import logging


def initialize_logger(name: str, level=logging.INFO) -> logging.Logger:
    """
    Initialize a logger with the specified name and level.

    Parameters:
    - name (str): Name of the logger.
    - level (int): Logging level (e.g., logging.INFO, logging.DEBUG).

    Returns:
    - logger (logging.Logger): Configured logger.
    """
    logger = logging.getLogger(name)
    if not logger.hasHandlers():
        handler = logging.StreamHandler()
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger