"""Logging module."""
import logging
from typing import Optional

logging.basicConfig(level=logging.INFO, stream=None)
for external_module in ("absl", "tensorflow"):
    logging.getLogger(external_module).disabled = True


def make_logger(name: str, filename: Optional[str] = None) -> logging.Logger:
    """Create a logger with configuration."""
    return configure_logger(logging.getLogger(name), filename)


def configure_logger(
    logger: logging.Logger, filename: Optional[str] = None
) -> logging.Logger:
    """Configure an already created logger."""
    default_format = "[ %(asctime)s | %(name)s ] %(message)s"
    s_format = logging.Formatter(default_format)
    s_handler = logging.StreamHandler()
    s_handler.setLevel(logging.INFO)
    s_handler.setFormatter(s_format)
    logger.addHandler(s_handler)
    logger.propagate = False

    if filename is not None:
        f_handler = logging.FileHandler(filename)
        f_handler.setLevel(logging.INFO)
        f_handler.setFormatter(logging.Formatter(default_format))
        logger.addHandler(f_handler)

    return logger
