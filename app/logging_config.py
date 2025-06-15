import logging
import os
from pathlib import Path
from logging.handlers import RotatingFileHandler

# Create logs directory if it doesn't exist
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)


def setup_logger(name: str, log_file: str, level=logging.INFO):
    """Configure a logger with file rotation"""

    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # File handler with rotation (5MB per file, max 3 backups)
    handler = RotatingFileHandler(
        filename=LOG_DIR / log_file,
        maxBytes=5 * 1024 * 1024,  # 5MB
        backupCount=3,
        encoding='utf-8'
    )
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)

    return logger