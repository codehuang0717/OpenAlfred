import os
import sys
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

# Try to import colorlog for colored console output
try:
    import colorlog
    HAS_COLORLOG = True
except ImportError:
    HAS_COLORLOG = False

# Default configuration
LOG_FORMAT = "[%(asctime)s] [%(levelname)s] [%(name)s] - %(message)s"
COLOR_FORMAT = "%(log_color)s[%(asctime)s] [%(levelname)s] [%(name)s]%(reset)s - %(message)s"

_initialized = False

def setup_logging(
    level=logging.INFO,
    log_file="agent.log",
    max_bytes=10 * 1024 * 1024,  # 10MB
    backup_count=5
):
    """
    Sets up a unified logging system for OpenAlfred.
    Configures both console and file output.
    """
    global _initialized
    if _initialized:
        return
    
    # Create logs directory if it doesn't exist
    log_dir = Path(__file__).parent.parent.parent / "logs"
    log_dir.mkdir(exist_ok=True)
    log_path = log_dir / log_file

    # Root logger configuration
    root_logger = logging.getLogger()
    
    # Remove existing handlers to avoid duplicates
    if root_logger.hasHandlers():
        root_logger.handlers.clear()
    
    root_logger.setLevel(level)

    # Console Handler
    if HAS_COLORLOG:
        console_formatter = colorlog.ColoredFormatter(
            COLOR_FORMAT,
            datefmt="%Y-%m-%d %H:%M:%S",
            log_colors={
                'DEBUG': 'cyan',
                'INFO': 'green',
                'WARNING': 'yellow',
                'ERROR': 'red',
                'CRITICAL': 'red,bg_white',
            }
        )
    else:
        console_formatter = logging.Formatter(LOG_FORMAT, datefmt="%Y-%m-%d %H:%M:%S")

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)

    # File Handler (Always use standard format, no colors)
    file_formatter = logging.Formatter(LOG_FORMAT, datefmt="%Y-%m-%d %H:%M:%S")
    file_handler = RotatingFileHandler(
        log_path, 
        maxBytes=max_bytes, 
        backupCount=backup_count,
        encoding="utf-8"
    )
    file_handler.setFormatter(file_formatter)
    root_logger.addHandler(file_handler)

    # Silence some noisy third-party loggers
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)

    root_logger.info(f"Logging initialized. Outputting to console and {log_path}")
    _initialized = True

def get_logger(name):
    """
    Get a logger with the given name.
    """
    return logging.getLogger(name)
