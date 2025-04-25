import os
import logging
import logging.config
from pathlib import Path

# logging config
LOG_DIR = os.getenv("LOG_DIR", str(Path(__file__).parent / "logs"))
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

def setup_logging():
    """setup logging configuration"""
    
    Path(LOG_DIR).mkdir(parents=True, exist_ok=True)
    
    logging_config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "verbose": {
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s [%(filename)s:%(lineno)d]"
            },
            "simple": {
                "format": "%(levelname)s %(message)s"
            },
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "level": LOG_LEVEL,
                "formatter": "simple",
                "stream": "ext://sys.stdout",
            },
            "file": {
                "class": "logging.handlers.RotatingFileHandler",
                "level": LOG_LEVEL,
                "formatter": "verbose",
                "filename": f"{LOG_DIR}/auto-commit.log",
                "maxBytes": 1024 * 1024 * 10,  # 10MB
                "backupCount": 5,
                "encoding": "utf8",
            },
        },
        "loggers": {
            "src": {
                "handlers": ["console", "file"],
                "level": LOG_LEVEL,
                "propagate": False,
            },
        },
        "root": {
            "handlers": ["console"],
            "level": "WARNING",
        },
    }

    logging.config.dictConfig(logging_config)