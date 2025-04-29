import sys
import logging
import logging.config
from pathlib import Path

# logging config
LOG_DIR = str(Path(__file__).parent.parent / "logs")
LOG_LEVEL = "INFO"

def setup_logging():
    """Setup comprehensive logging configuration"""
    Path(LOG_DIR).mkdir(parents=True, exist_ok=True)
    
    logging_config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "verbose": {
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s [%(filename)s:%(lineno)d]"
            },
            "simple": {
                "format": "%(asctime)s - %(levelname)s - %(message)s"
            },
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "level": LOG_LEVEL,
                "formatter": "simple",
                "stream": sys.stdout,
            },
            "file": {
                "class": "logging.handlers.RotatingFileHandler",
                "level": LOG_LEVEL,
                "formatter": "verbose",
                "filename": f"{LOG_DIR}/auto-commit.log",
                "maxBytes": 10 * 1024 * 1024,  # 10MB
                "backupCount": 5,
                "encoding": "utf8",
            },
        },
        "loggers": {
            "__main__": {
                "handlers": ["console", "file"],
                "level": LOG_LEVEL,
                "propagate": False,
            },
        },
        "root": {
            "level": "WARNING",
        },
    }

    logging.config.dictConfig(logging_config)