"""Logging configuration for the fes-ml package."""
import logging
import logging.config
import os


def config_logger() -> None:
    """Configure the logger for the fes-ml package."""
    # Define log level
    log_level = os.environ.get("FES_ML_LOG_LEVEL", default="INFO").upper()

    # Set up basicConfig
    LOGGING = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "default": {
                "format": "%(asctime)s %(levelname)-8s %(name)-15s %(message)s",
                "datefmt": "%Y-%m-%d %H:%M:%S",
            }
        },
        "handlers": {
            "stdout": {
                "class": "logging.StreamHandler",
                "stream": "ext://sys.stdout",
                "formatter": "default",
            }
        },
        "loggers": {"": {"handlers": ["stdout"], "level": log_level}},
    }

    logging.config.dictConfig(LOGGING)

    # Try to use colorful logs
    try:
        import coloredlogs

        coloredlogs.install(level=getattr(logging, log_level))

    except ImportError:
        pass


config_logger()
