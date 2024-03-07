import logging
import os


def setup_logging():
    # Define log level
    log_level = os.environ.get("FES_ML_LOG_LEVEL", default="INFO").upper()

    # Set up basicConfig
    logging.basicConfig(
        level=getattr(logging, log_level),
        format="%(name)-12s: %(levelname)-8s %(message)s",
    )

    # Get logger
    logger = logging.getLogger(__name__)

    # Try to use colorful logs
    try:
        import coloredlogs

        coloredlogs.install(level=getattr(logging, log_level), logger=logger)
    except ImportError:
        pass

    return logger


# Usage
logger = setup_logging()
