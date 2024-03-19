"""fes_ml base package."""
__version__ = "0.1.0"
__author__ = "Joao Morado"

from .fes import FES
from .log import config_logger

__all__ = ["FES"]
