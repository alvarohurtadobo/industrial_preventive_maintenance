"""Módulo FastAPI para recepción de datos de sensores en tiempo real."""

from .app import create_app
from .data_model import SensorDataModel

__all__ = ["create_app", "SensorDataModel"]

