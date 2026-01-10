"""Configuración y fixtures compartidos para los tests."""
import json
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Generator

import pytest
from fastapi.testclient import TestClient

from realtime_api.app import Settings, create_app, get_settings


@pytest.fixture
def temp_log_file() -> Generator[Path, None, None]:
    """Crea un archivo temporal para los logs durante los tests."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        temp_path = Path(f.name)
    yield temp_path
    # Limpiar después del test
    if temp_path.exists():
        temp_path.unlink()


@pytest.fixture
def test_settings(temp_log_file: Path) -> Settings:
    """Crea una configuración de prueba con archivo de log temporal."""
    return Settings()


@pytest.fixture
def app(temp_log_file: Path):
    """Crea una instancia de la aplicación FastAPI para testing."""
    import os
    
    # Guardar valores originales
    original_log_path = os.environ.get("FASTAPI_SENSOR_LOG_PATH")
    
    # Configurar variable de entorno para el test
    os.environ["FASTAPI_SENSOR_LOG_PATH"] = str(temp_log_file)
    
    try:
        # Crear la app (Settings leerá la variable de entorno actual)
        app_instance = create_app()
        yield app_instance
    finally:
        # Restaurar valores originales
        if original_log_path:
            os.environ["FASTAPI_SENSOR_LOG_PATH"] = original_log_path
        elif "FASTAPI_SENSOR_LOG_PATH" in os.environ:
            del os.environ["FASTAPI_SENSOR_LOG_PATH"]


@pytest.fixture
def client(app) -> Generator[TestClient, None, None]:
    """Crea un cliente de prueba para la API."""
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def sample_sensor_payload() -> dict:
    """Payload de ejemplo para tests de sensores."""
    return {
        "device_id": "PLC-MOTOR-001",
        "timestamp": "2025-01-15T10:30:45.123Z",
        "temp_01": 65.2,
        "temp_02": 63.9,
        "temp_03": 66.1,
        "curr_01": 12.4,
        "curr_02": 12.1,
        "curr_03": 11.8,
    }


@pytest.fixture
def sample_sensor_payload_with_int_values() -> dict:
    """Payload de ejemplo con valores enteros para probar conversión."""
    return {
        "device_id": "PLC-MOTOR-002",
        "timestamp": "2025-01-15T10:30:45.123Z",
        "temp_01": 65,
        "temp_02": 64,
        "temp_03": 66,
        "curr_01": 12,
        "curr_02": 11,
        "curr_03": 10,
    }
