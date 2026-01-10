"""Tests para las funciones auxiliares de la aplicación."""
import json
import logging
import os
import tempfile
from datetime import datetime
from pathlib import Path

import pytest

from realtime_api.app import Settings, get_settings, persist_payload, setup_logging


class TestSettings:
    """Tests para la clase Settings."""

    def test_default_settings(self):
        """Test que los valores por defecto son correctos."""
        # Limpiar variables de entorno para probar defaults
        original_env = {}
        env_vars = [
            "FASTAPI_HOST",
            "FASTAPI_PORT",
            "FASTAPI_RELOAD",
            "FASTAPI_SENSOR_LOG_PATH",
            "FASTAPI_APP_TITLE",
            "FASTAPI_APP_VERSION",
        ]
        
        for var in env_vars:
            if var in os.environ:
                original_env[var] = os.environ[var]
                del os.environ[var]
        
        try:
            settings = Settings()
            assert settings.host == "0.0.0.0"
            assert settings.port == 8000
            assert settings.reload is False
            assert settings.log_path == Path("./outputs/fastapi_sensor_stream.jsonl")
            assert settings.app_title == "Industrial Preventive Maintenance API"
            assert settings.app_version == "0.1.0"
        finally:
            # Restaurar variables de entorno
            for var, value in original_env.items():
                os.environ[var] = value

    def test_custom_settings_from_env(self):
        """Test que las variables de entorno se leen correctamente."""
        original_env = {}
        env_vars = {
            "FASTAPI_HOST": "127.0.0.1",
            "FASTAPI_PORT": "9000",
            "FASTAPI_RELOAD": "true",
            "FASTAPI_SENSOR_LOG_PATH": "/tmp/test_log.jsonl",
            "FASTAPI_APP_TITLE": "Test API",
            "FASTAPI_APP_VERSION": "1.0.0",
        }
        
        for var in env_vars:
            if var in os.environ:
                original_env[var] = os.environ[var]
            os.environ[var] = env_vars[var]
        
        try:
            settings = Settings()
            assert settings.host == "127.0.0.1"
            assert settings.port == 9000
            assert settings.reload is True
            assert settings.log_path == Path("/tmp/test_log.jsonl")
            assert settings.app_title == "Test API"
            assert settings.app_version == "1.0.0"
        finally:
            # Restaurar variables de entorno
            for var in env_vars:
                if var in original_env:
                    os.environ[var] = original_env[var]
                elif var in os.environ:
                    del os.environ[var]

    def test_get_settings(self):
        """Test que get_settings() retorna una instancia de Settings."""
        settings = get_settings()
        assert isinstance(settings, Settings)


class TestLogging:
    """Tests para la configuración de logging."""

    def test_setup_logging(self):
        """Test que setup_logging() retorna un logger configurado."""
        logger = setup_logging()
        assert isinstance(logger, logging.Logger)
        assert logger.name == "sensor_ingestion"

    def test_setup_logging_idempotent(self):
        """Test que setup_logging() puede llamarse múltiples veces sin problemas."""
        logger1 = setup_logging()
        logger2 = setup_logging()
        
        # Debería retornar el mismo logger si ya tiene handlers
        assert logger1 is logger2 or logger1.name == logger2.name


class TestPersistPayload:
    """Tests para la función persist_payload."""

    def test_persist_payload_creates_file(self):
        """Test que persist_payload crea el archivo si no existe."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "test_log.jsonl"
            settings = Settings()
            settings.log_path = log_path
            logger = setup_logging()
            
            payload = {
                "device_id": "TEST-DEVICE",
                "timestamp": "2025-01-15T10:30:45.123Z",
                "temp_01": 65.2,
            }
            
            persist_payload(settings, payload, logger)
            
            assert log_path.exists()

    def test_persist_payload_writes_correct_data(self):
        """Test que persist_payload escribe los datos correctamente."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "test_log.jsonl"
            settings = Settings()
            settings.log_path = log_path
            logger = setup_logging()
            
            payload = {
                "device_id": "TEST-DEVICE",
                "timestamp": "2025-01-15T10:30:45.123Z",
                "temp_01": 65.2,
                "temp_02": 63.9,
            }
            
            persist_payload(settings, payload, logger)
            
            # Leer y verificar
            with log_path.open("r") as f:
                logged_data = json.loads(f.readline())
                
                assert logged_data["device_id"] == "TEST-DEVICE"
                assert logged_data["temp_01"] == 65.2
                assert "received_at" in logged_data
                assert isinstance(logged_data["received_at"], str)

    def test_persist_payload_appends_multiple_entries(self):
        """Test que persist_payload agrega múltiples entradas."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "test_log.jsonl"
            settings = Settings()
            settings.log_path = log_path
            logger = setup_logging()
            
            for i in range(3):
                payload = {"device_id": f"DEVICE-{i}", "timestamp": "2025-01-15T10:30:45.123Z"}
                persist_payload(settings, payload, logger)
            
            # Verificar que hay 3 líneas
            with log_path.open("r") as f:
                lines = f.readlines()
                assert len(lines) == 3

    def test_persist_payload_creates_parent_directories(self):
        """Test que persist_payload crea los directorios padre si no existen."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "nested" / "dir" / "test_log.jsonl"
            settings = Settings()
            settings.log_path = log_path
            logger = setup_logging()
            
            payload = {"device_id": "TEST", "timestamp": "2025-01-15T10:30:45.123Z"}
            persist_payload(settings, payload, logger)
            
            assert log_path.exists()
            assert log_path.parent.exists()
