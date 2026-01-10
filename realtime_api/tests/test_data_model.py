"""Tests para el modelo de datos SensorDataModel."""
from datetime import datetime, timezone

import pytest

from realtime_api.data_model import SensorDataModel


class TestSensorDataModel:
    """Tests para la clase SensorDataModel."""

    def test_default_values(self):
        """Test que los valores por defecto son correctos."""
        model = SensorDataModel()
        assert model.device_id == "PLC-MOTOR-001"
        assert model.timestamp == "2025-11-07T12:34:56.789Z"
        assert model.temp_01 == 65.2
        assert model.temp_02 == 63.9
        assert model.temp_03 == 66.1
        assert model.curr_01 == 12.4
        assert model.curr_02 == 12.1
        assert model.curr_03 == 11.8

    def test_to_dict(self):
        """Test que to_dict() retorna un diccionario correcto."""
        model = SensorDataModel()
        result = model.to_dict()
        
        assert isinstance(result, dict)
        assert result["device_id"] == "PLC-MOTOR-001"
        assert result["temp_01"] == 65.2
        assert result["curr_01"] == 12.4

    def test_example(self):
        """Test que example() retorna un diccionario con valores por defecto."""
        result = SensorDataModel.example()
        
        assert isinstance(result, dict)
        assert "device_id" in result
        assert "timestamp" in result
        assert "temp_01" in result
        assert "curr_01" in result

    def test_with_timestamp(self):
        """Test que with_timestamp() genera un payload con timestamp dinámico."""
        test_timestamp = datetime(2025, 1, 15, 10, 30, 45, 123000, tzinfo=timezone.utc)
        result = SensorDataModel.with_timestamp(test_timestamp)
        
        assert isinstance(result, dict)
        assert result["timestamp"] == "2025-01-15T10:30:45.123000+00:00"
        assert result["device_id"] == "PLC-MOTOR-001"
        assert result["temp_01"] == 65.2

    def test_immutability(self):
        """Test que el modelo es inmutable (frozen dataclass)."""
        from dataclasses import FrozenInstanceError
        
        model = SensorDataModel()
        
        # frozen=True hace que sea inmutable y lanza FrozenInstanceError
        with pytest.raises(FrozenInstanceError):
            model.device_id = "NEW-DEVICE"
