"""Tests para los endpoints de la API."""
import json
from pathlib import Path

import pytest
from fastapi import status


class TestSensorIngestionEndpoint:
    """Tests para el endpoint POST /api/v1/sensors."""

    def test_successful_sensor_ingestion(
        self, client, sample_sensor_payload, temp_log_file
    ):
        """Test que el endpoint acepta correctamente un payload válido."""
        response = client.post("/api/v1/sensors", json=sample_sensor_payload)
        
        assert response.status_code == status.HTTP_202_ACCEPTED
        assert response.json() == {"message": "Lectura aceptada"}

    def test_sensor_data_persisted(
        self, client, sample_sensor_payload, temp_log_file
    ):
        """Test que los datos se persisten correctamente en el archivo de log."""
        response = client.post("/api/v1/sensors", json=sample_sensor_payload)
        
        assert response.status_code == status.HTTP_202_ACCEPTED
        
        # Verificar que el archivo se creó
        assert temp_log_file.exists()
        
        # Leer y verificar el contenido
        with temp_log_file.open("r") as f:
            lines = f.readlines()
            assert len(lines) == 1
            
            logged_data = json.loads(lines[0])
            assert logged_data["device_id"] == sample_sensor_payload["device_id"]
            assert logged_data["temp_01"] == sample_sensor_payload["temp_01"]
            assert "received_at" in logged_data  # Campo agregado por persist_payload

    def test_multiple_sensor_readings(
        self, client, sample_sensor_payload, temp_log_file
    ):
        """Test que múltiples lecturas se persisten correctamente."""
        # Enviar múltiples lecturas
        for i in range(3):
            payload = {**sample_sensor_payload, "device_id": f"DEVICE-{i}"}
            response = client.post("/api/v1/sensors", json=payload)
            assert response.status_code == status.HTTP_202_ACCEPTED
        
        # Verificar que todas se guardaron
        with temp_log_file.open("r") as f:
            lines = f.readlines()
            assert len(lines) == 3
            
            for i, line in enumerate(lines):
                data = json.loads(line)
                assert data["device_id"] == f"DEVICE-{i}"

    def test_int_values_converted_to_float(
        self, client, sample_sensor_payload_with_int_values, temp_log_file
    ):
        """Test que los valores enteros se convierten correctamente a float."""
        response = client.post(
            "/api/v1/sensors", json=sample_sensor_payload_with_int_values
        )
        
        assert response.status_code == status.HTTP_202_ACCEPTED
        
        # Verificar que los valores se guardaron como float
        with temp_log_file.open("r") as f:
            logged_data = json.loads(f.readline())
            assert isinstance(logged_data["temp_01"], float)
            assert isinstance(logged_data["curr_01"], float)

    def test_missing_required_field(self, client, sample_sensor_payload):
        """Test que falta un campo requerido retorna error 422."""
        payload = {**sample_sensor_payload}
        del payload["device_id"]
        
        response = client.post("/api/v1/sensors", json=payload)
        
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    def test_invalid_timestamp_format(self, client, sample_sensor_payload):
        """Test que un timestamp inválido retorna error 422."""
        payload = {**sample_sensor_payload, "timestamp": "invalid-timestamp"}
        
        response = client.post("/api/v1/sensors", json=payload)
        
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    def test_invalid_numeric_value(self, client, sample_sensor_payload):
        """Test que valores no numéricos retornan error 422."""
        payload = {**sample_sensor_payload, "temp_01": "not-a-number"}
        
        response = client.post("/api/v1/sensors", json=payload)
        
        # Pydantic v2 usa HTTP_422_UNPROCESSABLE_CONTENT en algunos casos
        assert response.status_code in (status.HTTP_422_UNPROCESSABLE_ENTITY, status.HTTP_422_UNPROCESSABLE_CONTENT)

    def test_negative_values_accepted(self, client, sample_sensor_payload):
        """Test que valores negativos son aceptados (pueden ser válidos en algunos casos)."""
        payload = {**sample_sensor_payload, "temp_01": -10.5}
        
        response = client.post("/api/v1/sensors", json=payload)
        
        assert response.status_code == status.HTTP_202_ACCEPTED

    def test_zero_values_accepted(self, client, sample_sensor_payload):
        """Test que valores cero son aceptados."""
        payload = {**sample_sensor_payload, "temp_01": 0.0, "curr_01": 0.0}
        
        response = client.post("/api/v1/sensors", json=payload)
        
        assert response.status_code == status.HTTP_202_ACCEPTED


class TestAPIDocumentation:
    """Tests para la documentación de la API."""

    def test_openapi_schema_available(self, client):
        """Test que el schema OpenAPI está disponible."""
        response = client.get("/openapi.json")
        
        assert response.status_code == status.HTTP_200_OK
        schema = response.json()
        assert "openapi" in schema
        assert "info" in schema
        assert schema["info"]["title"] == "Industrial Preventive Maintenance API"

    def test_docs_available(self, client):
        """Test que la documentación interactiva está disponible."""
        response = client.get("/docs")
        
        assert response.status_code == status.HTTP_200_OK
        assert "text/html" in response.headers["content-type"]

    def test_redoc_available(self, client):
        """Test que ReDoc está disponible."""
        response = client.get("/redoc")
        
        assert response.status_code == status.HTTP_200_OK
        assert "text/html" in response.headers["content-type"]
