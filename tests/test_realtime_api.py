import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

import pytest
from fastapi.testclient import TestClient

from realtime_api import SensorDataModel, create_app
from realtime_api.app import Settings, get_settings


@pytest.fixture()
def log_path(tmp_path: Path) -> Path:
    return tmp_path / "sensor_stream.jsonl"


@pytest.fixture()
def app_with_override(log_path: Path) -> Iterator[TestClient]:
    original_dependency = get_settings

    def override_settings() -> Settings:
        settings = Settings()
        settings.log_path = log_path
        settings.reload = False
        settings.host = "testserver"
        settings.port = 8000
        return settings

    app = create_app()
    app.dependency_overrides[get_settings] = override_settings

    client = TestClient(app)
    try:
        yield client
    finally:
        app.dependency_overrides.clear()


def test_ingest_sensor_data_persists_payload(app_with_override: TestClient, log_path: Path) -> None:
    payload = SensorDataModel.with_timestamp(datetime.now(timezone.utc))

    response = app_with_override.post("/api/v1/sensors", json=payload)

    assert response.status_code == 202
    assert response.json() == {"message": "Lectura aceptada"}
    assert log_path.exists()

    with log_path.open(encoding="utf-8") as stream:
        stored = json.loads(stream.readline())

    for key, value in payload.items():
        assert stored[key] == value
    assert "received_at" in stored


def test_ingest_sensor_data_rejects_invalid_numeric(app_with_override: TestClient) -> None:
    payload = SensorDataModel.with_timestamp(datetime.now(timezone.utc))
    payload["temp_01"] = "not-a-number"

    response = app_with_override.post("/api/v1/sensors", json=payload)

    assert response.status_code == 422

