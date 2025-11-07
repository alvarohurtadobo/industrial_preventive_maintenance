import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

from fastapi import Depends, FastAPI, status
from pydantic import BaseModel, Field, field_validator

from .data_model import SensorDataModel


class SensorPayload(BaseModel):
    device_id: str = Field(..., description="Identificador único del dispositivo o sensor.")
    timestamp: datetime = Field(
        ..., description="Marca de tiempo en formato ISO 8601 de la lectura capturada."
    )
    temp_01: float = Field(..., description="Temperatura medida en el canal 1 (°C).")
    temp_02: float = Field(..., description="Temperatura medida en el canal 2 (°C).")
    temp_03: float = Field(..., description="Temperatura medida en el canal 3 (°C).")
    curr_01: float = Field(..., description="Corriente medida en el canal 1 (A).")
    curr_02: float = Field(..., description="Corriente medida en el canal 2 (A).")
    curr_03: float = Field(..., description="Corriente medida en el canal 3 (A).")

    @field_validator("temp_01", "temp_02", "temp_03", "curr_01", "curr_02", "curr_03", mode="before")
    @classmethod
    def validate_numeric(cls, value: Any) -> float:
        if isinstance(value, (float, int)):
            return float(value)
        raise TypeError("El valor debe ser un número (int o float).")

    class Config:
        json_schema_extra = {"examples": [SensorDataModel.example()]}


class Settings:
    def __init__(self) -> None:
        self.host: str = os.getenv("FASTAPI_HOST", "0.0.0.0")
        self.port: int = int(os.getenv("FASTAPI_PORT", "8000"))
        self.reload: bool = os.getenv("FASTAPI_RELOAD", "false").lower() == "true"
        self.log_path: Path = Path(
            os.getenv(
                "FASTAPI_SENSOR_LOG_PATH", "./outputs/fastapi_sensor_stream.jsonl"
            )
        )
        self.app_title: str = os.getenv(
            "FASTAPI_APP_TITLE", "Industrial Preventive Maintenance API"
        )
        self.app_version: str = os.getenv("FASTAPI_APP_VERSION", "0.1.0")


def get_settings() -> Settings:
    return Settings()


def setup_logging() -> logging.Logger:
    logger = logging.getLogger("sensor_ingestion")
    if logger.handlers:
        return logger

    level = os.getenv("FASTAPI_LOG_LEVEL", "INFO").upper()
    logging.basicConfig(
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=level
    )
    return logger


def persist_payload(settings: Settings, payload: Dict[str, Any], logger: logging.Logger) -> None:
    settings.log_path.parent.mkdir(parents=True, exist_ok=True)
    enriched_payload = {
        **payload,
        "received_at": datetime.utcnow().isoformat(timespec="milliseconds") + "Z",
    }

    with settings.log_path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(enriched_payload))
        stream.write("\n")

    logger.debug("Datos de sensor persistidos en %s", settings.log_path)


def create_app() -> FastAPI:
    settings = get_settings()
    logger = setup_logging()

    app = FastAPI(
        title=settings.app_title,
        version=settings.app_version,
        description=(
            "Servicio de ingesta para la recepción de datos de sensores industriales "
            "a través de un endpoint REST."
        ),
    )

    @app.post(
        "/api/v1/sensors",
        status_code=status.HTTP_202_ACCEPTED,
        summary="Ingesta de lecturas de sensores",
        response_description="Confirmación de recepción de la lectura.",
    )
    def ingest_sensor_data(
        payload: SensorPayload, current_settings: Settings = Depends(get_settings)
    ) -> Dict[str, str]:
        data = payload.dict()
        logger.info("Lectura recibida de %s a las %s", data["device_id"], data["timestamp"])
        persist_payload(current_settings, data, logger)
        return {"message": "Lectura aceptada"}

    return app


app = create_app()

