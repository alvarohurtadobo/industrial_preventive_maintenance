"""API de predicción para mantenimiento predictivo industrial."""
import logging
import os
from datetime import datetime
from typing import Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from realtime_api.app import SensorPayload
from production.services.model_service import ModelService
from production.services.feature_engineering import FeatureEngineeringService
from production.services.preprocessing import PreprocessingService

logger = logging.getLogger(__name__)

# Router para los endpoints de predicción
router = APIRouter(prefix="/api/v1", tags=["prediction"])


class PredictionResponse(BaseModel):
    """Respuesta de predicción de fallo."""

    device_id: str = Field(..., description="ID del dispositivo")
    prediction: int = Field(..., description="Predicción: 0 (no fallo) o 1 (fallo)")
    probability: float = Field(..., description="Probabilidad de fallo (0-1)")
    model_used: str = Field(..., description="Nombre del modelo utilizado")
    timestamp: datetime = Field(..., description="Timestamp de la predicción")
    confidence: str = Field(..., description="Nivel de confianza: 'low', 'medium', 'high'")

    class Config:
        json_schema_extra = {
            "example": {
                "device_id": "PLC-MOTOR-001",
                "prediction": 0,
                "probability": 0.15,
                "model_used": "RandomForest",
                "timestamp": "2025-11-07T12:34:56.789Z",
                "confidence": "low",
            }
        }


class HealthResponse(BaseModel):
    """Respuesta del estado de salud de los modelos."""

    status: str = Field(..., description="Estado general: 'healthy' o 'unhealthy'")
    models_loaded: List[str] = Field(..., description="Lista de modelos cargados en memoria")
    models_available: List[str] = Field(..., description="Lista de modelos disponibles en disco")
    default_model: str = Field(..., description="Modelo por defecto configurado")

    class Config:
        json_schema_extra = {
            "example": {
                "status": "healthy",
                "models_loaded": ["RandomForest"],
                "models_available": ["RandomForest", "GradientBoosting", "SVM", "LogisticRegression"],
                "default_model": "RandomForest",
            }
        }


class ModelListResponse(BaseModel):
    """Respuesta con lista de modelos disponibles."""

    models: List[str] = Field(..., description="Lista de nombres de modelos disponibles")
    default_model: str = Field(..., description="Modelo por defecto")
    model_dir: str = Field(..., description="Directorio donde se encuentran los modelos")

    class Config:
        json_schema_extra = {
            "example": {
                "models": ["RandomForest", "GradientBoosting", "SVM", "LogisticRegression"],
                "default_model": "RandomForest",
                "model_dir": "./models",
            }
        }


# Dependencias para inyección de servicios
def get_model_service() -> ModelService:
    """Dependencia para obtener instancia del servicio de modelos."""
    return ModelService()


def get_feature_engineering_service() -> FeatureEngineeringService:
    """Dependencia para obtener instancia del servicio de feature engineering."""
    return FeatureEngineeringService()


def get_confidence_level(probability: float) -> str:
    """
    Determina el nivel de confianza basado en la probabilidad.

    Args:
        probability: Probabilidad de fallo (0-1)

    Returns:
        Nivel de confianza: 'low', 'medium', 'high'
    """
    if probability < 0.3:
        return "low"
    elif probability < 0.7:
        return "medium"
    else:
        return "high"


@router.post(
    "/predict",
    response_model=PredictionResponse,
    status_code=status.HTTP_200_OK,
    summary="Predicción de fallo de equipo",
    description=(
        "Recibe datos de sensores y devuelve una predicción de fallo del equipo. "
        "La predicción es binaria (0=no fallo, 1=fallo) con su probabilidad asociada."
    ),
)
def get_preprocessing_service() -> PreprocessingService:
    """Dependencia para obtener instancia del servicio de preprocesamiento."""
    return PreprocessingService()


def predict_failure(
    payload: SensorPayload,
    model_name: Optional[str] = None,
    model_service: ModelService = Depends(get_model_service),
    feature_service: FeatureEngineeringService = Depends(get_feature_engineering_service),
    preprocessing_service: PreprocessingService = Depends(get_preprocessing_service),
) -> PredictionResponse:
    """
    Endpoint para predecir fallos de equipos industriales.

    Args:
        payload: Datos del sensor (device_id, timestamp, temp_01-03, curr_01-03)
        model_name: Nombre del modelo a usar. Si no se especifica, usa el modelo por defecto.
        model_service: Servicio de modelos (inyectado)
        feature_service: Servicio de feature engineering (inyectado)

    Returns:
        PredictionResponse con la predicción y metadatos

    Raises:
        HTTPException: Si hay error al cargar el modelo o procesar los datos
    """
    try:
        logger.info(
            "Predicción solicitada para device_id=%s, modelo=%s",
            payload.device_id,
            model_name or "default",
        )

        # Transformar datos de sensores a features del modelo
        sensor_dict = payload.dict()
        features = feature_service.transform_sensor_data(sensor_dict)

        # Aplicar preprocesamiento (scaling)
        features_scaled = preprocessing_service.scale_features(features)

        # Realizar predicción
        prediction, probability = model_service.predict(model_name, features_scaled)

        # Determinar nivel de confianza
        confidence = get_confidence_level(probability)

        # Modelo usado (puede ser diferente al solicitado si se usa el default)
        actual_model = model_name or model_service.default_model_name

        response = PredictionResponse(
            device_id=payload.device_id,
            prediction=prediction,
            probability=probability,
            model_used=actual_model,
            timestamp=datetime.utcnow(),
            confidence=confidence,
        )

        logger.info(
            "Predicción completada: device_id=%s, prediction=%d, probability=%.4f",
            payload.device_id,
            prediction,
            probability,
        )

        return response

    except FileNotFoundError as e:
        logger.error("Modelo no encontrado: %s", e)
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Modelo no encontrado: {str(e)}",
        ) from e
    except ValueError as e:
        logger.error("Error al procesar predicción: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error al procesar predicción: {str(e)}",
        ) from e
    except Exception as e:
        logger.exception("Error inesperado en predicción: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error interno del servidor al procesar la predicción",
        ) from e


@router.get(
    "/models/health",
    response_model=HealthResponse,
    status_code=status.HTTP_200_OK,
    summary="Estado de salud de los modelos",
    description="Verifica el estado de los modelos ML y su disponibilidad.",
)
def get_models_health(
    model_service: ModelService = Depends(get_model_service),
) -> HealthResponse:
    """
    Endpoint para verificar el estado de salud de los modelos.

    Args:
        model_service: Servicio de modelos (inyectado)

    Returns:
        HealthResponse con el estado de los modelos
    """
    try:
        available_models = model_service.get_available_models()
        loaded_models = [
            name for name in available_models if model_service.is_model_loaded(name)
        ]

        # Intentar cargar el modelo por defecto si no está cargado
        if not loaded_models and available_models:
            try:
                model_service.load_model()
                loaded_models = [model_service.default_model_name]
            except Exception as e:
                logger.warning("No se pudo cargar el modelo por defecto: %s", e)

        status_value = "healthy" if loaded_models else "unhealthy"

        return HealthResponse(
            status=status_value,
            models_loaded=loaded_models,
            models_available=available_models,
            default_model=model_service.default_model_name,
        )

    except Exception as e:
        logger.exception("Error al verificar salud de modelos: %s", e)
        return HealthResponse(
            status="unhealthy",
            models_loaded=[],
            models_available=[],
            default_model=model_service.default_model_name,
        )


@router.get(
    "/models/list",
    response_model=ModelListResponse,
    status_code=status.HTTP_200_OK,
    summary="Lista de modelos disponibles",
    description="Retorna la lista de modelos ML disponibles en el sistema.",
)
def list_models(
    model_service: ModelService = Depends(get_model_service),
) -> ModelListResponse:
    """
    Endpoint para listar modelos disponibles.

    Args:
        model_service: Servicio de modelos (inyectado)

    Returns:
        ModelListResponse con la lista de modelos
    """
    try:
        available_models = model_service.get_available_models()

        return ModelListResponse(
            models=available_models,
            default_model=model_service.default_model_name,
            model_dir=str(model_service.model_dir),
        )

    except Exception as e:
        logger.exception("Error al listar modelos: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error al listar modelos disponibles",
        ) from e

