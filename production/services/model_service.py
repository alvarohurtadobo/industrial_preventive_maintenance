"""Servicio para gestión y carga de modelos ML."""
import logging
import os
from pathlib import Path
from typing import Dict, Optional, Any

import joblib
from sklearn.base import BaseEstimator

logger = logging.getLogger(__name__)


class ModelService:
    """Servicio para cargar y gestionar modelos de machine learning."""

    def __init__(self, model_dir: Optional[str] = None, default_model: Optional[str] = None):
        """
        Inicializa el servicio de modelos.

        Args:
            model_dir: Directorio donde se encuentran los modelos. Default: ./models
            default_model: Nombre del modelo a usar por defecto. Default: RandomForest
        """
        self.model_dir = Path(model_dir or os.getenv("MODEL_DIR", "./models"))
        self.default_model_name = default_model or os.getenv("MODEL_NAME", "RandomForest")
        self._models: Dict[str, BaseEstimator] = {}
        self._loaded_models: set = set()

    def load_model(self, model_name: Optional[str] = None) -> BaseEstimator:
        """
        Carga un modelo desde disco.

        Args:
            model_name: Nombre del modelo a cargar. Si es None, usa el modelo por defecto.

        Returns:
            Modelo cargado (sklearn estimator)

        Raises:
            FileNotFoundError: Si el modelo no existe
            ValueError: Si hay error al cargar el modelo
        """
        model_name = model_name or self.default_model_name
        model_key = f"{model_name}_model"

        # Retornar modelo si ya está cargado en memoria
        if model_key in self._models:
            logger.debug("Modelo %s ya cargado en memoria", model_name)
            return self._models[model_key]

        # Construir ruta del modelo
        model_path = self.model_dir / f"{model_name}_model.pkl"

        if not model_path.exists():
            raise FileNotFoundError(
                f"Modelo {model_name} no encontrado en {model_path}"
            )

        try:
            logger.info("Cargando modelo %s desde %s", model_name, model_path)
            model = joblib.load(model_path)
            self._models[model_key] = model
            self._loaded_models.add(model_name)
            logger.info("Modelo %s cargado exitosamente", model_name)
            return model
        except Exception as e:
            logger.error("Error al cargar modelo %s: %s", model_name, e)
            raise ValueError(f"Error al cargar modelo {model_name}: {e}") from e

    def get_available_models(self) -> list[str]:
        """
        Retorna lista de modelos disponibles en el directorio.

        Returns:
            Lista de nombres de modelos disponibles
        """
        if not self.model_dir.exists():
            return []

        models = []
        for pkl_file in self.model_dir.glob("*_model.pkl"):
            # Extraer nombre del modelo: "RandomForest_model.pkl" -> "RandomForest"
            model_name = pkl_file.stem.replace("_model", "")
            models.append(model_name)

        return sorted(models)

    def predict(self, model_name: Optional[str], features: Any) -> tuple[int, float]:
        """
        Realiza una predicción usando el modelo especificado.

        Args:
            model_name: Nombre del modelo a usar. Si es None, usa el modelo por defecto.
            features: Features preprocesadas para la predicción

        Returns:
            Tupla (predicción, probabilidad) donde:
            - predicción: 0 (no fallo) o 1 (fallo)
            - probabilidad: probabilidad de fallo (0-1)
        """
        model = self.load_model(model_name)
        prediction = model.predict([features])[0]
        probability = model.predict_proba([features])[0][1]  # Probabilidad de clase 1 (fallo)

        return int(prediction), float(probability)

    def is_model_loaded(self, model_name: str) -> bool:
        """
        Verifica si un modelo está cargado en memoria.

        Args:
            model_name: Nombre del modelo

        Returns:
            True si el modelo está cargado, False en caso contrario
        """
        model_key = f"{model_name}_model"
        return model_key in self._models

