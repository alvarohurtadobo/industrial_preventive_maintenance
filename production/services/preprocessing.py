"""Servicio para preprocesamiento de features (scaler, encoder)."""
import logging
import os
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class PreprocessingService:
    """Servicio para aplicar preprocesamiento (scaling, encoding) a las features."""

    def __init__(self, model_dir: Optional[str] = None):
        """
        Inicializa el servicio de preprocesamiento.

        Args:
            model_dir: Directorio donde se encuentran los preprocesadores guardados
        """
        self.model_dir = Path(model_dir or os.getenv("MODEL_DIR", "./models"))
        self._scaler: Optional[StandardScaler] = None

    def load_scaler(self) -> StandardScaler:
        """
        Carga el StandardScaler desde disco o crea uno por defecto.

        Returns:
            StandardScaler cargado o nuevo
        """
        if self._scaler is not None:
            return self._scaler

        scaler_path = self.model_dir / "scaler.pkl"

        if scaler_path.exists():
            try:
                logger.info("Cargando scaler desde %s", scaler_path)
                self._scaler = joblib.load(scaler_path)
                logger.info("Scaler cargado exitosamente")
                return self._scaler
            except Exception as e:
                logger.warning("Error al cargar scaler, usando scaler por defecto: %s", e)

        # Si no existe o hay error, crear un nuevo scaler
        # NOTA: En producción real, el scaler DEBE ser el mismo usado en entrenamiento
        logger.warning(
            "Scaler no encontrado en %s. Usando scaler por defecto. "
            "Las predicciones pueden ser inexactas si el scaler no coincide con el de entrenamiento.",
            scaler_path,
        )
        self._scaler = StandardScaler()
        return self._scaler

    def scale_features(self, features: np.ndarray) -> np.ndarray:
        """
        Aplica scaling a las features.

        Args:
            features: Array numpy con features sin escalar

        Returns:
            Array numpy con features escaladas
        """
        scaler = self.load_scaler()

        # Si el scaler fue entrenado, usar transform
        # Si es nuevo, usar fit_transform (aunque esto no es ideal en producción)
        if hasattr(scaler, "mean_") and scaler.mean_ is not None:
            return scaler.transform(features.reshape(1, -1))[0]
        else:
            logger.warning(
                "Scaler no entrenado. Aplicando fit_transform. "
                "Esto puede causar predicciones incorrectas."
            )
            return scaler.fit_transform(features.reshape(1, -1))[0]

