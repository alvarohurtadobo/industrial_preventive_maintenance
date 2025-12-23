"""Servicio para transformación de features de sensores a formato de modelo."""
import logging
from typing import Dict, Any
import numpy as np

logger = logging.getLogger(__name__)


class FeatureEngineeringService:
    """Servicio para transformar datos de sensores a features del modelo."""

    def __init__(self):
        """Inicializa el servicio de feature engineering."""
        # Cache de estadísticas por device_id para cálculos acumulativos
        self._device_stats: Dict[str, Dict[str, Any]] = {}

    def transform_sensor_data(self, sensor_data: Dict[str, Any]) -> np.ndarray:
        """
        Transforma datos de sensores a features esperadas por el modelo.

        Mapeo propuesto:
        - temperature: promedio de temp_01, temp_02, temp_03
        - vibration: varianza de temperaturas (indicador de vibración)
        - pressure: función de corrientes
        - oil_quality, contaminant_level, acidity: estimados basados en patrones
        - hours_operated: acumulado por device_id
        - maintenance_history: contador de mantenimientos
        - load: promedio de corrientes

        Args:
            sensor_data: Diccionario con datos del sensor (device_id, temp_01-03, curr_01-03, timestamp)

        Returns:
            Array numpy con features transformadas
        """
        device_id = sensor_data["device_id"]
        temp_01 = sensor_data["temp_01"]
        temp_02 = sensor_data["temp_02"]
        temp_03 = sensor_data["temp_03"]
        curr_01 = sensor_data["curr_01"]
        curr_02 = sensor_data["curr_02"]
        curr_03 = sensor_data["curr_03"]

        # Inicializar estadísticas del dispositivo si no existen
        if device_id not in self._device_stats:
            self._device_stats[device_id] = {
                "hours_operated": 0.0,
                "maintenance_history": 0,
                "first_timestamp": sensor_data.get("timestamp"),
                "readings_count": 0,
            }

        stats = self._device_stats[device_id]
        stats["readings_count"] += 1

        # Calcular features
        temperature = np.mean([temp_01, temp_02, temp_03])
        vibration = np.var([temp_01, temp_02, temp_03])  # Varianza como indicador de vibración
        pressure = 30 + 3 * (vibration ** 2)  # Función estimada basada en vibración
        load = np.mean([curr_01, curr_02, curr_03])

        # Estimaciones basadas en patrones de temperatura y corriente
        oil_quality = max(0, min(100, 50 + (temperature - 20) * 2))
        contaminant_level = 50 + 0.5 * oil_quality
        acidity = 10 + 0.3 * (oil_quality ** 1.5)

        # Hours operated: incrementar basado en tiempo transcurrido
        # Por simplicidad, incrementamos por lectura (asumiendo 1 hora por lectura)
        # En producción real, calcular basado en timestamps
        stats["hours_operated"] += 1.0

        # Maintenance history: inicialmente 0, actualizable externamente
        maintenance_history = stats["maintenance_history"]

        # Construir array de features en el orden esperado por el modelo
        # Nota: Este orden debe coincidir con el orden de features usado en el entrenamiento
        # Si hay variables categóricas (process_type), se deben agregar después del encoding
        features = np.array([
            temperature,
            vibration,
            pressure,
            oil_quality,
            contaminant_level,
            acidity,
            stats["hours_operated"],
            maintenance_history,
            load,
        ])

        logger.debug(
            "Features transformadas para device_id=%s: temp=%.2f, vib=%.2f, load=%.2f",
            device_id,
            temperature,
            vibration,
            load,
        )

        return features

    def update_maintenance_history(self, device_id: str, maintenance_count: int) -> None:
        """
        Actualiza el historial de mantenimiento de un dispositivo.

        Args:
            device_id: ID del dispositivo
            maintenance_count: Nuevo contador de mantenimientos
        """
        if device_id not in self._device_stats:
            self._device_stats[device_id] = {
                "hours_operated": 0.0,
                "maintenance_history": 0,
                "readings_count": 0,
            }

        self._device_stats[device_id]["maintenance_history"] = maintenance_count
        logger.info(
            "Historial de mantenimiento actualizado para %s: %d",
            device_id,
            maintenance_count,
        )

    def get_device_stats(self, device_id: str) -> Dict[str, Any]:
        """
        Obtiene estadísticas acumuladas de un dispositivo.

        Args:
            device_id: ID del dispositivo

        Returns:
            Diccionario con estadísticas del dispositivo
        """
        return self._device_stats.get(device_id, {})

