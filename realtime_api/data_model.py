from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Dict


@dataclass(frozen=True)
class SensorDataModel:
    """Modelo serializable con valores de ejemplo para lecturas de sensores."""

    device_id: str = "PLC-MOTOR-001"
    timestamp: str = "2025-11-07T12:34:56.789Z"
    temp_01: float = 65.2
    temp_02: float = 63.9
    temp_03: float = 66.1
    curr_01: float = 12.4
    curr_02: float = 12.1
    curr_03: float = 11.8

    def to_dict(self) -> Dict[str, Any]:
        """Devuelve una representación serializable (dict) del modelo."""
        return asdict(self)

    @classmethod
    def example(cls) -> Dict[str, Any]:
        """Retorna el payload de ejemplo definido en el modelo."""
        return cls().to_dict()

    @classmethod
    def with_timestamp(cls, timestamp: datetime) -> Dict[str, Any]:
        """Genera un payload utilizando un timestamp dinámico."""
        ts = timestamp.astimezone(timezone.utc).isoformat()
        return cls(timestamp=ts).to_dict()

