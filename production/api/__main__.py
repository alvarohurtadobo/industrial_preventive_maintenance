"""Punto de entrada para ejecutar la API de predicción."""
import uvicorn
from fastapi import FastAPI
from production.api.prediction_api import router as prediction_router


def create_app() -> FastAPI:
    """Crea y configura la aplicación FastAPI."""
    app = FastAPI(
        title="Industrial Preventive Maintenance - Prediction API",
        version="1.0.0",
        description=(
            "API de predicción de fallos para mantenimiento predictivo industrial. "
            "Proporciona endpoints para predecir fallos de equipos basándose en datos de sensores."
        ),
    )

    # Incluir router de predicción
    app.include_router(prediction_router)

    return app


def main() -> None:
    """Función principal para ejecutar el servidor."""
    import os

    host = os.getenv("PREDICTION_API_HOST", "0.0.0.0")
    port = int(os.getenv("PREDICTION_API_PORT", "8001"))
    reload = os.getenv("PREDICTION_API_RELOAD", "false").lower() == "true"

    app = create_app()

    uvicorn.run(
        app,
        host=host,
        port=port,
        reload=reload,
    )


if __name__ == "__main__":
    main()

