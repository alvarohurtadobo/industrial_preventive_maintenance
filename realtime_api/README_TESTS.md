# Tests para Realtime API

Este directorio contiene los tests unitarios e integración para la API de tiempo real.

## Estructura de Tests

```
tests/
├── __init__.py
├── conftest.py              # Fixtures y configuración compartida
├── test_api.py              # Tests para endpoints de la API
├── test_data_model.py       # Tests para el modelo de datos
└── test_app_functions.py    # Tests para funciones auxiliares
```

## Ejecutar Tests

### Instalar dependencias

```bash
pip install -r requirements.txt
```

### Ejecutar todos los tests

```bash
# Desde el directorio raíz del proyecto
pytest realtime_api/tests/

# O desde el directorio realtime_api
pytest tests/
```

### Ejecutar tests específicos

```bash
# Un archivo específico
pytest tests/test_api.py

# Una clase específica
pytest tests/test_api.py::TestSensorIngestionEndpoint

# Un test específico
pytest tests/test_api.py::TestSensorIngestionEndpoint::test_successful_sensor_ingestion
```

### Ejecutar con cobertura

```bash
# Instalar pytest-cov primero
pip install pytest-cov

# Ejecutar con cobertura
pytest tests/ --cov=realtime_api --cov-report=html --cov-report=term
```

### Ejecutar tests en modo verbose

```bash
pytest tests/ -v
```

### Ejecutar tests con marcadores

```bash
# Solo tests de API
pytest tests/ -m api

# Solo tests unitarios
pytest tests/ -m unit
```

## Ejecutar Tests en Docker

### Construir imagen con tests

```bash
docker build -t realtime-api-test -f realtime_api/Dockerfile .
```

### Ejecutar tests en contenedor

```bash
docker run --rm realtime-api-test pytest tests/
```

### Con docker-compose

Agregar al `docker-compose.yml`:

```yaml
services:
  test:
    build:
      context: ..
      dockerfile: realtime_api/Dockerfile
    command: pytest tests/ -v
    volumes:
      - ./tests:/app/tests
```

Luego ejecutar:

```bash
docker-compose run --rm test
```

## Cobertura de Tests

Los tests cubren:

- ✅ Endpoints de la API (`POST /api/v1/sensors`)
- ✅ Validación de payloads (campos requeridos, tipos de datos)
- ✅ Persistencia de datos en archivos JSONL
- ✅ Modelo de datos `SensorDataModel`
- ✅ Configuración y variables de entorno
- ✅ Funciones auxiliares (logging, persistencia)

## Escribir Nuevos Tests

### Estructura básica

```python
import pytest
from fastapi import status

def test_example(client, sample_sensor_payload):
    """Descripción del test."""
    response = client.post("/api/v1/sensors", json=sample_sensor_payload)
    assert response.status_code == status.HTTP_202_ACCEPTED
```

### Fixtures disponibles

- `client`: Cliente de prueba de FastAPI
- `app`: Instancia de la aplicación FastAPI
- `temp_log_file`: Archivo temporal para logs
- `test_settings`: Configuración de prueba
- `sample_sensor_payload`: Payload de ejemplo válido
- `sample_sensor_payload_with_int_values`: Payload con valores enteros

### Buenas prácticas

1. Usar fixtures para datos de prueba comunes
2. Limpiar archivos temporales después de cada test
3. Usar nombres descriptivos para los tests
4. Agrupar tests relacionados en clases
5. Probar casos límite y errores, no solo casos exitosos
