# Dockerización de Realtime API

Este directorio contiene la configuración Docker para ejecutar la API de tiempo real.

## Requisitos Previos

- Docker instalado
- Docker Compose instalado (opcional, pero recomendado)

## Construcción y Ejecución

### Opción 1: Usando Docker Compose (Recomendado)

```bash
# Construir y ejecutar el contenedor
docker-compose up --build

# Ejecutar en segundo plano
docker-compose up -d --build

# Ver logs
docker-compose logs -f

# Detener el contenedor
docker-compose down
```

### Opción 2: Usando Docker directamente

```bash
# Construir la imagen
docker build -t realtime-api .

# Ejecutar el contenedor
docker run -d \
  --name realtime-api \
  -p 8000:8000 \
  -v $(pwd)/outputs:/app/outputs \
  -e FASTAPI_LOG_LEVEL=INFO \
  realtime-api

# Ver logs
docker logs -f realtime-api

# Detener el contenedor
docker stop realtime-api
docker rm realtime-api
```

## Variables de Entorno

Puedes personalizar el comportamiento de la API mediante variables de entorno:

- `FASTAPI_HOST`: Host donde escucha la API (default: `0.0.0.0`)
- `FASTAPI_PORT`: Puerto donde escucha la API (default: `8000`)
- `FASTAPI_RELOAD`: Habilitar recarga automática (default: `false`)
- `FASTAPI_LOG_LEVEL`: Nivel de logging (default: `INFO`)
- `FASTAPI_SENSOR_LOG_PATH`: Ruta donde se guardan los logs (default: `/app/outputs/fastapi_sensor_stream.jsonl`)
- `FASTAPI_APP_TITLE`: Título de la aplicación (default: `Industrial Preventive Maintenance API`)
- `FASTAPI_APP_VERSION`: Versión de la aplicación (default: `0.1.0`)

## Acceso a la API

Una vez que el contenedor esté en ejecución:

- **Documentación interactiva**: http://localhost:8000/docs
- **Documentación alternativa**: http://localhost:8000/redoc
- **Endpoint de ingesta**: POST http://localhost:8000/api/v1/sensors

## Volúmenes

El directorio `outputs` se monta como volumen para persistir los logs de sensores fuera del contenedor.

## Health Check

El contenedor incluye un health check que verifica que la API esté respondiendo correctamente.
