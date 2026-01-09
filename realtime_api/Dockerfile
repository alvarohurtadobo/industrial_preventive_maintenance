# Usar imagen base de Python oficial
FROM python:3.11-slim

# Establecer directorio de trabajo
WORKDIR /app

# Instalar dependencias del sistema si son necesarias
RUN apt-get update && apt-get install -y --no-install-recommends \
    && rm -rf /var/lib/apt/lists/*

# Copiar requirements.txt e instalar dependencias de Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar el código de la aplicación
COPY . .

# Crear directorio para outputs si no existe
RUN mkdir -p /app/outputs

# Exponer el puerto que usa la aplicación
EXPOSE 8000

# Variables de entorno por defecto
ENV FASTAPI_HOST=0.0.0.0
ENV FASTAPI_PORT=8000
ENV FASTAPI_RELOAD=false
ENV FASTAPI_LOG_LEVEL=INFO
ENV FASTAPI_SENSOR_LOG_PATH=/app/outputs/fastapi_sensor_stream.jsonl

# Comando para ejecutar la aplicación
CMD ["python", "-m", "realtime_api"]
