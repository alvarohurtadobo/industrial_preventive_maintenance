# Configuración de Conda para Industrial Preventive Maintenance

Esta guía explica cómo configurar un entorno conda para el proyecto de mantenimiento predictivo industrial.

## Prerrequisitos

- Conda instalado (Miniconda o Anaconda)
- Python 3.8 o superior

## Opción 1: Crear Entorno desde Cero

### Paso 1: Crear el Entorno

```bash
# Crear entorno con Python 3.10 (recomendado)
conda create -n industrial_maintenance python=3.10 -y

# O con Python 3.9
conda create -n industrial_maintenance python=3.9 -y
```

### Paso 2: Activar el Entorno

```bash
conda activate industrial_maintenance
```

### Paso 3: Instalar Dependencias

```bash
# Navegar al directorio del proyecto
cd /path/to/industrial_preventive_maintenance

# Instalar dependencias principales
pip install -r requirements.txt
```

## Opción 2: Crear Entorno con Conda Environment File

### Crear environment.yml

Primero, crea un archivo `environment.yml` en la raíz del proyecto:

```yaml
name: industrial_maintenance
channels:
  - conda-forge
  - defaults
dependencies:
  - python=3.10
  - pip
  - numpy>=1.20.0
  - pandas>=1.3.0
  - scikit-learn>=1.0.0
  - matplotlib>=3.5.0
  - seaborn>=0.11.0
  - pip:
    - reportlab
    - svglib
    - ydata-profiling
    - imblearn
    - openpyxl
    - paho-mqtt
    - kedro
    - joblib
    - python-dotenv
    - fastapi
    - uvicorn
    - pytest
    - tensorflow>=2.10.0
```

### Crear el Entorno

```bash
# Crear entorno desde environment.yml
conda env create -f environment.yml

# Activar el entorno
conda activate industrial_maintenance
```

## Verificación de la Instalación

### Verificar Entorno Activo

```bash
# Verificar que el entorno está activo
conda info --envs

# Deberías ver un asterisco (*) junto a industrial_maintenance
```

### Verificar Instalación de Paquetes

```bash
# Verificar Python
python --version

# Verificar paquetes principales
python -c "import numpy; print('NumPy:', numpy.__version__)"
python -c "import pandas; print('Pandas:', pandas.__version__)"
python -c "import sklearn; print('Scikit-learn:', sklearn.__version__)"
python -c "import tensorflow as tf; print('TensorFlow:', tf.__version__)"
python -c "import fastapi; print('FastAPI:', fastapi.__version__)"
```

## Configuración de Variables de Entorno

### Crear Archivo .env

Copia el archivo `.env.example` y crea tu propio `.env`:

```bash
# Si existe .env.example
cp .env.example .env

# Edita .env con tus configuraciones
# nano .env  # o usa tu editor preferido
```

### Variables de Entorno Importantes

```bash
# MQTT Configuration (si usas server.py)
MQTT_BROKER=broker.emqx.io
MQTT_PORT=1883
MQTT_TOPIC=flutter/sensors/2

# FastAPI Configuration
FASTAPI_HOST=0.0.0.0
FASTAPI_PORT=8000
FASTAPI_RELOAD=false

# Prediction API Configuration
PREDICTION_API_HOST=0.0.0.0
PREDICTION_API_PORT=8001
PREDICTION_API_RELOAD=false

# Model Configuration
MODEL_DIR=./models
MODEL_NAME=RandomForest
```

## Estructura de Entornos por Módulo

Si prefieres entornos separados para diferentes módulos:

### Entorno Principal (Mantenimiento Predictivo)

```bash
conda create -n industrial_maintenance python=3.10 -y
conda activate industrial_maintenance
pip install -r requirements.txt
```

### Entorno para Generación de Datos (Edge Devices)

```bash
conda create -n data_generation python=3.10 -y
conda activate data_generation
pip install -r generate_simulated_data_for_edge/requirements.txt
```

### Entorno para Series Temporales

```bash
conda create -n timeseries python=3.10 -y
conda activate timeseries
pip install -r requirements.txt
# TensorFlow ya está incluido en requirements.txt principal
```

## Comandos Útiles de Conda

### Gestión de Entornos

```bash
# Listar todos los entornos
conda env list

# Activar entorno
conda activate industrial_maintenance

# Desactivar entorno
conda deactivate

# Eliminar entorno (cuidado)
conda env remove -n industrial_maintenance
```

### Actualización de Paquetes

```bash
# Actualizar conda
conda update conda

# Actualizar todos los paquetes del entorno
conda update --all

# Actualizar un paquete específico
conda update numpy
```

### Exportar Entorno

```bash
# Exportar lista de paquetes instalados
conda env export > environment.yml

# Exportar solo paquetes instalados con conda (sin pip)
conda env export --no-builds > environment.yml

# Exportar lista de paquetes pip
pip freeze > requirements.txt
```

## Solución de Problemas

### Error: "Conda command not found"

```bash
# En macOS/Linux, agregar conda al PATH
# Agregar a ~/.bashrc o ~/.zshrc:
export PATH="$HOME/anaconda3/bin:$PATH"
# o
export PATH="$HOME/miniconda3/bin:$PATH"

# Recargar shell
source ~/.bashrc  # o source ~/.zshrc
```

### Error: "Package conflicts"

```bash
# Resolver conflictos forzando reinstalación
conda install --force-reinstall package_name

# O usar mamba (más rápido para resolver dependencias)
conda install mamba -c conda-forge
mamba install package_name
```

### Error: "TensorFlow no se instala correctamente"

```bash
# Instalar TensorFlow con conda-forge
conda install -c conda-forge tensorflow

# O instalar con pip específicamente
pip install tensorflow>=2.10.0
```

### Error: "Permission denied"

```bash
# Si tienes problemas de permisos, instalar en modo usuario
pip install --user package_name

# O usar sudo (no recomendado)
sudo pip install package_name
```

## Verificación Final

### Ejecutar Tests Básicos

```bash
# Verificar que el proyecto funciona
python -c "from main import generate_simulated_temporal_data; print('OK')"

# Verificar API
python -c "from realtime_api.app import create_app; print('API OK')"

# Verificar modelos de series temporales
python -c "from native_temporal_series.models import create_lstm_model; print('Time Series OK')"
```

### Ejecutar Scripts del Proyecto

```bash
# Generar datos simulados
python main.py

# Generar datos para edge devices
python -m generate_simulated_data_for_edge.generate_data

# Ejecutar API de ingesta
python -m realtime_api

# Ejecutar API de predicción
python -m production.api
```

## Recomendaciones

1. **Usar Python 3.10**: Compatible con todas las dependencias
2. **Entorno Virtual**: Siempre usar entornos conda para aislar dependencias
3. **Versionado**: Mantener `environment.yml` actualizado
4. **Backup**: Exportar entorno antes de actualizaciones importantes
5. **Documentación**: Documentar cambios en dependencias

## Recursos Adicionales

- [Documentación de Conda](https://docs.conda.io/)
- [Conda Cheat Sheet](https://docs.conda.io/projects/conda/en/4.6.0/_downloads/52a95608c49671267e40c689e0bc00ca/conda-cheatsheet.pdf)
- [Mamba (Conda más rápido)](https://mamba.readthedocs.io/)

## Notas

- Si trabajas en múltiples proyectos, considera usar entornos separados
- Para producción, usa `conda env export --no-builds` para mayor portabilidad
- Mantén un `requirements.txt` y `environment.yml` actualizados
- Considera usar `pip-tools` para gestionar dependencias pip más complejas

