# Plan de Generación de Datos Simulados para Detección de Anomalías

## Objetivo

Generar datos simulados de sensores industriales en formato CSV para entrenamiento de modelos de detección de anomalías. Los datos simulan lecturas temporales de múltiples equipos con patrones normales y anómalos.

## Estructura de Datos

### Formato de Salida

Cada archivo CSV contiene una columna única de valores de temperatura (o sensor) con:
- **128 muestras por archivo** (truncado si es más largo)
- **Formato**: CSV con una sola columna de valores numéricos
- **Organización**: Directorios separados para operación normal y anómala

### Estructura de Directorios

```
datasets/
├── normal_operation/
│   ├── equipment_001_sample_001.csv
│   ├── equipment_001_sample_002.csv
│   └── ...
├── anomaly_operation/
│   ├── equipment_001_anomaly_001.csv
│   ├── equipment_001_anomaly_002.csv
│   └── ...
```

## Fórmulas de Generación de Datos

### 1. Datos de Operación Normal

#### Tipo: Vibraciones
```python
# Parámetros base
t = time_step  # Paso de tiempo (1, 2, 3, ..., n_time_steps)
seed = 42  # Para reproducibilidad

# Generación de vibración
vibration = sin(t / 5) + random.normal(0, 0.5)

# Temperatura correlacionada con vibración
temperature = 20 + 2 * vibration + random.normal(0, 0.5)

# Presión como función cuadrática de vibración
pressure = 30 + 3 * (vibration ** 2) + random.normal(0, 1)
```

#### Tipo: Análisis de Aceite
```python
# Calidad de aceite con tendencia temporal
oil_quality = random.uniform(0, 100) + t * 0.1

# Nivel de contaminante correlacionado
contaminant_level = 50 + 0.5 * oil_quality + random.normal(0, 5)

# Acidez como función potencia
acidity = 10 + 0.3 * (oil_quality ** 1.5) + random.normal(0, 2)
```

#### Tipo: Horas Operadas
```python
# Horas operadas con distribución exponencial y tendencia
hours_operated = random.exponential(scale=50) + t * 0.5

# Historial de mantenimiento (Poisson)
maintenance_history = random.poisson(lam=2)

# Carga con tendencia temporal
load = 100 + 0.1 * t + random.normal(0, 10)
```

### 2. Simulación de Fallos

#### Para Tipo Vibraciones
```python
failure = int((0.3 * vibration + 0.2 * temperature - 0.1 * pressure + random.normal(0, 0.5)) > 1)
```

#### Para Tipo Análisis de Aceite
```python
failure = int((0.2 * oil_quality - 0.1 * contaminant_level + 0.05 * acidity + random.normal(0, 1)) > 5)
```

#### Para Tipo Horas Operadas
```python
failure = int((0.05 * hours_operated + 0.1 * maintenance_history - 0.02 * load + random.normal(0, 1)) > 3)
```

### 3. Introducción de Anomalías

Las anomalías se introducen con una probabilidad del 2% en cada muestra:

```python
if random.rand() < 0.02:  # 2% probabilidad de anomalía
    anomaly = 1
    
    # Alteración según tipo de proceso
    if process_type == 'Vibrations':
        vibration += random.normal(10, 5)  # Incremento significativo
    
    elif process_type == 'Oil Analysis':
        oil_quality += random.uniform(50, 100)  # Incremento grande
    
    elif process_type == 'Hours Operated':
        load += random.uniform(50, 100)  # Incremento de carga
else:
    anomaly = 0
```

## Algoritmo de Generación

### Paso 1: Configuración Inicial

```python
# Parámetros de configuración
n_equipment = 100  # Número de equipos
n_time_steps = 40  # Muestras por equipo
samples_per_file = 128  # Muestras por archivo CSV
random.seed(42)  # Semilla para reproducibilidad
```

### Paso 2: Generación por Equipo

Para cada equipo (1 a n_equipment):

1. **Seleccionar tipo de proceso**:
   ```python
   process_type = random.choice(['Vibrations', 'Oil Analysis', 'Hours Operated'])
   ```

2. **Generar datos temporales**:
   - Para cada time_step (1 a n_time_steps):
     - Calcular variables según el tipo de proceso
     - Determinar si hay fallo usando fórmulas específicas
     - Introducir anomalía aleatoria (2% probabilidad)
     - Guardar registro

### Paso 3: Manejo de Valores Faltantes

```python
# Llenar NaN con la media de la columna
numeric_cols = data.select_dtypes(include=[np.number]).columns
data[numeric_cols] = data[numeric_cols].fillna(data[numeric_cols].mean())
```

### Paso 4: Exportación a CSV

#### Para Detección de Anomalías (Formato 1D)

Cada archivo CSV contiene una sola columna con valores de temperatura:

```python
# Extraer columna de temperatura (o sensor principal)
temperature_values = data['temperature'].values

# Truncar a 128 muestras si es necesario
if len(temperature_values) > samples_per_file:
    temperature_values = temperature_values[:samples_per_file]

# Guardar como CSV de una columna
pd.DataFrame(temperature_values).to_csv(
    f'equipment_{id}_sample_{sample_num}.csv',
    index=False,
    header=False
)
```

#### Para Mantenimiento Predictivo (Formato Completo)

```python
# Guardar todas las columnas
data.to_csv('emulated_data.csv', index=False)
```

## Implementación Completa

### Script de Generación

```python
import os
import numpy as np
import pandas as pd
from pathlib import Path

# Configuración
N_EQUIPMENT = 100
N_TIME_STEPS = 40
SAMPLES_PER_FILE = 128
RANDOM_SEED = 42
OUTPUT_DIR_NORMAL = "datasets/normal_operation"
OUTPUT_DIR_ANOMALY = "datasets/anomaly_operation"

np.random.seed(RANDOM_SEED)

# Crear directorios
Path(OUTPUT_DIR_NORMAL).mkdir(parents=True, exist_ok=True)
Path(OUTPUT_DIR_ANOMALY).mkdir(parents=True, exist_ok=True)

# Contadores
normal_sample_count = 0
anomaly_sample_count = 0

for equipment in range(1, N_EQUIPMENT + 1):
    # Seleccionar tipo de proceso
    process_type = np.random.choice(['Vibrations', 'Oil Analysis', 'Hours Operated'])
    
    for t in range(1, N_TIME_STEPS + 1):
        # Generar datos según tipo
        if process_type == 'Vibrations':
            vib = np.sin(t / 5) + np.random.normal(0, 0.5)
            temp = 20 + 2 * vib + np.random.normal(0, 0.5)
            pres = 30 + 3 * (vib ** 2) + np.random.normal(0, 1)
            
            # Valores NaN para otras variables
            oil_q = np.nan
            cont_level = np.nan
            acid = np.nan
            hours_op = np.nan
            maint_hist = np.nan
            ld = np.nan
            
            # Fallo
            fail = int((0.3 * vib + 0.2 * temp - 0.1 * pres + np.random.normal(0, 0.5)) > 1)
            
        elif process_type == 'Oil Analysis':
            oil_q = np.random.uniform(0, 100) + t * 0.1
            cont_level = 50 + 0.5 * oil_q + np.random.normal(0, 5)
            acid = 10 + 0.3 * (oil_q ** 1.5) + np.random.normal(0, 2)
            
            # Valores NaN para otras variables
            vib = np.nan
            temp = np.nan
            pres = np.nan
            hours_op = np.nan
            maint_hist = np.nan
            ld = np.nan
            
            # Fallo
            fail = int((0.2 * oil_q - 0.1 * cont_level + 0.05 * acid + np.random.normal(0, 1)) > 5)
            
        elif process_type == 'Hours Operated':
            hours_op = np.random.exponential(scale=50) + t * 0.5
            maint_hist = np.random.poisson(lam=2)
            ld = 100 + 0.1 * t + np.random.normal(0, 10)
            
            # Valores NaN para otras variables
            vib = np.nan
            temp = np.nan
            pres = np.nan
            oil_q = np.nan
            cont_level = np.nan
            acid = np.nan
            
            # Fallo
            fail = int((0.05 * hours_op + 0.1 * maint_hist - 0.02 * ld + np.random.normal(0, 1)) > 3)
        
        # Introducir anomalía (2% probabilidad)
        if np.random.rand() < 0.02:
            anomaly = 1
            if process_type == 'Vibrations':
                vib += np.random.normal(10, 5)
            elif process_type == 'Oil Analysis':
                oil_q += np.random.uniform(50, 100)
            elif process_type == 'Hours Operated':
                ld += np.random.uniform(50, 100)
        else:
            anomaly = 0
        
        # Crear registro
        record = {
            'equipment_id': equipment,
            'time_step': t,
            'process_type': process_type,
            'vibration': vib,
            'temperature': temp,
            'pressure': pres,
            'oil_quality': oil_q,
            'contaminant_level': cont_level,
            'acidity': acid,
            'hours_operated': hours_op,
            'maintenance_history': maint_hist,
            'load': ld,
            'failure': fail,
            'anomaly': anomaly
        }
        
        # Guardar como CSV individual (formato 1D para detección de anomalías)
        # Usar temperatura como variable principal
        if not np.isnan(temp):
            sensor_value = temp
        elif not np.isnan(oil_q):
            sensor_value = oil_q
        else:
            sensor_value = ld
        
        # Determinar si es normal o anómalo
        if anomaly == 0 and fail == 0:
            # Operación normal
            filename = f"{OUTPUT_DIR_NORMAL}/equipment_{equipment:03d}_sample_{normal_sample_count:04d}.csv"
            normal_sample_count += 1
        else:
            # Operación anómala
            filename = f"{OUTPUT_DIR_ANOMALY}/equipment_{equipment:03d}_anomaly_{anomaly_sample_count:04d}.csv"
            anomaly_sample_count += 1
        
        # Guardar como CSV de una columna (128 muestras máximo)
        # Para este ejemplo, guardamos cada registro individualmente
        # En producción, acumularíamos 128 muestras antes de guardar
        pd.DataFrame([sensor_value]).to_csv(filename, index=False, header=False)

# También guardar dataset completo para mantenimiento predictivo
# (acumular todos los registros y guardar en un solo CSV)
```

## Adaptación para Archivos de 128 Muestras

### Estrategia de Agrupación

Para generar archivos con exactamente 128 muestras:

```python
def generate_samples_in_batches(data, samples_per_file=128):
    """
    Agrupa datos en archivos de samples_per_file muestras.
    """
    samples = []
    file_count = 0
    
    for _, row in data.iterrows():
        # Extraer valor de sensor (temperatura, oil_quality, o load)
        sensor_value = row['temperature'] if not np.isnan(row['temperature']) else \
                      (row['oil_quality'] if not np.isnan(row['oil_quality']) else row['load'])
        
        samples.append(sensor_value)
        
        # Cuando alcanzamos samples_per_file, guardar archivo
        if len(samples) == samples_per_file:
            # Determinar si es normal o anómalo
            is_normal = (row['anomaly'] == 0 and row['failure'] == 0)
            
            if is_normal:
                filename = f"{OUTPUT_DIR_NORMAL}/normal_sample_{file_count:04d}.csv"
            else:
                filename = f"{OUTPUT_DIR_ANOMALY}/anomaly_sample_{file_count:04d}.csv"
            
            # Guardar archivo
            pd.DataFrame(samples).to_csv(filename, index=False, header=False)
            
            # Reset
            samples = []
            file_count += 1
    
    # Guardar muestras restantes
    if len(samples) > 0:
        # Rellenar o truncar a 128
        if len(samples) < samples_per_file:
            samples.extend([samples[-1]] * (samples_per_file - len(samples)))
        else:
            samples = samples[:samples_per_file]
        
        is_normal = (data.iloc[-1]['anomaly'] == 0 and data.iloc[-1]['failure'] == 0)
        if is_normal:
            filename = f"{OUTPUT_DIR_NORMAL}/normal_sample_{file_count:04d}.csv"
        else:
            filename = f"{OUTPUT_DIR_ANOMALY}/anomaly_sample_{file_count:04d}.csv"
        
        pd.DataFrame(samples).to_csv(filename, index=False, header=False)
```

## Parámetros Clave

### Distribuciones Estadísticas

- **Normal**: `np.random.normal(mean, std)`
- **Uniforme**: `np.random.uniform(low, high)`
- **Exponencial**: `np.random.exponential(scale)`
- **Poisson**: `np.random.poisson(lam)`

### Valores de Umbral para Fallos

- **Vibraciones**: Umbral = 1.0
- **Análisis de Aceite**: Umbral = 5.0
- **Horas Operadas**: Umbral = 3.0

### Probabilidades

- **Anomalía aleatoria**: 2% (0.02)
- **Tipo de proceso**: Uniforme entre 3 opciones

## Validación de Datos Generados

### Verificaciones

1. **Rango de valores**: Verificar que los valores estén en rangos esperados
2. **Distribución**: Verificar distribución de fallos y anomalías
3. **Tamaño de archivos**: Verificar que cada CSV tenga 128 muestras
4. **Separación normal/anómalo**: Verificar que los directorios estén correctamente separados

### Métricas Esperadas

- **Tasa de anomalías**: ~2% de muestras individuales
- **Tasa de fallos**: Variable según tipo de proceso
- **Distribución de tipos**: ~33% cada tipo (Vibrations, Oil Analysis, Hours Operated)

## Consideraciones de Implementación

### Rendimiento

- Usar `numpy` para operaciones vectorizadas
- Generar en lotes para grandes volúmenes
- Usar `pandas` para manejo eficiente de DataFrames

### Reproducibilidad

- Fijar semilla aleatoria (`np.random.seed(42)`)
- Documentar versión de librerías
- Guardar parámetros de generación

### Escalabilidad

- Para grandes volúmenes, considerar generación incremental
- Usar generadores para evitar cargar todo en memoria
- Paralelizar si es necesario

## Ejemplo de Uso

```python
# Generar datos
python generate_simulated_data.py

# Resultado esperado:
# datasets/normal_operation/ contiene ~3800 archivos CSV
# datasets/anomaly_operation/ contiene ~200 archivos CSV
# Cada archivo contiene 128 valores de temperatura (1 columna)
```

## Notas Finales

- Los datos simulados reflejan patrones realistas pero son sintéticos
- Para producción, validar con datos reales
- Ajustar parámetros según dominio específico
- Considerar variabilidad estacional o temporal adicional si es necesario

