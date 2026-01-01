# Plan: Conversión de Modelos a Series Temporales para Producción

## Objetivo

Convertir los modelos de clasificación tabular existentes en `models/` para que funcionen eficientemente con datos de series temporales en tiempo real, aprovechando patrones temporales para mejorar las predicciones de fallos.

## Estado Actual

### Modelos Existentes
- **RandomForest_model.pkl**: Clasificador de bosque aleatorio
- **GradientBoosting_model.pkl**: Clasificador de boosting
- **SVM_model.pkl**: Máquina de vectores de soporte
- **LogisticRegression_model.pkl**: Regresión logística

### Limitaciones Actuales
1. **Modelos tabulares**: Reciben un solo punto de datos (instante)
2. **Sin contexto temporal**: No aprovechan historial de lecturas
3. **Features estáticas**: Solo usan valores instantáneos
4. **Sin memoria**: Cada predicción es independiente

### Datos de Entrada Actuales
- `device_id`: Identificador del dispositivo
- `timestamp`: Marca de tiempo
- `temp_01`, `temp_02`, `temp_03`: Temperaturas
- `curr_01`, `curr_02`, `curr_03`: Corrientes

## Estrategias de Conversión

### Estrategia 1: Ventana Deslizante con Modelos Existentes (Recomendada para inicio rápido)

**Enfoque**: Usar modelos actuales con features temporales extraídas de ventanas deslizantes.

**Ventajas**:
- Reutiliza modelos existentes
- Implementación rápida
- Mantiene interpretabilidad
- Bajo costo computacional

**Desventajas**:
- Limitado a patrones de ventana fija
- No captura dependencias de largo plazo

### Estrategia 2: Modelos de Series Temporales Nativos (Recomendada para largo plazo)

**Enfoque**: Entrenar nuevos modelos específicos para series temporales (LSTM, GRU, Transformer).

**Ventajas**:
- Captura patrones temporales complejos
- Aprovecha dependencias de largo plazo
- Mejor para predicción de secuencias

**Desventajas**:
- Requiere retrenamiento completo
- Mayor complejidad
- Más recursos computacionales

### Estrategia 3: Híbrida (Recomendada para producción)

**Enfoque**: Combinar modelos tabulares con features temporales + modelos de series temporales.

**Ventajas**:
- Mejor de ambos mundos
- Ensamble para mayor robustez
- Fallback si un modelo falla

## Plan de Implementación Detallado

### Fase 1: Infraestructura de Almacenamiento Temporal

#### 1.1 Sistema de Buffer Temporal por Dispositivo

**Archivo**: `production/services/time_series_buffer.py`

**Funcionalidad**:
- Almacenar últimas N lecturas por `device_id`
- Buffer en memoria con límite de tiempo (ej: últimas 24 horas)
- Persistencia opcional en base de datos para análisis histórico

**Estructura de Datos**:
```python
{
    "device_id": {
        "readings": [
            {
                "timestamp": "2025-11-07T12:34:56Z",
                "temp_01": 65.2,
                "temp_02": 63.9,
                "temp_03": 66.1,
                "curr_01": 12.4,
                "curr_02": 12.1,
                "curr_03": 11.8
            },
            # ... más lecturas
        ],
        "max_size": 100,  # Ventana máxima
        "ttl_hours": 24   # Time to live
    }
}
```

**Tareas**:
- [ ] Crear clase `TimeSeriesBuffer`
- [ ] Implementar métodos: `add_reading()`, `get_window()`, `cleanup_old()`
- [ ] Agregar persistencia opcional a SQLite/PostgreSQL
- [ ] Tests unitarios

#### 1.2 Base de Datos para Historial Temporal

**Archivo**: `production/services/time_series_storage.py`

**Funcionalidad**:
- Almacenar todas las lecturas históricas
- Consultas eficientes por rango de tiempo
- Agregaciones temporales (promedios, tendencias)

**Schema Propuesto**:
```sql
CREATE TABLE sensor_readings (
    id SERIAL PRIMARY KEY,
    device_id VARCHAR(100) NOT NULL,
    timestamp TIMESTAMP NOT NULL,
    temp_01 FLOAT,
    temp_02 FLOAT,
    temp_03 FLOAT,
    curr_01 FLOAT,
    curr_02 FLOAT,
    curr_03 FLOAT,
    created_at TIMESTAMP DEFAULT NOW(),
    INDEX idx_device_time (device_id, timestamp)
);
```

**Tareas**:
- [ ] Diseñar schema de base de datos
- [ ] Implementar repositorio de datos
- [ ] Migraciones de base de datos
- [ ] Tests de integración

---

### Fase 2: Feature Engineering Temporal

#### 2.1 Extracción de Features Temporales

**Archivo**: `production/services/temporal_feature_engineering.py`

**Funcionalidad**:
- Extraer features estadísticas de ventanas temporales
- Calcular tendencias y derivadas
- Detectar patrones temporales

**Features Temporales a Extraer**:

1. **Estadísticas de Ventana** (últimas N lecturas):
   - Media, mediana, desviación estándar
   - Mínimo, máximo, rango
   - Percentiles (25, 50, 75)

2. **Tendencias**:
   - Pendiente de regresión lineal
   - Cambio porcentual
   - Aceleración (segunda derivada)

3. **Patrones Temporales**:
   - FFT para detectar frecuencias dominantes
   - Autocorrelación
   - Cambios de régimen

4. **Features de Comparación**:
   - Diferencia con lectura anterior
   - Diferencia con promedio de ventana
   - Z-score respecto a historial

**Implementación**:
```python
class TemporalFeatureEngineering:
    def extract_window_features(
        self, 
        window: List[Dict], 
        window_size: int = 10
    ) -> np.ndarray:
        """
        Extrae features temporales de una ventana de lecturas.
        
        Returns:
            Array con features: [instant_features, temporal_features]
        """
        # Features instantáneas (actuales)
        current = window[-1]
        instant_features = self._extract_instant_features(current)
        
        # Features temporales (de la ventana)
        temporal_features = self._extract_temporal_stats(window)
        temporal_features.extend(self._extract_trends(window))
        temporal_features.extend(self._extract_patterns(window))
        
        return np.concatenate([instant_features, temporal_features])
```

**Tareas**:
- [ ] Implementar extracción de estadísticas de ventana
- [ ] Calcular tendencias (regresión lineal)
- [ ] Detectar patrones (FFT, autocorrelación)
- [ ] Normalizar features temporales
- [ ] Tests unitarios

#### 2.2 Integración con Feature Engineering Existente

**Modificar**: `production/services/feature_engineering.py`

**Cambios**:
- Agregar método `transform_with_temporal_context()`
- Combinar features instantáneas con temporales
- Mantener compatibilidad con API actual

**Tareas**:
- [ ] Extender `FeatureEngineeringService`
- [ ] Agregar parámetro opcional `use_temporal=True`
- [ ] Mantener backward compatibility

---

### Fase 3: Adaptación de Modelos Existentes

#### 3.1 Wrapper para Modelos con Features Temporales

**Archivo**: `production/services/temporal_model_wrapper.py`

**Funcionalidad**:
- Envolver modelos existentes
- Agregar features temporales antes de predicción
- Manejar diferentes tamaños de ventana

**Implementación**:
```python
class TemporalModelWrapper:
    def __init__(
        self, 
        base_model: BaseEstimator,
        feature_engineer: TemporalFeatureEngineering,
        window_size: int = 10
    ):
        self.base_model = base_model
        self.feature_engineer = feature_engineer
        self.window_size = window_size
    
    def predict_with_window(
        self, 
        current_reading: Dict,
        window: List[Dict]
    ) -> tuple[int, float]:
        """
        Predice usando modelo base con features temporales.
        """
        # Extraer features temporales
        features = self.feature_engineer.extract_window_features(
            window + [current_reading]
        )
        
        # Predecir con modelo base
        prediction = self.base_model.predict([features])[0]
        probability = self.base_model.predict_proba([features])[0][1]
        
        return int(prediction), float(probability)
```

**Tareas**:
- [ ] Crear wrapper para modelos existentes
- [ ] Integrar con `TimeSeriesBuffer`
- [ ] Manejar casos edge (ventana incompleta)
- [ ] Tests de integración

#### 3.2 Retrenamiento con Features Temporales (Opcional)

**Archivo**: `production/scripts/retrain_with_temporal_features.py`

**Funcionalidad**:
- Re-entrenar modelos con features temporales
- Usar datos históricos con ventanas
- Comparar performance vs modelos originales

**Tareas**:
- [ ] Script de retrenamiento
- [ ] Generar dataset con ventanas temporales
- [ ] Evaluar mejora en métricas
- [ ] Versionado de modelos

---

### Fase 4: Modelos de Series Temporales Nativos

#### 4.1 Entrenamiento de Modelos LSTM/GRU

**Archivo**: `production/models/time_series_models.py`

**Modelos a Implementar**:
1. **LSTM (Long Short-Term Memory)**
   - Captura dependencias de largo plazo
   - Arquitectura: LSTM → Dense → Output

2. **GRU (Gated Recurrent Unit)**
   - Similar a LSTM pero más eficiente
   - Menos parámetros

3. **Transformer (Opcional)**
   - Attention mechanism
   - Mejor para patrones complejos

**Arquitectura Propuesta**:
```python
def create_lstm_model(
    input_shape: tuple,
    lstm_units: int = 64,
    dropout: float = 0.2
) -> tf.keras.Model:
    """
    Crea modelo LSTM para predicción de fallos.
    
    Args:
        input_shape: (window_size, n_features)
        lstm_units: Número de unidades LSTM
        dropout: Tasa de dropout
    
    Returns:
        Modelo Keras compilado
    """
    model = Sequential([
        LSTM(lstm_units, return_sequences=True, input_shape=input_shape),
        Dropout(dropout),
        LSTM(lstm_units // 2, return_sequences=False),
        Dropout(dropout),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')  # Clasificación binaria
    ])
    
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy', 'precision', 'recall']
    )
    
    return model
```

**Tareas**:
- [ ] Implementar modelos LSTM/GRU
- [ ] Script de entrenamiento
- [ ] Hyperparameter tuning
- [ ] Evaluación y comparación
- [ ] Guardar modelos entrenados

#### 4.2 Preprocesamiento para Modelos de Series Temporales

**Archivo**: `production/services/time_series_preprocessing.py`

**Funcionalidad**:
- Crear secuencias de ventanas deslizantes
- Normalización específica para series temporales
- Padding/truncamiento de secuencias

**Tareas**:
- [ ] Crear secuencias de entrenamiento
- [ ] Normalización por ventana
- [ ] Manejo de secuencias de longitud variable
- [ ] Tests

---

### Fase 5: Integración con API de Producción

#### 5.1 Modificar API de Predicción

**Modificar**: `production/api/prediction_api.py`

**Nuevos Endpoints**:
1. `POST /api/v1/predict/temporal` - Predicción con contexto temporal
2. `GET /api/v1/models/temporal/list` - Listar modelos temporales disponibles
3. `POST /api/v1/models/temporal/switch` - Cambiar modelo activo

**Cambios en Endpoint Existente**:
- Modificar `POST /api/v1/predict` para usar buffer temporal automáticamente
- Agregar parámetro `use_temporal=True/False`

**Implementación**:
```python
@router.post("/api/v1/predict/temporal")
def predict_with_temporal_context(
    payload: SensorPayload,
    window_size: int = 10,
    model_name: Optional[str] = None,
    buffer_service: TimeSeriesBuffer = Depends(get_buffer_service),
    temporal_model: TemporalModelWrapper = Depends(get_temporal_model)
) -> PredictionResponse:
    """
    Predicción usando contexto temporal.
    """
    # Obtener ventana histórica
    window = buffer_service.get_window(
        payload.device_id, 
        window_size=window_size
    )
    
    # Agregar lectura actual al buffer
    buffer_service.add_reading(payload.device_id, payload.dict())
    
    # Predecir con contexto temporal
    prediction, probability = temporal_model.predict_with_window(
        payload.dict(),
        window
    )
    
    return PredictionResponse(...)
```

**Tareas**:
- [ ] Agregar endpoints temporales
- [ ] Integrar `TimeSeriesBuffer` en API
- [ ] Agregar parámetros de configuración
- [ ] Tests de integración
- [ ] Documentación de API

#### 5.2 Configuración y Variables de Entorno

**Agregar a `.env`**:
```bash
# Time Series Configuration
TIMESERIES_WINDOW_SIZE=10
TIMESERIES_BUFFER_TTL_HOURS=24
TIMESERIES_USE_TEMPORAL=true
TIMESERIES_MODEL_TYPE=hybrid  # 'tabular', 'lstm', 'hybrid'
TIMESERIES_DB_PATH=./data/timeseries.db
```

**Tareas**:
- [ ] Agregar variables de entorno
- [ ] Actualizar `.env.example`
- [ ] Documentar configuración

---

### Fase 6: Sistema de Ensamble Híbrido

#### 6.1 Ensamble de Modelos

**Archivo**: `production/services/ensemble_service.py`

**Funcionalidad**:
- Combinar predicciones de modelos tabulares y temporales
- Pesos configurables por modelo
- Voting o stacking

**Estrategias**:
1. **Voting**: Mayoría simple o ponderada
2. **Stacking**: Meta-modelo que aprende a combinar
3. **Weighted Average**: Promedio ponderado de probabilidades

**Implementación**:
```python
class EnsembleService:
    def predict_ensemble(
        self,
        tabular_pred: tuple[int, float],
        temporal_pred: tuple[int, float],
        weights: Dict[str, float] = None
    ) -> tuple[int, float]:
        """
        Combina predicciones de múltiples modelos.
        """
        weights = weights or {"tabular": 0.4, "temporal": 0.6}
        
        # Promedio ponderado de probabilidades
        combined_prob = (
            weights["tabular"] * tabular_pred[1] +
            weights["temporal"] * temporal_pred[1]
        )
        
        prediction = 1 if combined_prob > 0.5 else 0
        
        return prediction, combined_prob
```

**Tareas**:
- [ ] Implementar ensamble
- [ ] Configurar pesos
- [ ] Evaluar performance
- [ ] Tests

---

### Fase 7: Monitoreo y Evaluación

#### 7.1 Métricas de Performance Temporal

**Archivo**: `production/services/temporal_metrics.py`

**Métricas a Implementar**:
- Accuracy por ventana de tiempo
- Latencia de predicción
- Uso de memoria del buffer
- Drift detection temporal

**Tareas**:
- [ ] Implementar métricas
- [ ] Dashboard de monitoreo
- [ ] Alertas de degradación

#### 7.2 A/B Testing

**Funcionalidad**:
- Comparar modelos tabulares vs temporales
- Métricas side-by-side
- Rollback automático si modelo nuevo falla

**Tareas**:
- [ ] Sistema de A/B testing
- [ ] Tracking de métricas
- [ ] Decisión automática de mejor modelo

---

## Arquitectura Final Propuesta

```
┌─────────────────┐
│  FastAPI        │
│  /predict       │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ TimeSeriesBuffer│ ◄─── Almacena últimas N lecturas
└────────┬────────┘
         │
         ▼
┌─────────────────────────────┐
│ TemporalFeatureEngineering  │ ◄─── Extrae features temporales
└────────┬────────────────────┘
         │
         ├─────────────────┬─────────────────┐
         ▼                 ▼                 ▼
┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│ Tabular      │  │ LSTM/GRU     │  │ Ensemble     │
│ Models       │  │ Models       │  │ Service      │
│ (Existing)   │  │ (New)        │  │              │
└──────────────┘  └──────────────┘  └──────────────┘
         │                 │                 │
         └─────────────────┴─────────────────┘
                           │
                           ▼
                  ┌──────────────┐
                  │ Prediction   │
                  │ Response     │
                  └──────────────┘
```

## Plan de Ejecución por Prioridad

### Prioridad Alta (MVP - 2-3 semanas)
1. ✅ Fase 1: Sistema de Buffer Temporal
2. ✅ Fase 2: Feature Engineering Temporal básico
3. ✅ Fase 3: Wrapper para modelos existentes
4. ✅ Fase 5: Integración con API

### Prioridad Media (Mejoras - 3-4 semanas)
5. ✅ Fase 4: Modelos LSTM/GRU
6. ✅ Fase 6: Sistema de Ensamble
7. ✅ Fase 1.2: Base de datos histórica

### Prioridad Baja (Optimización - 2-3 semanas)
8. ✅ Fase 7: Monitoreo avanzado
9. ✅ Fase 4.1: Modelos Transformer
10. ✅ Fase 3.2: Retrenamiento con features temporales

## Consideraciones Técnicas

### Performance
- **Latencia**: Buffer en memoria para acceso rápido (< 10ms)
- **Escalabilidad**: Redis para buffer distribuido si es necesario
- **Storage**: Base de datos para historial, buffer en memoria para predicción

### Compatibilidad
- Mantener backward compatibility con API actual
- Parámetro `use_temporal` para habilitar/deshabilitar
- Fallback a modelos tabulares si buffer está vacío

### Versionado
- Versionar modelos temporales separadamente
- Metadata de modelos (window_size, features usadas)
- Migración gradual de modelos

## Dependencias Adicionales

```txt
# Time Series
tensorflow>=2.10.0  # Para LSTM/GRU
keras>=2.10.0
tslearn>=0.6.0  # Utilidades de series temporales

# Storage
sqlalchemy>=2.0.0  # ORM para base de datos
alembic>=1.10.0  # Migraciones

# Optional
redis>=4.5.0  # Buffer distribuido
prometheus-client>=0.16.0  # Métricas
```

## Testing Strategy

### Unit Tests
- `TimeSeriesBuffer`: Agregar, obtener, limpiar
- `TemporalFeatureEngineering`: Extracción de features
- `TemporalModelWrapper`: Predicción con ventanas

### Integration Tests
- API con buffer temporal
- Flujo completo: lectura → buffer → predicción
- Persistencia en base de datos

### Performance Tests
- Latencia de predicción con diferentes window sizes
- Uso de memoria del buffer
- Throughput de la API

## Documentación

### Archivos a Crear
1. `docs/TIMESERIES_GUIDE.md` - Guía de uso
2. `docs/API_TIMESERIES.md` - Documentación de API
3. `docs/ARCHITECTURE_TIMESERIES.md` - Arquitectura detallada

## Riesgos y Mitigaciones

### Riesgo 1: Aumento de Latencia
- **Mitigación**: Buffer en memoria, procesamiento asíncrono

### Riesgo 2: Uso de Memoria
- **Mitigación**: TTL en buffer, límites de tamaño, persistencia a DB

### Riesgo 3: Complejidad de Features Temporales
- **Mitigación**: Feature engineering incremental, validación de features

### Riesgo 4: Modelos LSTM Requieren Muchos Datos
- **Mitigación**: Empezar con modelos tabulares + features temporales, LSTM después

## Métricas de Éxito

1. **Mejora en Accuracy**: +5-10% vs modelos tabulares
2. **Latencia**: < 50ms para predicción con contexto temporal
3. **Uptime**: > 99.9% disponibilidad
4. **Adopción**: 100% de predicciones usando contexto temporal en 3 meses

---

## Próximos Pasos Inmediatos

1. **Revisar y aprobar plan**
2. **Crear issues/tickets para cada fase**
3. **Asignar recursos y timeline**
4. **Comenzar con Fase 1 (Buffer Temporal)**

---

*Plan creado: 2025*
*Última actualización: 2025*

