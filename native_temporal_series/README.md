# Native Time Series Models

This module implements native time series models (LSTM and GRU) for predictive maintenance, following **Strategy 2** from the time series conversion plan.

## Overview

Unlike tabular models that use single data points, these models leverage temporal sequences to capture patterns and dependencies over time, making them more suitable for time series prediction.

## Features

- **LSTM Models**: Long Short-Term Memory networks for capturing long-term dependencies
- **GRU Models**: Gated Recurrent Units, more efficient than LSTM
- **Sequence-based Preprocessing**: Creates sliding windows from temporal data
- **Automatic Scaling**: Handles feature scaling for time series data
- **Equipment-aware**: Groups data by equipment ID to maintain temporal order

## Installation

Ensure you have TensorFlow installed:

```bash
pip install tensorflow
```

Or add to requirements.txt:

```txt
tensorflow>=2.10.0
```

## Usage

### Training Models

Train a model using the training script:

```bash
# Train both LSTM and GRU models
python -m native_temporal_series.train --input emulated_data.csv --model both --window-size 10 --epochs 50

# Train only LSTM
python -m native_temporal_series.train --input emulated_data.csv --model lstm --window-size 10

# Train only GRU
python -m native_temporal_series.train --input emulated_data.csv --model gru --window-size 15 --epochs 100
```

**Arguments:**
- `--input`: Input CSV file (default: `emulated_data.csv`)
- `--model`: Model type - `lstm`, `gru`, or `both` (default: `both`)
- `--window-size`: Number of time steps in each sequence (default: 10)
- `--epochs`: Number of training epochs (default: 50)
- `--batch-size`: Batch size for training (default: 32)

### Making Predictions

```python
from native_temporal_series.predict import load_predictor
import pandas as pd

# Load predictor
predictor = load_predictor(
    model_type='lstm',
    model_dir='models',
    window_size=10
)

# Load data
data = pd.read_csv('inputs/emulated_data.csv')

# Predict for specific equipment
results = predictor.predict_from_dataframe(
    data,
    equipment_id=1
)

# Access predictions
for result in results['results']:
    print(f"Equipment {result['equipment_id']}, "
          f"Time Step {result['time_step']}: "
          f"Prediction={result['prediction']}, "
          f"Probability={result['probability']:.4f}")
```

### Using Single Sequences

```python
import numpy as np

# Create a sequence (window_size, n_features)
sequence = np.array([
    # 10 time steps, each with n_features
    # ... your sequence data ...
])

# Predict
prediction, probability = predictor.predict_single_sequence(sequence)
print(f"Prediction: {prediction}, Probability: {probability:.4f}")
```

## Model Architecture

### LSTM Model
- Two LSTM layers (64 and 32 units)
- Dropout and BatchNormalization for regularization
- Dense layers for final classification
- Binary output (failure/no failure)

### GRU Model
- Two GRU layers (64 and 32 units)
- Similar architecture to LSTM but more efficient
- Same regularization and output structure

## Data Format

Input data should be a CSV with:
- `equipment_id`: Equipment identifier
- `time_step`: Time step index
- Feature columns: `vibration`, `temperature`, `pressure`, etc.
- `failure`: Target variable (0 or 1)

## Output Files

After training, the following files are saved in `models/`:
- `{model_type}_model.keras`: Trained Keras model
- `{model_type}_scaler.pkl`: Scaler used for preprocessing
- `{model_type}_best_model.keras`: Best model checkpoint (during training)

## Performance Considerations

- **Window Size**: Larger windows capture more context but require more data
- **Sequence Creation**: Sequences are created per equipment to maintain temporal order
- **Memory**: Large datasets may require batch processing
- **Training Time**: LSTM/GRU models take longer to train than tabular models

## Integration with Production

These models can be integrated with the production API by:
1. Loading the predictor in the API service
2. Maintaining a buffer of recent readings per equipment
3. Creating sequences from the buffer for prediction
4. Combining with tabular models using ensemble methods

See `PLAN_TIMESERIES_CONVERSION.md` for detailed integration plans.

## Troubleshooting

### TensorFlow Not Found
```bash
pip install tensorflow
```

### Insufficient Data
Ensure you have at least `window_size` samples per equipment.

### Shape Mismatch
Verify that feature names match between training and prediction.

### Memory Issues
Reduce batch size or window size if running out of memory.

## Next Steps

1. **Hyperparameter Tuning**: Experiment with different architectures
2. **Ensemble Methods**: Combine LSTM/GRU with tabular models
3. **Real-time Integration**: Integrate with production API
4. **Advanced Models**: Consider Transformer-based models for complex patterns

