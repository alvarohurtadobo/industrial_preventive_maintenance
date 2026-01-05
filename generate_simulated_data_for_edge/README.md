# Simulated Data Generation for Edge Devices

This module generates simulated industrial sensor data in CSV format for anomaly detection training on edge devices.

## Overview

The generator creates CSV files with exactly 128 samples each, organized in separate directories for normal and anomaly operation. Each file contains a single column of sensor values (temperature, oil quality, or load depending on the process type).

## Installation

No additional dependencies beyond the standard project requirements:
- numpy
- pandas

## Usage

### Basic Usage

```bash
# Generate data with default parameters
python -m generate_simulated_data_for_edge.generate_data

# This will create:
# - datasets/normal_operation/ with normal operation CSV files
# - datasets/anomaly_operation/ with anomaly operation CSV files
```

### Command Line Arguments

```bash
python -m generate_simulated_data_for_edge.generate_data \
    --equipment 100 \
    --time-steps 40 \
    --samples-per-file 128 \
    --seed 42 \
    --output-normal datasets/normal_operation \
    --output-anomaly datasets/anomaly_operation \
    --save-complete inputs/emulated_data.csv
```

**Arguments:**
- `--equipment`: Number of equipment units (default: 100)
- `--time-steps`: Number of time steps per equipment (default: 40)
- `--samples-per-file`: Number of samples per CSV file (default: 128)
- `--seed`: Random seed for reproducibility (default: 42)
- `--output-normal`: Output directory for normal operation samples
- `--output-anomaly`: Output directory for anomaly operation samples
- `--save-complete`: Optional path to save complete dataset as single CSV

### Programmatic Usage

```python
from generate_simulated_data_for_edge.generate_data import DataGenerator

# Create generator
generator = DataGenerator(
    n_equipment=100,
    n_time_steps=40,
    samples_per_file=128,
    random_seed=42
)

# Generate all data
df = generator.generate_all_data()

# Save complete dataset
generator.save_complete_dataset(df, "inputs/emulated_data.csv")
```

## Output Structure

### Directory Structure

```
datasets/
├── normal_operation/
│   ├── equipment_001_sample_0000.csv
│   ├── equipment_001_sample_0001.csv
│   └── ...
└── anomaly_operation/
    ├── equipment_001_anomaly_0000.csv
    ├── equipment_001_anomaly_0001.csv
    └── ...
```

### CSV File Format

Each CSV file contains exactly 128 rows with a single column (no header):

```csv
20.345
20.521
20.678
...
```

## Data Generation Formulas

### Process Types

1. **Vibrations**
   - `vibration = sin(t/5) + N(0, 0.5)`
   - `temperature = 20 + 2*vibration + N(0, 0.5)`
   - `pressure = 30 + 3*(vibration²) + N(0, 1)`
   - Sensor value: temperature

2. **Oil Analysis**
   - `oil_quality = U(0,100) + t*0.1`
   - `contaminant_level = 50 + 0.5*oil_quality + N(0, 5)`
   - `acidity = 10 + 0.3*(oil_quality^1.5) + N(0, 2)`
   - Sensor value: oil_quality

3. **Hours Operated**
   - `hours_operated = Exp(50) + t*0.5`
   - `maintenance_history = Poisson(2)`
   - `load = 100 + 0.1*t + N(0, 10)`
   - Sensor value: load

### Anomaly Introduction

- **Probability**: 2% per sample
- **Vibrations**: Add `N(10, 5)` to vibration
- **Oil Analysis**: Add `U(50, 100)` to oil_quality
- **Hours Operated**: Add `U(50, 100)` to load

### Failure Calculation

- **Vibrations**: `failure = (0.3*vib + 0.2*temp - 0.1*pres + noise) > 1`
- **Oil Analysis**: `failure = (0.2*oil_q - 0.1*cont_level + 0.05*acid + noise) > 5`
- **Hours Operated**: `failure = (0.05*hours + 0.1*maint_hist - 0.02*load + noise) > 3`

## Expected Output

With default parameters (100 equipment, 40 time steps):
- **Total samples**: ~4,000 individual sensor readings
- **Normal files**: ~3,800 files (128 samples each)
- **Anomaly files**: ~200 files (128 samples each)
- **Anomaly rate**: ~2% of individual samples

## Validation

After generation, verify:
1. Each CSV file has exactly 128 rows
2. Files are correctly separated into normal/anomaly directories
3. Sensor values are within expected ranges
4. Anomaly rate is approximately 2%

## Integration with Training

The generated CSV files can be directly used with:
- Autoencoder-based anomaly detection
- Edge device training pipelines
- TensorFlow Lite model conversion

## Notes

- Data is synthetic and reflects realistic patterns but should be validated with real data
- Random seed ensures reproducibility
- NaN values are filled with column means in the complete dataset
- Each equipment has a single process type assigned randomly

