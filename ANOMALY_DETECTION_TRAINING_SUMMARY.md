# Anomaly Detection Training with Autoencoder - Summary

## Overview

This Jupyter notebook (`anomaly-detection-training-autoencoder.ipynb`) implements an anomaly detection system using an autoencoder neural network. The system is designed to detect anomalies in temperature sensor data (MAX6675) by learning the normal operating patterns and identifying deviations from them.

## Inputs

### 1. **Raw Data Files (CSV format)**
   - **Location**: `../datasets/ceiling-fan/`
   - **Format**: CSV files containing temperature measurements (1D data)
   - **Structure**: Each CSV file contains a single column of temperature values
   - **Expected samples per file**: 128 measurements (truncated if longer)
   - **Data organization**:
     - Normal operation data: `fan_0_low_0_weight-moving/`
     - Anomaly operation data: Multiple directories including:
       - `fan_0_med_0_weight/`
       - `fan_0_high_0_weight/`
       - `fan_0_low_1_weight/`
       - `fan_0_med_1_weight/`
       - `fan_0_high_1_weight/`

### 2. **Configuration Parameters**
   - `dataset_path`: Path to the dataset directory
   - `normal_op_list`: List of directories containing normal operation samples
   - `anomaly_op_list`: List of directories containing anomaly samples
   - `val_ratio`: Percentage of samples for validation (default: 0.2)
   - `test_ratio`: Percentage of samples for testing (default: 0.2)
   - `raw_scale`: Scaling factor for raw values (default: 1)
   - `sensor_sample_rate`: Original sensor sampling rate in Hz (default: 200)
   - `desired_sample_rate`: Target sampling rate in Hz (default: 50)
   - `sample_time`: Duration of each sample in seconds (default: 0.64)
   - `samples_per_file`: Expected number of measurements per file (default: 128)

## Processing Pipeline

### 1. **Data Loading and Preparation**
   - Loads CSV files from specified directories
   - Creates lists of normal and anomaly sample filenames
   - Shuffles the data randomly
   - Splits normal samples into training (60%), validation (20%), and test (20%) sets

### 2. **Feature Extraction**
   - **Function**: `extract_features(sample, max_measurements, scale)`
   - **Process**:
     1. Truncates samples to `max_measurements` (128 samples)
     2. Ensures data is 1D (handles temperature from MAX6675 - single column)
     3. Scales the data by `raw_scale` factor
     4. Calculates **Median Absolute Deviation (MAD)** as the feature
   - **Output**: Single feature value per sample (1D array)
   - **Note**: Adapted for temperature sensor (MAX6675) - processes 1D data instead of 3D accelerometer data

### 3. **Model Architecture**
   - **Type**: Autoencoder (unsupervised learning)
   - **Structure**:
     - Input layer: Shape determined by feature dimension (1D for temperature)
     - Encoder: Dense layer with 2 neurons, ReLU activation
     - Dropout: 0.2 dropout rate for regularization
     - Decoder: Dense layer reconstructing original input shape
   - **Total parameters**: ~17 (very lightweight for edge deployment)
   - **Optimizer**: Adam
   - **Loss function**: Mean Squared Error (MSE)

### 4. **Training Process**
   - **Training data**: Normal operation samples only (autoencoder learns normal patterns)
   - **Epochs**: 50
   - **Batch size**: 100
   - **Validation**: Uses validation set to monitor overfitting
   - **Objective**: Minimize reconstruction error (MSE) for normal samples

### 5. **Anomaly Detection Logic**
   - **Principle**: Anomalies have higher reconstruction error than normal samples
   - **Process**:
     1. Feed sample through trained autoencoder
     2. Calculate MSE between input and reconstructed output
     3. Compare MSE to threshold
     4. If MSE > threshold → Anomaly (1)
     5. If MSE ≤ threshold → Normal (0)
   - **Threshold calculation**: 
     - Based on validation set: `3 × std_dev + average_MSE`
     - Default threshold: `3e-05` (can be adjusted)

### 6. **Evaluation**
   - Calculates MSE for normal validation/test sets
   - Calculates MSE for anomaly test set
   - Generates histograms comparing normal vs. anomaly MSE distributions
   - Creates confusion matrix to evaluate classification performance
   - Visualizes training/validation loss curves

## Outputs

### 1. **Trained Model**
   - **File**: `models/fan_low_model-moving.h5`
   - **Format**: Keras H5 model file
   - **Usage**: Can be converted to TensorFlow Lite for edge deployment

### 2. **Sample Data Files**
   - **File**: `../test_samples/normal_anomaly_samples.npz`
   - **Contents**:
     - `normal_sample`: Raw temperature data from a normal sample
     - `anomaly_sample`: Raw temperature data from an anomaly sample
   - **Purpose**: For testing on microcontroller (MCU)

### 3. **Representative Dataset**
   - **File**: `../test_samples/normal_anomaly_test_set.npz`
   - **Contents**: `x_test` - Feature-extracted test set
   - **Purpose**: Used for model quantization and conversion to TensorFlow Lite

### 4. **Visualizations**
   - **Training/Validation Loss Plot**: Shows model convergence over epochs
   - **MSE Histograms**: 
     - Distribution of MSE values for normal samples
     - Distribution of MSE values for anomaly samples
     - Log scale comparison
   - **Confusion Matrix**: 
     - True Positives, True Negatives
     - False Positives, False Negatives
     - Performance metrics visualization

### 5. **Performance Metrics**
   - Average MSE for normal validation set
   - Standard deviation of MSE for normal validation set
   - Recommended threshold value
   - Confusion matrix results
   - Classification accuracy on test and anomaly sets

## Key Characteristics

- **Unsupervised Learning**: Only uses normal samples for training
- **Lightweight Model**: Small number of parameters suitable for edge devices
- **1D Input**: Adapted for temperature sensor (single feature: MAD)
- **Reconstruction-based**: Detects anomalies by measuring reconstruction error
- **Threshold-based Classification**: Simple binary classification based on MSE threshold

## Workflow Summary

```
CSV Files → Feature Extraction (MAD) → Train/Val/Test Split
    ↓
Train Autoencoder (normal samples only)
    ↓
Calculate MSE on validation set → Determine threshold
    ↓
Evaluate on test set → Generate confusion matrix
    ↓
Save model (.h5) + Sample files (.npz) + Visualizations
```

## Dependencies

- Python 3.7.6
- NumPy 1.18.1
- TensorFlow 2.1.0
- Keras 2.2.4-tf
- SciPy (for statistical functions)
- Matplotlib (for visualization)
- Seaborn (for confusion matrix visualization)
- scikit-learn (for confusion matrix calculation)
- Pandas (for data manipulation)

