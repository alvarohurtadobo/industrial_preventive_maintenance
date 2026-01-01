# Industrial Preventive Maintenance - Main Pipeline Documentation

## Overview

The `main.py` file is the core pipeline for an industrial predictive maintenance system. It performs end-to-end machine learning workflow including data generation, exploratory data analysis (EDA), model training, evaluation, anomaly detection, and report generation.

## Purpose

This script automates the complete machine learning lifecycle for predicting equipment failures in industrial settings. It generates simulated temporal sensor data, trains multiple classification models, evaluates their performance, detects anomalies, and produces comprehensive reports.

## Architecture

The pipeline follows a sequential workflow:

```
Data Generation → Data Preprocessing → EDA → Model Training → 
Model Evaluation → Anomaly Detection → Report Generation
```

## Main Components

### 1. Configuration and Setup

#### Directory Structure
- `INPUTS_DIR`: Directory for input data files (default: `inputs/`)
- `RESULTS_DIR`: Directory for output files, plots, and reports (default: `outputs/`)
- `MODELS_DIR`: Directory for saved ML models (default: `models/`)
- `EXCEL_FILE`: Excel file for model evaluation metrics
- `PDF_REPORT`: PDF technical report file

#### `setup_directories()`
Creates necessary directories if they don't exist and initializes the Excel evaluation file.

**Returns:** None

---

### 2. Data Generation

#### `generate_simulated_temporal_data()`
Generates synthetic temporal data simulating industrial equipment sensor readings.

**Parameters:** None

**Returns:** `pd.DataFrame` with simulated sensor data

**Features Generated:**
- **Equipment ID**: Unique identifier for each equipment (1-100)
- **Time Step**: Sequential time steps (1-40 per equipment)
- **Process Type**: One of three types:
  - `Vibrations`: vibration, temperature, pressure
  - `Oil Analysis`: oil_quality, contaminant_level, acidity
  - `Hours Operated`: hours_operated, maintenance_history, load

**Data Characteristics:**
- 100 equipment units
- 40 time steps per equipment
- 2% anomaly rate
- Failure labels based on process-specific formulas
- Missing values filled with column means

**Key Variables:**
- `vibration`: Sinusoidal pattern with noise
- `temperature`: Correlated with vibration
- `pressure`: Quadratic function of vibration
- `oil_quality`: Uniform distribution with time trend
- `contaminant_level`: Linear function of oil quality
- `acidity`: Power function of oil quality
- `hours_operated`: Exponential distribution with time trend
- `maintenance_history`: Poisson distribution
- `load`: Normal distribution with time trend

#### `exportToCSV(data)`
Exports the generated data to CSV format.

**Parameters:**
- `data`: DataFrame to export

**Returns:** None

---

### 3. Data Preprocessing

#### `handle_data_types(data)`
Handles data type conversions and encoding.

**Parameters:**
- `data`: Input DataFrame

**Returns:** Preprocessed DataFrame

**Operations:**
1. Identifies categorical columns
2. Applies OneHotEncoder (version-aware for sklearn compatibility)
3. Converts all numeric columns to float
4. Validates that no categorical columns remain

**Note:** Handles sklearn version differences for `OneHotEncoder` parameters.

#### `preprocess_data(data)`
Main preprocessing function for model training.

**Parameters:**
- `data`: Input DataFrame

**Returns:** 
- `x_train`: Training features (numpy array)
- `x_test`: Testing features (numpy array)
- `y_train`: Training labels (numpy array)
- `y_test`: Testing labels (numpy array)
- `feature_names`: Column names of features (Index)

**Operations:**
1. Separates features (X) and target (y)
2. Drops non-feature columns: `failure`, `equipment_id`, `time_step`, `anomaly`
3. Applies `StandardScaler` for feature scaling
4. Uses SMOTE (Synthetic Minority Oversampling Technique) to balance classes
5. Splits data into train/test sets (70/30) with stratification

**Important:** The scaler is fitted but not saved. For production use, the scaler should be saved separately.

---

### 4. Exploratory Data Analysis (EDA)

#### `perform_eda(data)`
Performs comprehensive exploratory data analysis and generates visualizations.

**Parameters:**
- `data`: Input DataFrame

**Returns:** None

**Generated Visualizations:**
1. **Failure Distribution**: Count plot of failure vs non-failure
2. **Correlation Matrix**: Heatmap of feature correlations
3. **Histograms**: Distribution plots for all numeric variables
4. **Boxplots**: Outlier detection plots for all features
5. **Pairplot**: Pairwise relationships colored by failure status

**Output Files:**
- `dataset_EDA.html`: Interactive profiling report (ydata-profiling)
- `failure_distribution.png`
- `correlation_matrix.png`
- `histograms.png`
- `boxplots.png`
- `pairplot.png`

---

### 5. Model Training

#### `train_classification_models(x_train, y_train)`
Trains multiple classification models using GridSearchCV for hyperparameter tuning.

**Parameters:**
- `x_train`: Training features
- `y_train`: Training labels

**Returns:** Dictionary of best models (name -> model object)

**Models Trained:**
1. **RandomForest**
   - Hyperparameters: `n_estimators`, `max_depth`, `min_samples_split`
   - Scoring: ROC AUC
   - Cross-validation: 5-fold

2. **SVM (Support Vector Machine)**
   - Hyperparameters: `C`, `kernel`
   - Scoring: ROC AUC
   - Cross-validation: 5-fold

3. **GradientBoosting**
   - Hyperparameters: `n_estimators`, `learning_rate`, `max_depth`
   - Scoring: ROC AUC
   - Cross-validation: 5-fold

4. **LogisticRegression**
   - Hyperparameters: `C`, `penalty`
   - Scoring: ROC AUC
   - Cross-validation: 5-fold

**Optimization:**
- Uses `GridSearchCV` with 5-fold cross-validation
- Optimizes for ROC AUC score
- Parallel processing with `n_jobs=-1`

#### `export_models(best_models)`
Saves trained models to disk.

**Parameters:**
- `best_models`: Dictionary of trained models

**Returns:** None

**Output:** Saves each model as `{MODELS_DIR}/{model_name}_model.pkl`

---

### 6. Model Evaluation

#### `evaluate_classification_models(best_models, x_test, y_test)`
Evaluates trained models and saves comprehensive metrics.

**Parameters:**
- `best_models`: Dictionary of trained models
- `x_test`: Test features
- `y_test`: Test labels

**Returns:** Dictionary with model metrics

**Metrics Calculated:**
- Accuracy
- Precision
- Recall
- F1-score
- ROC AUC
- PR AUC (Average Precision)

**Outputs:**
1. **Excel File**: Classification reports for each model
2. **Confusion Matrices**: PNG images for each model
3. **Logs**: Performance metrics printed to console

#### `plot_classification_curves(model_metrics, y_test)`
Generates ROC and Precision-Recall curves for all models.

**Parameters:**
- `model_metrics`: Dictionary with model predictions and metrics
- `y_test`: True test labels

**Returns:** None

**Output:** `roc_pr_curves.png` with side-by-side plots

#### `plot_feature_importance(models, feature_names)`
Plots feature importance for tree-based models.

**Parameters:**
- `models`: Dictionary of trained models
- `feature_names`: Names of features

**Returns:** None

**Output:** `{model_name}_feature_importance.png` for RandomForest and GradientBoosting

---

### 7. Anomaly Detection

#### `detect_anomalies(data)`
Applies five different anomaly detection algorithms.

**Parameters:**
- `data`: Input DataFrame

**Returns:** DataFrame with anomaly detection results

**Algorithms Used:**
1. **Isolation Forest**: Contamination rate 0.02
2. **One-Class SVM**: Nu=0.02, RBF kernel
3. **Local Outlier Factor**: 20 neighbors, contamination 0.02
4. **DBSCAN**: eps=3, min_samples=5
5. **PCA-based Outlier Detection**: Top 2% as anomalies

**Preprocessing:**
- Encodes categorical variables (if present)
- Applies StandardScaler
- Handles sklearn version differences

**Outputs:**
1. **CSV File**: `anomaly_detection_results.csv` with predictions from all methods
2. **PNG Images**: Scatter plots for each detection method
3. **Metrics**: Precision, Recall, F1-score for each method

**Evaluation:**
Compares against ground truth `anomaly` column and logs performance metrics.

---

### 8. Report Generation

#### `create_pdf_report(data, model_metrics, feature_names, best_models, anomaly_results)`
Generates a comprehensive PDF technical report.

**Parameters:**
- `data`: Original dataset
- `model_metrics`: Model evaluation metrics
- `feature_names`: Feature column names
- `best_models`: Trained models dictionary
- `anomaly_results`: Anomaly detection results

**Returns:** None

**Report Sections:**
1. **Title and Description**
2. **Exploratory Data Analysis**: All EDA visualizations
3. **Anomaly Detection**: Results from all 5 methods
4. **Classification Model Evaluation**: Metrics and confusion matrices
5. **ROC and Precision-Recall Curves**
6. **Feature Importance**: For tree-based models
7. **Conclusions**: Summary of findings

**Output:** `technical_report.pdf` in `outputs/` directory

---

### 9. Main Execution Flow

#### `main()`
Orchestrates the complete pipeline execution.

**Execution Order:**
1. Setup directories
2. Generate simulated temporal data
3. Export data to CSV
4. Handle data types and encoding
5. Validate categorical encoding
6. Perform EDA
7. Preprocess data (scaling, balancing, splitting)
8. Train classification models
9. Evaluate models
10. Export models to disk
11. Plot classification curves
12. Plot feature importance
13. Detect anomalies
14. Generate PDF report

**Error Handling:**
- Logs errors at each step
- Continues execution even if individual steps fail
- Provides informative error messages

---

## Dependencies

### Required Libraries
- `numpy`: Numerical operations
- `pandas`: Data manipulation
- `matplotlib`: Plotting
- `seaborn`: Statistical visualizations
- `scikit-learn`: Machine learning models and preprocessing
- `imblearn`: SMOTE for class balancing
- `ydata-profiling`: EDA report generation
- `joblib`: Model serialization
- `openpyxl`: Excel file operations
- `reportlab`: PDF generation
- `packaging`: Version checking

### Version Compatibility
The script handles sklearn version differences:
- `OneHotEncoder` parameter `sparse_output` (sklearn >= 1.2) vs `sparse` (older versions)

---

## Usage

### Basic Execution
```bash
python main.py
```

### Expected Output
- Generated data in `inputs/emulated_data.csv`
- Trained models in `models/` directory
- Visualizations in `outputs/` directory
- Excel evaluation file: `outputs/model_evaluation.xlsx`
- PDF report: `outputs/technical_report.pdf`

### Logging
All operations are logged to console with INFO level. Logs include:
- Directory creation status
- Data generation progress
- Model training progress
- Evaluation metrics
- File save confirmations

---

## Key Design Decisions

### 1. Simulated Data
- Uses synthetic data generation instead of real sensor data
- Allows for controlled experimentation and testing
- Includes realistic temporal patterns and correlations

### 2. Multiple Models
- Trains 4 different classification algorithms
- Enables model comparison and selection
- Provides ensemble capabilities

### 3. Comprehensive Evaluation
- Multiple metrics (accuracy, precision, recall, F1, ROC AUC, PR AUC)
- Visual evaluation through curves and confusion matrices
- Excel export for further analysis

### 4. Anomaly Detection
- Five different algorithms for robustness
- Independent of classification models
- Provides complementary insights

### 5. Automated Reporting
- Single command generates complete analysis
- PDF report for stakeholders
- Excel file for detailed metrics

---

## Limitations and Considerations

### 1. Scaler Not Saved
The `StandardScaler` used in preprocessing is not saved. For production use:
- Save the scaler after fitting: `joblib.dump(scaler, 'models/scaler.pkl')`
- Load and use the same scaler for predictions

### 2. Synthetic Data
- Generated data may not reflect real-world patterns
- Model performance on synthetic data may not generalize
- Real sensor data should be used for production models

### 3. Fixed Random Seed
- Uses `np.random.seed(42)` for reproducibility
- Results are deterministic but may not reflect variability

### 4. Memory Considerations
- Large datasets may require significant memory
- SMOTE can increase dataset size significantly
- Consider data sampling for very large datasets

### 5. Processing Time
- GridSearchCV with multiple models can be time-consuming
- Consider reducing hyperparameter search space for faster execution
- Use parallel processing (already enabled with `n_jobs=-1`)

---

## Future Enhancements

### Recommended Improvements
1. **Save Preprocessors**: Export StandardScaler and OneHotEncoder
2. **Real Data Integration**: Support for loading real sensor data
3. **Model Versioning**: Track model versions and performance over time
4. **API Integration**: Connect with prediction API for real-time inference
5. **Automated Retraining**: Schedule periodic model retraining
6. **A/B Testing**: Compare model versions in production
7. **Feature Store**: Centralized feature engineering and storage
8. **Monitoring**: Track model drift and performance degradation

---

## Troubleshooting

### Common Issues

**Issue**: `FileNotFoundError` for models directory
- **Solution**: Ensure `setup_directories()` runs first

**Issue**: Categorical encoding errors
- **Solution**: Check sklearn version compatibility

**Issue**: Memory errors with large datasets
- **Solution**: Reduce dataset size or use data sampling

**Issue**: Model training takes too long
- **Solution**: Reduce hyperparameter search space or use fewer models

**Issue**: PDF generation fails
- **Solution**: Ensure all required images exist in `outputs/` directory

---

## Contact and Support

For issues or questions regarding this pipeline, refer to the project README or contact the development team.

---

*Last Updated: 2025*

