# Machine Learning Pipeline with MQTT Integration

This project provides a complete machine learning workflow for classification, anomaly detection, and reporting. It also includes an MQTT server implementation to receive information remotely.

## Features

### `test.py`
The `test.py` script runs the entire machine learning pipeline. Its `main()` function executes the following steps:

1. **Setup Directories**  
   Creates the required folder structure to store outputs and results.

2. **Generate Temporal Data**  
   Simulates temporal datasets for training and evaluation.

3. **Export to CSV**  
   Saves the generated data into CSV files for further processing.

4. **Data Type Handling**  
   Ensures categorical and numerical data are properly encoded and transformed before analysis.

5. **Exploratory Data Analysis (EDA)**  
   Produces statistical summaries and visualizations to understand the dataset.

6. **Preprocessing**  
   Splits the dataset into training and testing sets, applies scaling and encoding, and extracts feature names.

7. **Model Training**  
   Trains multiple classification models, including:
   - Random Forest  
   - Gradient Boosting  
   - Support Vector Machines (SVC, OneClassSVM)  
   - Logistic Regression  

8. **Model Evaluation**  
   Evaluates trained models using metrics such as accuracy, precision, recall, F1-score, ROC AUC, and generates confusion matrices.

9. **Curve Plotting**  
   Plots ROC curves, Precision-Recall curves, and other performance visualizations.

10. **Feature Importance**  
    Generates feature importance plots for interpretability of models.

11. **Anomaly Detection**  
    Detects anomalies using Isolation Forest and One-Class SVM techniques.

12. **PDF Report Generation**  
    Creates a structured PDF report containing:  
    - Dataset summary  
    - Model performance metrics  
    - Plots and visualizations  
    - Anomaly detection results  

13. **Logging**  
    Logs all important steps and final status into the console and output directory.

All generated results, plots, and reports are stored in the `resultados/` directory.

---

### `server.py`
A separate server implementation is provided in `server.py`.  
- It uses **paho-mqtt** to receive data over MQTT.  
- This allows integration with IoT devices or remote applications that send data for analysis.  

---

## Requirements
This is ment to run in python 3.11. Install the required dependencies using within your environment:

```bash
pip install -r requirements.txt
```

or using conda

```bash
conda install --file requirements.txt
```

## Usage:
To run the ML pipeline:
```bash
python test.py
```

## To start the MQTT server:
To start the MQTT server:
```bash
python server.py
```