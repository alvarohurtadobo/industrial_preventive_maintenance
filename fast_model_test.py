"""
Fast Model Test - Interactive script to test ML models on input data.

This script allows users to:
1. Select an input CSV file (default: inputs/emulated_data.csv)
2. Select a trained model from models/
3. Make predictions and display results
"""
import os
import sys
import joblib
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Tuple
from packaging import version
import sklearn

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

# Configuration
INPUTS_DIR = "inputs"
MODELS_DIR = "models"
DEFAULT_INPUT_FILE = "emulated_data.csv"

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def list_csv_files(directory: str = INPUTS_DIR) -> list[str]:
    """List all CSV files in the specified directory."""
    input_path = Path(directory)
    if not input_path.exists():
        logger.warning(f"Directory '{directory}' does not exist.")
        return []
    
    csv_files = [f.name for f in input_path.glob("*.csv")]
    return sorted(csv_files)


def list_available_models(directory: str = MODELS_DIR) -> list[str]:
    """List all available model files in the models directory."""
    models_path = Path(directory)
    if not models_path.exists():
        logger.warning(f"Directory '{directory}' does not exist.")
        return []
    
    model_files = []
    for pkl_file in models_path.glob("*_model.pkl"):
        # Extract model name: "RandomForest_model.pkl" -> "RandomForest"
        model_name = pkl_file.stem.replace("_model", "")
        model_files.append(model_name)
    
    return sorted(model_files)


def select_input_file(default: str = DEFAULT_INPUT_FILE) -> str:
    """Interactive selection of input CSV file."""
    csv_files = list_csv_files()
    
    if not csv_files:
        print(f"\n❌ No CSV files found in '{INPUTS_DIR}/' directory.")
        print(f"Using default: {default}")
        return default
    
    print(f"\n📁 Available CSV files in '{INPUTS_DIR}/':")
    print("-" * 50)
    
    for idx, file in enumerate(csv_files, 1):
        marker = "✓" if file == default else " "
        print(f"  {marker} [{idx}] {file}")
    
    print(f"\n  [0] Use default: {default}")
    print("-" * 50)
    
    while True:
        try:
            choice = input(f"\nSelect file (0-{len(csv_files)}, default=0): ").strip()
            
            if not choice or choice == "0":
                selected = default
                break
            
            idx = int(choice)
            if 1 <= idx <= len(csv_files):
                selected = csv_files[idx - 1]
                break
            else:
                print(f"❌ Invalid choice. Please enter a number between 0 and {len(csv_files)}.")
        except ValueError:
            print("❌ Invalid input. Please enter a number.")
        except KeyboardInterrupt:
            print("\n\n⚠️  Cancelled by user.")
            sys.exit(0)
    
    print(f"✅ Selected: {selected}")
    return selected


def select_model() -> Optional[str]:
    """Interactive selection of ML model."""
    models = list_available_models()
    
    if not models:
        print(f"\n❌ No model files found in '{MODELS_DIR}/' directory.")
        print("Please train models first using main.py")
        return None
    
    print(f"\n🤖 Available models in '{MODELS_DIR}/':")
    print("-" * 50)
    
    for idx, model in enumerate(models, 1):
        print(f"  [{idx}] {model}")
    
    print("-" * 50)
    
    while True:
        try:
            choice = input(f"\nSelect model (1-{len(models)}): ").strip()
            
            if not choice:
                print("❌ Please enter a number.")
                continue
            
            idx = int(choice)
            if 1 <= idx <= len(models):
                selected = models[idx - 1]
                break
            else:
                print(f"❌ Invalid choice. Please enter a number between 1 and {len(models)}.")
        except ValueError:
            print("❌ Invalid input. Please enter a number.")
        except KeyboardInterrupt:
            print("\n\n⚠️  Cancelled by user.")
            sys.exit(0)
    
    print(f"✅ Selected: {selected}")
    return selected


def load_data(file_path: str) -> pd.DataFrame:
    """Load and return data from CSV file."""
    full_path = Path(INPUTS_DIR) / file_path
    
    if not full_path.exists():
        raise FileNotFoundError(f"Input file not found: {full_path}")
    
    logger.info(f"Loading data from {full_path}...")
    data = pd.read_csv(full_path)
    
    logger.info(f"Data loaded: {data.shape[0]} rows, {data.shape[1]} columns")
    return data


def handle_data_types(data: pd.DataFrame) -> pd.DataFrame:
    """
    Handle data types: encode categorical variables and ensure numeric types.
    Based on main.py handle_data_types function.
    """
    categorical_cols = data.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # Determine scikit-learn version
    skl_version = version.parse(sklearn.__version__)
    
    # Define OneHotEncoder parameters according to version
    if skl_version >= version.parse("1.2"):
        encoder = OneHotEncoder(drop='first', sparse_output=False)
    else:
        encoder = OneHotEncoder(drop='first', sparse=False)
    
    # Encode categorical variables using OneHotEncoder
    if categorical_cols:
        try:
            encoded_data = encoder.fit_transform(data[categorical_cols])
            encoded_cols = encoder.get_feature_names_out(categorical_cols)
            encoded_df = pd.DataFrame(encoded_data, columns=encoded_cols, index=data.index)
            data = pd.concat([data.drop(categorical_cols, axis=1), encoded_df], axis=1)
            logger.info("Categorical variables encoded correctly.")
        except Exception as e:
            logger.error(f"Error encoding variables: {e}")
            raise
    else:
        logger.info("No categorical variable columns found.")
    
    # Ensure all numeric columns are of float type
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    data[numeric_cols] = data[numeric_cols].astype(float)
    logger.info("Ensured numeric data type as float.")
    
    # Additional verification
    remaining_categorical = data.select_dtypes(include=['object', 'category']).columns.tolist()
    if remaining_categorical:
        raise ValueError(
            f"Following columns have not been coded yet and are still categorical: "
            f"{remaining_categorical}"
        )
    else:
        logger.info("All columns have been coded")
    
    return data


def preprocess_for_prediction(data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, StandardScaler]:
    """
    Preprocess data for prediction (similar to main.py but without SMOTE and train/test split).
    Returns features, labels, and fitted scaler.
    """
    # Separate features and target
    X = data.drop(['failure', 'equipment_id', 'time_step', 'anomaly'], axis=1, errors='ignore')
    y = data['failure'].astype(int) if 'failure' in data.columns else None
    
    # Handle NaN values
    numeric_cols = X.select_dtypes(include=[np.number]).columns
    X[numeric_cols] = X[numeric_cols].fillna(X[numeric_cols].mean())
    
    # Scaling characteristics
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    logger.info("Features scaled successfully.")
    
    return X_scaled, y, scaler


def load_model(model_name: str) -> object:
    """Load a trained model from disk."""
    model_path = Path(MODELS_DIR) / f"{model_name}_model.pkl"
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    logger.info(f"Loading model: {model_name}...")
    model = joblib.load(model_path)
    logger.info(f"Model '{model_name}' loaded successfully.")
    
    return model


def make_predictions(model: object, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Make predictions using the loaded model."""
    logger.info("Making predictions...")
    
    y_pred = model.predict(X)
    y_pred_proba = model.predict_proba(X)[:, 1] if hasattr(model, 'predict_proba') else None
    
    logger.info(f"Predictions completed: {len(y_pred)} samples")
    
    return y_pred, y_pred_proba


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray, y_pred_proba: Optional[np.ndarray] = None):
    """Evaluate and display prediction results."""
    print("\n" + "=" * 70)
    print("📊 PREDICTION RESULTS")
    print("=" * 70)
    
    # Basic metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    print(f"\n📈 Classification Metrics:")
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1-Score:  {f1:.4f}")
    
    if y_pred_proba is not None:
        roc_auc = roc_auc_score(y_true, y_pred_proba)
        print(f"  ROC AUC:   {roc_auc:.4f}")
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    print(f"\n📋 Confusion Matrix:")
    print(f"                Predicted")
    print(f"              No Failure  Failure")
    print(f"  Actual No F    {cm[0][0]:6d}    {cm[0][1]:6d}")
    print(f"         Failure {cm[1][0]:6d}    {cm[1][1]:6d}")
    
    # Classification report
    print(f"\n📄 Detailed Classification Report:")
    print(classification_report(y_true, y_pred, target_names=['No Failure', 'Failure']))
    
    # Distribution
    unique, counts = np.unique(y_pred, return_counts=True)
    print(f"\n📊 Prediction Distribution:")
    for label, count in zip(unique, counts):
        label_name = "Failure" if label == 1 else "No Failure"
        percentage = (count / len(y_pred)) * 100
        print(f"  {label_name}: {count:6d} ({percentage:5.2f}%)")
    
    print("=" * 70)


def show_sample_predictions(
    data: pd.DataFrame,
    y_pred: np.ndarray,
    y_pred_proba: Optional[np.ndarray],
    y_true: Optional[np.ndarray] = None,
    n_samples: int = 10
):
    """Display sample predictions."""
    print(f"\n🔍 Sample Predictions (first {n_samples}):")
    print("-" * 100)
    
    # Select samples (mix of failures and non-failures if possible)
    if y_true is not None:
        failure_indices = np.where(y_true == 1)[0]
        no_failure_indices = np.where(y_true == 0)[0]
        
        # Mix samples
        sample_indices = []
        if len(failure_indices) > 0:
            sample_indices.extend(failure_indices[:n_samples // 2])
        if len(no_failure_indices) > 0:
            sample_indices.extend(no_failure_indices[:n_samples // 2])
        
        if len(sample_indices) < n_samples:
            sample_indices = list(range(min(n_samples, len(data))))
    else:
        sample_indices = list(range(min(n_samples, len(data))))
    
    for idx in sample_indices[:n_samples]:
        row = data.iloc[idx]
        pred = y_pred[idx]
        proba = y_pred_proba[idx] if y_pred_proba is not None else None
        true_label = y_true[idx] if y_true is not None else None
        
        # Get equipment info
        equipment_id = row.get('equipment_id', 'N/A')
        time_step = row.get('time_step', 'N/A')
        
        pred_label = "Failure" if pred == 1 else "No Failure"
        true_label_str = f"True: {'Failure' if true_label == 1 else 'No Failure'}" if true_label is not None else "N/A"
        
        match = "✓" if (true_label is None or pred == true_label) else "✗"
        
        print(f"\n  [{idx}] Equipment {equipment_id}, Time Step {time_step} {match}")
        print(f"      Prediction: {pred_label}", end="")
        if proba is not None:
            print(f" (Probability: {proba:.4f})", end="")
        print()
        if true_label is not None:
            print(f"      {true_label_str}")


def main():
    """Main execution function."""
    print("\n" + "=" * 70)
    print("🚀 Fast Model Test - Interactive Model Testing")
    print("=" * 70)
    
    try:
        # Step 1: Select input file
        input_file = select_input_file()
        
        # Step 2: Select model
        model_name = select_model()
        if model_name is None:
            print("\n❌ No model selected. Exiting.")
            return
        
        # Step 3: Load data
        print(f"\n📂 Loading data from '{input_file}'...")
        data = load_data(input_file)
        
        # Step 4: Handle data types
        print("\n🔧 Preprocessing data...")
        data = handle_data_types(data.copy())
        
        # Step 5: Preprocess for prediction
        X, y_true, scaler = preprocess_for_prediction(data)
        
        # Step 6: Load model
        print(f"\n🤖 Loading model '{model_name}'...")
        model = load_model(model_name)
        
        # Step 7: Make predictions
        print("\n🔮 Making predictions...")
        y_pred, y_pred_proba = make_predictions(model, X)
        
        # Step 8: Evaluate results
        if y_true is not None:
            evaluate_predictions(y_true, y_pred, y_pred_proba)
        
        # Step 9: Show sample predictions
        show_sample_predictions(data, y_pred, y_pred_proba, y_true, n_samples=10)
        
        print("\n✅ Testing completed successfully!")
        print("=" * 70)
        
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)
    except Exception as e:
        logger.exception("Unexpected error occurred:")
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user. Exiting.")
        sys.exit(0)


if __name__ == "__main__":
    main()

