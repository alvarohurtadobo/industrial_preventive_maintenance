"""
Training script for native time series models (LSTM, GRU).

This script trains time series models on temporal equipment data.
"""
import os
import sys
import logging
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

from native_temporal_series.models import create_lstm_model, create_gru_model, create_callbacks
from native_temporal_series.preprocessing import prepare_data_for_training

# Configuration
INPUTS_DIR = "inputs"
MODELS_DIR = "models"
OUTPUTS_DIR = "outputs"
DEFAULT_INPUT_FILE = "emulated_data.csv"

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_data(file_path: str) -> pd.DataFrame:
    """Load data from CSV file."""
    full_path = Path(INPUTS_DIR) / file_path
    
    if not full_path.exists():
        raise FileNotFoundError(f"Input file not found: {full_path}")
    
    logger.info(f"Loading data from {full_path}...")
    data = pd.read_csv(full_path)
    
    logger.info(f"Data loaded: {data.shape[0]} rows, {data.shape[1]} columns")
    return data


def handle_data_types(data: pd.DataFrame) -> pd.DataFrame:
    """Handle data types: encode categorical variables."""
    from sklearn.preprocessing import OneHotEncoder
    from packaging import version
    import sklearn
    
    categorical_cols = data.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # Determine scikit-learn version
    skl_version = version.parse(sklearn.__version__)
    
    # Define OneHotEncoder parameters according to version
    if skl_version >= version.parse("1.2"):
        encoder = OneHotEncoder(drop='first', sparse_output=False)
    else:
        encoder = OneHotEncoder(drop='first', sparse=False)
    
    # Encode categorical variables
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
    
    # Ensure all numeric columns are float
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    data[numeric_cols] = data[numeric_cols].astype(float)
    
    # Fill NaN values
    data[numeric_cols] = data[numeric_cols].fillna(data[numeric_cols].mean())
    
    return data


def train_model(
    model_type: str,
    data: dict,
    epochs: int = 50,
    batch_size: int = 32,
    model_dir: str = MODELS_DIR
) -> dict:
    """
    Train a time series model.
    
    Args:
        model_type: 'lstm' or 'gru'
        data: Dictionary with prepared data from prepare_data_for_training
        epochs: Number of training epochs
        batch_size: Batch size for training
        model_dir: Directory to save model
    
    Returns:
        Dictionary with training results and metrics
    """
    # Create model
    if model_type.lower() == 'lstm':
        model = create_lstm_model(
            input_shape=data['input_shape'],
            lstm_units=64,
            dropout=0.2
        )
    elif model_type.lower() == 'gru':
        model = create_gru_model(
            input_shape=data['input_shape'],
            gru_units=64,
            dropout=0.2
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}. Use 'lstm' or 'gru'")
    
    # Create callbacks
    model_dir_path = Path(model_dir)
    model_dir_path.mkdir(parents=True, exist_ok=True)
    
    checkpoint_path = model_dir_path / f"{model_type}_best_model.keras"
    callbacks = create_callbacks(
        checkpoint_path=str(checkpoint_path),
        patience=10,
        monitor='val_loss'
    )
    
    # Train model
    logger.info(f"Training {model_type.upper()} model...")
    logger.info(f"Training samples: {len(data['X_train'])}")
    logger.info(f"Validation samples: {len(data['X_val'])}")
    
    history = model.fit(
        data['X_train'],
        data['y_train'],
        validation_data=(data['X_val'], data['y_val']),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )
    
    # Load best model
    model.load_weights(str(checkpoint_path))
    
    # Evaluate on test set
    logger.info("Evaluating on test set...")
    y_pred_proba = model.predict(data['X_test'], verbose=0)
    y_pred = (y_pred_proba > 0.5).astype(int).flatten()
    
    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(data['y_test'], y_pred),
        'precision': precision_score(data['y_test'], y_pred, zero_division=0),
        'recall': recall_score(data['y_test'], y_pred, zero_division=0),
        'f1': f1_score(data['y_test'], y_pred, zero_division=0),
        'roc_auc': roc_auc_score(data['y_test'], y_pred_proba.flatten())
    }
    
    # Print results
    logger.info(f"\n{model_type.upper()} Model Results:")
    logger.info(f"  Accuracy:  {metrics['accuracy']:.4f}")
    logger.info(f"  Precision: {metrics['precision']:.4f}")
    logger.info(f"  Recall:    {metrics['recall']:.4f}")
    logger.info(f"  F1-Score:  {metrics['f1']:.4f}")
    logger.info(f"  ROC AUC:   {metrics['roc_auc']:.4f}")
    
    # Confusion matrix
    cm = confusion_matrix(data['y_test'], y_pred)
    logger.info(f"\nConfusion Matrix:")
    logger.info(f"                Predicted")
    logger.info(f"              No Failure  Failure")
    logger.info(f"  Actual No F    {cm[0][0]:6d}    {cm[0][1]:6d}")
    logger.info(f"         Failure {cm[1][0]:6d}    {cm[1][1]:6d}")
    
    # Save final model
    final_model_path = model_dir_path / f"{model_type}_model.keras"
    model.save(str(final_model_path))
    logger.info(f"\nModel saved to: {final_model_path}")
    
    # Save scaler
    import joblib
    scaler_path = model_dir_path / f"{model_type}_scaler.pkl"
    joblib.dump(data['scaler'], scaler_path)
    logger.info(f"Scaler saved to: {scaler_path}")
    
    return {
        'model': model,
        'history': history.history,
        'metrics': metrics,
        'model_path': str(final_model_path),
        'scaler_path': str(scaler_path)
    }


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train time series models for predictive maintenance')
    parser.add_argument(
        '--input',
        type=str,
        default=DEFAULT_INPUT_FILE,
        help=f'Input CSV file (default: {DEFAULT_INPUT_FILE})'
    )
    parser.add_argument(
        '--model',
        type=str,
        choices=['lstm', 'gru', 'both'],
        default='both',
        help='Model type to train (default: both)'
    )
    parser.add_argument(
        '--window-size',
        type=int,
        default=10,
        help='Window size for sequences (default: 10)'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=50,
        help='Number of training epochs (default: 50)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Batch size for training (default: 32)'
    )
    
    args = parser.parse_args()
    
    try:
        # Load and preprocess data
        logger.info("Loading and preprocessing data...")
        data = load_data(args.input)
        data = handle_data_types(data)
        
        # Prepare data for time series training
        logger.info(f"Preparing data with window_size={args.window_size}...")
        prepared_data = prepare_data_for_training(
            data,
            window_size=args.window_size,
            test_size=0.2,
            val_size=0.1
        )
        
        # Train model(s)
        models_to_train = ['lstm', 'gru'] if args.model == 'both' else [args.model]
        
        results = {}
        for model_type in models_to_train:
            logger.info(f"\n{'='*70}")
            logger.info(f"Training {model_type.upper()} model")
            logger.info(f"{'='*70}")
            
            result = train_model(
                model_type=model_type,
                data=prepared_data,
                epochs=args.epochs,
                batch_size=args.batch_size
            )
            results[model_type] = result
        
        logger.info("\n✅ Training completed successfully!")
        
    except Exception as e:
        logger.exception("Error during training:")
        sys.exit(1)


if __name__ == "__main__":
    main()


