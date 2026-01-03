"""
Prediction utilities for native time series models.

This module provides functions to load and use trained time series models
for making predictions on new data.
"""
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Tuple, List, Dict
import joblib

try:
    from tensorflow import keras
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    logging.warning("TensorFlow not available. Time series models cannot be used.")

from native_temporal_series.preprocessing import create_sequences, scale_sequences

logger = logging.getLogger(__name__)


class TimeSeriesPredictor:
    """Predictor for time series models."""
    
    def __init__(
        self,
        model_path: str,
        scaler_path: str,
        window_size: int,
        feature_names: List[str]
    ):
        """
        Initialize time series predictor.
        
        Args:
            model_path: Path to saved Keras model
            scaler_path: Path to saved scaler
            window_size: Window size used during training
            feature_names: List of feature names in order
        """
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow is required. Install with: pip install tensorflow")
        
        # Load model
        logger.info(f"Loading model from {model_path}...")
        self.model = keras.models.load_model(model_path)
        
        # Load scaler
        logger.info(f"Loading scaler from {scaler_path}...")
        self.scaler = joblib.load(scaler_path)
        
        self.window_size = window_size
        self.feature_names = feature_names
        
        logger.info(f"Predictor initialized with window_size={window_size}")
    
    def predict_single_sequence(
        self,
        sequence: np.ndarray,
        return_proba: bool = True
    ) -> Tuple[int, float]:
        """
        Predict on a single sequence.
        
        Args:
            sequence: Array of shape (window_size, n_features)
            return_proba: Whether to return probability
        
        Returns:
            Tuple of (prediction, probability)
        """
        # Ensure correct shape
        if sequence.shape != (self.window_size, len(self.feature_names)):
            raise ValueError(
                f"Sequence shape {sequence.shape} does not match expected "
                f"({self.window_size}, {len(self.feature_names)})"
            )
        
        # Scale sequence
        sequence_reshaped = sequence.reshape(1, -1)
        sequence_scaled = self.scaler.transform(sequence_reshaped)
        sequence_scaled = sequence_scaled.reshape(1, self.window_size, len(self.feature_names))
        
        # Predict
        proba = self.model.predict(sequence_scaled, verbose=0)[0][0]
        prediction = 1 if proba > 0.5 else 0
        
        if return_proba:
            return prediction, float(proba)
        return prediction
    
    def predict_batch(
        self,
        sequences: np.ndarray,
        return_proba: bool = True
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Predict on a batch of sequences.
        
        Args:
            sequences: Array of shape (n_samples, window_size, n_features)
            return_proba: Whether to return probabilities
        
        Returns:
            Tuple of (predictions, probabilities)
        """
        # Scale sequences
        n_samples, window_size, n_features = sequences.shape
        sequences_reshaped = sequences.reshape(-1, n_features)
        sequences_scaled = self.scaler.transform(sequences_reshaped)
        sequences_scaled = sequences_scaled.reshape(n_samples, window_size, n_features)
        
        # Predict
        probas = self.model.predict(sequences_scaled, verbose=0).flatten()
        predictions = (probas > 0.5).astype(int)
        
        if return_proba:
            return predictions, probas
        return predictions, None
    
    def predict_from_dataframe(
        self,
        data: pd.DataFrame,
        equipment_id: Optional[int] = None,
        group_by: str = 'equipment_id'
    ) -> Dict:
        """
        Predict from a DataFrame with temporal data.
        
        Args:
            data: DataFrame with temporal data
            equipment_id: Specific equipment ID to predict (if None, predicts all)
            group_by: Column to group by
        
        Returns:
            Dictionary with predictions
        """
        # Filter by equipment if specified
        if equipment_id is not None:
            data = data[data[group_by] == equipment_id].copy()
        
        if len(data) < self.window_size:
            raise ValueError(
                f"Insufficient data: need at least {self.window_size} samples, "
                f"got {len(data)}"
            )
        
        # Create sequences
        X, _, _ = create_sequences(
            data,
            window_size=self.window_size,
            target_col=None,  # No target needed for prediction
            group_by=group_by,
            feature_cols=self.feature_names
        )
        
        # Predict
        predictions, probas = self.predict_batch(X, return_proba=True)
        
        # Get corresponding timestamps/indices
        result_data = []
        for i, (equip_id, group) in enumerate(data.groupby(group_by)):
            group = group.sort_values('time_step' if 'time_step' in group.columns else group.index)
            
            for j in range(len(group) - self.window_size + 1):
                idx = j + self.window_size - 1
                result_data.append({
                    'equipment_id': equip_id,
                    'time_step': group.iloc[idx].get('time_step', idx),
                    'prediction': int(predictions[i * (len(group) - self.window_size + 1) + j]),
                    'probability': float(probas[i * (len(group) - self.window_size + 1) + j])
                })
        
        return {
            'predictions': predictions,
            'probabilities': probas,
            'results': result_data
        }


def load_predictor(
    model_type: str,
    model_dir: str = "models",
    window_size: int = 10,
    feature_names: Optional[List[str]] = None
) -> TimeSeriesPredictor:
    """
    Load a trained time series predictor.
    
    Args:
        model_type: 'lstm' or 'gru'
        model_dir: Directory containing models
        window_size: Window size used during training
        feature_names: Feature names (if None, will try to infer)
    
    Returns:
        TimeSeriesPredictor instance
    """
    model_dir_path = Path(model_dir)
    
    model_path = model_dir_path / f"{model_type}_model.keras"
    scaler_path = model_dir_path / f"{model_type}_scaler.pkl"
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not scaler_path.exists():
        raise FileNotFoundError(f"Scaler not found: {scaler_path}")
    
    # If feature_names not provided, try to load from metadata or use default
    if feature_names is None:
        # Default feature names based on typical data structure
        feature_names = [
            'vibration', 'temperature', 'pressure',
            'oil_quality', 'contaminant_level', 'acidity',
            'hours_operated', 'maintenance_history', 'load'
        ]
        logger.warning(
            f"Feature names not provided, using default. "
            f"Make sure these match your training data."
        )
    
    return TimeSeriesPredictor(
        model_path=str(model_path),
        scaler_path=str(scaler_path),
        window_size=window_size,
        feature_names=feature_names
    )

