"""
Time Series Models for Predictive Maintenance.

This module contains LSTM and GRU models specifically designed for
time series prediction of equipment failures.
"""
import logging
from typing import Optional, Tuple
import numpy as np

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout, BatchNormalization
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    logging.warning("TensorFlow not available. Time series models cannot be used.")

logger = logging.getLogger(__name__)


def create_lstm_model(
    input_shape: Tuple[int, int],
    lstm_units: int = 64,
    dropout: float = 0.2,
    learning_rate: float = 0.001,
    dense_units: int = 32
) -> keras.Model:
    """
    Create an LSTM model for failure prediction.
    
    Args:
        input_shape: Tuple (window_size, n_features) - shape of input sequences
        lstm_units: Number of units in LSTM layers
        dropout: Dropout rate for regularization
        learning_rate: Learning rate for optimizer
        dense_units: Number of units in dense layer
    
    Returns:
        Compiled Keras model
    """
    if not TENSORFLOW_AVAILABLE:
        raise ImportError("TensorFlow is required for LSTM models. Install with: pip install tensorflow")
    
    model = Sequential([
        # First LSTM layer with return_sequences=True to pass sequences to next layer
        LSTM(
            lstm_units,
            return_sequences=True,
            input_shape=input_shape,
            name='lstm_1'
        ),
        Dropout(dropout, name='dropout_1'),
        BatchNormalization(name='batch_norm_1'),
        
        # Second LSTM layer
        LSTM(
            lstm_units // 2,
            return_sequences=False,
            name='lstm_2'
        ),
        Dropout(dropout, name='dropout_2'),
        BatchNormalization(name='batch_norm_2'),
        
        # Dense layers
        Dense(dense_units, activation='relu', name='dense_1'),
        Dropout(dropout, name='dropout_3'),
        Dense(1, activation='sigmoid', name='output')  # Binary classification
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='binary_crossentropy',
        metrics=['accuracy', 'precision', 'recall']
    )
    
    logger.info(f"LSTM model created with input shape: {input_shape}")
    logger.info(f"Model parameters: {model.count_params():,}")
    
    return model


def create_gru_model(
    input_shape: Tuple[int, int],
    gru_units: int = 64,
    dropout: float = 0.2,
    learning_rate: float = 0.001,
    dense_units: int = 32
) -> keras.Model:
    """
    Create a GRU model for failure prediction.
    
    Args:
        input_shape: Tuple (window_size, n_features) - shape of input sequences
        gru_units: Number of units in GRU layers
        dropout: Dropout rate for regularization
        learning_rate: Learning rate for optimizer
        dense_units: Number of units in dense layer
    
    Returns:
        Compiled Keras model
    """
    if not TENSORFLOW_AVAILABLE:
        raise ImportError("TensorFlow is required for GRU models. Install with: pip install tensorflow")
    
    model = Sequential([
        # First GRU layer
        GRU(
            gru_units,
            return_sequences=True,
            input_shape=input_shape,
            name='gru_1'
        ),
        Dropout(dropout, name='dropout_1'),
        BatchNormalization(name='batch_norm_1'),
        
        # Second GRU layer
        GRU(
            gru_units // 2,
            return_sequences=False,
            name='gru_2'
        ),
        Dropout(dropout, name='dropout_2'),
        BatchNormalization(name='batch_norm_2'),
        
        # Dense layers
        Dense(dense_units, activation='relu', name='dense_1'),
        Dropout(dropout, name='dropout_3'),
        Dense(1, activation='sigmoid', name='output')  # Binary classification
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='binary_crossentropy',
        metrics=['accuracy', 'precision', 'recall']
    )
    
    logger.info(f"GRU model created with input shape: {input_shape}")
    logger.info(f"Model parameters: {model.count_params():,}")
    
    return model


def create_callbacks(
    checkpoint_path: Optional[str] = None,
    patience: int = 10,
    monitor: str = 'val_loss'
) -> list:
    """
    Create training callbacks for model training.
    
    Args:
        checkpoint_path: Path to save best model checkpoint
        patience: Patience for early stopping
        monitor: Metric to monitor
    
    Returns:
        List of callbacks
    """
    callbacks = [
        EarlyStopping(
            monitor=monitor,
            patience=patience,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor=monitor,
            factor=0.5,
            patience=patience // 2,
            min_lr=1e-7,
            verbose=1
        )
    ]
    
    if checkpoint_path:
        callbacks.append(
            ModelCheckpoint(
                checkpoint_path,
                monitor=monitor,
                save_best_only=True,
                verbose=1
            )
        )
    
    return callbacks


