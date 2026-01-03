"""
Time Series Preprocessing for Predictive Maintenance.

This module handles the creation of sequences from temporal data
and preprocessing for time series models.
"""
import logging
import numpy as np
import pandas as pd
from typing import Tuple, Optional
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from packaging import version
import sklearn

logger = logging.getLogger(__name__)


def create_sequences(
    data: pd.DataFrame,
    window_size: int,
    target_col: str = 'failure',
    group_by: str = 'equipment_id',
    feature_cols: Optional[list] = None
) -> Tuple[np.ndarray, np.ndarray, list]:
    """
    Create sequences from temporal data grouped by equipment.
    
    Args:
        data: DataFrame with temporal data
        window_size: Number of time steps in each sequence
        target_col: Name of target column
        group_by: Column to group by (typically equipment_id)
        feature_cols: List of feature columns to use. If None, uses all numeric columns
                     except group_by, target_col, and time_step
    
    Returns:
        Tuple of (X_sequences, y_sequences, feature_names)
        - X_sequences: Shape (n_sequences, window_size, n_features)
        - y_sequences: Shape (n_sequences,)
        - feature_names: List of feature names used
    """
    # Determine feature columns
    if feature_cols is None:
        exclude_cols = [group_by, target_col, 'time_step', 'anomaly']
        feature_cols = [
            col for col in data.select_dtypes(include=[np.number]).columns
            if col not in exclude_cols
        ]
    
    logger.info(f"Creating sequences with window_size={window_size}")
    logger.info(f"Using {len(feature_cols)} features: {feature_cols}")
    
    X_sequences = []
    y_sequences = []
    
    # Group by equipment
    for equipment_id, group in data.groupby(group_by):
        group = group.sort_values('time_step' if 'time_step' in group.columns else group.index)
        
        # Extract features and target
        X_group = group[feature_cols].values
        y_group = group[target_col].values if target_col in group.columns else None
        
        # Create sequences
        for i in range(len(group) - window_size + 1):
            # Extract sequence window
            X_seq = X_group[i:i + window_size]
            X_sequences.append(X_seq)
            
            # Target is the label at the end of the window
            if y_group is not None:
                y_seq = y_group[i + window_size - 1]
                y_sequences.append(y_seq)
            else:
                y_sequences.append(0)  # Default if no target
    
    X_sequences = np.array(X_sequences)
    y_sequences = np.array(y_sequences)
    
    logger.info(f"Created {len(X_sequences)} sequences")
    logger.info(f"Sequence shape: {X_sequences.shape}")
    logger.info(f"Target distribution: {np.bincount(y_sequences.astype(int))}")
    
    return X_sequences, y_sequences, feature_cols


def scale_sequences(
    X_train: np.ndarray,
    X_val: Optional[np.ndarray] = None,
    X_test: Optional[np.ndarray] = None,
    scaler_type: str = 'standard'
) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray], object]:
    """
    Scale sequences for time series models.
    
    Args:
        X_train: Training sequences (n_samples, window_size, n_features)
        X_val: Validation sequences (optional)
        X_test: Test sequences (optional)
        scaler_type: Type of scaler ('standard' or 'minmax')
    
    Returns:
        Tuple of (X_train_scaled, X_val_scaled, X_test_scaled, scaler)
    """
    # Reshape for scaling: (n_samples * window_size, n_features)
    n_samples, window_size, n_features = X_train.shape
    X_train_reshaped = X_train.reshape(-1, n_features)
    
    # Create and fit scaler
    if scaler_type == 'standard':
        scaler = StandardScaler()
    elif scaler_type == 'minmax':
        scaler = MinMaxScaler()
    else:
        raise ValueError(f"Unknown scaler_type: {scaler_type}")
    
    X_train_scaled = scaler.fit_transform(X_train_reshaped)
    X_train_scaled = X_train_scaled.reshape(n_samples, window_size, n_features)
    
    # Scale validation set if provided
    X_val_scaled = None
    if X_val is not None:
        n_val = X_val.shape[0]
        X_val_reshaped = X_val.reshape(-1, n_features)
        X_val_scaled = scaler.transform(X_val_reshaped)
        X_val_scaled = X_val_scaled.reshape(n_val, window_size, n_features)
    
    # Scale test set if provided
    X_test_scaled = None
    if X_test is not None:
        n_test = X_test.shape[0]
        X_test_reshaped = X_test.reshape(-1, n_features)
        X_test_scaled = scaler.transform(X_test_reshaped)
        X_test_scaled = X_test_scaled.reshape(n_test, window_size, n_features)
    
    logger.info(f"Sequences scaled using {scaler_type} scaler")
    
    return X_train_scaled, X_val_scaled, X_test_scaled, scaler


def prepare_data_for_training(
    data: pd.DataFrame,
    window_size: int,
    test_size: float = 0.2,
    val_size: float = 0.1,
    target_col: str = 'failure',
    group_by: str = 'equipment_id',
    scaler_type: str = 'standard',
    random_state: int = 42
) -> dict:
    """
    Complete data preparation pipeline for time series training.
    
    Args:
        data: DataFrame with temporal data
        window_size: Size of time window for sequences
        test_size: Proportion of data for testing
        val_size: Proportion of data for validation
        target_col: Name of target column
        group_by: Column to group by
        scaler_type: Type of scaler
        random_state: Random state for reproducibility
    
    Returns:
        Dictionary with prepared data:
        {
            'X_train', 'y_train',
            'X_val', 'y_val',
            'X_test', 'y_test',
            'scaler', 'feature_names',
            'input_shape'
        }
    """
    # Create sequences
    X, y, feature_names = create_sequences(
        data, window_size, target_col, group_by
    )
    
    # Split data (maintaining temporal order by equipment)
    n_samples = len(X)
    n_test = int(n_samples * test_size)
    n_val = int(n_samples * val_size)
    n_train = n_samples - n_test - n_val
    
    # Simple split (could be improved with time-based splitting)
    indices = np.arange(n_samples)
    np.random.seed(random_state)
    np.random.shuffle(indices)
    
    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]
    
    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    X_test, y_test = X[test_idx], y[test_idx]
    
    logger.info(f"Data split - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    
    # Scale sequences
    X_train_scaled, X_val_scaled, X_test_scaled, scaler = scale_sequences(
        X_train, X_val, X_test, scaler_type
    )
    
    # Determine input shape
    input_shape = (window_size, len(feature_names))
    
    return {
        'X_train': X_train_scaled,
        'y_train': y_train,
        'X_val': X_val_scaled,
        'y_val': y_val,
        'X_test': X_test_scaled,
        'y_test': y_test,
        'scaler': scaler,
        'feature_names': feature_names,
        'input_shape': input_shape
    }

