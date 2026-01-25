"""
Data Processor Module - Zero-Leakage Pipeline
==============================================

Critical: This module ensures NO data leakage by fitting scalers
ONLY on training data, then transforming validation/test sets.

This is the most common source of overly-optimistic results in
financial ML - fitting normalization on the entire dataset leaks
future information into training.

Usage:
    processor = DataProcessor(target_col='Target')
    data = processor.split_and_scale(df, train_ratio=0.7, val_ratio=0.15)
    # data contains: X_train, y_train, X_val, y_val, X_test, y_test
    
    # For inference, load saved scalers:
    processor.load_scalers('models/scalers.joblib')
    X_scaled = processor.transform_features(new_data)
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler, StandardScaler, MinMaxScaler

logger = logging.getLogger(__name__)


class DataProcessor:
    """
    Zero-leakage data processor for time-series financial data.
    
    Key principle: Fit scalers ONLY on training data, transform all sets
    using those same statistics. This prevents lookahead bias.
    
    Uses RobustScaler by default for cryptocurrency data due to:
    - Extreme outliers (flash crashes, pump-and-dumps)
    - Fat-tailed return distributions
    - Better handling of non-normal data
    """
    
    def __init__(
        self,
        target_col: str = 'Target',
        scaler_type: str = 'robust',
        feature_cols: Optional[List[str]] = None
    ):
        """
        Parameters
        ----------
        target_col : str
            Name of the target column
        scaler_type : str
            Type of scaler: 'robust', 'standard', or 'minmax'
        feature_cols : Optional[List[str]]
            Specific feature columns to use. If None, uses all except target.
        """
        self.target_col = target_col
        self.feature_cols = feature_cols
        self.scaler_type = scaler_type
        
        # Initialize scalers based on type
        if scaler_type == 'robust':
            # RobustScaler uses median and IQR - resistant to outliers
            self.feature_scaler = RobustScaler()
            self.target_scaler = RobustScaler()
        elif scaler_type == 'standard':
            self.feature_scaler = StandardScaler()
            self.target_scaler = StandardScaler()
        elif scaler_type == 'minmax':
            self.feature_scaler = MinMaxScaler()
            self.target_scaler = MinMaxScaler()
        else:
            raise ValueError(f"Unknown scaler_type: {scaler_type}")
        
        self.is_fitted = False
        self._feature_names = None
        
        logger.info(f"DataProcessor initialized with {scaler_type} scaler")
    
    def split_and_scale(
        self,
        df: pd.DataFrame,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15
    ) -> Dict[str, np.ndarray]:
        """
        Split data chronologically, then scale using ONLY training statistics.
        
        CRITICAL: This is the correct order for time-series:
            1. Split FIRST (chronologically)
            2. Fit scalers on TRAIN only
            3. Transform all sets using train statistics
        
        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with features and target column
        train_ratio : float
            Proportion for training set
        val_ratio : float
            Proportion for validation set
        test_ratio : float
            Proportion for test set (should equal 1 - train - val)
            
        Returns
        -------
        Dict[str, np.ndarray]
            Dictionary with keys: X_train, y_train, X_val, y_val, X_test, y_test
            Also includes: train_dates, val_dates, test_dates
        """
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 0.01, \
            "Ratios must sum to 1.0"
        
        n = len(df)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))
        
        logger.info(f"Splitting {n} rows: train={train_end}, val={val_end - train_end}, test={n - val_end}")
        
        # 1. SPLIT FIRST (chronological - no shuffling!)
        train_df = df.iloc[:train_end].copy()
        val_df = df.iloc[train_end:val_end].copy()
        test_df = df.iloc[val_end:].copy()
        
        # Store dates for later reference
        train_dates = train_df.index if isinstance(train_df.index, pd.DatetimeIndex) else None
        val_dates = val_df.index if isinstance(val_df.index, pd.DatetimeIndex) else None
        test_dates = test_df.index if isinstance(test_df.index, pd.DatetimeIndex) else None
        
        # 2. Separate features and target
        if self.feature_cols is None:
            # Use all columns except target
            self._feature_names = [c for c in df.columns if c != self.target_col]
        else:
            self._feature_names = self.feature_cols
        
        X_train = train_df[self._feature_names].values
        y_train = train_df[[self.target_col]].values
        
        X_val = val_df[self._feature_names].values
        y_val = val_df[[self.target_col]].values
        
        X_test = test_df[self._feature_names].values
        y_test = test_df[[self.target_col]].values
        
        # 3. FIT ON TRAIN ONLY
        self.feature_scaler.fit(X_train)
        self.target_scaler.fit(y_train)
        self.is_fitted = True
        
        logger.info(f"Scalers fitted on {len(X_train)} training samples")
        
        # 4. TRANSFORM ALL SETS using train statistics
        data = {
            'X_train': self.feature_scaler.transform(X_train),
            'y_train': self.target_scaler.transform(y_train).flatten(),
            'X_val': self.feature_scaler.transform(X_val),
            'y_val': self.target_scaler.transform(y_val).flatten(),
            'X_test': self.feature_scaler.transform(X_test),
            'y_test': self.target_scaler.transform(y_test).flatten(),
            'train_dates': train_dates,
            'val_dates': val_dates,
            'test_dates': test_dates,
            'feature_names': self._feature_names
        }
        
        return data
    
    def transform_features(self, X: np.ndarray) -> np.ndarray:
        """
        Transform new features using fitted scaler.
        
        For inference on new data after model is trained.
        """
        if not self.is_fitted:
            raise RuntimeError("DataProcessor not fitted. Call split_and_scale first.")
        return self.feature_scaler.transform(X)
    
    def inverse_transform_target(self, y_scaled: np.ndarray) -> np.ndarray:
        """
        Convert scaled predictions back to original scale.
        
        Essential for interpretable predictions and metrics.
        """
        if not self.is_fitted:
            raise RuntimeError("DataProcessor not fitted. Call split_and_scale first.")
        
        if y_scaled.ndim == 1:
            y_scaled = y_scaled.reshape(-1, 1)
        return self.target_scaler.inverse_transform(y_scaled).flatten()
    
    def save_scalers(self, path: str) -> None:
        """
        Save fitted scalers for later inference.
        
        Critical for production: must use SAME scalers used during training.
        """
        import joblib
        
        if not self.is_fitted:
            raise RuntimeError("DataProcessor not fitted. Nothing to save.")
        
        state = {
            'feature_scaler': self.feature_scaler,
            'target_scaler': self.target_scaler,
            'scaler_type': self.scaler_type,
            'feature_names': self._feature_names,
            'target_col': self.target_col
        }
        
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(state, path)
        logger.info(f"Scalers saved to {path}")
    
    def load_scalers(self, path: str) -> None:
        """
        Load previously fitted scalers for inference.
        """
        import joblib
        
        state = joblib.load(path)
        self.feature_scaler = state['feature_scaler']
        self.target_scaler = state['target_scaler']
        self.scaler_type = state['scaler_type']
        self._feature_names = state['feature_names']
        self.target_col = state['target_col']
        self.is_fitted = True
        
        logger.info(f"Scalers loaded from {path}")


def create_sequences(
    X: np.ndarray,
    y: np.ndarray,
    sequence_length: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create sequences for LSTM input.
    
    Parameters
    ----------
    X : np.ndarray
        Feature array, shape (n_samples, n_features)
    y : np.ndarray
        Target array, shape (n_samples,)
    sequence_length : int
        Number of time steps per sequence
        
    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        X_seq: shape (n_sequences, sequence_length, n_features)
        y_seq: shape (n_sequences,)
    """
    n_samples = len(X)
    n_sequences = n_samples - sequence_length
    
    if n_sequences <= 0:
        raise ValueError(
            f"Not enough samples ({n_samples}) for sequence_length ({sequence_length})"
        )
    
    X_seq = np.zeros((n_sequences, sequence_length, X.shape[1]))
    y_seq = np.zeros(n_sequences)
    
    for i in range(n_sequences):
        X_seq[i] = X[i:i + sequence_length]
        y_seq[i] = y[i + sequence_length]  # Predict next value after sequence
    
    return X_seq, y_seq


if __name__ == "__main__":
    # Test the processor
    logging.basicConfig(level=logging.INFO)
    
    # Create dummy data
    dates = pd.date_range('2020-01-01', periods=1000, freq='D')
    df = pd.DataFrame({
        'Feature1': np.random.randn(1000),
        'Feature2': np.random.randn(1000),
        'Target': np.random.randn(1000)
    }, index=dates)
    
    processor = DataProcessor(target_col='Target', scaler_type='robust')
    data = processor.split_and_scale(df, train_ratio=0.7, val_ratio=0.15)
    
    print(f"X_train shape: {data['X_train'].shape}")
    print(f"y_train shape: {data['y_train'].shape}")
    print(f"Feature names: {data['feature_names']}")
    
    # Test sequence creation
    X_seq, y_seq = create_sequences(data['X_train'], data['y_train'], sequence_length=60)
    print(f"X_seq shape: {X_seq.shape}")
    print(f"y_seq shape: {y_seq.shape}")
    
    # Test save/load
    processor.save_scalers('/tmp/test_scalers.joblib')
    processor2 = DataProcessor()
    processor2.load_scalers('/tmp/test_scalers.joblib')
    print("Scaler save/load successful")
