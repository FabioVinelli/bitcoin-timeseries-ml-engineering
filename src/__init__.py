# Bitcoin LSTM Forecasting Package
# =================================
# Production-grade time-series forecasting for Bitcoin
# Based on Lou et al. (2023) and Yang et al. (2023)

"""
BTC-LSTM Forecasting

Two architectures available:
1. CNN-Bi-LSTM-Attention (Yang et al., 2023) - Best for R² regression
2. At-BiLSTM (Lou et al., 2023) - Best for directional accuracy

Usage:
    from src import create_model, DataProcessor, InferenceEngine
"""

__version__ = "1.0.0"
__author__ = "IBM AI Engineering Certificate Project"

# Lazy imports to avoid circular dependencies
def __getattr__(name):
    """Lazy loading of submodules."""
    if name == 'create_model':
        from src.models.lstm import create_model
        return create_model
    elif name == 'LSTMForecaster':
        from src.models.lstm import LSTMForecaster
        return LSTMForecaster
    elif name == 'CNNBiLSTMAttention':
        from src.models.lstm import CNNBiLSTMAttention
        return CNNBiLSTMAttention
    elif name == 'DataProcessor':
        from src.data.processor import DataProcessor
        return DataProcessor
    elif name == 'InferenceEngine':
        from src.utils.inference import InferenceEngine
        return InferenceEngine
    elif name == 'Trainer':
        from src.training.trainer import Trainer
        return Trainer
    raise AttributeError(f"module 'src' has no attribute '{name}'")

__all__ = [
    'create_model',
    'LSTMForecaster',
    'CNNBiLSTMAttention',
    'DataProcessor',
    'InferenceEngine',
    'Trainer',
]
