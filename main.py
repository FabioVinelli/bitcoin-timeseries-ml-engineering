#!/usr/bin/env python3
"""
Bitcoin LSTM Forecasting - Main Entry Point
============================================

Implementation based on:
    Lou, J., Cui, L., & Li, Y. (2023). Bi-LSTM Price Prediction based on 
    Attention Mechanism. arXiv:2212.03443v2

Usage:
    python main.py --mode train
    python main.py --mode evaluate
    python main.py --mode predict --days 7
"""

import argparse
import logging
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_config(config_path: str = "config.yaml") -> dict:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def get_device(config: dict) -> torch.device:
    """Get the appropriate device for training."""
    device_name = config.get("device", "auto")
    
    if device_name == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(device_name)
    
    logger.info(f"Using device: {device}")
    return device


def prepare_data(config: dict) -> tuple:
    """
    Load and prepare data for training.
    
    Returns tuple of arrays and scalers for train/val/test splits.
    """
    from src.data.fetch_data import fetch_bitcoin_data, train_val_test_split, load_cached_data
    from src.data.features import prepare_features, get_feature_columns
    from src.training.trainer import TimeSeriesDataset
    from sklearn.preprocessing import StandardScaler
    
    data_config = config.get("data", {})
    feature_config = config.get("features", {})
    paths_config = config.get("paths", {})
    
    # Check if synthetic data is requested
    data_source = data_config.get("data_source", "real")
    
    if data_source == "synthetic":
        # Generate synthetic OHLCV data for CI/testing
        logger.info("Generating synthetic data for CI/testing")
        n_samples = 500
        dates = pd.date_range(start="2020-01-01", periods=n_samples, freq="D")
        
        # Generate realistic-looking price data
        np.random.seed(42)
        price_base = 50000
        returns = np.random.randn(n_samples) * 0.02  # 2% daily volatility
        prices = price_base * np.exp(np.cumsum(returns))
        
        df = pd.DataFrame({
            "Date": dates,
            "Open": prices * (1 + np.random.randn(n_samples) * 0.001),
            "High": prices * (1 + np.abs(np.random.randn(n_samples)) * 0.01),
            "Low": prices * (1 - np.abs(np.random.randn(n_samples)) * 0.01),
            "Close": prices,
            "Volume": np.random.lognormal(15, 0.5, n_samples)
        })
        df.set_index("Date", inplace=True)
        df["Target"] = df["Close"].pct_change(1).shift(-1)
        df = df.dropna()
    else:
        # Real data path
        # Create directories
        data_dir = Path(paths_config.get("data_dir", "data"))
        data_dir.mkdir(parents=True, exist_ok=True)
        
        raw_path = data_dir / "raw" / "btc_data.csv"
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Try to load cached data
        df = load_cached_data(raw_path)
        
        # Fetch new data if needed
        if df is None:
            df = fetch_bitcoin_data(
                ticker=data_config.get("ticker", "BTC-USD"),
                start_date=data_config.get("start_date", "2018-01-01"),
                end_date=data_config.get("end_date"),
                save_path=raw_path
            )
    
    # Feature engineering
    df = prepare_features(df, feature_config)
    
    # Get feature columns
    feature_cols = get_feature_columns(df)
    logger.info(f"Using {len(feature_cols)} features: {feature_cols[:5]}...")
    
    # Split chronologically (critical for time-series!)
    train_df, val_df, test_df = train_val_test_split(
        df,
        train_ratio=data_config.get("train_ratio", 0.7),
        val_ratio=data_config.get("val_ratio", 0.15),
        test_ratio=data_config.get("test_ratio", 0.15)
    )
    
    # Normalize features (fit on train only to prevent data leakage)
    feature_scaler = StandardScaler()
    target_scaler = StandardScaler()
    
    train_features = feature_scaler.fit_transform(train_df[feature_cols])
    val_features = feature_scaler.transform(val_df[feature_cols])
    test_features = feature_scaler.transform(test_df[feature_cols])
    
    train_target = target_scaler.fit_transform(train_df[["Target"]]).flatten()
    val_target = target_scaler.transform(val_df[["Target"]]).flatten()
    test_target = target_scaler.transform(test_df[["Target"]]).flatten()
    
    # Create sequences
    seq_len = feature_config.get("sequence_length", 60)
    
    train_ds = TimeSeriesDataset(train_features, train_target, seq_len)
    val_ds = TimeSeriesDataset(val_features, val_target, seq_len)
    test_ds = TimeSeriesDataset(test_features, test_target, seq_len)
    
    X_train, y_train = train_ds.create_sequences()
    X_val, y_val = val_ds.create_sequences()
    X_test, y_test = test_ds.create_sequences()
    
    logger.info(f"Data shapes - Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    
    return (X_train, y_train, X_val, y_val, X_test, y_test, 
            feature_scaler, target_scaler, feature_cols, test_df)


def train(config: dict):
    """Main training function."""
    from src.models.lstm import create_model
    from src.training.trainer import Trainer, create_data_loaders
    from src.training.metrics import evaluate_predictions, format_metrics
    
    logger.info("=" * 60)
    logger.info("Starting Bitcoin LSTM Training Pipeline")
    logger.info("Based on Lou et al. (2023) At-BiLSTM architecture")
    logger.info("=" * 60)
    
    # Set seed
    set_seed(config.get("seed", 42))
    
    # Get device
    device = get_device(config)
    
    # Prepare data
    (X_train, y_train, X_val, y_val, X_test, y_test,
     feature_scaler, target_scaler, feature_cols, test_df) = prepare_data(config)
    
    # Create data loaders
    training_config = config.get("training", {})
    train_loader, val_loader, test_loader = create_data_loaders(
        X_train, y_train, X_val, y_val, X_test, y_test,
        batch_size=training_config.get("batch_size", 128)  # Paper: 128
    )
    
    # Create model
    model_config = config.get("model", {})
    model_config["input_size"] = X_train.shape[2]  # Number of features
    model = create_model(model_config)
    
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {total_params:,}")
    
    # Create trainer
    trainer = Trainer(model, device, training_config)
    
    # Create model directory
    models_dir = Path(config.get("paths", {}).get("models_dir", "models"))
    models_dir.mkdir(parents=True, exist_ok=True)
    model_path = models_dir / "best_model.pt"
    
    # Train
    history = trainer.fit(
        train_loader,
        val_loader,
        epochs=training_config.get("epochs", 300),  # Paper: 300
        save_path=model_path
    )
    
    # Evaluate on test set
    logger.info("\n" + "=" * 60)
    logger.info("Final Evaluation on Test Set")
    logger.info("=" * 60)
    
    test_loss, predictions, actuals = trainer.validate(test_loader)
    
    # Inverse transform to get actual returns
    predictions_orig = target_scaler.inverse_transform(predictions.reshape(-1, 1)).flatten()
    actuals_orig = target_scaler.inverse_transform(actuals.reshape(-1, 1)).flatten()
    
    # Calculate metrics
    metrics = evaluate_predictions(actuals_orig, predictions_orig, include_financial=True)
    
    print("\n" + format_metrics(metrics))
    
    # Save results
    outputs_dir = Path(config.get("paths", {}).get("outputs_dir", "outputs"))
    outputs_dir.mkdir(parents=True, exist_ok=True)
    
    # Save predictions
    results_df = pd.DataFrame({
        "actual_return": actuals_orig,
        "predicted_return": predictions_orig,
        "actual_direction": np.sign(actuals_orig),
        "predicted_direction": np.sign(predictions_orig)
    })
    results_df.to_csv(outputs_dir / "test_predictions.csv", index=False)
    
    # Save metrics
    with open(outputs_dir / "metrics.txt", "w") as f:
        f.write(format_metrics(metrics))
    
    logger.info(f"\nResults saved to {outputs_dir}")
    logger.info(f"Best model saved to {model_path}")
    
    return metrics


def evaluate(config: dict):
    """Evaluate a saved model."""
    from src.models.lstm import create_model
    from src.training.trainer import Trainer, create_data_loaders
    from src.training.metrics import evaluate_predictions, format_metrics
    
    logger.info("Loading saved model for evaluation...")
    
    device = get_device(config)
    
    # Prepare data
    (X_train, y_train, X_val, y_val, X_test, y_test,
     feature_scaler, target_scaler, feature_cols, test_df) = prepare_data(config)
    
    # Create model
    model_config = config.get("model", {})
    model_config["input_size"] = X_train.shape[2]
    model = create_model(model_config)
    
    # Load weights
    models_dir = Path(config.get("paths", {}).get("models_dir", "models"))
    model_path = models_dir / "best_model.pt"
    
    if not model_path.exists():
        logger.error(f"No saved model found at {model_path}. Run training first.")
        return
    
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.to(device)
    model.eval()
    
    # Create test loader
    _, _, test_loader = create_data_loaders(
        X_train, y_train, X_val, y_val, X_test, y_test,
        batch_size=config.get("training", {}).get("batch_size", 128)
    )
    
    # Evaluate
    trainer = Trainer(model, device, config.get("training", {}))
    test_loss, predictions, actuals = trainer.validate(test_loader)
    
    # Inverse transform
    predictions_orig = target_scaler.inverse_transform(predictions.reshape(-1, 1)).flatten()
    actuals_orig = target_scaler.inverse_transform(actuals.reshape(-1, 1)).flatten()
    
    # Calculate metrics
    metrics = evaluate_predictions(actuals_orig, predictions_orig, include_financial=True)
    
    print("\n" + format_metrics(metrics))
    
    return metrics


def main():
    parser = argparse.ArgumentParser(
        description="Bitcoin LSTM Price Forecasting",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python main.py --mode train
    python main.py --mode evaluate
    python main.py --config custom_config.yaml --mode train
        """
    )
    parser.add_argument(
        "--mode",
        choices=["train", "evaluate", "predict"],
        default="train",
        help="Operation mode"
    )
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--days",
        type=int,
        default=7,
        help="Number of days to predict (for predict mode)"
    )
    parser.add_argument(
        "--max_epochs",
        type=int,
        default=None,
        help="Override max epochs from config (optional)"
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Override epochs if --max_epochs is provided
    if args.max_epochs is not None:
        if "training" not in config:
            config["training"] = {}
        config["training"]["epochs"] = args.max_epochs
        logger.info(f"Overriding epochs to {args.max_epochs} from CLI")
    
    if args.mode == "train":
        train(config)
    elif args.mode == "evaluate":
        evaluate(config)
    elif args.mode == "predict":
        logger.info(f"Prediction mode for {args.days} days - Coming soon!")


if __name__ == "__main__":
    main()
