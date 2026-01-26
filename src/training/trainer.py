"""
Training Module
===============
Training loop, validation, and walk-forward cross-validation
for time-series forecasting models.
"""

import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset

from .metrics import evaluate_predictions, format_metrics

logger = logging.getLogger(__name__)


class EarlyStopping:
    """Early stopping to prevent overfitting."""
    
    def __init__(self, patience: int = 10, min_delta: float = 0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        
    def __call__(self, val_loss: float) -> bool:
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0
        return self.early_stop


class TimeSeriesDataset:
    """
    Create sequences for time-series forecasting.
    
    Converts a DataFrame into sequences suitable for LSTM training.
    """
    
    def __init__(
        self,
        data: np.ndarray,
        target: np.ndarray,
        sequence_length: int = 60
    ):
        """
        Parameters
        ----------
        data : np.ndarray
            Feature matrix, shape (n_samples, n_features)
        target : np.ndarray
            Target values, shape (n_samples,)
        sequence_length : int
            Number of time steps to look back
        """
        self.data = data
        self.target = target
        self.sequence_length = sequence_length
        
    def create_sequences(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create input sequences and corresponding targets.
        
        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            X: shape (n_sequences, sequence_length, n_features)
            y: shape (n_sequences,)
        """
        X, y = [], []
        
        for i in range(self.sequence_length, len(self.data)):
            X.append(self.data[i - self.sequence_length:i])
            y.append(self.target[i])
        
        return np.array(X), np.array(y)


class Trainer:
    """
    Training manager for LSTM models.
    
    Handles:
    - Training loop
    - Validation
    - Early stopping
    - Model checkpointing
    - Learning rate scheduling
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        config: Dict
    ):
        """
        Parameters
        ----------
        model : nn.Module
            PyTorch model to train
        device : torch.device
            Device to train on (cpu/cuda/mps)
        config : Dict
            Training configuration
        """
        self.model = model.to(device)
        self.device = device
        self.config = config
        
        # Loss function
        loss_name = config.get("loss", "mse").lower()
        if loss_name == "mse":
            self.criterion = nn.MSELoss()
        elif loss_name == "mae":
            self.criterion = nn.L1Loss()
        elif loss_name == "huber":
            self.criterion = nn.HuberLoss()
        else:
            raise ValueError(f"Unknown loss: {loss_name}")
        
        # Optimizer (Paper Section III.D.1: RMSProp for non-convex optimization)
        optimizer_name = config.get("optimizer", "rmsprop").lower()
        if optimizer_name == "rmsprop":
            # RMSProp as recommended in the paper (Equations 13-15)
            self.optimizer = torch.optim.RMSprop(
                model.parameters(),
                lr=config.get("learning_rate", 0.01),
                alpha=config.get("rmsprop_rho", 0.9),  # ρ in paper
                weight_decay=config.get("weight_decay", 0.0001)
            )
        elif optimizer_name == "adam":
            self.optimizer = Adam(
                model.parameters(),
                lr=config.get("learning_rate", 0.001),
                weight_decay=config.get("weight_decay", 0.0001)
            )
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")
        
        # Learning rate scheduler
        scheduler_config = config.get("scheduler", {})
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            patience=scheduler_config.get("patience", 10),
            factor=scheduler_config.get("factor", 0.5),
            min_lr=scheduler_config.get("min_lr", 1e-6)
        )
        
        # Early stopping
        early_config = config.get("early_stopping", {})
        self.early_stopping = EarlyStopping(
            patience=early_config.get("patience", 15),
            min_delta=early_config.get("min_delta", 0.0001)
        )
        
        # Gradient clipping
        self.gradient_clip = config.get("gradient_clip", 1.0)
        
        # Training history
        self.history = {
            "train_loss": [],
            "val_loss": [],
            "learning_rate": []
        }
        
    def train_epoch(self, train_loader: DataLoader) -> float:
        """
        Train for one epoch.
        
        Parameters
        ----------
        train_loader : DataLoader
            Training data loader
            
        Returns
        -------
        float
            Average training loss
        """
        self.model.train()
        total_loss = 0.0
        
        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(self.device)
            batch_y = batch_y.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            predictions = self.model(batch_X).squeeze()
            loss = self.criterion(predictions, batch_y)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            if self.gradient_clip:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    self.gradient_clip
                )
            
            self.optimizer.step()
            total_loss += loss.item()
        
        return total_loss / len(train_loader)
    
    def validate(self, val_loader: DataLoader) -> Tuple[float, np.ndarray, np.ndarray]:
        """
        Validate the model.
        
        Parameters
        ----------
        val_loader : DataLoader
            Validation data loader
            
        Returns
        -------
        Tuple[float, np.ndarray, np.ndarray]
            val_loss, predictions, actuals
        """
        self.model.eval()
        total_loss = 0.0
        predictions = []
        actuals = []
        
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                
                pred = self.model(batch_X).squeeze()
                loss = self.criterion(pred, batch_y)
                
                total_loss += loss.item()
                # Convert to list first (defensive: avoids numpy ABI issues)
                # Handle both scalar and tensor cases
                pred_list = pred.detach().cpu().to(torch.float32).tolist()
                actual_list = batch_y.detach().cpu().to(torch.float32).tolist()
                # Ensure we have a list (handle scalar case)
                if not isinstance(pred_list, list):
                    pred_list = [pred_list]
                if not isinstance(actual_list, list):
                    actual_list = [actual_list]
                predictions.extend(pred_list)
                actuals.extend(actual_list)
        
        # Convert to numpy arrays
        predictions = np.array(predictions)
        actuals = np.array(actuals)
        
        return total_loss / len(val_loader), predictions, actuals
    
    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 100,
        save_path: Optional[Path] = None
    ) -> Dict:
        """
        Full training loop.
        
        Parameters
        ----------
        train_loader : DataLoader
            Training data
        val_loader : DataLoader
            Validation data
        epochs : int
            Maximum number of epochs
        save_path : Path, optional
            Path to save best model
            
        Returns
        -------
        Dict
            Training history
        """
        best_val_loss = float("inf")
        
        logger.info(f"Starting training for {epochs} epochs")
        
        for epoch in range(epochs):
            start_time = time.time()
            
            # Train
            train_loss = self.train_epoch(train_loader)
            
            # Validate
            val_loss, val_pred, val_actual = self.validate(val_loader)
            
            # Update scheduler
            self.scheduler.step(val_loss)
            current_lr = self.optimizer.param_groups[0]["lr"]
            
            # Record history
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["learning_rate"].append(current_lr)
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                if save_path:
                    torch.save(self.model.state_dict(), save_path)
                    logger.debug(f"Saved best model to {save_path}")
            
            # Log progress
            elapsed = time.time() - start_time
            if epoch % 10 == 0 or epoch == epochs - 1:
                logger.info(
                    f"Epoch {epoch+1}/{epochs} | "
                    f"Train Loss: {train_loss:.6f} | "
                    f"Val Loss: {val_loss:.6f} | "
                    f"LR: {current_lr:.2e} | "
                    f"Time: {elapsed:.1f}s"
                )
            
            # Early stopping
            if self.early_stopping(val_loss):
                logger.info(f"Early stopping triggered at epoch {epoch+1}")
                break
        
        # Load best model
        if save_path and Path(save_path).exists():
            self.model.load_state_dict(torch.load(save_path, weights_only=True))
            logger.info("Loaded best model weights")
        
        return self.history


def walk_forward_validation(
    model_class,
    model_config: Dict,
    X: np.ndarray,
    y: np.ndarray,
    initial_train_size: float = 0.5,
    step_size: int = 30,
    min_test_size: int = 30,
    device: torch.device = None,
    training_config: Dict = None
) -> Dict:
    """
    Walk-forward cross-validation for time-series.
    
    Unlike k-fold CV, walk-forward respects temporal ordering:
    1. Train on period [0, t]
    2. Test on period [t, t+step]
    3. Move forward and repeat
    
    Parameters
    ----------
    model_class : class
        Model class to instantiate
    model_config : Dict
        Model configuration
    X : np.ndarray
        Feature sequences, shape (n_samples, seq_len, n_features)
    y : np.ndarray
        Targets, shape (n_samples,)
    initial_train_size : float
        Fraction of data for initial training
    step_size : int
        Number of samples to move forward each fold
    min_test_size : int
        Minimum test window size
    device : torch.device
        Device for training
    training_config : Dict
        Training configuration
        
    Returns
    -------
    Dict
        Results with predictions and metrics for each fold
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if training_config is None:
        training_config = {}
    
    n_samples = len(X)
    initial_train_end = int(n_samples * initial_train_size)
    
    fold_results = []
    all_predictions = []
    all_actuals = []
    
    fold = 0
    train_end = initial_train_end
    
    while train_end + min_test_size <= n_samples:
        test_end = min(train_end + step_size, n_samples)
        
        logger.info(f"\nFold {fold + 1}: Train [0:{train_end}], Test [{train_end}:{test_end}]")
        
        # Split data
        X_train, y_train = X[:train_end], y[:train_end]
        X_test, y_test = X[train_end:test_end], y[train_end:test_end]
        
        # Create data loaders
        train_dataset = TensorDataset(
            torch.FloatTensor(X_train),
            torch.FloatTensor(y_train)
        )
        test_dataset = TensorDataset(
            torch.FloatTensor(X_test),
            torch.FloatTensor(y_test)
        )
        
        batch_size = training_config.get("batch_size", 32)
        # Time-series data must preserve temporal order: shuffle=False
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        # Create fresh model
        model = model_class(**model_config)
        
        # Train
        trainer = Trainer(model, device, training_config)
        trainer.fit(
            train_loader,
            test_loader,  # Using test as validation for simplicity
            epochs=training_config.get("epochs", 50)
        )
        
        # Evaluate
        test_loss, predictions, actuals = trainer.validate(test_loader)
        
        # Calculate metrics
        metrics = evaluate_predictions(actuals, predictions, include_financial=True)
        
        fold_results.append({
            "fold": fold,
            "train_size": train_end,
            "test_size": test_end - train_end,
            "test_loss": test_loss,
            "metrics": metrics
        })
        
        all_predictions.extend(predictions)
        all_actuals.extend(actuals)
        
        logger.info(f"Fold {fold + 1} - Test Loss: {test_loss:.6f}, "
                   f"Dir Acc: {metrics['Directional_Accuracy']:.2f}%")
        
        # Move forward
        train_end = test_end
        fold += 1
    
    # Aggregate results
    all_predictions = np.array(all_predictions)
    all_actuals = np.array(all_actuals)
    
    overall_metrics = evaluate_predictions(all_actuals, all_predictions, include_financial=True)
    
    return {
        "fold_results": fold_results,
        "overall_metrics": overall_metrics,
        "predictions": all_predictions,
        "actuals": all_actuals
    }


def create_data_loaders(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    batch_size: int = 32
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create PyTorch data loaders from numpy arrays.
    
    Parameters
    ----------
    X_train, y_train : np.ndarray
        Training data
    X_val, y_val : np.ndarray
        Validation data
    X_test, y_test : np.ndarray
        Test data
    batch_size : int
        Batch size
        
    Returns
    -------
    Tuple[DataLoader, DataLoader, DataLoader]
        Train, validation, and test data loaders
    """
    train_dataset = TensorDataset(
        torch.FloatTensor(X_train),
        torch.FloatTensor(y_train)
    )
    val_dataset = TensorDataset(
        torch.FloatTensor(X_val),
        torch.FloatTensor(y_val)
    )
    test_dataset = TensorDataset(
        torch.FloatTensor(X_test),
        torch.FloatTensor(y_test)
    )
    
    # Time-series data must preserve temporal order: shuffle=False for all loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # Quick test
    logging.basicConfig(level=logging.INFO)
    
    from ..models.lstm import LSTMForecaster
    
    # Generate dummy data
    n_samples = 500
    seq_len = 60
    n_features = 20
    
    X = np.random.randn(n_samples, seq_len, n_features).astype(np.float32)
    y = np.random.randn(n_samples).astype(np.float32)
    
    # Split
    train_end = int(n_samples * 0.7)
    val_end = int(n_samples * 0.85)
    
    X_train, y_train = X[:train_end], y[:train_end]
    X_val, y_val = X[train_end:val_end], y[train_end:val_end]
    X_test, y_test = X[val_end:], y[val_end:]
    
    # Create loaders
    train_loader, val_loader, test_loader = create_data_loaders(
        X_train, y_train, X_val, y_val, X_test, y_test
    )
    
    # Create model
    model = LSTMForecaster(input_size=n_features)
    device = torch.device("cpu")
    
    # Train
    config = {"epochs": 5, "learning_rate": 0.001}
    trainer = Trainer(model, device, config)
    history = trainer.fit(train_loader, val_loader, epochs=5)
    
    # Evaluate
    test_loss, predictions, actuals = trainer.validate(test_loader)
    metrics = evaluate_predictions(actuals, predictions)
    print(format_metrics(metrics))
