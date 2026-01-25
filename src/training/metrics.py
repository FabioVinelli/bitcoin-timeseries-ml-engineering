"""
Evaluation Metrics Module
=========================
Standard ML metrics and financial-specific metrics for model evaluation.
"""

import logging
from typing import Dict, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ========== Standard ML Metrics ==========

def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Squared Error."""
    return np.mean((y_true - y_pred) ** 2)


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error."""
    return np.sqrt(mse(y_true, y_pred))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error."""
    return np.mean(np.abs(y_true - y_pred))


def mape(y_true: np.ndarray, y_pred: np.ndarray, epsilon: float = 1e-8) -> float:
    """Mean Absolute Percentage Error."""
    return np.mean(np.abs((y_true - y_pred) / (y_true + epsilon))) * 100


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """R-squared (coefficient of determination)."""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / (ss_tot + 1e-8))


# ========== Financial Metrics ==========

def directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Directional Accuracy: percentage of correct up/down predictions.
    
    Critical for trading strategies - even if magnitude is wrong,
    getting direction right can be profitable.
    
    Parameters
    ----------
    y_true : np.ndarray
        Actual returns or price changes
    y_pred : np.ndarray
        Predicted returns or price changes
        
    Returns
    -------
    float
        Percentage of correct direction predictions (0-100)
    """
    true_direction = np.sign(y_true)
    pred_direction = np.sign(y_pred)
    return np.mean(true_direction == pred_direction) * 100


def hit_rate(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Hit Rate: percentage of profitable trades based on predictions.
    
    A trade is "profitable" if:
    - We predict positive and actual is positive
    - We predict negative and actual is negative
    
    Parameters
    ----------
    y_true : np.ndarray
        Actual returns
    y_pred : np.ndarray
        Predicted returns
        
    Returns
    -------
    float
        Hit rate percentage (0-100)
    """
    # Only count predictions with clear direction
    mask = np.abs(y_pred) > 1e-8
    if np.sum(mask) == 0:
        return 50.0  # No clear predictions
    
    true_dir = np.sign(y_true[mask])
    pred_dir = np.sign(y_pred[mask])
    
    return np.mean(true_dir == pred_dir) * 100


def sharpe_ratio(
    returns: np.ndarray,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252
) -> float:
    """
    Sharpe Ratio: risk-adjusted return metric.
    
    Parameters
    ----------
    returns : np.ndarray
        Array of returns (daily, weekly, etc.)
    risk_free_rate : float
        Annual risk-free rate
    periods_per_year : int
        Number of periods per year (252 for daily)
        
    Returns
    -------
    float
        Annualized Sharpe ratio
    """
    if len(returns) == 0 or np.std(returns) == 0:
        return 0.0
    
    excess_returns = returns - risk_free_rate / periods_per_year
    return np.sqrt(periods_per_year) * np.mean(excess_returns) / np.std(excess_returns)


def strategy_sharpe_ratio(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    risk_free_rate: float = 0.0,
    cost_bps: float = 0.0
) -> float:
    """
    Calculate Sharpe ratio of a simple strategy based on predictions.
    
    Strategy: Go long when prediction > 0, go short when prediction < 0.
    
    Parameters
    ----------
    y_true : np.ndarray
        Actual returns
    y_pred : np.ndarray
        Predicted returns
    risk_free_rate : float
        Annual risk-free rate
    cost_bps : float
        Transaction cost in basis points (10 bps = 0.1%)
        Typical crypto exchange fees: 10-30 bps per trade
        
    Returns
    -------
    float
        Sharpe ratio of the strategy
    """
    # Strategy returns: sign(prediction) * actual_return
    strategy_returns = np.sign(y_pred) * y_true
    
    # Deduct transaction costs when position changes
    if cost_bps > 0:
        cost_pct = cost_bps / 10000.0
        # Detect position changes (trade occurs)
        position_changes = np.abs(np.diff(np.sign(y_pred), prepend=0))
        # Cost is incurred on position change (0->1, 1->-1, etc.)
        trade_costs = (position_changes > 0).astype(float) * cost_pct
        strategy_returns = strategy_returns - trade_costs
    
    return sharpe_ratio(strategy_returns, risk_free_rate)


def strategy_sharpe_ratio_net(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    cost_bps: float = 10.0
) -> float:
    """
    Net Sharpe ratio after transaction costs.
    
    Default 10 bps (0.1%) is typical for major crypto exchanges.
    """
    return strategy_sharpe_ratio(y_true, y_pred, cost_bps=cost_bps)


def calculate_strategy_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    prices: Optional[np.ndarray] = None,
    cost_bps: float = 10.0
) -> Dict[str, float]:
    """
    Calculate comprehensive strategy metrics with transaction costs.
    
    Parameters
    ----------
    y_true : np.ndarray
        Actual returns
    y_pred : np.ndarray
        Predicted returns
    prices : Optional[np.ndarray]
        Price series (for PnL calculations)
    cost_bps : float
        Transaction cost in basis points (default: 10 bps = 0.1%)
        
    Returns
    -------
    Dict[str, float]
        Dictionary of strategy metrics
    """
    # Basic strategy: long if pred > 0, short if pred < 0
    signals = np.sign(y_pred)
    
    # Gross strategy returns (no costs)
    gross_returns = signals * y_true
    
    # Net strategy returns (with costs)
    cost_pct = cost_bps / 10000.0
    position_changes = np.abs(np.diff(signals, prepend=0))
    trade_costs = (position_changes > 0).astype(float) * cost_pct
    net_returns = gross_returns - trade_costs
    
    # Count trades
    n_trades = np.sum(position_changes > 0)
    
    # Calculate metrics
    metrics = {
        'sharpe_gross': sharpe_ratio(gross_returns, periods_per_year=365),
        'sharpe_net': sharpe_ratio(net_returns, periods_per_year=365),
        'total_return_gross': np.sum(gross_returns) * 100,  # percentage
        'total_return_net': np.sum(net_returns) * 100,
        'n_trades': n_trades,
        'avg_trade_return': np.mean(gross_returns[position_changes > 0]) * 100 if n_trades > 0 else 0,
        'win_rate': np.mean(gross_returns > 0) * 100,
        'cost_drag': (np.sum(gross_returns) - np.sum(net_returns)) * 100  # Total cost impact
    }
    
    # Add max drawdown if we can compute equity curve
    equity_gross = np.cumsum(gross_returns) + 1
    equity_net = np.cumsum(net_returns) + 1
    metrics['max_drawdown_gross'] = max_drawdown(equity_gross)
    metrics['max_drawdown_net'] = max_drawdown(equity_net)
    
    return metrics


def max_drawdown(equity_curve: np.ndarray) -> float:
    """
    Maximum Drawdown: worst peak-to-trough decline.
    
    Parameters
    ----------
    equity_curve : np.ndarray
        Cumulative returns or portfolio values
        
    Returns
    -------
    float
        Maximum drawdown as a percentage (0-100)
    """
    # Calculate running maximum
    running_max = np.maximum.accumulate(equity_curve)
    
    # Calculate drawdown at each point
    drawdowns = (running_max - equity_curve) / running_max
    
    return np.max(drawdowns) * 100


def calculate_equity_curve(returns: np.ndarray, initial_value: float = 1.0) -> np.ndarray:
    """
    Calculate cumulative equity curve from returns.
    
    Parameters
    ----------
    returns : np.ndarray
        Period returns
    initial_value : float
        Starting portfolio value
        
    Returns
    -------
    np.ndarray
        Cumulative portfolio value
    """
    return initial_value * np.cumprod(1 + returns)


def profit_factor(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Profit Factor: ratio of gross profits to gross losses.
    
    Parameters
    ----------
    y_true : np.ndarray
        Actual returns
    y_pred : np.ndarray
        Predicted returns
        
    Returns
    -------
    float
        Profit factor (>1 means profitable)
    """
    strategy_returns = np.sign(y_pred) * y_true
    
    gross_profit = np.sum(strategy_returns[strategy_returns > 0])
    gross_loss = np.abs(np.sum(strategy_returns[strategy_returns < 0]))
    
    if gross_loss == 0:
        return np.inf if gross_profit > 0 else 0.0
    
    return gross_profit / gross_loss


def calmar_ratio(
    returns: np.ndarray,
    periods_per_year: int = 252
) -> float:
    """
    Calmar Ratio: annualized return / max drawdown.
    
    Parameters
    ----------
    returns : np.ndarray
        Period returns
    periods_per_year : int
        Number of periods per year
        
    Returns
    -------
    float
        Calmar ratio
    """
    equity = calculate_equity_curve(returns)
    mdd = max_drawdown(equity)
    
    if mdd == 0:
        return 0.0
    
    annualized_return = np.mean(returns) * periods_per_year * 100
    return annualized_return / mdd


# ========== Combined Evaluation ==========

def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    include_financial: bool = True
) -> Dict[str, float]:
    """
    Comprehensive evaluation of predictions.
    
    Parameters
    ----------
    y_true : np.ndarray
        Actual values
    y_pred : np.ndarray
        Predicted values
    include_financial : bool
        Whether to include financial metrics
        
    Returns
    -------
    Dict[str, float]
        Dictionary of metric names and values
    """
    metrics = {
        "MSE": mse(y_true, y_pred),
        "RMSE": rmse(y_true, y_pred),
        "MAE": mae(y_true, y_pred),
        "MAPE": mape(y_true, y_pred),
        "R2": r2_score(y_true, y_pred)
    }
    
    if include_financial:
        # Financial metrics (assumes y_true/y_pred are returns)
        metrics.update({
            "Directional_Accuracy": directional_accuracy(y_true, y_pred),
            "Hit_Rate": hit_rate(y_true, y_pred),
            "Strategy_Sharpe": strategy_sharpe_ratio(y_true, y_pred),
            "Profit_Factor": profit_factor(y_true, y_pred)
        })
        
        # Calculate strategy equity curve for max drawdown
        strategy_returns = np.sign(y_pred) * y_true
        equity = calculate_equity_curve(strategy_returns)
        metrics["Max_Drawdown"] = max_drawdown(equity)
        metrics["Calmar_Ratio"] = calmar_ratio(strategy_returns)
    
    return metrics


def format_metrics(metrics: Dict[str, float]) -> str:
    """
    Format metrics for display.
    
    Parameters
    ----------
    metrics : Dict[str, float]
        Dictionary of metrics
        
    Returns
    -------
    str
        Formatted string
    """
    lines = []
    
    # ML metrics
    lines.append("=" * 40)
    lines.append("Standard ML Metrics")
    lines.append("=" * 40)
    for name in ["MSE", "RMSE", "MAE", "MAPE", "R2"]:
        if name in metrics:
            if name == "MAPE":
                lines.append(f"{name:20s}: {metrics[name]:.2f}%")
            elif name == "R2":
                lines.append(f"{name:20s}: {metrics[name]:.4f}")
            else:
                lines.append(f"{name:20s}: {metrics[name]:.6f}")
    
    # Financial metrics
    financial = ["Directional_Accuracy", "Hit_Rate", "Strategy_Sharpe", 
                 "Profit_Factor", "Max_Drawdown", "Calmar_Ratio"]
    if any(m in metrics for m in financial):
        lines.append("")
        lines.append("=" * 40)
        lines.append("Financial Metrics")
        lines.append("=" * 40)
        for name in financial:
            if name in metrics:
                if name in ["Directional_Accuracy", "Hit_Rate", "Max_Drawdown"]:
                    lines.append(f"{name:20s}: {metrics[name]:.2f}%")
                else:
                    lines.append(f"{name:20s}: {metrics[name]:.4f}")
    
    return "\n".join(lines)


if __name__ == "__main__":
    # Quick test with random data
    np.random.seed(42)
    
    # Simulate returns
    y_true = np.random.randn(100) * 0.02  # 2% daily volatility
    y_pred = y_true + np.random.randn(100) * 0.01  # Add noise
    
    metrics = evaluate_predictions(y_true, y_pred, include_financial=True)
    print(format_metrics(metrics))
