"""
Feature Engineering Module
==========================
Creates features for Bitcoin price prediction including:
- Technical indicators (RSI, MACD, Bollinger Bands)
- Price transformations (log returns, volatility)
- Time-based features
- GARCH(1,1) derived features (based on Lou et al. 2023 paper)

Reference:
    Lou, J., Cui, L., & Li, Y. (2023). Bi-LSTM Price Prediction based on 
    Attention Mechanism. arXiv:2212.03443v2
"""

import logging
from typing import Dict, List, Optional
import warnings

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def add_garch_features(
    df: pd.DataFrame,
    p: int = 1,
    q: int = 1
) -> pd.DataFrame:
    """
    Add GARCH(1,1) derived features as described in the research paper.
    
    The paper demonstrates that GARCH features (stochastic disturbance μt
    and conditional variance σ²t) have high coincidence with asset returns
    and improve prediction accuracy.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with 'Close' price column
    p : int
        ARCH lag order (default 1)
    q : int
        GARCH lag order (default 1)
        
    Returns
    -------
    pd.DataFrame
        DataFrame with GARCH features added
    """
    df = df.copy()
    
    try:
        from arch import arch_model
        
        # Calculate returns for GARCH model
        returns = df["Close"].pct_change().dropna() * 100  # Scale for numerical stability
        
        if len(returns) < 100:
            logger.warning("Insufficient data for GARCH model, using simplified features")
            df["GARCH_mu"] = df["Close"].pct_change()
            df["GARCH_sigma2"] = df["Close"].pct_change().rolling(20).var()
            return df
        
        # Fit GARCH(1,1) model
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = arch_model(returns, vol='Garch', p=p, q=q, mean='AR', lags=1)
            result = model.fit(disp='off', show_warning=False)
        
        # Extract features
        # μt: stochastic disturbance term (residuals)
        mu = result.resid
        # σ²t: conditional variance
        sigma2 = result.conditional_volatility ** 2
        
        # Align with original dataframe
        df.loc[returns.index, "GARCH_mu"] = mu.values
        df.loc[returns.index, "GARCH_sigma2"] = sigma2.values
        
        # Normalize GARCH features
        df["GARCH_mu_norm"] = df["GARCH_mu"] / (df["GARCH_sigma2"].apply(np.sqrt) + 1e-8)
        
        logger.info("Added GARCH(1,1) features: GARCH_mu, GARCH_sigma2")
        
    except ImportError:
        logger.warning("arch package not installed, using simplified volatility features")
        # Fallback: simplified GARCH-like features
        returns = df["Close"].pct_change()
        df["GARCH_mu"] = returns
        df["GARCH_sigma2"] = returns.rolling(20).var()
        df["GARCH_mu_norm"] = returns / (returns.rolling(20).std() + 1e-8)
    
    except Exception as e:
        logger.warning(f"GARCH fitting failed: {e}, using simplified features")
        returns = df["Close"].pct_change()
        df["GARCH_mu"] = returns
        df["GARCH_sigma2"] = returns.rolling(20).var()
        df["GARCH_mu_norm"] = returns / (returns.rolling(20).std() + 1e-8)
    
    return df


def add_technical_indicators(
    df: pd.DataFrame,
    rsi_period: int = 14,
    macd_fast: int = 12,
    macd_slow: int = 26,
    macd_signal: int = 9,
    bollinger_period: int = 20,
    bollinger_std: float = 2.0,
    sma_periods: List[int] = [7, 21, 50],
    ema_periods: List[int] = [12, 26]
) -> pd.DataFrame:
    """
    Add technical analysis indicators to the DataFrame.
    
    Includes indicators from Lou et al. (2023) paper:
    - RSI (Relative Strength Index)
    - MACD (Moving Average Convergence Divergence)
    - Bollinger Bands (Boll index)
    - PSY (Psychological index) - measures investor sentiment
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with OHLCV columns
    rsi_period : int
        Period for RSI calculation
    macd_fast, macd_slow, macd_signal : int
        MACD parameters
    bollinger_period : int
        Period for Bollinger Bands
    bollinger_std : float
        Standard deviation multiplier for Bollinger Bands
    sma_periods : List[int]
        Periods for Simple Moving Averages
    ema_periods : List[int]
        Periods for Exponential Moving Averages
        
    Returns
    -------
    pd.DataFrame
        DataFrame with added technical indicators
    """
    df = df.copy()
    close = df["Close"]
    high = df["High"]
    low = df["Low"]
    volume = df["Volume"]
    
    # ========== RSI (from paper Equation 6) ==========
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=rsi_period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_period).mean()
    rs = gain / loss
    df["RSI"] = 100 - (100 / (1 + rs))
    
    # ========== MACD ==========
    ema_fast = close.ewm(span=macd_fast, adjust=False).mean()
    ema_slow = close.ewm(span=macd_slow, adjust=False).mean()
    df["MACD"] = ema_fast - ema_slow
    df["MACD_Signal"] = df["MACD"].ewm(span=macd_signal, adjust=False).mean()
    df["MACD_Hist"] = df["MACD"] - df["MACD_Signal"]
    
    # ========== Bollinger Bands (from paper Equation 4) ==========
    sma = close.rolling(window=bollinger_period).mean()
    std = close.rolling(window=bollinger_period).std()
    df["BB_Upper"] = sma + (std * bollinger_std)
    df["BB_Middle"] = sma
    df["BB_Lower"] = sma - (std * bollinger_std)
    df["BB_Width"] = (df["BB_Upper"] - df["BB_Lower"]) / df["BB_Middle"]
    df["BB_Pct"] = (close - df["BB_Lower"]) / (df["BB_Upper"] - df["BB_Lower"])
    
    # ========== PSY - Psychological Index (from paper Equation 5) ==========
    # Measures the ratio of up days in the last N days
    # PSY_t = N_up / N
    for period in [10, 20]:
        up_days = (close.diff() > 0).rolling(window=period).sum()
        df[f"PSY_{period}"] = up_days / period * 100
    
    # ========== Moving Averages ==========
    for period in sma_periods:
        df[f"SMA_{period}"] = close.rolling(window=period).mean()
        # Price relative to SMA
        df[f"Close_SMA_{period}_Ratio"] = close / df[f"SMA_{period}"]
        
    for period in ema_periods:
        df[f"EMA_{period}"] = close.ewm(span=period, adjust=False).mean()
    
    # ========== Volume Indicators ==========
    df["Volume_SMA_20"] = volume.rolling(window=20).mean()
    df["Volume_Ratio"] = volume / df["Volume_SMA_20"]
    
    # ========== Price Range Indicators ==========
    df["High_Low_Pct"] = (high - low) / close
    df["Close_Open_Pct"] = (close - df["Open"]) / df["Open"]
    
    # ========== Average True Range (ATR) ==========
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df["ATR_14"] = tr.rolling(window=14).mean()
    
    # ========== Price Variance (from paper) ==========
    df["Price_Var_10"] = close.rolling(window=10).var()
    df["Price_Var_20"] = close.rolling(window=20).var()
    
    logger.info(f"Added technical indicators: {len(df.columns) - 6} new features")
    
    return df


def add_returns_and_volatility(
    df: pd.DataFrame,
    volatility_window: int = 20
) -> pd.DataFrame:
    """
    Add return and volatility features.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with Close price
    volatility_window : int
        Window for volatility calculation
        
    Returns
    -------
    pd.DataFrame
        DataFrame with return and volatility features
    """
    df = df.copy()
    close = df["Close"]
    
    # Log returns (more stationary than price)
    df["Log_Return"] = np.log(close / close.shift(1))
    
    # Multiple return periods
    for period in [1, 5, 10, 20]:
        df[f"Return_{period}d"] = close.pct_change(period)
    
    # Realized volatility (annualized)
    df["Volatility_20d"] = df["Log_Return"].rolling(window=volatility_window).std() * np.sqrt(252)
    
    # Rolling Sharpe-like ratio (return / volatility)
    df["Return_Vol_Ratio"] = df["Return_5d"] / (df["Volatility_20d"] + 1e-8)
    
    # Momentum
    df["Momentum_10d"] = close / close.shift(10) - 1
    df["Momentum_20d"] = close / close.shift(20) - 1
    
    logger.info("Added returns and volatility features")
    
    return df


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add time-based features.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with DatetimeIndex
        
    Returns
    -------
    pd.DataFrame
        DataFrame with time features
    """
    df = df.copy()
    
    # Basic time features
    df["Day_of_Week"] = df.index.dayofweek
    df["Month"] = df.index.month
    df["Quarter"] = df.index.quarter
    df["Day_of_Month"] = df.index.day
    df["Week_of_Year"] = df.index.isocalendar().week.values
    
    # Cyclical encoding for periodicity
    df["Day_of_Week_Sin"] = np.sin(2 * np.pi * df["Day_of_Week"] / 7)
    df["Day_of_Week_Cos"] = np.cos(2 * np.pi * df["Day_of_Week"] / 7)
    df["Month_Sin"] = np.sin(2 * np.pi * df["Month"] / 12)
    df["Month_Cos"] = np.cos(2 * np.pi * df["Month"] / 12)
    
    # Is weekend (crypto trades 24/7 but behavior may differ)
    df["Is_Weekend"] = (df["Day_of_Week"] >= 5).astype(int)
    
    logger.info("Added time features")
    
    return df


def create_target(
    df: pd.DataFrame,
    forecast_horizon: int = 1,
    target_type: str = "return"
) -> pd.DataFrame:
    """
    Create target variable for prediction.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with Close price
    forecast_horizon : int
        Number of days ahead to predict
    target_type : str
        "return" for percentage return, "price" for raw price
        
    Returns
    -------
    pd.DataFrame
        DataFrame with target column
    """
    df = df.copy()
    
    if target_type == "return":
        df["Target"] = df["Close"].pct_change(forecast_horizon).shift(-forecast_horizon)
    elif target_type == "price":
        df["Target"] = df["Close"].shift(-forecast_horizon)
    elif target_type == "direction":
        # Binary: 1 if price goes up, 0 if down
        df["Target"] = (df["Close"].shift(-forecast_horizon) > df["Close"]).astype(int)
    else:
        raise ValueError(f"Unknown target_type: {target_type}")
    
    logger.info(f"Created target with {forecast_horizon}-day horizon, type={target_type}")
    
    return df


def prepare_features(
    df: pd.DataFrame,
    config: Dict
) -> pd.DataFrame:
    """
    Full feature engineering pipeline.
    
    Implements the feature engineering approach from Lou et al. (2023):
    - Economical attributes (RSI, MACD, Bollinger, PSY)
    - Mathematical attributes (GARCH μt, σ²t)
    - Time-based features
    
    Parameters
    ----------
    df : pd.DataFrame
        Raw OHLCV DataFrame
    config : Dict
        Configuration dictionary with feature parameters
        
    Returns
    -------
    pd.DataFrame
        DataFrame with all features and target
    """
    logger.info(f"Starting feature engineering on {len(df)} rows")
    
    # Add technical indicators (economical attributes from paper)
    tech_config = config.get("technical", {})
    df = add_technical_indicators(
        df,
        rsi_period=tech_config.get("rsi_period", 14),
        macd_fast=tech_config.get("macd_fast", 12),
        macd_slow=tech_config.get("macd_slow", 26),
        macd_signal=tech_config.get("macd_signal", 9),
        bollinger_period=tech_config.get("bollinger_period", 20),
        bollinger_std=tech_config.get("bollinger_std", 2),
        sma_periods=tech_config.get("sma_periods", [7, 21, 50]),
        ema_periods=tech_config.get("ema_periods", [12, 26])
    )
    
    # Add GARCH features (mathematical attributes from paper)
    if config.get("use_garch", True):
        df = add_garch_features(df, p=1, q=1)
    
    # Add returns and volatility
    df = add_returns_and_volatility(
        df,
        volatility_window=config.get("volatility_window", 20)
    )
    
    # Add time features
    df = add_time_features(df)
    
    # Create target
    df = create_target(
        df,
        forecast_horizon=config.get("forecast_horizon", 1),
        target_type=config.get("target_type", "return")
    )
    
    # Handle NaN values intelligently
    initial_len = len(df)
    initial_cols = len(df.columns)
    
    # Step 1: Drop columns that are mostly NaN (>50%)
    nan_threshold = 0.5
    nan_pct = df.isna().sum() / len(df)
    cols_to_drop = nan_pct[nan_pct > nan_threshold].index.tolist()
    if cols_to_drop:
        logger.info(f"Dropping {len(cols_to_drop)} columns with >{nan_threshold*100:.0f}% NaN: {cols_to_drop[:5]}...")
        df = df.drop(columns=cols_to_drop)
    
    # Step 2: Forward-fill remaining NaN (for on-chain metrics that may have gaps)
    df = df.ffill()
    
    # Step 3: Drop rows that still have NaN (primarily at the start due to lookback windows)
    df = df.dropna()
    
    logger.info(f"Dropped {initial_cols - len(df.columns)} columns, {initial_len - len(df)} rows with NaN, {len(df)} remaining")
    
    logger.info(f"Feature engineering complete: {len(df.columns)} total columns")
    
    return df


def get_feature_columns(df: pd.DataFrame) -> List[str]:
    """
    Get list of feature columns (excluding target and raw OHLCV).
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with features
        
    Returns
    -------
    List[str]
        List of feature column names
    """
    exclude = ["Target", "Open", "High", "Low", "Close", "Volume", "Adj Close"]
    features = [col for col in df.columns if col not in exclude]
    return features


if __name__ == "__main__":
    # Quick test
    logging.basicConfig(level=logging.INFO)
    from fetch_data import fetch_bitcoin_data
    
    df = fetch_bitcoin_data(start_date="2023-01-01")
    
    config = {
        "technical": {
            "rsi_period": 14,
            "macd_fast": 12,
            "macd_slow": 26,
            "macd_signal": 9,
        },
        "forecast_horizon": 1,
        "target_type": "return"
    }
    
    df = prepare_features(df, config)
    print(f"\nFeature columns: {get_feature_columns(df)}")
    print(f"\nSample data:\n{df.head()}")
