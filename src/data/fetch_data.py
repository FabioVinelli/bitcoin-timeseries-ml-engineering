"""
Data Fetching and Loading Module
================================
Loads Bitcoin price data and optional on-chain metrics.

Supports:
1. Local CSV files (preferred - user's historical archive)
2. yfinance API (fallback for missing data)

On-Chain Metrics Integration:
    If btc_onchain.csv is provided, merges on-chain metrics with OHLCV data.
    Supports 32+ on-chain metrics from services like Glassnode, CryptoQuant, etc.
"""

import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Tuple, List

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


def load_cached_data(filepath: str) -> Optional[pd.DataFrame]:
    """
    Load cached data from a file.
    
    Parameters
    ----------
    filepath : str
        Path to the cached data file
        
    Returns
    -------
    Optional[pd.DataFrame]
        DataFrame if file exists, None otherwise
    """
    path = Path(filepath)
    if not path.exists():
        return None
    return load_local_ohlcv(str(filepath))


def load_local_ohlcv(
    filepath: str,
    date_column: str = 'Date'
) -> Optional[pd.DataFrame]:
    """
    Load OHLCV data from local CSV file.
    
    Handles various date formats and column naming conventions.
    
    Parameters
    ----------
    filepath : str
        Path to CSV file
    date_column : str
        Name of the date column
        
    Returns
    -------
    Optional[pd.DataFrame]
        DataFrame with DatetimeIndex or None if file not found
    """
    path = Path(filepath)
    if not path.exists():
        logger.warning(f"Local file not found: {filepath}")
        return None
    
    try:
        # Try different date parsing approaches
        df = pd.read_csv(filepath)
        
        # Find date column (case-insensitive)
        date_cols = [c for c in df.columns if c.lower() in ['date', 'datetime', 'timestamp', 'time', 'unix', 'unix_timestamp']]
        if date_cols:
            date_column = date_cols[0]
        elif df.columns[0].lower() not in ['open', 'high', 'low', 'close', 'volume']:
            # Assume first column is date if not OHLCV
            date_column = df.columns[0]
        
        # Handle Unix timestamps
        if df[date_column].dtype in ['int64', 'float64'] and df[date_column].iloc[0] > 1e9:
            df[date_column] = pd.to_datetime(df[date_column], unit='s')
        else:
            df[date_column] = pd.to_datetime(df[date_column])
        
        df = df.set_index(date_column)
        df = df.sort_index()
        
        # Standardize column names
        column_mapping = {}
        for col in df.columns:
            col_lower = col.lower()
            if 'open' in col_lower:
                column_mapping[col] = 'Open'
            elif 'high' in col_lower:
                column_mapping[col] = 'High'
            elif 'low' in col_lower:
                column_mapping[col] = 'Low'
            elif 'close' in col_lower and 'adj' not in col_lower:
                column_mapping[col] = 'Close'
            elif 'volume' in col_lower:
                column_mapping[col] = 'Volume'
            elif 'adj' in col_lower and 'close' in col_lower:
                column_mapping[col] = 'Adj Close'
        
        df = df.rename(columns=column_mapping)
        
        # Ensure required columns exist
        required = ['Open', 'High', 'Low', 'Close', 'Volume']
        missing = [c for c in required if c not in df.columns]
        if missing:
            logger.warning(f"Missing required columns: {missing}")
            # Try to fill missing with Close if available
            if 'Close' in df.columns:
                for col in ['Open', 'High', 'Low']:
                    if col not in df.columns:
                        df[col] = df['Close']
                if 'Volume' not in df.columns:
                    df['Volume'] = 0
        
        # Convert to numeric
        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        logger.info(f"Loaded {len(df)} rows from {filepath}")
        logger.info(f"Date range: {df.index[0]} to {df.index[-1]}")
        return df
        
    except Exception as e:
        logger.error(f"Error loading {filepath}: {e}")
        return None


def load_onchain_metrics(
    filepath: str,
    date_column: str = 'Date'
) -> Optional[pd.DataFrame]:
    """
    Load on-chain metrics from local CSV file.
    
    Supports various on-chain metrics:
    - Network metrics: active_addresses, transaction_count, hash_rate
    - Supply metrics: realized_cap, nvt_ratio, mvrv_ratio
    - Exchange metrics: exchange_balance, inflow, outflow
    - Sentiment: fear_greed_index, google_trends
    - Derivatives: futures_open_interest, funding_rate
    
    Parameters
    ----------
    filepath : str
        Path to on-chain metrics CSV
    date_column : str
        Name of date column
        
    Returns
    -------
    Optional[pd.DataFrame]
        DataFrame with on-chain metrics or None
    """
    path = Path(filepath)
    if not path.exists():
        logger.info(f"On-chain file not found (optional): {filepath}")
        return None
    
    try:
        df = pd.read_csv(filepath)
        
        # Find date column
        date_cols = [c for c in df.columns if c.lower() in ['date', 'datetime', 'timestamp', 'time', 'unix']]
        if date_cols:
            date_column = date_cols[0]
        else:
            date_column = df.columns[0]
        
        # Handle Unix timestamps
        if df[date_column].dtype in ['int64', 'float64'] and df[date_column].iloc[0] > 1e9:
            df[date_column] = pd.to_datetime(df[date_column], unit='s')
        else:
            df[date_column] = pd.to_datetime(df[date_column])
        
        df = df.set_index(date_column)
        df = df.sort_index()
        
        # Convert all columns to numeric
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Log available metrics
        logger.info(f"Loaded {len(df)} rows of on-chain data with {len(df.columns)} metrics")
        logger.info(f"On-chain metrics: {df.columns.tolist()[:10]}{'...' if len(df.columns) > 10 else ''}")
        
        return df
        
    except Exception as e:
        logger.error(f"Error loading on-chain data: {e}")
        return None


def fetch_from_yfinance(
    ticker: str = 'BTC-USD',
    start_date: str = '2018-01-01',
    end_date: Optional[str] = None
) -> Optional[pd.DataFrame]:
    """
    Fetch Bitcoin data from yfinance (fallback).
    
    Parameters
    ----------
    ticker : str
        Yahoo Finance ticker symbol
    start_date : str
        Start date for data
    end_date : Optional[str]
        End date (default: today)
        
    Returns
    -------
    Optional[pd.DataFrame]
        OHLCV DataFrame or None
    """
    try:
        import yfinance as yf
        
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
        
        logger.info(f"Fetching {ticker} from yfinance: {start_date} to {end_date}")
        
        df = yf.download(ticker, start=start_date, end=end_date, progress=False)
        
        if df.empty:
            logger.warning("yfinance returned empty DataFrame")
            return None
        
        # Handle multi-level columns from yfinance
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        
        logger.info(f"Fetched {len(df)} rows from yfinance")
        return df
        
    except ImportError:
        logger.warning("yfinance not installed, skipping API fetch")
        return None
    except Exception as e:
        logger.error(f"yfinance error: {e}")
        return None


def fetch_bitcoin_data(
    ticker: str = 'BTC-USD',
    start_date: str = '2018-01-01',
    end_date: Optional[str] = None,
    ohlcv_path: str = 'data/raw/btc_ohlcv_daily.csv',
    onchain_path: str = 'data/raw/btc_onchain.csv',
    save_path: Optional[str] = None,
    use_onchain: bool = True
) -> pd.DataFrame:
    """
    Load Bitcoin data from local files or fetch from API.
    
    Priority:
    1. Load from local OHLCV CSV (user's historical archive)
    2. Merge with on-chain metrics if available
    3. Fallback to yfinance for missing/recent data
    
    Parameters
    ----------
    ticker : str
        Ticker symbol for API fallback
    start_date : str
        Start date for data
    end_date : Optional[str]
        End date (default: today)
    ohlcv_path : str
        Path to local OHLCV CSV
    onchain_path : str
        Path to on-chain metrics CSV
    save_path : Optional[str]
        Path to save merged data
    use_onchain : bool
        Whether to include on-chain metrics
        
    Returns
    -------
    pd.DataFrame
        Complete Bitcoin dataset
    """
    # Try local OHLCV first
    df = load_local_ohlcv(ohlcv_path)
    
    # Fallback to yfinance if no local data
    if df is None:
        logger.info("No local data found, fetching from yfinance...")
        df = fetch_from_yfinance(ticker, start_date, end_date)
        
        if df is None:
            raise ValueError(
                f"Could not load data from local files or yfinance. "
                f"Please provide OHLCV data at: {ohlcv_path}"
            )
    
    # Check if we need to append recent data from yfinance
    if df is not None and end_date:
        end_dt = pd.to_datetime(end_date)
        if df.index[-1] < end_dt - timedelta(days=2):
            logger.info("Local data is stale, fetching recent data from yfinance...")
            recent = fetch_from_yfinance(
                ticker,
                start_date=(df.index[-1] + timedelta(days=1)).strftime('%Y-%m-%d'),
                end_date=end_date
            )
            if recent is not None and not recent.empty:
                df = pd.concat([df, recent])
                df = df[~df.index.duplicated(keep='last')]
                df = df.sort_index()
    
    # Merge on-chain metrics if requested and available
    if use_onchain:
        onchain = load_onchain_metrics(onchain_path)
        if onchain is not None:
            # Merge on date index
            df = df.join(onchain, how='left')
            logger.info(f"Merged on-chain data. Total columns: {len(df.columns)}")
            
            # Log on-chain coverage
            onchain_cols = [c for c in onchain.columns if c in df.columns]
            coverage = df[onchain_cols].notna().mean().mean() * 100
            logger.info(f"On-chain data coverage: {coverage:.1f}%")
    
    # Save if requested
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(save_path)
        logger.info(f"Saved data to {save_path}")
    
    logger.info(f"Final dataset: {len(df)} rows, {len(df.columns)} columns")
    logger.info(f"Date range: {df.index[0]} to {df.index[-1]}")
    
    return df


def get_onchain_feature_groups() -> dict:
    """
    Return categorized on-chain feature groups for feature engineering.
    
    These groups help with feature selection and interpretation.
    """
    return {
        'network': [
            'active_addresses', 'transaction_count', 'transfer_volume',
            'avg_transaction_value', 'median_transaction_value'
        ],
        'mining': [
            'hash_rate', 'difficulty', 'block_size', 'avg_block_size',
            'miner_revenue', 'fees_mean', 'fees_median', 'fees_total'
        ],
        'supply': [
            'realized_cap', 'market_cap', 'nvt_ratio', 'mvrv_ratio',
            'sopr', 'puell_multiple', 'reserve_risk', 'stock_to_flow'
        ],
        'exchange': [
            'exchange_balance', 'exchange_inflow', 'exchange_outflow',
            'exchange_netflow', 'exchange_whale_ratio'
        ],
        'derivatives': [
            'futures_open_interest', 'futures_volume', 'funding_rate',
            'liquidations_long', 'liquidations_short', 'options_volume'
        ],
        'sentiment': [
            'fear_greed_index', 'google_trends', 'twitter_sentiment',
            'reddit_sentiment', 'social_volume'
        ],
        'holder': [
            'whale_transactions', 'addresses_1k_btc', 'addresses_10k_btc',
            'hodl_waves', 'lth_supply', 'sth_supply'
        ],
        'stablecoin': [
            'stablecoin_supply', 'usdt_supply', 'usdc_supply',
            'stablecoin_ratio'
        ]
    }


def train_val_test_split(
    df: pd.DataFrame,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Chronological split for time series data.
    
    IMPORTANT: Never shuffle time series data!
    
    Parameters
    ----------
    df : pd.DataFrame
        Full dataset
    train_ratio, val_ratio, test_ratio : float
        Split proportions (must sum to 1.0)
        
    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        Train, validation, test DataFrames
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 0.01
    
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    
    train = df.iloc[:train_end].copy()
    val = df.iloc[train_end:val_end].copy()
    test = df.iloc[val_end:].copy()
    
    logger.info(f"Split sizes - Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")
    
    return train, val, test


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Test data loading
    df = fetch_bitcoin_data(
        ohlcv_path='data/raw/btc_ohlcv_daily.csv',
        onchain_path='data/raw/btc_onchain.csv',
        use_onchain=True
    )
    
    print(f"\nLoaded {len(df)} rows")
    print(f"Columns: {df.columns.tolist()}")
    print(f"\nSample data:")
    print(df.tail())
