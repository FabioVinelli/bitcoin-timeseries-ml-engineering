"""
Feature Engineering Pipeline
=============================
Combined feature pipeline with proper leak-free processing order.

Critical Order:
    1. Sort by timestamp
    2. Apply +1 day shift to ONCHAIN_COLS (prevents look-ahead bias)
    3. Create regime labels from shifted Layer 3 columns
    4. Create return targets with shift(-7)
    5. Drop unscorable rows (NaNs from shifting)
"""

from typing import List, Tuple, Optional
import pandas as pd
import numpy as np

from .ark_layers import (
    LAYER_1_FEATURES, 
    LAYER_2_FEATURES, 
    LAYER_3_LABELS,
    ONCHAIN_COLS,
    FORBIDDEN_COLS,
    validate_no_forbidden_features
)
from .halving_features import add_halving_features


# =============================================================================
# Constants
# =============================================================================

HORIZON_DAYS = 7  # Forecast horizon for return targets

# ARK Fixed Threshold Labels (BENCHMARK ONLY - not used for training)
MVRV_BUY_THRESHOLD = 1.0
SLRV_BUY_THRESHOLD = 0.04
MVRV_SELL_THRESHOLD = 10.0

# Percentile-Based Labels (DEFAULT for training)
PERCENTILE_WINDOW = 730  # 2-year rolling window
PERCENTILE_BUY = 0.42    # BUY if MVRV < 42nd percentile 
PERCENTILE_SELL = 0.58   # SELL if MVRV > 58th percentile


def create_regime_labels(
    df: pd.DataFrame,
    mvrv_col: str = 'mvrv_ratio',
    slrv_col: str = 'slrv_ratio', 
    rpv_col: str = 'rpv_ratio',
    rpv_threshold: Optional[float] = None,
    verbose: bool = True
) -> Tuple[pd.DataFrame, dict]:
    """
    Create BUY/SELL/HOLD regime labels using Layer 3 ratios.
    
    Regime Rules:
        - BUY:  (MVRV < 1) AND (SLRV < 0.04)
        - SELL: (MVRV > 10) AND (RPV > threshold)
        - HOLD: else
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with Layer 3 columns
    mvrv_col, slrv_col, rpv_col : str
        Column names for Layer 3 ratios
    rpv_threshold : float, optional
        RPV threshold for SELL signal (auto-determined if None)
    verbose : bool
        Print regime distribution
        
    Returns
    -------
    Tuple[pd.DataFrame, dict]
        (DataFrame with 'regime' column, config dict with thresholds used)
    """
    df = df.copy()
    
    # Initialize regime as HOLD (1)
    df['regime'] = 1  # 0=BUY, 1=HOLD, 2=SELL
    
    # Check which columns are available
    has_mvrv = mvrv_col in df.columns
    has_slrv = slrv_col in df.columns
    has_rpv = rpv_col in df.columns
    
    config = {
        'mvrv_col': mvrv_col if has_mvrv else None,
        'slrv_col': slrv_col if has_slrv else None,
        'rpv_col': rpv_col if has_rpv else None,
        'mvrv_buy_threshold': MVRV_BUY_THRESHOLD,
        'slrv_buy_threshold': SLRV_BUY_THRESHOLD,
        'mvrv_sell_threshold': MVRV_SELL_THRESHOLD,
        'rpv_sell_threshold': None,
    }
    
    # Determine RPV threshold if needed
    if has_rpv and rpv_threshold is None:
        rpv_stats = df[rpv_col].describe()
        if verbose:
            print(f"📊 RPV Scale Check:")
            print(f"   median: {rpv_stats['50%']:.4f}")
            print(f"   mean:   {rpv_stats['mean']:.4f}")
            print(f"   max:    {rpv_stats['max']:.4f}")
        
        # If median > 1.0, data is likely in percent units
        if rpv_stats['50%'] > 1.0:
            rpv_threshold = 2.0  # Adjust for percent scale
            print(f"   → Using RPV threshold: {rpv_threshold} (percent scale)")
        else:
            rpv_threshold = 0.02  # Standard ratio scale
            print(f"   → Using RPV threshold: {rpv_threshold} (ratio scale)")
    
    config['rpv_sell_threshold'] = rpv_threshold
    
    # Create BUY signal: (MVRV < 1) AND (SLRV < 0.04)
    if has_mvrv and has_slrv:
        buy_mask = (df[mvrv_col] < MVRV_BUY_THRESHOLD) & (df[slrv_col] < SLRV_BUY_THRESHOLD)
        df.loc[buy_mask, 'regime'] = 0
    elif has_mvrv:
        # Fallback: MVRV only
        buy_mask = df[mvrv_col] < MVRV_BUY_THRESHOLD
        df.loc[buy_mask, 'regime'] = 0
    
    # Create SELL signal: (MVRV > 10) AND (RPV > threshold)
    if has_mvrv and has_rpv and rpv_threshold is not None:
        sell_mask = (df[mvrv_col] > MVRV_SELL_THRESHOLD) & (df[rpv_col] > rpv_threshold)
        df.loc[sell_mask, 'regime'] = 2
    elif has_mvrv:
        # Fallback: MVRV only
        sell_mask = df[mvrv_col] > MVRV_SELL_THRESHOLD
        df.loc[sell_mask, 'regime'] = 2
    
    # Count regime distribution
    regime_counts = df['regime'].value_counts().sort_index()
    regime_pcts = (regime_counts / len(df) * 100).round(2)
    
    config['regime_distribution'] = {
        'BUY': int(regime_counts.get(0, 0)),
        'HOLD': int(regime_counts.get(1, 0)),
        'SELL': int(regime_counts.get(2, 0)),
        'BUY_pct': float(regime_pcts.get(0, 0)),
        'HOLD_pct': float(regime_pcts.get(1, 0)),
        'SELL_pct': float(regime_pcts.get(2, 0)),
    }
    
    if verbose:
        print(f"\n🏷️  Regime Distribution:")
        print(f"   BUY:  {config['regime_distribution']['BUY']:5d} ({config['regime_distribution']['BUY_pct']:.1f}%)")
        print(f"   HOLD: {config['regime_distribution']['HOLD']:5d} ({config['regime_distribution']['HOLD_pct']:.1f}%)")
        print(f"   SELL: {config['regime_distribution']['SELL']:5d} ({config['regime_distribution']['SELL_pct']:.1f}%)")
    
    return df, config


def create_percentile_regime_labels(
    df: pd.DataFrame,
    mvrv_col: str = 'mvrv_ratio',
    window: int = PERCENTILE_WINDOW,
    buy_percentile: float = PERCENTILE_BUY,
    sell_percentile: float = PERCENTILE_SELL,
    min_periods: int = 365,
    verbose: bool = True
) -> Tuple[pd.DataFrame, dict]:
    """
    Create BUY/SELL/HOLD regime labels using rolling percentiles.
    
    This method is LEAK-SAFE: percentiles are computed using only past data
    (rolling window ending at time t, not including future).
    
    Regime Rules:
        - BUY:  MVRV < rolling 20th percentile (undervalued relative to history)
        - SELL: MVRV > rolling 80th percentile (overvalued relative to history)
        - HOLD: else
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with MVRV column (already shifted +1 day)
    mvrv_col : str
        Column name for MVRV ratio
    window : int
        Rolling window size in days (default: 730 = 2 years)
    buy_percentile : float
        Percentile threshold for BUY signal (default: 0.20)
    sell_percentile : float
        Percentile threshold for SELL signal (default: 0.80)
    min_periods : int
        Minimum observations required for percentile calc (default: 365)
    verbose : bool
        Print regime distribution
        
    Returns
    -------
    Tuple[pd.DataFrame, dict]
        (DataFrame with 'regime' column, config dict with thresholds used)
    """
    df = df.copy()
    
    config = {
        'labeling_method': 'percentile',
        'mvrv_col': mvrv_col,
        'window': window,
        'buy_percentile': buy_percentile,
        'sell_percentile': sell_percentile,
        'min_periods': min_periods,
    }
    
    if mvrv_col not in df.columns:
        if verbose:
            print(f"⚠️  MVRV column '{mvrv_col}' not found, defaulting to all HOLD")
        df['regime'] = 1
        config['regime_distribution'] = {'BUY': 0, 'HOLD': len(df), 'SELL': 0,
                                         'BUY_pct': 0.0, 'HOLD_pct': 100.0, 'SELL_pct': 0.0}
        return df, config
    
    # Compute rolling percentiles (LEAK-SAFE: uses only past data)
    mvrv = df[mvrv_col]
    rolling_p20 = mvrv.rolling(window=window, min_periods=min_periods).quantile(buy_percentile)
    rolling_p80 = mvrv.rolling(window=window, min_periods=min_periods).quantile(sell_percentile)
    
    # Store percentile columns for debugging (optional)
    df['mvrv_p20'] = rolling_p20
    df['mvrv_p80'] = rolling_p80
    
    # Initialize regime as HOLD (1)
    df['regime'] = 1  # 0=BUY, 1=HOLD, 2=SELL
    
    # Create BUY signal: MVRV < rolling 20th percentile
    buy_mask = mvrv < rolling_p20
    df.loc[buy_mask, 'regime'] = 0
    
    # Create SELL signal: MVRV > rolling 80th percentile
    sell_mask = mvrv > rolling_p80
    df.loc[sell_mask, 'regime'] = 2
    
    # Count regime distribution
    regime_counts = df['regime'].value_counts().sort_index()
    regime_pcts = (regime_counts / len(df) * 100).round(2)
    
    config['regime_distribution'] = {
        'BUY': int(regime_counts.get(0, 0)),
        'HOLD': int(regime_counts.get(1, 0)),
        'SELL': int(regime_counts.get(2, 0)),
        'BUY_pct': float(regime_pcts.get(0, 0)),
        'HOLD_pct': float(regime_pcts.get(1, 0)),
        'SELL_pct': float(regime_pcts.get(2, 0)),
    }
    
    # Record percentile stats for auditability
    config['percentile_stats'] = {
        'p20_mean': float(rolling_p20.mean()) if rolling_p20.notna().any() else None,
        'p80_mean': float(rolling_p80.mean()) if rolling_p80.notna().any() else None,
        'p20_median': float(rolling_p20.median()) if rolling_p20.notna().any() else None,
        'p80_median': float(rolling_p80.median()) if rolling_p80.notna().any() else None,
    }
    
    if verbose:
        print(f"\n🏷️  Percentile-Based Regime Distribution (window={window}d):")
        print(f"   BUY (MVRV < p{int(buy_percentile*100)}):  {config['regime_distribution']['BUY']:5d} ({config['regime_distribution']['BUY_pct']:.1f}%)")
        print(f"   HOLD:                     {config['regime_distribution']['HOLD']:5d} ({config['regime_distribution']['HOLD_pct']:.1f}%)")
        print(f"   SELL (MVRV > p{int(sell_percentile*100)}): {config['regime_distribution']['SELL']:5d} ({config['regime_distribution']['SELL_pct']:.1f}%)")
        print(f"   Percentile thresholds: p20 avg={config['percentile_stats']['p20_mean']:.2f}, p80 avg={config['percentile_stats']['p80_mean']:.2f}")
    
    return df, config


def apply_onchain_shift(
    df: pd.DataFrame,
    shift_days: int = 1,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Apply +1 day shift to on-chain columns to prevent look-ahead bias.
    
    This ensures that on the prediction day, we only use on-chain data
    that would have been available the previous day.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with on-chain metrics
    shift_days : int
        Number of days to shift (default: 1)
    verbose : bool
        Print shift summary
        
    Returns
    -------
    pd.DataFrame
        DataFrame with shifted on-chain columns
    """
    df = df.copy()
    
    # Find columns that exist in ONCHAIN_COLS
    cols_to_shift = [c for c in ONCHAIN_COLS if c in df.columns]
    
    if verbose:
        print(f"📈 Applying +{shift_days} day shift to {len(cols_to_shift)} on-chain columns")
    
    for col in cols_to_shift:
        df[col] = df[col].shift(shift_days)
    
    return df


def create_return_target(
    df: pd.DataFrame,
    horizon: int = HORIZON_DAYS,
    price_col: str = 'Close'
) -> pd.DataFrame:
    """
    Create forward-looking return target.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with price column
    horizon : int
        Forecast horizon in days
    price_col : str
        Price column name
        
    Returns
    -------
    pd.DataFrame
        DataFrame with return_7d target column
    """
    df = df.copy()
    
    if price_col in df.columns:
        df[f'return_{horizon}d'] = df[price_col].pct_change(horizon).shift(-horizon)
    
    return df


def prepare_proof_fold_features(
    df: pd.DataFrame,
    apply_shift: bool = True,
    create_labels: bool = True,
    create_targets: bool = True,
    use_percentile_labels: bool = True,
    verbose: bool = True
) -> Tuple[pd.DataFrame, dict]:
    """
    Full feature preparation pipeline with leak-free processing.
    
    Critical Order:
        1. Sort by timestamp
        2. Apply +1 day shift to ONCHAIN_COLS
        3. Add halving features
        4. Create regime labels (from shifted L3)
        5. Create return targets
    
    Parameters
    ----------
    df : pd.DataFrame
        Raw DataFrame with on-chain metrics
    apply_shift : bool
        Apply +1 day shift to on-chain columns
    create_labels : bool
        Create BUY/SELL/HOLD regime labels
    create_targets : bool
        Create return targets
    verbose : bool
        Print processing summary
        
    Returns
    -------
    Tuple[pd.DataFrame, dict]
        (Processed DataFrame, config dict with processing details)
    """
    config = {
        'apply_shift': apply_shift,
        'shift_days': 1,
        'create_labels': create_labels,
        'create_targets': create_targets,
        'horizon_days': HORIZON_DAYS,
        'use_percentile_labels': use_percentile_labels,
    }
    
    if verbose:
        print("=" * 60)
        print("🔧 Feature Engineering Pipeline")
        print("=" * 60)
    
    # Step 1: Sort by timestamp
    df = df.sort_index()
    if verbose:
        print(f"1. Sorted by timestamp: {df.index[0]} → {df.index[-1]}")
    
    # Step 2: Apply +1 day shift to on-chain columns
    if apply_shift:
        df = apply_onchain_shift(df, shift_days=1, verbose=verbose)
    
    # Step 3: Add halving features
    df = add_halving_features(df, verbose=verbose)
    
    # Step 4: Create regime labels
    if create_labels:
        if use_percentile_labels:
            # DEFAULT: Percentile-based labeling (stable across cycles)
            df, label_config = create_percentile_regime_labels(df, verbose=verbose)
        else:
            # BENCHMARK ONLY: ARK fixed thresholds
            df, label_config = create_regime_labels(df, verbose=verbose)
        config.update(label_config)
    
    # Step 5: Create return targets
    if create_targets:
        df = create_return_target(df, horizon=HORIZON_DAYS)
        if verbose:
            print(f"5. Created return_{HORIZON_DAYS}d target")
    
    # ==============================================
    # Step 6: Handle NaN values intelligently
    # ==============================================
    initial_len = len(df)
    
    # First, forward-fill feature columns (on-chain data may have gaps)
    # Exclude target columns from forward-fill
    exclude_from_ffill = ['regime', f'return_{HORIZON_DAYS}d', 'Target']
    fill_cols = [c for c in df.columns if c not in exclude_from_ffill]
    df[fill_cols] = df[fill_cols].ffill()
    
    # Then backward-fill any remaining leading NaNs (start of dataset)
    df[fill_cols] = df[fill_cols].bfill()
    
    # Identify essential columns that must not be NaN
    essential_cols = ['regime']
    if f'return_{HORIZON_DAYS}d' in df.columns:
        essential_cols.append(f'return_{HORIZON_DAYS}d')
    
    # Drop rows only where essential columns are NaN
    df = df.dropna(subset=essential_cols)
    
    # Also drop rows where ALL feature columns are NaN (completely empty rows)
    # Get feature columns for this check
    feature_candidates = [c for c in df.columns if c not in exclude_from_ffill + 
                         ['Open', 'High', 'Low', 'Close', 'Volume', 'halving_cycle', 
                          'cycle_phase', 'cycle_phase_name']]
    if feature_candidates:
        df = df.dropna(subset=feature_candidates, how='all')
    
    if verbose:
        print(f"\n📋 Final dataset: {len(df)} rows ({initial_len - len(df)} dropped due to missing essential values)")
        if len(df) > 0:
            nan_counts = df.isna().sum()
            cols_with_nan = nan_counts[nan_counts > 0]
            if len(cols_with_nan) > 0:
                print(f"   Columns with remaining NaN: {len(cols_with_nan)} (will be handled during training)")
    
    config['rows_processed'] = len(df)
    config['rows_dropped'] = initial_len - len(df)
    
    return df, config


if __name__ == "__main__":
    print("Feature Engineering Pipeline Test")
    print("This module requires actual data to run.")
