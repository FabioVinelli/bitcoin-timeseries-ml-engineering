"""
Halving Cycle Feature Engineering
==================================
Bitcoin halving-cycle awareness features for regime classification.

Halving Dates (block reward halvings):
    - Cycle 2: 2016-07-09 (25 → 12.5 BTC)
    - Cycle 3: 2020-05-11 (12.5 → 6.25 BTC)  
    - Cycle 4: 2024-04-20 (6.25 → 3.125 BTC)

Reference:
    Meynkhard, A. (2019). Fair Value of Bitcoin Based on Scarcity
"""

from datetime import datetime
from typing import Dict, List, Tuple, Optional
import pandas as pd
import numpy as np


# =============================================================================
# Halving Cycle Constants
# =============================================================================

HALVING_DATES = {
    1: datetime(2012, 11, 28),  # 50 → 25 BTC
    2: datetime(2016, 7, 9),     # 25 → 12.5 BTC
    3: datetime(2020, 5, 11),    # 12.5 → 6.25 BTC
    4: datetime(2024, 4, 20),    # 6.25 → 3.125 BTC
}

BLOCK_REWARDS = {
    1: 50.0,
    2: 25.0,
    3: 12.5,
    4: 6.25,
    5: 3.125,
}

# Average blocks per day (144 blocks @ 10 min target)
BLOCKS_PER_DAY = 144

# Cycle phase definitions (days from halving)
CYCLE_PHASES = {
    'post_halving': (0, 180),      # 0-6 months
    'early_bull': (180, 365),       # 6-12 months
    'late_bull': (365, 540),        # 12-18 months
    'distribution': (540, 730),     # 18-24 months
    'bear': (730, 1095),            # 24-36 months
    'accumulation': (1095, 1460),   # 36-48 months
}


def get_halving_cycle(date: datetime) -> int:
    """
    Get the halving cycle number for a given date.
    
    Parameters
    ----------
    date : datetime
        Date to check
        
    Returns
    -------
    int
        Cycle number (2, 3, 4, or 5 for current)
    """
    if date < HALVING_DATES[2]:
        return 1  # Before 2nd halving
    elif date < HALVING_DATES[3]:
        return 2  # Cycle 2: 2016-07-09 to 2020-05-10
    elif date < HALVING_DATES[4]:
        return 3  # Cycle 3: 2020-05-11 to 2024-04-19
    else:
        return 4  # Cycle 4: 2024-04-20 onwards


def get_days_since_halving(date: datetime) -> int:
    """
    Get days since most recent halving.
    
    Parameters
    ----------
    date : datetime
        Date to check
        
    Returns
    -------
    int
        Days since most recent halving
    """
    halvings_before = [h for h in HALVING_DATES.values() if h <= date]
    if not halvings_before:
        return 0
    most_recent = max(halvings_before)
    return (date - most_recent).days


def get_cycle_phase(days_since_halving: int) -> str:
    """
    Get cycle phase name based on days since halving.
    
    Parameters
    ----------
    days_since_halving : int
        Days since most recent halving
        
    Returns
    -------
    str
        Phase name (e.g., 'post_halving', 'early_bull', etc.)
    """
    for phase_name, (start, end) in CYCLE_PHASES.items():
        if start <= days_since_halving < end:
            return phase_name
    return 'accumulation'  # Default for >48 months


def get_cycle_phase_normalized(days_since_halving: int) -> float:
    """
    Get normalized cycle phase (0-1 position within 4-year cycle).
    
    Parameters
    ----------
    days_since_halving : int
        Days since most recent halving
        
    Returns
    -------
    float
        Normalized phase (0.0 = halving day, 1.0 = next halving)
    """
    cycle_length = 4 * 365  # ~4 years between halvings
    return min(days_since_halving / cycle_length, 1.0)


def get_block_reward(cycle: int) -> float:
    """Get block reward for a given cycle."""
    return BLOCK_REWARDS.get(cycle, BLOCK_REWARDS[5])


def add_halving_features(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Add halving cycle features to dataframe.
    
    Features added:
        - halving_cycle: Cycle number (2, 3, 4, ...)
        - days_since_halving: Days since most recent halving
        - cycle_phase: Normalized position (0-1)
        - cycle_phase_name: Categorical phase name
        - block_reward: Current block reward
        - daily_new_supply: Estimated daily BTC issuance
        - annual_inflation: Annualized inflation rate
        - is_post_halving_150d: Boolean for first 150 days after halving
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with DatetimeIndex
    verbose : bool
        Print summary of added features
        
    Returns
    -------
    pd.DataFrame
        DataFrame with halving features added
    """
    df = df.copy()
    
    # Ensure datetime index
    if not isinstance(df.index, pd.DatetimeIndex):
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
            df.set_index('Date', inplace=True)
        elif 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
    
    # Vectorized halving cycle computation
    dates = df.index.to_pydatetime()
    
    df['halving_cycle'] = [get_halving_cycle(d) for d in dates]
    df['days_since_halving'] = [get_days_since_halving(d) for d in dates]
    df['cycle_phase'] = df['days_since_halving'].apply(get_cycle_phase_normalized)
    df['cycle_phase_name'] = df['days_since_halving'].apply(get_cycle_phase)
    
    # Block reward and supply features
    df['block_reward'] = df['halving_cycle'].apply(get_block_reward)
    df['daily_new_supply'] = df['block_reward'] * BLOCKS_PER_DAY
    
    # Estimate circulating supply (rough approximation)
    estimated_supply = 19_500_000  # Approximate as of 2024
    df['annual_inflation'] = (df['daily_new_supply'] * 365) / estimated_supply
    
    # Post-halving window indicator (first 150 days)
    df['is_post_halving_150d'] = (df['days_since_halving'] <= 150).astype(int)
    
    if verbose:
        cycles = df['halving_cycle'].value_counts().sort_index()
        print(f"📅 Halving Features Added:")
        for cycle, count in cycles.items():
            print(f"   Cycle {cycle}: {count} days")
        print(f"   Post-halving (150d) windows: {df['is_post_halving_150d'].sum()} days")
    
    return df


def get_cycle_split(
    df: pd.DataFrame,
    train_cycles: List[int] = [2],
    val_phase_max: float = 0.3,
    test_min_samples: int = 60
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split data by halving cycles for temporal validation.
    
    Strategy:
        - If Cycle 4 data available: Train=C2, Val=C3 early, Test=C4
        - If no Cycle 4: Train=C2, Val=C3 early, Test=C3 late
        - Fallback: 70/15/15 temporal split if cycles don't work
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with halving features
    train_cycles : List[int]
        Cycle numbers for training (default: Cycle 2)
    val_phase_max : float
        Maximum phase for validation (default: 0.3)
    test_min_samples : int
        Minimum test samples
        
    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        (train_df, val_df, test_df)
    """
    # Ensure halving features exist
    if 'halving_cycle' not in df.columns:
        df = add_halving_features(df, verbose=False)
    
    # Check available data
    available_cycles = df['halving_cycle'].unique()
    has_cycle_4 = 4 in available_cycles
    
    if has_cycle_4:
        # Original strategy: Train=C2, Val=C3 early, Test=C4
        train_df = df[df['halving_cycle'].isin(train_cycles)].copy()
        val_mask = (df['halving_cycle'] == 3) & (df['cycle_phase'] < val_phase_max)
        val_df = df[val_mask].copy()
        test_df = df[df['halving_cycle'] == 4].copy()
        
        if len(test_df) < test_min_samples:
            print(f"⚠️  Test set has {len(test_df)} samples (< {test_min_samples}), using all Cycle 4 data")
    else:
        # Fallback: Split within available data (cycles 2-3)
        print(f"ℹ️  No Cycle 4 data found. Using Cycle 2 + Cycle 3 split strategy.")
        
        # Get Cycle 2 and Cycle 3 data
        c2_df = df[df['halving_cycle'] == 2].copy()
        c3_df = df[df['halving_cycle'] == 3].copy()
        
        if len(c2_df) >= test_min_samples and len(c3_df) >= test_min_samples:
            # Train=C2, split C3 for val/test
            train_df = c2_df
            
            # Split C3 by phase: early (< 0.5) = val, late (>= 0.5) = test
            c3_early = c3_df[c3_df['cycle_phase'] < 0.5]
            c3_late = c3_df[c3_df['cycle_phase'] >= 0.5]
            
            # Further split if needed
            if len(c3_late) >= test_min_samples:
                val_df = c3_early
                test_df = c3_late
            else:
                # Split C3 50/50
                mid_idx = len(c3_df) // 2
                val_df = c3_df.iloc[:mid_idx]
                test_df = c3_df.iloc[mid_idx:]
            
            print(f"   Train (Cycle 2): {len(train_df)} rows")
            print(f"   Val (Cycle 3 early): {len(val_df)} rows")
            print(f"   Test (Cycle 3 late): {len(test_df)} rows")
        else:
            # Ultimate fallback: 70/15/15 temporal split
            print(f"⚠️  Insufficient cycle data. Using 70/15/15 temporal split.")
            df_sorted = df.sort_index()
            n = len(df_sorted)
            train_end = int(n * 0.70)
            val_end = int(n * 0.85)
            
            train_df = df_sorted.iloc[:train_end].copy()
            val_df = df_sorted.iloc[train_end:val_end].copy()
            test_df = df_sorted.iloc[val_end:].copy()
            
            print(f"   Train: {len(train_df)} rows")
            print(f"   Val: {len(val_df)} rows")
            print(f"   Test: {len(test_df)} rows")
    
    return train_df, val_df, test_df


if __name__ == "__main__":
    print("Halving Cycle Definitions:")
    for cycle, date in HALVING_DATES.items():
        reward = get_block_reward(cycle)
        print(f"  Cycle {cycle}: {date.strftime('%Y-%m-%d')} (reward: {reward} BTC)")
    
    print(f"\nCycle Phases:")
    for phase, (start, end) in CYCLE_PHASES.items():
        print(f"  {phase}: days {start}-{end}")
