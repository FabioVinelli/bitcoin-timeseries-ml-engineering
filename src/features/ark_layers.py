"""
ARK Invest 3-Layer On-Chain Framework
======================================
Feature extraction based on ARK Invest's on-chain analysis framework.

Layer 1: Network Health
    - Infrastructure metrics indicating Bitcoin network strength
    
Layer 2: Buy/Sell Behavior  
    - On-chain activity patterns indicating market participant behavior
    
Layer 3: Valuation Signals
    - Ratios used for regime LABELING only (not features to avoid tautology)
    
Reference:
    ARK Invest "Big Ideas 2024" Bitcoin On-Chain Analysis Framework
"""

from typing import List, Optional
import pandas as pd
import numpy as np


# =============================================================================
# ARK Layer Definitions
# =============================================================================

# Layer 1: Network Health (features for classification)
LAYER_1_FEATURES = [
    'hash_rate',
    'active_addresses', 
    'transaction_count',
    'adjusted_transaction_volume',
    'circulating_supply',
    'issuance_rate',
    'miner_revenue'
]

# Layer 2: Buy/Sell Behavior (features for classification)
LAYER_2_FEATURES = [
    'coin_days_destroyed',
    'realized_price_lth',
    'realized_price_sth',
    'supply_in_profit',
    'supply_in_loss',
    'realized_profit_loss',
    'total_supply_1y_plus_hodl'
]

# Layer 3: Valuation Signals (for regime LABELS only, NOT features)
# Using these as features would create tautological predictions
LAYER_3_LABELS = [
    'mvrv_ratio',
    'slrv_ratio', 
    'rpv_ratio'
]

# Combined feature columns for regime classification (L1 + L2 only)
FEATURE_COLS = LAYER_1_FEATURES + LAYER_2_FEATURES

# Columns that require +1 day shift to prevent look-ahead bias
ONCHAIN_COLS = LAYER_1_FEATURES + LAYER_2_FEATURES + LAYER_3_LABELS

# Forbidden columns (never use as features)
FORBIDDEN_COLS = [
    'Close', 'close', 'price',
    'return_7d', 'return_1d', 'return_5d', 'return_10d', 'return_20d',
    'regime', 'target', 'Target'
]


def get_available_layer1_features(df: pd.DataFrame) -> List[str]:
    """Get Layer 1 features available in the dataframe."""
    available = []
    for col in LAYER_1_FEATURES:
        # Check for exact match or case-insensitive match
        matches = [c for c in df.columns if c.lower() == col.lower().replace('_', '')]
        if col in df.columns:
            available.append(col)
        elif matches:
            available.append(matches[0])
    return available


def get_available_layer2_features(df: pd.DataFrame) -> List[str]:
    """Get Layer 2 features available in the dataframe."""
    available = []
    for col in LAYER_2_FEATURES:
        if col in df.columns:
            available.append(col)
        else:
            # Try common column name variations
            alt_names = [
                col.replace('_', ''),
                col.replace('_', '-'),
                col.title().replace('_', ''),
            ]
            for alt in alt_names:
                if alt in df.columns:
                    available.append(alt)
                    break
    return available


def get_available_layer3_labels(df: pd.DataFrame) -> List[str]:
    """Get Layer 3 columns available for labeling."""
    available = []
    for col in LAYER_3_LABELS:
        if col in df.columns:
            available.append(col)
        else:
            # Try common variations
            alt_names = [
                col.replace('_ratio', ''),
                col.upper(),
                col.title().replace('_', ''),
            ]
            for alt in alt_names:
                if alt in df.columns:
                    available.append(alt)
                    break
    return available


def extract_ark_features(
    df: pd.DataFrame,
    include_layer3: bool = False,
    verbose: bool = True
) -> tuple[pd.DataFrame, List[str]]:
    """
    Extract ARK framework features from dataframe.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with on-chain metrics
    include_layer3 : bool
        If True, include Layer 3 in features (for return forecasting only)
    verbose : bool
        Print feature availability summary
        
    Returns
    -------
    tuple[pd.DataFrame, List[str]]
        (DataFrame with extracted features, list of feature column names)
    """
    l1_cols = get_available_layer1_features(df)
    l2_cols = get_available_layer2_features(df)
    l3_cols = get_available_layer3_labels(df) if include_layer3 else []
    
    feature_cols = l1_cols + l2_cols + l3_cols
    
    if verbose:
        print(f"📊 ARK Feature Extraction:")
        print(f"   Layer 1 (Network Health): {len(l1_cols)}/{len(LAYER_1_FEATURES)} available")
        print(f"   Layer 2 (Buy/Sell):       {len(l2_cols)}/{len(LAYER_2_FEATURES)} available")
        if include_layer3:
            print(f"   Layer 3 (Valuation):      {len(l3_cols)}/{len(LAYER_3_LABELS)} available")
        print(f"   Total features: {len(feature_cols)}")
    
    # Return only available feature columns
    available_in_df = [c for c in feature_cols if c in df.columns]
    
    return df[available_in_df].copy(), available_in_df


def validate_no_forbidden_features(feature_cols: List[str]) -> bool:
    """
    Validate that no forbidden columns are in the feature list.
    
    Parameters
    ----------
    feature_cols : List[str]
        List of feature column names
        
    Returns
    -------
    bool
        True if no forbidden columns, raises ValueError otherwise
    """
    forbidden_found = [c for c in feature_cols if c in FORBIDDEN_COLS]
    
    if forbidden_found:
        raise ValueError(
            f"Forbidden columns found in features (would cause leakage): {forbidden_found}"
        )
    
    # Also check Layer 3 isn't accidentally included for regime classification
    layer3_in_features = [c for c in feature_cols if c in LAYER_3_LABELS]
    if layer3_in_features:
        print(f"⚠️  Warning: Layer 3 columns in features (OK for returns, NOT for regimes): {layer3_in_features}")
    
    return True


if __name__ == "__main__":
    print("ARK 3-Layer Feature Definitions:")
    print(f"  Layer 1 ({len(LAYER_1_FEATURES)} features): {LAYER_1_FEATURES}")
    print(f"  Layer 2 ({len(LAYER_2_FEATURES)} features): {LAYER_2_FEATURES}")
    print(f"  Layer 3 ({len(LAYER_3_LABELS)} labels):   {LAYER_3_LABELS}")
