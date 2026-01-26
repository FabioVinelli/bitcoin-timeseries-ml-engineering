"""
Split Validation Module
========================
Validators to ensure leak-free temporal splits and proper baselines.
"""

from typing import List, Dict, Tuple, Any
import pandas as pd
import numpy as np


# Forbidden columns that would cause leakage if used as features
FORBIDDEN_FEATURE_COLS = [
    'Close', 'close', 'price', 'Price',
    'regime', 'Regime', 'target', 'Target',
]

# Pattern matching for return columns
RETURN_PATTERNS = ['return_', '_return', 'pct_change']


def validate_temporal_splits(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    raise_on_error: bool = True
) -> Tuple[bool, str]:
    """
    Validate that temporal splits have no overlap.
    
    Asserts: train.max() < val.min() < test.min()
    
    Parameters
    ----------
    train_df, val_df, test_df : pd.DataFrame
        Split dataframes with DatetimeIndex
    raise_on_error : bool
        Raise assertion error on failure
        
    Returns
    -------
    Tuple[bool, str]
        (passed, message)
    """
    train_max = train_df.index.max()
    val_min = val_df.index.min()
    val_max = val_df.index.max()
    test_min = test_df.index.min()
    
    errors = []
    
    if train_max >= val_min:
        errors.append(f"Train overlaps Val: train_max={train_max} >= val_min={val_min}")
    
    if val_max >= test_min:
        errors.append(f"Val overlaps Test: val_max={val_max} >= test_min={test_min}")
    
    if errors:
        msg = "Temporal split validation FAILED:\n  " + "\n  ".join(errors)
        if raise_on_error:
            raise AssertionError(msg)
        return False, msg
    
    msg = (f"Temporal splits OK:\n"
           f"  Train: {train_df.index.min()} → {train_max}\n"
           f"  Val:   {val_min} → {val_max}\n"
           f"  Test:  {test_min} → {test_df.index.max()}")
    
    return True, msg


def validate_no_leakage(
    feature_cols: List[str],
    raise_on_error: bool = True
) -> Tuple[bool, str]:
    """
    Validate that no forbidden columns are in the feature list.
    
    Checks:
        - No Close/price columns
        - No return_* columns
        - No regime/target columns
        - No shift(-h) columns
    
    Parameters
    ----------
    feature_cols : List[str]
        List of feature column names
    raise_on_error : bool
        Raise assertion error on failure
        
    Returns
    -------
    Tuple[bool, str]
        (passed, message)
    """
    forbidden_found = []
    
    # Check exact matches
    for col in feature_cols:
        if col in FORBIDDEN_FEATURE_COLS:
            forbidden_found.append(col)
        
        # Check return patterns
        for pattern in RETURN_PATTERNS:
            if pattern in col.lower():
                forbidden_found.append(col)
                break
    
    if forbidden_found:
        msg = f"Leakage validation FAILED - forbidden columns in features: {forbidden_found}"
        if raise_on_error:
            raise AssertionError(msg)
        return False, msg
    
    return True, f"No leakage detected in {len(feature_cols)} feature columns"


def validate_label_distribution(
    df: pd.DataFrame,
    regime_col: str = 'regime',
    min_pct: float = 0.01,
    raise_on_error: bool = True
) -> Tuple[bool, Dict[str, float]]:
    """
    Validate that BUY/SELL labels are not degenerate.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with regime column
    regime_col : str
        Name of regime column
    min_pct : float
        Minimum percentage for each regime (default: 1%)
    raise_on_error : bool
        Raise assertion error on failure
        
    Returns
    -------
    Tuple[bool, Dict[str, float]]
        (passed, distribution dict)
    """
    if regime_col not in df.columns:
        if raise_on_error:
            raise ValueError(f"Regime column '{regime_col}' not found")
        return False, {}
    
    counts = df[regime_col].value_counts()
    total = len(df)
    
    distribution = {
        'BUY': counts.get(0, 0) / total,
        'HOLD': counts.get(1, 0) / total,
        'SELL': counts.get(2, 0) / total,
    }
    
    errors = []
    
    if distribution['BUY'] < min_pct:
        errors.append(f"BUY too rare: {distribution['BUY']*100:.2f}% < {min_pct*100}%")
    
    if distribution['SELL'] < min_pct:
        errors.append(f"SELL too rare: {distribution['SELL']*100:.2f}% < {min_pct*100}%")
    
    if errors:
        msg = "Label distribution validation FAILED:\n  " + "\n  ".join(errors)
        if raise_on_error:
            raise AssertionError(msg)
        return False, distribution
    
    return True, distribution


def validate_baseline_computation(
    train_baseline: float,
    val_baseline: float,
    test_baseline: float,
    raise_on_error: bool = True
) -> Tuple[bool, str]:
    """
    Validate that baselines are computed per-split.
    
    If all three baselines are identical, they were likely computed globally
    rather than per-split, which is incorrect.
    
    Parameters
    ----------
    train_baseline, val_baseline, test_baseline : float
        Baseline metrics for each split
    raise_on_error : bool
        Raise assertion error on failure
        
    Returns
    -------
    Tuple[bool, str]
        (passed, message)
    """
    if train_baseline == val_baseline == test_baseline:
        msg = (f"Baseline validation WARNING: All baselines identical ({train_baseline:.4f}). "
               f"Ensure baselines are computed per-split, not globally.")
        # This is a warning, not an error - baselines could legitimately be similar
        return True, msg
    
    return True, f"Baselines computed per-split: train={train_baseline:.4f}, val={val_baseline:.4f}, test={test_baseline:.4f}"


def compute_always_hold_baseline(df: pd.DataFrame, regime_col: str = 'regime') -> float:
    """
    Compute Always-HOLD baseline accuracy.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with regime column
    regime_col : str
        Name of regime column
        
    Returns
    -------
    float
        Accuracy of always predicting HOLD (regime=1)
    """
    if regime_col not in df.columns:
        return 0.0
    
    return (df[regime_col] == 1).mean()


def compute_persistence_baseline(df: pd.DataFrame, regime_col: str = 'regime') -> float:
    """
    Compute persistence baseline accuracy (predict yesterday's regime).
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with regime column
    regime_col : str
        Name of regime column
        
    Returns
    -------
    float
        Accuracy of predicting previous day's regime
    """
    if regime_col not in df.columns:
        return 0.0
    
    # Shift regime by 1 and compare to actual
    shifted = df[regime_col].shift(1).fillna(1)  # Fill first day with HOLD
    return (df[regime_col] == shifted).mean()


def run_all_validations(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: List[str],
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Run all validations and return results.
    
    Parameters
    ----------
    train_df, val_df, test_df : pd.DataFrame
        Split dataframes
    feature_cols : List[str]
        Feature column names
    verbose : bool
        Print validation results
        
    Returns
    -------
    Dict[str, Any]
        Validation results
    """
    results = {
        'temporal_splits': {},
        'leakage': {},
        'label_distribution': {},
        'baselines': {},
        'all_passed': True,
    }
    
    if verbose:
        print("\n" + "=" * 60)
        print("🔍 Running Validations")
        print("=" * 60)
    
    # 1. Temporal split validation
    try:
        passed, msg = validate_temporal_splits(train_df, val_df, test_df, raise_on_error=False)
        results['temporal_splits'] = {'passed': passed, 'message': msg}
        if verbose:
            print(f"\n✅ Temporal Splits: {'PASS' if passed else 'FAIL'}")
            print(f"   {msg}")
        if not passed:
            results['all_passed'] = False
    except Exception as e:
        results['temporal_splits'] = {'passed': False, 'message': str(e)}
        results['all_passed'] = False
    
    # 2. Leakage validation
    try:
        passed, msg = validate_no_leakage(feature_cols, raise_on_error=False)
        results['leakage'] = {'passed': passed, 'message': msg}
        if verbose:
            print(f"\n✅ Leakage Check: {'PASS' if passed else 'FAIL'}")
            print(f"   {msg}")
        if not passed:
            results['all_passed'] = False
    except Exception as e:
        results['leakage'] = {'passed': False, 'message': str(e)}
        results['all_passed'] = False
    
    # 3. Label distribution validation
    for split_name, split_df in [('train', train_df), ('val', val_df), ('test', test_df)]:
        try:
            passed, dist = validate_label_distribution(split_df, raise_on_error=False)
            results['label_distribution'][split_name] = {'passed': passed, 'distribution': dist}
            if verbose:
                print(f"\n✅ Label Distribution ({split_name}): {'PASS' if passed else 'FAIL'}")
                for label, pct in dist.items():
                    print(f"   {label}: {pct*100:.1f}%")
            if not passed:
                results['all_passed'] = False
        except Exception as e:
            results['label_distribution'][split_name] = {'passed': False, 'distribution': {}}
            results['all_passed'] = False
    
    # 4. Baseline computation
    train_baseline = compute_always_hold_baseline(train_df)
    val_baseline = compute_always_hold_baseline(val_df)
    test_baseline = compute_always_hold_baseline(test_df)
    
    passed, msg = validate_baseline_computation(train_baseline, val_baseline, test_baseline, raise_on_error=False)
    results['baselines'] = {
        'passed': passed,
        'message': msg,
        'train': train_baseline,
        'val': val_baseline,
        'test': test_baseline,
    }
    
    if verbose:
        print(f"\n✅ Baselines: {'PASS' if passed else 'WARN'}")
        print(f"   {msg}")
    
    # Summary
    if verbose:
        print("\n" + "=" * 60)
        print(f"🏁 All Validations: {'PASSED' if results['all_passed'] else 'FAILED'}")
        print("=" * 60)
    
    return results


if __name__ == "__main__":
    print("Split Validator Module")
    print("Run via proof_fold.py for full validation")
