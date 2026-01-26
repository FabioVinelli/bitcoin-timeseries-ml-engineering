#!/usr/bin/env python3
"""
Proof Fold: ARK 3-Layer + Halving-Aware Bitcoin Regime Classification
======================================================================

Main entry point for the leak-free proof fold pipeline.

Usage:
    python src/proof_fold.py
    python src/proof_fold.py --verbose
    
Outputs:
    reports/split_summary.json   - Split configuration and regime config
    reports/label_distribution.csv - Regime distribution per split
    reports/metrics.json         - Model performance metrics

Reference:
    ARK Invest "Big Ideas 2024" On-Chain Framework
    Halving cycles 2, 3, 4 (2016-2024+)
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import pandas as pd

# Ensure project root is in path (for running as python src/proof_fold.py)
src_dir = Path(__file__).parent
project_root = src_dir.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.metrics import f1_score, accuracy_score, classification_report
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from src.features.ark_layers import (
    LAYER_1_FEATURES, LAYER_2_FEATURES, LAYER_3_LABELS,
    FEATURE_COLS, FORBIDDEN_COLS,
    extract_ark_features, validate_no_forbidden_features
)
from src.features.halving_features import (
    HALVING_DATES, add_halving_features, get_cycle_split
)
from src.features.feature_engineering import (
    prepare_proof_fold_features, create_regime_labels, HORIZON_DAYS
)
from src.validation.split_validator import (
    run_all_validations, 
    compute_always_hold_baseline,
    compute_persistence_baseline
)


# =============================================================================
# Configuration
# =============================================================================

# Dataset configuration - search paths in priority order
DATASET_SEARCH_PATHS = [
    # Environment variable or default
    "data/raw/bitcoin_transformer_dataset_full.csv",
    # BTC-CRYPTO-DATA location (note: filename has space)
    "/Users/FV/Dev.Ops/IBM-GEN-AI-CERT/Projects-Gitub-Dev/BTC-LSTM-Project/data/BTC-CRYPTO-DATA/bitcoin_transformer_dataset_full - bitcoin_transformer_dataset_full.csv",
    "/Users/FV/Dev.Ops/IBM-GEN-AI-CERT/Projects-Gitub-Dev/BTC-LSTM-Project/data/raw/bitcoin_transformer_dataset_full - bitcoin_transformer_dataset_full.csv",
]
FALLBACK_DATASET_DIR = "/Users/FV/Dev.Ops/IBM-GEN-AI-CERT/Projects-Gitub-Dev/BTC-LSTM-Project/data/BTC-CRYPTO-DATA"

# Model configuration
USE_LAYER3_FOR_RETURNS = False  # Config switch - written to split_summary.json

# Reports directory
REPORTS_DIR = Path("reports")


def load_dataset(verbose: bool = True) -> pd.DataFrame:
    """
    Load Bitcoin on-chain dataset.
    
    Priority:
        1. BTC_DATASET_PATH environment variable
        2. DATASET_SEARCH_PATHS in order
        3. Fallback to BTC-CRYPTO-DATA loader
    
    Returns
    -------
    pd.DataFrame
        Dataset with on-chain metrics
    """
    # Try environment variable first
    env_path = os.environ.get("BTC_DATASET_PATH")
    search_paths = [env_path] if env_path else []
    search_paths.extend(DATASET_SEARCH_PATHS)
    
    # Search for dataset
    for dataset_path in search_paths:
        if dataset_path and os.path.exists(dataset_path):
            if verbose:
                print(f"📂 Loading dataset from: {dataset_path}")
            
            # Read first row to detect columns
            sample = pd.read_csv(dataset_path, nrows=1)
            
            # Find date column
            date_col = None
            for col in sample.columns:
                if col.lower() in ['date', 'datetime', 'time', 'timestamp']:
                    date_col = col
                    break
            
            df = pd.read_csv(dataset_path)
            
            # Set date index
            if date_col:
                df[date_col] = pd.to_datetime(df[date_col])
                df.set_index(date_col, inplace=True)
            elif 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date'])
                df.set_index('Date', inplace=True)
            elif 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
                df.set_index('date', inplace=True)
                
            df = df[~df.index.duplicated(keep='first')]
            df.sort_index(inplace=True)
            
            if verbose:
                print(f"   Loaded {len(df)} rows, {len(df.columns)} columns")
                print(f"   Date range: {df.index[0]} → {df.index[-1]}")
            
            return df
    
    # Fallback to BTC-CRYPTO-DATA loader
    if verbose:
        print(f"⚠️  No dataset found in search paths")
        print(f"   Attempting fallback loader from: {FALLBACK_DATASET_DIR}")
    
    # Try to use the HalvingDataLoader from the BTC-LSTM-Project
    try:
        halving_loader_path = Path(FALLBACK_DATASET_DIR).parent.parent / "src" / "alpha" / "halving_data_loader.py"
        if halving_loader_path.exists():
            import importlib.util
            spec = importlib.util.spec_from_file_location("halving_data_loader", halving_loader_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            loader = module.HalvingDataLoader(FALLBACK_DATASET_DIR)
            df = loader.merge_all_data(start_date='2016-07-09')
            
            if verbose:
                print(f"   Loaded {len(df)} rows via HalvingDataLoader")
            
            return df
    except Exception as e:
        if verbose:
            print(f"   Fallback loader failed: {e}")
    
    raise FileNotFoundError(
        f"Could not load dataset. Set BTC_DATASET_PATH env var or check DATASET_SEARCH_PATHS"
    )


def get_feature_cols_available(df: pd.DataFrame) -> List[str]:
    """
    Get available feature columns from dataframe.
    
    Uses Layer 1 + Layer 2 + halving features for regime classification.
    """
    # ARK Layer 1 + Layer 2 features
    ark_features = [c for c in FEATURE_COLS if c in df.columns]
    
    # Halving features
    halving_features = [c for c in df.columns if c.startswith('halving_') or 
                       c in ['days_since_halving', 'cycle_phase', 'block_reward', 
                             'daily_new_supply', 'annual_inflation', 'is_post_halving_150d']]
    
    # Combine and validate
    feature_cols = ark_features + halving_features
    
    # Remove any forbidden columns that might have slipped through
    feature_cols = [c for c in feature_cols if c not in FORBIDDEN_COLS]
    
    return feature_cols


def train_regime_classifier(
    X_train: np.ndarray,
    y_train: np.ndarray,
    random_state: int = 42
) -> Pipeline:
    """
    Train regime classifier using RandomForest.
    
    Parameters
    ----------
    X_train : np.ndarray
        Training features
    y_train : np.ndarray
        Training labels (0=BUY, 1=HOLD, 2=SELL)
    random_state : int
        Random seed
        
    Returns
    -------
    Pipeline
        Fitted classifier pipeline with scaler
    """
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight='balanced',
            random_state=random_state,
            n_jobs=-1
        ))
    ])
    
    pipeline.fit(X_train, y_train)
    
    return pipeline


def train_return_forecaster(
    X_train: np.ndarray,
    y_train: np.ndarray,
    random_state: int = 42
) -> Pipeline:
    """
    Train return forecaster using GradientBoosting.
    
    Parameters
    ----------
    X_train : np.ndarray
        Training features
    y_train : np.ndarray
        Training targets (7-day returns)
    random_state : int
        Random seed
        
    Returns
    -------
    Pipeline
        Fitted regressor pipeline with scaler
    """
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('reg', GradientBoostingRegressor(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            min_samples_split=5,
            random_state=random_state
        ))
    ])
    
    pipeline.fit(X_train, y_train)
    
    return pipeline


def evaluate_regime_classifier(
    clf: Pipeline,
    X: np.ndarray,
    y: np.ndarray,
    split_name: str = "test"
) -> Dict[str, float]:
    """
    Evaluate regime classifier.
    
    Returns
    -------
    Dict[str, float]
        Metrics including macro-F1, accuracy, per-class F1
    """
    y_pred = clf.predict(X)
    
    metrics = {
        f'{split_name}_macro_f1': f1_score(y, y_pred, average='macro', zero_division=0),
        f'{split_name}_accuracy': accuracy_score(y, y_pred),
        f'{split_name}_f1_buy': f1_score(y == 0, y_pred == 0, zero_division=0),
        f'{split_name}_f1_hold': f1_score(y == 1, y_pred == 1, zero_division=0),
        f'{split_name}_f1_sell': f1_score(y == 2, y_pred == 2, zero_division=0),
    }
    
    return metrics


def evaluate_return_forecaster(
    reg: Pipeline,
    X: np.ndarray,
    y: np.ndarray,
    split_name: str = "test"
) -> Dict[str, float]:
    """
    Evaluate return forecaster.
    
    Returns
    -------
    Dict[str, float]
        Metrics including RMSE, MAE, directional accuracy
    """
    y_pred = reg.predict(X)
    
    # RMSE and MAE
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    mae = mean_absolute_error(y, y_pred)
    
    # Directional accuracy
    direction_actual = np.sign(y)
    direction_pred = np.sign(y_pred)
    directional_acc = (direction_actual == direction_pred).mean()
    
    # Predict-zero baseline RMSE
    predict_zero_rmse = np.sqrt(mean_squared_error(y, np.zeros_like(y)))
    
    metrics = {
        f'{split_name}_rmse': rmse,
        f'{split_name}_mae': mae,
        f'{split_name}_directional_accuracy': directional_acc,
        f'{split_name}_predict_zero_baseline_rmse': predict_zero_rmse,
        f'{split_name}_rmse_beats_baseline': rmse < predict_zero_rmse,
    }
    
    return metrics


def save_reports(
    split_summary: Dict[str, Any],
    label_distribution: pd.DataFrame,
    metrics: Dict[str, Any],
    reports_dir: Path = REPORTS_DIR,
    verbose: bool = True
) -> None:
    """
    Save reports to disk.
    
    Outputs:
        reports/split_summary.json
        reports/label_distribution.csv
        reports/metrics.json
    """
    reports_dir.mkdir(parents=True, exist_ok=True)
    
    # Save split summary
    summary_path = reports_dir / "split_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(split_summary, f, indent=2, default=str)
    
    # Save label distribution
    dist_path = reports_dir / "label_distribution.csv"
    label_distribution.to_csv(dist_path)
    
    # Save metrics
    metrics_path = reports_dir / "metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2, default=str)
    
    if verbose:
        print(f"\n📁 Reports saved to {reports_dir}:")
        print(f"   ✓ {summary_path.name}")
        print(f"   ✓ {dist_path.name}")
        print(f"   ✓ {metrics_path.name}")


def main(verbose: bool = True) -> Dict[str, Any]:
    """
    Main proof fold pipeline.
    
    Returns
    -------
    Dict[str, Any]
        Combined metrics and configuration
    """
    print("=" * 70)
    print("🔮 Proof Fold: ARK 3-Layer + Halving-Aware Regime Classification")
    print("=" * 70)
    print(f"   Timestamp: {datetime.now().isoformat()}")
    print(f"   USE_LAYER3_FOR_RETURNS: {USE_LAYER3_FOR_RETURNS}")
    print()
    
    # =========================================================================
    # Step 1: Load Data
    # =========================================================================
    df = load_dataset(verbose=verbose)
    
    # =========================================================================
    # Step 2: Feature Engineering (leak-free order)
    # =========================================================================
    df, feature_config = prepare_proof_fold_features(df, verbose=verbose)
    
    # =========================================================================
    # Step 3: Cycle-Aware Splitting
    # =========================================================================
    print("\n" + "=" * 60)
    print("📊 Cycle-Aware Temporal Splitting")
    print("=" * 60)
    
    train_df, val_df, test_df = get_cycle_split(df)
    
    print(f"   Train (Cycle 2): {len(train_df)} rows")
    print(f"   Val (Cycle 3 early): {len(val_df)} rows")
    print(f"   Test (Cycle 4): {len(test_df)} rows")
    
    # =========================================================================
    # Step 4: Get Feature Columns
    # =========================================================================
    feature_cols = get_feature_cols_available(train_df)
    
    if verbose:
        print(f"\n📋 Features ({len(feature_cols)} total):")
        print(f"   {feature_cols[:5]}...")
    
    # Validate no leakage
    validate_no_forbidden_features(feature_cols)
    
    # =========================================================================
    # Step 5: Run Validations
    # =========================================================================
    validation_results = run_all_validations(
        train_df, val_df, test_df, 
        feature_cols, 
        verbose=verbose
    )
    
    # =========================================================================
    # Step 6: Train Regime Classifier
    # =========================================================================
    print("\n" + "=" * 60)
    print("🤖 Training Regime Classifier")
    print("=" * 60)
    
    X_train = train_df[feature_cols].values
    y_train = train_df['regime'].values
    X_val = val_df[feature_cols].values
    y_val = val_df['regime'].values
    X_test = test_df[feature_cols].values
    y_test = test_df['regime'].values
    
    clf = train_regime_classifier(X_train, y_train)
    
    # Evaluate
    train_metrics = evaluate_regime_classifier(clf, X_train, y_train, 'train')
    val_metrics = evaluate_regime_classifier(clf, X_val, y_val, 'val')
    test_metrics = evaluate_regime_classifier(clf, X_test, y_test, 'test')
    
    print(f"\n   Train macro-F1: {train_metrics['train_macro_f1']:.4f}")
    print(f"   Val macro-F1:   {val_metrics['val_macro_f1']:.4f}")
    print(f"   Test macro-F1:  {test_metrics['test_macro_f1']:.4f}")
    
    # Baselines
    train_always_hold = compute_always_hold_baseline(train_df)
    val_always_hold = compute_always_hold_baseline(val_df)
    test_always_hold = compute_always_hold_baseline(test_df)
    
    print(f"\n   Baselines (Always-HOLD accuracy):")
    print(f"   Train: {train_always_hold:.4f}")
    print(f"   Val:   {val_always_hold:.4f}")
    print(f"   Test:  {test_always_hold:.4f}")
    
    # Check if model beats baseline
    test_beats_baseline = test_metrics['test_macro_f1'] > test_always_hold
    print(f"\n   Test beats Always-HOLD: {'✅ YES' if test_beats_baseline else '❌ NO'}")
    
    # =========================================================================
    # Step 7: Train Return Forecaster (optional)
    # =========================================================================
    return_col = f'return_{HORIZON_DAYS}d'
    return_metrics = {}
    
    if return_col in train_df.columns:
        print("\n" + "=" * 60)
        print("📈 Training Return Forecaster")
        print("=" * 60)
        
        y_train_ret = train_df[return_col].values
        y_val_ret = val_df[return_col].values
        y_test_ret = test_df[return_col].values
        
        # Use same features (optionally add Layer 3 for returns)
        reg = train_return_forecaster(X_train, y_train_ret)
        
        return_metrics = evaluate_return_forecaster(reg, X_test, y_test_ret, 'test')
        
        print(f"\n   Test RMSE: {return_metrics['test_rmse']:.6f}")
        print(f"   Predict-0 Baseline RMSE: {return_metrics['test_predict_zero_baseline_rmse']:.6f}")
        print(f"   Beats baseline: {'✅ YES' if return_metrics['test_rmse_beats_baseline'] else '❌ NO'}")
    
    # =========================================================================
    # Step 8: Prepare Reports
    # =========================================================================
    
    # Split summary
    split_summary = {
        'timestamp': datetime.now().isoformat(),
        'USE_LAYER3_FOR_RETURNS': USE_LAYER3_FOR_RETURNS,
        'splits': {
            'train': {'rows': len(train_df), 'start': str(train_df.index.min()), 'end': str(train_df.index.max())},
            'val': {'rows': len(val_df), 'start': str(val_df.index.min()), 'end': str(val_df.index.max())},
            'test': {'rows': len(test_df), 'start': str(test_df.index.min()), 'end': str(test_df.index.max())},
        },
        'feature_config': feature_config,
        'feature_cols': feature_cols,
        'validation_results': validation_results,
    }
    
    # Label distribution DataFrame
    label_dist_data = []
    for split_name, split_df in [('train', train_df), ('val', val_df), ('test', test_df)]:
        counts = split_df['regime'].value_counts().sort_index()
        total = len(split_df)
        label_dist_data.append({
            'split': split_name,
            'BUY_count': counts.get(0, 0),
            'HOLD_count': counts.get(1, 0),
            'SELL_count': counts.get(2, 0),
            'BUY_pct': counts.get(0, 0) / total * 100,
            'HOLD_pct': counts.get(1, 0) / total * 100,
            'SELL_pct': counts.get(2, 0) / total * 100,
        })
    label_distribution = pd.DataFrame(label_dist_data)
    
    # Combined metrics
    all_metrics = {
        'timestamp': datetime.now().isoformat(),
        'regime_classification': {
            **train_metrics,
            **val_metrics,
            **test_metrics,
            'train_always_hold_baseline': train_always_hold,
            'val_always_hold_baseline': val_always_hold,
            'test_always_hold_baseline': test_always_hold,
            'test_beats_baseline': test_beats_baseline,
        },
        'return_forecasting': return_metrics,
        'validation_passed': validation_results['all_passed'],
    }
    
    # =========================================================================
    # Step 9: Save Reports
    # =========================================================================
    save_reports(split_summary, label_distribution, all_metrics, verbose=verbose)
    
    # =========================================================================
    # Final Summary
    # =========================================================================
    print("\n" + "=" * 70)
    print("🏁 PROOF FOLD COMPLETE")
    print("=" * 70)
    print(f"\n   ✅ All validations: {'PASSED' if validation_results['all_passed'] else 'FAILED'}")
    print(f"   ✅ Test macro-F1: {test_metrics['test_macro_f1']:.4f} (baseline: {test_always_hold:.4f})")
    if return_metrics:
        print(f"   ✅ Test RMSE: {return_metrics['test_rmse']:.6f}")
    print(f"\n   📁 Reports saved to: {REPORTS_DIR.absolute()}")
    
    return all_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Proof Fold: ARK 3-Layer + Halving-Aware Regime Classification")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    args = parser.parse_args()
    
    metrics = main(verbose=args.verbose or True)
