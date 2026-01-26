"""
Features Module
===============
ARK 3-layer on-chain features and halving-cycle features for Bitcoin regime classification.
"""

from .ark_layers import (
    LAYER_1_FEATURES,
    LAYER_2_FEATURES, 
    LAYER_3_LABELS,
    FEATURE_COLS,
    extract_ark_features
)
from .halving_features import (
    HALVING_DATES,
    add_halving_features,
    get_halving_cycle,
    get_cycle_phase
)
from .feature_engineering import (
    prepare_proof_fold_features,
    create_regime_labels,
    create_percentile_regime_labels,
    ONCHAIN_COLS,
    PERCENTILE_WINDOW,
    PERCENTILE_BUY,
    PERCENTILE_SELL
)

__all__ = [
    'LAYER_1_FEATURES',
    'LAYER_2_FEATURES',
    'LAYER_3_LABELS',
    'FEATURE_COLS',
    'extract_ark_features',
    'HALVING_DATES',
    'add_halving_features',
    'get_halving_cycle',
    'get_cycle_phase',
    'prepare_proof_fold_features',
    'create_regime_labels',
    'create_percentile_regime_labels',
    'ONCHAIN_COLS',
    'PERCENTILE_WINDOW',
    'PERCENTILE_BUY',
    'PERCENTILE_SELL'
]
