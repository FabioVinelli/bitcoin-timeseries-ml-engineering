# Proof Fold Walkthrough: ARK 3-Layer + Halving-Aware Bitcoin Regime Classification

> **Project**: IBM AI Engineer Certification Portfolio  
> **Date**: 2026-01-26  
> **Status**: Milestone A Complete

---

## Executive Summary

This document details the implementation of a **leak-free Bitcoin regime classification pipeline** that combines ARK Invest's 3-layer on-chain framework with Bitcoin halving cycle awareness. The pipeline produces cycle-aware temporal splits, validates for data leakage, and generates reproducible model metrics.

### Key Results

| Metric | Value |
|--------|-------|
| Dataset | 6,207 daily observations (2009-2025) |
| Features | 17 (ARK Layer 1+2 + halving features) |
| Train Split | 1,402 rows (Cycle 2: 2016-2020) |
| Val Split | 438 rows (Cycle 3 early: 2020-2021) |
| Test Split | 621 rows (Cycle 4: 2024-2025) |
| Regime Distribution | BUY 10.9%, HOLD 89.1%, SELL 0.03% |

---

## 1. Methodology

### 1.1 Problem Framing

Bitcoin markets exhibit strong **cyclical behavior** aligned with the ~4-year halving schedule. Traditional ML approaches that use random train/test splits create:

1. **Temporal leakage** — future data informs past predictions
2. **Cycle contamination** — model learns from same market regime it's evaluated on
3. **Unrealistic performance** — metrics don't reflect real-world deployment

Our approach addresses these by implementing **halving-cycle-aware temporal splitting**.

### 1.2 ARK Invest 3-Layer Framework

We adapted ARK Invest's on-chain analysis framework (Big Ideas 2024):

```
┌─────────────────────────────────────────────────────────────┐
│                    ARK 3-LAYER FRAMEWORK                    │
├─────────────────────────────────────────────────────────────┤
│ LAYER 1: Network Health (FEATURES)                         │
│   • hash_rate, active_addresses, transaction_count         │
│   • adjusted_transaction_volume, miner_revenue             │
├─────────────────────────────────────────────────────────────┤
│ LAYER 2: Buy/Sell Behavior (FEATURES)                      │
│   • coin_days_destroyed, realized_price_lth/sth            │
│   • supply_in_profit/loss, realized_profit_loss            │
├─────────────────────────────────────────────────────────────┤
│ LAYER 3: Valuation Signals (LABELS ONLY)                   │
│   • mvrv_ratio → BUY/SELL/HOLD regime labels               │
│   ⚠️ NOT used as features to avoid tautological prediction │
└─────────────────────────────────────────────────────────────┘
```

### 1.3 Halving-Cycle Temporal Splitting

```
Timeline: ──────────────────────────────────────────────────────────►

Cycle 2          │  Cycle 3           │ Gap │    Cycle 4
2016-07-09       │  2020-05-11        │     │    2024-04-20
    ─────────────┼────────────────────┼─────┼─────────────────►
    TRAIN        │  VAL (early)       │     │    TEST
    1,402 days   │  438 days          │     │    621 days
```

**Critical constraint**: `train.max_date < val.min_date < test.min_date`

---

## 2. Data Pipeline

### 2.1 Dataset Source

```
File: bitcoin_transformer_dataset_full.csv
Rows: 6,207 daily observations
Date Range: 2009-01-03 → 2025-12-31
Columns: 37 (price, on-chain metrics, derived features)
```

**Column categories**:
- Price data: open, high, low, close, volume
- On-chain: mvrv_ratio, hash_rate, active_addresses, coin_days_destroyed
- Behavior: supply_in_profit, realized_price_lth/sth
- Sentiment: Fear_Greed_Index, Volatility_Index

### 2.2 Feature Engineering Pipeline

The pipeline follows a **critical order** to prevent data leakage:

```python
# LEAK-FREE PROCESSING ORDER
1. Sort by timestamp (chronological order)
2. Apply +1 day shift to on-chain columns  # Prevents look-ahead bias
3. Add halving cycle features              # days_since_halving, cycle_phase
4. Create regime labels from shifted L3    # BUY/SELL/HOLD
5. Create return targets with shift(-7)    # 7-day forward returns
6. Forward-fill NaN in feature columns     # Handle data gaps
7. Drop rows missing essential columns     # regime, return_7d
```

### 2.3 Regime Labeling Logic

```python
# Explicit BUY/SELL/HOLD conditions (no nested np.where)
BUY  = (mvrv_ratio < 1.0)                    # Undervalued
SELL = (mvrv_ratio > 10.0)                   # Extremely overvalued
HOLD = else                                   # Default state
```

---

## 3. Tools & Technologies

### 3.1 Core Stack

| Component | Tool | Purpose |
|-----------|------|---------|
| Language | Python 3.11 | Core implementation |
| ML Framework | scikit-learn 1.3+ | RandomForest, StandardScaler |
| Data Processing | pandas, numpy | DataFrame operations |
| Validation | Custom validators | Temporal split, leakage checks |

### 3.2 Project Structure

```
bitcoin-timeseries-ml-engineering/
├── src/
│   ├── features/
│   │   ├── ark_layers.py         # ARK 3-layer definitions
│   │   ├── halving_features.py   # Halving cycle computation
│   │   └── feature_engineering.py # Leak-free pipeline
│   ├── validation/
│   │   └── split_validator.py    # Temporal/leakage validators
│   └── proof_fold.py             # Main entrypoint
├── reports/
│   ├── split_summary.json        # Configuration + splits
│   ├── metrics.json              # Model performance
│   └── label_distribution.csv    # Regime counts per split
└── data/
    └── raw/                      # Dataset location
```

### 3.3 Validation Framework

Four automated validators ensure pipeline integrity:

| Validator | Check | Status |
|-----------|-------|--------|
| `validate_temporal_splits()` | train.max < val.min < test.min | ✅ PASS |
| `validate_no_leakage()` | No forbidden columns in features | ✅ PASS |
| `validate_label_distribution()` | BUY/SELL each ≥ 1% | ⚠️ FAIL |
| `validate_baseline_computation()` | Baselines per-split, not global | ✅ PASS |

---

## 4. Key Findings

### 4.1 Temporal Split Validation

```json
{
  "train": {"start": "2016-07-09", "end": "2020-05-10", "rows": 1402},
  "val":   {"start": "2020-05-11", "end": "2021-07-22", "rows": 438},
  "test":  {"start": "2024-04-20", "end": "2025-12-31", "rows": 621}
}
```

✅ **No temporal overlap** — strict chronological ordering maintained.

### 4.2 Feature Composition

17 features extracted from ARK Layer 1+2 plus halving cycle:

```
Layer 1 (Network Health):
  hash_rate, active_addresses, adjusted_transaction_volume, miner_revenue

Layer 2 (Buy/Sell Behavior):  
  coin_days_destroyed, realized_price_lth, realized_price_sth,
  supply_in_profit, supply_in_loss, realized_profit_loss

Halving Features:
  halving_cycle, days_since_halving, cycle_phase, block_reward,
  daily_new_supply, annual_inflation, is_post_halving_150d
```

### 4.3 Regime Distribution Issue

| Split | BUY | HOLD | SELL |
|-------|-----|------|------|
| Train (Cycle 2) | 9.9% | 90.1% | 0.0% |
| Val (Cycle 3) | 0.0% | 100.0% | 0.0% |
| Test (Cycle 4) | 0.0% | 100.0% | 0.0% |

**Root cause**: MVRV thresholds (< 1 for BUY, > 10 for SELL) are calibrated for extreme market conditions:
- BUY signals occur during deep bear markets (MVRV < 1)
- SELL signals require extreme overvaluation (MVRV > 10, occurred only 2× in dataset)
- Cycle 3-4 validation/test periods were predominantly bull/neutral markets

---

## 5. Leveraging the Findings

### 5.1 Immediate Applications

1. **Portfolio Position Sizing**
   - Use predicted regime to adjust BTC allocation
   - BUY regime → increase exposure
   - SELL regime → reduce/hedge exposure
   - HOLD → maintain current allocation

2. **Risk Management**
   - Cycle phase indicates expected volatility regime
   - Post-halving windows (150 days) historically show higher returns

3. **Backtesting Framework**
   - Leak-free splits enable realistic strategy backtests
   - Cross-cycle evaluation tests regime transfer learning

### 5.2 Model Improvements (Milestone B)

To address the label distribution imbalance:

**Option 1: Adjust Thresholds**
```python
# More sensitive thresholds
BUY  = (mvrv_ratio < 1.5)   # Was < 1.0
SELL = (mvrv_ratio > 3.5)   # Was > 10.0
```

**Option 2: Percentile-Based Labeling**
```python
# Use rolling percentiles instead of fixed thresholds
BUY  = (mvrv_ratio < mvrv_rolling_20pct)
SELL = (mvrv_ratio > mvrv_rolling_80pct)
```

**Option 3: Multi-Indicator Signals**
```python
# Combine multiple on-chain signals
BUY  = (mvrv_ratio < 1.5) OR (nupl < 0) OR (sth_mvrv < 0.8)
SELL = (mvrv_ratio > 3.5) AND (nupl > 0.7)
```

### 5.3 Production Deployment Path

```
Phase 1: Regime Classification API
  └─ Deploy model as REST endpoint
  └─ Input: current on-chain metrics
  └─ Output: BUY/SELL/HOLD probability

Phase 2: Real-Time Pipeline
  └─ Integrate Glassnode MCP server for live data
  └─ Daily regime updates with confidence scores

Phase 3: Trading Integration
  └─ Connect to exchange APIs
  └─ Position sizing based on regime + confidence
```

---

## 6. Reproducibility

### 6.1 Run the Pipeline

```bash
cd bitcoin-timeseries-ml-engineering
python src/proof_fold.py
```

### 6.2 Output Files

| File | Description |
|------|-------------|
| `reports/split_summary.json` | Full configuration, features, validation results |
| `reports/metrics.json` | Model performance metrics |
| `reports/label_distribution.csv` | Regime counts per split |

### 6.3 Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `BTC_DATASET_PATH` | Override dataset location | Search `DATASET_SEARCH_PATHS` |

---

## 7. Next Steps

### Milestone B: Label Calibration
- [ ] Implement percentile-based regime thresholds
- [ ] Add STH-MVRV and NUPL as alternative signals
- [ ] Re-run validation with balanced labels

### Milestone C: Tests + Documentation
- [ ] Add pytest unit tests
- [ ] Update README with execution guide
- [ ] Create model card for portfolio

---

## Appendix: Report Files

### A. Split Summary (excerpt)

```json
{
  "USE_LAYER3_FOR_RETURNS": false,
  "splits": {
    "train": {"rows": 1402, "start": "2016-07-09", "end": "2020-05-10"},
    "val":   {"rows": 438,  "start": "2020-05-11", "end": "2021-07-22"},
    "test":  {"rows": 621,  "start": "2024-04-20", "end": "2025-12-31"}
  },
  "validation_results": {
    "temporal_splits": {"passed": true},
    "leakage": {"passed": true},
    "baselines": {"passed": true}
  }
}
```

### B. Metrics Summary

```json
{
  "regime_classification": {
    "train_macro_f1": 1.0,
    "val_macro_f1": 1.0,
    "test_macro_f1": 1.0,
    "train_always_hold_baseline": 0.9009,
    "test_beats_baseline": false
  }
}
```

> **Note**: 100% macro-F1 with 100% HOLD in val/test indicates degenerate labels, not perfect prediction. This is the primary issue for Milestone B.
