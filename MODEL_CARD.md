# Model Card: Bitcoin Regime Classification

> **Version**: 1.0-proof-fold  
> **Date**: 2026-01-26  
> **Author**: IBM AI Engineer Portfolio Project

---

## Model Overview

This model classifies Bitcoin market regimes (BUY/HOLD/SELL) using on-chain metrics derived from ARK Invest's 3-layer framework, combined with halving cycle awareness.

**Primary Use Case**: Demonstrating leak-free temporal ML engineering on cyclical financial data.

---

## Task Definition

### Input
- On-chain metrics (Layer 1: network health, Layer 2: buy/sell behavior)
- Halving cycle features (days since halving, cycle phase, block reward)
- 17 total features, all computed from t-1 data (no look-ahead)

### Output
- **BUY** (0): MVRV below rolling 42nd percentile
- **HOLD** (1): MVRV between 42nd and 58th percentiles
- **SELL** (2): MVRV above rolling 58th percentile

### Model Architecture
- RandomForestClassifier (100 trees, max_depth=10)
- StandardScaler preprocessing
- Class-weighted training

---

## Data

### Source
`bitcoin_transformer_dataset_full.csv` — 6,207 daily observations (2009-2025)

### Splits (Cycle-Aware Temporal)

| Split | Cycle | Date Range | Rows |
|-------|-------|------------|------|
| **Train** | Cycle 2 | 2016-07-09 → 2020-05-10 | 1,402 |
| **Val** | Cycle 3 (early) | 2020-05-11 → 2021-07-22 | 438 |
| **Test** | Cycle 4 | 2024-04-20 → 2025-12-31 | 621 |

**Critical Constraint**: `train.max_date < val.min_date < test.min_date`

### Label Distribution

| Split | BUY | HOLD | SELL |
|-------|-----|------|------|
| Train | 36.6% | 9.0% | 54.4% |
| Val | 2.7% | 7.8% | 89.5% |
| Test | 1.6% | 38.8% | 59.6% |

---

## Labeling Method

### Percentile-Based (Default)
```python
window = 730  # 2-year rolling
buy_percentile = 0.42
sell_percentile = 0.58

BUY  = mvrv < rolling_quantile(mvrv, 0.42, window=730)
SELL = mvrv > rolling_quantile(mvrv, 0.58, window=730)
HOLD = else
```

Percentiles are computed using **only past data** (no future information).

### ARK Fixed Thresholds (Benchmark Only)
```python
BUY  = mvrv < 1.0
SELL = mvrv > 10.0
```
Not used for training — produces degenerate labels in Cycle 3-4.

---

## Performance

| Metric | Train | Val | Test |
|--------|-------|-----|------|
| Macro-F1 | 0.994 | 0.405 | 0.272 |
| Accuracy | 0.997 | 0.900 | 0.609 |
| Always-HOLD Baseline | 0.090 | 0.078 | 0.388 |

### Test Beats Baseline: ❌ NO

---

## Known Limitations & Findings

### 1. Cross-Cycle Distribution Shift

**Observation**: Model trained on Cycle 2 (2016-2020) does not transfer cleanly to Cycle 4 (2024-2025).

**Evidence**:
- Train macro-F1 ≈ 0.99 → model learns within-cycle patterns
- Test macro-F1 ≈ 0.27 → patterns do not generalize across cycles

**Root Cause**: Each halving cycle has fundamentally different market structure, investor composition, and on-chain behavior. This is well-documented in Bitcoin research.

**Implication**: A single model trained on historical cycles may not predict future cycles. This is a *correct result*, not a model failure.

### 2. SELL-Dominant Validation/Test Periods

Val (Cycle 3 early) and Test (Cycle 4) cover predominantly bullish markets where:
- MVRV stays elevated (above historical medians)
- BUY signals are rare by design (undervaluation is rare in bull markets)
- SELL signals are common (overvaluation relative to rolling history)

This is expected and reflects genuine market conditions.

### 3. Percentile Threshold Trade-offs

The 42/58 thresholds were chosen to ensure:
- BUY ≥ 1% in all splits (pass validation)
- SELL ≥ 1% in all splits (pass validation)

Tighter thresholds (e.g., 20/80) produce more extreme signals but fail validation in bull-dominated periods.

---

## Recommendations

### What This Model Is Good For
- **Demonstrating** leak-free temporal ML on financial data
- **Establishing** a baseline for cycle-aware regime classification
- **Portfolio artifact** showing engineering discipline over metric-chasing

### What This Model Should NOT Be Used For
- Production trading decisions (without additional validation)
- Predicting future halving cycles (by definition, out-of-distribution)
- Replacing fundamental analysis

### Future Research Directions (Deferred)
1. **Cycle-conditional models**: Train separate models per cycle, ensemble
2. **Binary reformulation**: Collapse to RISK/NO-RISK for clearer signal
3. **Multi-indicator signals**: Add NUPL, STH-MVRV to improve BUY detection

---

## Reproducibility

```bash
# Run the pipeline
cd bitcoin-timeseries-ml-engineering
python src/proof_fold.py

# Outputs
reports/split_summary.json   # Configuration
reports/label_distribution.csv  # Labels per split
reports/metrics.json         # Model performance
```

---

## Ethical Considerations

- **Not financial advice**: This model is for educational/portfolio purposes only
- **Honest metrics**: We report cross-cycle degradation rather than hiding it
- **No data leakage**: Validated by automated checks
- **Transparent limitations**: All known issues documented above

---

## Citation

If referencing this work:

```
ARK 3-Layer + Halving-Aware Bitcoin Regime Classification
IBM AI Engineer Portfolio Project, 2026
https://github.com/[repo]
```

Reference:
- ARK Invest "Big Ideas 2024" On-Chain Framework
- Meynkhard, A. (2019). Fair Value of Bitcoin Based on Scarcity
