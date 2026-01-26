## Objective

Decide whether your **ARK 3-layer + halving-aware + multi-head (forecast/regime/anomaly)** approach is the right backbone for your Bitcoin ML repo.

## Decision/Constraint

**APPROVE (with constraints).**
This approach is directionally correct **and** portfolio-grade **if** you enforce: (1) no-leakage alignment, (2) conservative on-chain availability timing, (3) walk-forward + cross-cycle testing, and (4) non-degenerate regime labels. 

## Next step

Run a single “proof fold”:

1. build dataset (features + targets),
2. run **one** walk-forward fold for **regime classification** + **return forecasting**,
3. publish `reports/` artifacts (label distribution, leakage checks, baseline comparison).

## Measurement

Success = on the same fold:

* Dataset passes **leakage checks** (no future data; rolling windows past-only; targets aligned).
* Models beat **simple baselines** (predict-0 returns; moving-average/last-return; trivial HOLD classifier).
* Regime labels have sane balance (e.g., HOLD 60–90%, BUY/SELL not near 0%).
  Failure = any fold shows suspiciously high accuracy/ROC (typical leakage), or BUY/SELL almost never occur.

## Review trigger

Review after **1 dataset build + 1 walk-forward fold**, or immediately if:

* SELL precision collapses,
* label distribution is skewed,
* cross-cycle generalization is near-random.

---

# Do I agree? Yes — with the 4 critical constraints below

### 1) Treat on-chain metrics as “delayed publication” by default

Even if your CSV is daily, you don’t always know when the metric becomes available. Conservative fix:

* shift on-chain features by **+1 day** (or provider-specific lag) before training.
  This single step prevents “silent look-ahead,” the #1 way crypto ML looks amazing and fails live.

### 2) Separate *targets* from *label features* to avoid tautology

If your regime label uses MVRV/RPV/etc, and the model also trains on those same features, you’ll get an overly easy task (model learns your rules).
**Fix:** keep two tracks:

* **Rule-based regime signals** (ARK-style thresholds) as a benchmark/explainer.
* **Learned regime classifier** trained on *broader* features and/or *different* inputs than the label definition (or use soft scores instead of hard labels).

### 3) Halving-aware splits are correct — but you must also test cross-cycle

Within-cycle walk-forward is good engineering.
But the real question is: **does it survive regime drift?**
Minimum evaluation set:

* Train Cycle 2→ validate late Cycle 2
* Train Cycle 2–3→ test early Cycle 4
  If cross-cycle collapses, that’s not failure — it’s a signal to move to **ensemble-by-phase** or **domain adaptation**.

### 4) Use returns as the primary forecasting target

Forecasting price level invites “trend proxies” and makes metrics look better than they are.
Use:

* `y = log(close_{t+h}/close_t)` for h ∈ {7,14,30}
  and evaluate directional + trading-aware metrics.

---

If you execute the “proof fold” above and it passes the measurements, your approach is not just valid — it’s the right spine for a serious repo.
