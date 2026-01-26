#!/usr/bin/env python3
"""
Generate regime heatmap and drawdown comparison charts (synthetic-safe for public repo).
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ----------------------------
# Synthetic-safe generation
# ----------------------------
np.random.seed(42)
dates = pd.date_range("2024-01-01", periods=500, freq="D")
n = len(dates)

def simulate_equity(mu, sigma, start=1.0):
    r = np.random.normal(mu, sigma, n)
    eq = start * np.cumprod(1 + r)
    return pd.Series(eq, index=dates)

def max_drawdown(equity: pd.Series) -> pd.Series:
    peak = equity.cummax()
    dd = (equity / peak) - 1.0
    return dd

def label_regimes(returns: pd.Series, lookback=30, bull_thr=0.0, bear_thr=0.0):
    """
    Simple regime labeling:
    - Use rolling return + rolling volatility to label bull/bear/sideways.
    This is illustrative; swap with your production regime detector if desired.
    """
    roll_ret = returns.rolling(lookback).mean()
    roll_vol = returns.rolling(lookback).std()

    # Basic rule:
    # bull if rolling mean > bull_thr and vol not extreme
    # bear if rolling mean < -bear_thr
    # sideways otherwise
    regime = pd.Series("sideways", index=returns.index)
    regime[roll_ret > bull_thr] = "bull"
    regime[roll_ret < -bear_thr] = "bear"

    # optional: if vol is extremely low/high you can refine; kept simple.
    return regime

# ----------------------------
# 1) Drawdown comparison chart
# ----------------------------
# Buy & Hold: higher drift, higher vol
equity_bh = simulate_equity(mu=0.0015, sigma=0.020)
# ML strategy: slightly lower drift, moderate vol
equity_ml = simulate_equity(mu=0.0010, sigma=0.015)
# QFL-DCA benchmark: modest drift, lower vol (drawdown-controlled shape)
equity_qfl = simulate_equity(mu=0.0012, sigma=0.010)

dd_bh = max_drawdown(equity_bh)
dd_ml = max_drawdown(equity_ml)
dd_qfl = max_drawdown(equity_qfl)

plt.figure(figsize=(14, 6))
plt.plot(dd_bh.index, dd_bh.values * 100, label="Buy & Hold", color="gray")
plt.plot(dd_ml.index, dd_ml.values * 100, label="ML-driven Strategy", color="#1f77b4")
plt.plot(dd_qfl.index, dd_qfl.values * 100, label="QFL-DCA (Benchmark)", color="green")
plt.axhline(0, linewidth=1, color="black", linestyle="--", alpha=0.3)

plt.title("Drawdown Comparison (Synthetic, % from peak)")
plt.xlabel("Date")
plt.ylabel("Drawdown (%)")
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()

output_dir = Path("docs/images")
output_dir.mkdir(parents=True, exist_ok=True)
plt.savefig(output_dir / "drawdown_comparison.png", dpi=150)
plt.close()

# ----------------------------
# 2) Regime heatmap (bull/bear/sideways)
# ----------------------------
# Create a synthetic price series and compute returns for regime classification
price = 100 * (equity_bh / equity_bh.iloc[0])  # use BH as "market proxy"
ret = price.pct_change().fillna(0)

regime = label_regimes(ret, lookback=30, bull_thr=0.0002, bear_thr=0.0002)

# Aggregate by month: count days in each regime (heatmap-friendly table)
monthly = pd.DataFrame({"regime": regime})
monthly["month"] = monthly.index.to_period("M").astype(str)

counts = monthly.groupby(["month", "regime"]).size().unstack(fill_value=0)
# Ensure consistent column order
for col in ["bull", "bear", "sideways"]:
    if col not in counts.columns:
        counts[col] = 0
counts = counts[["bull", "bear", "sideways"]]

# Convert counts to percentages of days in month
row_sums = counts.sum(axis=1).replace(0, 1)
pct = (counts.T / row_sums).T * 100.0  # month x regime

# Plot heatmap (manual imshow to avoid seaborn dependency)
plt.figure(figsize=(14, 6))
data = pct.values  # shape: months x 3
im = plt.imshow(data, aspect="auto", cmap="RdYlGn")

plt.title("Regime Heatmap (% of Days per Month) — bull / bear / sideways (Synthetic)")
plt.xlabel("Regime")
plt.ylabel("Month")

plt.xticks([0, 1, 2], ["bull", "bear", "sideways"])
plt.yticks(np.arange(len(pct.index)), pct.index)

# annotate cells
for i in range(data.shape[0]):
    for j in range(data.shape[1]):
        plt.text(j, i, f"{data[i, j]:.0f}%", ha="center", va="center", fontsize=8)

plt.colorbar(im, fraction=0.02, pad=0.02, label="% of days")
plt.tight_layout()
plt.savefig(output_dir / "regime_heatmap.png", dpi=150)
plt.close()

print("✓ Generated:")
print(f"  - {output_dir / 'drawdown_comparison.png'}")
print(f"  - {output_dir / 'regime_heatmap.png'}")
