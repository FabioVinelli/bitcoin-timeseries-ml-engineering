#!/usr/bin/env python3
"""
Generate Model vs Strategy comparison chart (synthetic-safe for public repo).
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ---------- Synthetic setup (safe for public repos) ----------
np.random.seed(42)
dates = pd.date_range("2024-01-01", periods=500, freq="D")

# Simulated training history
epochs = np.arange(1, 31)
train_loss = np.exp(-epochs/10) + 0.05*np.random.rand(len(epochs))
val_loss   = np.exp(-epochs/9)  + 0.08*np.random.rand(len(epochs))

# Simulated equity curves (normalized)
bh = 1 + np.cumsum(np.random.normal(0.0015, 0.02, len(dates)))
ml = 1 + np.cumsum(np.random.normal(0.0009, 0.015, len(dates)))
qfl = 1 + np.cumsum(np.random.normal(0.0011, 0.01, len(dates)))

# ---------- Plot ----------
plt.figure(figsize=(14, 8))

# (1) Training vs Validation loss
ax1 = plt.subplot(2, 1, 1)
ax1.plot(epochs, train_loss, label="Train Loss", color="#1f77b4")
ax1.plot(epochs, val_loss, label="Val Loss", color="#ff7f0e")
ax1.set_title("Model Training Dynamics (Synthetic)")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Loss")
ax1.legend()
ax1.grid(alpha=0.3)

# (2) Strategy outcomes
ax2 = plt.subplot(2, 1, 2)
ax2.plot(dates, bh / bh[0], label="Buy & Hold", color="gray")
ax2.plot(dates, ml / ml[0], label="ML-driven Strategy", color="#1f77b4")
ax2.plot(dates, qfl / qfl[0], label="QFL-DCA (Benchmark)", color="green")
ax2.set_title("Strategy Outcomes (Normalized Equity, Synthetic)")
ax2.set_xlabel("Date")
ax2.set_ylabel("Equity (Normalized)")
ax2.legend()
ax2.grid(alpha=0.3)

plt.tight_layout()

# Ensure output directory exists
output_dir = Path("docs/images")
output_dir.mkdir(parents=True, exist_ok=True)

plt.savefig(output_dir / "model_vs_strategy.png", dpi=150)
plt.close()

print(f"✓ Generated: {output_dir / 'model_vs_strategy.png'}")
