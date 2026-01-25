## Objective

Ship a **public GitHub repo** (`bitcoin-timeseries-ml-engineering`) that **proves ML engineering skill** while **explicitly protecting** your private trading data + alpha.

## Decision/Constraint

**Allowed:** publish architecture, training pipeline, metrics framework, reproducibility, and a “bring-your-own-data” workflow.
**Blocked:** pushing your `data/`, private `outputs/`, full rulebooks, and proprietary QFL-DCA edge.

## Next step

Replace your current `README.md` + `AGENT.md` with the hardened versions below (copy/paste), then create the hardening branch and do the pre-push checklist.

---

## Measurement

Success = repo passes these checks:

* README makes the boundary unmissable (“no alpha/data included”), and shows how to run with user-owned data.
* AGENT.md gives deterministic steps: hardening branch → sanitize → smoke test → release.
* No private artifacts are tracked by git (models/data/outputs are ignored by default).

## Review trigger

Re-check after **8 checklist items completed** or after **first dry-run “public build” succeeds** (train/evaluate on a small public sample dataset or synthetic demo).

---

# 1) New README.md (repo = `bitcoin-timeseries-ml-engineering`)

```md
# bitcoin-timeseries-ml-engineering
## Bitcoin Time-Series ML Engineering (LSTM + Walk-Forward Validation) — portfolio-grade pipeline with private alpha redaction

> **Portfolio project for the IBM AI Engineering Certificate**  
> This repo demonstrates end-to-end ML engineering for financial time-series (data processing → model training → evaluation → reproducible runs).  
> **Important boundary:** my proprietary trading datasets, QFL-DCA rules, and “alpha discoveries” are intentionally **not** published.

---

## What this repo is
A **production-style ML pipeline** for Bitcoin forecasting experiments using:
- **Time-series-safe splits** (chronological + walk-forward evaluation)
- **Leakage-resistant scaling** (fit on train only)
- **Model architectures** (LSTM variants + optional attention/CNN)
- **Evaluation** (standard ML metrics + trading-aware metrics)

This is designed to be **auditable, reproducible, and recruiter-readable**.

---

## What this repo is NOT
- Not a “copy-paste profitable strategy”
- Not a data dump of my historical trades
- Not a release of my full QFL-DCA alpha rulebook
- Not financial advice

If you want to reproduce results, you must use **your own datasets** (or the optional public sample described below).

---

## Privacy / Alpha Redaction Policy (Explicit)
To protect years of research and private trading data, the following are **excluded from this public repo**:
- Private `data/` (raw, processed, or labeled datasets)
- Private `outputs/` (full experiment result tables, matched trades, correlations)
- Certain strategy-specific configurations / thresholds
- Full QFL-DCA optimization documents or proprietary rule sets

What is included instead:
- The **pipeline** and **engineering rigor**
- Config-driven training/evaluation
- A clear “Bring Your Own Data” interface
- Optional **synthetic or public sample** dataset support (recommended)

---

## Repo Structure (public-safe)
```

.
├── src/
│   ├── data/            # feature engineering + processor (train-only scaling)
│   ├── models/          # LSTM architectures
│   ├── training/        # trainer + metrics + walk-forward
│   └── utils/           # inference utilities
├── tests/               # unit tests (pipeline sanity checks)
├── notebooks/           # exploration / colab (sanitized)
├── configs/             # PUBLIC configs (no private thresholds)
├── docs/                # methodology notes (portfolio narrative)
├── scripts/             # helper scripts (sanitized)
├── main.py              # CLI entry point (train/evaluate)
├── requirements.txt
├── README.md
└── AGENT.md             # contributor/agent runbook (hardening + release flow)

````

---

## Engineering Rigor Highlights (what recruiters should notice)
- **Reproducibility:** seed control + config-driven runs (`config.yaml`)  
  (see seed handling in `main.py`)  
- **Leakage control:** scaling is fit on train split only (zero-leakage design)
- **Time-series validation:** walk-forward cross-validation (no random k-fold)
- **Checkpointing:** best model is saved and reloaded for evaluation
- **Metrics:** ML metrics + trading-aware metrics (Sharpe, drawdown, directional accuracy)

---

## Quick Start (Bring Your Own Data)
### 1) Install
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
````

### 2) Provide your BTC data

You provide a CSV at minimum containing OHLCV daily bars.

Expected location (default):

* `data/raw/btc_ohlcv_daily.csv` (not committed)

Optional:

* `data/raw/btc_onchain.csv` (not committed)

> If you don’t want to use private data, use the optional “public sample mode” once you add it (see below).

### 3) Train / Evaluate

```bash
python main.py --mode train --config config.yaml
python main.py --mode evaluate --config config.yaml
```

Outputs (local only, not committed):

* `models/best_model.pt`
* `models/scalers.joblib`
* `outputs/metrics.json` / `outputs/metrics.txt`
* `outputs/test_predictions.csv`

---

## Public Sample Mode (Recommended)

To keep this repo reproducible without exposing private data:

* Add a **small public dataset** (or generate synthetic data) under:

  * `data/sample/` (committed)
* Keep real datasets in:

  * `data/raw/` (gitignored)

This keeps the project runnable for reviewers while protecting your edge.

---

## Results (How to interpret)

This repo focuses on **engineering quality** and correct evaluation for noisy markets.

Important:

* Any “paper numbers” (e.g., R²=0.991) are **literature-reported benchmarks**, not guaranteed reproduction.
* Your actual results depend heavily on data window, feature set, regime shifts, and cost assumptions.

---

## Safety / Compliance

This is educational and for portfolio demonstration only.
No financial advice. Use at your own risk.

---

## License

Choose a permissive license for portfolio visibility (MIT/Apache-2.0).
If you want to restrict use, choose a more protective license. (Decide before first push.)

---

## Contact

If you’re a recruiter/hiring manager: this repo is intended to show ML engineering skill in time-series forecasting + evaluation discipline.

````

---

# 2) New AGENT.md (hardening + release runbook)

```md
# AGENT.md — bitcoin-timeseries-ml-engineering
Instructions for contributors / AI agents to harden, sanitize, and ship this repo publicly **without leaking private alpha**.

---

## Prime Directive (Non-Negotiable)
This public repo must **demonstrate engineering** without exposing:
- Private trading data
- Strategy thresholds / proprietary alpha rules
- Full backtest tables and matched trades
- Private model weights trained on private datasets

If unsure, **do not publish** the artifact.

---

## Branch Strategy
### Create hardening branch (before first push)
```bash
git checkout -b portfolio-hardening
````

All changes below happen in `portfolio-hardening` first.

---

## Portfolio Hardening Checklist (8 items)

Complete these **before first push**:

1. **Sanitize tracked files**

* Remove or untrack any private datasets, outputs, trade logs, or full result tables.
* Confirm `git status` and `git ls-files` show no private artifacts.

2. **Lock down .gitignore**
   Must ignore at minimum:

* `data/raw/`, `data/processed/`
* `outputs/`
* `models/*.pt`, `models/*.joblib` (unless you intentionally ship a toy model)
* `.venv/`, `__pycache__/`, `.pytest_cache/`

3. **Public-safe configs**

* Provide `configs/config.public.yaml` that runs without private thresholds.
* If needed, add `configs/config.private.yaml` to `.gitignore`.

4. **Public sample dataset or synthetic demo**

* Add `data/sample/` (small, public-safe) OR synthetic generator script:

  * `scripts/make_synthetic_btc_sample.py`
* Ensure the pipeline runs end-to-end using sample mode.

5. **Reproducible CLI + deterministic seed**

* Ensure `main.py` sets seeds and logs config clearly.
* Keep default run short (quick mode) for reviewers.

6. **Tests + CI-ready**

* `pytest` passes locally.
* Add minimal tests that validate:

  * data processor fit-on-train-only
  * model forward pass
  * trainer can run 1 epoch without NaNs

7. **Docs that explain boundaries**

* README contains explicit “alpha redaction” policy and BYO-data instructions.
* Add `docs/SECURITY_AND_PRIVACY.md` if needed.

8. **Release hygiene**

* Add `LICENSE`
* Add `CHANGELOG.md` (optional)
* Tag v0.1.0 after merge
* Confirm no secrets:

  * API keys, tokens, Glassnode keys, etc.

---

## Verification Commands (run these before push)

```bash
# 1) Ensure no private artifacts are tracked
git ls-files | rg "(data/raw|data/processed|outputs|full_matched_trades|.*\.pt$|.*\.joblib$)" || true

# 2) Confirm ignore works (should show ignored files)
git status --ignored

# 3) Run tests
pytest -v

# 4) Quick pipeline smoke test (should finish fast)
python main.py --mode train --config configs/config.public.yaml
python main.py --mode evaluate --config configs/config.public.yaml
```

---

## Public vs Private Data Contract

**Public repo contains:**

* pipeline code, training loop, metrics, docs, sample data/synthetic mode

**Private (never commit):**

* full datasets, trade logs, strategy rules, optimized thresholds, private outputs

---

## “If it feels like alpha, it is alpha”

When in doubt:

* replace with a **placeholder**
* publish a **toy example**
* or describe at a high level without revealing parameters

---

## Done Definition

This repo is ready to push when:

* Sample mode runs end-to-end
* Tests pass
* README boundary is explicit
* `git ls-files` shows no private artifacts

```

---

## What you *don’t* need to send me (you already did)
You’ve already shared the core runtime pieces (`main.py`, `config.yaml`, model/training modules). The uncertainty you’re feeling is normal because your folder has mixed “research outputs” + “portfolio code.”

**Your next irreversible move** is not sharing more files — it’s enforcing the boundary in git (ignore + sanitize) and making the README/AGENT match that boundary.

If you want, paste your current `.gitignore` and I’ll tighten it to guarantee you can’t accidentally push private artifacts.
::contentReference[oaicite:0]{index=0}
```
------

## Objective

Update the repo narrative so recruiters see **credible local training on your Intel iMac** (no Colab), with **repeatable CPU-safe runs**.

## Decision/Constraint

**Allowed:** publish your hardware + “local-only training” claim + CPU-first runtime limits.
**Blocked:** implying GPU/Colab performance, or publishing private datasets/alpha to “prove” it.

## Next step

Add this “Local Training” block to your README (drop-in), and add a matching “Runtime/Compute Contract” block to AGENT.md.

---

## Measurement

Success = README clearly states **training was done locally** on your exact machine, and gives **expected run constraints** (epochs/batch/sequence length) that fit CPU.

## Review trigger

Re-check after you run **one full public-safe smoke test** (train+evaluate) on your Mac with `configs/config.public.yaml`.

---

# README.md — drop-in section (paste under “Engineering Rigor Highlights”)

```md
## Local-Only Training (No Colab)
All model training and experiments in this project were executed locally on my machine (CPU-first).  
This repo is intentionally designed to be **reproducible on a standard developer workstation**.

**Training hardware (local):**
- Machine: iMac (iMac19,1)
- CPU: 6-Core Intel Core i5 @ 3.0 GHz
- RAM: 32 GB
- OS: macOS (local execution)

**Design implication:** defaults favor **CPU-safe** settings (short runs, deterministic seeds, minimal dependencies).  
For longer training, adjust epochs/sequence length/batch size in config.
```

---

# AGENT.md — drop-in section (paste near “Prime Directive”)

```md
## Compute Contract (CPU-First)
This repository must remain runnable on a CPU-only workstation.

Reference training environment (author machine):
- iMac19,1 — 6-Core Intel i5 @ 3.0 GHz, 32 GB RAM
- Local-only training (no Colab, no cloud dependence)

Rules:
- Default configs must complete a smoke run quickly on CPU.
- Avoid heavy dependencies and long downloads.
- Prefer small sample/synthetic data for public runs.
- Deterministic runs: fixed seeds; log config + versions.
```

---

## Optional (high-value, low-risk) polish

Add a single-line badge-style callout near the top of README:

```md
**Execution:** Local CPU-only ✅  |  **Data:** BYO-data ✅  |  **Private Alpha:** Redacted ✅
```

----


## Objective

Ship a **public GitHub-ready** repo (`bitcoin-timeseries-ml-engineering`) that proves ML engineering skill **without leaking your private alpha/data**.

## Decision/Constraint

**Allowed:** open-source the architecture, training pipeline, evaluation rigor, reproducibility scaffolding, and a “public demo” path using tiny synthetic/sample data.
**Blocked:** publishing your full QFL-DCA findings, proprietary features, raw trade logs, or any dataset that recreates your edge.

## Next step

Copy/paste the two files below into your project root as **`README.md`** and **`AGENT.md`** (overwrite your current versions). They explicitly enforce the boundary + create the “Portfolio Hardening” workflow before first push.

---

# README.md (rewrite)

````md
# bitcoin-timeseries-ml-engineering
**Bitcoin time-series ML engineering portfolio repo — reproducible LSTM forecasting + evaluation harness (public-safe).**

> This repository is intentionally “portfolio-hardened”: it shows *how* I build and validate time-series ML systems on a local CPU machine, while **excluding proprietary datasets, private trading logs, and alpha-revealing details**.

---

## What this repo is
A compact, professional ML engineering project focused on:
- **Time-series forecasting** for Bitcoin (LSTM / BiLSTM / Attention variants)
- **Leakage-safe data processing** (chronological split, fit scalers on train only)
- **Training discipline** (early stopping, LR scheduling, checkpointing)
- **Evaluation rigor** including ML + trading-aware metrics (directional accuracy, Sharpe, drawdown, profit factor)
- **Local-first execution** (trained on Intel iMac CPU, 32GB RAM)

## What this repo is NOT
To protect years of private work, this repo does **not** include:
- My private QFL-DCA research notebooks, edge logic, or parameterized alpha rules
- Full datasets / trade logs / exchange keys / webhooks / automation configs
- Any “copy-paste profitable strategy” implementation

If you’re a recruiter/hiring manager and want to validate deeper results, I can provide **screenshare evidence** and a **redacted evaluation report**.

---

## Quickstart (public-safe)
### 1) Create venv + install
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
````

### 2) Run a smoke train/eval (small + deterministic)

This uses minimal settings and should run on CPU.

```bash
python main.py --mode train --config config.public.yaml
python main.py --mode evaluate --config config.public.yaml
```

---

## Reproducibility & rigor checklist

This project emphasizes *engineering proof*, not hype:

* ✅ Chronological train/val/test split (no shuffling across time)
* ✅ Standardization fit on train only (prevents leakage)
* ✅ Early stopping + best-checkpoint restore
* ✅ ReduceLROnPlateau scheduler
* ✅ Deterministic seeds
* ✅ Metrics include trading-aware evaluation (cost-aware Sharpe, drawdown, profit factor)

> Note: if you see `shuffle=True` inside the training DataLoader, it only shuffles **within the already-trimmed training window** (sequence order is preserved by the dataset construction). If you prefer strict sequential batching, set shuffle to false.

---

## Project structure (public)

```
.
├── README.md
├── AGENT.md
├── main.py
├── config.yaml                # full research config (private-oriented; do not publish raw data paths)
├── config.public.yaml         # portfolio-safe config for public runs
├── requirements.txt
└── src/
    ├── models/
    │   └── lstm.py
    ├── training/
    │   ├── trainer.py
    │   └── metrics.py
    ├── data/
    │   └── processor.py
    └── utils/
        └── inference.py
```

---

## Hardware used (local training)

* iMac19,1 — 6-Core Intel i5 3.0GHz
* 32GB RAM
* macOS Darwin 24.6.0
* Training executed **locally** (no Colab)

---

## Results policy (important)

This repo avoids publishing “marketing numbers” unless they are:

1. reproducible from the public-safe pipeline, and
2. generated from a dataset that is publishable.

If you see performance claims elsewhere in the repo history, treat them as **research notes** unless backed by a reproducible run + artifacts.

---

## License

MIT (or your preferred license). See `LICENSE`.

---

## Contact

If you want a recruiter-ready walkthrough:

* I can demo: training run, metrics, and evaluation flow live.
* I can share: redacted artifacts and performance summaries under NDA.

````

---

# AGENT.md (new “Portfolio Hardening” operator file)

```md
# AGENT.md — Portfolio Hardening Operator (first push gate)

This repo is a **public portfolio**. Goal: show ML engineering maturity while protecting private alpha/data.

## Non-negotiables (repo rules)
- Python 3.11+ and `venv` (.venv). No conda.
- Minimal dependencies; prefer stdlib.
- Local-first; CPU-safe; deterministic seeds.
- Every runnable step must be scriptable (`scripts/*.sh`) and documented in README.
- Do not publish private datasets, trade logs, API keys, or alpha rules.

---

# One-time workflow: create Portfolio Hardening branch
Run these commands from repo root:

```bash
git checkout -b portfolio-hardening
````

---

# The 8 items to implement BEFORE first push

## 1) Sanitize what’s public

**Goal:** no proprietary data or alpha leakage.

* Add/verify `.gitignore` excludes:

  * `data/**` (raw, processed, caches)
  * `models/*.pt`, `models/*.joblib`, `outputs/**`, `logs/**`
  * `.env`, `*.key`, `*webhook*`, `*3commas*`, `*secrets*`
* Replace any “private findings” docs with short redacted summaries if needed.

## 2) Add `config.public.yaml` (public-safe)

Create a smaller config for smoke runs (fast CPU).

* Smaller model
* Fewer epochs
* Smaller batch size
* Deterministic seed
* No external paid APIs
* Uses cached BTC CSV if present; otherwise fetches BTC-USD via fallback

**Template to create:**

```yaml
seed: 42
device: cpu

paths:
  data_dir: data
  models_dir: models
  outputs_dir: outputs
  logs_dir: logs

data:
  ticker: BTC-USD
  start_date: "2020-01-01"
  end_date: null
  train_ratio: 0.70
  val_ratio: 0.15
  test_ratio: 0.15

features:
  sequence_length: 60

model:
  type: lstm_attention
  input_size: null
  hidden_size: 64
  num_layers: 2
  dropout: 0.2
  bidirectional: true
  use_attention: true
  attention_size: 64
  fc_layers: [64, 32]
  output_size: 1

training:
  batch_size: 64
  epochs: 10
  learning_rate: 0.001
  weight_decay: 0.0001
  optimizer: adam
  loss: mse
  gradient_clip: 1.0
  scheduler:
    type: reduce_on_plateau
    patience: 3
    factor: 0.5
    min_lr: 0.0001
  early_stopping:
    patience: 5
    min_delta: 0.0001
```

## 3) Add scripts (reproducible operator commands)

Create:

* `scripts/bootstrap.sh` (venv + install)
* `scripts/smoke_train.sh` (train with config.public.yaml)
* `scripts/smoke_eval.sh` (evaluate with config.public.yaml)
* `scripts/lint.sh` (ruff)
* `scripts/test.sh` (pytest)

## 4) Add minimal CI

Add GitHub Actions workflow:

* install deps
* run `scripts/lint.sh`
* run `scripts/test.sh`
* optional: run smoke train with tiny epochs (keep under a few minutes)

## 5) Add tests for “rigor surfaces”

Add/confirm tests that assert:

* chronological split sizes sum to full length
* scalers are fit only on train (no leakage)
* TimeSeriesDataset sequence shapes correct
* metrics functions return finite values

## 6) Add “Public vs Private” boundary docs

Create `docs/BOUNDARY.md`:

* what’s included publicly
* what’s withheld and why
* how to request verification (screenshare/NDA)

## 7) Add release-grade repo hygiene

* `LICENSE`
* `SECURITY.md` (no secrets, reporting guidance)
* `CONTRIBUTING.md` (lightweight)
* badges + version pin guidance

## 8) Final gate: no secrets + clean tree

Before push:

```bash
git status
git grep -n "api_key\|secret\|webhook\|3commas\|PRIVATE\|glassnode" .
python -m compileall src
```

Then:

```bash
git add -A
git commit -m "Portfolio hardening: public-safe training pipeline + boundaries"
git push -u origin portfolio-hardening
```

---

# Agent operating rules (when assisting in this repo)

* Prefer edits that improve reproducibility and professionalism.
* Do not add new “learning content” unless a weekly gate passes.
* If asked to include private alpha: refuse and propose redacted alternative.

```

---

## Measurement
Success = you can run these two commands on a clean machine and get a complete output without leaking private assets:
- `python main.py --mode train --config config.public.yaml`
- `python main.py --mode evaluate --config config.public.yaml`

Failure = any of:
- repo contains raw data/trade logs/model checkpoints
- README makes non-reproducible performance claims
- CI fails or smoke run is not deterministic

## Review trigger
Re-review after **(1)** your first successful smoke run + CI pass, or **(2)** any time you add a new dataset/feature source.

---

## Objective

Ship a **GitHub-ready public repo** (professional + recruiter-friendly) from `/Users/FV/Dev.Ops/IBM-GEN-AI-CERT/Github-ready`, with **private alpha/data protected**, and a single **Portfolio Hardening** branch that must pass gates before first push.

## Decision/Constraint

**Allowed:** Publish engineering rigor, architecture, reproducible pipeline, synthetic sample data, evaluation harness, and redacted examples.
**Blocked:** Publishing proprietary QFL-DCA “alpha”, private datasets, keys, full signal formulas, or anything that recreates your edge.

 
---

## 5) “Portfolio Hardening” — 8 Required Items (must complete before first push)

### (1) Repo Sanitization & Secrets Safety

* Add/verify `.gitignore` excludes: `.env`, `private/`, `data/raw/`, `models/*.pt`, `*.joblib`, `**/trading_logs*`
* Add `.env.example` (no real keys)
* Run a local “redaction check” script before committing

### (2) Public Narrative (Recruiter-Readable)

* README positioned as **ML engineering** + **time-series rigor** + **Bitcoin domain**
* Clear “What’s public vs private” section
* Clear “Reproduce demo in <10 minutes on CPU” instructions

### (3) Deterministic Demo Dataset

* Add `data/sample/` with synthetic OHLCV (tiny)
* Provide `scripts/make_synth_data.sh` to regenerate deterministically

### (4) Reproducible CPU Training Demo

* `scripts/train_demo.sh` trains a tiny model on synthetic data in minutes
* Fixed random seeds and small epochs
* Outputs metrics to `artifacts/demo_metrics.json` (ignored or committed if non-sensitive)

### (5) Evaluation Harness (Time-Series Safe)

* Walk-forward or temporal split only (no leakage)
* Confirm scaling is fit on train only (processor already enforces this)
* Provide `scripts/eval_demo.sh`

### (6) Minimal Tests (Smoke + Metrics)

* `pytest` includes:

  * metric correctness sanity checks
  * data processor no-leakage behavior
  * training loop runs on tiny synthetic data

### (7) Quality Gates

* `scripts/run_gates.sh` runs:

  * `python -m compileall src`
  * `ruff check .`
  * `pytest -q`
  * quick demo train/eval (optional but preferred)

### (8) CI (Optional but recommended)

* Add GitHub Actions workflow to run `run_gates.sh` on PRs and pushes to main

---

## 6) Mandatory Scripts (create if missing)

### scripts/setup_venv.sh

* Creates `.venv`, installs deps, prints python version.

### scripts/run_gates.sh

* The single command recruiters can run:

```bash
./scripts/run_gates.sh
```

### scripts/redact_check.sh

* Fails if forbidden paths/files are staged or present in tracked files:

  * `private/`
  * `*.pt`, `*.joblib` (unless explicitly allowed)
  * `.env`
  * large datasets

---

## 7) Release Procedure (First Push)

1. Move curated public content into:
   `/Users/FV/Dev.Ops/IBM-GEN-AI-CERT/Github-ready/bitcoin-timeseries-ml-engineering`
2. Initialize git + hardening branch
3. Run gates locally
4. Create empty repo on GitHub (public)
5. Add remote + push hardening branch
6. Open PR: `portfolio-hardening` → `main`
7. Merge only if CI green (if CI is enabled)

---

## 8) Notes for Recruiters (Intended Read Path)

* Start: README “10-minute demo”
* Then: `src/data/processor.py` (no leakage)
* Then: `src/training/trainer.py` (walk-forward validation)
* Then: `src/training/metrics.py` (financial metrics)
* Then: `src/models/lstm.py` (architectures)

---

## 9) Definition of Done

* `./scripts/run_gates.sh` passes on CPU
* README is clear and honest about what is private
* No sensitive files are tracked by git
* Repo looks intentional: clean top-level, no research sprawl

````

---

## Steps to create the repo on your GitHub (FabioVinelli)

### Option A — GitHub Web UI (fast, visual)
1. Go to your GitHub profile and click the **“+”** menu → **New repository**. :contentReference[oaicite:0]{index=0}  
2. Repository name: **`bitcoin-timeseries-ml-engineering`**  
3. Visibility: **Public** (recommended for recruiters)  
4. Initialize with README: **No** (you already have one locally)  
5. Click **Create repository**. :contentReference[oaicite:1]{index=1}  

### Option B — GitHub CLI (cleanest for dev workflow)
If you have `gh` installed, you can create directly from your local folder using `gh repo create`. :contentReference[oaicite:2]{index=2}  
Example flow (from inside your GitHub-ready repo folder):
```bash
cd "/Users/FV/Dev.Ops/IBM-GEN-AI-CERT/Github-ready/bitcoin-timeseries-ml-engineering"

git init
git checkout -b portfolio-hardening

git add -A
git commit -m "Portfolio hardening: initial public release scaffold"

# creates repo under your account and sets remote
gh repo create FabioVinelli/bitcoin-timeseries-ml-engineering --public --source=. --remote=origin --push
````

---

## Measurement

Success =

* `AGENT.md` exists in your Github-ready repo, and
* you can run `./scripts/run_gates.sh` locally with **all green**, and
* `git status` shows **no private artifacts tracked**, and
* repo is created under `FabioVinelli/bitcoin-timeseries-ml-engineering` with first push from `portfolio-hardening`.

## Review trigger

Trigger a review **after 1 full gate pass** OR **after first push** (whichever happens first).

```

