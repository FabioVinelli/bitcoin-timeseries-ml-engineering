# AGENT.md — Portfolio Hardening Orchestrator (Public GitHub Release)

## 0) Mission (what this repo is)
This repository is a **public, recruiter-facing** Bitcoin time-series ML engineering portfolio project.
It demonstrates:
- clean architecture and reproducibility
- time-series-safe training + evaluation
- disciplined engineering workflow (tests, lint, gates)
- practical trading-oriented evaluation metrics **without exposing private alpha**

It intentionally **does not** publish proprietary signals, datasets, or full strategy logic.

---

## 1) Non-Negotiables (Repo Rules)
- Python **3.11+** with **venv (.venv)**. No conda.
- Minimal dependencies; prefer stdlib; avoid cloud-only assumptions.
- Every runnable step must be scriptable (`scripts/*.sh`) and documented in README.
- Weekly gates are mandatory: **no new content** until the current gate passes.
- Prefer deterministic runs: fixed seeds, tiny **synthetic** sample data, no downloads.
- Keep commands fast enough for CPU laptop/desktop (Intel iMac OK).

Quality:
- Run format/lint/tests after changes (**ruff/pytest**).
- Add/adjust tests when behavior changes.
- Keep modules importable via `python -m src...`.

---

## 2) Public vs Private Boundary (Hard Rule)
### Public (OK to publish)
- Model/training code, data processing code, evaluation metrics harness
- Synthetic data generation + small public sample dataset
- Architecture docs, methodology, limitations, reproducibility steps
- Screenshots of dashboards (optional), but no private endpoints/keys

### Private (DO NOT publish)
- Private trading logs, 3Commas exports, proprietary labels, QFL-DCA parameterizations
- “Alpha guides” and full signal/threshold tables
- Real feature matrices derived from private sources
- `.pt` trained weights if they encode proprietary data patterns (default: keep private)
- Any API keys / credentials / tokens

**Enforcement:** anything private must be excluded by `.gitignore` AND placed under `private/`.

---

## 3) Target Public Directory Layout (GitHub)
You will publish only this structure:

.
├─ README.md
├─ LICENSE
├─ AGENT.md
├─ CHANGELOG.md
├─ .gitignore
├─ .env.example
├─ requirements.txt (or pyproject.toml)
├─ scripts/
│  ├─ setup_venv.sh
│  ├─ run_gates.sh
│  ├─ train_demo.sh
│  ├─ eval_demo.sh
│  ├─ make_synth_data.sh
│  └─ redact_check.sh
├─ src/
│  ├─ data/        (processor + feature pipeline)
│  ├─ models/      (LSTM variants)
│  ├─ training/    (trainer + metrics)
│  ├─ utils/       (inference utilities)
│  └─ __init__.py
├─ tests/
│  ├─ test_data_processor.py
│  ├─ test_training_loop_smoke.py
│  └─ test_metrics.py
└─ data/
   └─ sample/
      ├─ synth_btc_ohlcv.csv
      └─ README_DATA.md

### Private (local-only, never committed)
private/
├─ qfl_dca_alpha/
├─ real_data/
├─ trading_logs/
├─ model_weights/
└─ notes/

---

## 4) Portfolio Hardening Branch — One Way In
### Branch policy
- Create branch: `portfolio-hardening`
- All changes for the first public push happen ONLY on this branch
- Merge to `main` only after gates pass

Command:
```bash
git checkout -b portfolio-hardening
