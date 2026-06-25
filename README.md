# cement_fineness_opt

Standalone, deployment-oriented pipeline for cement-fineness APC:

```
data_preprocess.py  →  model_train.py  →  apc_predict_optimize.py
   (merge + power band)   (score + freeze weight)   (predict + optimize sep_motor_speed)
```

This repo is a **slim, pinned** extraction from the research repo `CementFinenessPrediction`. The four
shared libraries (`config`, `utils`, `opt_algo`, `feat_eng`) are **vendored** here, not imported from
the research tree, so this repo is self-contained and safe to port to the Ignition/Jython runtime.

## Setup (conda env `CBM`)

```bash
pip install -e . --no-deps     # registers config/utils/opt_algo/feat_eng as importable packages
```

`--no-deps` so it does not touch the env's existing packages. Runtime dependencies (already present in
`CBM`): `pandas`, `numpy`, `scikit-learn`, `xgboost`, `lightgbm`, `catboost`, `optuna`, `joblib`,
`mlflow`.

## Usage (canonical CM08 / psd_r30 / PCC, backward-fill 1h)

```bash
# 1) Preprocess: raw sensor + quality → merged q75 bf1h CSV + power band (writes to data/)
python scripts/data_preprocess.py --cm CM08

# 2) Train: nested-CV score (Phase A) + refit on full data → frozen weight (Phase B)
python scripts/model_train.py --cm CM08_bf1h --target psd_r30 --brand PCC --optuna-trials 25

# 3) Predict + optimize against the frozen artifact
python scripts/apc_predict_optimize.py --pot CM08_bf1h --desired-psd-r30 15.0
```

Note the pot names: `data_preprocess --cm CM08` is the *physical mill*; `model_train`/`apc` use the
**`CM08_bf1h` pot** (the bf1h recipe variant in `config/pot_config.py`, which globs the bf1h CSV).

### Outputs
- `data/…_q75_bf1h_CM08.csv`, `data/power_band_CM08.json` (+ `data/archive/<stamp>/` history).
- `results/CM08_bf1h/optimization/frozen_psd_r30_PCC/{model.joblib, metadata.json, cv_report_*.csv}`
  (+ `runs/<stamp>/` history). `metadata.json` is the train→predict contract the APC script reads.

## MLflow

`model_train.py` logs the nested-CV folds + final summary to a repo-local SQLite store. Browse:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db    # from the repo root
```

Experiment name: `<cm>_apc_training` (e.g. `CM08_bf1h_apc_training`). Disable with `--no-mlflow`.

## Data (standalone)

Raw inputs live under `data/original/` (copied in, not symlinked). For CM08 that is
`data/original/process_parameter_CM08_20240707-20260331/` (the sensor files + the
`SKK - CM8 Quality (2024-2026).xlsx`). To enable other mills, copy their raw dirs from the research
repo. `data/` and `results/` are gitignored except `.gitkeep`.

## Vendored / pinned libraries

`config`, `utils`, `opt_algo`, `feat_eng` are copies of the research repo's versions at extraction
time, with only the path anchors retargeted (`pot_config.py` → `DATA_DIR`/`RESULTS_DIR` at this repo
root; `experiment_tracker.py` → `mlflow.db`/`mlruns` at this repo root). A deployment repo **should**
pin these rather than track every research-side edit. **Re-sync only when intentionally promoting a
vetted change** — re-copy the file(s) from `CementFinenessPrediction/notebooks/{pkg}/` and re-apply the
two path-anchor edits. `opt_algo` here ships only the deterministic grid-search optimizer (the
Optuna/NSGA variants were omitted to stay dependency-light).

## Deployment caveats (carried over)

- **Forward-holdout gate (TODO):** `model_train.py` reports only the shuffled-IID nested-CV MAE. A
  most-recent-window (forward) holdout as a deployment acceptance gate is **not** yet implemented —
  production predicts forward in time, so an IID score is not by itself proof of forward performance.
- **MAE target is blaine-specific:** the ≈20 bar is for blaine; `psd_r30` is a different scale and
  needs its own acceptance threshold.
