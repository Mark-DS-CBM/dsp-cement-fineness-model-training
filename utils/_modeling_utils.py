"""Shared utilities for the modelling notebooks (01_2, 02_1) + model_train.py.

Two evaluation schemes are supported via `run_nested_cv(split_mode=..., inner_split=...)`:

- **Forward holdout** (`split_mode="holdout"`, `inner_split="ts"`) — the current
  forward-evaluation gate. Time-sort the brand rows, hold out the most-recent
  `test_frac` as the test set, tune hyperparameters on the earlier train span via inner
  `TimeSeriesSplit`, refit on the train span, score once. Production predicts forward in
  time, so this is the honest deployment estimate. It is *not* nested (one train/test
  split, not an outer loop).

- **Nested IID CV** (`split_mode="iid"`, `inner_split="kfold"`, the legacy defaults) —
  outer shuffled KFold = honest estimate (every row is a val row once); inner shuffled
  KFold inside each outer-train slice = Optuna tuning. Each outer fold gets its own best
  hyperparameters. Retained as the baseline for comparison.

02_1 does NOT consume hyperparameters from 01_2; it re-tunes inside its own evaluation
so the FE comparison is end-to-end consistent with whatever feature set it produces.

Encoding / scaling is fit on the train slice and reused for the val slice plus the inner
CV — same convention as the legacy single-split pipeline.

Models supported: KNN, XGBoost, LightGBM, CatBoost. AutoGluon is intentionally
excluded.
"""
from __future__ import annotations

import time
from typing import Any, Callable, Iterator

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    mean_absolute_percentage_error,
)
from sklearn.model_selection import KFold, TimeSeriesSplit, cross_val_score

import xgboost as xgb
import lightgbm as lgb
import catboost as cb
import optuna


VALID_MODELS = ("KNN", "XGBoost", "LightGBM", "CatBoost")


# ────────────────────────────────────────────────────────────────────────────
# Encoding & scaling
# ────────────────────────────────────────────────────────────────────────────

def encode_and_scale_features_train_test(X_train, X_test, num_cols, verbose=True):
    """Label-encode + one-hot encode `brand`, then standardize numeric columns.

    Scaler and encoders are fit on `X_train` only and applied to `X_test`. Returns
    both label-encoded (`le_sc`, for tree models) and one-hot (`oh_sc`, for KNN)
    versions in a dict.
    """
    if 'brand' in X_train.columns:
        le = LabelEncoder()
        le.fit(X_train['brand'])
        if verbose:
            print(f"Brand encoding: {dict(zip(le.classes_, le.transform(le.classes_)))}")

        X_train_le = X_train.copy()
        X_test_le = X_test.copy()
        X_train_le['brand'] = le.transform(X_train_le['brand'])
        X_test_le['brand'] = le.transform(X_test_le['brand'])

        X_train_oh = pd.get_dummies(X_train, columns=['brand'], drop_first=True, dtype=float)
        X_test_oh = pd.get_dummies(X_test, columns=['brand'], drop_first=True, dtype=float)
        for col in X_train_oh.columns:
            if col not in X_test_oh.columns:
                X_test_oh[col] = 0
        X_test_oh = X_test_oh[X_train_oh.columns]
    else:
        X_train_le = X_train.copy()
        X_test_le = X_test.copy()
        X_train_oh = X_train.copy()
        X_test_oh = X_test.copy()

    scaler = StandardScaler()
    scaler.fit(X_train_oh[num_cols])

    X_train_oh_sc = X_train_oh.copy()
    X_test_oh_sc = X_test_oh.copy()
    X_train_oh_sc[num_cols] = scaler.transform(X_train_oh[num_cols])
    X_test_oh_sc[num_cols] = scaler.transform(X_test_oh[num_cols])

    X_train_le_sc = X_train_le.copy()
    X_test_le_sc = X_test_le.copy()
    X_train_le_sc[num_cols] = scaler.transform(X_train_le[num_cols])
    X_test_le_sc[num_cols] = scaler.transform(X_test_le[num_cols])

    if verbose:
        print(f"One-hot encoded features: {X_train_oh_sc.columns.tolist()}")
        print(f"Shape after one-hot: {X_train_oh_sc.shape}")

    return {
        'oh_sc': (X_train_oh_sc, X_test_oh_sc),
        'le_sc': (X_train_le_sc, X_test_le_sc),
        'scaler': scaler,
    }


# ────────────────────────────────────────────────────────────────────────────
# Metric helpers
# ────────────────────────────────────────────────────────────────────────────

def evaluate_model(y_true, y_pred, set_name=""):
    """Regression metrics: MAE, RMSE, R², MAPE(%)."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred) * 100
    return {'Set': set_name, 'MAE': mae, 'RMSE': rmse, 'R2': r2, 'MAPE(%)': mape}


def evaluate_residuals(y_true, y_pred):
    """Residual statistics: Max/Min |Residual|, Mean Residual, SD Residual."""
    residuals = np.asarray(y_true) - np.asarray(y_pred)
    abs_residuals = np.abs(residuals)
    return {
        'Max |Residual|': abs_residuals.max(),
        'Min |Residual|': abs_residuals.min(),
        'Mean Residual': residuals.mean(),
        'SD Residual': residuals.std(),
    }


# ────────────────────────────────────────────────────────────────────────────
# Model factory
# ────────────────────────────────────────────────────────────────────────────

def make_model(model_name: str, params: dict[str, Any], random_state: int = 42):
    """Instantiate a regressor with shared infra params injected.

    `params` should be a hyperparameter dict (e.g., `study.best_params`).
    Infra params (random seed, n_jobs, verbosity) are added here so callers
    don't have to remember them per model.
    """
    p = dict(params)  # copy
    if model_name == "KNN":
        p.setdefault('n_jobs', -1)
        return KNeighborsRegressor(**p)
    if model_name == "XGBoost":
        p.update({'random_state': random_state, 'n_jobs': -1, 'tree_method': 'hist'})
        return xgb.XGBRegressor(**p)
    if model_name == "LightGBM":
        p.update({'random_state': random_state, 'n_jobs': -1, 'verbose': -1})
        return lgb.LGBMRegressor(**p)
    if model_name == "CatBoost":
        p.update({'random_seed': random_state, 'verbose': 0})
        return cb.CatBoostRegressor(**p)
    raise ValueError(f"Unknown model: {model_name!r}. Valid: {VALID_MODELS}")


# ────────────────────────────────────────────────────────────────────────────
# Optuna search spaces
# ────────────────────────────────────────────────────────────────────────────

def _suggest_knn(trial, cv_train_size: int):
    # `cv_train_size` = rows available in the *smallest* inner-CV train fold (computed
    # by make_objective per splitter), so n_neighbors never exceeds the train rows.
    max_neighbors = min(50, max(3, cv_train_size - 1))
    max_leaf_size = min(100, max(10, cv_train_size - 1))
    return {
        'n_neighbors': trial.suggest_int('n_neighbors', 3, max_neighbors),
        'weights': trial.suggest_categorical('weights', ['uniform', 'distance']),
        'metric': trial.suggest_categorical('metric', ['euclidean', 'manhattan', 'minkowski']),
        'leaf_size': trial.suggest_int('leaf_size', 10, max_leaf_size),
        'p': trial.suggest_int('p', 1, 3),
    }


def _suggest_xgb(trial, n_train: int):
    return {
        'n_estimators': trial.suggest_int('n_estimators', 100, 1500),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.3, log=True),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 20),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
        'gamma': trial.suggest_float('gamma', 1e-8, 5.0, log=True),
    }


def _suggest_lgb(trial, n_train: int):
    return {
        'n_estimators': trial.suggest_int('n_estimators', 100, 1500),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.3, log=True),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 50),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 20, 300),
        'min_split_gain': trial.suggest_float('min_split_gain', 1e-8, 1.0, log=True),
    }


def _suggest_cat(trial, n_train: int):
    return {
        'iterations': trial.suggest_int('iterations', 100, 1500),
        'depth': trial.suggest_int('depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.3, log=True),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-2, 10.0, log=True),
        'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 1.0),
        'random_strength': trial.suggest_float('random_strength', 1e-8, 10.0, log=True),
        'border_count': trial.suggest_int('border_count', 32, 255),
    }


_SUGGESTERS = {
    "KNN": _suggest_knn,
    "XGBoost": _suggest_xgb,
    "LightGBM": _suggest_lgb,
    "CatBoost": _suggest_cat,
}


def make_objective(model_name: str, X_train, y_train,
                   n_inner_folds: int, random_state: int,
                   inner_split: str = "kfold") -> Callable:
    """Build an Optuna objective that scores params by inner CV mean MAE.

    `inner_split` selects the inner splitter used for hyperparameter search:
    - ``"kfold"`` (default): shuffled ``KFold`` — IID tuning (legacy nested-CV path).
    - ``"ts"``: ``TimeSeriesSplit`` — chronological tuning on the (time-sorted) train
      span, used by the forward-holdout scheme. The forward-holdout has only this one
      TimeSeriesSplit level, so the inner-train slices stay usable even on short windows.
    """
    if model_name not in _SUGGESTERS:
        raise ValueError(f"Unknown model: {model_name!r}. Valid: {VALID_MODELS}")
    if inner_split not in ("kfold", "ts"):
        raise ValueError(f"Unknown inner_split: {inner_split!r}. Valid: 'kfold', 'ts'.")

    n_train = len(X_train)
    suggester = _SUGGESTERS[model_name]
    # Rows in the *smallest* inner-CV train fold (only KNN's neighbor cap uses this;
    # the tree suggesters ignore the size arg). TimeSeriesSplit's first fold trains on
    # ~n/(k+1); shuffled KFold trains on (k-1)/k of the rows.
    if inner_split == "ts":
        min_cv_train = max(1, n_train // (n_inner_folds + 1))
    else:
        min_cv_train = int(n_train * (n_inner_folds - 1) / n_inner_folds)

    def objective(trial):
        params = suggester(trial, min_cv_train)
        model = make_model(model_name, params, random_state)
        if inner_split == "ts":
            cv = TimeSeriesSplit(n_splits=n_inner_folds)
        else:
            cv = KFold(n_splits=n_inner_folds, shuffle=True, random_state=random_state)
        scores = cross_val_score(
            model, X_train, y_train, cv=cv,
            scoring='neg_mean_absolute_error', n_jobs=-1,
        )
        return -scores.mean()

    return objective


# ────────────────────────────────────────────────────────────────────────────
# Per-brand fold generator (handles IID + value-identity test source)
# ────────────────────────────────────────────────────────────────────────────

def make_brand_folds(
    df_current: pd.DataFrame,
    df_test_source: pd.DataFrame | None,
    brand,
    target: str,
    feature_cols: list[str],
    n_splits: int = 5,
    random_state: int = 42,
    split_mode: str = "iid",
    test_frac: float = 0.2,
) -> Iterator[tuple]:
    """Yield evaluation folds for one brand.

    Each yield: ``(X_train, X_val, y_train, y_val, hours_train, hours_val)``.

    `split_mode="iid"` (default) — shuffled `KFold` cross-validation:
        `df_test_source is None` (regular pots, e.g. CM04, CM08):
            Standard shuffled `n_splits`-fold KFold on `df_current[brand==b]`.
        `df_test_source` provided (e.g. CM08_off75 → CM08):
            Folds are defined on `df_test_source`'s brand rows by `hour`. For each
            fold, val rows come from `df_test_source` (preserving the source pot's
            preprocessed values exactly). Train rows are `df_current` rows whose
            `hour` is NOT in that fold's val hours. Off75-only hours all flow into
            training across all folds.

    `split_mode="holdout"` — single forward (chronological) holdout:
        Sort the brand rows by `hour` and yield ONE fold: train = the earliest
        `(1 - test_frac)` of rows, val = the most-recent `test_frac` of rows. This
        is the forward-evaluation gate (production predicts forward in time). The
        value-identity `df_test_source` path is IID-only and unsupported here.
    """
    if split_mode not in ("iid", "holdout"):
        raise ValueError(f"Unknown split_mode: {split_mode!r}. Valid: 'iid', 'holdout'.")

    feats_no_brand = [c for c in feature_cols if c != 'brand']

    df_cur_b = (df_current[df_current['brand'] == brand]
                .dropna(subset=[target])
                .reset_index(drop=True))

    if split_mode == "holdout":
        if df_test_source is not None:
            raise NotImplementedError(
                "split_mode='holdout' does not support the value-identity df_test_source "
                "path (off75 augmentation). Use split_mode='iid' for those pots."
            )
        # Chronological order is essential — model_train.py does not pre-sort.
        df_cur_b = df_cur_b.sort_values('hour').reset_index(drop=True)
        n = len(df_cur_b)
        n_test = max(1, int(round(n * test_frac)))
        cut = n - n_test
        X_b = df_cur_b[feats_no_brand]
        y_b = df_cur_b[target]
        hours_b = df_cur_b['hour']
        yield (
            X_b.iloc[:cut], X_b.iloc[cut:],
            y_b.iloc[:cut], y_b.iloc[cut:],
            hours_b.iloc[:cut], hours_b.iloc[cut:],
        )
        return

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    if df_test_source is None:
        X_b = df_cur_b[feats_no_brand]
        y_b = df_cur_b[target]
        hours_b = df_cur_b['hour']
        for train_idx, val_idx in kf.split(np.arange(len(df_cur_b))):
            yield (
                X_b.iloc[train_idx], X_b.iloc[val_idx],
                y_b.iloc[train_idx], y_b.iloc[val_idx],
                hours_b.iloc[train_idx], hours_b.iloc[val_idx],
            )
        return

    # Value-identity: fold partitions defined on the source pot.
    df_src_b = (df_test_source[df_test_source['brand'] == brand]
                .dropna(subset=[target])
                .reset_index(drop=True))

    for _, val_idx in kf.split(np.arange(len(df_src_b))):
        val_hours = set(df_src_b.loc[val_idx, 'hour'])
        X_val = df_src_b.loc[val_idx, feats_no_brand]
        y_val = df_src_b.loc[val_idx, target]
        hours_val = df_src_b.loc[val_idx, 'hour']

        train_mask = ~df_cur_b['hour'].isin(val_hours)
        X_train = df_cur_b.loc[train_mask, feats_no_brand]
        y_train = df_cur_b.loc[train_mask, target]
        hours_train = df_cur_b.loc[train_mask, 'hour']

        yield X_train, X_val, y_train, y_val, hours_train, hours_val


# ────────────────────────────────────────────────────────────────────────────
# Per-fold time-membership scatter
# ────────────────────────────────────────────────────────────────────────────

def plot_target_over_time_fold(hours_train, y_train, hours_val, y_val,
                                target_name, brand, fold_idx, save_path=None):
    """Sanity scatter: train (steelblue) and val (coral X) points over time.

    Under the forward holdout, val is the most-recent contiguous block (the future
    tail) and train is everything earlier — they should NOT interleave. Under the
    legacy shuffled KFold they should interleave across the full time range; mixing
    there would indicate a stale `random_state` or upstream sort bug.
    """
    fig, ax = plt.subplots(figsize=(14, 4))

    ax.scatter(hours_train, y_train, s=20, alpha=0.7, color='steelblue',
               label=f'Train ({len(hours_train)})')
    ax.scatter(hours_val, y_val, s=40, alpha=0.9, color='coral', marker='x',
               label=f'Val ({len(hours_val)})')

    ax.set_title(f'{target_name} — Fold {fold_idx} train/val membership over time '
                 f'(Brand: {brand})', fontsize=13, fontweight='bold')
    ax.set_xlabel('Time (Month)')
    ax.set_ylabel(target_name)

    all_hours = pd.concat([pd.Series(hours_train), pd.Series(hours_val)])
    ax.set_xlim(all_hours.min(), all_hours.max())
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.xticks(rotation=45)

    ax.legend(markerscale=1.5)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


# ────────────────────────────────────────────────────────────────────────────
# Top-level driver: nested 5×5 CV for one (brand, model, target)
# ────────────────────────────────────────────────────────────────────────────

def run_nested_cv(
    *,
    model_name: str,
    df_current: pd.DataFrame,
    df_test_source: pd.DataFrame | None,
    brand,
    target: str,
    feature_cols: list[str],
    num_cols: list[str],
    n_outer_folds: int = 5,
    n_inner_folds: int = 5,
    n_optuna_trials: int = 50,
    random_state: int = 42,
    optuna_seed: int = 42,
    plot_save_dir: str | None = None,
    plot_target_label: str | None = None,
    split_mode: str = "iid",
    inner_split: str = "kfold",
    test_frac: float = 0.2,
):
    """Evaluate one (brand, target, model) and return per-fold records.

    `split_mode="iid"` + `inner_split="kfold"` (defaults) runs the legacy **nested 5×5
    CV** (outer shuffled KFold = honest estimate; inner shuffled KFold = Optuna search).

    `split_mode="holdout"` + `inner_split="ts"` runs the **forward holdout**: a single
    most-recent-window test (last `test_frac`, by sample fraction), with hyperparameters
    tuned on the earlier train span via inner `TimeSeriesSplit`. This is *not* nested —
    there is one train/test split, not an outer loop — so it returns a single record.

    Each record is a dict with keys:
        fold, train_samples, val_samples, MAE, RMSE, R2, MAPE(%),
        Max |Residual|, Min |Residual|, Mean Residual, SD Residual,
        Train_Time(s), best_params, last_fold_model, last_fold_X_val_le_sc.

    The last two are populated only on the *final* outer fold and are intended
    for downstream SHAP computation. They are absent from earlier-fold records
    to keep the in-memory footprint low.
    """
    if model_name not in VALID_MODELS:
        raise ValueError(f"Unknown model: {model_name!r}. Valid: {VALID_MODELS}")
    use_oh = (model_name == "KNN")

    target_label = plot_target_label or target
    brand_safe = str(brand).replace(" ", "_").replace("/", "_")

    fold_records = []
    fold_iter = make_brand_folds(
        df_current, df_test_source, brand, target, feature_cols,
        n_splits=n_outer_folds, random_state=random_state,
        split_mode=split_mode, test_frac=test_frac,
    )
    is_holdout = (split_mode == "holdout")

    for k, (X_otr, X_oval, y_otr, y_oval, hours_otr, hours_oval) in enumerate(fold_iter, 1):
        # Sample-size guard. Inner CV needs enough rows to split 5 ways.
        if len(y_otr) < n_inner_folds * 2 or len(y_oval) < 1:
            print(f"  -> Fold {k}: skipped (outer-train={len(y_otr)}, outer-val={len(y_oval)})")
            continue

        encoded = encode_and_scale_features_train_test(
            X_otr, X_oval, num_cols, verbose=(k == 1),
        )
        X_otr_enc, X_oval_enc = encoded['oh_sc'] if use_oh else encoded['le_sc']

        # Inner-CV Optuna search on the outer-train slice.
        sampler = optuna.samplers.TPESampler(seed=optuna_seed, n_startup_trials=5)
        study = optuna.create_study(
            direction='minimize', sampler=sampler,
            study_name=f'{model_name}_{target}_{brand_safe}_outer{k}',
        )
        objective = make_objective(model_name, X_otr_enc, y_otr,
                                    n_inner_folds, random_state,
                                    inner_split=inner_split)
        study.optimize(objective, n_trials=n_optuna_trials, show_progress_bar=False)

        # Refit best params on the full outer-train, evaluate on outer-val.
        start = time.time()
        model = make_model(model_name, study.best_params, random_state)
        model.fit(X_otr_enc, y_otr)
        train_time = time.time() - start

        y_pred_val = model.predict(X_oval_enc)
        metrics = evaluate_model(y_oval.values, y_pred_val, set_name=f"Fold {k}")
        res_stats = evaluate_residuals(y_oval.values, y_pred_val)

        record = {
            'fold': k,
            'train_samples': len(y_otr),
            'val_samples': len(y_oval),
            **metrics, **res_stats,
            'Train_Time(s)': train_time,
            'best_params': dict(study.best_params),
        }

        # Per-fold time-membership sanity plot.
        if plot_save_dir is not None:
            save_path = (f"{plot_save_dir}/timeseries_{target}_brand_{brand_safe}"
                         f"_fold{k}.png")
            plot_target_over_time_fold(
                hours_train=hours_otr, y_train=y_otr,
                hours_val=hours_oval, y_val=y_oval,
                target_name=target_label, brand=brand, fold_idx=k,
                save_path=save_path,
            )

        # Keep last fold's fitted model + encoded val for SHAP. For the single
        # forward holdout there is only one fold, so retain on it.
        if is_holdout or k == n_outer_folds:
            record['last_fold_model'] = model
            record['last_fold_X_val_encoded'] = X_oval_enc

        fold_label = "holdout" if is_holdout else f"{k}/{n_outer_folds}"
        print(f"  Fold {fold_label}: "
              f"MAE={metrics['MAE']:.4f}, RMSE={metrics['RMSE']:.4f}, "
              f"R²={metrics['R2']:.4f}, MAPE={metrics['MAPE(%)']:.2f}%  "
              f"({train_time:.1f}s)")

        fold_records.append(record)

    return fold_records


def summarize_fold_records(fold_records: list[dict]) -> dict:
    """Mean ± std across folds for the standard metric set + total train time.

    With a single fold (the forward holdout) there is no spread, so the ``*_std``
    fields are reported as 0.0 rather than pandas ``NaN`` to keep downstream
    CSVs/plots well-formed.
    """
    if not fold_records:
        return {}
    metric_cols = ['MAE', 'RMSE', 'R2', 'MAPE(%)',
                   'Max |Residual|', 'Min |Residual|', 'Mean Residual', 'SD Residual']
    df = pd.DataFrame(fold_records)
    single = len(df) == 1
    summary = {}
    for m in metric_cols:
        summary[m] = float(df[m].mean())
        summary[f'{m}_std'] = 0.0 if single else float(df[m].std())
    summary['Train_Time(s)'] = float(df['Train_Time(s)'].sum())
    return summary
