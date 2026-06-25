"""Lightweight MLflow wrapper for the cement-fineness modelling pipeline.

Goals:
- Keep downstream notebooks free of MLflow API details. They call four functions.
- Pin the tracking URI to a SQLite DB at the project root so the UI can be launched
  with one shell command (`mlflow ui --backend-store-uri sqlite:///mlflow.db`).
- Build a stable `setup_hash` from the data-prep manifest + feature set so runs
  with identical pre-modelling configuration can be grouped in the UI.

Typical 01_2 usage (run once at the top of the notebook):

    from utils.experiment_tracker import (
        init_tracking, start_parent_run, log_model_result,
        finalize_parent_summary, end_parent_run,
    )

    init_tracking(pot=POT, workflow="hyperparameter_tuning")

    manifest = {
        "pot": POT,
        "data_source": str(cfg["merged_data_path"].name),
        "n_rows": len(df),
        "aggregation": "mean_hourly",
        "mill_power_filter": "q75_minus_200_plus_200",
        "outlier_capping": "1st-99th per brand",
        "feature_set": cfg["model_features"],
        "split": {"test_size": 0.20, "shuffle": True, "seed": RANDOM_STATE},
        "cv": {"folds": N_CV_FOLDS, "shuffle": True, "seed": RANDOM_STATE},
        "n_optuna_trials": N_OPTUNA_TRIALS,
    }
    start_parent_run(manifest, run_name=None)        # opens parent run

    # ... inside the per-(target, brand, model) loop:
    log_model_result(
        target="blaine", brand=b, model_name="XGBoost",
        best_params=xgb_study.best_params,
        test_metrics={"mae": ..., "rmse": ..., "r2": ..., "mape": ...},
        train_metrics=None,           # optional
        train_time_s=...,
    )

    # at the very end:
    finalize_parent_summary({"blaine_mae_weighted": ..., "psd_r30_mae_weighted": ...})
    end_parent_run()
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import mlflow

# Standalone-repo layout: this file lives at <repo>/utils/experiment_tracker.py, so the repo root is
# one parent up (utils -> repo root). parents[1] keeps mlflow.db / mlruns at the repo root (vendored
# copy — the research repo's version used parents[2] for the notebooks/utils nesting).
PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRACKING_URI = f"sqlite:///{PROJECT_ROOT / 'mlflow.db'}"
ARTIFACT_ROOT = PROJECT_ROOT / "mlruns"


def _compute_setup_hash(manifest: dict[str, Any]) -> str:
    """SHA256 of the canonical manifest, truncated to 8 hex chars.

    Two runs with the same setup_hash share: data source file, filter regime,
    aggregation, capping rule, feature list (order-insensitive), and CV config
    (legacy `split`/`cv` keys plus the nested-CV `outer_cv`/`inner_cv`/`n_optuna_trials`
    keys are all read; whichever the manifest provides participates in the hash).
    Model hyperparameters are NOT part of the hash — those are the experiment knobs.

    Note: existing nested-CV runs created before this fix will have hashes that
    differ from any future re-run, since their `outer_cv`/`inner_cv` values were
    silently absent from the previous canonical form. No data is lost — just the
    "Setup" grouping in the UI for those legacy runs no longer matches new runs.
    """
    canonical = {
        "data_source":       manifest.get("data_source"),
        "aggregation":       manifest.get("aggregation"),
        "mill_power_filter": manifest.get("mill_power_filter"),
        "outlier_capping":   manifest.get("outlier_capping"),
        "feature_set":       sorted(manifest.get("feature_set", [])),
        # Legacy single-split / single-CV keys (01_2 pre-nested refactor).
        "split":             manifest.get("split"),
        "cv":                manifest.get("cv"),
        # Nested 5×5 CV keys (01_2 / 02_1 post-refactor).
        "outer_cv":          manifest.get("outer_cv"),
        "inner_cv":          manifest.get("inner_cv"),
        "n_optuna_trials":   manifest.get("n_optuna_trials"),
    }
    blob = json.dumps(canonical, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:8]


def init_tracking(pot: str, workflow: str) -> str:
    """Point MLflow at the project-local SQLite store and select / create the experiment.

    Returns the active experiment name, e.g. ``"CM08_hyperparameter_tuning"``.
    """
    ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(TRACKING_URI)
    experiment_name = f"{pot}_{workflow}"
    mlflow.set_experiment(experiment_name)
    return experiment_name


def start_parent_run(
    manifest: dict[str, Any],
    run_name: str | None = None,
    extra_tags: dict[str, Any] | None = None,
) -> str:
    """Start a parent MLflow run, log the manifest as params + tags, return run_id.

    Notebook cells can call this once near the top; subsequent ``log_model_result``
    calls create nested children under it.

    `extra_tags` is for caller-specific labels (e.g. `model_name`, `fe_experiment`)
    that aren't part of the universal manifest schema. Values are stringified;
    `None` values are dropped.
    """
    setup_hash = _compute_setup_hash(manifest)
    pot = manifest.get("pot", "UNKNOWN")
    if run_name is None:
        run_name = f"{pot}_setup_{setup_hash}"

    run = mlflow.start_run(run_name=run_name)

    feature_set = list(manifest.get("feature_set", []))
    feature_set_sorted = sorted(feature_set)
    features_hash = hashlib.sha256(
        json.dumps(feature_set_sorted).encode()
    ).hexdigest()[:8]

    # Params: scalar manifest entries. Pulls outer/inner-CV folds from the new
    # nested keys when present; falls back to the legacy `cv` and `split` keys.
    outer_cv = manifest.get("outer_cv") or {}
    inner_cv = manifest.get("inner_cv") or {}
    legacy_cv = manifest.get("cv") or {}
    legacy_split = manifest.get("split") or {}
    flat_params = {
        "pot": manifest.get("pot"),
        "data_source": manifest.get("data_source"),
        "n_rows": manifest.get("n_rows"),
        "aggregation": manifest.get("aggregation"),
        "mill_power_filter": manifest.get("mill_power_filter"),
        "outlier_capping": manifest.get("outlier_capping"),
        "n_features": len(feature_set),
        "features_hash": features_hash,
        "n_optuna_trials": manifest.get("n_optuna_trials"),
        "outer_cv_folds": outer_cv.get("folds"),
        "inner_cv_folds": inner_cv.get("folds"),
        "cv_folds": legacy_cv.get("folds"),
        "split_test_size": legacy_split.get("test_size"),
    }
    mlflow.log_params({k: v for k, v in flat_params.items() if v is not None})

    # Tags: anything you'd want to filter on in the UI
    mlflow.set_tags({
        "setup_hash": setup_hash,
        "pot": str(manifest.get("pot")),
        "workflow_role": "parent",
        # Full feature-name list (sorted) so runs with the same `n_features`
        # but a different selection are distinguishable in the UI.
        "features": ",".join(feature_set_sorted),
    })
    if extra_tags:
        mlflow.set_tags({k: str(v) for k, v in extra_tags.items() if v is not None})

    # Full manifest as a JSON artefact (paths, lists, nested dicts — keeps params clean)
    mlflow.log_dict(manifest, "manifest.json")
    return run.info.run_id


def log_parent_artifact(path: str | Path) -> None:
    """Upload a file as an artifact on the still-open parent run.

    Use this to attach summary CSVs / plots so they're visible in the MLflow UI
    alongside the run's metrics. Must be called while the parent run is the active
    run (i.e. outside any `log_model_result` context). Silently no-ops if no run
    is active.
    """
    if mlflow.active_run() is None:
        return
    mlflow.log_artifact(str(path))


def log_model_result(
    *,
    target: str,
    brand: str,
    model_name: str,
    best_params: dict[str, Any],
    test_metrics: dict[str, float],
    train_metrics: dict[str, float] | None = None,
    train_time_s: float | None = None,
    fold: int | None = None,
) -> str:
    """Create a nested child run for one (target, brand, model[, fold]) result.

    Logs best_params (as params) and test/train MAE/RMSE/R²/MAPE (as metrics).
    Saves best_params as a JSON artefact too — convenient for downstream notebooks
    that want to reload the dict directly.

    Pass `fold=k` (1-indexed) to log a per-outer-fold child under nested CV. The
    `test_metrics` payload then represents the outer-fold validation metrics for
    that fold; `best_params` are the params chosen by that fold's inner Optuna CV.
    """
    run_name = f"{target}__{brand}__{model_name}"
    if fold is not None:
        run_name += f"__fold{fold}"
    with mlflow.start_run(run_name=run_name, nested=True):
        tags = {
            "target": target,
            "brand": str(brand),
            "model_name": model_name,
            "workflow_role": "child",
        }
        if fold is not None:
            tags["outer_fold"] = str(fold)
        mlflow.set_tags(tags)

        # Hyperparameters as params (MLflow casts to str; that's fine for retrieval)
        if best_params:
            mlflow.log_params({f"hp_{k}": v for k, v in best_params.items()})

        # Metrics
        for k, v in test_metrics.items():
            mlflow.log_metric(f"test_{k}", float(v))
        if train_metrics:
            for k, v in train_metrics.items():
                mlflow.log_metric(f"train_{k}", float(v))
        if train_time_s is not None:
            mlflow.log_metric("train_time_s", float(train_time_s))

        # Artefact: best params as JSON for easy reload
        mlflow.log_dict(best_params or {}, "best_params.json")

        return mlflow.active_run().info.run_id


def start_intermediate_run(
    *,
    target: str,
    brand: str,
    model_name: str,
    run_name: str | None = None,
) -> str:
    """Open a nested 'intermediate' run for one (target, brand) summary.

    Sits between the parent run (one per notebook execution) and the per-fold
    child runs (one per outer fold). After calling this, subsequent
    ``log_model_result(fold=k)`` calls nest under it (MLflow uses the most
    recently opened run as the parent for ``nested=True``).

    Caller MUST close it with ``end_intermediate_run()`` before opening the
    next (target, brand) intermediate.
    """
    if run_name is None:
        run_name = f"{brand}_{target}_avg"
    run = mlflow.start_run(run_name=run_name, nested=True)
    mlflow.set_tags({
        "target": target,
        "brand": str(brand),
        "model_name": model_name,
        "workflow_role": "intermediate",
    })
    return run.info.run_id


def log_intermediate_summary(
    *,
    summary: dict[str, float],
    fold_best_params: list[dict[str, Any]],
) -> None:
    """Log fold-aggregated metrics + per-fold best_params on the active intermediate run.

    ``summary`` is the dict returned by ``summarize_fold_records`` (keys like
    ``MAE``, ``MAE_std``, ...). Each metric is logged with ``_mean`` / ``_std``
    suffixes, matching the convention used by other MLflow summary panels.

    ``fold_best_params[i]`` is the best_params dict from outer fold ``i+1``; it's
    logged as a JSON-string param ``param_fold_{i+1}`` and also written as a JSON
    artefact for easy reload.
    """
    metric_renames = {
        'MAE': 'mae', 'RMSE': 'rmse', 'R2': 'r2', 'MAPE(%)': 'mape',
        'Max |Residual|': 'max_abs_residual',
        'Min |Residual|': 'min_abs_residual',
        'Mean Residual':  'mean_residual',
        'SD Residual':    'sd_residual',
    }
    for src, dst in metric_renames.items():
        if src in summary:
            mlflow.log_metric(f"{dst}_mean", float(summary[src]))
        std_key = f"{src}_std"
        if std_key in summary:
            mlflow.log_metric(f"{dst}_std", float(summary[std_key]))
    if 'Train_Time(s)' in summary:
        mlflow.log_metric("train_time_s_total", float(summary['Train_Time(s)']))

    for i, params in enumerate(fold_best_params, 1):
        mlflow.log_param(f"param_fold_{i}", json.dumps(params or {}))
    mlflow.log_dict(
        {f"fold_{i}": p for i, p in enumerate(fold_best_params, 1)},
        "fold_best_params.json",
    )


def end_intermediate_run() -> None:
    """Close the intermediate run, leaving the parent run active. Idempotent."""
    if mlflow.active_run() is not None:
        mlflow.end_run()


def finalize_parent_summary(summary_metrics: dict[str, float]) -> None:
    """Log final weighted-average / overall metrics on the still-open parent run.

    Call from the last cell of the notebook, before ``end_parent_run()``.
    """
    for k, v in summary_metrics.items():
        mlflow.log_metric(k, float(v))


def end_parent_run() -> None:
    """Close the parent run. Idempotent: safe to call when no run is active."""
    if mlflow.active_run() is not None:
        mlflow.end_run()
