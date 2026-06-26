#!/usr/bin/env python
"""Train + freeze the predictor for the APC loop (two phases).

Extracts the modelling core of ``notebooks/02_1_FeatEng.py`` (drops EDA / correlation / SHAP;
keeps MLflow logging). The model architecture is locked per mill by ``pot_config[cm]["model_name"]``
(CatBoost for CM08, XGBoost CM04, LightGBM CM07); hyperparameters are re-tuned with Optuna, as the
notebook does — no JSON is read back.

Two phases (per the deployment requirement):
  • Phase A — Evaluate & report. Run the same nested 5×5 CV the notebook uses and report the honest
    score (MAE/RMSE/R²/MAPE, mean±std across the 5 outer folds). The reported MAE feeds ``model_mae``
    in the frozen metadata. This estimates the *procedure*; the deployed model (Phase B) is the same
    procedure fit on all data, so this number is a slightly conservative estimate of its performance.
  • Phase B — Final weight. Re-tune once on the full filtered data (inner 5-fold Optuna), refit on
    100% of it, and write the frozen artifact the APC script consumes.

Artifact (written to both the canonical out-dir = apc default, AND out-dir/runs/<stamp>/ for history):
  • model.joblib          — trained regressor (raw numeric features, no scaler; tree model).
  • metadata.json         — apc contract: target, pot, brand, feature_order (numeric, no brand),
                            sep_motor_speed_bounds [min,max,step], model_mae (Phase A), is_autogluon,
                            trained_through, + provenance (trained_at, source_data_file).
  • cv_report_<t>_<b>.csv  — per-fold + summary nested-CV metrics.

MLflow: the nested-CV folds + final summary are logged to a repo-local SQLite store
(``sqlite:///mlflow.db`` at the repo root, experiment ``<cm>_apc_training``). Browse with
``mlflow ui --backend-store-uri sqlite:///mlflow.db``. Disable with ``--no-mlflow``.

Usage (conda env CBM):
  python model_train.py --cm CM08_bf1h --target psd_r30 --brand PCC --optuna-trials 25
"""
import argparse
import json
from datetime import datetime
from pathlib import Path

import joblib
import optuna
import pandas as pd

# Vendored packages (config/, utils/, feat_eng/) are importable via the editable install (pip install -e .).
from config.pot_config import get_config, RESULTS_DIR  # noqa: E402
from feat_eng.registry import get_experiment  # noqa: E402
from utils._modeling_utils import (  # noqa: E402
    VALID_MODELS, run_nested_cv, summarize_fold_records, make_objective, make_model,
)
from utils.experiment_tracker import (  # noqa: E402
    init_tracking, start_parent_run, log_model_result,
    start_intermediate_run, log_intermediate_summary, end_intermediate_run,
    log_parent_artifact, finalize_parent_summary, end_parent_run,
)

optuna.logging.set_verbosity(optuna.logging.WARNING)

N_OUTER_FOLDS = 5
N_INNER_FOLDS = 5
RANDOM_STATE = 42
OPTUNA_SEED = 42
ACTIONABLE = "sep_motor_speed"
# feat_eng experiment applied to every pot by default (override with --feat-eng / --no-feat-eng).
# 04_seasonal adds time-of-arrival cyclic + Thai-season features derived from the `hour` column;
# they are independent of the actionable param (AFFECTS_ACTIONABLE=False), so they shift the
# baseline prediction by time-of-year/day without changing the sep_motor_speed search surface.
DEFAULT_FEAT_ENG = "04_seasonal"


def apply_feat_eng(df, experiment):
    """Apply a feat_eng experiment to ``df``; return (df, new_feature_names).

    Applied before the training-window filter so the experiment sees the raw string ``hour``
    column it parses. ``experiment`` of None/"" /"none" disables FE and returns no new features.
    """
    if not experiment or str(experiment).lower() == "none":
        print("Feature engineering: disabled (no feat_eng experiment applied).")
        return df, []
    exp = get_experiment(experiment)
    print(f"Feature engineering: {experiment} — {exp['description']}")
    df, new_features, _ = exp["apply"](df)
    return df, new_features


def _filter_training_window(df, cfg):
    """Restrict to the post-drift training regime if the pot defines one (AGENTS.md)."""
    df["hour"] = pd.to_datetime(df["hour"])
    start = cfg.get("training_start")
    if start:
        n0 = len(df)
        df = df[df["hour"] >= pd.Timestamp(start)].reset_index(drop=True)
        print(f"Training window ≥ {start}: {n0:,} → {len(df):,} rows")
    else:
        print("No training_start set for this pot — using all rows.")
    return df


def phase_a_report(df, cfg, model_name, brand, target, n_trials, feature_cols):
    """Nested 5×5 CV exactly as the notebook; returns (summary, fold_records).

    ``feature_cols`` is the full model-input list (incl. 'brand' and any feat_eng columns).
    """
    feature_cols = list(feature_cols)                        # includes 'brand'
    num_cols = [c for c in feature_cols if c != "brand"]     # all non-brand features are numeric

    df_b = df[df["brand"] == brand].dropna(subset=[target])
    print(f"\n── Phase A: nested {N_OUTER_FOLDS}×{N_INNER_FOLDS} CV "
          f"({model_name}, brand={brand}, target={target}, rows={len(df_b):,}) ──")
    if len(df_b) < N_OUTER_FOLDS * 2:
        raise RuntimeError(f"Insufficient samples for {brand}/{target}: {len(df_b)}")

    fold_records = run_nested_cv(
        model_name=model_name, df_current=df, df_test_source=None,
        brand=brand, target=target, feature_cols=feature_cols, num_cols=num_cols,
        n_outer_folds=N_OUTER_FOLDS, n_inner_folds=N_INNER_FOLDS,
        n_optuna_trials=n_trials, random_state=RANDOM_STATE, optuna_seed=OPTUNA_SEED,
        plot_save_dir=None, plot_target_label=target,
    )
    if not fold_records:
        raise RuntimeError(f"No outer folds completed for {brand}/{target}.")

    summary = summarize_fold_records(fold_records)
    print(f"\n  Score  MAE={summary['MAE']:.4f} ± {summary['MAE_std']:.4f}  "
          f"RMSE={summary['RMSE']:.4f} ± {summary['RMSE_std']:.4f}  "
          f"R²={summary['R2']:.4f} ± {summary['R2_std']:.4f}  "
          f"MAPE={summary['MAPE(%)']:.2f}%")
    return summary, fold_records


def phase_b_freeze(df, cfg, model_name, brand, target, n_trials, feature_cols):
    """Re-tune on the full filtered data, refit on 100% of it, return (model, feature_order).

    ``feature_cols`` is the full model-input list (incl. 'brand' and any feat_eng columns).
    """
    feature_order = [c for c in feature_cols if c != "brand"]  # numeric, no brand
    df_b = df[df["brand"] == brand].dropna(subset=[target]).reset_index(drop=True)
    X_full = df_b[feature_order]
    y_full = df_b[target]

    print(f"\n── Phase B: final weight ({model_name}, {len(X_full):,} rows, "
          f"{len(feature_order)} features) ──")

    # One Optuna pass on the full data (inner 5-fold CV), same search space as the notebook.
    sampler = optuna.samplers.TPESampler(seed=OPTUNA_SEED, n_startup_trials=5)
    study = optuna.create_study(direction="minimize", sampler=sampler,
                                study_name=f"{model_name}_{target}_{brand}_full")
    objective = make_objective(model_name, X_full, y_full, N_INNER_FOLDS, RANDOM_STATE)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    print(f"  Best inner-CV MAE: {study.best_value:.4f}")

    model = make_model(model_name, study.best_params, RANDOM_STATE)
    model.fit(X_full, y_full)
    print(f"  Refit on all {len(X_full):,} rows.")
    return model, feature_order, df_b, study.best_params, float(study.best_value)


def _write_artifact(out_dir, stamp, model, metadata, cv_report_df, target, brand):
    """Write model.joblib + metadata.json + cv_report to canonical out_dir AND runs/<stamp>/."""
    cv_name = f"cv_report_{target}_{brand}.csv"
    for d in (out_dir, out_dir / "runs" / stamp):
        d.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, d / "model.joblib")
        (d / "metadata.json").write_text(json.dumps(metadata, indent=2))
        cv_report_df.to_csv(d / cv_name, index=False)
    print(f"  ✅ Artifact → {out_dir}  (+ runs/{stamp}/)")


def main():
    ap = argparse.ArgumentParser(description="Train + freeze APC predictor (nested-CV report + refit).")
    ap.add_argument("--cm", required=True, help="Cement mill / pot (locks model arch via pot_config).")
    ap.add_argument("--target", default="psd_r30", help="Target variable (default psd_r30).")
    ap.add_argument("--brand", default=None, help="Brand (default = pot's first optimization brand).")
    ap.add_argument("--data", default=None, help="Merged CSV (default = pot_config merged_data_path).")
    ap.add_argument("--optuna-trials", type=int, default=25, help="Optuna trials per tuning (default 25).")
    ap.add_argument("--feat-eng", default=DEFAULT_FEAT_ENG,
                    help=f"feat_eng experiment applied to every pot (default {DEFAULT_FEAT_ENG!r}). "
                         "Use 'none' to disable.")
    ap.add_argument("--no-feat-eng", action="store_true",
                    help="Disable feature engineering (equivalent to --feat-eng none).")
    ap.add_argument("--step", type=float, default=1.0,
                    help="sep_motor_speed grid step recorded in metadata bounds (default 1.0).")
    ap.add_argument("--out-dir", default=None,
                    help="Artifact dir (default results/<CM>/optimization/frozen_<target>_<brand>).")
    ap.add_argument("--no-mlflow", action="store_true", help="Skip MLflow logging.")
    args = ap.parse_args()

    cfg = get_config(args.cm)
    model_name = cfg["model_name"]
    assert model_name in VALID_MODELS, f"Pot model {model_name!r} not in {VALID_MODELS}"
    brand = args.brand or cfg["optimization_brands"][0]
    target = args.target
    if target not in cfg["target_cols"]:
        raise ValueError(f"target {target!r} not in {cfg['target_cols']}")

    data_path = Path(args.data) if args.data else Path(cfg["merged_data_path"])
    out_dir = (Path(args.out_dir) if args.out_dir
               else RESULTS_DIR / args.cm / "optimization" / f"frozen_{target}_{brand}")
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    use_mlflow = not args.no_mlflow

    feat_eng_exp = "none" if args.no_feat_eng else args.feat_eng

    print(f"=== Train {args.cm} / {target} / {brand} — model={model_name} — run {stamp} ===")
    print(f"Data: {data_path}")
    df = pd.read_csv(data_path)
    # Feature engineering first (sees raw string `hour`), then restrict to the training window.
    df, fe_features = apply_feat_eng(df, feat_eng_exp)
    df = _filter_training_window(df, cfg)

    # Full model-input list = pot's canonical features + any feat_eng columns (all numeric).
    feature_cols = list(cfg["model_features"]) + fe_features

    if use_mlflow:
        init_tracking(pot=args.cm, workflow="apc_training")
        manifest = {
            "pot": args.cm, "data_source": data_path.name, "n_rows": len(df),
            **cfg["data_recipe"], "feature_set": list(feature_cols),
            "feat_eng": feat_eng_exp, "fe_features": fe_features,
            "outer_cv": {"folds": N_OUTER_FOLDS, "shuffle": True, "seed": RANDOM_STATE},
            "inner_cv": {"folds": N_INNER_FOLDS, "shuffle": True, "seed": RANDOM_STATE},
            "n_optuna_trials": args.optuna_trials,
            "model_name": model_name, "target": target, "brand": brand,
        }
        start_parent_run(manifest, run_name=f"{args.cm}_{model_name}_{target}_{brand}",
                         extra_tags={"model_name": model_name, "target": target, "brand": brand})

    try:
        # Phase A — honest score.
        summary, fold_records = phase_a_report(df, cfg, model_name, brand, target,
                                               args.optuna_trials, feature_cols)

        if use_mlflow:
            start_intermediate_run(target=target, brand=brand, model_name=model_name)
            for r in fold_records:
                log_model_result(
                    target=target, brand=brand, model_name=model_name,
                    best_params=r["best_params"],
                    test_metrics={"mae": r["MAE"], "rmse": r["RMSE"],
                                  "r2": r["R2"], "mape": r["MAPE(%)"]},
                    train_time_s=r["Train_Time(s)"], fold=r["fold"],
                )
            log_intermediate_summary(summary=summary,
                                     fold_best_params=[r["best_params"] for r in fold_records])
            end_intermediate_run()

        # Phase B — final weight on full data.
        model, feature_order, df_b, best_params, best_inner_mae = phase_b_freeze(
            df, cfg, model_name, brand, target, args.optuna_trials, feature_cols)

        # CV report (per-fold rows + a summary row).
        cv_rows = [{"scope": f"fold_{r['fold']}", **{k: r[k] for k in
                   ("train_samples", "val_samples", "MAE", "RMSE", "R2", "MAPE(%)")}}
                   for r in fold_records]
        cv_rows.append({"scope": "summary(mean)", "train_samples": "", "val_samples": "",
                        "MAE": summary["MAE"], "RMSE": summary["RMSE"],
                        "R2": summary["R2"], "MAPE(%)": summary["MAPE(%)"]})
        cv_rows.append({"scope": "summary(std)", "train_samples": "", "val_samples": "",
                        "MAE": summary["MAE_std"], "RMSE": summary["RMSE_std"],
                        "R2": summary["R2_std"], "MAPE(%)": summary["MAPE(%)_std"]})
        cv_report_df = pd.DataFrame(cv_rows)

        # Metadata (apc contract + provenance).
        sep = df_b[ACTIONABLE]
        metadata = {
            "target": target, "pot": args.cm, "brand": brand,
            "feature_order": feature_order,
            "feat_eng": feat_eng_exp, "fe_features": fe_features,
            "sep_motor_speed_bounds": [float(sep.min()), float(sep.max()), float(args.step)],
            "model_mae": float(summary["MAE"]),
            "is_autogluon": False,
            "trained_through": str(pd.to_datetime(df_b["hour"]).max().date()),
            "trained_at": stamp,
            "source_data_file": data_path.name,
            "model_name": model_name,
            "final_best_params": best_params,
        }

        _write_artifact(out_dir, stamp, model, metadata, cv_report_df, target, brand)

        if use_mlflow:
            finalize_parent_summary({"model_mae": float(summary["MAE"]),
                                     "final_inner_cv_mae": best_inner_mae})
            for fname in ("model.joblib", "metadata.json", f"cv_report_{target}_{brand}.csv"):
                log_parent_artifact(out_dir / fname)
    finally:
        if use_mlflow:
            end_parent_run()

    print(f"\n✅ Done. model_mae={metadata['model_mae']:.4f}, "
          f"bounds={metadata['sep_motor_speed_bounds']}, "
          f"trained_through={metadata['trained_through']}")


if __name__ == "__main__":
    main()
