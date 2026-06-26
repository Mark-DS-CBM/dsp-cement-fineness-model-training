#!/usr/bin/env python
"""APC core: Predict + Optimize (single-step inverse optimization for CM08 psd_r30).

Implements the "Predict + Optimize Loop" box from the APC design diagram:

    features (current 60-min snapshot) ─┐
    desired psd_r30 setpoint ───────────┤
    constraints (sep_motor_speed range) ─┼──▶  Predict + Optimize  ──▶  recommended
    frozen predictor (CM08 psd_r30) ─────┘                              sep_motor_speed
                                                                        + alternatives

Scope:
  • Predict + optimize CORE ONLY — no 5-min tick, no brand/power/drift gates, no Ignition read or
    OPC UA write. Those live in the surrounding APC loop.
  • Standalone CPython (conda env ``CBM``).
  • Loads a FROZEN model artifact produced by ``model_train.py`` and errors clearly if it is missing.

Reuse from the notebook codebase (not reinvented):
  • ``build_predictor`` mirrors notebooks/04_1_Optimization.py (dict -> float).
  • The optimizer is loaded verbatim from ``opt_algo/`` (default grid search) — the same
    plug-and-play contract 04_1 uses. Its literal key ``'desired_psd_r30'`` carries the setpoint.

Frozen artifact layout (``<ARTIFACT_DIR>/``), written by ``model_train.py``:
  • ``model.joblib``     — trained sklearn-style regressor (CM08 PCC psd_r30).
  • ``metadata.json``    — {target, pot, brand, feature_order (no 'brand'),
        sep_motor_speed_bounds:[min,max,step], model_mae, is_autogluon, trained_through, ...}.
"""
import argparse
import importlib
import json
import os
from pathlib import Path

import joblib
import pandas as pd

# Vendored packages (config/, opt_algo/, feat_eng/) are importable via the editable install.
from config.pot_config import get_config, RESULTS_DIR  # noqa: E402
from feat_eng.registry import get_experiment  # noqa: E402

# ── Configuration ──────────────────────────────────────────────────────────────
DEFAULT_POT = "CM08"             # override via --pot (e.g. CM08_bf1h for the bf1h-trained artifact)
TARGET = "psd_r30"
BRAND = "PCC"
ACTIONABLE = "sep_motor_speed"
# The canonical notebooks/opt_algo/ optimizers read a single generic setpoint key, literally
# named 'desired_blaine' (target-agnostic: the predictor decides which target it scores). We put
# the psd_r30 setpoint under it, so the optimizer minimizes |setpoint - predicted psd_r30|.
DESIRED_KEY = "desired_blaine"

# Default optimizer module — same plug-and-play registry as 04_1. Override via --optimizer.
OPT_ALGO_MODULE = "opt_algo.03_grid_search_minimize"


def default_artifact_dir(pot):
    """Frozen-artifact location for a pot (matches model_train.py's --out-dir default)."""
    return RESULTS_DIR / pot / "optimization" / f"frozen_{TARGET}_{BRAND}"


# ── Frozen artifact loading ──────────────────────────────────────────────────
def load_frozen_model(artifact_dir):
    """Load the frozen model + metadata. Raises FileNotFoundError if either is missing."""
    artifact_dir = Path(artifact_dir)
    model_path = artifact_dir / "model.joblib"
    meta_path = artifact_dir / "metadata.json"

    missing = [str(p) for p in (model_path, meta_path) if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Frozen APC artifact not found — expected both files:\n"
            f"  {model_path}\n  {meta_path}\n"
            f"Missing: {missing}\n"
            "Run model_train.py (e.g. `python model_train.py --cm CM08 --target psd_r30 "
            "--brand PCC`) to produce the frozen artifact before running this script."
        )

    model = joblib.load(model_path)
    with open(meta_path) as f:
        metadata = json.load(f)

    _validate_metadata(metadata)
    return model, metadata


def _validate_metadata(metadata):
    """Fail fast on a malformed artifact so we never optimize on the wrong contract."""
    required = ["feature_order", "sep_motor_speed_bounds", "model_mae", "target"]
    missing = [k for k in required if k not in metadata]
    if missing:
        raise ValueError(f"metadata.json missing required keys: {missing}")
    if metadata["target"] != TARGET:
        raise ValueError(
            f"Artifact target is {metadata['target']!r}, but this script drives {TARGET!r}."
        )
    bounds = metadata["sep_motor_speed_bounds"]
    if not (isinstance(bounds, (list, tuple)) and len(bounds) >= 2):
        raise ValueError(f"sep_motor_speed_bounds must be [min, max(, step)]; got {bounds!r}")
    if ACTIONABLE not in metadata["feature_order"]:
        raise ValueError(f"{ACTIONABLE!r} absent from feature_order; cannot optimize it.")


# ── Predictor wrapper (mirrors 04_1_Optimization.py:build_predictor) ──────────
def build_predictor(model, feature_order, is_autogluon=False):
    """Wrap a trained model into a dict -> float predictor.

    Only keys in ``feature_order`` are read, so extra keys in the feature dict
    (e.g. 'brand', or the optimizer's bookkeeping) are ignored harmlessly.
    """
    if is_autogluon:
        def predictor(feature_dict):
            row = pd.DataFrame([{k: feature_dict[k] for k in feature_order if k in feature_dict}])
            return float(model.predict(row).iloc[0])
    else:
        def predictor(feature_dict):
            row = pd.DataFrame([{k: feature_dict[k] for k in feature_order}])
            return float(model.predict(row)[0])
    return predictor


# ── Core: Predict + Optimize ──────────────────────────────────────────────────
def recommend(current_features, desired_psd_r30, model, metadata, optimizer):
    """Run the single-step inverse optimization and return the recommendation."""
    feature_order = metadata["feature_order"]
    mae = float(metadata["model_mae"])
    bounds = list(metadata["sep_motor_speed_bounds"])
    lo, hi = float(bounds[0]), float(bounds[1])

    predictor = build_predictor(model, feature_order, metadata.get("is_autogluon", False))
    constraints = {ACTIONABLE: bounds}

    features = dict(current_features)
    features[DESIRED_KEY] = float(desired_psd_r30)

    result = optimizer.optimize(features, constraints, predictor)
    alternates = result.pop("_alternates", []) if isinstance(result, dict) else []

    def _post_process(candidate):
        speed = float(min(max(candidate[ACTIONABLE], lo), hi))
        scored = dict(candidate)
        scored[ACTIONABLE] = speed
        pred = predictor(scored)
        return {
            "sep_motor_speed": speed,
            "predicted_psd_r30": pred,
            "objective_error": abs(float(desired_psd_r30) - pred),
        }

    best = _post_process(result)

    return {
        "recommended_sep_motor_speed": best["sep_motor_speed"],
        "predicted_psd_r30": best["predicted_psd_r30"],
        "desired_psd_r30": float(desired_psd_r30),
        "objective_error": best["objective_error"],
        "confidence_interval": [best["predicted_psd_r30"] - mae,
                                best["predicted_psd_r30"] + mae],
        "model_mae": mae,
        "sep_motor_speed_bounds": [lo, hi],
        "alternatives": [_post_process(a) for a in alternates],
    }


def load_optimizer(module_name=OPT_ALGO_MODULE):
    """Load a plug-and-play optimizer module from opt_algo/ (same registry as 04_1)."""
    opt = importlib.import_module(module_name)
    if not hasattr(opt, "optimize"):
        raise ImportError(f"{module_name} does not expose an optimize() function.")
    return opt


# ── Demo / CLI ────────────────────────────────────────────────────────────────
def _build_demo_snapshot(metadata, pot):
    """Pull the most recent PCC row from the pot's merged dataset as a stand-in for a
    live 60-min snapshot. (In production the snapshot is built from Ignition tags.)"""
    cfg = get_config(pot)
    df = pd.read_csv(cfg["merged_data_path"])
    # Recompute the feat_eng columns the model was trained on (they are derived at runtime from
    # `hour`, not stored in the merged CSV). Must match metadata["feat_eng"] so feature_order resolves.
    fe_exp = metadata.get("feat_eng", "none")
    if fe_exp and str(fe_exp).lower() != "none":
        df, _, _ = get_experiment(fe_exp)["apply"](df)
    df_b = df[df["brand"] == BRAND].dropna(subset=[TARGET])
    if df_b.empty:
        raise RuntimeError(f"No {BRAND} rows with {TARGET} in {cfg['merged_data_path']}")
    row = df_b.iloc[-1]
    snapshot = {k: float(row[k]) for k in metadata["feature_order"]}
    snapshot["brand"] = BRAND
    observed_psd = float(row[TARGET])
    return snapshot, observed_psd


def main():
    parser = argparse.ArgumentParser(description="APC Predict + Optimize core (CM08 psd_r30).")
    parser.add_argument("--pot", default=DEFAULT_POT,
                        help="Pot whose frozen artifact + demo snapshot to use (e.g. CM08, CM08_bf1h).")
    parser.add_argument("--artifact-dir", default=None,
                        help="Directory holding model.joblib + metadata.json "
                             "(default: results/<pot>/optimization/frozen_psd_r30_PCC).")
    parser.add_argument("--desired-psd-r30", type=float, default=None,
                        help="Desired psd_r30 setpoint. Defaults to the demo row's own value.")
    parser.add_argument("--optimizer", default=OPT_ALGO_MODULE,
                        help="opt_algo module to use (e.g. opt_algo.03_grid_search_minimize).")
    args = parser.parse_args()

    artifact_dir = (args.artifact_dir
                    or os.environ.get("APC_ARTIFACT_DIR")
                    or str(default_artifact_dir(args.pot)))
    model, metadata = load_frozen_model(artifact_dir)
    optimizer = load_optimizer(args.optimizer)
    print(f"Loaded frozen artifact: target={metadata['target']}, "
          f"model_mae={metadata['model_mae']:.4f}, "
          f"bounds={metadata['sep_motor_speed_bounds']}, optimizer={args.optimizer}")

    snapshot, observed_psd = _build_demo_snapshot(metadata, args.pot)
    desired = args.desired_psd_r30 if args.desired_psd_r30 is not None else observed_psd
    print(f"Demo snapshot (latest {BRAND} row) — observed psd_r30={observed_psd:.3f}, "
          f"current {ACTIONABLE}={snapshot[ACTIONABLE]:.1f}")
    print(f"Desired psd_r30 setpoint: {desired:.3f}")

    out = recommend(snapshot, desired, model, metadata, optimizer)

    print("\n── Recommendation ──")
    print(f"  recommended {ACTIONABLE}: {out['recommended_sep_motor_speed']:.1f}")
    print(f"  predicted psd_r30:       {out['predicted_psd_r30']:.3f} "
          f"(desired {out['desired_psd_r30']:.3f}, obj err {out['objective_error']:.3f})")
    print(f"  CI (±MAE):               [{out['confidence_interval'][0]:.3f}, "
          f"{out['confidence_interval'][1]:.3f}]")
    print(f"  alternatives ({len(out['alternatives'])}):")
    for i, alt in enumerate(out["alternatives"], 1):
        print(f"    {i}. {ACTIONABLE}={alt['sep_motor_speed']:.1f} "
              f"-> psd_r30={alt['predicted_psd_r30']:.3f} "
              f"(obj err {alt['objective_error']:.3f})")


if __name__ == "__main__":
    main()
