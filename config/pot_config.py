"""Per-pot configuration for the cement-fineness modelling pipeline.

Add a new pot by inserting a key in POT_CONFIGS. Every downstream notebook reads
its data paths, brand lists, and result directory through this module so that
swapping ``POT = "CM04"`` to ``POT = "CM08"`` (or future pots) at the top of a
notebook redirects all I/O without further edits.
"""
from pathlib import Path

# Standalone-repo layout: this module lives at <repo>/config/pot_config.py, so the repo root is two
# parents up. DATA_DIR and RESULTS_DIR both sit directly under the repo root here (vendored copy —
# the research repo's version anchored RESULTS_DIR at notebooks/results instead).
REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
RESULTS_DIR = REPO_ROOT / "results"

POT_CONFIGS = {
    "CM04": {
        # --- Data sources ---
        # Main dataset for CM04 now uses the sensor-as-label-source `_03_` family produced by
        # 00_EDA_CM04 §2.3 (sensor file's forward-filled `blaine`/`psd_r30` columns are the
        # canonical labels per §1.3 / §1.4; bit-identical to the lab CSV on overlap). Expands the
        # merged window from the lab-CSV-bottlenecked 316 rows to several thousand. The shorter
        # CSV (CM04-window, 88 rows) stays on the legacy `_01_` family and is used by 03_2/03_3
        # for the drift-window subset comparison.
        # Update the date range below to match the actual min/max printed by
        # 00_EDA_CM04.py §2.3 after the first run.
        "merged_data_path":     DATA_DIR / "merged_sensor_03_quality_20241001-20260401_q75_CM04.csv",
        "merged_data_shorter":  DATA_DIR / "merged_sensor_01_20250605-20251023_lab_CM04.csv",
        "merged_data_expanded": DATA_DIR / "merged_sensor_03_quality_20241001-20260401_q75_CM04.csv",
        "sensor_dir":           DATA_DIR / "original" / "process_parameter_CM04_20240101-20260331",
        "lab_csv_path":         DATA_DIR / "original" / "SKK-New-CM04&08.csv",
        "lab_no_filter":        "CM4",      # SKK-New-CM04&08.csv mixes CM4 and CM8 rows

        # --- Brand metadata ---
        "brand_list":           ["PCC", "Type III", "TI", "SC", "ELPre", "EL"],
        "brands_of_interest":   ["PCC"],     # 01_1, 03_2, 03_3, 03_4
        "brands_excluded":      ["ELPre",  "Type III", "EL"],                # 01_2, 01_3, 02_1, 03_3
        "has_type_iii":         True,                     # gates 03_1
        "optimization_brands":  ["PCC"],                  # 04_1

        # --- Features ---
        "target_cols":          ["blaine", "psd_r30"],
        "actionable_params":    ["sep_motor_speed"],
        "control_params":       ["sep fan_damper", "be-mill_out_power",
                                 #"sep_motor_amp",
                                 "sep fan_amp", "cement_temp-mv", "cement_temp-wsp"],
        "exclude_from_features": ["blaine", "psd_r30", "hour"],
        # Canonical model-input allowlist. Every downstream notebook (01_*, 02_*,
        # 03_2, 03_3, 04_*) restricts itself to these columns so feature sets are
        # consistent across pots.
        # Invariant: set(model_features) == {"brand"} | set(actionable_params) | set(control_params)
        "model_features": [
            "brand",
            "sep_motor_speed", "sep fan_damper", "be-mill_out_power",
            #"sep_motor_amp",
            "sep fan_amp",
            "cement_temp-mv", "cement_temp-wsp",
            "sep_motor_speed_std", "sep fan_damper_std", "be-mill_out_power_std",
            #"sep_motor_amp_std",
            "sep fan_amp_std",
            "cement_temp-mv_std", "cement_temp-wsp_std",
        ],
        # Model used by 02_1_FeatEng and 04_1_Optimization (and the queue script)
        # for this pot. MODEL_NAME env var overrides this. Must be one of
        # _modeling_utils.VALID_MODELS (KNN, XGBoost, LightGBM, CatBoost) — or
        # "AutoGluon" via env-var override (04_1-only path).
        "model_name":           "XGBoost",
        # Description of how this pot's merged CSV was prepared. Read verbatim into
        # the MLflow manifest so the setup_hash changes when the CSV is regenerated
        # with different settings.
        "data_recipe": {
            "aggregation": "mean_hourly",
            "mill_power_filter": "q75_minus_200_plus_200",
            "outlier_capping": "1st-99th per brand",
        },

        # --- Raw-sensor schema (used by 03_4 which loads the unmerged sensor files) ---
        "mill_power_col":       "mill drive_power",
        "brand_col_in_sensor":  "brand",                 # CM04 sensor has a `brand` column

        # --- Date cutoffs (CM04 post-drift regime, see AGENTS.md) ---
        "training_start":       "2025-06-01",
        "drift_period_cutoff":  "2025-11-01",

        # --- Reference MAE per training regime (CM04 AutoGluon baseline, used by 03_2) ---
        # Source: results/CM04/hyperparameter_tuning/tuned_{target}_model_comparison_results.csv
        "reference_mae": {
            "expanded": {
                "blaine":  {"PCC": 84.8, "Type III": 83.7},
                "psd_r30": {"PCC": 2.07, "Type III": 1.83},
            },
            "shorter": {
                "blaine":  {"PCC": 62.3, "Type III": 66.0},
                "psd_r30": {"PCC": 1.97, "Type III": 0.31},
            },
        },
    },
    "CM08": {
        # Main dataset for CM08 now uses the Quality-XLSX-merged `_03_` family produced by
        # 00_EDA_CM08 §2.3 (Quality XLSX is the canonical label + brand source per §1.3 / §1.4;
        # bit-identical to the lab CSV on overlap). Expands the merged window from the
        # lab-CSV-bottlenecked 383 rows to several thousand. The shorter CSV (CM04-window)
        # is used only by 03_2/03_3 to compare regimes and stays on the legacy `_01_` family.
        "merged_data_path":     DATA_DIR / "merged_sensor_03_quality_xlsx_20240707-20260331_q75_CM08.csv",
        "merged_data_shorter":  DATA_DIR / "merged_sensor_01_20250605-20251023_lab_CM08.csv",
        "merged_data_expanded": DATA_DIR / "merged_sensor_03_quality_xlsx_20240707-20260331_q75_CM08.csv",
        "sensor_dir":           DATA_DIR / "original" / "process_parameter_CM08_20240707-20260331",
        "lab_csv_path":         DATA_DIR / "original" / "SKK-New-CM04&08.csv",
        "lab_no_filter":        "CM8",

        "brand_list":           ["PCC", "SC", "EL", "EL_CH"],
        "brands_of_interest":   ["PCC"],
        "brands_excluded":      ["EL_CH", "EL", "SC"],                # smallest sample
        "has_type_iii":         False,
        "optimization_brands":  ["PCC"],

        "target_cols":          ["blaine", "psd_r30"],
        "actionable_params":    ["sep_motor_speed"],
        # Control set. The first six columns are the CM04-parity channels; the
        # last four (rp drive-m1/m2_amp, Mill_Reject, Mill_dPres(mmAq)) are
        # CM08-only and included here because they are part of CM08's
        # operational instrumentation. Mill_Water Spray_Flow Rate-MV remains
        # excluded.
        "control_params":       ["sep fan_damper", "be_mill outlet_power",
                                 # "sep_motor_amp",
                                 "sep fan_amp", "mill_outlet_temp-mv", "mill_outlet_temp-wsp",
                                 "rp drive-m1_amp", "rp drive-m2_amp"],
        "exclude_from_features": ["blaine", "psd_r30", "hour"],
        # CM08 model-input allowlist. The first six numeric columns map 1:1 onto
        # CM04's `model_features` (same physical quantities); the last four are
        # CM08-only channels added for richer modelling.
        # Invariant: set(model_features) == {"brand"} | set(actionable_params) | set(control_params)
        "model_features": [
            "brand",
            "sep_motor_speed", "sep fan_damper", "be_mill outlet_power",
            # "sep_motor_amp",
            "sep fan_amp",
            "mill_outlet_temp-mv", "mill_outlet_temp-wsp",
            "rp drive-m1_amp", "rp drive-m2_amp",
            "sep_motor_speed_std", "sep fan_damper_std", "be_mill outlet_power_std",
            # "sep_motor_amp_std",
            "sep fan_amp_std",
            "mill_outlet_temp-mv_std", "mill_outlet_temp-wsp_std",
            "rp drive-m1_amp_std", "rp drive-m2_amp_std",
        ],
        # Model used by 02_1_FeatEng and 04_1_Optimization (and the queue script)
        # for this pot. MODEL_NAME env var overrides this.
        "model_name":           "CatBoost",
        # Description of how this pot's merged CSV was prepared. Read verbatim into
        # the MLflow manifest so the setup_hash changes when the CSV is regenerated
        # with different settings.
        "data_recipe": {
            "aggregation": "mean_hourly",
            "mill_power_filter": "q75_minus_200_plus_200",
            "outlier_capping": "1st-99th per brand",
        },

        # Raw-sensor schema
        "mill_power_col":       "mill_motor_power",
        "brand_col_in_sensor":  None,                     # CM08 sensor has no `brand` column

        "training_start":       None,                     # TBD pending CM08 drift study
        "drift_period_cutoff":  "2025-11-01",             # matches CM04; P3 = 2025-06-01 .. 2025-10-31

        "reference_mae":        None,                     # populate after first CM08 baseline run
    },
    "CM07": {
        # CM07 labels (`blaine`, `psd_r30`) live in the sensor files themselves as
        # forward-filled lab values at 10-second cadence; there is no separate hourly
        # lab CSV (SKK-New-CM04&08.csv has only CM4/CM8 rows). The merged dataset
        # therefore covers the full sensor window (Aug 2024 – Mar 2026).
        # Update the date range below to match the actual min/max printed by
        # 00_EDA_CM07.py §2.3 after the first run.
        "merged_data_path":     DATA_DIR / "merged_sensor_02_20240801-20260331_lab_CM07.csv",
        "merged_data_shorter":  DATA_DIR / "merged_sensor_02_20240801-20260331_lab_CM07.csv",
        "merged_data_expanded": DATA_DIR / "merged_sensor_02_20240801-20260331_lab_CM07.csv",
        "sensor_dir":           DATA_DIR / "original" / "process_parameter_CM07_202408-202603",
        "lab_csv_path":         None,
        "lab_no_filter":        None,

        # Populate brand_list / brands_of_interest / brands_excluded after the first
        # run of §1.2 (brand distribution) and §3.2 (post-merge brand counts).
        "brand_list":           ['PCC', 'EL', 'SC'],
        "brands_of_interest":   ["PCC"],
        "brands_excluded":      ["EL", "SC"],
        "has_type_iii":         False,
        "optimization_brands":  ["PCC"],

        "target_cols":          ["blaine", "psd_r30"],
        "actionable_params":    ["sep_motor_speed"],
        # CM07-specific control set. `be-mill_out_amp` is the CM07 analog of CM04's
        # `be-mill_out_power` / CM08's `be_mill outlet_power`. Both roller-press
        # drives are kept (CM08 convention).
        "control_params":       ["sep fan_damper", "be-mill_out_amp", 
                                 #"sep_motor_amp",
                                 "sep fan_amp", "cement_temp-mv", "cement_temp-wsp",
                                 "rp drive-m1_amp", "rp drive-m2_amp"],
        "exclude_from_features": ["blaine", "psd_r30", "hour"],
        "model_features": [
            "brand",
            "sep_motor_speed", "sep fan_damper", "be-mill_out_amp",
            #"sep_motor_amp",
            "sep fan_amp",
            "cement_temp-mv", "cement_temp-wsp",
            "rp drive-m1_amp", "rp drive-m2_amp",
            "sep_motor_speed_std", "sep fan_damper_std", "be-mill_out_amp_std",
            #"sep_motor_amp_std",
            "sep fan_amp_std",
            "cement_temp-mv_std", "cement_temp-wsp_std",
            "rp drive-m1_amp_std", "rp drive-m2_amp_std",
        ],
        # Model used by 02_1_FeatEng and 04_1_Optimization (and the queue script)
        # for this pot. MODEL_NAME env var overrides this.
        "model_name":           "LightGBM",
        "data_recipe": {
            "aggregation": "mean_hourly",
            "mill_power_filter": "q75_minus_200_plus_200",
            "outlier_capping": "1st-99th per brand",
        },

        "mill_power_col":       "mill drive_power",
        "brand_col_in_sensor":  "brand",

        "training_start":       None,
        "drift_period_cutoff":  "2025-11-01",

        "reference_mae":        None,
    },
}

# CM08_off75: CM08 with the per-quarter Q75±200 mill_motor_power band trim skipped
# (built by 00_EDA_CM08 §2.3.1). Defined here, after POT_CONFIGS, so it can spread
# `CM08`'s entry and override only the recipe-relevant fields. `test_set_source_pot`
# encodes the value-identity coupling: the modeling notebook derives the test split
# from the source pot's `merged_data_path` and uses off75 rows for training only.
POT_CONFIGS["CM08_off75"] = {
    **POT_CONFIGS["CM08"],
    "merged_data_path": DATA_DIR / "merged_sensor_03_quality_xlsx_20240707-20260331_off75_CM08.csv",
    "data_recipe": {
        **POT_CONFIGS["CM08"]["data_recipe"],
        "mill_power_filter": "none",
    },
    "test_set_source_pot": "CM08",
}

# CM07_off75: CM07 with the per-quarter Q75±200 mill drive_power band trim skipped
# (built by 00_EDA_CM07 §2.3.1). Same coupling pattern as CM08_off75: the modeling
# notebook activates this pot with `POT = "CM07_off75"` and derives the test split
# from the source pot's `merged_data_path` (CM07's Q75-filtered CSV).
POT_CONFIGS["CM07_off75"] = {
    **POT_CONFIGS["CM07"],
    "merged_data_path": DATA_DIR / "merged_sensor_02_20240801-20260331_lab_CM07_off75.csv",
    "data_recipe": {
        **POT_CONFIGS["CM07"]["data_recipe"],
        "mill_power_filter": "none",
    },
    "test_set_source_pot": "CM07",
}

# CM04_off75: CM04 with the per-quarter Q75±200 mill drive_power band trim skipped
# (built by 00_EDA_CM04 §2.3.1). Same coupling pattern as CM07_off75 / CM08_off75: the
# modeling notebook activates this pot with `POT = "CM04_off75"` and derives the test
# split from the source pot's `merged_data_path` (CM04's Q75-filtered CSV). Update the
# date range below to match the actual min/max printed by 00_EDA_CM04.py §2.3.1.
POT_CONFIGS["CM04_off75"] = {
    **POT_CONFIGS["CM04"],
    "merged_data_path": DATA_DIR / "merged_sensor_03_quality_20241001-20260401_off75_CM04.csv",
    "data_recipe": {
        **POT_CONFIGS["CM04"]["data_recipe"],
        "mill_power_filter": "none",
    },
    "test_set_source_pot": "CM04",
}

# --- Backward-fill lag-variant pots (1h / 30m / 15m) ---
# 00_EDA_CM0X.py §2.2/§2.3 now align features to the backward-looking window [H - lag, H)
# (resolving the old forward-looking TODO) and emit, per lag, a Q75 + an off75 CSV carrying a
# `_bf{lag}_` token. We register a pot per (mill, lag, filter): the q75-bf pot is the canonical
# data source, and the off75-bf pot trains on the wider regime while deriving its test split from
# the matching q75-bf pot (`test_set_source_pot`). The forward-looking CM04/CM07/CM08 (+_off75)
# pots above are left untouched so prior artifacts stay reproducible.
#
# Each CSV path is resolved by glob so the pot tracks the actual data-derived filename regardless
# of the exact date range; before the EDA has been run, it falls back to the base pot's date tokens.
_BF_LAGS = ["bf1h", "bf30m", "bf15m"]

# (q75 glob, q75 fallback, off75 glob, off75 fallback) per mill, templated on {lag}.
_BF_SPECS = {
    "CM04": (
        "merged_sensor_03_quality_*_q75_{lag}_CM04.csv",
        "merged_sensor_03_quality_20241001-20260401_q75_{lag}_CM04.csv",
        "merged_sensor_03_quality_*_off75_{lag}_CM04.csv",
        "merged_sensor_03_quality_20241001-20260401_off75_{lag}_CM04.csv",
    ),
    "CM07": (
        "merged_sensor_02_*_lab_{lag}_CM07.csv",
        "merged_sensor_02_20240801-20260331_lab_{lag}_CM07.csv",
        "merged_sensor_02_*_lab_{lag}_CM07_off75.csv",
        "merged_sensor_02_20240801-20260331_lab_{lag}_CM07_off75.csv",
    ),
    "CM08": (
        "merged_sensor_03_quality_xlsx_*_q75_{lag}_CM08.csv",
        "merged_sensor_03_quality_xlsx_20240707-20260331_q75_{lag}_CM08.csv",
        "merged_sensor_03_quality_xlsx_*_off75_{lag}_CM08.csv",
        "merged_sensor_03_quality_xlsx_20240707-20260331_off75_{lag}_CM08.csv",
    ),
}


def _resolve_bf(glob_pat: str, fallback_name: str) -> Path:
    """Latest data CSV matching ``glob_pat``; the constructed fallback if none exist yet."""
    hits = sorted(DATA_DIR.glob(glob_pat))
    return hits[-1] if hits else DATA_DIR / fallback_name


for _mill, (_q_glob, _q_fb, _o_glob, _o_fb) in _BF_SPECS.items():
    for _lag in _BF_LAGS:
        _q75_csv = _resolve_bf(_q_glob.format(lag=_lag), _q_fb.format(lag=_lag))
        _off75_csv = _resolve_bf(_o_glob.format(lag=_lag), _o_fb.format(lag=_lag))
        _align = f"backward_fill_{_lag[2:]}"  # e.g. "backward_fill_1h"

        POT_CONFIGS[f"{_mill}_{_lag}"] = {
            **POT_CONFIGS[_mill],
            "merged_data_path":     _q75_csv,
            "merged_data_expanded": _q75_csv,
            "data_recipe": {
                **POT_CONFIGS[_mill]["data_recipe"],
                "temporal_alignment": _align,
            },
        }
        POT_CONFIGS[f"{_mill}_{_lag}_off75"] = {
            **POT_CONFIGS[_mill],
            "merged_data_path":     _off75_csv,
            "merged_data_expanded": _off75_csv,
            "data_recipe": {
                **POT_CONFIGS[_mill]["data_recipe"],
                "mill_power_filter":  "none",
                "temporal_alignment": _align,
            },
            "test_set_source_pot": f"{_mill}_{_lag}",
        }


def _validate_config(pot: str, cfg: dict) -> None:
    """Fail fast if model_features and the actionable/control split disagree.

    The optimizer (04_1) builds its feature set from `actionable_params + control_params`
    while the modelling notebooks (01_*, 02_*, 03_2/3) build theirs from `model_features`.
    If the two ever drift apart, the optimizer would be predicting on a different
    feature set than 01_2 trained on. This check makes that impossible.
    """
    expected = {"brand"} | set(cfg["actionable_params"]) | set(cfg["control_params"])
    actual = set(cfg["model_features"])
    actual_base = {f for f in actual if not f.endswith("_std")}
    actual_std_bases = {f[:-4] for f in actual if f.endswith("_std")}
    if expected != actual_base:
        only_in_model = actual_base - expected
        only_in_split = expected - actual_base
        raise AssertionError(
            f"[{pot}] model_features and (brand + actionable_params + control_params) disagree.\n"
            f"  Only in model_features:                  {sorted(only_in_model)}\n"
            f"  Only in (actionable + control + brand):  {sorted(only_in_split)}"
        )
    orphan_std = actual_std_bases - expected
    if orphan_std:
        raise AssertionError(
            f"[{pot}] _std features with no corresponding base feature in model_features: "
            f"{sorted(orphan_std)}"
        )


def get_config(pot: str) -> dict:
    """Return the config dict for ``pot`` (e.g. ``"CM04"``). Raises KeyError if unknown."""
    if pot not in POT_CONFIGS:
        raise KeyError(f"Unknown pot '{pot}'. Available: {list(POT_CONFIGS)}")
    cfg = POT_CONFIGS[pot]
    _validate_config(pot, cfg)
    return cfg


def prepare_results_dir(pot: str, name: str) -> Path:
    """Return ``results/<POT>/<name>/`` and create it if missing."""
    p = RESULTS_DIR / pot / name
    p.mkdir(parents=True, exist_ok=True)
    return p
