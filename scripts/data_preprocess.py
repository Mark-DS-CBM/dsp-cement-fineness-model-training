#!/usr/bin/env python
"""Inference preprocessing — sensor + quality → merged hourly q75 backward-fill dataset.

This is the production path of ``notebooks/00_EDA_CM0X.py`` (§0, §1.1.6, §2.1, §2.2.3, §2.3),
stripped of all EDA / plot / overview cells. It loads a mill's raw sensor files, applies the
per-quarter mill-power Q75±tolerance band, cleans (99999→NaN, column drops, percentile cap,
dropna), aggregates sensor features over the backward-looking window ``[H - lag, H)``, attaches
the lab targets (``blaine`` / ``psd_r30``) at hour ``H``, and writes one merged CSV.

Output (locked decision: q75 + backward-fill 1h only):
  • data/<merged>_q75_bf{lag}_{CM}.csv   — canonical "latest" (the file pot_config globs for).
  • data/power_band_{CM}.json            — per-quarter mill-power Q75 band (the "power band").
  • data/archive/<stamp>/...             — timestamped history copies (canonical is always refreshed).

The three mills differ only in how labels arrive: CM04/CM07 carry forward-filled ``blaine`` /
``psd_r30`` in the sensor file (``label_mode="in_file"``); CM08 has no sensor-side brand/labels and
joins the Quality XLSX (``label_mode="quality_xlsx"``). Everything else (s_t_stamp, Q75 filter,
clean, backward aggregation, target cap, run-collapse) is shared.

Usage (conda env CBM):
  python data_preprocess.py --cm CM08
  python data_preprocess.py --cm CM04 --lag 1h --agg mean --power-tolerance 200
"""
import argparse
import json
import shutil
from datetime import datetime

import numpy as np
import pandas as pd

# Single-source the data root from the vendored config (editable-installed package).
from config.pot_config import DATA_DIR  # noqa: E402

ORIGINAL_DIR = DATA_DIR / "original"

LAG_TIMEDELTAS = {
    "1h": pd.Timedelta(hours=1),
    "30m": pd.Timedelta(minutes=30),
    "15m": pd.Timedelta(minutes=15),
}

# ── Per-CM specs (mirrors each notebook's §0 load + §2.1 clean lists) ──────────
CM_SPECS = {
    "CM04": {
        "sensor_dir": "process_parameter_CM04_20240101-20260331",
        # XLSX/CSV files in this dir are read by extension; consistency is checked at load.
        "column_renames": {},
        "power_col": "mill drive_power",
        "special_code_cols": ["fresh_feed", "sep_motor_speed", "sep fan_damper",
                              "sep_motor_amp", "rp drive-m1_motor_amp-p"],
        "drop_cols": ["Hemi in CM", "fresh_feed", "rp drive-m1_motor_amp-p",
                      "cement_temp-pout", "mill drive_power"],
        "cap_mode": "per_brand",
        "cap_cols": ["sep_motor_speed", "sep fan_damper", "be-mill_out_power",
                     "sep_motor_amp", "sep fan_amp", "cement_temp-wsp"],
        "label_mode": "in_file",
        "brands_excluded_eda": ["ELPre"],
        "out_name": lambda s, e, lag: f"merged_sensor_03_quality_{s}-{e}_q75_bf{lag}_CM04.csv",
    },
    "CM07": {
        "sensor_dir": "process_parameter_CM07_202408-202603",
        "column_renames": {"hemi in cm": "Hemi in CM"},
        "power_col": "mill drive_power",
        "special_code_cols": ["fresh_feed", "sep_motor_speed", "sep fan_damper",
                              "sep_motor_amp", "rp drive-m1_amp", "rp drive-m2_amp"],
        "drop_cols": ["Hemi in CM", "fresh_feed", "cement_temp-pout", "mill drive_power"],
        "cap_mode": "per_brand",
        "cap_cols": ["sep_motor_speed", "sep fan_damper", "be-mill_out_amp",
                     "sep_motor_amp", "sep fan_amp", "rp drive-m1_amp", "rp drive-m2_amp",
                     "cement_temp-mv", "cement_temp-wsp"],
        "label_mode": "in_file",
        "brands_excluded_eda": [],
        "out_name": lambda s, e, lag: f"merged_sensor_02_{s}-{e}_lab_bf{lag}_CM07.csv",
    },
    "CM08": {
        "sensor_dir": "process_parameter_CM08_20240707-20260331",
        "column_renames": {"mill_motor_power-mv": "mill_motor_power",
                           "be_mill outlet_power-mv": "be_mill outlet_power"},
        "power_col": "mill_motor_power",
        "special_code_cols": ["fresh_feed", "sep_motor_speed", "sep fan_damper",
                              "sep_motor_amp", "rp drive-m1_amp", "rp drive-m2_amp"],
        "drop_cols": ["fresh_feed", "mill_motor_power"],
        "cap_mode": "global",      # CM08 sensor has no brand column → global cap
        "cap_cols": None,          # None ⇒ cap all numeric columns
        "label_mode": "quality_xlsx",
        "quality_file": "SKK - CM8 Quality (2024-2026).xlsx",
        "quality_renames": {"Date": "date", "Brand": "brand",
                            "Blaine(cm2/g)": "quality_blaine", "PSD R30(%)": "quality_psd_r30"},
        "brands_excluded_eda": [],
        "out_name": lambda s, e, lag: f"merged_sensor_03_quality_xlsx_{s}-{e}_q75_bf{lag}_CM08.csv",
    },
}

TARGET_COLS = ["blaine", "psd_r30"]


# ── Loaders ───────────────────────────────────────────────────────────────────
def load_sensor(spec):
    """Concat all sensor files in the mill's dir; verify columns line up after renames."""
    sensor_dir = ORIGINAL_DIR / spec["sensor_dir"]
    files = sorted([p for p in sensor_dir.iterdir()
                    if p.suffix.lower() in (".csv", ".xlsx")
                    and "quality" not in p.name.lower()])
    if not files:
        raise FileNotFoundError(f"No sensor files found in {sensor_dir}")

    file_dfs, file_columns = [], {}
    print("=== Loading sensor files ===")
    for fpath in files:
        df = pd.read_csv(fpath) if fpath.suffix.lower() == ".csv" else pd.read_excel(fpath)
        if spec["column_renames"]:
            df = df.rename(columns=spec["column_renames"])
        file_columns[fpath.name] = list(df.columns)
        file_dfs.append(df)
        print(f"  ✅ {fpath.name}: {len(df):,} rows, {len(df.columns)} cols")

    ref = file_columns[files[0].name]
    for name, cols in file_columns.items():
        if cols != ref:
            missing, extra = set(ref) - set(cols), set(cols) - set(ref)
            raise ValueError(f"Column mismatch in {name}: missing={missing}, extra={extra}")

    df_raw = pd.concat(file_dfs, ignore_index=True)
    print(f"Merged sensor shape: {df_raw.shape}")
    return df_raw


def load_quality_cm08(spec):
    """CM08 label/brand source: Quality XLSX → one (hour, brand) row with blaine/psd_r30."""
    path = ORIGINAL_DIR / spec["sensor_dir"] / spec["quality_file"]
    dfq = pd.read_excel(path).rename(columns=spec["quality_renames"])
    dfq["date"] = pd.to_datetime(dfq["date"])
    dfq["hour"] = dfq["date"].dt.floor("h")
    dfq = dfq.dropna(subset=["quality_blaine", "quality_psd_r30"], how="all")
    dfq = (dfq.groupby(["hour", "brand"])[["quality_blaine", "quality_psd_r30"]]
              .first().reset_index()
              .rename(columns={"quality_blaine": "blaine", "quality_psd_r30": "psd_r30"}))
    print(f"Quality merge keys: {len(dfq):,} (hour, brand) rows")
    return dfq


# ── Shared core ────────────────────────────────────────────────────────────────
def add_s_t_stamp(df_raw):
    """Recover 10-second resolution: cumulative 10s offsets within each t_stamp group."""
    df_raw = df_raw.copy()
    df_raw["t_stamp_parsed"] = pd.to_datetime(df_raw["t_stamp"], errors="coerce")
    df_raw["s_t_stamp"] = df_raw["t_stamp_parsed"] + pd.to_timedelta(
        df_raw.groupby("t_stamp_parsed").cumcount() * 10, unit="s")
    return df_raw


def quarter_q75_filter(df_raw, power_col, tolerance):
    """Keep rows whose mill power is within Q75±tolerance of their calendar quarter."""
    df = df_raw.copy()
    df["quarter"] = df["s_t_stamp"].dt.to_period("Q")
    parts, q75_by_quarter = [], {}
    print(f"=== Per-quarter {power_col} Q75±{tolerance} filter ===")
    for q in sorted(df["quarter"].unique()):
        chunk = df[df["quarter"] == q]
        q75 = chunk[power_col].quantile(0.75)
        q75_by_quarter[str(q)] = float(q75)
        lo, hi = q75 - tolerance, q75 + tolerance
        kept = chunk[(chunk[power_col] >= lo) & (chunk[power_col] <= hi)]
        print(f"  {q}: Q75={q75:.1f}, range=[{lo:.1f}, {hi:.1f}], "
              f"pass={len(kept):,}/{len(chunk):,} ({len(kept)/len(chunk)*100:.1f}%)")
        parts.append(kept)
    df_domain = pd.concat(parts, ignore_index=True).drop(columns=["quarter"])
    print(f"Rows retained: {len(df_domain):,} / {len(df_raw):,} "
          f"({len(df_domain)/len(df_raw)*100:.1f}%)")
    return df_domain, q75_by_quarter


def clean_sensor(df_domain, spec):
    """99999→NaN, drop unused/power columns, percentile cap (global or per-brand), dropna."""
    df = df_domain.copy()
    for col in [c for c in spec["special_code_cols"] if c in df.columns]:
        df.loc[df[col] == 99999, col] = np.nan

    drop = [c for c in (spec["drop_cols"] + ["t_stamp_parsed"]) if c in df.columns]
    df = df.drop(columns=drop)
    print(f"Dropped columns: {drop}")

    if spec["cap_mode"] == "global":
        cap_cols = list(df.select_dtypes("number").columns)
        for col in cap_cols:
            df[col] = df[col].clip(df[col].quantile(0.01), df[col].quantile(0.99))
        print(f"Capped (global 1st–99th): {len(cap_cols)} numeric columns")
    else:  # per_brand
        cap_cols = [c for c in spec["cap_cols"] if c in df.columns]
        for col in cap_cols:
            for b in df["brand"].dropna().unique():
                m = df["brand"] == b
                bd = df.loc[m, col]
                df.loc[m, col] = bd.clip(bd.quantile(0.01), bd.quantile(0.99))
        print(f"Capped (per-brand 1st–99th): {cap_cols}")

    before = len(df)
    df = df.dropna()
    print(f"dropna: {before:,} → {len(df):,}")
    return df


def collapse_target_runs(df):
    """Drop consecutive rows that repeat the same (brand, blaine, psd_r30) — forward-fill runs."""
    df = df.sort_values(["brand", "hour"]).reset_index(drop=True)
    is_dup = ((df["brand"] == df["brand"].shift()) &
              (df["blaine"] == df["blaine"].shift()) &
              (df["psd_r30"] == df["psd_r30"].shift()))
    return df.loc[~is_dup].reset_index(drop=True)


def build_backward(df_clean, lag_td, agg, spec, df_quality_merge=None):
    """Aggregate sensor features over [H - lag, H), attach targets at H, cap, collapse runs."""
    label_hour = df_clean["s_t_stamp"].dt.floor("h") + pd.Timedelta(hours=1)
    within = (label_hour - df_clean["s_t_stamp"]) <= lag_td
    feat = df_clean.loc[within].copy()
    feat["hour"] = label_hour.loc[within]

    if spec["label_mode"] == "quality_xlsx":
        # No sensor-side brand: aggregate by hour, then join brand + targets from Quality.
        sensor_cols = [c for c in feat.select_dtypes("number").columns if c != "hour"]
        df_feat = feat.groupby("hour")[sensor_cols].agg(agg)
        df_std = feat.groupby("hour")[sensor_cols].std()
        df_std.columns = [f"{c}_std" for c in sensor_cols]
        df_feat = df_feat.join(df_std).reset_index()
        df_m = df_feat.merge(
            df_quality_merge[["hour", "brand", "blaine", "psd_r30"]], on="hour", how="inner")
    else:  # in_file: sensor file already carries brand + forward-filled targets
        sensor_num = [c for c in feat.select_dtypes("number").columns if c != "hour"]
        sensor_feature_cols = [c for c in sensor_num if c not in TARGET_COLS]
        df_feat = feat.groupby(["hour", "brand"])[sensor_feature_cols].agg(agg)
        df_std = feat.groupby(["hour", "brand"])[sensor_feature_cols].std()
        df_std.columns = [f"{c}_std" for c in sensor_feature_cols]
        df_feat = df_feat.join(df_std).reset_index()
        # Target AT hour H from the natural-hour grid (constant within hour, per EDA §1.3.0).
        tgt = (df_clean.assign(hour=df_clean["s_t_stamp"].dt.floor("h"))
               .groupby(["hour", "brand"])[TARGET_COLS].first().reset_index())
        df_m = df_feat.merge(tgt, on=["hour", "brand"], how="inner")

    # Per-brand 1st–99th cap on lab targets.
    for col in TARGET_COLS:
        for b in df_m["brand"].dropna().unique():
            bm = df_m["brand"] == b
            bd = df_m.loc[bm, col].dropna()
            if len(bd) < 10:
                continue
            df_m.loc[bm, col] = df_m.loc[bm, col].clip(bd.quantile(0.01), bd.quantile(0.99))

    n0 = len(df_m)
    df_m = collapse_target_runs(df_m)
    if spec["brands_excluded_eda"]:
        df_m = df_m[~df_m["brand"].isin(spec["brands_excluded_eda"])].reset_index(drop=True)
    print(f"build_backward: {n0:,} → {len(df_m):,} rows (collapse+brand filter), "
          f"brands={df_m['brand'].value_counts().to_dict()}")
    return df_m


# ── Versioned output ────────────────────────────────────────────────────────────
def write_versioned(obj, filename, stamp, kind):
    """Write to canonical data/<filename> AND data/archive/<stamp>/<filename>.

    ``kind`` is "csv" (obj is a DataFrame) or "json" (obj is a dict).
    """
    canonical = DATA_DIR / filename
    archive = DATA_DIR / "archive" / stamp
    archive.mkdir(parents=True, exist_ok=True)

    if kind == "csv":
        obj.to_csv(canonical, index=False)
    else:
        canonical.write_text(json.dumps(obj, indent=2))
    shutil.copy2(canonical, archive / filename)
    print(f"  ✅ {filename}  →  {canonical}  (+ archive/{stamp}/)")


def main():
    ap = argparse.ArgumentParser(description="Preprocess sensor+quality → merged q75 bf dataset.")
    ap.add_argument("--cm", required=True, choices=sorted(CM_SPECS), help="Cement mill.")
    ap.add_argument("--lag", default="1h", choices=sorted(LAG_TIMEDELTAS),
                    help="Backward-fill window (default 1h).")
    ap.add_argument("--agg", default="mean", choices=["mean", "median"],
                    help="Hourly aggregation method (default mean).")
    ap.add_argument("--power-tolerance", type=int, default=200,
                    help="Mill-power Q75 ± tolerance band (default 200).")
    args = ap.parse_args()

    spec = CM_SPECS[args.cm]
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    print(f"\n=== Preprocess {args.cm} (lag={args.lag}, agg={args.agg}, "
          f"tol={args.power_tolerance}) — run {stamp} ===\n")

    df_raw = load_sensor(spec)
    df_raw = add_s_t_stamp(df_raw)
    df_domain, q75_by_quarter = quarter_q75_filter(
        df_raw, spec["power_col"], args.power_tolerance)

    # Power-band sidecar (the only downstream record of the trained-domain mill-power band).
    power_band = {"pot": args.cm, "power_col": spec["power_col"],
                  "tolerance": args.power_tolerance, "per_quarter": q75_by_quarter}
    write_versioned(power_band, f"power_band_{args.cm}.json", stamp, "json")

    df_quality_merge = load_quality_cm08(spec) if spec["label_mode"] == "quality_xlsx" else None
    df_clean = clean_sensor(df_domain, spec)

    df_merged = build_backward(
        df_clean, LAG_TIMEDELTAS[args.lag], args.agg, spec, df_quality_merge)

    s = df_merged["hour"].min().strftime("%Y%m%d")
    e = df_merged["hour"].max().strftime("%Y%m%d")
    write_versioned(df_merged, spec["out_name"](s, e, args.lag), stamp, "csv")

    print(f"\n✅ Done: {len(df_merged):,} rows, {df_merged.shape[1]} cols, "
          f"window {s} → {e}.")


if __name__ == "__main__":
    main()
