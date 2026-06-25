"""
Feature engineering experiment: Seasonal features.
Adds datetime-based cyclic features and Thai seasonal flags.
"""

import numpy as np
import pandas as pd

EXPERIMENT_NAME = "04_seasonal"
DESCRIPTION = "Extracts cyclic datetime and Thai seasonal indicators"
AFFECTS_ACTIONABLE = False  # Pure time-of-arrival features — independent of actionable params


def apply(df):
    """Apply feature engineering transformations to the DataFrame.

    Args:
        df: Raw DataFrame after loading and timestamp sorting.

    Returns:
        df:                 Transformed DataFrame with engineered features.
        new_feature_names:  List of newly created feature names (for tracking).
        experiment_name:    Name of this experiment (used for results subfolder).
    """
    df = df.copy()
    new_feature_names = []

    # ──────────────────────────────────────────────────────────
    # Feature engineering code
    # ──────────────────────────────────────────────────────────

    if 'hour' in df.columns:
        dt_series = pd.to_datetime(df['hour'], format="%Y-%m-%d %H:%M:%S")
        hour_val = dt_series.dt.hour
        doy = dt_series.dt.dayofyear
        month = dt_series.dt.month

        # Cyclic Hour of Day
        df['FE_HOUR_SIN'] = np.sin(2 * np.pi * hour_val / 24.0)
        df['FE_HOUR_COS'] = np.cos(2 * np.pi * hour_val / 24.0)

        # Cyclic Day of Year
        df['FE_DOY_SIN'] = np.sin(2 * np.pi * doy / 365.0)
        df['FE_DOY_COS'] = np.cos(2 * np.pi * doy / 365.0)

        # Cyclic Month
        df['FE_MONTH_SIN'] = np.sin(2 * np.pi * month / 12.0)
        df['FE_MONTH_COS'] = np.cos(2 * np.pi * month / 12.0)

        # Thai Season Flags
        # Note: Month 5 overlaps in definition; we create flags exactly as specified.
        df['FE_THAI_RAINY_FLAG'] = month.isin([5, 6, 7, 8, 9, 10]).astype(int)
        df['FE_THAI_HOT_FLAG'] = month.isin([3, 4, 5]).astype(int)
        df['FE_THAI_COOL_FLAG'] = month.isin([11, 12, 1, 2]).astype(int)

        # Season Index: ordered list hot (0), rainy (1), cool (2)
        # Mapping overlapping month 5 to hot (0) for the categorical index
        conditions = [
            month.isin([3, 4, 5]),
            month.isin([6, 7, 8, 9, 10]),
            month.isin([11, 12, 1, 2])
        ]
        choices = [0, 1, 2]
        season_idx = np.select(conditions, choices, default=0)

        # Cyclic Season Index
        df['FE_THAI_SEASON_SIN'] = np.sin(2 * np.pi * season_idx / 3.0)
        df['FE_THAI_SEASON_COS'] = np.cos(2 * np.pi * season_idx / 3.0)

        new_features = [
            'FE_HOUR_SIN', 'FE_HOUR_COS',
            'FE_DOY_SIN', 'FE_DOY_COS', 'FE_MONTH_SIN', 'FE_MONTH_COS',
            'FE_THAI_RAINY_FLAG', 'FE_THAI_HOT_FLAG', 'FE_THAI_COOL_FLAG',
            'FE_THAI_SEASON_SIN', 'FE_THAI_SEASON_COS'
        ]
        new_feature_names.extend(new_features)

    # ──────────────────────────────────────────────────────────

    print(f"  Experiment: {EXPERIMENT_NAME}")
    if len(new_feature_names) > 5:
        print(f"  New features created ({len(new_feature_names)}): {new_feature_names[:5]} ... and more")
    else:
        print(f"  New features created ({len(new_feature_names)}): {new_feature_names}")

    return df, new_feature_names, EXPERIMENT_NAME
