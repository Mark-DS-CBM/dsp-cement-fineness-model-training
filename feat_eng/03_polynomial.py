"""
Feature engineering experiment: Polynomial features.
Creates squared and cubed features.
"""

EXPERIMENT_NAME = "03_polynomial"
DESCRIPTION = "Polynomial (squared and cubed) transformations of numeric features"
AFFECTS_ACTIONABLE = True  # `sep_motor_speed_sq` / `_cube` derived from the actionable param


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

    numeric_cols = [
        'sep_motor_speed', 'sep fan_damper', 'be-mill_out_power',
        'sep_motor_amp', 'sep fan_amp', 'cement_temp-mv', 'cement_temp-wsp'
    ]

    for col in numeric_cols:
        if col in df.columns:
            # Squared feature
            new_col_sq = f"{col}_sq"
            df[new_col_sq] = df[col] ** 2
            new_feature_names.append(new_col_sq)
            
            # Cubed feature
            new_col_cube = f"{col}_cube"
            df[new_col_cube] = df[col] ** 3
            new_feature_names.append(new_col_cube)

    # ──────────────────────────────────────────────────────────

    print(f"  Experiment: {EXPERIMENT_NAME}")
    if len(new_feature_names) > 5:
        print(f"  New features created ({len(new_feature_names)}): {new_feature_names[:5]} ... and more")
    else:
        print(f"  New features created ({len(new_feature_names)}): {new_feature_names}")

    return df, new_feature_names, EXPERIMENT_NAME
