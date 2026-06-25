"""
Feature engineering experiment: Ratio features.
Creates ratio features of related logical pairs.
"""

EXPERIMENT_NAME = "02_ratio"
DESCRIPTION = "Ratios between related process features"
AFFECTS_ACTIONABLE = True  # `sep_motor_speed` appears as denominator in several ratios


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

    # Pairs of features for ratio: (numerator, denominator)
    pairs = [
        ('sep_motor_amp', 'sep_motor_speed'),
        ('sep fan_amp', 'sep fan_damper'),
        ('cement_temp-mv', 'cement_temp-wsp'),
        ('be-mill_out_power', 'sep_motor_speed')
    ]

    for num, den in pairs:
        if num in df.columns and den in df.columns:
            new_col = f"{num}_over_{den}"
            # Add a small epsilon to avoid division by zero
            df[new_col] = df[num] / (df[den] + 1e-8)
            new_feature_names.append(new_col)

    # ──────────────────────────────────────────────────────────

    print(f"  Experiment: {EXPERIMENT_NAME}")
    print(f"  New features created ({len(new_feature_names)}): {new_feature_names}")

    return df, new_feature_names, EXPERIMENT_NAME
