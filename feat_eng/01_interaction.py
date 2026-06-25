"""
Feature engineering experiment: Interaction features.
Creates pairwise multiplication features of key numeric columns.
"""

EXPERIMENT_NAME = "01_interaction"
DESCRIPTION = "Pairwise multiplication of key numeric features"
AFFECTS_ACTIONABLE = True  # `sep_motor_speed` participates in interactions


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

    for i in range(len(numeric_cols)):
        for j in range(i + 1, len(numeric_cols)):
            col1 = numeric_cols[i]
            col2 = numeric_cols[j]
            # Verify columns exist before interacting
            if col1 in df.columns and col2 in df.columns:
                new_col = f"{col1}_x_{col2}"
                df[new_col] = df[col1] * df[col2]
                new_feature_names.append(new_col)

    # ──────────────────────────────────────────────────────────

    print(f"  Experiment: {EXPERIMENT_NAME}")
    if len(new_feature_names) > 5:
        print(f"  New features created ({len(new_feature_names)}): {new_feature_names[:5]} ... and more")
    else:
        print(f"  New features created ({len(new_feature_names)}): {new_feature_names}")

    return df, new_feature_names, EXPERIMENT_NAME
