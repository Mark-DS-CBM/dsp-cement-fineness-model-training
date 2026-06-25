"""
Template for a new feature engineering experiment.

HOW TO USE:
  1. Copy this file:  cp _template.py 01_your_experiment.py
  2. Fill in EXPERIMENT_NAME, DESCRIPTION, and the apply() function body.
  3. In 03_1_FeatEng.py, change the experiment module string:
       _exp = importlib.import_module("feat_eng.01_your_experiment")
  4. Run the notebook. Results will be saved to:
       results/feature_engineering/<EXPERIMENT_NAME>/
"""

EXPERIMENT_NAME = "01_your_experiment"  # ← Change this (must be unique)
DESCRIPTION = "Describe what this experiment does in one line"  # ← Change this
# AFFECTS_ACTIONABLE: set True if ANY engineered feature is derived from an
# actionable parameter (e.g. `sep_motor_speed`). 04_1_Optimization refuses to
# load modules with this flag set to True — under inverse optimization, the
# predictor's engineered features would not update when the optimizer perturbs
# the actionable param, so the loss surface becomes inconsistent with training.
# Leave True unless you've verified your transforms touch only control features.
AFFECTS_ACTIONABLE = True  # ← Change to False ONLY if no engineered feature touches an actionable param


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
    # TODO: Add your feature engineering code below
    # ──────────────────────────────────────────────────────────

    # Example: Interaction feature
    # df['feat_A_x_feat_B'] = df['feat_A'] * df['feat_B']
    # new_feature_names.append('feat_A_x_feat_B')

    # Example: Ratio feature
    # df['feat_A_over_feat_B'] = df['feat_A'] / (df['feat_B'] + 1e-8)
    # new_feature_names.append('feat_A_over_feat_B')

    # Example: Rolling feature (make sure df is sorted by timestamp)
    # df['feat_A_rolling_mean5'] = df['feat_A'].rolling(5, min_periods=1).mean()
    # new_feature_names.append('feat_A_rolling_mean5')

    # Example: Polynomial feature
    # df['feat_A_sq'] = df['feat_A'] ** 2
    # new_feature_names.append('feat_A_sq')

    # ──────────────────────────────────────────────────────────

    print(f"  Experiment: {EXPERIMENT_NAME}")
    print(f"  New features created ({len(new_feature_names)}): {new_feature_names}")

    return df, new_feature_names, EXPERIMENT_NAME
