"""
Baseline experiment — no feature engineering applied.
Use this as the control to compare all other experiments against.
"""

EXPERIMENT_NAME = "baseline"
DESCRIPTION = "No feature engineering (raw features only)"
# AFFECTS_ACTIONABLE: True if any engineered feature is derived from an actionable
# parameter (e.g. `sep_motor_speed`). 04_1_Optimization refuses to load modules
# with this flag set to True because the predictor would see stale engineered
# values when the optimizer perturbs the actionable param. See 04_1 §3 for the
# check. Default is True (fail-safe) if the flag is omitted.
AFFECTS_ACTIONABLE = False  # No engineered features → safe for inverse optimization


def apply(df):
    """No-op: returns the DataFrame unchanged.

    Args:
        df: Raw DataFrame after loading and timestamp sorting.

    Returns:
        df:                 Unchanged DataFrame.
        new_feature_names:  Empty list.
        experiment_name:    'baseline'.
    """
    df = df.copy()
    new_feature_names = []

    print(f"  Experiment: {EXPERIMENT_NAME}")
    print(f"  New features created ({len(new_feature_names)}): {new_feature_names}")

    return df, new_feature_names, EXPERIMENT_NAME
