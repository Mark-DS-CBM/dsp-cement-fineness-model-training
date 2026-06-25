"""
Template for a new optimization algorithm.

HOW TO USE:
  1. Copy this file:  cp _template.py 01_your_algorithm.py
  2. Fill in ALGORITHM_NAME, DESCRIPTION, and the optimize() function body.
  3. In 05_1_Optimization.py, change the algorithm module string:
       _opt = importlib.import_module("opt_algo.01_your_algorithm")
  4. Run the notebook. Results will be saved to:
       results/optimization/<ALGORITHM_NAME>/

STANDARDIZED I/O CONTRACT:

  Input 1 — features (dict):
      Contains the desired target, current actionable parameter values,
      and all control parameters.
      Example:
          {
              'desired_blaine': 4550.0,
              'sep_motor_speed': 1380.0,      # actionable (to be optimized)
              'sep fan_damper': 60.0,          # control (fixed)
              'be-mill_out_power': 20.0,       # control (fixed)
              'sep_motor_amp': 155.0,          # control (fixed)
              'sep fan_amp': 40.0,             # control (fixed)
              'cement_temp-mv': 122.0,         # control (fixed)
              'cement_temp-wsp': 113.0,        # control (fixed)
          }

  Input 2 — constraints (dict):
      Valid ranges for actionable parameters. May include a 3rd item for step size.
      Example:
          {'sep_motor_speed': [1299.0, 1449.0, 5.0]}

  Input 3 — predictor (callable):
      A function: predictor(feature_dict) -> float
      Takes a dict of ALL feature columns (same keys as `features` minus
      'desired_blaine') and returns the predicted blaine value.

  Output — result (dict):
      A complete feature dict with the newly optimized actionable parameter(s)
      and all control parameters unchanged.
      Example:
          {'sep_motor_speed': 1395.2, 'sep fan_damper': 60.0, ...}

  Output (optional) — '_alternates' key:
      Population-based algorithms (e.g. NSGA-II) may include an '_alternates'
      key carrying a list of alternate candidate dicts (ranked best-first),
      each with the same shape as the top-level result. The harness records
      them as extra recommendations for the operator-rejection loop described
      in AGENTS.md (User journey step 3). Algorithms without population-based
      output should omit this key.
      Example:
          {
              'sep_motor_speed': 1395.2, 'sep fan_damper': 60.0, ...,
              '_alternates': [
                  {'sep_motor_speed': 1385.0, 'sep fan_damper': 60.0, ...},
                  {'sep_motor_speed': 1410.0, 'sep fan_damper': 60.0, ...},
              ],
          }
"""

ALGORITHM_NAME = "01_your_algorithm"  # ← Change this (must be unique)
DESCRIPTION = "Describe what this algorithm does in one line"  # ← Change this


def optimize(features, constraints, predictor):
    """Find optimal actionable parameters to achieve the desired target.

    Args:
        features:    Dict with 'desired_blaine' and all current feature values.
        constraints: Dict of {param_name: [min, max]} for actionable params.
        predictor:   Callable(feature_dict) -> float predicted blaine.

    Returns:
        dict: Complete feature dict with optimized actionable params.
    """
    desired = features['desired_blaine']

    # Build the starting point (all features except the target)
    result = {k: v for k, v in features.items() if k != 'desired_blaine'}

    # ──────────────────────────────────────────────────────────
    # TODO: Implement your optimization logic here.
    #
    # You must:
    #   1. Only modify keys listed in `constraints` (actionable params).
    #   2. Keep all other values in `result` unchanged (control params).
    #   3. Stay within the bounds specified in `constraints`.
    #   4. Minimize |desired - predictor(result)|.
    # ──────────────────────────────────────────────────────────

    return result
