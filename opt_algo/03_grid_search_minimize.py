"""
Exhaustive grid-search optimizer.

Sweeps every grid-spaced combination of actionable parameters inside the
provided constraints and returns the candidate minimizing
|desired_blaine - predictor(x)|. Deterministic by construction: repeated
calls with the same inputs produce the same recommendation, so the §4
stability metrics (SD of recommended params, SD of objective error) will
all be 0.0 — the comparative advantage of an exhaustive search.

Grid resolution:
  - If a constraint provides an explicit step (`[lo, hi, step]`), the grid
    uses `np.arange(lo, hi + step/2, step)` so the upper endpoint isn't
    dropped by floating-point drift.
  - Otherwise the grid falls back to `np.linspace(lo, hi, GRID_POINTS)`.

Like 02_optuna_nsga2_minimize, this optimizer also returns an `_alternates`
list (top-K diverse runners-up along the primary actionable param) to feed
the operator-rejection loop described in AGENTS.md.
"""

import itertools

import numpy as np

ALGORITHM_NAME = "grid_search_minimize"
DESCRIPTION = "Exhaustive grid search over actionable params; top-5 diverse alternates"

GRID_POINTS = 100         # Fallback resolution when a constraint has no explicit step
TOP_K = 5
MIN_GAP_MULTIPLIER = 2.0  # Min spacing between alternates, in units of constraint step
DEFAULT_STEP = 1.0        # Fallback gap unit when a constraint has no explicit step


def _build_grid(bounds):
    """Return a 1-D numpy array of grid points for a single param's bounds."""
    if len(bounds) == 3:
        lo, hi, step = bounds
        return np.arange(lo, hi + step / 2.0, step)
    lo, hi = bounds[0], bounds[1]
    return np.linspace(lo, hi, GRID_POINTS)


def optimize(features, constraints, predictor):
    """Find optimal actionable parameters via exhaustive grid search.

    Args:
        features:    Dict with 'desired_blaine' and all current feature values.
        constraints: Dict of {param_name: [min, max]} or [min, max, step] for
                     actionable params.
        predictor:   Callable(feature_dict) -> float predicted blaine.

    Returns:
        dict: Complete feature dict with optimized actionable params. Includes
              an '_alternates' key with up to TOP_K diverse alternate
              candidates (same dict shape, best-first).
    """
    desired = features['desired_blaine']
    base = {k: v for k, v in features.items() if k != 'desired_blaine'}
    actionable_params = list(constraints.keys())

    grids = [_build_grid(constraints[p]) for p in actionable_params]

    # Evaluate every grid combo; remember (err, combo) pairs for alternates.
    evaluated = []
    for combo in itertools.product(*grids):
        candidate = base.copy()
        for p, v in zip(actionable_params, combo):
            candidate[p] = float(v)
        err = abs(desired - predictor(candidate))
        evaluated.append((err, combo))

    evaluated.sort(key=lambda x: x[0])
    best_combo = evaluated[0][1]

    result = base.copy()
    for p, v in zip(actionable_params, best_combo):
        result[p] = float(v)

    # ── Top-K diverse alternates along the primary actionable param ──
    primary = actionable_params[0]
    primary_idx = 0
    primary_bounds = constraints[primary]
    step = primary_bounds[2] if len(primary_bounds) == 3 else DEFAULT_STEP
    min_gap = step * MIN_GAP_MULTIPLIER

    selected_values = [result[primary]]
    alternates = []
    for _err, combo in evaluated[1:]:
        v = float(combo[primary_idx])
        if all(abs(v - sv) >= min_gap for sv in selected_values):
            alt = base.copy()
            for p, av in zip(actionable_params, combo):
                alt[p] = float(av)
            alternates.append(alt)
            selected_values.append(v)
            if len(alternates) >= TOP_K:
                break

    result['_alternates'] = alternates
    return result
