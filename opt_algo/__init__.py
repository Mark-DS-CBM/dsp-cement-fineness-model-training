# opt_algo — Optimization Algorithm Registry
#
# Each algorithm module exports:
#   ALGORITHM_NAME : str   — unique short name (used as results subfolder)
#   DESCRIPTION    : str   — one-line description of what this algorithm does
#   optimize(features, constraints, predictor) : func
#       → returns dict of optimized feature values
#
# Usage in 05_1_Optimization.py:
#   import importlib
#   _opt = importlib.import_module("opt_algo.00_optuna_minimize")
#   optimize = _opt.optimize
#
# To list all registered algorithms:
#   from opt_algo.registry import list_algorithms
#   for name, info in list_algorithms().items():
#       print(f"{name}: {info['description']}")
