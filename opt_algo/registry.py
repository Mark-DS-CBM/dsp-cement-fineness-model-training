"""
Optimization Algorithm Registry.

Auto-discovers all algorithm modules in the opt_algo package.
Modules prefixed with '_' (like _template.py) are excluded.

Usage:
    from opt_algo.registry import list_algorithms, get_algorithm

    # List all algorithms
    for name, info in list_algorithms().items():
        print(f"  {name}: {info['description']}  (module: {info['module']})")

    # Get a specific algorithm by name
    algo = get_algorithm("optuna_minimize")
    result = algo['optimize'](features, constraints, predictor)
"""

import importlib
import pkgutil
from pathlib import Path


def list_algorithms():
    """Discover all algorithm modules and return a dict of metadata.

    Returns:
        dict: {algorithm_name: {'module': str, 'description': str, 'optimize': callable}}
    """
    import opt_algo  # import the package itself for path resolution

    algorithms = {}
    pkg_path = opt_algo.__path__

    for _, module_name, _ in pkgutil.iter_modules(pkg_path):
        # Skip private/internal modules
        if module_name.startswith('_') or module_name == 'registry':
            continue

        mod = importlib.import_module(f'opt_algo.{module_name}')

        # Validate the module has required attributes
        if not hasattr(mod, 'ALGORITHM_NAME') or not hasattr(mod, 'optimize'):
            print(f"  ⚠️  Skipping opt_algo.{module_name} — missing ALGORITHM_NAME or optimize()")
            continue

        algorithms[mod.ALGORITHM_NAME] = {
            'module': module_name,
            'description': getattr(mod, 'DESCRIPTION', '(no description)'),
            'optimize': mod.optimize,
        }

    return algorithms


def get_algorithm(name):
    """Get a specific algorithm by its ALGORITHM_NAME.

    Args:
        name: The algorithm name to look up.

    Returns:
        dict with 'module', 'description', 'optimize' keys.

    Raises:
        KeyError: If algorithm name not found.
    """
    algorithms = list_algorithms()
    if name not in algorithms:
        available = ', '.join(sorted(algorithms.keys()))
        raise KeyError(
            f"Algorithm '{name}' not found. Available: [{available}]"
        )
    return algorithms[name]
