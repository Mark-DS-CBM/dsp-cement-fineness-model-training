"""
Feature Engineering Experiment Registry.

Auto-discovers all experiment modules in the feat_eng package.
Modules prefixed with '_' (like _template.py) are excluded.

Usage:
    from feat_eng.registry import list_experiments, get_experiment

    # List all experiments
    for name, info in list_experiments().items():
        print(f"  {name}: {info['description']}  (module: {info['module']})")

    # Get a specific experiment by name
    exp = get_experiment("baseline")
    df, new_feats, exp_name = exp['apply'](df)
"""

import importlib
import pkgutil
from pathlib import Path


def list_experiments():
    """Discover all experiment modules and return a dict of metadata.

    Returns:
        dict: {experiment_name: {'module': str, 'description': str, 'apply': callable}}
    """
    import feat_eng  # import the package itself for path resolution

    experiments = {}
    pkg_path = feat_eng.__path__

    for _, module_name, _ in pkgutil.iter_modules(pkg_path):
        # Skip private/internal modules
        if module_name.startswith('_') or module_name == 'registry':
            continue

        mod = importlib.import_module(f'feat_eng.{module_name}')

        # Validate the module has required attributes
        if not hasattr(mod, 'EXPERIMENT_NAME') or not hasattr(mod, 'apply'):
            print(f"  ⚠️  Skipping feat_eng.{module_name} — missing EXPERIMENT_NAME or apply()")
            continue

        experiments[mod.EXPERIMENT_NAME] = {
            'module': module_name,
            'description': getattr(mod, 'DESCRIPTION', '(no description)'),
            'apply': mod.apply,
        }

    return experiments


def get_experiment(name):
    """Get a specific experiment by its EXPERIMENT_NAME.

    Args:
        name: The experiment name to look up.

    Returns:
        dict with 'module', 'description', 'apply' keys.

    Raises:
        KeyError: If experiment name not found.
    """
    experiments = list_experiments()
    if name not in experiments:
        available = ', '.join(sorted(experiments.keys()))
        raise KeyError(
            f"Experiment '{name}' not found. Available: [{available}]"
        )
    return experiments[name]
