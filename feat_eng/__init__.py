# feat_eng — Feature Engineering Experiment Registry
#
# Each experiment module exports:
#   EXPERIMENT_NAME : str   — unique short name (used as results subfolder)
#   DESCRIPTION     : str   — one-line description of what this experiment does
#   apply(df)       : func  — transforms df → (df, new_feature_names, experiment_name)
#
# Usage in 03_1_FeatEng.py:
#   from feat_eng.baseline import apply as apply_feature_engineering
#
# To list all registered experiments:
#   from feat_eng.registry import list_experiments
#   for name, info in list_experiments().items():
#       print(f"{name}: {info['description']}")
