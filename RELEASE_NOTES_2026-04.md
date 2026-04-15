# Release Notes (2026-04)

## Scope
This release adds full analysis reporting, fixes cross-validation/model compatibility issues, and stabilizes CI/data loading.

## Data Source
- UCI Student Performance dataset (ID 320): `student-mat.csv`

## Reporting Added
- New `analysis/` package with:
  - `report.py`, `plotting.py`, `stats_utils.py`, `json_util.py`, module entrypoint
- Generated outputs:
  - `reports/summary.json`
  - `reports/REPORT.md`
  - `reports/figures/roc_auc_encoding_comparison.png`
  - `reports/figures/folds_*.png`

## Latest Report Snapshot (ROC-AUC mean ± std)
- `random_forest_onehot`: `0.6862 ± 0.0401`
- `xgboost_ordinal`: `0.6790 ± 0.0486`
- `catboost_native`: `0.6711 ± 0.0588`

## Modeling/CV Fixes
- Replaced scorer path with manual StratifiedKFold + `roc_auc_score`.
- Added safe fallback (`deepcopy`) when estimator cloning fails for CatBoost in CV contexts.
- Set explicit XGBoost binary objective for classifier behavior consistency.

## Reliability and CI
- Added ZIP-based UCI fetch fallback via `app/uci_fetch.py`.
- Ensured local/offline stability with vendored `data/student-mat.csv`.
- CI runs tests and `python -m analysis` smoke step.
- Upgraded actions to:
  - `actions/checkout@v6`
  - `actions/setup-python@v6`

## Latest CI Status
- Latest successful run: https://github.com/milos-plavsic/categorical-boost-lab/actions/runs/24447654888

## Dependency Notes
- `xgboost` pinned to `>=2.0,<2.1` to reduce CUDA-related install issues in generic CPU CI.
