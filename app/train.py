from __future__ import annotations

import copy

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from ml_core import configure_logging
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, OrdinalEncoder
from xgboost import XGBClassifier

from app.data import discover_columns

logger = configure_logging("app.train")


def _catboost_prep(X: pd.DataFrame, num_cols: list[str], cat_cols: list[str]) -> pd.DataFrame:
    """Catboost prep.."""
    out = X[num_cols + cat_cols].copy()
    for c in cat_cols:
        out[c] = out[c].astype(str)
    return out


def build_pipelines(
    num_cols: list[str],
    cat_cols: list[str],
    random_state: int = 42,
) -> dict[str, Pipeline]:
    """Three stacks: RF + one-hot, XGB + ordinal encoding, CatBoost with native categoricals."""
    rf = RandomForestClassifier(
        n_estimators=32,
        max_depth=12,
        min_samples_leaf=2,
        n_jobs=1,
        random_state=random_state,
        class_weight="balanced_subsample",
    )
    try:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)
    pre_rf = ColumnTransformer(
        [
            ("num", "passthrough", num_cols),
            ("cat", ohe, cat_cols),
        ],
        remainder="drop",
    )
    rf_pipe = Pipeline([("prep", pre_rf), ("clf", rf)])

    xgb = XGBClassifier(
        n_estimators=32,
        max_depth=5,
        learning_rate=0.08,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_lambda=1.0,
        random_state=random_state,
        objective="binary:logistic",
        eval_metric="logloss",
        verbosity=0,
        n_jobs=1,
    )
    pre_xgb = ColumnTransformer(
        [
            ("num", "passthrough", num_cols),
            (
                "cat",
                OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
                cat_cols,
            ),
        ],
        remainder="drop",
    )
    xgb_pipe = Pipeline([("prep", pre_xgb), ("clf", xgb)])

    cat_idx = tuple(range(len(num_cols), len(num_cols) + len(cat_cols)))
    cb = CatBoostClassifier(
        iterations=32,
        depth=6,
        learning_rate=0.08,
        loss_function="Logloss",
        verbose=False,
        random_seed=random_state,
        allow_writing_files=False,
        cat_features=cat_idx,
    )
    cb_prep = FunctionTransformer(
        lambda X, nc=num_cols, cc=cat_cols: _catboost_prep(X, nc, cc),
        validate=False,
    )
    cb_pipe = Pipeline([("prep", cb_prep), ("clf", cb)])

    return {
        "random_forest_onehot": rf_pipe,
        "xgboost_ordinal": xgb_pipe,
        "catboost_native": cb_pipe,
    }


def run_comparison(
    X: pd.DataFrame,
    y: np.ndarray,
    cv_splits: int = 3,
    random_state: int = 42,
) -> dict[str, dict]:
    """Run comparison."""
    num, cat = discover_columns(X)
    models = build_pipelines(num, cat, random_state=random_state)
    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=random_state)
    results: dict[str, dict] = {}
    for name, clf in models.items():
        fold_scores: list[float] = []
        for train_idx, val_idx in cv.split(X, y):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            if len(np.unique(y_val)) < 2:
                fold_scores.append(float("nan"))
                continue
            try:
                est = clone(clf)
            except (RuntimeError, TypeError, ValueError) as exc:
                logger.debug(
                    "sklearn.clone failed for %s (%s); falling back to deepcopy",
                    name,
                    exc,
                )
                est = copy.deepcopy(clf)
            est.fit(X_train, y_train)
            proba = est.predict_proba(X_val)[:, 1]
            fold_scores.append(float(roc_auc_score(y_val, proba)))
        arr = np.asarray(fold_scores, dtype=float)
        results[name] = {
            "roc_auc_mean": float(np.nanmean(arr)),
            "roc_auc_std": float(np.nanstd(arr)),
            "folds": [float(s) for s in fold_scores],
        }
    return results


def leaderboard(results: dict[str, dict]) -> list[tuple[str, float, float]]:
    """Leaderboard."""
    rows = [(name, v["roc_auc_mean"], v["roc_auc_std"]) for name, v in results.items()]
    rows.sort(key=lambda r: r[1], reverse=True)
    return rows
