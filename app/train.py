from __future__ import annotations

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from xgboost import XGBClassifier

from app.data import CAT_COLS, N_NUMERIC, ordered_frame


def _num_cols() -> list[str]:
    return [f"n{i}" for i in range(N_NUMERIC)]


def build_pipelines(random_state: int = 42) -> dict[str, object]:
    num = _num_cols()
    cat = CAT_COLS

    pre_rf = ColumnTransformer(
        transformers=[
            ("num", "passthrough", num),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", max_categories=32, sparse_output=False),
                cat,
            ),
        ]
    )
    rf = Pipeline(
        steps=[
            ("prep", pre_rf),
            (
                "clf",
                RandomForestClassifier(
                    n_estimators=100,
                    max_depth=12,
                    min_samples_leaf=2,
                    n_jobs=4,
                    random_state=random_state,
                    class_weight="balanced_subsample",
                ),
            ),
        ]
    )

    pre_xgb = ColumnTransformer(
        transformers=[
            ("num", "passthrough", num),
            (
                "cat",
                OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
                cat,
            ),
        ]
    )
    xgb = Pipeline(
        steps=[
            ("prep", pre_xgb),
            (
                "clf",
                XGBClassifier(
                    n_estimators=100,
                    max_depth=5,
                    learning_rate=0.08,
                    subsample=0.85,
                    colsample_bytree=0.85,
                    reg_lambda=1.0,
                    random_state=random_state,
                    eval_metric="logloss",
                    verbosity=0,
                ),
            ),
        ]
    )

    cb = CatBoostClassifier(
        iterations=100,
        depth=5,
        learning_rate=0.08,
        loss_function="Logloss",
        cat_features=cat,
        verbose=False,
        random_seed=random_state,
        allow_writing_files=False,
    )

    return {"random_forest_onehot": rf, "xgboost_ordinal": xgb, "catboost_native": cb}


def run_comparison(
    df: pd.DataFrame,
    y: np.ndarray,
    cv_splits: int = 3,
    random_state: int = 42,
) -> dict[str, dict]:
    X = ordered_frame(df)
    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=random_state)
    models = build_pipelines(random_state)
    results: dict[str, dict] = {}
    for name, est in models.items():
        scores = cross_val_score(
            est,
            X,
            y,
            cv=cv,
            scoring="roc_auc",
            n_jobs=1,
        )
        results[name] = {
            "roc_auc_mean": float(np.mean(scores)),
            "roc_auc_std": float(np.std(scores)),
            "folds": [float(s) for s in scores],
        }
    return results


def leaderboard(results: dict[str, dict]) -> list[tuple[str, float, float]]:
    rows = [(name, v["roc_auc_mean"], v["roc_auc_std"]) for name, v in results.items()]
    rows.sort(key=lambda r: r[1], reverse=True)
    return rows
