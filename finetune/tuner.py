from __future__ import annotations

import os

import numpy as np
from scipy.stats import randint
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RandomizedSearchCV, train_test_split

from app.data import DATA_SOURCE, discover_columns, load_student_math, ordered_frame
from app.train import build_pipelines


def run_rf_pipeline_finetune(random_state: int = 42) -> dict:
    """Tune the OneHot + RandomForest pipeline (same stack as `random_forest_onehot`)."""
    n_iter = int(os.getenv("FINETUNE_N_ITER", "8"))
    df, y = load_student_math()
    X = ordered_frame(df)
    num, cat = discover_columns(X)
    models = build_pipelines(num, cat, random_state=random_state)
    pipe = models["random_forest_onehot"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y
    )
    param = {
        "clf__n_estimators": randint(60, 220),
        "clf__max_depth": [None, 8, 10, 14],
        "clf__min_samples_leaf": randint(1, 5),
    }
    search = RandomizedSearchCV(
        pipe,
        param,
        n_iter=n_iter,
        cv=3,
        scoring="roc_auc",
        random_state=random_state,
        n_jobs=1,
        refit=True,
    )
    search.fit(X_train, y_train)
    proba = search.predict_proba(X_test)[:, 1]
    auc = float(roc_auc_score(y_test, proba))
    best = {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in search.best_params_.items()}
    return {
        "model": "random_forest_onehot_pipeline",
        "best_params": best,
        "test_roc_auc": auc,
        "n_iter": n_iter,
        "data_source": DATA_SOURCE,
    }


def main() -> None:
    out = run_rf_pipeline_finetune()
    print("RF + OneHot pipeline fine-tune")
    for k, v in out.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
