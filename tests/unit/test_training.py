"""Comprehensive unit tests for categorical-boost-lab training pipeline.

Covers data loading, column discovery, model construction, cross-validation,
leaderboard ordering, and reproducibility with a fixed random state.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone

from app.data import discover_columns, load_student_math, ordered_frame
from app.train import build_pipelines, leaderboard, run_comparison

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def full_dataset():
    """Load the full student-math dataset once per module for speed."""
    X, y = load_student_math()
    return X, y


@pytest.fixture(scope="module")
def small_dataset(full_dataset):
    """150-row stratified subset for fast individual tests."""
    X, y = full_dataset
    rng = np.random.default_rng(0)
    idx_pass = np.where(y == 1)[0]
    idx_fail = np.where(y == 0)[0]
    # Take up to 90 pass, 60 fail
    chosen = np.concatenate(
        [
            rng.choice(idx_pass, size=min(90, len(idx_pass)), replace=False),
            rng.choice(idx_fail, size=min(60, len(idx_fail)), replace=False),
        ]
    )
    rng.shuffle(chosen)
    return X.iloc[chosen].reset_index(drop=True), y[chosen]


# ---------------------------------------------------------------------------
# Data loading tests
# ---------------------------------------------------------------------------


def test_load_student_math_returns_dataframe(full_dataset):
    """load_student_math() must return a pd.DataFrame, not a plain array."""
    X, y = full_dataset
    assert isinstance(X, pd.DataFrame), "X should be a DataFrame"
    assert isinstance(y, np.ndarray), "y should be a numpy array"


def test_load_student_math_shape(full_dataset):
    """Dataset must have at least 300 rows and more than 20 features."""
    X, y = full_dataset
    assert X.shape[0] >= 300, f"Expected >=300 rows, got {X.shape[0]}"
    assert X.shape[1] >= 20, f"Expected >=20 columns, got {X.shape[1]}"


def test_load_student_math_target_binary(full_dataset):
    """Target array must be binary (0/1 only)."""
    _, y = full_dataset
    unique = set(np.unique(y))
    assert unique == {0, 1}, f"y should be binary, got {unique}"


def test_load_student_math_no_target_column(full_dataset):
    """G3, G1, G2 must not appear in X when include_prior_grades=False."""
    X, _ = full_dataset
    assert "G3" not in X.columns, "G3 should be dropped"
    assert "G1" not in X.columns, "G1 should be dropped by default"
    assert "G2" not in X.columns, "G2 should be dropped by default"


def test_load_student_math_with_prior_grades():
    """When include_prior_grades=True, G1 and G2 are retained but G3 dropped."""
    X, _ = load_student_math(include_prior_grades=True)
    assert "G3" not in X.columns
    assert "G1" in X.columns
    assert "G2" in X.columns


def test_load_student_math_expected_columns(full_dataset):
    """Dataset must include known UCI student-performance columns."""
    X, _ = full_dataset
    expected_cols = {"age", "absences", "Medu", "Fedu", "failures"}
    assert expected_cols.issubset(
        set(X.columns)
    ), f"Missing expected columns from {X.columns.tolist()}"


def test_load_student_math_class_balance(full_dataset):
    """Neither class should account for fewer than 10% of samples."""
    _, y = full_dataset
    pass_rate = y.mean()
    assert 0.10 <= pass_rate <= 0.90, f"Unusual class balance: pass_rate={pass_rate:.2f}"


# ---------------------------------------------------------------------------
# Column discovery tests
# ---------------------------------------------------------------------------


def test_discover_columns_types(full_dataset):
    """discover_columns must partition columns into numeric and categorical."""
    X, _ = full_dataset
    num, cat = discover_columns(X)
    assert isinstance(num, list)
    assert isinstance(cat, list)
    # Every column must appear in exactly one list
    assert set(num + cat) == set(X.columns)
    assert set(num) & set(cat) == set(), "Overlap between num and cat"


def test_discover_columns_numeric_content(full_dataset):
    """Columns flagged as numeric must contain numeric dtypes."""
    X, _ = full_dataset
    num, _ = discover_columns(X)
    for col in num:
        assert pd.api.types.is_numeric_dtype(
            X[col]
        ), f"{col} flagged numeric but has dtype {X[col].dtype}"


def test_ordered_frame_sorts_columns(full_dataset):
    """ordered_frame() must return a DataFrame with sorted column names."""
    X, _ = full_dataset
    ordered = ordered_frame(X)
    assert list(ordered.columns) == sorted(X.columns)


# ---------------------------------------------------------------------------
# Pipeline construction tests
# ---------------------------------------------------------------------------


def test_build_pipelines_returns_three_models(full_dataset):
    """build_pipelines must return exactly three named pipelines."""
    X, _ = full_dataset
    num, cat = discover_columns(X)
    pipelines = build_pipelines(num, cat)
    assert set(pipelines.keys()) == {"random_forest_onehot", "xgboost_ordinal", "catboost_native"}


def test_build_pipelines_are_sklearn_estimators(full_dataset):
    """Each returned pipeline must be a sklearn-compatible estimator."""
    from sklearn.pipeline import Pipeline

    X, _ = full_dataset
    num, cat = discover_columns(X)
    pipelines = build_pipelines(num, cat)
    for name, pipe in pipelines.items():
        assert isinstance(pipe, Pipeline), f"{name} is not a Pipeline"


# ---------------------------------------------------------------------------
# Model training / ROC-AUC tests
# ---------------------------------------------------------------------------


def test_each_model_trains_successfully(small_dataset):
    """Every pipeline must fit on training data without raising."""
    X, y = small_dataset
    num, cat = discover_columns(X)
    pipelines = build_pipelines(num, cat, random_state=7)
    n_train = int(0.7 * len(X))
    X_tr, _X_val = X.iloc[:n_train], X.iloc[n_train:]
    y_tr = y[:n_train]
    for name, pipe in pipelines.items():
        est = clone(pipe) if not hasattr(pipe[-1], "allow_writing_files") else pipe
        try:
            est = clone(pipe)
        except Exception:
            import copy

            est = copy.deepcopy(pipe)
        est.fit(X_tr, y_tr)  # must not raise


def test_each_model_achieves_roc_auc_gt_0_7(small_dataset):
    """Each model should beat 0.7 ROC-AUC on a hold-out set."""
    from sklearn.metrics import roc_auc_score

    X, y = small_dataset
    num, cat = discover_columns(X)
    pipelines = build_pipelines(num, cat, random_state=99)
    n_train = int(0.7 * len(X))
    X_tr, X_val = X.iloc[:n_train], X.iloc[n_train:]
    y_tr, y_val = y[:n_train], y[n_train:]
    for name, pipe in pipelines.items():
        import copy

        try:
            from sklearn.base import clone

            est = clone(pipe)
        except Exception:
            est = copy.deepcopy(pipe)
        est.fit(X_tr, y_tr)
        proba = est.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y_val, proba)
        assert auc > 0.7, f"{name} achieved ROC-AUC={auc:.3f} (<0.7 threshold)"


# ---------------------------------------------------------------------------
# run_comparison tests
# ---------------------------------------------------------------------------


def test_run_comparison_returns_all_models(small_dataset):
    """run_comparison must return results for all three model families."""
    X, y = small_dataset
    results = run_comparison(X, y, cv_splits=2, random_state=0)
    assert set(results.keys()) == {"random_forest_onehot", "xgboost_ordinal", "catboost_native"}


def test_run_comparison_roc_auc_range(small_dataset):
    """Every model's mean ROC-AUC should be in (0.5, 1.0]."""
    X, y = small_dataset
    results = run_comparison(X, y, cv_splits=2, random_state=1)
    for name, metrics in results.items():
        mean = metrics["roc_auc_mean"]
        assert 0.5 < mean <= 1.0, f"{name}: roc_auc_mean={mean:.4f} outside (0.5, 1.0]"


def test_run_comparison_folds_count(small_dataset):
    """Number of fold scores must equal cv_splits."""
    X, y = small_dataset
    cv_splits = 3
    results = run_comparison(X, y, cv_splits=cv_splits, random_state=0)
    for name, metrics in results.items():
        assert len(metrics["folds"]) == cv_splits, f"{name}: expected {cv_splits} folds"


def test_run_comparison_std_non_negative(small_dataset):
    """Standard deviation of ROC-AUC scores must be non-negative."""
    X, y = small_dataset
    results = run_comparison(X, y, cv_splits=2, random_state=5)
    for name, metrics in results.items():
        assert metrics["roc_auc_std"] >= 0, f"{name}: negative std"


def test_run_comparison_mean_consistent_with_folds(small_dataset):
    """Reported mean must match np.mean of the fold scores."""
    X, y = small_dataset
    results = run_comparison(X, y, cv_splits=3, random_state=2)
    for name, metrics in results.items():
        expected_mean = float(np.nanmean(metrics["folds"]))
        assert (
            abs(metrics["roc_auc_mean"] - expected_mean) < 1e-9
        ), f"{name}: reported mean {metrics['roc_auc_mean']} != np.nanmean {expected_mean}"


def test_run_comparison_reproducible_with_same_seed(small_dataset):
    """run_comparison with the same random_state must produce identical results."""
    X, y = small_dataset
    r1 = run_comparison(X, y, cv_splits=2, random_state=17)
    r2 = run_comparison(X, y, cv_splits=2, random_state=17)
    for name in r1:
        assert (
            abs(r1[name]["roc_auc_mean"] - r2[name]["roc_auc_mean"]) < 1e-9
        ), f"{name}: results differ between identical random_state runs"


def test_run_comparison_different_seeds_may_differ(small_dataset):
    """Different random seeds should (almost always) produce different folds."""
    X, y = small_dataset
    r1 = run_comparison(X, y, cv_splits=2, random_state=0)
    r2 = run_comparison(X, y, cv_splits=2, random_state=999)
    # At least one model should differ (very unlikely to be equal)
    all_equal = all(abs(r1[n]["roc_auc_mean"] - r2[n]["roc_auc_mean"]) < 1e-12 for n in r1)
    assert not all_equal, "All models identical across different seeds — suspicious"


# ---------------------------------------------------------------------------
# leaderboard tests
# ---------------------------------------------------------------------------


def test_leaderboard_returns_all_models(small_dataset):
    """leaderboard() must return one entry per model."""
    X, y = small_dataset
    results = run_comparison(X, y, cv_splits=2, random_state=0)
    board = leaderboard(results)
    assert len(board) == len(results)


def test_leaderboard_sorted_descending(small_dataset):
    """leaderboard() must be sorted best-first by mean ROC-AUC."""
    X, y = small_dataset
    results = run_comparison(X, y, cv_splits=2, random_state=3)
    board = leaderboard(results)
    means = [m for _, m, _ in board]
    assert means == sorted(means, reverse=True), f"Leaderboard not sorted: {means}"


def test_leaderboard_tuple_structure(small_dataset):
    """Each leaderboard row must be (str, float, float)."""
    X, y = small_dataset
    results = run_comparison(X, y, cv_splits=2, random_state=0)
    board = leaderboard(results)
    for row in board:
        name, mean, std = row
        assert isinstance(name, str)
        assert isinstance(mean, float)
        assert isinstance(std, float)
