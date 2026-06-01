import numpy as np

from app.data import load_student_math
from app.train import leaderboard, run_comparison


def test_three_model_families() -> None:
    """Test three model families."""
    df, y = load_student_math()
    out = run_comparison(df, y, cv_splits=3, random_state=0)
    assert set(out.keys()) == {"random_forest_onehot", "xgboost_ordinal", "catboost_native"}
    for name, m in out.items():
        assert np.isfinite(m["roc_auc_mean"])
        assert len(m["folds"]) == 3
        assert m["roc_auc_mean"] >= 0.5, name


def test_leaderboard_non_increasing() -> None:
    """Test leaderboard non increasing."""
    df, y = load_student_math()
    out = run_comparison(df, y, cv_splits=3, random_state=2)
    board = leaderboard(out)
    means = [m for _, m, _ in board]
    for i in range(len(means) - 1):
        assert means[i] >= means[i + 1]
