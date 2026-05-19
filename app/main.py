import os

from ml_core import configure_logging

from app.data import DATA_SOURCE, load_student_math
from app.train import leaderboard, run_comparison

logger = configure_logging("app.main")


def main() -> None:
    """Execute the main routine."""
    cv_splits = int(os.getenv("CV_SPLITS", "3"))
    df, y = load_student_math()
    results = run_comparison(df, y, cv_splits=cv_splits)
    logger.info("Categorical Boost Lab — UCI student math (pass/fail) — ROC-AUC")
    logger.info(DATA_SOURCE)
    for name, mean, std in leaderboard(results):
        logger.info(f"  {name:24s}  {mean:.4f} ± {std:.4f}")


if __name__ == "__main__":
    main()
