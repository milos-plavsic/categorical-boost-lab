import os

from app.data import DATA_SOURCE, load_student_math
from app.train import leaderboard, run_comparison


def main() -> None:
    cv_splits = int(os.getenv("CV_SPLITS", "3"))
    df, y = load_student_math()
    results = run_comparison(df, y, cv_splits=cv_splits)
    print("Categorical Boost Lab — UCI student math (pass/fail) — ROC-AUC")
    print(DATA_SOURCE)
    for name, mean, std in leaderboard(results):
        print(f"  {name:24s}  {mean:.4f} ± {std:.4f}")


if __name__ == "__main__":
    main()
