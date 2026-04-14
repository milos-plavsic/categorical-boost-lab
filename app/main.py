import os

from app.data import load_mixed_dataframe
from app.train import leaderboard, run_comparison


def main() -> None:
    n_samples = int(os.getenv("N_SAMPLES", "1200"))
    cv_splits = int(os.getenv("CV_SPLITS", "3"))
    df, y = load_mixed_dataframe(n_samples=n_samples)
    results = run_comparison(df, y, cv_splits=cv_splits)
    print("Categorical Boost Lab — ROC-AUC (higher is better)")
    for name, mean, std in leaderboard(results):
        print(f"  {name:24s}  {mean:.4f} ± {std:.4f}")


if __name__ == "__main__":
    main()
