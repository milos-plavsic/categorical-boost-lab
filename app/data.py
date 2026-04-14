from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification

N_NUMERIC = 10
CAT_COLS = ["region", "sku_bucket", "channel"]


def load_mixed_dataframe(
    n_samples: int = 1200,
    random_state: int = 42,
) -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(random_state)
    Xn, y = make_classification(
        n_samples=n_samples,
        n_features=N_NUMERIC,
        n_informative=6,
        n_redundant=2,
        random_state=random_state,
    )
    num_cols = [f"n{i}" for i in range(N_NUMERIC)]
    df = pd.DataFrame(Xn, columns=num_cols)
    k = 10
    df["region"] = pd.Series(rng.integers(0, k, size=n_samples), dtype="int64").astype(str)
    df["sku_bucket"] = pd.Series(rng.integers(0, k, size=n_samples), dtype="int64").astype(str)
    df["channel"] = pd.Series(rng.integers(0, 5, size=n_samples), dtype="int64").astype(str)
    return df, y.astype(np.int64)


def ordered_frame(df: pd.DataFrame) -> pd.DataFrame:
    num_cols = [f"n{i}" for i in range(N_NUMERIC)]
    return df[num_cols + CAT_COLS].copy()
