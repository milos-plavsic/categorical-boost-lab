from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

DATA_SOURCE = (
    "UCI — Student Performance (Math), secondary schools Portugal. "
    "https://archive.ics.uci.edu/dataset/320/student+performance"
)


def project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def load_student_math() -> tuple[pd.DataFrame, np.ndarray]:
    path = project_root() / "data" / "student-mat.csv"
    df = pd.read_csv(path, sep=";")
    y = (df["G3"] >= 10).astype(int).to_numpy()
    X = df.drop(columns=["G3", "G1", "G2"])
    return X, y


def discover_columns(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    cat = df.select_dtypes(include=["object", "category"]).columns.tolist()
    num = [c for c in df.columns if c not in cat]
    return num, cat


def ordered_frame(df: pd.DataFrame) -> pd.DataFrame:
    num, cat = discover_columns(df)
    return df[num + cat].copy()
