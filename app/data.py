from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from app.uci_fetch import fetch_uci_student_csv

DATA_SOURCE = (
    "UCI — Student Performance (Math), secondary schools Portugal. "
    "https://archive.ics.uci.edu/dataset/320/student+performance"
)


def project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _ensure_student_mat_csv(path: Path) -> None:
    if path.exists():
        return
    try:
        fetch_uci_student_csv("student-mat.csv", path)
    except Exception as e:
        raise RuntimeError("Could not obtain student-mat.csv from UCI") from e


def load_student_math() -> tuple[pd.DataFrame, np.ndarray]:
    """Features + binary pass label (G3 >= 10); prior grades G1/G2 dropped."""
    path = project_root() / "data" / "student-mat.csv"
    _ensure_student_mat_csv(path)
    df = pd.read_csv(path, sep=";")
    y = (df["G3"] >= 10).astype(int).to_numpy()
    X = df.drop(columns=["G1", "G2", "G3"])
    return X, y


def discover_columns(X: pd.DataFrame) -> tuple[list[str], list[str]]:
    num = list(X.select_dtypes(include=[np.number]).columns)
    cat = list(X.select_dtypes(exclude=[np.number]).columns)
    return num, cat


def ordered_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Stable column order for pipelines and tuning."""
    cols = sorted(df.columns)
    return df[cols].copy()
