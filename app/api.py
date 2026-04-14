from fastapi import FastAPI
from pydantic import BaseModel, Field

from app.data import load_mixed_dataframe
from app.train import leaderboard, run_comparison

app = FastAPI(title="Categorical Boost Lab", version="0.1.0")


class CompareRequest(BaseModel):
    n_samples: int = Field(1200, ge=400, le=30_000)
    cv_splits: int = Field(3, ge=2, le=10)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/v1/compare")
def compare(body: CompareRequest) -> dict:
    df, y = load_mixed_dataframe(n_samples=body.n_samples)
    results = run_comparison(df, y, cv_splits=body.cv_splits)
    board = [{"model": n, "roc_auc_mean": m, "roc_auc_std": s} for n, m, s in leaderboard(results)]
    return {"leaderboard": board, "raw": results}
