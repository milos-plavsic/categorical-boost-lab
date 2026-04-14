from fastapi import FastAPI
from pydantic import BaseModel, Field

from app.data import DATA_SOURCE, load_student_math
from app.train import leaderboard, run_comparison
from finetune.tuner import run_rf_pipeline_finetune

app = FastAPI(title="Categorical Boost Lab", version="0.1.0")


class CompareRequest(BaseModel):
    cv_splits: int = Field(3, ge=2, le=10)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/v1/compare")
def compare(body: CompareRequest) -> dict:
    df, y = load_student_math()
    results = run_comparison(df, y, cv_splits=body.cv_splits)
    board = [{"model": n, "roc_auc_mean": m, "roc_auc_std": s} for n, m, s in leaderboard(results)]
    return {"leaderboard": board, "raw": results, "data_source": DATA_SOURCE}


@app.post("/v1/finetune/rf_pipeline")
def finetune_rf_pipeline() -> dict:
    return run_rf_pipeline_finetune()
