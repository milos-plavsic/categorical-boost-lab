"""FastAPI application for Categorical Boost Lab.

Endpoints:
  GET  /health                 — liveness probe
  GET  /metrics                — Prometheus metrics
  POST /v1/compare             — agentic compare (legacy)
  POST /v1/finetune/rf_pipeline— RF hyper-parameter finetune (legacy)
  POST /v1/train               — train all models, return leaderboard
  POST /v1/predict             — predict with best registered model
  GET  /v1/models              — list model types and default hyper-parameters
  GET  /v1/data-profile        — dataset statistics
"""

from __future__ import annotations

from typing import Any

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from ml_core import configure_logging, install_middleware
from ml_core.observability import metrics_router, observe_request
from pydantic import BaseModel, Field

from app.data import DATA_SOURCE, discover_columns, load_student_math
from app.langgraph_compare import run_agentic_compare
from app.model_registry import registry
from app.train import build_pipelines, leaderboard, run_comparison
from finetune.tuner import run_rf_pipeline_finetune

logger = configure_logging("app.api")

app = FastAPI(title="Categorical Boost Lab", version="0.3.0")

# -- Middleware ----------------------------------------------------------
install_middleware(app, cors_allow_origins=("*",))
app.include_router(metrics_router)


@app.middleware("http")
async def _observe(request: Request, call_next):
    return await observe_request(request, call_next)


# -----------------------------------------------------------------------
# Request / Response schemas
# -----------------------------------------------------------------------


class CompareRequest(BaseModel):
    """Pydantic schema for the compare request."""

    cv_splits: int = Field(3, ge=2, le=10)
    confidence_threshold: float = Field(0.69, ge=0.0, le=1.0)
    max_iterations: int = Field(3, ge=1, le=8)


class TrainRequest(BaseModel):
    """Pydantic schema for POST /v1/train."""

    cv_splits: int = Field(3, ge=2, le=10, description="Number of CV folds")
    random_state: int = Field(42, ge=0, description="Random seed for reproducibility")
    include_prior_grades: bool = Field(False, description="If true, include G1/G2 as features")


class PredictRequest(BaseModel):
    """Pydantic schema for POST /v1/predict."""

    features: dict[str, Any] = Field(
        ..., description="Feature dictionary matching the training columns"
    )


# -----------------------------------------------------------------------
# Routes — legacy
# -----------------------------------------------------------------------


@app.get("/health")
async def health() -> dict[str, str]:
    """Liveness probe."""
    return {"status": "ok"}


@app.post("/v1/compare")
async def compare(body: CompareRequest) -> dict:
    """Agentic comparison (legacy endpoint — wraps langgraph)."""
    out = run_agentic_compare(
        cv_splits=body.cv_splits,
        confidence_threshold=body.confidence_threshold,
        max_iterations=body.max_iterations,
    )
    return {**out, "data_source": DATA_SOURCE}


@app.post("/v1/finetune/rf_pipeline")
async def finetune_rf_pipeline() -> dict:
    """Run RF hyper-parameter fine-tune (legacy)."""
    return run_rf_pipeline_finetune()


# -----------------------------------------------------------------------
# Routes — v1 ML API
# -----------------------------------------------------------------------


@app.post("/v1/train")
async def train(body: TrainRequest) -> dict:
    """Train all model families on the student dataset and return a leaderboard.

    Side effect: registers all trained models in the in-process model registry
    so that subsequent calls to ``POST /v1/predict`` work without retraining.
    """
    try:
        X, y = load_student_math(include_prior_grades=body.include_prior_grades)
        results = run_comparison(X, y, cv_splits=body.cv_splits, random_state=body.random_state)
        board = leaderboard(results)

        # Register the best model from each fold (re-train once on full data).
        num_cols, cat_cols = discover_columns(X)
        pipelines = build_pipelines(num_cols, cat_cols, random_state=body.random_state)
        for name, pipeline in pipelines.items():
            pipeline.fit(X, y)
            registry.register(
                name,
                pipeline,
                {
                    "roc_auc_mean": results[name]["roc_auc_mean"],
                    "roc_auc_std": results[name]["roc_auc_std"],
                    "cv_splits": body.cv_splits,
                    "random_state": body.random_state,
                },
            )

        leaderboard_response = [
            {"name": name, "roc_auc_mean": mean, "roc_auc_std": std} for name, mean, std in board
        ]
        return {
            "leaderboard": leaderboard_response,
            "data_source": DATA_SOURCE,
            "n_samples": int(len(X)),
            "n_features": int(X.shape[1]),
        }
    except Exception as exc:
        logger.exception("Training failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/v1/predict")
async def predict(body: PredictRequest) -> dict:
    """Return positive-class probability from the best registered model.

    The registry must be populated first by calling ``POST /v1/train``.
    """
    import pandas as pd

    if not registry.list_models():
        raise HTTPException(
            status_code=400,
            detail="No models trained yet. Call POST /v1/train first.",
        )
    try:
        X = pd.DataFrame([body.features])
        probabilities = registry.predict(X)
        best_name, _ = registry.get_best()
        return {
            "best_model": best_name,
            "probability_class_1": float(probabilities[0]),
            "predicted_class": int(probabilities[0] >= 0.5),
        }
    except Exception as exc:
        logger.exception("Prediction failed: %s", exc)
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.get("/v1/models")
async def list_models() -> dict:
    """Return the available model families and their default hyper-parameters."""
    model_specs: dict[str, dict[str, Any]] = {
        "random_forest_onehot": {
            "type": "RandomForestClassifier",
            "preprocessing": "OneHotEncoder for categoricals",
            "hyperparameters": {
                "n_estimators": 32,
                "max_depth": 12,
                "min_samples_leaf": 2,
                "class_weight": "balanced_subsample",
            },
        },
        "xgboost_ordinal": {
            "type": "XGBClassifier",
            "preprocessing": "OrdinalEncoder for categoricals",
            "hyperparameters": {
                "n_estimators": 32,
                "max_depth": 5,
                "learning_rate": 0.08,
                "subsample": 0.85,
                "colsample_bytree": 0.85,
                "reg_lambda": 1.0,
                "objective": "binary:logistic",
            },
        },
        "catboost_native": {
            "type": "CatBoostClassifier",
            "preprocessing": "Native categorical support (no encoding needed)",
            "hyperparameters": {
                "iterations": 32,
                "depth": 6,
                "learning_rate": 0.08,
                "loss_function": "Logloss",
            },
        },
    }

    registered = {m["name"] for m in registry.list_models()}
    for name in model_specs:
        model_specs[name]["registered"] = name in registered

    return {"models": model_specs, "total": len(model_specs)}


@app.get("/v1/data-profile")
async def data_profile() -> dict:
    """Return dataset statistics: shape, feature types, class balance."""
    try:
        X, y = load_student_math()
        num_cols, cat_cols = discover_columns(X)

        # Class balance
        n_total = int(len(y))
        n_pass = int(np.sum(y == 1))
        n_fail = int(np.sum(y == 0))

        # Basic stats per numeric column
        numeric_stats: dict[str, dict[str, float]] = {}
        for col in num_cols:
            series = X[col]
            numeric_stats[col] = {
                "mean": float(series.mean()),
                "std": float(series.std()),
                "min": float(series.min()),
                "max": float(series.max()),
                "median": float(series.median()),
            }

        return {
            "shape": {"rows": int(X.shape[0]), "columns": int(X.shape[1])},
            "numeric_columns": num_cols,
            "categorical_columns": cat_cols,
            "class_balance": {
                "total": n_total,
                "pass_n": n_pass,
                "fail_n": n_fail,
                "pass_rate": round(n_pass / n_total, 4),
            },
            "numeric_stats": numeric_stats,
            "data_source": DATA_SOURCE,
        }
    except Exception as exc:
        logger.exception("Data profile failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/v1/leaderboard")
async def leaderboard_endpoint() -> dict:
    """Return the current registry leaderboard (populated after POST /v1/train)."""
    models = registry.list_models()
    if not models:
        return {
            "leaderboard": [],
            "message": "No models trained yet. Call POST /v1/train first.",
        }
    return {
        "leaderboard": models,
        "best_model": models[0]["name"] if models else None,
    }
