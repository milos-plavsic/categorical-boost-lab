"""API integration tests for categorical-boost-lab.

Each test uses the FastAPI test client (patched to httpx in conftest).
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.api import app
from app.model_registry import registry

client = TestClient(app)


@pytest.fixture(autouse=True)
def _clear_registry():
    """Reset the model registry before every test so tests are independent."""
    registry.clear()
    yield
    registry.clear()


# ---------------------------------------------------------------------------
# /health
# ---------------------------------------------------------------------------


def test_health_returns_200():
    r = client.get("/health")
    assert r.status_code == 200


def test_health_body_ok():
    r = client.get("/health")
    assert r.json() == {"status": "ok"}


# ---------------------------------------------------------------------------
# /v1/data-profile
# ---------------------------------------------------------------------------


def test_data_profile_returns_200():
    r = client.get("/v1/data-profile")
    assert r.status_code == 200


def test_data_profile_has_shape():
    r = client.get("/v1/data-profile")
    body = r.json()
    assert "shape" in body
    assert body["shape"]["rows"] >= 300
    assert body["shape"]["columns"] >= 20


def test_data_profile_has_class_balance():
    r = client.get("/v1/data-profile")
    body = r.json()
    assert "class_balance" in body
    cb = body["class_balance"]
    assert "pass_rate" in cb
    assert 0.0 < cb["pass_rate"] < 1.0


def test_data_profile_has_numeric_and_categorical():
    r = client.get("/v1/data-profile")
    body = r.json()
    assert isinstance(body["numeric_columns"], list)
    assert isinstance(body["categorical_columns"], list)
    assert len(body["numeric_columns"]) > 0
    assert len(body["categorical_columns"]) > 0


def test_data_profile_has_numeric_stats():
    r = client.get("/v1/data-profile")
    body = r.json()
    assert "numeric_stats" in body
    # At least one numeric column should have stats
    assert len(body["numeric_stats"]) > 0
    # Each stat entry must have mean, std, min, max, median
    for col_stats in body["numeric_stats"].values():
        for key in ("mean", "std", "min", "max", "median"):
            assert key in col_stats, f"Missing '{key}' in numeric_stats"


# ---------------------------------------------------------------------------
# /v1/models
# ---------------------------------------------------------------------------


def test_models_returns_200():
    r = client.get("/v1/models")
    assert r.status_code == 200


def test_models_lists_all_three_types():
    r = client.get("/v1/models")
    body = r.json()
    assert "models" in body
    model_names = set(body["models"].keys())
    assert model_names == {"random_forest_onehot", "xgboost_ordinal", "catboost_native"}


def test_models_has_hyperparameters():
    r = client.get("/v1/models")
    body = r.json()
    for name, spec in body["models"].items():
        assert "hyperparameters" in spec, f"{name} missing 'hyperparameters'"
        assert len(spec["hyperparameters"]) > 0


# ---------------------------------------------------------------------------
# /v1/train
# ---------------------------------------------------------------------------


def test_train_returns_200():
    r = client.post("/v1/train", json={"cv_splits": 2, "random_state": 42})
    assert r.status_code == 200


def test_train_returns_three_models():
    r = client.post("/v1/train", json={"cv_splits": 2, "random_state": 42})
    body = r.json()
    assert "leaderboard" in body
    assert len(body["leaderboard"]) == 3


def test_train_leaderboard_has_roc_auc():
    r = client.post("/v1/train", json={"cv_splits": 2, "random_state": 42})
    body = r.json()
    for entry in body["leaderboard"]:
        assert "roc_auc_mean" in entry
        assert entry["roc_auc_mean"] > 0.5, f"Model {entry['name']} ROC-AUC too low"


def test_train_leaderboard_sorted_descending():
    r = client.post("/v1/train", json={"cv_splits": 2, "random_state": 42})
    body = r.json()
    means = [e["roc_auc_mean"] for e in body["leaderboard"]]
    assert means == sorted(means, reverse=True), f"Leaderboard not sorted: {means}"


def test_train_populates_registry():
    client.post("/v1/train", json={"cv_splits": 2, "random_state": 42})
    registered = registry.list_models()
    assert len(registered) == 3


# ---------------------------------------------------------------------------
# /v1/predict
# ---------------------------------------------------------------------------


def test_predict_requires_prior_training():
    """Predict without training should return 400."""
    r = client.post("/v1/predict", json={"features": {"age": 17}})
    assert r.status_code == 400


def test_predict_returns_probability_after_training():
    """After training, predict returns a valid probability."""
    client.post("/v1/train", json={"cv_splits": 2, "random_state": 42})

    # Build a valid feature dict using the data profile
    profile = client.get("/v1/data-profile").json()
    # Create a minimal feature dict with median values for numeric columns
    features: dict = {}
    for col, stats in profile["numeric_stats"].items():
        features[col] = stats["median"]
    # Fill categorical columns with a plausible value
    # We don't know exact categories so we use "GP" (common school value)
    for col in profile["categorical_columns"]:
        features[col] = "GP"

    r = client.post("/v1/predict", json={"features": features})
    assert r.status_code == 200
    body = r.json()
    assert "probability_class_1" in body
    prob = body["probability_class_1"]
    assert 0.0 <= prob <= 1.0, f"Probability out of range: {prob}"
    assert "predicted_class" in body
    assert body["predicted_class"] in (0, 1)


# ---------------------------------------------------------------------------
# Legacy endpoint smoke tests
# ---------------------------------------------------------------------------


def test_legacy_compare():
    r = client.post("/v1/compare", json={"cv_splits": 2})
    assert r.status_code == 200
    body = r.json()
    assert "leaderboard" in body
    assert "data_source" in body
