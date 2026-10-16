
from fastapi.testclient import TestClient

from api.main import app


def test_health_endpoint():
    with TestClient(app) as client:
        response = client.get("/health")

        assert response.status_code == 200
        assert response.json()["status"] == "healthy"
        assert response.json()["model_loaded"] is True


def test_valid_prediction():
    payload = {
        "center_id": "55",
        "meal_id": "1885",
        "week_index": 144,
        "checkout_price": 150.0,
        "base_price": 160.0,
        "emailer_for_promotion": 0,
        "homepage_featured": 0,
        "discount_amount": 10.0,
        "discount_percentage": 6.25,
    }

    with TestClient(app) as client:
        response = client.post("/predict", json=payload)

        assert response.status_code == 200
        assert isinstance(response.json()["predicted_orders"], (int, float))
        assert response.json()["predicted_orders"] >= 0


def test_invalid_prediction_input():
    payload = {
        "center_id": "55",
        "meal_id": "1885",
        "week_index": 144,
        "checkout_price": 0,
        "base_price": 160.0,
        "emailer_for_promotion": 0,
        "homepage_featured": 0,
        "discount_amount": 10.0,
        "discount_percentage": 6.25,
    }

    with TestClient(app) as client:
        response = client.post("/predict", json=payload)

        assert response.status_code == 422
