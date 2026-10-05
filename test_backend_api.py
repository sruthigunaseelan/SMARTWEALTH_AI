import os
import sys
import pytest
from fastapi.testclient import TestClient

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.backend.main import app

client = TestClient(app)

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"

def test_predict_category_endpoint():
    payload = {
        "transactions": [
            {"description": "WALMART SUPERCENTER BENTONVILLE", "amount": 54.20},
            {"description": "SHELL OIL GAS STATION #102", "amount": 35.00}
        ]
    }
    response = client.post("/api/v1/predict/category", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["count"] == 2
    assert len(data["predictions"]) == 2
    assert "latency_ms" in data
    assert data["predictions"][0]["category"] == "Groceries"
    assert data["predictions"][1]["category"] == "Transportation & Fuel"

def test_get_monthly_forecast_endpoint():
    response = client.get("/api/v1/forecast/monthly?user_id=USER_001&months=6")
    assert response.status_code == 200
    data = response.json()
    assert data["forecast_horizon"] == 6
    assert len(data["forecasts"]) == 6
    assert data["total_forecasted"] > 0
    assert "latency_ms" in data

def test_anomaly_scan_endpoint():
    payload = {
        "transactions": [
            {"tx_id": "TX_01", "description": "COFFEE SHOP", "amount": 4.50, "price_change_pct": 0.0, "merchant_rarity_score": 0.1},
            {"tx_id": "TX_02", "description": "UNKNOWN LUXURY DEPOSIT", "amount": 6200.00, "price_change_pct": 3.0, "merchant_rarity_score": 0.95}
        ]
    }
    response = client.post("/api/v1/anomaly/scan", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["total_scanned"] == 2
    assert data["anomalies_detected"] >= 1
    assert "latency_ms" in data

def test_get_metrics_endpoint():
    response = client.get("/api/v1/metrics")
    assert response.status_code == 200
    data = response.json()
    assert "category_classifier" in data
    assert data["category_classifier"]["test_accuracy"] >= 0.80
