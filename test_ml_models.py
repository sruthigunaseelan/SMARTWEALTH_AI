import os
import sys
import json
import pytest

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.ml.category_classifier import CategoryClassifierPipeline
from src.ml.expense_forecaster import ExpenseForecasterPipeline
from src.ml.anomaly_detector import AnomalyDetectorPipeline

MODELS_DIR = os.path.join(BASE_DIR, "saved_models")

def test_metrics_file_exists():
    metrics_path = os.path.join(MODELS_DIR, "metrics.json")
    assert os.path.exists(metrics_path), "saved_models/metrics.json does not exist. Run train_all.py."
    with open(metrics_path, "r") as f:
        data = json.load(f)
    assert "category_classifier" in data
    assert "expense_forecaster" in data
    assert "anomaly_detector" in data

def test_category_classifier_accuracy_threshold():
    """Requirement: Held-out test accuracy MUST be > 80%."""
    metrics_path = os.path.join(MODELS_DIR, "metrics.json")
    with open(metrics_path, "r") as f:
        data = json.load(f)
    test_acc = data["category_classifier"]["test_accuracy"]
    assert test_acc >= 0.80, f"Category classifier accuracy {test_acc} is below 0.80 requirement!"

def test_category_classifier_inference():
    clf = CategoryClassifierPipeline()
    vec_path = os.path.join(MODELS_DIR, "tfidf_vectorizer.joblib")
    model_path = os.path.join(MODELS_DIR, "rf_category_model.joblib")
    clf.load_artifacts(vec_path, model_path)
    
    samples = ["WALMART SUPERCENTER GROCERY", "NETFLIX DIGITAL MONTHLY STREAMING"]
    results = clf.predict(samples)
    assert len(results) == 2
    assert results[0]["category"] == "Groceries"
    assert results[1]["category"] == "Subscriptions & Streaming"
    assert results[0]["confidence"] > 0.0

def test_expense_forecaster_inference():
    fc = ExpenseForecasterPipeline()
    model_path = os.path.join(MODELS_DIR, "xgb_forecast_model.joblib")
    fc.load_artifacts(model_path)
    
    history = [2200.0, 2400.0, 2100.0, 2300.0, 2500.0, 2700.0]
    forecasts = fc.forecast_next_months(history, months_ahead=6)
    assert len(forecasts) == 6
    for item in forecasts:
        assert item["predicted_expense"] > 0.0
        assert "ds" in item

def test_anomaly_detector_inference():
    ad = AnomalyDetectorPipeline()
    scaler_path = os.path.join(MODELS_DIR, "anomaly_scaler.joblib")
    model_path = os.path.join(MODELS_DIR, "isolation_forest.joblib")
    ad.load_artifacts(scaler_path, model_path)
    
    sample_txs = [
        {"tx_id": "T1", "amount": 25.0, "price_change_pct": 0.0, "merchant_rarity_score": 0.1},
        {"tx_id": "T2", "amount": 5500.0, "price_change_pct": 2.5, "merchant_rarity_score": 0.95}
    ]
    results = ad.scan_transactions(sample_txs)
    assert len(results) == 2
    assert results[1]["is_anomaly"] is True
