import os
import sys
import json
import time

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from data.generate_dataset import main as generate_data
from src.ml.category_classifier import CategoryClassifierPipeline
from src.ml.expense_forecaster import ExpenseForecasterPipeline
from src.ml.anomaly_detector import AnomalyDetectorPipeline

DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "saved_models")

def run_master_training():
    print("=========================================================")
    print("      SMARTWEALTH AI - Master Model Training Runner     ")
    print("=========================================================\n")

    start_time = time.time()
    
    # Step 1: Ensure datasets exist
    tx_csv = os.path.join(DATA_DIR, "transactions_dataset.csv")
    expense_csv = os.path.join(DATA_DIR, "monthly_expenses.csv")
    anom_csv = os.path.join(DATA_DIR, "anomalous_transactions.csv")

    if not (os.path.exists(tx_csv) and os.path.exists(expense_csv) and os.path.exists(anom_csv)):
        print("Dataset files missing. Generating reproducible datasets...")
        generate_data()
    else:
        print("Existing reproducible datasets found.")

    os.makedirs(MODELS_DIR, exist_ok=True)

    all_metrics = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "random_seed": 42
    }

    # Step 2: Train Category Classifier (Random Forest + TF-IDF)
    print("\n[1/3] Training Category Classifier (Random Forest + TF-IDF n-grams)...")
    clf = CategoryClassifierPipeline()
    clf_metrics = clf.train_and_evaluate(tx_csv, output_dir=MODELS_DIR)
    all_metrics["category_classifier"] = clf_metrics

    # Check requirement: >80% held-out test accuracy
    if clf_metrics["test_accuracy"] < 0.80:
        raise ValueError(f"Category Classifier failed accuracy requirement: {clf_metrics['test_accuracy'] * 100:.2f}% < 80.0%")
    print(f"[OK] Category Classifier passed requirement! Test Accuracy: {clf_metrics['test_accuracy'] * 100:.2f}%")

    # Step 3: Train Expense Forecaster (XGBoost Regressor)
    print("\n[2/3] Training Expense Forecaster (XGBoost Regressor)...")
    forecaster = ExpenseForecasterPipeline()
    fc_metrics = forecaster.train_and_evaluate(expense_csv, output_dir=MODELS_DIR)
    all_metrics["expense_forecaster"] = fc_metrics
    print(f"[OK] Expense Forecaster trained! Test RMSE: ${fc_metrics['test_rmse']:.2f}, MAE: ${fc_metrics['test_mae']:.2f}")

    # Step 4: Train Anomaly Detector (Isolation Forest)
    print("\n[3/3] Training Anomaly Detector (Isolation Forest)...")
    anomaly_det = AnomalyDetectorPipeline(contamination=0.15)
    anom_metrics = anomaly_det.train_and_evaluate(anom_csv, output_dir=MODELS_DIR)
    all_metrics["anomaly_detector"] = anom_metrics
    print(f"[OK] Anomaly Detector trained! F1-Score: {anom_metrics['f1_score']:.4f}")

    elapsed = time.time() - start_time
    all_metrics["total_training_time_seconds"] = round(elapsed, 2)

    # Export metrics.json
    metrics_json_path = os.path.join(MODELS_DIR, "metrics.json")
    with open(metrics_json_path, "w") as f:
        json.dump(all_metrics, f, indent=2)

    print(f"\n=========================================================")
    print(f"All models trained successfully in {elapsed:.2f} seconds.")
    print(f"Metrics programmatically exported to: {metrics_json_path}")
    print("=========================================================")

    return all_metrics

if __name__ == "__main__":
    run_master_training()
