import os
import joblib
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, precision_recall_fscore_support

SEED = 42

class AnomalyDetectorPipeline:
    def __init__(self, contamination=0.15):
        self.scaler = StandardScaler()
        self.model = IsolationForest(
            n_estimators=100,
            contamination=contamination,
            random_state=SEED,
            n_jobs=-1
        )
        self.feature_cols = ["amount", "price_change_pct", "merchant_rarity_score"]
        self.is_trained = False

    def train_and_evaluate(self, data_path, output_dir=None):
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Dataset not found at {data_path}")

        df = pd.read_csv(data_path)
        X_raw = df[self.feature_cols]
        y_true = df["is_anomaly"] # 0 for normal, 1 for anomaly

        # Fit Scaler and Isolation Forest
        X_scaled = self.scaler.fit_transform(X_raw)
        self.model.fit(X_scaled)
        self.is_trained = True

        # Isolation forest returns 1 for inliers, -1 for outliers (anomalies)
        preds_raw = self.model.predict(X_scaled)
        preds_binary = np.where(preds_raw == -1, 1, 0) # Convert -1 -> 1 (anomaly)

        anomaly_scores = -self.model.score_samples(X_scaled) # higher score = more anomalous

        # Metrics evaluation
        precision, recall, f1, _ = precision_recall_fscore_support(y_true, preds_binary, average='binary')

        metrics = {
            "model_name": "Isolation Forest Anomaly Detector",
            "contamination_parameter": self.model.contamination,
            "num_samples_evaluated": len(df),
            "num_anomalies_detected": int(np.sum(preds_binary)),
            "precision": round(float(precision), 4),
            "recall": round(float(recall), 4),
            "f1_score": round(float(f1), 4)
        }

        print("\n=== Isolation Forest Anomaly Detector Evaluation ===")
        print(f"Detected Anomalies: {metrics['num_anomalies_detected']} / {len(df)}")
        print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}")
        print("Detailed Anomaly Detection Report:")
        print(classification_report(y_true, preds_binary, target_names=["Normal", "Anomaly"]))

        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            scaler_path = os.path.join(output_dir, "anomaly_scaler.joblib")
            model_path = os.path.join(output_dir, "isolation_forest.joblib")
            joblib.dump(self.scaler, scaler_path)
            joblib.dump(self.model, model_path)
            print(f"Saved scaler to {scaler_path}")
            print(f"Saved Isolation Forest model to {model_path}")

        return metrics

    def scan_transactions(self, transactions):
        """Scans a batch of transaction dictionaries for anomalies."""
        if not self.is_trained:
            raise RuntimeError("Model is not trained yet.")

        records = []
        for tx in transactions:
            amt = float(tx.get("amount", 0.0))
            p_change = float(tx.get("price_change_pct", 0.0))
            rarity = float(tx.get("merchant_rarity_score", 0.1))
            records.append([amt, p_change, rarity])

        X_scaled = self.scaler.transform(records)
        raw_preds = self.model.predict(X_scaled)
        scores = -self.model.score_samples(X_scaled)

        results = []
        for i, tx in enumerate(transactions):
            is_anom = bool(raw_preds[i] == -1)
            score = float(scores[i])
            
            # Rule-based tag enrichment
            tags = []
            if tx.get("amount", 0.0) > 1000.0:
                tags.append("High Transaction Amount")
            if tx.get("price_change_pct", 0.0) > 0.50:
                tags.append("Unusual Price Increase")
            if tx.get("merchant_rarity_score", 0.0) > 0.80:
                tags.append("Unusual Merchant")
            if is_anom and not tags:
                tags.append("Statistical Outlier")

            results.append({
                "tx_id": tx.get("tx_id", f"TX_{i:04d}"),
                "description": tx.get("raw_description", tx.get("description", "Unknown Merchant")),
                "amount": tx.get("amount", 0.0),
                "is_anomaly": is_anom,
                "anomaly_score": round(score, 4),
                "flagged_reasons": tags
            })
        return results

    def load_artifacts(self, scaler_path, model_path):
        self.scaler = joblib.load(scaler_path)
        self.model = joblib.load(model_path)
        self.is_trained = True
