import os
from src.backend.config import settings
from src.ml.anomaly_detector import AnomalyDetectorPipeline

class AnomalyService:
    def __init__(self):
        self.pipeline = AnomalyDetectorPipeline()
        self._load_model()

    def _load_model(self):
        scaler_path = os.path.join(settings.MODELS_DIR, "anomaly_scaler.joblib")
        model_path = os.path.join(settings.MODELS_DIR, "isolation_forest.joblib")
        
        if os.path.exists(scaler_path) and os.path.exists(model_path):
            self.pipeline.load_artifacts(scaler_path, model_path)
            print("Loaded Isolation Forest Anomaly Detector into memory.")
        else:
            print("Warning: Isolation Forest artifacts not found. Please run training pipeline.")

    def scan_batch(self, transactions):
        return self.pipeline.scan_transactions(transactions)

anomaly_service = AnomalyService()
