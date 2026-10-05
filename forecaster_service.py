import os
from src.backend.config import settings
from src.ml.expense_forecaster import ExpenseForecasterPipeline

class ForecasterService:
    def __init__(self):
        self.pipeline = ExpenseForecasterPipeline()
        self._load_model()

    def _load_model(self):
        model_path = os.path.join(settings.MODELS_DIR, "xgb_forecast_model.joblib")
        if os.path.exists(model_path):
            self.pipeline.load_artifacts(model_path)
            print("Loaded XGBoost Expense Forecaster model into memory.")
        else:
            print("Warning: XGBoost Forecaster model artifact not found. Please run training pipeline.")

    def forecast_months(self, recent_history, months_ahead=6):
        return self.pipeline.forecast_next_months(recent_history, months_ahead=months_ahead)

forecaster_service = ForecasterService()
