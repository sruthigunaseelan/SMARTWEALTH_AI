import os
from src.backend.config import settings
from src.ml.category_classifier import CategoryClassifierPipeline

class ClassifierService:
    def __init__(self):
        self.pipeline = CategoryClassifierPipeline()
        self._load_model()

    def _load_model(self):
        vec_path = os.path.join(settings.MODELS_DIR, "tfidf_vectorizer.joblib")
        model_path = os.path.join(settings.MODELS_DIR, "rf_category_model.joblib")
        
        if os.path.exists(vec_path) and os.path.exists(model_path):
            self.pipeline.load_artifacts(vec_path, model_path)
            print("Loaded Category Classifier model into memory.")
        else:
            print("Warning: Category Classifier model artifacts not found. Please run training pipeline.")

    def predict_batch(self, descriptions):
        return self.pipeline.predict(descriptions)

classifier_service = ClassifierService()
