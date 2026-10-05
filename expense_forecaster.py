import os
import joblib
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

SEED = 42

class ExpenseForecasterPipeline:
    def __init__(self):
        self.model = XGBRegressor(
            n_estimators=150,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=SEED,
            n_jobs=-1
        )
        self.feature_cols = ["lag_1", "lag_2", "lag_3", "rolling_mean_3", "month"]
        self.is_trained = False

    def create_features(self, df):
        """Extract time-series features (lags, rolling stats, month seasonality)."""
        df = df.copy()
        df["ds"] = pd.to_datetime(df["ds"])
        df = df.sort_values(by=["user_id", "ds"]).reset_index(drop=True)
        
        # Shift lags per user
        df["lag_1"] = df.groupby("user_id")["total_expense"].shift(1)
        df["lag_2"] = df.groupby("user_id")["total_expense"].shift(2)
        df["lag_3"] = df.groupby("user_id")["total_expense"].shift(3)
        
        # Rolling mean feature
        df["rolling_mean_3"] = df.groupby("user_id")["lag_1"].transform(lambda x: x.rolling(3, min_periods=1).mean())
        
        # Month feature (1 to 12)
        df["month"] = df["ds"].dt.month
        
        # Drop rows with NaNs from lag creation
        clean_df = df.dropna(subset=["lag_1", "lag_2", "lag_3"]).reset_index(drop=True)
        return clean_df

    def train_and_evaluate(self, data_path, output_dir=None):
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Dataset not found at {data_path}")

        raw_df = pd.read_csv(data_path)
        feature_df = self.create_features(raw_df)

        # Split into Train (first 30 months) and Test (last 6 months) to evaluate 6-month forecast
        unique_dates = sorted(feature_df["ds"].unique())
        split_date = unique_dates[-6] # Last 6 months held out for evaluation

        train_df = feature_df[feature_df["ds"] < split_date].copy()
        test_df = feature_df[feature_df["ds"] >= split_date].copy()

        X_train = train_df[self.feature_cols]
        y_train = train_df["total_expense"]

        X_test = test_df[self.feature_cols]
        y_test = test_df["total_expense"]

        # Train XGBoost Regressor
        self.model.fit(X_train, y_train)
        self.is_trained = True

        # Predict on held-out 6-month test period
        y_pred = self.model.predict(X_test)

        # Programmatic RMSE and MAE calculation
        mse = mean_squared_error(y_test, y_pred)
        rmse = float(np.sqrt(mse))
        mae = float(mean_absolute_error(y_test, y_pred))

        metrics = {
            "model_name": "XGBoost Expense Forecaster",
            "task_type": "Regression",
            "num_train_records": len(y_train),
            "num_test_records": len(y_test),
            "test_rmse": round(rmse, 4),
            "test_mae": round(mae, 4),
            "forecast_horizon_months": 6
        }

        print("\n=== XGBoost Expense Forecaster Evaluation ===")
        print(f"Test RMSE: ${rmse:.2f}")
        print(f"Test MAE:  ${mae:.2f}")

        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            model_path = os.path.join(output_dir, "xgb_forecast_model.joblib")
            joblib.dump(self.model, model_path)
            print(f"Saved XGBoost forecast model to {model_path}")

        return metrics

    def forecast_next_months(self, recent_monthly_expenses, months_ahead=6):
        """Generates 6-month iterative forecast given recent historical monthly expenses."""
        if not self.is_trained:
            raise RuntimeError("Model is not trained yet.")

        # Ensure we have at least 3 historical months
        if len(recent_monthly_expenses) < 3:
            # Fallback padding if less than 3 months provided
            padding = [np.mean(recent_monthly_expenses) if len(recent_monthly_expenses) > 0 else 2000.0] * (3 - len(recent_monthly_expenses))
            history = padding + list(recent_monthly_expenses)
        else:
            history = list(recent_monthly_expenses)

        forecasts = []
        last_date = pd.to_datetime("today")
        current_history = history.copy()

        for step in range(1, months_ahead + 1):
            next_date = last_date + pd.DateOffset(months=step)
            target_month = next_date.month

            l1 = current_history[-1]
            l2 = current_history[-2]
            l3 = current_history[-3]
            r_mean = np.mean([l1, l2, l3])

            features = np.array([[l1, l2, l3, r_mean, target_month]])
            pred_val = float(self.model.predict(features)[0])
            pred_val = max(100.0, pred_val) # non-negative clip

            forecasts.append({
                "month_offset": step,
                "ds": next_date.strftime("%Y-%m"),
                "predicted_expense": round(pred_val, 2)
            })
            current_history.append(pred_val)

        return forecasts

    def load_artifacts(self, model_path):
        self.model = joblib.load(model_path)
        self.is_trained = True
