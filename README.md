# SMARTWEALTH AI
> **Machine Learning Based Personal Finance Analysis & Recommendation System**
> College PBL Project

---

## 📌 Project Overview

SMARTWEALTH AI is an end-to-end Machine Learning personal finance system designed to automatically categorize raw merchant transactions, forecast future monthly expenses, and detect financial anomalies such as price hikes or fraudulent-looking charges.

---

## 🚀 Key Features & Architectural Components

1. **Transaction Categorization**:
   - **Model**: Random Forest Classifier with TF-IDF n-grams (1, 2).
   - **Target**: 12 finance categories (`Groceries`, `Dining & Restaurants`, `Utilities & Bills`, `Subscriptions & Streaming`, `Transportation & Fuel`, `Healthcare & Pharmacy`, `Entertainment & Leisure`, `Shopping & Electronics`, `Travel & Lodging`, `Education & Courses`, `Personal Care`, `Investments & Wealth`).
   - **Requirement**: Held-out test accuracy **>80%** (Achieved **100.0%** on held-out test set).

2. **Expense Forecasting**:
   - **Model**: XGBoost Regressor.
   - **Task**: 6-month monthly expense forecast based on historical spending lag features & seasonality.
   - **Evaluation**: Programmatic **RMSE** ($231.73) and **MAE** ($190.83).

3. **Anomaly & Fraud Detection**:
   - **Model**: Isolation Forest.
   - **Task**: Scans transaction amounts, price change percentages, and merchant rarity scores to flag unusual subscriptions, price spikes, and potential fraud.

4. **FastAPI REST API**:
   - `POST /api/v1/predict/category` — Fast batch category prediction.
   - `GET  /api/v1/forecast/monthly` — 6-month expense forecast.
   - `POST /api/v1/anomaly/scan` — Anomaly transaction scanner.
   - `GET  /api/v1/metrics` — Held-out test metric report.

5. **Caching & DB Layer**:
   - PostgreSQL connector with SQLite in-memory fallback.
   - Redis cache connector with fast memory fallback.

6. **Interactive Dashboard**:
   - Modern web dashboard built with React, Tailwind CSS, and Chart.js.

7. **Performance & Latency**:
   - Response latency **<120 ms** for batch queries (Achieved **~2-12 ms** average latency).

---

## 🛠 Quickstart Guide

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Generate Reproducible Datasets & Train Models
```bash
python src/ml/train_all.py
```
This script generates seed datasets (`SEED=42`), trains all 3 ML models, outputs held-out evaluation reports, and exports `.joblib` model weights to `saved_models/metrics.json`.

### 3. Run FastAPI Backend Server
```bash
python -m uvicorn src.backend.main:app --reload --port 8000
```
Open `http://localhost:8000` in your browser to launch the interactive UI dashboard and OpenAPI documentation (`http://localhost:8000/docs`).

### 4. Run Test Suites
```bash
# Unit & Integration Tests
python -m pytest tests/test_ml_models.py tests/test_backend_api.py -v

# Automated API Latency Benchmark (<120 ms verification)
python tests/test_performance.py
```

### 5. Docker Deployment
```bash
docker-compose up --build
```
