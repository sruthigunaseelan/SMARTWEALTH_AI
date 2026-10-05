from typing import List, Optional
from pydantic import BaseModel, Field

# Category Prediction Schemas
class CategoryPredictionInput(BaseModel):
    description: str = Field(..., json_schema_extra={"example": "SWIGGY FOOD DELIVERY"})
    amount: Optional[float] = Field(default=0.0, json_schema_extra={"example": 45.50})

class CategoryBatchRequest(BaseModel):
    transactions: List[CategoryPredictionInput]

class CategoryPredictionOutput(BaseModel):
    description: str
    category: str
    confidence: float

class CategoryBatchResponse(BaseModel):
    count: int
    predictions: List[CategoryPredictionOutput]
    latency_ms: float

# Monthly Forecast Schemas
class MonthlyForecastRequest(BaseModel):
    user_id: Optional[str] = Field(default="USER_001", json_schema_extra={"example": "USER_001"})
    recent_history: Optional[List[float]] = Field(
        default=[2200.0, 2450.0, 2100.0, 2350.0, 2600.0, 2800.0],
        description="List of recent monthly totals"
    )
    forecast_horizon: Optional[int] = Field(default=6, ge=1, le=12)

class ForecastMonthItem(BaseModel):
    month_offset: int
    ds: str
    predicted_expense: float

class MonthlyForecastResponse(BaseModel):
    user_id: str
    forecast_horizon: int
    forecasts: List[ForecastMonthItem]
    total_forecasted: float
    latency_ms: float

# Anomaly Scan Schemas
class TransactionScanItem(BaseModel):
    tx_id: Optional[str] = Field(default="TX_001")
    description: str = Field(..., json_schema_extra={"example": "UNKNOWN MERCHANT ONLINE"})
    amount: float = Field(..., json_schema_extra={"example": 1250.0})
    price_change_pct: Optional[float] = Field(default=0.0, json_schema_extra={"example": 1.5})
    merchant_rarity_score: Optional[float] = Field(default=0.1, json_schema_extra={"example": 0.9})

class AnomalyScanRequest(BaseModel):
    transactions: List[TransactionScanItem]

class AnomalyScanResultItem(BaseModel):
    tx_id: str
    description: str
    amount: float
    is_anomaly: bool
    anomaly_score: float
    flagged_reasons: List[str]

class AnomalyScanResponse(BaseModel):
    total_scanned: int
    anomalies_detected: int
    results: List[AnomalyScanResultItem]
    latency_ms: float

# Financial Health Score Schema
class HealthScoreResponse(BaseModel):
    overall_score: int
    health_rating: str
    savings_ratio_pct: float
    spending_stability_score: int
    anomaly_risk_penalty: int
    insights: List[str]

# Goal Tracker Schemas
class GoalInput(BaseModel):
    goal_name: str
    target_amount: float
    current_savings: float
    monthly_contribution: float

class GoalRecommendationResponse(BaseModel):
    goal_name: str
    target_amount: float
    current_savings: float
    months_to_reach_goal: int
    completion_date: str
    ai_saving_nudges: List[str]
