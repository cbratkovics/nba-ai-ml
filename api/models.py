"""
Pydantic models for API requests and responses
"""
from pydantic import BaseModel, Field, validator
from typing import Optional, List, Dict, Any
from datetime import datetime, date
from enum import Enum


class ModelVersion(str, Enum):
    """Available model versions"""
    V1_0 = "v1.0.0"
    V2_0 = "v2.0.0"
    V2_1 = "v2.1.0"
    LATEST = "latest"


class PredictionTarget(str, Enum):
    """Prediction targets"""
    POINTS = "points"
    REBOUNDS = "rebounds"
    ASSISTS = "assists"
    ALL = "all"


class PredictionRequest(BaseModel):
    """Request model for predictions"""
    player_id: str = Field(..., description="NBA player ID")
    game_date: date = Field(..., description="Game date for prediction")
    opponent_team: str = Field(..., description="Opponent team abbreviation (e.g., 'LAL')")
    targets: Optional[List[PredictionTarget]] = Field(
        default=[PredictionTarget.ALL],
        description="Statistics to predict"
    )
    model_version: Optional[ModelVersion] = Field(
        default=ModelVersion.LATEST,
        description="Model version to use"
    )
    include_explanation: Optional[bool] = Field(
        default=False,
        description="Include natural language explanation"
    )
    include_confidence_intervals: Optional[bool] = Field(
        default=False,
        description="Include 95% confidence intervals"
    )
    
    @validator('game_date')
    def validate_game_date(cls, v):
        # Allow past dates for testing and historical analysis
        # In production, you might want to enforce future dates only
        return v
    
    class Config:
        schema_extra = {
            "example": {
                "player_id": "203999",
                "game_date": "2025-01-20",
                "opponent_team": "LAL",
                "targets": ["points", "rebounds", "assists"],
                "model_version": "latest",
                "include_explanation": True
            }
        }


class PredictionResponse(BaseModel):
    """Response model for predictions"""
    player_id: str
    player_name: str
    game_date: date
    opponent_team: str
    predictions: Dict[str, float] = Field(..., description="Predicted statistics")
    confidence: float = Field(..., description="Overall prediction confidence (0-1)")
    confidence_intervals: Optional[Dict[str, Dict[str, float]]] = Field(
        None,
        description="95% confidence intervals for each prediction"
    )
    model_version: str
    model_accuracy: Dict[str, float] = Field(..., description="Historical model accuracy metrics")
    explanation: Optional[str] = Field(None, description="Natural language explanation")
    factors: Optional[List[Dict[str, Any]]] = Field(
        None,
        description="Key factors influencing the prediction"
    )
    prediction_id: str = Field(..., description="Unique prediction identifier")
    timestamp: datetime = Field(default_factory=datetime.now)
    
    class Config:
        schema_extra = {
            "example": {
                "player_id": "203999",
                "player_name": "Nikola Jokic",
                "game_date": "2025-01-20",
                "opponent_team": "LAL",
                "predictions": {
                    "points": 27.3,
                    "rebounds": 13.8,
                    "assists": 8.2
                },
                "confidence": 0.92,
                "confidence_intervals": {
                    "points": {"lower": 22.1, "upper": 32.5},
                    "rebounds": {"lower": 10.2, "upper": 17.4},
                    "assists": {"lower": 5.8, "upper": 10.6}
                },
                "model_version": "v2.1.0",
                "model_accuracy": {
                    "r2_score": 0.942,
                    "mae": 3.1,
                    "rmse": 4.2
                },
                "explanation": "Jokic is expected to have a strong performance...",
                "factors": [
                    {"factor": "10-game average", "value": 28.5, "impact": "positive"},
                    {"factor": "rest days", "value": 2, "impact": "positive"},
                    {"factor": "opponent defense rank", "value": 18, "impact": "neutral"}
                ],
                "prediction_id": "pred_20250120_203999_LAL",
                "timestamp": "2025-01-15T10:30:00"
            }
        }


class BatchPredictionRequest(BaseModel):
    """Request model for batch predictions"""
    predictions: List[PredictionRequest]
    
    @validator('predictions')
    def validate_batch_size(cls, v):
        if len(v) > 100:
            raise ValueError("Batch size cannot exceed 100 predictions")
        return v


class BatchPredictionResponse(BaseModel):
    """Response model for batch predictions"""
    predictions: List[PredictionResponse]
    batch_id: str
    processing_time: float
    timestamp: datetime


class ExperimentRequest(BaseModel):
    """Request model for A/B testing experiments"""
    name: str = Field(..., description="Experiment name")
    description: str = Field(..., description="Experiment description")
    control_model: ModelVersion = Field(..., description="Control model version")
    treatment_model: ModelVersion = Field(..., description="Treatment model version")
    traffic_percentage: float = Field(
        default=50.0,
        ge=0,
        le=100,
        description="Percentage of traffic for treatment model"
    )
    start_date: datetime = Field(default_factory=datetime.now)
    end_date: Optional[datetime] = None
    
    class Config:
        schema_extra = {
            "example": {
                "name": "neural_net_vs_ensemble",
                "description": "Testing neural network against ensemble",
                "control_model": "v2.0.0",
                "treatment_model": "v2.1.0",
                "traffic_percentage": 50.0,
                "start_date": "2025-01-15T00:00:00",
                "end_date": "2025-01-30T00:00:00"
            }
        }


class ExperimentResponse(BaseModel):
    """Response model for experiment results"""
    experiment_id: str
    name: str
    status: str = Field(..., description="active, completed, or paused")
    control_metrics: Dict[str, float]
    treatment_metrics: Dict[str, float]
    lift: Dict[str, float] = Field(..., description="Improvement metrics")
    sample_size: Dict[str, int]
    statistical_significance: Dict[str, float]
    recommendation: str
    created_at: datetime
    updated_at: datetime
    
    class Config:
        schema_extra = {
            "example": {
                "experiment_id": "exp_20250115_001",
                "name": "neural_net_vs_ensemble",
                "status": "active",
                "control_metrics": {
                    "r2_score": 0.935,
                    "mae": 3.3,
                    "rmse": 4.4
                },
                "treatment_metrics": {
                    "r2_score": 0.942,
                    "mae": 3.1,
                    "rmse": 4.2
                },
                "lift": {
                    "r2_improvement": 0.007,
                    "mae_reduction": 0.2,
                    "rmse_reduction": 0.2
                },
                "sample_size": {
                    "control": 5000,
                    "treatment": 5000
                },
                "statistical_significance": {
                    "p_value": 0.03,
                    "confidence": 0.97
                },
                "recommendation": "Treatment model shows significant improvement",
                "created_at": "2025-01-15T00:00:00",
                "updated_at": "2025-01-20T12:00:00"
            }
        }


class InsightRequest(BaseModel):
    """Request model for player insights"""
    player_id: str
    period: Optional[str] = Field(
        default="season",
        description="Time period: 'recent', 'season', 'career'"
    )
    include_predictions: Optional[bool] = Field(
        default=True,
        description="Include future game predictions"
    )


class InsightResponse(BaseModel):
    """Response model for player insights"""
    player_id: str
    player_name: str
    insights: str = Field(..., description="LLM-generated insights")
    key_stats: Dict[str, float]
    trends: List[Dict[str, Any]]
    strengths: List[str]
    areas_for_improvement: List[str]
    upcoming_predictions: Optional[List[Dict[str, Any]]]
    generated_at: datetime
    
    class Config:
        schema_extra = {
            "example": {
                "player_id": "203999",
                "player_name": "Nikola Jokic",
                "insights": "Jokic continues to demonstrate elite playmaking...",
                "key_stats": {
                    "ppg": 26.8,
                    "rpg": 13.2,
                    "apg": 8.9,
                    "fg_pct": 0.582
                },
                "trends": [
                    {"metric": "assists", "direction": "up", "change": 1.2},
                    {"metric": "turnovers", "direction": "down", "change": -0.5}
                ],
                "strengths": [
                    "Elite court vision and passing",
                    "Exceptional rebounding for position",
                    "High shooting efficiency"
                ],
                "areas_for_improvement": [
                    "Three-point volume",
                    "Defensive intensity"
                ],
                "upcoming_predictions": [
                    {
                        "date": "2025-01-20",
                        "opponent": "LAL",
                        "predicted_points": 27.3
                    }
                ],
                "generated_at": "2025-01-15T10:30:00"
            }
        }


class HealthCheckResponse(BaseModel):
    """Response model for health checks"""
    status: str
    version: str
    timestamp: datetime
    services: Dict[str, str]
    metrics: Dict[str, Any]
    
    class Config:
        schema_extra = {
            "example": {
                "status": "healthy",
                "version": "2.1.0",
                "timestamp": "2025-01-15T10:30:00",
                "services": {
                    "database": "connected",
                    "redis": "connected",
                    "model_service": "ready"
                },
                "metrics": {
                    "requests_per_second": 150,
                    "average_latency_ms": 45,
                    "error_rate": 0.001,
                    "model_cache_hit_rate": 0.92
                }
            }
        }


class ErrorResponse(BaseModel):
    """Standard error response"""
    error: str
    status_code: int
    path: str
    timestamp: datetime
    request_id: Optional[str] = None
    
    class Config:
        schema_extra = {
            "example": {
                "error": "Player not found",
                "status_code": 404,
                "path": "/v1/predict",
                "timestamp": "2025-01-15T10:30:00",
                "request_id": "req_abc123"
            }
        }