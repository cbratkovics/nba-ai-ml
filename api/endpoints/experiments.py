"""
A/B testing experiment endpoints
"""
from fastapi import APIRouter, Depends, HTTPException
from typing import List
import uuid
from datetime import datetime
import logging

from api.models import ExperimentRequest, ExperimentResponse
from api.middleware.auth import verify_api_key

logger = logging.getLogger(__name__)

router = APIRouter()

# Mock experiment storage (would be database in production)
experiments = {}

@router.post("/experiments", response_model=dict)
async def create_experiment(
    request: ExperimentRequest,
    api_key: str = Depends(verify_api_key)
):
    """Create new A/B testing experiment"""
    experiment_id = f"exp_{datetime.now():%Y%m%d}_{uuid.uuid4().hex[:8]}"
    
    experiment = {
        "id": experiment_id,
        "name": request.name,
        "description": request.description,
        "control_model": request.control_model,
        "treatment_model": request.treatment_model,
        "traffic_percentage": request.traffic_percentage,
        "status": "active",
        "created_at": datetime.now(),
        "updated_at": datetime.now(),
        "metrics": {
            "control": {"requests": 0, "accuracy": 0},
            "treatment": {"requests": 0, "accuracy": 0}
        }
    }
    
    experiments[experiment_id] = experiment
    
    return {
        "experiment_id": experiment_id,
        "message": "Experiment created successfully",
        "status": "active"
    }

@router.get("/experiments/{experiment_id}", response_model=ExperimentResponse)
async def get_experiment_results(
    experiment_id: str,
    api_key: str = Depends(verify_api_key)
):
    """Get A/B testing experiment results"""
    if experiment_id not in experiments:
        raise HTTPException(status_code=404, detail="Experiment not found")
    
    exp = experiments[experiment_id]
    
    # Placeholder results. This endpoint is a demonstration of the experiment API shape,
    # not a report of a real experiment. There is no metrics store wired up behind it, so
    # it must not return numbers that could be mistaken for measured model performance.
    control_metrics = {
        "r2_score": None,
        "mae": None,
        "rmse": None
    }
    
    treatment_metrics = {
        "r2_score": None,
        "mae": None,
        "rmse": None
    }
    
    lift = {
        "r2_improvement": None,
        "mae_reduction": None,
        "rmse_reduction": None
    }
    
    sample_size = {
        "control": exp["metrics"]["control"]["requests"],
        "treatment": exp["metrics"]["treatment"]["requests"]
    }
    
    # No significance test is performed. Reporting a p-value without an actual
    # experiment would be a fabricated statistic.
    significance = {
        "p_value": None,
        "confidence": None
    }
    
    recommendation = "No experiment metrics are recorded; connect a metrics store to populate this endpoint"
    
    return ExperimentResponse(
        experiment_id=experiment_id,
        name=exp["name"],
        status=exp["status"],
        control_metrics=control_metrics,
        treatment_metrics=treatment_metrics,
        lift=lift,
        sample_size=sample_size,
        statistical_significance=significance,
        recommendation=recommendation,
        created_at=exp["created_at"],
        updated_at=exp["updated_at"]
    )

@router.get("/experiments", response_model=List[dict])
async def list_experiments(api_key: str = Depends(verify_api_key)):
    """List all A/B testing experiments"""
    return [
        {
            "experiment_id": exp_id,
            "name": exp["name"],
            "status": exp["status"],
            "created_at": exp["created_at"]
        }
        for exp_id, exp in experiments.items()
    ]