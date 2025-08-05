"""
Enhanced experiment API endpoints for A/B testing
"""
from fastapi import APIRouter, Depends, HTTPException, Query
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field
from datetime import datetime
import logging

from api.middleware.auth import verify_api_key
from api.ml.experiments import ExperimentManager, ExperimentStatus

logger = logging.getLogger(__name__)

router = APIRouter()

# Initialize experiment manager
experiment_manager = ExperimentManager()


class CreateExperimentRequest(BaseModel):
    """Request to create an experiment"""
    name: str = Field(..., description="Experiment name")
    description: str = Field(..., description="Experiment description")
    variants: Dict[str, Dict[str, Any]] = Field(..., description="Variant configurations")
    traffic_allocation: Dict[str, float] = Field(..., description="Traffic split between variants")
    success_metrics: List[str] = Field(..., description="Metrics to optimize")
    guardrail_metrics: Optional[List[str]] = Field(None, description="Metrics that shouldn't degrade")
    minimum_sample_size: int = Field(1000, description="Minimum samples per variant")


class TrackMetricRequest(BaseModel):
    """Request to track a metric"""
    experiment_id: str
    variant_id: str
    metric_name: str
    value: float
    user_id: Optional[str] = None


class ExperimentAssignmentRequest(BaseModel):
    """Request for experiment assignment"""
    user_id: str
    attributes: Optional[Dict[str, Any]] = None


@router.post("/experiments/create", response_model=Dict[str, str])
async def create_experiment(
    request: CreateExperimentRequest,
    api_key: str = Depends(verify_api_key)
):
    """
    Create a new A/B testing experiment
    
    This endpoint allows you to create experiments with multiple variants
    to test different model configurations, features, or parameters.
    """
    try:
        experiment_id = experiment_manager.create_experiment(
            name=request.name,
            description=request.description,
            variants=request.variants,
            traffic_allocation=request.traffic_allocation,
            success_metrics=request.success_metrics,
            guardrail_metrics=request.guardrail_metrics,
            minimum_sample_size=request.minimum_sample_size,
            created_by=api_key
        )
        
        return {
            "experiment_id": experiment_id,
            "status": "created",
            "message": f"Experiment '{request.name}' created successfully"
        }
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error creating experiment: {e}")
        raise HTTPException(status_code=500, detail="Failed to create experiment")


@router.post("/experiments/{experiment_id}/start")
async def start_experiment(
    experiment_id: str,
    api_key: str = Depends(verify_api_key)
):
    """
    Start an experiment
    
    Activates the experiment and begins traffic allocation.
    """
    try:
        success = experiment_manager.start_experiment(experiment_id)
        
        if success:
            return {
                "experiment_id": experiment_id,
                "status": "started",
                "message": "Experiment started successfully"
            }
        else:
            raise HTTPException(status_code=400, detail="Failed to start experiment")
            
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error starting experiment {experiment_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to start experiment")


@router.get("/experiments/{experiment_id}/results")
async def get_experiment_results(
    experiment_id: str,
    api_key: str = Depends(verify_api_key)
):
    """
    Get current results for an experiment
    
    Returns detailed metrics, statistical analysis, and recommendations.
    """
    try:
        results = experiment_manager.get_experiment_results(experiment_id)
        return results
        
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error getting results for experiment {experiment_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to get experiment results")


@router.post("/experiments/{experiment_id}/conclude")
async def conclude_experiment(
    experiment_id: str,
    winning_variant: Optional[str] = None,
    conclusion_notes: str = "",
    api_key: str = Depends(verify_api_key)
):
    """
    Conclude an experiment
    
    Stops the experiment and optionally declares a winner.
    """
    try:
        success = experiment_manager.conclude_experiment(
            experiment_id=experiment_id,
            winning_variant=winning_variant,
            conclusion_notes=conclusion_notes
        )
        
        if success:
            return {
                "experiment_id": experiment_id,
                "status": "concluded",
                "winning_variant": winning_variant,
                "message": "Experiment concluded successfully"
            }
        else:
            raise HTTPException(status_code=400, detail="Failed to conclude experiment")
            
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error concluding experiment {experiment_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to conclude experiment")


@router.get("/experiments/active")
async def get_active_experiments(
    api_key: str = Depends(verify_api_key)
):
    """
    Get all currently active experiments
    
    Returns a list of experiments that are currently running.
    """
    try:
        active_experiments = experiment_manager.get_active_experiments()
        
        return {
            "count": len(active_experiments),
            "experiments": [
                {
                    "experiment_id": exp.experiment_id,
                    "name": exp.name,
                    "description": exp.description,
                    "variants": list(exp.variants.keys()),
                    "created_at": exp.created_at,
                    "status": exp.status.value
                }
                for exp in active_experiments
            ]
        }
        
    except Exception as e:
        logger.error(f"Error getting active experiments: {e}")
        raise HTTPException(status_code=500, detail="Failed to get active experiments")


@router.post("/experiments/{experiment_id}/assignment")
async def get_variant_assignment(
    experiment_id: str,
    request: ExperimentAssignmentRequest,
    api_key: str = Depends(verify_api_key)
):
    """
    Get variant assignment for a user
    
    Returns the assigned variant for A/B testing.
    """
    try:
        variant = experiment_manager.get_variant_assignment(
            experiment_id=experiment_id,
            user_id=request.user_id,
            attributes=request.attributes
        )
        
        if variant:
            return {
                "experiment_id": experiment_id,
                "user_id": request.user_id,
                "variant": variant,
                "timestamp": datetime.now().isoformat()
            }
        else:
            return {
                "experiment_id": experiment_id,
                "user_id": request.user_id,
                "variant": None,
                "message": "User not eligible for experiment"
            }
            
    except Exception as e:
        logger.error(f"Error getting assignment for experiment {experiment_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to get variant assignment")


@router.post("/experiments/track")
async def track_metric(
    request: TrackMetricRequest,
    api_key: str = Depends(verify_api_key)
):
    """
    Track a metric value for an experiment
    
    Records metric data for analysis.
    """
    try:
        experiment_manager.track_metric(
            experiment_id=request.experiment_id,
            variant_id=request.variant_id,
            metric_name=request.metric_name,
            value=request.value,
            user_id=request.user_id
        )
        
        return {
            "status": "tracked",
            "experiment_id": request.experiment_id,
            "variant_id": request.variant_id,
            "metric_name": request.metric_name,
            "value": request.value
        }
        
    except Exception as e:
        logger.error(f"Error tracking metric: {e}")
        raise HTTPException(status_code=500, detail="Failed to track metric")


@router.get("/experiments/recommendations")
async def get_experiment_recommendations(
    min_confidence: float = Query(0.95, description="Minimum confidence level"),
    api_key: str = Depends(verify_api_key)
):
    """
    Get recommendations across all experiments
    
    Provides actionable insights based on experiment results.
    """
    try:
        recommendations = []
        
        # Get all active experiments
        active_experiments = experiment_manager.get_active_experiments()
        
        for experiment in active_experiments:
            try:
                results = experiment_manager.get_experiment_results(experiment.experiment_id)
                
                # Check for significant winners
                for metric, sig_results in results.get("statistical_significance", {}).items():
                    for variant_id, sig_data in sig_results.items():
                        if sig_data.get("significant") and sig_data.get("confidence_level", 0) >= min_confidence:
                            recommendations.append({
                                "experiment_id": experiment.experiment_id,
                                "experiment_name": experiment.name,
                                "variant": variant_id,
                                "metric": metric,
                                "improvement": sig_data.get("relative_improvement", 0),
                                "confidence": sig_data.get("confidence_level", 0),
                                "action": "Consider promoting this variant"
                            })
                            
            except Exception as e:
                logger.error(f"Error getting recommendations for {experiment.experiment_id}: {e}")
                continue
        
        return {
            "count": len(recommendations),
            "min_confidence": min_confidence,
            "recommendations": recommendations
        }
        
    except Exception as e:
        logger.error(f"Error getting experiment recommendations: {e}")
        raise HTTPException(status_code=500, detail="Failed to get recommendations")


# Example experiment configurations for demo
DEMO_EXPERIMENTS = {
    "model_comparison": {
        "name": "Model Algorithm Comparison",
        "description": "Compare Random Forest vs XGBoost vs Neural Network",
        "variants": {
            "control_rf": {"model": "random_forest", "params": {"n_estimators": 100}},
            "treatment_xgb": {"model": "xgboost", "params": {"n_estimators": 100}},
            "treatment_nn": {"model": "neural_network", "params": {"hidden_layers": [128, 64]}}
        },
        "traffic_allocation": {"control_rf": 0.4, "treatment_xgb": 0.3, "treatment_nn": 0.3},
        "success_metrics": ["mae", "prediction_accuracy"],
        "guardrail_metrics": ["latency_ms", "memory_usage_mb"]
    },
    "feature_set_test": {
        "name": "Feature Set Comparison",
        "description": "Test basic vs advanced feature sets",
        "variants": {
            "basic_features": {"features": ["last_10_avg", "season_avg"]},
            "advanced_features": {"features": ["last_10_avg", "season_avg", "opponent_specific", "fatigue_index"]}
        },
        "traffic_allocation": {"basic_features": 0.5, "advanced_features": 0.5},
        "success_metrics": ["mae", "r2_score"],
        "guardrail_metrics": ["feature_calculation_time_ms"]
    }
}


@router.post("/experiments/create-demo")
async def create_demo_experiment(
    experiment_type: str = Query(..., description="Type of demo experiment"),
    api_key: str = Depends(verify_api_key)
):
    """
    Create a demo experiment for showcase
    
    Available types: model_comparison, feature_set_test
    """
    if experiment_type not in DEMO_EXPERIMENTS:
        raise HTTPException(
            status_code=400, 
            detail=f"Invalid experiment type. Choose from: {list(DEMO_EXPERIMENTS.keys())}"
        )
    
    config = DEMO_EXPERIMENTS[experiment_type]
    
    try:
        experiment_id = experiment_manager.create_experiment(
            name=config["name"],
            description=config["description"],
            variants=config["variants"],
            traffic_allocation=config["traffic_allocation"],
            success_metrics=config["success_metrics"],
            guardrail_metrics=config.get("guardrail_metrics", []),
            created_by="demo"
        )
        
        # Auto-start the experiment
        experiment_manager.start_experiment(experiment_id)
        
        return {
            "experiment_id": experiment_id,
            "type": experiment_type,
            "status": "started",
            "message": f"Demo experiment '{config['name']}' created and started"
        }
        
    except Exception as e:
        logger.error(f"Error creating demo experiment: {e}")
        raise HTTPException(status_code=500, detail="Failed to create demo experiment")