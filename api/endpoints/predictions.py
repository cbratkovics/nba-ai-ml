"""
Prediction endpoints for NBA player performance
"""
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from typing import Dict, List
import uuid
import asyncio
import time
import logging
from datetime import datetime

from api.models import (
    PredictionRequest, PredictionResponse, 
    BatchPredictionRequest, BatchPredictionResponse
)
from api.middleware.auth import verify_api_key
from api.middleware.rate_limiting import check_rate_limit
from ml.serving.predictor import PredictionService

logger = logging.getLogger(__name__)

router = APIRouter()

# Initialize prediction service (would be dependency injected in production)
prediction_service = PredictionService()


@router.post("/predict", response_model=PredictionResponse)
async def predict_player_performance(
    request: PredictionRequest,
    background_tasks: BackgroundTasks,
    api_key: str = Depends(verify_api_key),
    rate_limit_check: bool = Depends(check_rate_limit)
):
    """
    Predict player performance for a specific game
    
    This endpoint provides AI-powered predictions for NBA player statistics
    including points, rebounds, assists, and other key metrics.
    """
    start_time = time.time()
    prediction_id = f"pred_{request.game_date}_{request.player_id}_{request.opponent_team}"
    
    try:
        logger.info(f"Processing prediction request: {prediction_id}")
        
        # Get prediction from service
        result = await prediction_service.predict(
            player_id=request.player_id,
            game_date=request.game_date,
            opponent_team=request.opponent_team,
            targets=request.targets,
            model_version=request.model_version,
            include_explanation=request.include_explanation,
            include_confidence_intervals=request.include_confidence_intervals
        )
        
        # Track prediction metrics in background
        background_tasks.add_task(
            track_prediction_metrics,
            prediction_id=prediction_id,
            processing_time=time.time() - start_time,
            model_version=str(request.model_version),
            api_key=api_key
        )
        
        return PredictionResponse(
            player_id=request.player_id,
            player_name=result["player_name"],
            game_date=request.game_date,
            opponent_team=request.opponent_team,
            predictions=result["predictions"],
            confidence=result["confidence"],
            confidence_intervals=result.get("confidence_intervals"),
            model_version=result["model_version"],
            model_accuracy=result["model_accuracy"],
            explanation=result.get("explanation"),
            factors=result.get("factors"),
            prediction_id=prediction_id,
            timestamp=datetime.now()
        )
        
    except ValueError as e:
        logger.error(f"Validation error for {prediction_id}: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error processing prediction {prediction_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/predict/batch", response_model=BatchPredictionResponse)
async def batch_predict(
    request: BatchPredictionRequest,
    background_tasks: BackgroundTasks,
    api_key: str = Depends(verify_api_key),
    rate_limit_check: bool = Depends(check_rate_limit)
):
    """
    Process multiple predictions in a single request
    
    Efficiently handles up to 100 predictions with parallel processing
    for improved performance and reduced API calls.
    """
    start_time = time.time()
    batch_id = f"batch_{datetime.now():%Y%m%d_%H%M%S}_{uuid.uuid4().hex[:8]}"
    
    try:
        logger.info(f"Processing batch prediction: {batch_id} with {len(request.predictions)} requests")
        
        # Process predictions in parallel
        tasks = [
            prediction_service.predict(
                player_id=pred.player_id,
                game_date=pred.game_date,
                opponent_team=pred.opponent_team,
                targets=pred.targets,
                model_version=pred.model_version,
                include_explanation=pred.include_explanation,
                include_confidence_intervals=pred.include_confidence_intervals
            )
            for pred in request.predictions
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results and handle errors
        predictions = []
        for i, (pred_request, result) in enumerate(zip(request.predictions, results)):
            if isinstance(result, Exception):
                logger.error(f"Error in batch item {i}: {str(result)}")
                continue
                
            prediction_id = f"pred_{pred_request.game_date}_{pred_request.player_id}_{pred_request.opponent_team}"
            
            predictions.append(PredictionResponse(
                player_id=pred_request.player_id,
                player_name=result["player_name"],
                game_date=pred_request.game_date,
                opponent_team=pred_request.opponent_team,
                predictions=result["predictions"],
                confidence=result["confidence"],
                confidence_intervals=result.get("confidence_intervals"),
                model_version=result["model_version"],
                model_accuracy=result["model_accuracy"],
                explanation=result.get("explanation"),
                factors=result.get("factors"),
                prediction_id=prediction_id,
                timestamp=datetime.now()
            ))
        
        processing_time = time.time() - start_time
        
        # Track batch metrics
        background_tasks.add_task(
            track_batch_metrics,
            batch_id=batch_id,
            batch_size=len(request.predictions),
            successful_predictions=len(predictions),
            processing_time=processing_time,
            api_key=api_key
        )
        
        return BatchPredictionResponse(
            predictions=predictions,
            batch_id=batch_id,
            processing_time=processing_time,
            timestamp=datetime.now()
        )
        
    except Exception as e:
        logger.error(f"Error processing batch {batch_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/predict/{player_id}/next-game", response_model=PredictionResponse)
async def predict_next_game(
    player_id: str,
    api_key: str = Depends(verify_api_key),
    rate_limit_check: bool = Depends(check_rate_limit)
):
    """
    Predict performance for player's next scheduled game
    
    Automatically determines the next game and opponent from the schedule.
    """
    try:
        # Get next game from schedule
        next_game = await prediction_service.get_next_game(player_id)
        
        if not next_game:
            raise HTTPException(
                status_code=404, 
                detail="No upcoming games found for this player"
            )
        
        # Create prediction request
        request = PredictionRequest(
            player_id=player_id,
            game_date=next_game["date"],
            opponent_team=next_game["opponent"],
            targets=["all"],
            include_explanation=True
        )
        
        # Process prediction
        result = await prediction_service.predict(
            player_id=request.player_id,
            game_date=request.game_date,
            opponent_team=request.opponent_team,
            targets=request.targets,
            include_explanation=True
        )
        
        prediction_id = f"pred_next_{player_id}_{next_game['date']}"
        
        return PredictionResponse(
            player_id=player_id,
            player_name=result["player_name"],
            game_date=request.game_date,
            opponent_team=request.opponent_team,
            predictions=result["predictions"],
            confidence=result["confidence"],
            model_version=result["model_version"],
            model_accuracy=result["model_accuracy"],
            explanation=result.get("explanation"),
            factors=result.get("factors"),
            prediction_id=prediction_id,
            timestamp=datetime.now()
        )
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error predicting next game for {player_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/predict/feedback")
async def submit_prediction_feedback(
    prediction_id: str,
    actual_stats: Dict[str, float],
    api_key: str = Depends(verify_api_key)
):
    """
    Submit actual game results for prediction accuracy tracking
    
    This endpoint allows clients to provide actual game statistics
    for continuous model improvement and accuracy monitoring.
    """
    try:
        # Store feedback for model evaluation
        await prediction_service.store_feedback(prediction_id, actual_stats)
        
        # Calculate accuracy metrics
        accuracy_metrics = await prediction_service.calculate_accuracy(
            prediction_id, actual_stats
        )
        
        logger.info(f"Feedback received for {prediction_id}: {accuracy_metrics}")
        
        return {
            "message": "Feedback received successfully",
            "prediction_id": prediction_id,
            "accuracy_metrics": accuracy_metrics,
            "timestamp": datetime.now()
        }
        
    except Exception as e:
        logger.error(f"Error processing feedback for {prediction_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Error processing feedback")


async def track_prediction_metrics(
    prediction_id: str,
    processing_time: float,
    model_version: str,
    api_key: str
):
    """Background task to track prediction metrics"""
    try:
        # Would integrate with monitoring system
        logger.info(f"Prediction metrics - ID: {prediction_id}, "
                   f"Time: {processing_time:.3f}s, Version: {model_version}")
        
        # Track in time series database
        # metrics_db.record_prediction(
        #     prediction_id=prediction_id,
        #     processing_time=processing_time,
        #     model_version=model_version,
        #     api_key=api_key,
        #     timestamp=datetime.now()
        # )
        
    except Exception as e:
        logger.error(f"Error tracking prediction metrics: {str(e)}")


async def track_batch_metrics(
    batch_id: str,
    batch_size: int,
    successful_predictions: int,
    processing_time: float,
    api_key: str
):
    """Background task to track batch prediction metrics"""
    try:
        success_rate = successful_predictions / batch_size if batch_size > 0 else 0
        
        logger.info(f"Batch metrics - ID: {batch_id}, Size: {batch_size}, "
                   f"Success: {success_rate:.2%}, Time: {processing_time:.3f}s")
        
        # Track in monitoring system
        # metrics_db.record_batch(
        #     batch_id=batch_id,
        #     batch_size=batch_size,
        #     success_rate=success_rate,
        #     processing_time=processing_time,
        #     api_key=api_key,
        #     timestamp=datetime.now()
        # )
        
    except Exception as e:
        logger.error(f"Error tracking batch metrics: {str(e)}")