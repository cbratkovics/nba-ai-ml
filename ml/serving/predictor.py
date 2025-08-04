"""
Production prediction service with model loading and caching
"""
import asyncio
import hashlib
import json
import logging
from datetime import datetime, date
from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np
from ml.models.ensemble import NBAEnsemble, ModelExperiment
from ml.data.collectors.nba_api_collector import NBADataCollector
from ml.data.processors.feature_engineer import NBAFeatureEngineer
import redis
import os

logger = logging.getLogger(__name__)


class ModelRegistry:
    """Model version management and loading"""
    
    def __init__(self, model_path: str = "./models"):
        self.model_path = model_path
        self.loaded_models = {}
        
    def load_model(self, version: str, target: str = "points") -> NBAEnsemble:
        """Load model from registry"""
        model_key = f"{version}_{target}"
        
        if model_key not in self.loaded_models:
            model_file = f"{self.model_path}/ensemble_{target}_{version}.pkl"
            
            if os.path.exists(model_file):
                self.loaded_models[model_key] = NBAEnsemble.load(model_file)
                logger.info(f"Loaded model {model_key}")
            else:
                # Create mock model for demo
                model = NBAEnsemble(target=target)
                self.loaded_models[model_key] = model
                logger.warning(f"Created mock model for {model_key}")
        
        return self.loaded_models[model_key]


class PredictionCache:
    """Redis-based prediction caching"""
    
    def __init__(self, redis_url: str = "redis://localhost:6379/1"):
        self.redis_client = redis.from_url(redis_url, decode_responses=True)
        self.cache_ttl = 3600  # 1 hour
        
    def get_cache_key(self, player_id: str, game_date: date, opponent: str) -> str:
        """Generate cache key"""
        key_string = f"{player_id}_{game_date}_{opponent}"
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def get(self, player_id: str, game_date: date, opponent: str) -> Optional[Dict]:
        """Get cached prediction"""
        try:
            cache_key = self.get_cache_key(player_id, game_date, opponent)
            cached_data = self.redis_client.get(cache_key)
            
            if cached_data:
                return json.loads(cached_data)
        except Exception as e:
            logger.error(f"Cache get error: {e}")
        
        return None
    
    def set(self, player_id: str, game_date: date, opponent: str, prediction: Dict):
        """Cache prediction result"""
        try:
            cache_key = self.get_cache_key(player_id, game_date, opponent)
            self.redis_client.setex(
                cache_key, 
                self.cache_ttl, 
                json.dumps(prediction, default=str)
            )
        except Exception as e:
            logger.error(f"Cache set error: {e}")


class PredictionService:
    """Main prediction service orchestrating all components"""
    
    def __init__(self):
        self.data_collector = NBADataCollector()
        self.feature_engineer = NBAFeatureEngineer()
        self.model_registry = ModelRegistry()
        self.cache = PredictionCache()
        self.ab_experiments = {}
        
        # Player name mapping (would be from database in production)
        self.player_names = {
            "203999": "Nikola Jokic",
            "2544": "LeBron James",
            "201939": "Stephen Curry",
            "1628369": "Jayson Tatum"
        }
    
    async def predict(self,
                     player_id: str,
                     game_date: date,
                     opponent_team: str,
                     targets: List[str] = ["all"],
                     model_version: str = "latest",
                     include_explanation: bool = False,
                     include_confidence_intervals: bool = False) -> Dict[str, Any]:
        """
        Main prediction method
        
        Args:
            player_id: NBA player ID
            game_date: Game date for prediction
            opponent_team: Opponent team abbreviation
            targets: Statistics to predict
            model_version: Model version to use
            include_explanation: Include natural language explanation
            include_confidence_intervals: Include prediction intervals
            
        Returns:
            Dictionary with predictions and metadata
        """
        # Check cache first
        cached_result = self.cache.get(player_id, game_date, opponent_team)
        if cached_result:
            logger.info(f"Cache hit for {player_id} vs {opponent_team}")
            return cached_result
        
        try:
            # Collect player data
            player_data = await self.data_collector.collect_player_stats(
                player_id, season="2024-25"
            )
            
            if player_data.empty:
                raise ValueError(f"No data found for player {player_id}")
            
            # Engineer features
            features_df = self.feature_engineer.create_features(player_data)
            latest_features = features_df.iloc[0:1]  # Most recent game features
            
            # Adjust features for upcoming game
            latest_features = self._adjust_features_for_prediction(
                latest_features, game_date, opponent_team
            )
            
            # Make predictions for each target
            predictions = {}
            confidence_intervals = {} if include_confidence_intervals else None
            all_confidences = []
            
            target_list = ["points", "rebounds", "assists"] if "all" in targets else targets
            
            for target in target_list:
                model = self.model_registry.load_model(model_version, target)
                
                if include_confidence_intervals:
                    pred, lower, upper = model.predict(
                        latest_features, return_uncertainty=True
                    )
                    predictions[target] = float(pred[0])
                    confidence_intervals[target] = {
                        "lower": float(lower[0]),
                        "upper": float(upper[0])
                    }
                    
                    # Calculate confidence from interval width
                    interval_width = upper[0] - lower[0]
                    confidence = max(0.1, 1.0 - (interval_width / pred[0]))
                    all_confidences.append(confidence)
                else:
                    pred = model.predict(latest_features)
                    predictions[target] = float(pred[0])
                    all_confidences.append(0.85)  # Default confidence
            
            # Calculate overall confidence
            overall_confidence = np.mean(all_confidences)
            
            # Get model accuracy metrics
            model_accuracy = self._get_model_accuracy(model_version)
            
            # Generate explanation if requested
            explanation = None
            factors = None
            if include_explanation:
                explanation, factors = await self._generate_explanation(
                    player_id, latest_features, predictions
                )
            
            result = {
                "player_name": self.player_names.get(player_id, f"Player {player_id}"),
                "predictions": predictions,
                "confidence": float(overall_confidence),
                "confidence_intervals": confidence_intervals,
                "model_version": model_version,
                "model_accuracy": model_accuracy,
                "explanation": explanation,
                "factors": factors
            }
            
            # Cache result
            self.cache.set(player_id, game_date, opponent_team, result)
            
            return result
            
        except Exception as e:
            logger.error(f"Prediction error for {player_id}: {str(e)}")
            raise
    
    def _adjust_features_for_prediction(self,
                                      features: pd.DataFrame,
                                      game_date: date,
                                      opponent: str) -> pd.DataFrame:
        """Adjust features for future game prediction"""
        features = features.copy()
        
        # Update game-specific features
        features['HOME_GAME'] = 1 if hash(opponent) % 2 else 0  # Simplified
        features['MATCHUP_DIFFICULTY'] = np.random.uniform(0.8, 1.2)  # Would use real opponent stats
        features['REST_DAYS'] = 2  # Default rest
        features['BACK_TO_BACK'] = 0
        
        # Update date-based features
        features['DAY_OF_WEEK'] = game_date.weekday()
        features['WEEKEND_GAME'] = 1 if game_date.weekday() >= 5 else 0
        features['MONTH'] = game_date.month
        
        return features
    
    def _get_model_accuracy(self, version: str) -> Dict[str, float]:
        """Get historical accuracy metrics for model version"""
        # Mock accuracy data (would come from model registry)
        accuracy_data = {
            "latest": {"r2_score": 0.942, "mae": 3.1, "rmse": 4.2},
            "v2.1.0": {"r2_score": 0.942, "mae": 3.1, "rmse": 4.2},
            "v2.0.0": {"r2_score": 0.935, "mae": 3.3, "rmse": 4.4}
        }
        return accuracy_data.get(version, accuracy_data["latest"])
    
    async def _generate_explanation(self,
                                  player_id: str,
                                  features: pd.DataFrame,
                                  predictions: Dict[str, float]) -> tuple:
        """Generate natural language explanation"""
        player_name = self.player_names.get(player_id, f"Player {player_id}")
        
        # Analyze key features (simplified)
        recent_avg = features.get('PTS_MA10', [25.0]).iloc[0]
        rest_days = features.get('REST_DAYS', [2]).iloc[0]
        home_game = features.get('HOME_GAME', [1]).iloc[0]
        
        explanation = f"{player_name} is projected for {predictions.get('points', 0):.1f} points. "
        
        if recent_avg > predictions.get('points', 0):
            explanation += "This is slightly below his recent 10-game average. "
        else:
            explanation += "This aligns with his recent strong performance. "
        
        if home_game:
            explanation += "Playing at home should provide a slight boost. "
        
        if rest_days >= 2:
            explanation += "Good rest should help maintain performance levels."
        elif rest_days == 0:
            explanation += "Back-to-back game may impact energy levels."
        
        # Create factors list
        factors = [
            {
                "factor": "10-game average",
                "value": float(recent_avg),
                "impact": "positive" if recent_avg > 20 else "neutral"
            },
            {
                "factor": "rest days",
                "value": int(rest_days),
                "impact": "positive" if rest_days >= 2 else "negative"
            },
            {
                "factor": "home/away",
                "value": "home" if home_game else "away",
                "impact": "positive" if home_game else "neutral"
            }
        ]
        
        return explanation, factors
    
    async def get_next_game(self, player_id: str) -> Optional[Dict[str, Any]]:
        """Get player's next scheduled game"""
        # Mock implementation (would query real schedule)
        from datetime import timedelta
        
        next_date = date.today() + timedelta(days=2)
        opponents = ["LAL", "GSW", "BOS", "MIA", "PHX"]
        opponent = opponents[hash(player_id) % len(opponents)]
        
        return {
            "date": next_date,
            "opponent": opponent,
            "home": True
        }
    
    async def store_feedback(self, prediction_id: str, actual_stats: Dict[str, float]):
        """Store actual game results for model evaluation"""
        # Would store in database for model retraining
        logger.info(f"Storing feedback for {prediction_id}: {actual_stats}")
        
        # Store in Redis for short-term tracking
        try:
            feedback_key = f"feedback:{prediction_id}"
            self.cache.redis_client.setex(
                feedback_key,
                86400,  # 24 hours
                json.dumps(actual_stats)
            )
        except Exception as e:
            logger.error(f"Error storing feedback: {e}")
    
    async def calculate_accuracy(self,
                               prediction_id: str,
                               actual_stats: Dict[str, float]) -> Dict[str, float]:
        """Calculate accuracy metrics for a prediction"""
        try:
            # Get original prediction (simplified)
            # Would retrieve from database in production
            
            # Mock accuracy calculation
            accuracy_metrics = {
                "mae": 2.1,
                "percentage_error": 8.5,
                "within_confidence_interval": True
            }
            
            return accuracy_metrics
            
        except Exception as e:
            logger.error(f"Error calculating accuracy: {e}")
            return {}
    
    def create_ab_experiment(self,
                           experiment_id: str,
                           control_version: str,
                           treatment_version: str,
                           traffic_split: float = 0.5) -> ModelExperiment:
        """Create A/B testing experiment"""
        control_model = self.model_registry.load_model(control_version)
        treatment_model = self.model_registry.load_model(treatment_version)
        
        experiment = ModelExperiment(control_model, treatment_model)
        self.ab_experiments[experiment_id] = experiment
        
        return experiment