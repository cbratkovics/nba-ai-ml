"""
Redis-cached SHAP explainability for NBA predictions
"""
import json
import hashlib
import logging
from typing import Optional, Dict, Any, List
import numpy as np
import redis
import os
from datetime import datetime
import asyncio
from sqlalchemy import text
from database.connection import get_db_session

logger = logging.getLogger(__name__)


class CachedExplainer:
    """SHAP explanations with Railway Redis caching"""
    
    def __init__(self):
        self.redis_url = os.getenv("REDIS_URL")
        self.redis_client = redis.from_url(self.redis_url, decode_responses=True) if self.redis_url else None
        self.explainers = {}
        self.cache_ttl = 3600  # 1 hour cache
        
        # Try to import SHAP
        try:
            import shap
            self.shap = shap
            self.shap_available = True
            logger.info("SHAP library available for explanations")
        except ImportError:
            self.shap_available = False
            logger.warning("SHAP not available - explanations will use feature importance fallback")
    
    def _get_cache_key(self, prediction_id: str) -> str:
        """Generate cache key for explanation"""
        return f"explanation:{prediction_id}"
    
    def _hash_features(self, features: dict) -> str:
        """Create hash of features for cache validation"""
        # Sort features for consistent hashing
        sorted_features = json.dumps(features, sort_keys=True)
        return hashlib.md5(sorted_features.encode()).hexdigest()
    
    async def get_explanation(self, 
                            prediction_id: str,
                            features: dict,
                            model_name: str,
                            model: Optional[Any] = None) -> Dict[str, Any]:
        """Get cached or compute explanation"""
        start_time = datetime.now()
        
        # Check cache first
        cache_key = self._get_cache_key(prediction_id)
        
        if self.redis_client:
            try:
                cached = self.redis_client.get(cache_key)
                if cached:
                    logger.info(f"Explanation cache hit for prediction {prediction_id}")
                    return json.loads(cached)
            except Exception as e:
                logger.error(f"Redis cache read error: {e}")
        
        # Compute explanation
        logger.info(f"Computing explanation for prediction {prediction_id}")
        
        if self.shap_available and model is not None:
            explanation = await self._compute_shap(features, model_name, model)
        else:
            explanation = await self._compute_feature_importance(features, model_name)
        
        # Add metadata
        computation_time_ms = int((datetime.now() - start_time).total_seconds() * 1000)
        explanation["computation_time_ms"] = computation_time_ms
        explanation["prediction_id"] = prediction_id
        explanation["model_name"] = model_name
        explanation["timestamp"] = datetime.now().isoformat()
        
        # Cache the explanation
        if self.redis_client:
            try:
                self.redis_client.setex(
                    cache_key,
                    self.cache_ttl,
                    json.dumps(explanation)
                )
                logger.info(f"Cached explanation for prediction {prediction_id}")
            except Exception as e:
                logger.error(f"Redis cache write error: {e}")
        
        # Store in database for persistence
        await self._store_explanation(prediction_id, explanation)
        
        return explanation
    
    async def _compute_shap(self, 
                          features: dict, 
                          model_name: str,
                          model: Any) -> Dict[str, Any]:
        """Compute SHAP values for explanation"""
        try:
            # Prepare features as array
            feature_names = sorted(features.keys())
            feature_values = np.array([[features[f] for f in feature_names]])
            
            # Get or create explainer
            explainer_key = f"explainer:{model_name}"
            
            if explainer_key not in self.explainers:
                # Create appropriate explainer based on model type
                if hasattr(model, 'predict_proba'):
                    # For tree-based models
                    self.explainers[explainer_key] = self.shap.TreeExplainer(model)
                else:
                    # For other models, use Kernel explainer with sampling
                    # Create background dataset (sample of training data)
                    background = await self._get_background_data(feature_names)
                    self.explainers[explainer_key] = self.shap.KernelExplainer(
                        model.predict, 
                        background,
                        link="identity"
                    )
            
            explainer = self.explainers[explainer_key]
            
            # Calculate SHAP values
            shap_values = explainer.shap_values(feature_values)
            
            # Handle multi-output models
            if isinstance(shap_values, list):
                # For multi-output, take the first output (points prediction)
                shap_values = shap_values[0]
            
            # Create feature importance dictionary
            feature_importances = {}
            shap_dict = {}
            
            for i, feature_name in enumerate(feature_names):
                importance = float(shap_values[0][i])
                feature_importances[feature_name] = abs(importance)
                shap_dict[feature_name] = importance
            
            # Get base value
            if hasattr(explainer, 'expected_value'):
                base_value = float(explainer.expected_value[0] if isinstance(explainer.expected_value, list) 
                                  else explainer.expected_value)
            else:
                base_value = 0.0
            
            explanation = {
                "method": "shap",
                "feature_importances": feature_importances,
                "shap_values": shap_dict,
                "base_value": base_value,
                "features": features
            }
            
            # Generate natural language explanation
            explanation["natural_language"] = self._generate_shap_narrative(
                feature_importances, 
                shap_dict, 
                features,
                base_value
            )
            
            return explanation
            
        except Exception as e:
            logger.error(f"Error computing SHAP values: {e}")
            # Fallback to feature importance
            return await self._compute_feature_importance(features, model_name)
    
    async def _compute_feature_importance(self, 
                                        features: dict, 
                                        model_name: str) -> Dict[str, Any]:
        """Compute feature importance as fallback when SHAP unavailable"""
        # Simple heuristic-based importance for key NBA features
        importance_weights = {
            # Recent performance has high impact
            "pts_last_10": 0.25,
            "pts_last_5": 0.20,
            "pts_last_3": 0.15,
            
            # Season averages
            "pts_season_avg": 0.10,
            "min_season_avg": 0.08,
            
            # Matchup factors
            "vs_opponent_avg": 0.08,
            "home_away": 0.05,
            
            # Other stats
            "reb_last_10": 0.04,
            "ast_last_10": 0.03,
            
            # Rest and usage
            "days_rest": 0.02
        }
        
        feature_importances = {}
        total_weight = 0
        
        for feature, value in features.items():
            if feature in importance_weights:
                # Scale importance by feature value
                normalized_value = min(value / 30.0, 1.0) if value > 0 else 0
                importance = importance_weights[feature] * (0.5 + 0.5 * normalized_value)
                feature_importances[feature] = importance
                total_weight += importance
        
        # Normalize importances
        if total_weight > 0:
            feature_importances = {
                k: v / total_weight 
                for k, v in feature_importances.items()
            }
        
        explanation = {
            "method": "heuristic",
            "feature_importances": feature_importances,
            "features": features
        }
        
        # Generate narrative
        explanation["natural_language"] = self._generate_heuristic_narrative(
            feature_importances, 
            features
        )
        
        return explanation
    
    async def _get_background_data(self, feature_names: List[str]) -> np.ndarray:
        """Get background data for SHAP Kernel explainer"""
        # Get sample of recent predictions for background
        async with get_db_session() as session:
            # This is a simplified query - in production you'd want to get actual feature values
            query = """
            SELECT COUNT(*) as count
            FROM predictions
            WHERE created_at > NOW() - INTERVAL '7 days'
            """
            result = await session.execute(text(query))
            count = result.scalar()
        
        # Create synthetic background data based on typical ranges
        n_samples = min(100, count if count else 100)
        background = np.zeros((n_samples, len(feature_names)))
        
        # Fill with typical values
        for i, feature in enumerate(feature_names):
            if 'pts' in feature:
                background[:, i] = np.random.normal(20, 8, n_samples)
            elif 'reb' in feature:
                background[:, i] = np.random.normal(5, 3, n_samples)
            elif 'ast' in feature:
                background[:, i] = np.random.normal(5, 3, n_samples)
            elif 'min' in feature:
                background[:, i] = np.random.normal(30, 8, n_samples)
            else:
                background[:, i] = np.random.normal(0, 1, n_samples)
        
        return background
    
    async def _store_explanation(self, prediction_id: str, explanation: Dict[str, Any]):
        """Store explanation in database"""
        try:
            async with get_db_session() as session:
                # Check if prediction exists
                check_query = """
                SELECT id, model_version FROM predictions WHERE id = :prediction_id
                """
                result = await session.execute(
                    text(check_query),
                    {"prediction_id": int(prediction_id)}
                )
                prediction = result.fetchone()
                
                if not prediction:
                    logger.warning(f"Prediction {prediction_id} not found for explanation storage")
                    return
                
                # Store explanation
                insert_query = """
                INSERT INTO prediction_explanations 
                (prediction_id, feature_importances, shap_values, natural_language, 
                 explanation_method, model_version, computation_time_ms)
                VALUES (:prediction_id, :feature_importances, :shap_values, :natural_language,
                        :method, :model_version, :computation_time_ms)
                ON CONFLICT (prediction_id) 
                DO UPDATE SET 
                    feature_importances = EXCLUDED.feature_importances,
                    shap_values = EXCLUDED.shap_values,
                    natural_language = EXCLUDED.natural_language,
                    computation_time_ms = EXCLUDED.computation_time_ms,
                    created_at = NOW()
                """
                
                await session.execute(
                    text(insert_query),
                    {
                        "prediction_id": int(prediction_id),
                        "feature_importances": json.dumps(explanation.get("feature_importances", {})),
                        "shap_values": json.dumps(explanation.get("shap_values", {})),
                        "natural_language": explanation.get("natural_language", ""),
                        "method": explanation.get("method", "unknown"),
                        "model_version": prediction.model_version,
                        "computation_time_ms": explanation.get("computation_time_ms", 0)
                    }
                )
                await session.commit()
                
                logger.info(f"Stored explanation for prediction {prediction_id}")
                
        except Exception as e:
            logger.error(f"Error storing explanation: {e}")
    
    def _generate_shap_narrative(self, 
                               feature_importances: dict,
                               shap_values: dict,
                               features: dict,
                               base_value: float) -> str:
        """Generate natural language explanation from SHAP values"""
        # Sort features by absolute SHAP value
        sorted_features = sorted(
            shap_values.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )[:5]  # Top 5 features
        
        narrative = f"Starting from a baseline prediction of {base_value:.1f} points, "
        narrative += "the following factors most influenced this prediction: "
        
        explanations = []
        for feature, shap_value in sorted_features:
            feature_value = features.get(feature, 0)
            impact = "increased" if shap_value > 0 else "decreased"
            
            # Humanize feature names and add context
            if feature == "pts_last_10":
                explanation = f"recent 10-game average of {feature_value:.1f} points {impact} prediction by {abs(shap_value):.1f}"
            elif feature == "pts_last_5":
                explanation = f"last 5 games average ({feature_value:.1f} pts) {impact} by {abs(shap_value):.1f}"
            elif feature == "vs_opponent_avg":
                explanation = f"career average vs this opponent ({feature_value:.1f} pts) {impact} by {abs(shap_value):.1f}"
            elif feature == "days_rest":
                rest_desc = "on back-to-back" if feature_value == 0 else f"with {feature_value} days rest"
                explanation = f"playing {rest_desc} {impact} by {abs(shap_value):.1f}"
            elif feature == "home_away":
                location = "at home" if feature_value == 1 else "on the road"
                explanation = f"playing {location} {impact} by {abs(shap_value):.1f}"
            else:
                explanation = f"{self._humanize_feature(feature)} ({feature_value:.1f}) {impact} by {abs(shap_value):.1f}"
            
            explanations.append(explanation)
        
        narrative += "; ".join(explanations) + "."
        
        return narrative
    
    def _generate_heuristic_narrative(self, 
                                    feature_importances: dict,
                                    features: dict) -> str:
        """Generate narrative from heuristic importance"""
        # Sort by importance
        sorted_features = sorted(
            feature_importances.items(),
            key=lambda x: x[1],
            reverse=True
        )[:5]
        
        narrative = "Key factors influencing this prediction: "
        
        factors = []
        for feature, importance in sorted_features:
            feature_value = features.get(feature, 0)
            importance_pct = importance * 100
            
            if feature == "pts_last_10":
                factors.append(f"recent form ({feature_value:.1f} pts over last 10 games, {importance_pct:.0f}% influence)")
            elif feature == "pts_season_avg":
                factors.append(f"season average of {feature_value:.1f} points ({importance_pct:.0f}% weight)")
            elif feature == "vs_opponent_avg":
                factors.append(f"historical performance vs opponent ({feature_value:.1f} pts avg, {importance_pct:.0f}% factor)")
            else:
                factors.append(f"{self._humanize_feature(feature)} ({importance_pct:.0f}% importance)")
        
        narrative += "; ".join(factors) + "."
        
        return narrative
    
    def _humanize_feature(self, feature_name: str) -> str:
        """Convert feature name to human-readable format"""
        replacements = {
            "pts": "points",
            "reb": "rebounds", 
            "ast": "assists",
            "min": "minutes",
            "avg": "average",
            "pct": "percentage",
            "_": " "
        }
        
        result = feature_name
        for old, new in replacements.items():
            result = result.replace(old, new)
        
        return result.title()
    
    async def get_batch_explanations(self, 
                                   prediction_ids: List[str]) -> Dict[str, Dict[str, Any]]:
        """Get explanations for multiple predictions efficiently"""
        explanations = {}
        
        # Check cache for all predictions
        if self.redis_client:
            cache_keys = [self._get_cache_key(pid) for pid in prediction_ids]
            try:
                cached_values = self.redis_client.mget(cache_keys)
                
                for pid, cached in zip(prediction_ids, cached_values):
                    if cached:
                        explanations[pid] = json.loads(cached)
            except Exception as e:
                logger.error(f"Redis batch read error: {e}")
        
        # Get missing explanations from database
        missing_ids = [pid for pid in prediction_ids if pid not in explanations]
        
        if missing_ids:
            async with get_db_session() as session:
                query = """
                SELECT 
                    prediction_id,
                    feature_importances,
                    shap_values,
                    natural_language,
                    explanation_method,
                    computation_time_ms
                FROM prediction_explanations
                WHERE prediction_id = ANY(:prediction_ids)
                """
                
                result = await session.execute(
                    text(query),
                    {"prediction_ids": [int(pid) for pid in missing_ids]}
                )
                
                for row in result.fetchall():
                    explanation = {
                        "method": row.explanation_method,
                        "feature_importances": json.loads(row.feature_importances),
                        "shap_values": json.loads(row.shap_values) if row.shap_values else {},
                        "natural_language": row.natural_language,
                        "computation_time_ms": row.computation_time_ms
                    }
                    explanations[str(row.prediction_id)] = explanation
        
        return explanations