"""
Production feature store with caching and versioning
"""
import json
import logging
import hashlib
import os
from datetime import datetime, timedelta, date
from typing import Dict, Any, Optional, List, Tuple
import numpy as np
import pandas as pd
from redis import Redis
import redis.asyncio as aioredis
from dataclasses import dataclass, asdict
import asyncio
from sqlalchemy import select, text, and_
from database.connection import get_db_session
from api.models.game_data import PlayerFeatures, GameLog
from api.features.player_features import FeatureEngineer

logger = logging.getLogger(__name__)


@dataclass
class FeatureSet:
    """Feature set with metadata"""
    feature_id: str
    player_id: str
    features: Dict[str, float]
    version: str
    calculated_at: str
    valid_until: str
    metadata: Dict[str, Any]


class FeatureStore:
    """
    Production feature store with Redis caching and versioning
    """
    
    def __init__(self, redis_url: Optional[str] = None):
        """
        Initialize feature store
        
        Args:
            redis_url: Redis connection URL
        """
        self.redis_url = redis_url or os.getenv("REDIS_URL")
        self.redis_client = None
        self.async_redis_client = None
        self.feature_engineer = FeatureEngineer()
        
        # Cache configuration
        self.cache_ttl = {
            "hot": 300,      # 5 minutes for frequently accessed
            "warm": 3600,    # 1 hour for recent features
            "cold": 86400    # 24 hours for historical
        }
        
        # Feature versions
        self.current_version = "v2.0"
        
        # Initialize Redis clients
        self._init_redis()
    
    def _init_redis(self):
        """Initialize Redis connections"""
        if self.redis_url:
            try:
                self.redis_client = Redis.from_url(self.redis_url, decode_responses=True)
                self.async_redis_client = aioredis.from_url(self.redis_url, decode_responses=True)
                self.redis_client.ping()
                logger.info("Feature store Redis initialized")
            except Exception as e:
                logger.error(f"Redis initialization failed: {e}")
                self.redis_client = None
                self.async_redis_client = None
    
    async def get_features(self, 
                          player_id: str, 
                          game_date: str,
                          opponent_team: str,
                          version: Optional[str] = None) -> Dict[str, float]:
        """
        Get features with caching and fallback
        
        Args:
            player_id: NBA player ID
            game_date: Game date
            opponent_team: Opponent team
            version: Feature version (optional)
            
        Returns:
            Feature dictionary
        """
        version = version or self.current_version
        
        # Try cache first
        cached_features = await self._get_from_cache(player_id, game_date, opponent_team, version)
        if cached_features:
            return cached_features
        
        # Calculate features
        features = await self.feature_engineer.calculate_player_features(
            player_id, game_date, opponent_team
        )
        
        # Store in cache
        await self._store_in_cache(player_id, game_date, opponent_team, features, version)
        
        # Store in database for persistence
        await self._store_in_database(player_id, game_date, features)
        
        return features
    
    async def get_features_batch(self, 
                               requests: List[Tuple[str, str, str]],
                               version: Optional[str] = None) -> Dict[str, Dict[str, float]]:
        """
        Get features for multiple players efficiently
        
        Args:
            requests: List of (player_id, game_date, opponent_team) tuples
            version: Feature version
            
        Returns:
            Dictionary mapping player_id to features
        """
        version = version or self.current_version
        results = {}
        
        # Check cache for all requests
        cache_keys = []
        cache_misses = []
        
        for player_id, game_date, opponent_team in requests:
            cache_key = self._get_cache_key(player_id, game_date, opponent_team, version)
            cache_keys.append((cache_key, player_id, game_date, opponent_team))
        
        # Batch get from Redis
        if self.async_redis_client:
            try:
                # Use pipeline for efficiency
                pipe = self.async_redis_client.pipeline()
                for cache_key, _, _, _ in cache_keys:
                    pipe.get(cache_key)
                
                cache_results = await pipe.execute()
                
                for i, cached in enumerate(cache_results):
                    cache_key, player_id, game_date, opponent_team = cache_keys[i]
                    if cached:
                        results[player_id] = json.loads(cached)
                    else:
                        cache_misses.append((player_id, game_date, opponent_team))
                        
            except Exception as e:
                logger.error(f"Batch cache get failed: {e}")
                cache_misses = requests
        else:
            cache_misses = requests
        
        # Calculate missing features in parallel
        if cache_misses:
            tasks = []
            for player_id, game_date, opponent_team in cache_misses:
                task = self.get_features(player_id, game_date, opponent_team, version)
                tasks.append(task)
            
            missing_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for i, result in enumerate(missing_results):
                player_id = cache_misses[i][0]
                if not isinstance(result, Exception):
                    results[player_id] = result
                else:
                    logger.error(f"Error getting features for {player_id}: {result}")
                    results[player_id] = self.feature_engineer._get_default_features()
        
        return results
    
    async def get_point_in_time_features(self,
                                       player_id: str,
                                       timestamp: datetime,
                                       opponent_team: str) -> Dict[str, float]:
        """
        Get features as they were at a specific point in time
        
        Args:
            player_id: NBA player ID
            timestamp: Point in time
            opponent_team: Opponent team
            
        Returns:
            Features valid at that timestamp
        """
        # Convert timestamp to date
        game_date = timestamp.date()
        
        # Get historical features from database
        async with get_db_session() as session:
            result = await session.execute(
                select(PlayerFeatures)
                .where(
                    and_(
                        PlayerFeatures.player_id == player_id,
                        PlayerFeatures.calculation_date <= game_date
                    )
                )
                .order_by(PlayerFeatures.calculation_date.desc())
                .limit(1)
            )
            
            feature_record = result.scalar_one_or_none()
            
            if feature_record:
                # Convert to dictionary
                features = {
                    'pts_last_5': feature_record.pts_last_5,
                    'pts_last_10': feature_record.pts_last_10,
                    'pts_last_20': feature_record.pts_last_20,
                    'pts_season': feature_record.pts_season,
                    'reb_last_5': feature_record.reb_last_5,
                    'reb_last_10': feature_record.reb_last_10,
                    'reb_last_20': feature_record.reb_last_20,
                    'reb_season': feature_record.reb_season,
                    'ast_last_5': feature_record.ast_last_5,
                    'ast_last_10': feature_record.ast_last_10,
                    'ast_last_20': feature_record.ast_last_20,
                    'ast_season': feature_record.ast_season,
                    'fg_pct_last_10': feature_record.fg_pct_last_10,
                    'ft_pct_last_10': feature_record.ft_pct_last_10,
                    'fg3_pct_last_10': feature_record.fg3_pct_last_10,
                    'minutes_last_10': feature_record.minutes_last_10,
                    'games_played_season': feature_record.games_played_season,
                    'days_since_last_game': feature_record.days_since_last_game,
                    'back_to_backs_last_10': feature_record.back_to_backs_last_10,
                    'home_ppg': feature_record.home_ppg,
                    'away_ppg': feature_record.away_ppg,
                    'vs_opponent_avg_pts': feature_record.vs_opponent_avg_pts,
                    'vs_opponent_avg_reb': feature_record.vs_opponent_avg_reb,
                    'vs_opponent_avg_ast': feature_record.vs_opponent_avg_ast
                }
                return features
            else:
                # No historical features, calculate fresh
                return await self.get_features(player_id, str(game_date), opponent_team)
    
    async def invalidate_features(self, player_id: str):
        """
        Invalidate cached features for a player
        
        Args:
            player_id: NBA player ID
        """
        if not self.async_redis_client:
            return
        
        try:
            # Find all cache keys for this player
            pattern = f"features:{player_id}:*"
            keys = []
            
            async for key in self.async_redis_client.scan_iter(match=pattern):
                keys.append(key)
            
            if keys:
                await self.async_redis_client.delete(*keys)
                logger.info(f"Invalidated {len(keys)} cache entries for player {player_id}")
                
        except Exception as e:
            logger.error(f"Error invalidating cache for {player_id}: {e}")
    
    async def monitor_feature_drift(self, player_id: str) -> Dict[str, Any]:
        """
        Monitor feature drift for a player
        
        Returns:
            Drift metrics
        """
        # Get recent features
        recent_features = []
        today = datetime.now().date()
        
        for days_back in range(30):
            date = today - timedelta(days=days_back)
            features = await self._get_from_database(player_id, date)
            if features:
                recent_features.append(features)
        
        if len(recent_features) < 10:
            return {"status": "insufficient_data"}
        
        # Convert to DataFrame for analysis
        df = pd.DataFrame(recent_features)
        
        # Calculate drift metrics
        drift_metrics = {}
        
        for col in df.columns:
            if col.endswith(('_avg', '_pct', '_ppg')):
                # Calculate statistics
                mean = df[col].mean()
                std = df[col].std()
                recent_mean = df[col].head(7).mean()
                
                # Calculate drift
                if std > 0:
                    z_score = abs(recent_mean - mean) / std
                    drift_metrics[col] = {
                        "historical_mean": mean,
                        "recent_mean": recent_mean,
                        "std": std,
                        "z_score": z_score,
                        "drift_detected": z_score > 2.0
                    }
        
        return {
            "status": "analyzed",
            "player_id": player_id,
            "features_analyzed": len(drift_metrics),
            "drift_metrics": drift_metrics,
            "drift_detected": any(m["drift_detected"] for m in drift_metrics.values())
        }
    
    def _get_cache_key(self, player_id: str, game_date: str, opponent_team: str, version: str) -> str:
        """Generate cache key"""
        return f"features:{player_id}:{game_date}:{opponent_team}:{version}"
    
    async def _get_from_cache(self, 
                            player_id: str, 
                            game_date: str, 
                            opponent_team: str,
                            version: str) -> Optional[Dict[str, float]]:
        """Get features from cache"""
        if not self.async_redis_client:
            return None
        
        try:
            cache_key = self._get_cache_key(player_id, game_date, opponent_team, version)
            cached = await self.async_redis_client.get(cache_key)
            
            if cached:
                # Update access time for hot/warm/cold tiering
                await self._update_access_time(cache_key)
                return json.loads(cached)
                
        except Exception as e:
            logger.error(f"Cache get error: {e}")
        
        return None
    
    async def _store_in_cache(self,
                            player_id: str,
                            game_date: str,
                            opponent_team: str,
                            features: Dict[str, float],
                            version: str):
        """Store features in cache"""
        if not self.async_redis_client:
            return
        
        try:
            cache_key = self._get_cache_key(player_id, game_date, opponent_team, version)
            
            # Determine TTL based on recency
            today = datetime.now().date()
            game_date_obj = datetime.strptime(game_date, '%Y-%m-%d').date() if isinstance(game_date, str) else game_date
            days_ago = (today - game_date_obj).days
            
            if days_ago <= 1:
                ttl = self.cache_ttl["hot"]
            elif days_ago <= 7:
                ttl = self.cache_ttl["warm"]
            else:
                ttl = self.cache_ttl["cold"]
            
            # Store with expiration
            await self.async_redis_client.setex(
                cache_key,
                ttl,
                json.dumps(features)
            )
            
            # Store access time
            await self._update_access_time(cache_key)
            
        except Exception as e:
            logger.error(f"Cache store error: {e}")
    
    async def _update_access_time(self, cache_key: str):
        """Update last access time for cache management"""
        if not self.async_redis_client:
            return
        
        try:
            access_key = f"{cache_key}:accessed"
            await self.async_redis_client.setex(
                access_key,
                86400,  # 24 hour TTL for access tracking
                datetime.now().isoformat()
            )
        except Exception as e:
            logger.debug(f"Error updating access time: {e}")
    
    async def _store_in_database(self, player_id: str, game_date: str, features: Dict[str, float]):
        """Store features in database for persistence"""
        try:
            date_obj = datetime.strptime(game_date, '%Y-%m-%d').date() if isinstance(game_date, str) else game_date
            await self.feature_engineer.save_features(player_id, date_obj, features)
        except Exception as e:
            logger.error(f"Error storing features in database: {e}")
    
    async def _get_from_database(self, player_id: str, game_date: date) -> Optional[Dict[str, float]]:
        """Get features from database"""
        try:
            async with get_db_session() as session:
                result = await session.execute(
                    select(PlayerFeatures)
                    .where(
                        and_(
                            PlayerFeatures.player_id == player_id,
                            PlayerFeatures.calculation_date == game_date
                        )
                    )
                )
                
                feature_record = result.scalar_one_or_none()
                
                if feature_record:
                    return {
                        col: getattr(feature_record, col)
                        for col in self.feature_engineer.feature_names
                    }
                    
        except Exception as e:
            logger.error(f"Error getting features from database: {e}")
        
        return None
    
    async def get_feature_statistics(self) -> Dict[str, Any]:
        """Get feature store statistics"""
        stats = {
            "cache_enabled": self.async_redis_client is not None,
            "current_version": self.current_version,
            "cache_stats": {},
            "feature_stats": {}
        }
        
        if self.async_redis_client:
            try:
                # Get cache statistics
                info = await self.async_redis_client.info()
                stats["cache_stats"] = {
                    "used_memory": info.get("used_memory_human", "N/A"),
                    "total_keys": await self.async_redis_client.dbsize(),
                    "hit_rate": info.get("keyspace_hits", 0) / max(info.get("keyspace_hits", 0) + info.get("keyspace_misses", 1), 1)
                }
                
                # Count feature keys
                feature_count = 0
                async for _ in self.async_redis_client.scan_iter(match="features:*"):
                    feature_count += 1
                
                stats["cache_stats"]["feature_keys"] = feature_count
                
            except Exception as e:
                logger.error(f"Error getting cache stats: {e}")
        
        return stats