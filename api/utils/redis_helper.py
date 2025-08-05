"""
Redis connection helper with fallback handling for Railway deployment
"""
import os
import redis
import logging
from typing import Optional, Any
import json

logger = logging.getLogger(__name__)


def get_redis_client() -> Optional[redis.Redis]:
    """
    Get Redis client with fallback handling
    Returns None if Redis is not available
    """
    redis_url = os.getenv("REDIS_URL")
    
    if not redis_url:
        logger.warning("REDIS_URL not configured, Redis caching disabled")
        return None
    
    try:
        # If using Railway internal URL, try with short timeout
        if "redis.railway.internal" in redis_url:
            logger.info("Attempting to connect to Railway internal Redis...")
            client = redis.from_url(
                redis_url,
                socket_connect_timeout=5,
                socket_timeout=5,
                retry_on_timeout=True,
                retry_on_error=[redis.ConnectionError, redis.TimeoutError],
                health_check_interval=30,
                decode_responses=True
            )
        else:
            # External URL
            logger.info("Connecting to external Redis...")
            client = redis.from_url(
                redis_url,
                socket_connect_timeout=10,
                decode_responses=True
            )
        
        # Test connection
        client.ping()
        logger.info("Redis connected successfully")
        return client
        
    except redis.ConnectionError as e:
        logger.error(f"Redis connection failed: {e}")
        logger.info("Continuing without Redis caching")
        return None
    except redis.TimeoutError as e:
        logger.error(f"Redis connection timeout: {e}")
        logger.info("Continuing without Redis caching")
        return None
    except Exception as e:
        logger.error(f"Unexpected Redis error: {e}")
        logger.info("Continuing without Redis caching")
        return None


class RedisCacheWrapper:
    """
    Wrapper for Redis operations with fallback to None
    Provides safe methods that don't fail if Redis is unavailable
    """
    
    def __init__(self):
        self.client = get_redis_client()
        self._warned = False
    
    def _log_warning_once(self):
        """Log warning about Redis being unavailable only once"""
        if not self._warned and not self.client:
            logger.warning("Redis not available - caching disabled")
            self._warned = True
    
    def get(self, key: str) -> Optional[Any]:
        """Safely get value from Redis"""
        if not self.client:
            self._log_warning_once()
            return None
        
        try:
            value = self.client.get(key)
            if value:
                try:
                    return json.loads(value)
                except json.JSONDecodeError:
                    return value
            return None
        except Exception as e:
            logger.debug(f"Redis get failed for key {key}: {e}")
            return None
    
    def set(self, key: str, value: Any, ex: Optional[int] = None) -> bool:
        """Safely set value in Redis"""
        if not self.client:
            self._log_warning_once()
            return False
        
        try:
            if not isinstance(value, str):
                value = json.dumps(value)
            
            if ex:
                self.client.setex(key, ex, value)
            else:
                self.client.set(key, value)
            return True
        except Exception as e:
            logger.debug(f"Redis set failed for key {key}: {e}")
            return False
    
    def delete(self, key: str) -> bool:
        """Safely delete key from Redis"""
        if not self.client:
            return False
        
        try:
            self.client.delete(key)
            return True
        except Exception as e:
            logger.debug(f"Redis delete failed for key {key}: {e}")
            return False
    
    def exists(self, key: str) -> bool:
        """Check if key exists in Redis"""
        if not self.client:
            return False
        
        try:
            return bool(self.client.exists(key))
        except Exception as e:
            logger.debug(f"Redis exists check failed for key {key}: {e}")
            return False
    
    def expire(self, key: str, seconds: int) -> bool:
        """Set expiration on a key"""
        if not self.client:
            return False
        
        try:
            return bool(self.client.expire(key, seconds))
        except Exception as e:
            logger.debug(f"Redis expire failed for key {key}: {e}")
            return False
    
    def ping(self) -> bool:
        """Check if Redis is available"""
        if not self.client:
            return False
        
        try:
            return self.client.ping()
        except Exception:
            return False
    
    @property
    def is_available(self) -> bool:
        """Check if Redis client is available"""
        return self.client is not None


# Global cache instance
redis_cache = RedisCacheWrapper()


# Convenience functions for direct import
def get_cache() -> RedisCacheWrapper:
    """Get the global Redis cache wrapper"""
    return redis_cache


def cache_get(key: str) -> Optional[Any]:
    """Get value from cache"""
    return redis_cache.get(key)


def cache_set(key: str, value: Any, ttl: Optional[int] = None) -> bool:
    """Set value in cache with optional TTL"""
    return redis_cache.set(key, value, ex=ttl)


def cache_delete(key: str) -> bool:
    """Delete key from cache"""
    return redis_cache.delete(key)


def is_redis_available() -> bool:
    """Check if Redis is available"""
    return redis_cache.is_available