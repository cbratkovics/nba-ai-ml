"""
Rate limiting middleware
"""
from fastapi import HTTPException, Request
import time
import redis
import os
import logging

logger = logging.getLogger(__name__)

class RateLimiter:
    def __init__(self, redis_url: str = None):
        self.redis_client = redis.from_url(
            redis_url or os.getenv("REDIS_URL", "redis://localhost:6379/0"),
            decode_responses=True
        )
    
    def is_rate_limited(self, key: str, limit: int, window: int) -> bool:
        """Check if request should be rate limited"""
        try:
            current = self.redis_client.get(key)
            if current is None:
                self.redis_client.setex(key, window, 1)
                return False
            
            if int(current) >= limit:
                return True
            
            self.redis_client.incr(key)
            return False
            
        except Exception as e:
            logger.error(f"Rate limiting error: {e}")
            return False  # Allow on error

rate_limiter = RateLimiter()

def check_rate_limit(request: Request) -> bool:
    """Check rate limit for request"""
    # Get client IP
    client_ip = request.client.host
    
    # Create rate limit key
    current_minute = int(time.time() // 60)
    rate_key = f"rate_limit:{client_ip}:{current_minute}"
    
    # Check rate limit (100 requests per minute per IP)
    if rate_limiter.is_rate_limited(rate_key, 100, 60):
        raise HTTPException(
            status_code=429,
            detail="Rate limit exceeded. Please try again later."
        )
    
    return True