"""
Health check endpoints
"""
from fastapi import APIRouter
from datetime import datetime
import redis
import os
import logging

logger = logging.getLogger(__name__)

router = APIRouter()

@router.get("/status")
async def health_check():
    """Basic health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "2.1.0"
    }

@router.get("/detailed")
async def detailed_health_check():
    """Detailed health check with service status"""
    services = {}
    
    # Check Redis
    try:
        redis_client = redis.from_url(os.getenv("REDIS_URL", "redis://localhost:6379/0"))
        redis_client.ping()
        services["redis"] = "connected"
    except Exception as e:
        services["redis"] = f"error: {str(e)}"
    
    return {
        "status": "healthy" if all("error" not in status for status in services.values()) else "degraded",
        "version": "2.1.0",
        "timestamp": datetime.now().isoformat(),
        "services": services,
        "system": {
            "status": "operational"
        }
    }