"""
Health check endpoints
"""
from fastapi import APIRouter
from datetime import datetime
from sqlalchemy import text
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
    
    # Check Database
    try:
        from database.connection import engine
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
            services["database"] = "connected"
    except Exception as e:
        error_msg = str(e)
        if "dummy" in error_msg:
            services["database"] = "not configured (DATABASE_URL missing)"
        else:
            services["database"] = f"error: {error_msg[:100]}"
    
    # Check Redis
    redis_url = os.getenv("REDIS_URL")
    if redis_url:
        try:
            redis_client = redis.from_url(redis_url)
            redis_client.ping()
            services["redis"] = "connected"
        except Exception as e:
            services["redis"] = f"error: {str(e)[:100]}"
    else:
        services["redis"] = "not configured"
    
    # Check Models
    try:
        from pathlib import Path
        models_dir = Path("models")
        if models_dir.exists():
            model_files = list(models_dir.glob("*.pkl"))
            services["models"] = f"loaded ({len(model_files)} files)"
        else:
            services["models"] = "directory not found"
    except Exception as e:
        services["models"] = f"error: {str(e)[:100]}"
    
    # Determine overall status
    if any("error" in str(status) for status in services.values()):
        overall_status = "degraded"
    elif any("not configured" in str(status) for status in services.values()):
        overall_status = "partial"
    else:
        overall_status = "healthy"
    
    return {
        "status": overall_status,
        "version": "2.1.0",
        "timestamp": datetime.now().isoformat(),
        "services": services,
        "environment": {
            "has_database_url": bool(os.getenv("DATABASE_URL")),
            "has_redis_url": bool(os.getenv("REDIS_URL")),
            "has_supabase_url": bool(os.getenv("SUPABASE_URL")),
            "environment": os.getenv("ENVIRONMENT", "development"),
            "port": os.getenv("PORT", "8000")
        },
        "deployment": {
            "platform": "Railway" if os.getenv("RAILWAY_ENVIRONMENT") else "Local",
            "region": os.getenv("RAILWAY_REGION", "unknown")
        }
    }