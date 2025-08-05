"""
Production health check endpoints with comprehensive monitoring
"""
from fastapi import APIRouter, Depends
from datetime import datetime
from sqlalchemy import text
from typing import Dict, Any
import redis
import psutil
import aioredis
import os
import logging
import time
import asyncio
from pathlib import Path

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/status")
async def health_check():
    """Basic health check endpoint for load balancer/uptime monitoring"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": os.getenv("MODEL_VERSION", "2.1.0")
    }


@router.get("/ready")
async def readiness_check():
    """
    Kubernetes-style readiness probe
    Returns 200 only if the service is ready to accept traffic
    """
    try:
        # Check if critical services are available
        from database.connection import engine
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        
        # Check if models are loaded
        models_dir = Path("models")
        if not models_dir.exists() or not list(models_dir.glob("*.pkl")):
            return {"status": "not_ready", "reason": "models_not_loaded"}, 503
        
        return {"status": "ready", "timestamp": datetime.utcnow().isoformat()}
    
    except Exception as e:
        logger.error(f"Readiness check failed: {e}")
        return {"status": "not_ready", "reason": str(e)}, 503


@router.get("/live")
async def liveness_check():
    """
    Kubernetes-style liveness probe
    Returns 200 if the service is alive (even if not ready)
    """
    return {
        "status": "alive",
        "timestamp": datetime.utcnow().isoformat(),
        "pid": os.getpid(),
        "uptime_seconds": time.time() - psutil.Process().create_time()
    }


@router.get("/detailed")
async def detailed_health_check() -> Dict[str, Any]:
    """
    Comprehensive health check with all system components
    Perfect for monitoring dashboards and debugging
    """
    start_time = time.time()
    health_status = {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": os.getenv("MODEL_VERSION", "2.1.0"),
        "environment": os.getenv("ENVIRONMENT", "development"),
        "components": {},
        "metrics": {},
        "system": {}
    }
    
    # Database health check with connection pool stats
    try:
        # Try async pool first
        try:
            from api.db.connection_pool import db_pool
            db_health = await db_pool.health_check()
            health_status["components"]["database"] = db_health
        except:
            # Fallback to sync connection
            from database.connection import engine
            db_start = time.time()
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            db_time = (time.time() - db_start) * 1000
            
            health_status["components"]["database"] = {
                "status": "healthy",
                "response_time_ms": round(db_time, 2),
                "pool_size": getattr(engine.pool, 'size', lambda: 'N/A')(),
                "pool_checked_out": getattr(engine.pool, 'checked_out_connections', lambda: 'N/A')()
            }
    except Exception as e:
        health_status["components"]["database"] = {
            "status": "unhealthy",
            "error": str(e)[:200]
        }
        health_status["status"] = "degraded"
    
    # Redis health check
    redis_url = os.getenv("REDIS_URL")
    if redis_url:
        try:
            redis_start = time.time()
            # Try async redis first
            try:
                redis_client = await aioredis.from_url(redis_url)
                await redis_client.ping()
                info = await redis_client.info()
                await redis_client.close()
                
                health_status["components"]["redis"] = {
                    "status": "healthy",
                    "response_time_ms": round((time.time() - redis_start) * 1000, 2),
                    "version": info.get("redis_version", "unknown"),
                    "used_memory_mb": round(info.get("used_memory", 0) / 1024 / 1024, 2),
                    "connected_clients": info.get("connected_clients", 0)
                }
            except:
                # Fallback to sync redis
                redis_client = redis.from_url(redis_url)
                redis_client.ping()
                redis_time = (time.time() - redis_start) * 1000
                
                health_status["components"]["redis"] = {
                    "status": "healthy",
                    "response_time_ms": round(redis_time, 2)
                }
        except Exception as e:
            health_status["components"]["redis"] = {
                "status": "unhealthy",
                "error": str(e)[:200]
            }
    else:
        health_status["components"]["redis"] = {
            "status": "not_configured"
        }
    
    # Model health check
    try:
        from ml.serving.predictor_v2 import PredictionService
        models_dir = Path("models")
        model_files = list(models_dir.glob("*.pkl")) if models_dir.exists() else []
        
        # Try to get model registry stats
        try:
            service = PredictionService()
            model_stats = {
                "status": "healthy",
                "loaded_models": len(model_files),
                "model_files": [f.name for f in model_files[:5]],  # First 5 files
                "cache_enabled": bool(redis_url)
            }
        except:
            model_stats = {
                "status": "healthy",
                "loaded_models": len(model_files),
                "model_files": [f.name for f in model_files[:5]]
            }
        
        health_status["components"]["models"] = model_stats
        
    except Exception as e:
        health_status["components"]["models"] = {
            "status": "unhealthy",
            "error": str(e)[:200]
        }
        health_status["status"] = "unhealthy"
    
    # Feature store health (if implemented)
    try:
        from api.features.feature_store import FeatureStore
        health_status["components"]["feature_store"] = {
            "status": "healthy",
            "cache_backend": "redis" if redis_url else "memory"
        }
    except:
        health_status["components"]["feature_store"] = {
            "status": "not_implemented"
        }
    
    # System metrics
    try:
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        health_status["system"] = {
            "cpu_percent": cpu_percent,
            "memory": {
                "total_mb": round(memory.total / 1024 / 1024, 2),
                "used_mb": round(memory.used / 1024 / 1024, 2),
                "percent": memory.percent
            },
            "disk": {
                "total_gb": round(disk.total / 1024 / 1024 / 1024, 2),
                "used_gb": round(disk.used / 1024 / 1024 / 1024, 2),
                "percent": disk.percent
            },
            "load_average": os.getloadavg() if hasattr(os, 'getloadavg') else [0, 0, 0],
            "process": {
                "pid": os.getpid(),
                "threads": psutil.Process().num_threads(),
                "memory_mb": round(psutil.Process().memory_info().rss / 1024 / 1024, 2),
                "uptime_seconds": time.time() - psutil.Process().create_time()
            }
        }
    except Exception as e:
        logger.error(f"Failed to get system metrics: {e}")
        health_status["system"] = {"error": str(e)[:200]}
    
    # API metrics (if available)
    health_status["metrics"] = {
        "health_check_duration_ms": round((time.time() - start_time) * 1000, 2),
        "deployment": {
            "platform": "Railway" if os.getenv("RAILWAY_ENVIRONMENT") else "Local",
            "region": os.getenv("RAILWAY_REGION", "unknown"),
            "replica_id": os.getenv("RAILWAY_REPLICA_ID", "unknown")
        }
    }
    
    # Determine overall status
    component_statuses = [
        comp.get("status", "unknown") 
        for comp in health_status["components"].values()
    ]
    
    if any(status == "unhealthy" for status in component_statuses):
        health_status["status"] = "unhealthy"
    elif any(status in ["degraded", "not_configured"] for status in component_statuses):
        health_status["status"] = "degraded"
    
    return health_status


@router.get("/metrics")
async def prometheus_metrics():
    """
    Prometheus-compatible metrics endpoint
    Returns metrics in Prometheus text format
    """
    metrics = []
    
    # Get current metrics
    try:
        # System metrics
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()
        
        metrics.extend([
            f'# HELP nba_ml_api_cpu_usage_percent CPU usage percentage',
            f'# TYPE nba_ml_api_cpu_usage_percent gauge',
            f'nba_ml_api_cpu_usage_percent {cpu_percent}',
            '',
            f'# HELP nba_ml_api_memory_usage_bytes Memory usage in bytes',
            f'# TYPE nba_ml_api_memory_usage_bytes gauge',
            f'nba_ml_api_memory_usage_bytes {memory.used}',
            '',
            f'# HELP nba_ml_api_memory_percent Memory usage percentage',
            f'# TYPE nba_ml_api_memory_percent gauge',
            f'nba_ml_api_memory_percent {memory.percent}',
            ''
        ])
        
        # Database metrics (if available)
        try:
            from api.db.connection_pool import db_pool
            if db_pool._initialized:
                pool_impl = db_pool.engine.pool
                metrics.extend([
                    f'# HELP nba_ml_api_db_pool_size Database connection pool size',
                    f'# TYPE nba_ml_api_db_pool_size gauge',
                    f'nba_ml_api_db_pool_size {pool_impl.size()}',
                    '',
                    f'# HELP nba_ml_api_db_pool_checked_out Database connections checked out',
                    f'# TYPE nba_ml_api_db_pool_checked_out gauge',
                    f'nba_ml_api_db_pool_checked_out {pool_impl.checked_out_connections()}',
                    ''
                ])
        except:
            pass
        
        # Model metrics
        models_dir = Path("models")
        if models_dir.exists():
            model_count = len(list(models_dir.glob("*.pkl")))
            metrics.extend([
                f'# HELP nba_ml_api_loaded_models Number of loaded ML models',
                f'# TYPE nba_ml_api_loaded_models gauge',
                f'nba_ml_api_loaded_models {model_count}',
                ''
            ])
        
    except Exception as e:
        logger.error(f"Failed to generate metrics: {e}")
    
    return "\n".join(metrics)


@router.get("/debug")
async def debug_info():
    """
    Debug endpoint with detailed system information
    Should be protected in production!
    """
    # Check if debug mode is enabled
    if os.getenv("ENVIRONMENT", "development").lower() == "production":
        return {"error": "Debug endpoint disabled in production"}, 403
    
    return {
        "timestamp": datetime.utcnow().isoformat(),
        "environment_variables": {
            k: v[:50] + "..." if len(v) > 50 else v
            for k, v in os.environ.items()
            if not any(secret in k.lower() for secret in ["password", "secret", "key", "token"])
        },
        "python_version": os.sys.version,
        "working_directory": os.getcwd(),
        "loaded_modules": sorted([
            name for name in os.sys.modules.keys()
            if name.startswith(("api", "ml", "database"))
        ])[:20],  # First 20 modules
        "file_structure": {
            "models": [f.name for f in Path("models").glob("*.pkl")] if Path("models").exists() else [],
            "api_endpoints": [f.name for f in Path("api/endpoints").glob("*.py")] if Path("api/endpoints").exists() else []
        }
    }