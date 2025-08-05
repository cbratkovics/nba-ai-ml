"""
FastAPI application for NBA prediction service
"""
from fastapi import FastAPI, Depends, HTTPException, Request, Response, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from prometheus_fastapi_instrumentator import Instrumentator
from contextlib import asynccontextmanager
import uvicorn
import os
from datetime import datetime
from typing import Optional
import logging

from api.models import PredictionRequest, PredictionResponse, ExperimentResponse, InsightResponse
from api.endpoints import predictions, experiments, health, players, experiments_v2
from api.middleware.auth import verify_api_key
from api.middleware.rate_limiting import RateLimiter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Lifespan context manager for startup/shutdown
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Starting NBA AI/ML Prediction API...")
    logger.info(f"Environment: {os.getenv('ENVIRONMENT', 'development')}")
    logger.info(f"Python version: {os.sys.version}")
    
    # Check if models exist
    from pathlib import Path
    models_dir = Path("models")
    if models_dir.exists():
        model_files = list(models_dir.glob("*.pkl"))
        logger.info(f"Found {len(model_files)} model files")
        if model_files:
            for mf in model_files[:3]:  # Log first 3 model files
                logger.info(f"  - {mf.name}")
    else:
        logger.warning("Models directory not found. Creating dummy models may be needed.")
    
    # Log Redis connection status
    redis_url = os.getenv("REDIS_URL")
    if redis_url:
        logger.info("Redis URL configured")
    else:
        logger.info("Running without Redis cache")
    
    # Log database connection status
    db_url = os.getenv("DATABASE_URL")
    if db_url:
        logger.info("Database URL configured")
    else:
        logger.warning("No DATABASE_URL set - some features may be limited")
    
    yield
    # Shutdown
    logger.info("Shutting down NBA AI/ML Prediction API...")

# Create FastAPI app
app = FastAPI(
    title="NBA AI/ML Prediction API",
    description="Production-grade NBA player performance prediction system",
    version="2.1.0",
    lifespan=lifespan
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "https://*.vercel.app",
        os.getenv("FRONTEND_URL", "")
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Prometheus metrics
instrumentator = Instrumentator()
instrumentator.instrument(app).expose(app)

# Initialize rate limiter
rate_limiter = RateLimiter()

# Include routers
app.include_router(predictions.router, prefix="/v1", tags=["predictions"])
app.include_router(experiments.router, prefix="/v1", tags=["experiments"])
app.include_router(health.router, prefix="/health", tags=["health"])
app.include_router(players.router, prefix="/v1", tags=["players"])
app.include_router(experiments_v2.router, prefix="/v2", tags=["experiments-v2"])

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "name": "NBA AI/ML Prediction API",
        "version": "2.1.0",
        "status": "operational",
        "endpoints": {
            "predictions": "/v1/predict",
            "experiments": "/v1/experiments",
            "insights": "/v1/insights",
            "health": "/health/status",
            "metrics": "/metrics"
        },
        "documentation": "/docs",
        "timestamp": datetime.now().isoformat()
    }

@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    """Add X-Process-Time header to track request processing time"""
    import time
    start_time = time.time()
    
    response = await call_next(request)
    
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    
    # Log slow requests
    if process_time > 1.0:
        logger.warning(f"Slow request: {request.url.path} took {process_time:.2f}s")
    
    return response

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Custom HTTP exception handler"""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status_code": exc.status_code,
            "path": request.url.path,
            "timestamp": datetime.now().isoformat()
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """General exception handler for unexpected errors"""
    logger.error(f"Unexpected error: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "status_code": 500,
            "path": request.url.path,
            "timestamp": datetime.now().isoformat()
        }
    )


# WebSocket endpoint for real-time updates
@app.websocket("/ws/live")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for live game updates"""
    from api.streaming.game_monitor import GameMonitor, websocket_endpoint as ws_handler
    
    # Create game monitor instance
    monitor = GameMonitor()
    
    # Start monitoring if not already running
    await monitor.start_monitoring()
    
    # Handle WebSocket connection
    await ws_handler(websocket, monitor)


# Demo showcase endpoints
@app.get("/demo/tonight")
async def demo_tonight_predictions():
    """Demo endpoint: Predict tonight's games"""
    from api.demo.showcase import ShowcaseDemo
    demo = ShowcaseDemo()
    return await demo.predict_tonights_games()


@app.get("/demo/metrics")
async def demo_platform_metrics():
    """Demo endpoint: Show impressive platform metrics"""
    from api.demo.showcase import ShowcaseDemo
    demo = ShowcaseDemo()
    return await demo.generate_impressive_metrics()


if __name__ == "__main__":
    # Run with uvicorn for development
    uvicorn.run(
        "api.main:app",
        host=os.getenv("API_HOST", "0.0.0.0"),
        port=int(os.getenv("API_PORT", 8000)),
        reload=os.getenv("ENVIRONMENT", "development") == "development"
    )