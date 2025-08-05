"""
Railway Monitoring Service for NBA ML Platform
"""
from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import CollectorRegistry, Counter, Histogram, Gauge, generate_latest
from fastapi.responses import Response
import asyncpg
import asyncio
import os
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import redis
from contextlib import asynccontextmanager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Prometheus metrics
registry = CollectorRegistry()
prediction_counter = Counter('nba_predictions_total', 'Total predictions made', ['model_version', 'status'], registry=registry)
prediction_latency = Histogram('nba_prediction_latency_seconds', 'Prediction latency', ['model_version'], registry=registry)
model_accuracy = Gauge('nba_model_accuracy', 'Model accuracy metrics', ['model_version', 'metric'], registry=registry)
active_users = Gauge('nba_active_users', 'Active users in last hour', registry=registry)
cache_hits = Counter('nba_cache_hits_total', 'Cache hit rate', ['cache_type'], registry=registry)
anomaly_counter = Counter('nba_anomalies_detected_total', 'Anomalies detected', ['anomaly_type'], registry=registry)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle"""
    # Startup
    monitor = app.state.monitor = RailwayMonitor()
    await monitor.initialize()
    
    # Start background tasks
    asyncio.create_task(monitor.collect_metrics_loop())
    
    yield
    
    # Shutdown
    await monitor.cleanup()


app = FastAPI(
    title="NBA ML Monitoring Dashboard",
    version="1.0.0",
    lifespan=lifespan
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class RailwayMonitor:
    """Railway-specific monitoring service"""
    
    def __init__(self):
        self.supabase_url = os.getenv("DATABASE_URL")
        self.redis_url = os.getenv("REDIS_URL")
        self.registry = registry
        self.db_pool: Optional[asyncpg.Pool] = None
        self.redis_client: Optional[redis.Redis] = None
        
    async def initialize(self):
        """Initialize connections"""
        try:
            # Create Postgres connection pool
            self.db_pool = await asyncpg.create_pool(
                self.supabase_url,
                min_size=2,
                max_size=10,
                command_timeout=60
            )
            logger.info("Connected to Supabase PostgreSQL")
            
            # Connect to Redis
            if self.redis_url:
                self.redis_client = redis.from_url(self.redis_url, decode_responses=True)
                logger.info("Connected to Railway Redis")
                
        except Exception as e:
            logger.error(f"Failed to initialize connections: {e}")
    
    async def cleanup(self):
        """Cleanup connections"""
        if self.db_pool:
            await self.db_pool.close()
        if self.redis_client:
            self.redis_client.close()
    
    async def collect_supabase_metrics(self) -> Dict[str, Any]:
        """Collect metrics directly from Supabase"""
        metrics = {}
        
        try:
            async with self.db_pool.acquire() as conn:
                # Prediction count by model version
                query = """
                SELECT model_version, COUNT(*) as count
                FROM predictions
                WHERE created_at > NOW() - INTERVAL '1 hour'
                GROUP BY model_version
                """
                rows = await conn.fetch(query)
                for row in rows:
                    prediction_counter.labels(
                        model_version=row['model_version'],
                        status='success'
                    )._value._value = float(row['count'])
                
                # Model accuracy metrics
                accuracy_query = """
                SELECT 
                    model_version,
                    AVG(ABS(points_predicted - points_actual)) as mae_points,
                    AVG(ABS(rebounds_predicted - rebounds_actual)) as mae_rebounds,
                    AVG(ABS(assists_predicted - assists_actual)) as mae_assists,
                    COUNT(*) as sample_size
                FROM predictions
                WHERE created_at > NOW() - INTERVAL '24 hours'
                    AND points_actual IS NOT NULL
                GROUP BY model_version
                """
                accuracy_rows = await conn.fetch(accuracy_query)
                
                metrics['accuracy'] = []
                for row in accuracy_rows:
                    model_accuracy.labels(
                        model_version=row['model_version'],
                        metric='mae_points'
                    ).set(row['mae_points'])
                    model_accuracy.labels(
                        model_version=row['model_version'],
                        metric='mae_rebounds'
                    ).set(row['mae_rebounds'])
                    model_accuracy.labels(
                        model_version=row['model_version'],
                        metric='mae_assists'
                    ).set(row['mae_assists'])
                    
                    metrics['accuracy'].append({
                        'model_version': row['model_version'],
                        'mae_points': float(row['mae_points']),
                        'mae_rebounds': float(row['mae_rebounds']),
                        'mae_assists': float(row['mae_assists']),
                        'sample_size': row['sample_size']
                    })
                
                # Active users
                user_query = """
                SELECT COUNT(DISTINCT user_id) as active_users
                FROM predictions
                WHERE created_at > NOW() - INTERVAL '1 hour'
                """
                user_count = await conn.fetchval(user_query)
                active_users.set(user_count or 0)
                metrics['active_users'] = user_count
                
                # Top predicted players
                player_query = """
                SELECT player_name, COUNT(*) as prediction_count
                FROM predictions
                WHERE created_at > NOW() - INTERVAL '24 hours'
                GROUP BY player_name
                ORDER BY prediction_count DESC
                LIMIT 10
                """
                top_players = await conn.fetch(player_query)
                metrics['top_players'] = [dict(row) for row in top_players]
                
        except Exception as e:
            logger.error(f"Error collecting Supabase metrics: {e}")
            
        return metrics
    
    async def collect_railway_metrics(self) -> Dict[str, Any]:
        """Use Railway's metrics and system stats"""
        metrics = {
            'timestamp': datetime.now().isoformat(),
            'service': 'nba-ml-monitoring'
        }
        
        try:
            # Memory usage
            import psutil
            process = psutil.Process()
            memory_info = process.memory_info()
            metrics['memory_mb'] = memory_info.rss / 1024 / 1024
            metrics['memory_percent'] = process.memory_percent()
            
            # CPU usage
            metrics['cpu_percent'] = process.cpu_percent(interval=1)
            
            # Redis stats
            if self.redis_client:
                info = self.redis_client.info()
                metrics['redis'] = {
                    'connected_clients': info.get('connected_clients', 0),
                    'used_memory_mb': info.get('used_memory', 0) / 1024 / 1024,
                    'total_commands': info.get('total_commands_processed', 0)
                }
                
                # Cache hit rate
                cache_stats = self.redis_client.info('stats')
                hits = cache_stats.get('keyspace_hits', 0)
                misses = cache_stats.get('keyspace_misses', 0)
                if hits + misses > 0:
                    metrics['cache_hit_rate'] = hits / (hits + misses)
                    cache_hits.labels(cache_type='redis')._value._value = float(hits)
            
        except Exception as e:
            logger.error(f"Error collecting Railway metrics: {e}")
            
        return metrics
    
    async def detect_anomalies(self) -> List[Dict[str, Any]]:
        """Detect anomalies in predictions"""
        anomalies = []
        
        try:
            async with self.db_pool.acquire() as conn:
                # Statistical anomaly detection
                anomaly_query = """
                WITH stats AS (
                    SELECT 
                        AVG(points_predicted) as mean_points,
                        STDDEV(points_predicted) as std_points,
                        AVG(confidence) as mean_confidence,
                        STDDEV(confidence) as std_confidence
                    FROM predictions
                    WHERE created_at > NOW() - INTERVAL '7 days'
                )
                SELECT 
                    p.id,
                    p.player_name,
                    p.points_predicted,
                    p.confidence,
                    p.created_at,
                    ABS(p.points_predicted - s.mean_points) / NULLIF(s.std_points, 0) as points_z_score,
                    ABS(p.confidence - s.mean_confidence) / NULLIF(s.std_confidence, 0) as confidence_z_score
                FROM predictions p, stats s
                WHERE p.created_at > NOW() - INTERVAL '1 hour'
                  AND (
                    ABS(p.points_predicted - s.mean_points) > 3 * s.std_points
                    OR ABS(p.confidence - s.mean_confidence) > 3 * s.std_confidence
                    OR p.points_predicted < 0
                    OR p.points_predicted > 60
                    OR p.confidence < 0.3
                    OR p.confidence > 1.0
                  )
                ORDER BY p.created_at DESC
                LIMIT 50
                """
                
                anomaly_rows = await conn.fetch(anomaly_query)
                
                for row in anomaly_rows:
                    anomaly = {
                        'id': row['id'],
                        'player_name': row['player_name'],
                        'points_predicted': float(row['points_predicted']),
                        'confidence': float(row['confidence']),
                        'created_at': row['created_at'].isoformat(),
                        'anomaly_type': []
                    }
                    
                    # Classify anomaly type
                    if row['points_z_score'] and row['points_z_score'] > 3:
                        anomaly['anomaly_type'].append('statistical_outlier')
                        anomaly_counter.labels(anomaly_type='statistical_outlier').inc()
                    
                    if row['points_predicted'] < 0 or row['points_predicted'] > 60:
                        anomaly['anomaly_type'].append('impossible_value')
                        anomaly_counter.labels(anomaly_type='impossible_value').inc()
                    
                    if row['confidence'] < 0.3:
                        anomaly['anomaly_type'].append('low_confidence')
                        anomaly_counter.labels(anomaly_type='low_confidence').inc()
                    
                    anomalies.append(anomaly)
                
                # Check for drift
                drift_query = """
                WITH hourly_stats AS (
                    SELECT 
                        date_trunc('hour', created_at) as hour,
                        AVG(points_predicted) as mean_points,
                        AVG(confidence) as mean_confidence
                    FROM predictions
                    WHERE created_at > NOW() - INTERVAL '24 hours'
                    GROUP BY date_trunc('hour', created_at)
                    ORDER BY hour
                )
                SELECT 
                    hour,
                    mean_points,
                    mean_confidence,
                    mean_points - LAG(mean_points) OVER (ORDER BY hour) as points_change,
                    mean_confidence - LAG(mean_confidence) OVER (ORDER BY hour) as confidence_change
                FROM hourly_stats
                """
                
                drift_rows = await conn.fetch(drift_query)
                
                for i, row in enumerate(drift_rows):
                    if row['points_change'] and abs(row['points_change']) > 5:
                        anomalies.append({
                            'type': 'distribution_drift',
                            'hour': row['hour'].isoformat(),
                            'metric': 'points',
                            'change': float(row['points_change'])
                        })
                        anomaly_counter.labels(anomaly_type='distribution_drift').inc()
                
        except Exception as e:
            logger.error(f"Error detecting anomalies: {e}")
            
        return anomalies
    
    async def collect_metrics_loop(self):
        """Background task to collect metrics"""
        while True:
            try:
                # Collect metrics
                supabase_metrics = await self.collect_supabase_metrics()
                railway_metrics = await self.collect_railway_metrics()
                anomalies = await self.detect_anomalies()
                
                # Store latest metrics in Redis
                if self.redis_client:
                    self.redis_client.setex(
                        'monitoring:latest_metrics',
                        300,  # 5 minutes
                        json.dumps({
                            'supabase': supabase_metrics,
                            'railway': railway_metrics,
                            'anomalies': anomalies,
                            'timestamp': datetime.now().isoformat()
                        })
                    )
                
                # Log metrics for Railway dashboard
                logger.info(f"METRICS: {json.dumps(railway_metrics)}")
                
                # Alert on critical anomalies
                critical_anomalies = [a for a in anomalies if 'impossible_value' in a.get('anomaly_type', [])]
                if critical_anomalies:
                    logger.error(f"CRITICAL ANOMALIES: {len(critical_anomalies)} impossible values detected")
                
            except Exception as e:
                logger.error(f"Error in metrics collection loop: {e}")
            
            # Run every 60 seconds
            await asyncio.sleep(60)


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "NBA ML Monitoring Dashboard",
        "status": "operational",
        "endpoints": {
            "metrics": "/metrics",
            "dashboard": "/dashboard/metrics",
            "anomalies": "/dashboard/anomalies",
            "health": "/health"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    monitor = app.state.monitor
    
    status = {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "connections": {
            "database": monitor.db_pool is not None and not monitor.db_pool._closed,
            "redis": monitor.redis_client is not None
        }
    }
    
    if not all(status["connections"].values()):
        status["status"] = "degraded"
    
    return status


@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint"""
    return Response(
        content=generate_latest(registry),
        media_type="text/plain"
    )


@app.get("/dashboard/metrics")
async def get_dashboard_metrics():
    """Real-time metrics for frontend dashboard"""
    monitor = app.state.monitor
    
    # Get latest metrics from Redis cache
    if monitor.redis_client:
        cached = monitor.redis_client.get('monitoring:latest_metrics')
        if cached:
            return json.loads(cached)
    
    # Fallback to direct collection
    supabase_metrics = await monitor.collect_supabase_metrics()
    railway_metrics = await monitor.collect_railway_metrics()
    
    return {
        "timestamp": datetime.now().isoformat(),
        "metrics": {
            "predictions_24h": supabase_metrics.get('total_predictions', 0),
            "active_users": supabase_metrics.get('active_users', 0),
            "model_accuracy": supabase_metrics.get('accuracy', []),
            "top_players": supabase_metrics.get('top_players', []),
            "system": railway_metrics,
            "cache_hit_rate": railway_metrics.get('cache_hit_rate', 0)
        }
    }


@app.get("/dashboard/anomalies")
async def get_anomalies():
    """Get recent anomalies"""
    monitor = app.state.monitor
    anomalies = await monitor.detect_anomalies()
    
    return {
        "timestamp": datetime.now().isoformat(),
        "count": len(anomalies),
        "anomalies": anomalies
    }


@app.websocket("/dashboard/live")
async def dashboard_websocket(websocket: WebSocket):
    """WebSocket for real-time updates"""
    await websocket.accept()
    monitor = app.state.monitor
    
    try:
        while True:
            # Send metrics every 5 seconds
            metrics = await get_dashboard_metrics()
            await websocket.send_json(metrics)
            await asyncio.sleep(5)
            
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        await websocket.close()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8001)),
        reload=os.getenv("ENVIRONMENT", "development") == "development"
    )