"""
Real-time dashboard API endpoints for monitoring
"""
from fastapi import APIRouter, Depends, HTTPException, WebSocket, WebSocketDisconnect
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import json
import asyncio
import logging
from supabase import create_client
import os
import redis
from sqlalchemy import text
from database.connection import get_db_session

logger = logging.getLogger(__name__)

router = APIRouter()


class DashboardAPI:
    """Real-time metrics for Vercel frontend"""
    
    def __init__(self):
        self.supabase_url = os.getenv("SUPABASE_URL", os.getenv("DATABASE_URL"))
        self.supabase_key = os.getenv("SUPABASE_ANON_KEY", "")
        self.redis_url = os.getenv("REDIS_URL")
        
        # Initialize Supabase client if URL and key available
        if self.supabase_url and self.supabase_key and "supabase" in self.supabase_url:
            self.supabase = create_client(self.supabase_url, self.supabase_key)
        else:
            self.supabase = None
            
        # Initialize Redis
        self.redis_client = redis.from_url(self.redis_url, decode_responses=True) if self.redis_url else None
    
    async def get_model_accuracy(self) -> Dict[str, Any]:
        """Get current model accuracy metrics"""
        async with get_db_session() as session:
            query = """
            SELECT 
                model_version,
                COUNT(*) as total_predictions,
                AVG(ABS(points_predicted - points_actual)) as mae_points,
                AVG(ABS(rebounds_predicted - rebounds_actual)) as mae_rebounds,
                AVG(ABS(assists_predicted - assists_actual)) as mae_assists,
                AVG(confidence) as avg_confidence
            FROM predictions
            WHERE created_at > NOW() - INTERVAL '24 hours'
                AND points_actual IS NOT NULL
            GROUP BY model_version
            ORDER BY model_version DESC
            """
            
            result = await session.execute(text(query))
            rows = result.fetchall()
            
            return [
                {
                    "model_version": row.model_version,
                    "total_predictions": row.total_predictions,
                    "mae_points": float(row.mae_points) if row.mae_points else 0,
                    "mae_rebounds": float(row.mae_rebounds) if row.mae_rebounds else 0,
                    "mae_assists": float(row.mae_assists) if row.mae_assists else 0,
                    "avg_confidence": float(row.avg_confidence) if row.avg_confidence else 0
                }
                for row in rows
            ]
    
    async def get_prediction_volume(self) -> Dict[str, Any]:
        """Get prediction volume statistics"""
        async with get_db_session() as session:
            # Hourly volume for last 24 hours
            hourly_query = """
            SELECT 
                date_trunc('hour', created_at) as hour,
                COUNT(*) as prediction_count,
                COUNT(DISTINCT user_id) as unique_users,
                AVG(confidence) as avg_confidence
            FROM predictions
            WHERE created_at > NOW() - INTERVAL '24 hours'
            GROUP BY date_trunc('hour', created_at)
            ORDER BY hour DESC
            """
            
            hourly_result = await session.execute(text(hourly_query))
            hourly_data = [
                {
                    "hour": row.hour.isoformat(),
                    "count": row.prediction_count,
                    "unique_users": row.unique_users,
                    "avg_confidence": float(row.avg_confidence) if row.avg_confidence else 0
                }
                for row in hourly_result.fetchall()
            ]
            
            # Total stats
            total_query = """
            SELECT 
                COUNT(*) as total_24h,
                COUNT(DISTINCT user_id) as unique_users_24h,
                COUNT(DISTINCT player_id) as unique_players_24h
            FROM predictions
            WHERE created_at > NOW() - INTERVAL '24 hours'
            """
            
            total_result = await session.execute(text(total_query))
            total_row = total_result.fetchone()
            
            return {
                "hourly": hourly_data,
                "total_24h": total_row.total_24h if total_row else 0,
                "unique_users_24h": total_row.unique_users_24h if total_row else 0,
                "unique_players_24h": total_row.unique_players_24h if total_row else 0
            }
    
    async def get_popular_players(self) -> List[Dict[str, Any]]:
        """Get most predicted players"""
        async with get_db_session() as session:
            query = """
            SELECT 
                player_id,
                player_name,
                COUNT(*) as prediction_count,
                AVG(points_predicted) as avg_points_predicted,
                AVG(confidence) as avg_confidence,
                COUNT(DISTINCT user_id) as unique_users
            FROM predictions
            WHERE created_at > NOW() - INTERVAL '24 hours'
            GROUP BY player_id, player_name
            ORDER BY prediction_count DESC
            LIMIT 10
            """
            
            result = await session.execute(text(query))
            
            return [
                {
                    "player_id": row.player_id,
                    "player_name": row.player_name,
                    "prediction_count": row.prediction_count,
                    "avg_points_predicted": float(row.avg_points_predicted) if row.avg_points_predicted else 0,
                    "avg_confidence": float(row.avg_confidence) if row.avg_confidence else 0,
                    "unique_users": row.unique_users
                }
                for row in result.fetchall()
            ]
    
    async def get_system_status(self) -> Dict[str, Any]:
        """Get system health status"""
        status = {
            "timestamp": datetime.now().isoformat(),
            "services": {}
        }
        
        # Check database
        try:
            async with get_db_session() as session:
                await session.execute(text("SELECT 1"))
            status["services"]["database"] = "healthy"
        except Exception as e:
            status["services"]["database"] = "unhealthy"
            logger.error(f"Database health check failed: {e}")
        
        # Check Redis
        if self.redis_client:
            try:
                self.redis_client.ping()
                status["services"]["redis"] = "healthy"
                
                # Get cache stats
                info = self.redis_client.info("stats")
                hits = info.get("keyspace_hits", 0)
                misses = info.get("keyspace_misses", 0)
                if hits + misses > 0:
                    status["cache_hit_rate"] = hits / (hits + misses)
                else:
                    status["cache_hit_rate"] = 0
            except Exception as e:
                status["services"]["redis"] = "unhealthy"
                logger.error(f"Redis health check failed: {e}")
        else:
            status["services"]["redis"] = "not_configured"
        
        # Check Supabase
        if self.supabase:
            status["services"]["supabase"] = "configured"
        else:
            status["services"]["supabase"] = "not_configured"
        
        # Overall status
        unhealthy_services = [k for k, v in status["services"].items() if v == "unhealthy"]
        if unhealthy_services:
            status["overall"] = "degraded"
        else:
            status["overall"] = "healthy"
        
        return status
    
    async def get_recent_anomalies(self) -> List[Dict[str, Any]]:
        """Get recent anomalies detected"""
        async with get_db_session() as session:
            query = """
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
                CASE 
                    WHEN p.points_predicted < 0 OR p.points_predicted > 60 THEN 'impossible_value'
                    WHEN ABS(p.points_predicted - s.mean_points) > 3 * s.std_points THEN 'statistical_outlier'
                    WHEN p.confidence < 0.3 THEN 'low_confidence'
                    ELSE 'normal'
                END as anomaly_type
            FROM predictions p, stats s
            WHERE p.created_at > NOW() - INTERVAL '1 hour'
                AND (
                    p.points_predicted < 0 
                    OR p.points_predicted > 60
                    OR ABS(p.points_predicted - s.mean_points) > 3 * s.std_points
                    OR p.confidence < 0.3
                )
            ORDER BY p.created_at DESC
            LIMIT 20
            """
            
            result = await session.execute(text(query))
            
            return [
                {
                    "id": row.id,
                    "player_name": row.player_name,
                    "points_predicted": float(row.points_predicted),
                    "confidence": float(row.confidence),
                    "created_at": row.created_at.isoformat(),
                    "anomaly_type": row.anomaly_type
                }
                for row in result.fetchall()
            ]
    
    async def get_feature_drift(self) -> Dict[str, Any]:
        """Get feature drift metrics"""
        async with get_db_session() as session:
            query = """
            WITH daily_stats AS (
                SELECT 
                    DATE(created_at) as date,
                    AVG(points_predicted) as avg_points,
                    AVG(confidence) as avg_confidence,
                    COUNT(*) as prediction_count
                FROM predictions
                WHERE created_at > NOW() - INTERVAL '7 days'
                GROUP BY DATE(created_at)
                ORDER BY date
            )
            SELECT 
                date,
                avg_points,
                avg_confidence,
                prediction_count,
                avg_points - LAG(avg_points) OVER (ORDER BY date) as points_change,
                avg_confidence - LAG(avg_confidence) OVER (ORDER BY date) as confidence_change
            FROM daily_stats
            """
            
            result = await session.execute(text(query))
            rows = result.fetchall()
            
            drift_data = []
            for row in rows:
                drift_data.append({
                    "date": row.date.isoformat(),
                    "avg_points": float(row.avg_points) if row.avg_points else 0,
                    "avg_confidence": float(row.avg_confidence) if row.avg_confidence else 0,
                    "prediction_count": row.prediction_count,
                    "points_change": float(row.points_change) if row.points_change else 0,
                    "confidence_change": float(row.confidence_change) if row.confidence_change else 0
                })
            
            # Detect significant drift
            significant_drift = any(
                abs(d["points_change"]) > 5 or abs(d["confidence_change"]) > 0.1
                for d in drift_data if d["points_change"] is not None
            )
            
            return {
                "daily_stats": drift_data,
                "significant_drift_detected": significant_drift
            }


# Initialize dashboard API
dashboard_api = DashboardAPI()


@router.get("/v1/dashboard/metrics")
async def get_dashboard_metrics():
    """Real-time metrics for frontend dashboard"""
    try:
        # Check cache first
        if dashboard_api.redis_client:
            cached = dashboard_api.redis_client.get("dashboard:metrics")
            if cached:
                return json.loads(cached)
        
        # Collect all metrics
        metrics = {
            "timestamp": datetime.now().isoformat(),
            "current_accuracy": await dashboard_api.get_model_accuracy(),
            "predictions_24h": await dashboard_api.get_prediction_volume(),
            "top_players": await dashboard_api.get_popular_players(),
            "system_health": await dashboard_api.get_system_status(),
            "recent_anomalies": await dashboard_api.get_recent_anomalies(),
            "feature_drift": await dashboard_api.get_feature_drift()
        }
        
        # Cache for 1 minute
        if dashboard_api.redis_client:
            dashboard_api.redis_client.setex(
                "dashboard:metrics",
                60,
                json.dumps(metrics)
            )
        
        return metrics
        
    except Exception as e:
        logger.error(f"Error getting dashboard metrics: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch dashboard metrics")


@router.get("/v1/dashboard/anomalies")
async def get_anomalies():
    """Get detailed anomaly information"""
    try:
        anomalies = await dashboard_api.get_recent_anomalies()
        
        # Group by type
        grouped = {}
        for anomaly in anomalies:
            anomaly_type = anomaly["anomaly_type"]
            if anomaly_type not in grouped:
                grouped[anomaly_type] = []
            grouped[anomaly_type].append(anomaly)
        
        return {
            "timestamp": datetime.now().isoformat(),
            "total_count": len(anomalies),
            "by_type": grouped,
            "anomalies": anomalies
        }
        
    except Exception as e:
        logger.error(f"Error getting anomalies: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch anomalies")


@router.get("/v1/dashboard/performance/{model_version}")
async def get_model_performance(model_version: str):
    """Get detailed performance metrics for a specific model"""
    try:
        async with get_db_session() as session:
            query = """
            SELECT 
                DATE(created_at) as date,
                COUNT(*) as predictions,
                AVG(ABS(points_predicted - points_actual)) as mae_points,
                AVG(ABS(rebounds_predicted - rebounds_actual)) as mae_rebounds,
                AVG(ABS(assists_predicted - assists_actual)) as mae_assists,
                AVG(confidence) as avg_confidence,
                PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY ABS(points_predicted - points_actual)) as median_error
            FROM predictions
            WHERE model_version = :model_version
                AND points_actual IS NOT NULL
                AND created_at > NOW() - INTERVAL '30 days'
            GROUP BY DATE(created_at)
            ORDER BY date DESC
            """
            
            result = await session.execute(
                text(query),
                {"model_version": model_version}
            )
            
            performance_data = []
            for row in result.fetchall():
                performance_data.append({
                    "date": row.date.isoformat(),
                    "predictions": row.predictions,
                    "mae_points": float(row.mae_points) if row.mae_points else 0,
                    "mae_rebounds": float(row.mae_rebounds) if row.mae_rebounds else 0,
                    "mae_assists": float(row.mae_assists) if row.mae_assists else 0,
                    "avg_confidence": float(row.avg_confidence) if row.avg_confidence else 0,
                    "median_error": float(row.median_error) if row.median_error else 0
                })
            
            return {
                "model_version": model_version,
                "performance_history": performance_data
            }
            
    except Exception as e:
        logger.error(f"Error getting model performance: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch model performance")


# WebSocket manager for live updates
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
    
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
    
    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
    
    async def broadcast(self, data: dict):
        """Broadcast to all connected clients"""
        for connection in self.active_connections:
            try:
                await connection.send_json(data)
            except Exception:
                # Remove dead connections
                self.active_connections.remove(connection)


manager = ConnectionManager()


@router.websocket("/v1/dashboard/live")
async def dashboard_websocket(websocket: WebSocket):
    """WebSocket for real-time dashboard updates"""
    await manager.connect(websocket)
    
    try:
        while True:
            # Send updates every 5 seconds
            metrics = await get_dashboard_metrics()
            await websocket.send_json(metrics)
            
            # Wait for next update or client message
            try:
                await asyncio.wait_for(websocket.receive_text(), timeout=5.0)
            except asyncio.TimeoutError:
                continue
                
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        manager.disconnect(websocket)