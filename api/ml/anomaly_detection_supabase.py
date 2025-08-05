"""
Supabase-based anomaly detection for NBA ML Platform
"""
import asyncio
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import logging
import json
from sqlalchemy import text
from database.connection import get_db_session
import numpy as np
from scipy import stats
import redis
import os

logger = logging.getLogger(__name__)


class SupabaseAnomalyDetector:
    """Anomaly detection using Supabase functions and queries"""
    
    def __init__(self):
        self.redis_url = os.getenv("REDIS_URL")
        self.redis_client = redis.from_url(self.redis_url, decode_responses=True) if self.redis_url else None
        self.alert_webhook = os.getenv("ALERT_WEBHOOK_URL")
        
        # Anomaly thresholds
        self.thresholds = {
            "z_score": 3.0,  # Standard deviations from mean
            "min_points": 0,
            "max_points": 60,
            "min_confidence": 0.3,
            "max_confidence": 0.99,
            "drift_threshold": 0.2,  # 20% change
            "volume_spike_factor": 3.0
        }
    
    async def detect_prediction_anomalies(self, 
                                        time_window: str = "1 hour") -> List[Dict[str, Any]]:
        """Detect anomalies in recent predictions"""
        anomalies = []
        
        async with get_db_session() as session:
            # Statistical anomaly detection
            stats_query = f"""
            WITH prediction_stats AS (
                SELECT 
                    AVG(points_predicted) as mean_points,
                    STDDEV(points_predicted) as std_points,
                    AVG(rebounds_predicted) as mean_rebounds,
                    STDDEV(rebounds_predicted) as std_rebounds,
                    AVG(assists_predicted) as mean_assists,
                    STDDEV(assists_predicted) as std_assists,
                    AVG(confidence) as mean_confidence,
                    STDDEV(confidence) as std_confidence,
                    PERCENTILE_CONT(0.01) WITHIN GROUP (ORDER BY points_predicted) as p1_points,
                    PERCENTILE_CONT(0.99) WITHIN GROUP (ORDER BY points_predicted) as p99_points
                FROM predictions
                WHERE created_at > NOW() - INTERVAL '7 days'
            ),
            recent_predictions AS (
                SELECT 
                    p.*,
                    ps.mean_points,
                    ps.std_points,
                    ps.mean_confidence,
                    ps.std_confidence,
                    ps.p1_points,
                    ps.p99_points
                FROM predictions p, prediction_stats ps
                WHERE p.created_at > NOW() - INTERVAL '{time_window}'
            )
            SELECT 
                id,
                player_id,
                player_name,
                game_date,
                points_predicted,
                rebounds_predicted,
                assists_predicted,
                confidence,
                model_version,
                created_at,
                CASE WHEN std_points > 0 THEN 
                    ABS(points_predicted - mean_points) / std_points 
                ELSE 0 END as points_z_score,
                CASE WHEN std_confidence > 0 THEN 
                    ABS(confidence - mean_confidence) / std_confidence 
                ELSE 0 END as confidence_z_score
            FROM recent_predictions
            WHERE 
                -- Statistical outliers
                (std_points > 0 AND ABS(points_predicted - mean_points) / std_points > :z_threshold)
                -- Impossible values
                OR points_predicted < :min_points
                OR points_predicted > :max_points
                -- Confidence anomalies
                OR confidence < :min_confidence
                OR confidence > :max_confidence
                -- Extreme percentiles
                OR points_predicted < p1_points
                OR points_predicted > p99_points
            ORDER BY created_at DESC
            """
            
            result = await session.execute(
                text(stats_query),
                {
                    "z_threshold": self.thresholds["z_score"],
                    "min_points": self.thresholds["min_points"],
                    "max_points": self.thresholds["max_points"],
                    "min_confidence": self.thresholds["min_confidence"],
                    "max_confidence": self.thresholds["max_confidence"]
                }
            )
            
            for row in result.fetchall():
                anomaly = {
                    "id": row.id,
                    "type": "prediction",
                    "player_id": row.player_id,
                    "player_name": row.player_name,
                    "timestamp": row.created_at.isoformat(),
                    "values": {
                        "points_predicted": float(row.points_predicted),
                        "rebounds_predicted": float(row.rebounds_predicted),
                        "assists_predicted": float(row.assists_predicted),
                        "confidence": float(row.confidence)
                    },
                    "anomaly_scores": {
                        "points_z_score": float(row.points_z_score) if row.points_z_score else 0,
                        "confidence_z_score": float(row.confidence_z_score) if row.confidence_z_score else 0
                    },
                    "anomaly_types": []
                }
                
                # Classify anomaly types
                if row.points_z_score and row.points_z_score > self.thresholds["z_score"]:
                    anomaly["anomaly_types"].append("statistical_outlier")
                
                if row.points_predicted < self.thresholds["min_points"] or \
                   row.points_predicted > self.thresholds["max_points"]:
                    anomaly["anomaly_types"].append("impossible_value")
                
                if row.confidence < self.thresholds["min_confidence"]:
                    anomaly["anomaly_types"].append("low_confidence")
                elif row.confidence > self.thresholds["max_confidence"]:
                    anomaly["anomaly_types"].append("suspiciously_high_confidence")
                
                anomaly["severity"] = self._calculate_severity(anomaly)
                anomalies.append(anomaly)
        
        return anomalies
    
    async def detect_drift(self, lookback_hours: int = 24) -> Dict[str, Any]:
        """Detect feature and prediction drift"""
        drift_results = {
            "timestamp": datetime.now().isoformat(),
            "drift_detected": False,
            "metrics": []
        }
        
        async with get_db_session() as session:
            # Hourly statistics for drift detection
            drift_query = """
            WITH hourly_stats AS (
                SELECT 
                    date_trunc('hour', created_at) as hour,
                    AVG(points_predicted) as avg_points,
                    STDDEV(points_predicted) as std_points,
                    AVG(rebounds_predicted) as avg_rebounds,
                    AVG(assists_predicted) as avg_assists,
                    AVG(confidence) as avg_confidence,
                    COUNT(*) as prediction_count,
                    COUNT(DISTINCT player_id) as unique_players,
                    COUNT(DISTINCT model_version) as model_versions
                FROM predictions
                WHERE created_at > NOW() - INTERVAL ':hours hours'
                GROUP BY date_trunc('hour', created_at)
                ORDER BY hour
            ),
            drift_analysis AS (
                SELECT 
                    hour,
                    avg_points,
                    avg_rebounds,
                    avg_assists,
                    avg_confidence,
                    prediction_count,
                    LAG(avg_points, 1) OVER (ORDER BY hour) as prev_avg_points,
                    LAG(avg_confidence, 1) OVER (ORDER BY hour) as prev_avg_confidence,
                    LAG(prediction_count, 1) OVER (ORDER BY hour) as prev_count
                FROM hourly_stats
            )
            SELECT 
                hour,
                avg_points,
                avg_confidence,
                prediction_count,
                CASE WHEN prev_avg_points > 0 THEN 
                    ABS(avg_points - prev_avg_points) / prev_avg_points 
                ELSE 0 END as points_drift,
                CASE WHEN prev_avg_confidence > 0 THEN 
                    ABS(avg_confidence - prev_avg_confidence) / prev_avg_confidence 
                ELSE 0 END as confidence_drift,
                CASE WHEN prev_count > 0 THEN 
                    ABS(prediction_count - prev_count)::float / prev_count 
                ELSE 0 END as volume_drift
            FROM drift_analysis
            WHERE prev_avg_points IS NOT NULL
            """
            
            result = await session.execute(
                text(drift_query.replace(":hours", str(lookback_hours)))
            )
            
            for row in result.fetchall():
                hour_metrics = {
                    "hour": row.hour.isoformat(),
                    "avg_points": float(row.avg_points),
                    "avg_confidence": float(row.avg_confidence),
                    "prediction_count": row.prediction_count,
                    "drift": {
                        "points": float(row.points_drift) if row.points_drift else 0,
                        "confidence": float(row.confidence_drift) if row.confidence_drift else 0,
                        "volume": float(row.volume_drift) if row.volume_drift else 0
                    }
                }
                
                # Check for significant drift
                if (hour_metrics["drift"]["points"] > self.thresholds["drift_threshold"] or
                    hour_metrics["drift"]["confidence"] > self.thresholds["drift_threshold"] or
                    hour_metrics["drift"]["volume"] > self.thresholds["volume_spike_factor"]):
                    hour_metrics["significant_drift"] = True
                    drift_results["drift_detected"] = True
                else:
                    hour_metrics["significant_drift"] = False
                
                drift_results["metrics"].append(hour_metrics)
        
        # Perform statistical tests
        if len(drift_results["metrics"]) > 10:
            drift_results["statistical_tests"] = await self._perform_drift_tests(drift_results["metrics"])
        
        return drift_results
    
    async def detect_model_degradation(self, model_version: str) -> Dict[str, Any]:
        """Detect if model performance is degrading"""
        async with get_db_session() as session:
            # Get performance over time
            performance_query = """
            WITH daily_performance AS (
                SELECT 
                    DATE(created_at) as date,
                    COUNT(*) as predictions,
                    AVG(ABS(points_predicted - points_actual)) as mae_points,
                    AVG(confidence) as avg_confidence,
                    COUNT(CASE WHEN points_actual IS NOT NULL THEN 1 END) as evaluated
                FROM predictions
                WHERE model_version = :model_version
                    AND created_at > NOW() - INTERVAL '14 days'
                GROUP BY DATE(created_at)
                ORDER BY date
            )
            SELECT 
                date,
                predictions,
                mae_points,
                avg_confidence,
                evaluated,
                AVG(mae_points) OVER (ORDER BY date ROWS BETWEEN 6 PRECEDING AND CURRENT ROW) as ma7_mae
            FROM daily_performance
            """
            
            result = await session.execute(
                text(performance_query),
                {"model_version": model_version}
            )
            
            performance_data = []
            for row in result.fetchall():
                performance_data.append({
                    "date": row.date.isoformat(),
                    "mae_points": float(row.mae_points) if row.mae_points else None,
                    "ma7_mae": float(row.ma7_mae) if row.ma7_mae else None,
                    "predictions": row.predictions,
                    "evaluated": row.evaluated
                })
            
            # Detect degradation trends
            degradation = {
                "model_version": model_version,
                "degradation_detected": False,
                "performance_history": performance_data,
                "alerts": []
            }
            
            if len(performance_data) >= 7:
                recent_mae = [d["mae_points"] for d in performance_data[-7:] if d["mae_points"]]
                older_mae = [d["mae_points"] for d in performance_data[-14:-7] if d["mae_points"]]
                
                if recent_mae and older_mae:
                    # Mann-Whitney U test for degradation
                    statistic, p_value = stats.mannwhitneyu(recent_mae, older_mae, alternative='greater')
                    
                    if p_value < 0.05:
                        degradation["degradation_detected"] = True
                        degradation["alerts"].append({
                            "type": "performance_degradation",
                            "message": f"Model {model_version} shows significant performance degradation",
                            "p_value": p_value,
                            "recent_mae": np.mean(recent_mae),
                            "previous_mae": np.mean(older_mae)
                        })
            
            return degradation
    
    async def detect_volume_anomalies(self) -> List[Dict[str, Any]]:
        """Detect anomalies in prediction volume"""
        anomalies = []
        
        async with get_db_session() as session:
            # Volume analysis
            volume_query = """
            WITH hourly_volume AS (
                SELECT 
                    date_trunc('hour', created_at) as hour,
                    COUNT(*) as prediction_count,
                    COUNT(DISTINCT user_id) as unique_users,
                    COUNT(DISTINCT player_id) as unique_players
                FROM predictions
                WHERE created_at > NOW() - INTERVAL '48 hours'
                GROUP BY date_trunc('hour', created_at)
            ),
            volume_stats AS (
                SELECT 
                    AVG(prediction_count) as mean_volume,
                    STDDEV(prediction_count) as std_volume,
                    PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY prediction_count) as p95_volume
                FROM hourly_volume
                WHERE hour < NOW() - INTERVAL '1 hour'
            )
            SELECT 
                h.hour,
                h.prediction_count,
                h.unique_users,
                h.unique_players,
                v.mean_volume,
                v.std_volume,
                v.p95_volume,
                CASE WHEN v.std_volume > 0 THEN 
                    (h.prediction_count - v.mean_volume) / v.std_volume 
                ELSE 0 END as z_score
            FROM hourly_volume h, volume_stats v
            WHERE h.hour >= NOW() - INTERVAL '2 hours'
                AND (
                    h.prediction_count > v.p95_volume * 2
                    OR h.prediction_count < v.mean_volume * 0.1
                    OR (v.std_volume > 0 AND ABS(h.prediction_count - v.mean_volume) / v.std_volume > 3)
                )
            """
            
            result = await session.execute(text(volume_query))
            
            for row in result.fetchall():
                anomaly = {
                    "type": "volume",
                    "hour": row.hour.isoformat(),
                    "prediction_count": row.prediction_count,
                    "unique_users": row.unique_users,
                    "expected_volume": float(row.mean_volume),
                    "z_score": float(row.z_score) if row.z_score else 0,
                    "anomaly_types": []
                }
                
                if row.prediction_count > row.p95_volume * 2:
                    anomaly["anomaly_types"].append("volume_spike")
                elif row.prediction_count < row.mean_volume * 0.1:
                    anomaly["anomaly_types"].append("volume_drop")
                
                anomaly["severity"] = "high" if abs(row.z_score) > 4 else "medium"
                anomalies.append(anomaly)
        
        return anomalies
    
    async def create_alert(self, 
                          anomaly_type: str,
                          severity: str,
                          details: Dict[str, Any],
                          prediction_id: Optional[int] = None) -> int:
        """Create an anomaly alert in the database"""
        async with get_db_session() as session:
            query = """
            INSERT INTO anomaly_alerts (anomaly_type, severity, prediction_id, details)
            VALUES (:anomaly_type, :severity, :prediction_id, :details)
            RETURNING id
            """
            
            result = await session.execute(
                text(query),
                {
                    "anomaly_type": anomaly_type,
                    "severity": severity,
                    "prediction_id": prediction_id,
                    "details": json.dumps(details)
                }
            )
            await session.commit()
            
            alert_id = result.scalar()
            
            # Send webhook notification for critical alerts
            if severity == "critical" and self.alert_webhook:
                await self._send_webhook_alert(anomaly_type, details)
            
            return alert_id
    
    async def run_anomaly_detection_batch(self) -> Dict[str, Any]:
        """Run all anomaly detection checks"""
        logger.info("Running batch anomaly detection")
        
        results = {
            "timestamp": datetime.now().isoformat(),
            "anomalies": {
                "predictions": [],
                "drift": {},
                "volume": [],
                "model_degradation": []
            },
            "summary": {
                "total_anomalies": 0,
                "critical_count": 0,
                "alerts_created": 0
            }
        }
        
        try:
            # Detect prediction anomalies
            prediction_anomalies = await self.detect_prediction_anomalies()
            results["anomalies"]["predictions"] = prediction_anomalies
            
            # Detect drift
            drift_results = await self.detect_drift()
            results["anomalies"]["drift"] = drift_results
            
            # Detect volume anomalies
            volume_anomalies = await self.detect_volume_anomalies()
            results["anomalies"]["volume"] = volume_anomalies
            
            # Check for model degradation
            # Get active model versions
            async with get_db_session() as session:
                model_query = """
                SELECT DISTINCT model_version 
                FROM predictions 
                WHERE created_at > NOW() - INTERVAL '24 hours'
                """
                model_result = await session.execute(text(model_query))
                model_versions = [row[0] for row in model_result.fetchall()]
            
            for model_version in model_versions:
                degradation = await self.detect_model_degradation(model_version)
                if degradation["degradation_detected"]:
                    results["anomalies"]["model_degradation"].append(degradation)
            
            # Create alerts for critical anomalies
            critical_anomalies = [
                a for a in prediction_anomalies 
                if a["severity"] == "critical"
            ]
            
            for anomaly in critical_anomalies:
                alert_id = await self.create_alert(
                    anomaly_type="prediction_anomaly",
                    severity="critical",
                    details=anomaly,
                    prediction_id=anomaly["id"]
                )
                results["summary"]["alerts_created"] += 1
            
            # Update summary
            results["summary"]["total_anomalies"] = (
                len(prediction_anomalies) + 
                len(volume_anomalies) +
                len(results["anomalies"]["model_degradation"])
            )
            results["summary"]["critical_count"] = len(critical_anomalies)
            
            # Cache results
            if self.redis_client:
                self.redis_client.setex(
                    "anomaly_detection:latest",
                    300,  # 5 minutes
                    json.dumps(results)
                )
            
        except Exception as e:
            logger.error(f"Error in batch anomaly detection: {e}")
            results["error"] = str(e)
        
        return results
    
    def _calculate_severity(self, anomaly: Dict[str, Any]) -> str:
        """Calculate anomaly severity"""
        severity_score = 0
        
        # Check z-scores
        if anomaly["anomaly_scores"]["points_z_score"] > 4:
            severity_score += 3
        elif anomaly["anomaly_scores"]["points_z_score"] > 3:
            severity_score += 2
        
        # Check for impossible values
        if "impossible_value" in anomaly["anomaly_types"]:
            severity_score += 5
        
        # Check confidence
        if anomaly["values"]["confidence"] < 0.2:
            severity_score += 3
        
        # Map to severity levels
        if severity_score >= 5:
            return "critical"
        elif severity_score >= 3:
            return "high"
        elif severity_score >= 1:
            return "medium"
        else:
            return "low"
    
    async def _perform_drift_tests(self, metrics: List[Dict]) -> Dict[str, Any]:
        """Perform statistical tests for drift detection"""
        # Extract time series
        points_series = [m["avg_points"] for m in metrics]
        confidence_series = [m["avg_confidence"] for m in metrics]
        
        # Split into two halves for comparison
        mid_point = len(points_series) // 2
        
        results = {}
        
        # Kolmogorov-Smirnov test for distribution drift
        if mid_point > 5:
            ks_stat_points, ks_p_points = stats.ks_2samp(
                points_series[:mid_point], 
                points_series[mid_point:]
            )
            results["ks_test_points"] = {
                "statistic": ks_stat_points,
                "p_value": ks_p_points,
                "significant": ks_p_points < 0.05
            }
            
            ks_stat_conf, ks_p_conf = stats.ks_2samp(
                confidence_series[:mid_point], 
                confidence_series[mid_point:]
            )
            results["ks_test_confidence"] = {
                "statistic": ks_stat_conf,
                "p_value": ks_p_conf,
                "significant": ks_p_conf < 0.05
            }
        
        return results
    
    async def _send_webhook_alert(self, anomaly_type: str, details: Dict[str, Any]):
        """Send alert to webhook"""
        if not self.alert_webhook:
            return
        
        import aiohttp
        
        payload = {
            "text": f"🚨 Critical Anomaly Detected: {anomaly_type}",
            "attachments": [{
                "color": "danger",
                "fields": [
                    {"title": "Type", "value": anomaly_type, "short": True},
                    {"title": "Time", "value": datetime.now().isoformat(), "short": True},
                    {"title": "Details", "value": json.dumps(details, indent=2)}
                ]
            }]
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(self.alert_webhook, json=payload) as response:
                    if response.status != 200:
                        logger.error(f"Failed to send webhook alert: {response.status}")
        except Exception as e:
            logger.error(f"Error sending webhook alert: {e}")