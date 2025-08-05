"""
Background monitoring jobs for Railway
"""
import asyncio
import logging
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from datetime import datetime
import os
from sqlalchemy import text
from database.connection import get_db_session
from api.ml.anomaly_detection_supabase import SupabaseAnomalyDetector

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MonitoringJobs:
    """Background monitoring jobs for Railway deployment"""
    
    def __init__(self):
        self.scheduler = AsyncIOScheduler()
        self.anomaly_detector = SupabaseAnomalyDetector()
        self.setup_jobs()
    
    def setup_jobs(self):
        """Configure scheduled jobs"""
        # Every 5 minutes: Check for anomalies
        self.scheduler.add_job(
            self.check_anomalies,
            'interval',
            minutes=5,
            id='anomaly_check',
            name='Anomaly Detection'
        )
        
        # Every hour: Update materialized views
        self.scheduler.add_job(
            self.refresh_analytics,
            'interval', 
            hours=1,
            id='refresh_analytics',
            name='Refresh Analytics Views'
        )
        
        # Every 10 minutes: Calculate drift metrics
        self.scheduler.add_job(
            self.calculate_drift,
            'interval',
            minutes=10,
            id='drift_detection',
            name='Drift Detection'
        )
        
        # Every 30 minutes: Check model performance
        self.scheduler.add_job(
            self.check_model_performance,
            'interval',
            minutes=30,
            id='model_performance',
            name='Model Performance Check'
        )
        
        # Every hour: Clean up old data
        self.scheduler.add_job(
            self.cleanup_old_data,
            'interval',
            hours=1,
            id='data_cleanup',
            name='Data Cleanup'
        )
        
        # Daily at 2 AM: Generate daily report
        self.scheduler.add_job(
            self.generate_daily_report,
            'cron',
            hour=2,
            minute=0,
            id='daily_report',
            name='Daily Report Generation'
        )
    
    async def check_anomalies(self):
        """Check for anomalies in predictions"""
        logger.info("Running anomaly detection check")
        
        try:
            # Run comprehensive anomaly detection
            results = await self.anomaly_detector.run_anomaly_detection_batch()
            
            # Log summary
            logger.info(f"Anomaly detection complete: {results['summary']['total_anomalies']} anomalies found, "
                       f"{results['summary']['critical_count']} critical")
            
            # Handle critical anomalies
            if results['summary']['critical_count'] > 0:
                logger.warning(f"Critical anomalies detected! Created {results['summary']['alerts_created']} alerts")
                
                # You could send additional notifications here
                await self._notify_critical_anomalies(results)
            
        except Exception as e:
            logger.error(f"Error in anomaly detection job: {e}")
    
    async def refresh_analytics(self):
        """Refresh materialized views for analytics"""
        logger.info("Refreshing analytics views")
        
        try:
            async with get_db_session() as session:
                # Refresh prediction metrics
                await session.execute(
                    text("REFRESH MATERIALIZED VIEW CONCURRENTLY prediction_metrics_hourly")
                )
                
                # Refresh feature drift view
                await session.execute(
                    text("REFRESH MATERIALIZED VIEW CONCURRENTLY feature_drift_daily")
                )
                
                # Refresh feature importance summary
                await session.execute(
                    text("REFRESH MATERIALIZED VIEW CONCURRENTLY feature_importance_summary")
                )
                
                await session.commit()
                
            logger.info("Analytics views refreshed successfully")
            
        except Exception as e:
            logger.error(f"Error refreshing analytics views: {e}")
    
    async def calculate_drift(self):
        """Calculate and monitor feature drift"""
        logger.info("Calculating drift metrics")
        
        try:
            # Detect drift
            drift_results = await self.anomaly_detector.detect_drift(lookback_hours=24)
            
            if drift_results["drift_detected"]:
                logger.warning("Significant drift detected!")
                
                # Create drift alert
                async with get_db_session() as session:
                    await session.execute(
                        text("""
                        INSERT INTO anomaly_alerts (anomaly_type, severity, details)
                        VALUES ('feature_drift', 'high', :details)
                        """),
                        {"details": drift_results}
                    )
                    await session.commit()
            
            # Log drift summary
            drift_count = sum(1 for m in drift_results["metrics"] if m.get("significant_drift", False))
            logger.info(f"Drift check complete: {drift_count} hours with significant drift")
            
        except Exception as e:
            logger.error(f"Error in drift detection job: {e}")
    
    async def check_model_performance(self):
        """Check model performance and detect degradation"""
        logger.info("Checking model performance")
        
        try:
            # Get active model versions
            async with get_db_session() as session:
                result = await session.execute(
                    text("""
                    SELECT DISTINCT model_version 
                    FROM predictions 
                    WHERE created_at > NOW() - INTERVAL '24 hours'
                    """)
                )
                model_versions = [row[0] for row in result.fetchall()]
            
            degradation_detected = False
            
            for model_version in model_versions:
                degradation = await self.anomaly_detector.detect_model_degradation(model_version)
                
                if degradation["degradation_detected"]:
                    degradation_detected = True
                    logger.warning(f"Performance degradation detected for model {model_version}")
                    
                    # Create alert
                    for alert in degradation["alerts"]:
                        await self.anomaly_detector.create_alert(
                            anomaly_type="model_degradation",
                            severity="high",
                            details=alert
                        )
            
            if not degradation_detected:
                logger.info("All models performing within expected parameters")
                
        except Exception as e:
            logger.error(f"Error in model performance check: {e}")
    
    async def cleanup_old_data(self):
        """Clean up old monitoring data"""
        logger.info("Cleaning up old monitoring data")
        
        try:
            async with get_db_session() as session:
                # Clean old resolved alerts (keep 30 days)
                alerts_deleted = await session.execute(
                    text("""
                    DELETE FROM anomaly_alerts 
                    WHERE resolved = TRUE 
                    AND resolved_at < NOW() - INTERVAL '30 days'
                    """)
                )
                
                # Clean old explanations (keep 90 days)
                explanations_deleted = await session.execute(
                    text("""
                    DELETE FROM prediction_explanations 
                    WHERE created_at < NOW() - INTERVAL '90 days'
                    """)
                )
                
                await session.commit()
                
                logger.info(f"Cleanup complete: {alerts_deleted.rowcount} alerts, "
                           f"{explanations_deleted.rowcount} explanations deleted")
                
        except Exception as e:
            logger.error(f"Error in data cleanup job: {e}")
    
    async def generate_daily_report(self):
        """Generate daily monitoring report"""
        logger.info("Generating daily monitoring report")
        
        try:
            async with get_db_session() as session:
                # Get daily statistics
                stats_query = """
                SELECT 
                    COUNT(*) as total_predictions,
                    COUNT(DISTINCT user_id) as unique_users,
                    COUNT(DISTINCT player_id) as unique_players,
                    COUNT(DISTINCT model_version) as model_versions,
                    AVG(confidence) as avg_confidence,
                    COUNT(CASE WHEN points_actual IS NOT NULL THEN 1 END) as evaluated_predictions
                FROM predictions
                WHERE created_at >= CURRENT_DATE - INTERVAL '1 day'
                    AND created_at < CURRENT_DATE
                """
                
                stats = await session.execute(text(stats_query))
                daily_stats = stats.fetchone()
                
                # Get anomaly counts
                anomaly_query = """
                SELECT 
                    anomaly_type,
                    severity,
                    COUNT(*) as count
                FROM anomaly_alerts
                WHERE created_at >= CURRENT_DATE - INTERVAL '1 day'
                    AND created_at < CURRENT_DATE
                GROUP BY anomaly_type, severity
                """
                
                anomalies = await session.execute(text(anomaly_query))
                anomaly_counts = anomalies.fetchall()
                
                # Get model accuracy
                accuracy_query = """
                SELECT 
                    model_version,
                    AVG(ABS(points_predicted - points_actual)) as mae_points,
                    COUNT(*) as predictions
                FROM predictions
                WHERE created_at >= CURRENT_DATE - INTERVAL '1 day'
                    AND created_at < CURRENT_DATE
                    AND points_actual IS NOT NULL
                GROUP BY model_version
                """
                
                accuracy = await session.execute(text(accuracy_query))
                model_accuracy = accuracy.fetchall()
            
            # Format report
            report = {
                "date": (datetime.now().date() - timedelta(days=1)).isoformat(),
                "statistics": {
                    "total_predictions": daily_stats.total_predictions,
                    "unique_users": daily_stats.unique_users,
                    "unique_players": daily_stats.unique_players,
                    "model_versions": daily_stats.model_versions,
                    "avg_confidence": float(daily_stats.avg_confidence) if daily_stats.avg_confidence else 0,
                    "evaluated_predictions": daily_stats.evaluated_predictions
                },
                "anomalies": [
                    {
                        "type": row.anomaly_type,
                        "severity": row.severity,
                        "count": row.count
                    }
                    for row in anomaly_counts
                ],
                "model_performance": [
                    {
                        "model_version": row.model_version,
                        "mae_points": float(row.mae_points) if row.mae_points else None,
                        "predictions": row.predictions
                    }
                    for row in model_accuracy
                ]
            }
            
            logger.info(f"Daily report generated: {report['statistics']['total_predictions']} predictions")
            
            # You could store this report or send it via email/webhook
            await self._send_daily_report(report)
            
        except Exception as e:
            logger.error(f"Error generating daily report: {e}")
    
    async def _notify_critical_anomalies(self, results: dict):
        """Send notifications for critical anomalies"""
        # This is where you'd integrate with your notification system
        # For now, just log
        critical_count = results['summary']['critical_count']
        logger.critical(f"CRITICAL ANOMALIES: {critical_count} detected")
    
    async def _send_daily_report(self, report: dict):
        """Send daily report (webhook, email, etc.)"""
        # This is where you'd send the report
        # For now, just log the summary
        logger.info(f"Daily report ready: {report['statistics']['total_predictions']} predictions, "
                   f"{len(report['anomalies'])} anomaly types")
    
    def start(self):
        """Start the scheduler"""
        self.scheduler.start()
        logger.info("Monitoring jobs scheduler started")
    
    def stop(self):
        """Stop the scheduler"""
        self.scheduler.shutdown()
        logger.info("Monitoring jobs scheduler stopped")


# Create and run the monitoring jobs if this is the main script
if __name__ == "__main__":
    monitor = MonitoringJobs()
    monitor.start()
    
    try:
        # Keep the script running
        asyncio.get_event_loop().run_forever()
    except KeyboardInterrupt:
        logger.info("Shutting down monitoring jobs...")
        monitor.stop()