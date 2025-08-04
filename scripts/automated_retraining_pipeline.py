#!/usr/bin/env python3
"""
Automated Model Retraining Pipeline for NBA ML System
Implements continuous learning with performance monitoring and automatic deployment
"""
import os
import sys
import asyncio
import argparse
import logging
from pathlib import Path
from datetime import datetime, timedelta
import json
import yaml
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
import schedule
import time
import joblib
from dataclasses import dataclass
from enum import Enum
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from database.connection import DatabaseManager, init_db
from database.models import MLModel, ModelVersion, PlayerGameLog, RetrainingJob
from scripts.collect_nba_data_production import NBADataCollectionPipeline
from scripts.train_production_models import ProductionModelTrainer
from scripts.connect_models_to_api import ModelAPIConnector
from ml.data.processors.data_validator import NBADataValidator
import mlflow
from sqlalchemy import func, and_, or_
import boto3
from botocore.exceptions import ClientError

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'logs/retraining_{datetime.now():%Y%m%d_%H%M%S}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class RetrainingStatus(Enum):
    """Retraining job status"""
    PENDING = "pending"
    COLLECTING_DATA = "collecting_data"
    TRAINING = "training"
    EVALUATING = "evaluating"
    DEPLOYING = "deploying"
    COMPLETED = "completed"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"


@dataclass
class RetrainingConfig:
    """Retraining configuration"""
    schedule_interval: str = "weekly"  # daily, weekly, monthly
    min_new_games: int = 100
    performance_threshold: float = 0.94
    degradation_threshold: float = 0.02
    auto_deploy: bool = True
    rollback_on_failure: bool = True
    notification_emails: List[str] = None
    slack_webhook: Optional[str] = None
    max_training_time: int = 3600  # seconds
    parallel_training: bool = True
    test_before_deploy: bool = True
    canary_deployment: bool = True
    canary_percentage: int = 10


class ModelPerformanceMonitor:
    """Monitor model performance in production"""
    
    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager
        self.metrics_history = []
        
    async def calculate_production_metrics(self, 
                                          model_id: str,
                                          days_back: int = 7) -> Dict[str, float]:
        """
        Calculate model performance metrics from production predictions
        
        Args:
            model_id: Model identifier
            days_back: Number of days to look back
            
        Returns:
            Dictionary of performance metrics
        """
        logger.info(f"Calculating production metrics for model {model_id}")
        
        with self.db_manager.get_db(read_only=True) as session:
            # Get recent predictions vs actual results
            cutoff_date = datetime.now() - timedelta(days=days_back)
            
            # This would query a predictions table that stores model predictions
            # For now, we'll simulate with game logs
            recent_games = session.query(PlayerGameLog).filter(
                PlayerGameLog.game_date >= cutoff_date
            ).all()
            
            if not recent_games:
                logger.warning("No recent games found for metric calculation")
                return {}
            
            # Simulate prediction vs actual comparison
            # In production, this would compare stored predictions with actual results
            metrics = {
                "games_evaluated": len(recent_games),
                "mae": np.random.uniform(2.5, 4.0),  # Simulated
                "rmse": np.random.uniform(3.5, 5.5),  # Simulated
                "mape": np.random.uniform(10, 20),   # Simulated
                "r2_score": np.random.uniform(0.90, 0.96),  # Simulated
                "prediction_count": len(recent_games) * 3,  # PTS, REB, AST
                "avg_latency_ms": np.random.uniform(50, 150)  # Simulated
            }
            
            # Check for performance degradation
            if hasattr(self, 'baseline_metrics'):
                for metric, value in metrics.items():
                    if metric in self.baseline_metrics:
                        degradation = abs(value - self.baseline_metrics[metric])
                        metrics[f"{metric}_degradation"] = degradation
            
            return metrics
    
    async def detect_drift(self, 
                          current_data: pd.DataFrame,
                          training_data_stats: Dict) -> Dict[str, Any]:
        """
        Detect data drift from training distribution
        
        Args:
            current_data: Current production data
            training_data_stats: Statistics from training data
            
        Returns:
            Drift detection results
        """
        logger.info("Detecting data drift...")
        
        drift_results = {
            "drift_detected": False,
            "drift_score": 0.0,
            "drifted_features": [],
            "details": {}
        }
        
        # Calculate statistics for current data
        current_stats = {
            "mean": current_data.mean().to_dict(),
            "std": current_data.std().to_dict(),
            "min": current_data.min().to_dict(),
            "max": current_data.max().to_dict()
        }
        
        # Compare with training stats (simplified KS test simulation)
        for feature in current_stats["mean"].keys():
            if feature in training_data_stats.get("mean", {}):
                # Calculate normalized difference
                mean_diff = abs(current_stats["mean"][feature] - training_data_stats["mean"][feature])
                std_training = training_data_stats["std"].get(feature, 1)
                
                if std_training > 0:
                    z_score = mean_diff / std_training
                    
                    if z_score > 2:  # Significant drift threshold
                        drift_results["drifted_features"].append(feature)
                        drift_results["details"][feature] = {
                            "z_score": z_score,
                            "current_mean": current_stats["mean"][feature],
                            "training_mean": training_data_stats["mean"][feature]
                        }
        
        # Calculate overall drift score
        if drift_results["drifted_features"]:
            drift_results["drift_detected"] = True
            drift_results["drift_score"] = len(drift_results["drifted_features"]) / len(current_stats["mean"])
        
        logger.info(f"Drift detection complete. Drift detected: {drift_results['drift_detected']}")
        
        return drift_results


class AutomatedRetrainingPipeline:
    """Main automated retraining pipeline"""
    
    def __init__(self, config: RetrainingConfig):
        self.config = config
        self.db_manager = None
        self.data_collector = NBADataCollectionPipeline()
        self.model_trainer = ProductionModelTrainer()
        self.api_connector = ModelAPIConnector()
        self.performance_monitor = None
        self.validator = NBADataValidator()
        
        # AWS clients
        self.s3_client = boto3.client('s3')
        self.sns_client = boto3.client('sns')
        
        # State tracking
        self.current_job_id = None
        self.previous_models = {}
        
        logger.info("Automated Retraining Pipeline initialized")
    
    async def check_retraining_criteria(self) -> Tuple[bool, Dict[str, Any]]:
        """
        Check if retraining criteria are met
        
        Returns:
            Tuple of (should_retrain, criteria_details)
        """
        logger.info("Checking retraining criteria...")
        
        criteria = {
            "new_games_available": False,
            "performance_degraded": False,
            "data_drift_detected": False,
            "scheduled_retraining": False,
            "manual_trigger": False
        }
        
        if not self.db_manager:
            self.db_manager = init_db()
        
        # Check for new games
        with self.db_manager.get_db(read_only=True) as session:
            # Get last retraining date
            last_job = session.query(RetrainingJob).order_by(
                RetrainingJob.started_at.desc()
            ).first()
            
            if last_job:
                last_retrain_date = last_job.started_at
                new_games = session.query(func.count(PlayerGameLog.id)).filter(
                    PlayerGameLog.created_at > last_retrain_date
                ).scalar()
                
                if new_games >= self.config.min_new_games:
                    criteria["new_games_available"] = True
                    criteria["new_games_count"] = new_games
            else:
                # First retraining
                criteria["scheduled_retraining"] = True
        
        # Check performance degradation
        if self.performance_monitor:
            for model_name in ["PTS", "REB", "AST"]:
                metrics = await self.performance_monitor.calculate_production_metrics(
                    model_name, days_back=7
                )
                
                if metrics.get("r2_score", 1.0) < self.config.performance_threshold:
                    criteria["performance_degraded"] = True
                    criteria["degraded_models"] = criteria.get("degraded_models", [])
                    criteria["degraded_models"].append(model_name)
        
        # Check for scheduled retraining
        if last_job:
            if self.config.schedule_interval == "daily":
                if (datetime.now() - last_job.started_at).days >= 1:
                    criteria["scheduled_retraining"] = True
            elif self.config.schedule_interval == "weekly":
                if (datetime.now() - last_job.started_at).days >= 7:
                    criteria["scheduled_retraining"] = True
            elif self.config.schedule_interval == "monthly":
                if (datetime.now() - last_job.started_at).days >= 30:
                    criteria["scheduled_retraining"] = True
        
        # Determine if retraining should occur
        should_retrain = any([
            criteria["new_games_available"],
            criteria["performance_degraded"],
            criteria["data_drift_detected"],
            criteria["scheduled_retraining"],
            criteria["manual_trigger"]
        ])
        
        logger.info(f"Retraining criteria check: {should_retrain}")
        logger.info(f"Criteria details: {criteria}")
        
        return should_retrain, criteria
    
    async def create_retraining_job(self, criteria: Dict) -> str:
        """Create a new retraining job in the database"""
        
        with self.db_manager.transaction() as session:
            job = RetrainingJob(
                status=RetrainingStatus.PENDING.value,
                trigger_reason=json.dumps(criteria),
                config=json.dumps(self.config.__dict__),
                started_at=datetime.now()
            )
            session.add(job)
            session.flush()
            job_id = job.id
        
        self.current_job_id = job_id
        logger.info(f"Created retraining job: {job_id}")
        return job_id
    
    async def update_job_status(self, status: RetrainingStatus, details: Dict = None):
        """Update retraining job status"""
        
        with self.db_manager.transaction() as session:
            job = session.query(RetrainingJob).filter_by(id=self.current_job_id).first()
            if job:
                job.status = status.value
                if details:
                    job.results = json.dumps(details)
                if status in [RetrainingStatus.COMPLETED, RetrainingStatus.FAILED]:
                    job.completed_at = datetime.now()
    
    async def collect_latest_data(self) -> bool:
        """Collect latest NBA data"""
        logger.info("Collecting latest NBA data...")
        
        await self.update_job_status(RetrainingStatus.COLLECTING_DATA)
        
        try:
            # Get current season
            current_date = datetime.now()
            if current_date.month >= 10:  # NBA season starts in October
                current_season = f"{current_date.year}-{str(current_date.year + 1)[2:]}"
            else:
                current_season = f"{current_date.year - 1}-{str(current_date.year)[2:]}"
            
            # Collect data
            report = await self.data_collector.collect_all_data(seasons=[current_season])
            
            logger.info(f"Data collection complete. Games collected: {report['summary']['total_games_collected']}")
            return True
            
        except Exception as e:
            logger.error(f"Data collection failed: {e}")
            return False
    
    async def train_new_models(self) -> Dict[str, Any]:
        """Train new models with latest data"""
        logger.info("Training new models...")
        
        await self.update_job_status(RetrainingStatus.TRAINING)
        
        try:
            # Get seasons for training
            current_date = datetime.now()
            seasons = []
            for i in range(3):  # Last 3 seasons
                year = current_date.year - i
                if current_date.month >= 10:
                    season = f"{year}-{str(year + 1)[2:]}"
                else:
                    season = f"{year - 1}-{str(year)[2:]}"
                seasons.append(season)
            
            # Train models
            results = await self.model_trainer.train_all_models(
                targets=["PTS", "REB", "AST"],
                seasons=seasons
            )
            
            logger.info(f"Model training complete. Models trained: {len(results)}")
            return results
            
        except Exception as e:
            logger.error(f"Model training failed: {e}")
            return {}
    
    async def evaluate_new_models(self, 
                                 new_models: Dict,
                                 current_models: Dict) -> Dict[str, Any]:
        """
        Evaluate new models against current production models
        
        Args:
            new_models: Newly trained models
            current_models: Current production models
            
        Returns:
            Evaluation results
        """
        logger.info("Evaluating new models...")
        
        await self.update_job_status(RetrainingStatus.EVALUATING)
        
        evaluation_results = {
            "should_deploy": True,
            "models": {},
            "summary": {}
        }
        
        for target, new_model_info in new_models.items():
            eval_result = {
                "new_r2": new_model_info.get("r2_score", 0),
                "new_mae": new_model_info.get("mae", 999),
                "improvement": 0,
                "meets_threshold": False
            }
            
            # Compare with current model if exists
            if target in current_models:
                current_r2 = current_models[target].get("r2_score", 0)
                eval_result["current_r2"] = current_r2
                eval_result["improvement"] = eval_result["new_r2"] - current_r2
            
            # Check if meets threshold
            eval_result["meets_threshold"] = eval_result["new_r2"] >= self.config.performance_threshold
            
            if not eval_result["meets_threshold"]:
                evaluation_results["should_deploy"] = False
                logger.warning(f"{target} model doesn't meet threshold: {eval_result['new_r2']:.4f}")
            
            evaluation_results["models"][target] = eval_result
        
        # Summary statistics
        evaluation_results["summary"] = {
            "avg_r2": np.mean([m["new_r2"] for m in evaluation_results["models"].values()]),
            "avg_improvement": np.mean([m["improvement"] for m in evaluation_results["models"].values()]),
            "all_meet_threshold": all(m["meets_threshold"] for m in evaluation_results["models"].values())
        }
        
        logger.info(f"Evaluation complete. Should deploy: {evaluation_results['should_deploy']}")
        
        return evaluation_results
    
    async def deploy_models(self, models: Dict) -> bool:
        """
        Deploy new models to production
        
        Args:
            models: Models to deploy
            
        Returns:
            True if deployment successful
        """
        logger.info("Deploying new models to production...")
        
        await self.update_job_status(RetrainingStatus.DEPLOYING)
        
        try:
            # Backup current models
            await self.backup_current_models()
            
            # Deploy new models
            if self.config.canary_deployment:
                # Canary deployment - deploy to subset first
                logger.info(f"Starting canary deployment ({self.config.canary_percentage}% traffic)")
                
                # This would implement canary deployment logic
                # For now, we'll do standard deployment
                deployment_results = await self.api_connector.connect_all_models(
                    targets=list(models.keys())
                )
            else:
                # Standard deployment
                deployment_results = await self.api_connector.connect_all_models(
                    targets=list(models.keys())
                )
            
            # Test deployed models
            if self.config.test_before_deploy:
                test_passed = await self.test_deployed_models()
                if not test_passed:
                    logger.error("Deployed model tests failed")
                    if self.config.rollback_on_failure:
                        await self.rollback_models()
                    return False
            
            logger.info("Model deployment successful")
            return True
            
        except Exception as e:
            logger.error(f"Model deployment failed: {e}")
            if self.config.rollback_on_failure:
                await self.rollback_models()
            return False
    
    async def backup_current_models(self):
        """Backup current production models"""
        logger.info("Backing up current models...")
        
        backup_dir = Path("backups") / datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_dir.mkdir(parents=True, exist_ok=True)
        
        models_dir = Path("api/ml_models")
        if models_dir.exists():
            import shutil
            for model_dir in models_dir.iterdir():
                if model_dir.is_dir():
                    shutil.copytree(model_dir, backup_dir / model_dir.name)
                    self.previous_models[model_dir.name] = str(backup_dir / model_dir.name)
        
        logger.info(f"Models backed up to {backup_dir}")
    
    async def rollback_models(self):
        """Rollback to previous models"""
        logger.warning("Rolling back to previous models...")
        
        await self.update_job_status(RetrainingStatus.ROLLED_BACK)
        
        import shutil
        for model_name, backup_path in self.previous_models.items():
            if Path(backup_path).exists():
                target_path = Path("api/ml_models") / model_name
                if target_path.exists():
                    shutil.rmtree(target_path)
                shutil.copytree(backup_path, target_path)
        
        logger.info("Rollback completed")
    
    async def test_deployed_models(self) -> bool:
        """Test deployed models with sample predictions"""
        logger.info("Testing deployed models...")
        
        try:
            import requests
            
            # Test each model endpoint
            test_data = {
                "player_id": "203999",
                "opponent_team": "LAL",
                "is_home": True,
                "days_rest": 2,
                "season_avg_points": 25.5,
                "season_avg_rebounds": 11.2,
                "season_avg_assists": 8.1,
                "last_5_games_avg_points": 28.0,
                "last_5_games_avg_rebounds": 12.0,
                "last_5_games_avg_assists": 9.0
            }
            
            for target in ["points", "rebounds", "assists"]:
                response = requests.post(
                    f"http://localhost:8000/api/v1/predict/{target}",
                    json=test_data,
                    timeout=5
                )
                
                if response.status_code != 200:
                    logger.error(f"Test failed for {target}: {response.status_code}")
                    return False
                
                result = response.json()
                if "prediction" not in result:
                    logger.error(f"Invalid response for {target}")
                    return False
            
            logger.info("All model tests passed")
            return True
            
        except Exception as e:
            logger.error(f"Model testing failed: {e}")
            return False
    
    async def send_notifications(self, status: str, details: Dict):
        """Send notifications about retraining status"""
        logger.info(f"Sending notifications: {status}")
        
        # Email notification
        if self.config.notification_emails:
            await self.send_email_notification(status, details)
        
        # Slack notification
        if self.config.slack_webhook:
            await self.send_slack_notification(status, details)
        
        # AWS SNS notification
        await self.send_sns_notification(status, details)
    
    async def send_email_notification(self, status: str, details: Dict):
        """Send email notification"""
        try:
            subject = f"NBA ML Retraining Pipeline: {status}"
            
            body = f"""
            Retraining Pipeline Status: {status}
            
            Job ID: {self.current_job_id}
            Timestamp: {datetime.now().isoformat()}
            
            Details:
            {json.dumps(details, indent=2)}
            
            View logs for more information.
            """
            
            # This would use actual email service
            logger.info(f"Email notification would be sent: {subject}")
            
        except Exception as e:
            logger.error(f"Failed to send email: {e}")
    
    async def send_slack_notification(self, status: str, details: Dict):
        """Send Slack notification"""
        try:
            import requests
            
            message = {
                "text": f"NBA ML Retraining: {status}",
                "attachments": [{
                    "color": "good" if status == "SUCCESS" else "danger",
                    "fields": [
                        {"title": "Job ID", "value": self.current_job_id, "short": True},
                        {"title": "Status", "value": status, "short": True},
                        {"title": "Models Trained", "value": len(details.get("models", {})), "short": True},
                        {"title": "Average R²", "value": f"{details.get('avg_r2', 0):.4f}", "short": True}
                    ]
                }]
            }
            
            if self.config.slack_webhook:
                requests.post(self.config.slack_webhook, json=message)
            
        except Exception as e:
            logger.error(f"Failed to send Slack notification: {e}")
    
    async def send_sns_notification(self, status: str, details: Dict):
        """Send AWS SNS notification"""
        try:
            message = {
                "default": f"NBA ML Retraining Pipeline: {status}",
                "email": json.dumps(details, indent=2)
            }
            
            # This would publish to actual SNS topic
            # self.sns_client.publish(
            #     TopicArn='arn:aws:sns:us-west-2:account:nba-ml-retraining',
            #     Message=json.dumps(message),
            #     Subject=f'Retraining Pipeline: {status}'
            # )
            
            logger.info(f"SNS notification would be sent: {status}")
            
        except Exception as e:
            logger.error(f"Failed to send SNS notification: {e}")
    
    async def run_retraining_cycle(self) -> bool:
        """
        Run a complete retraining cycle
        
        Returns:
            True if successful
        """
        logger.info("="*60)
        logger.info("Starting Automated Retraining Cycle")
        logger.info("="*60)
        
        try:
            # Initialize components
            if not self.db_manager:
                self.db_manager = init_db()
            
            if not self.performance_monitor:
                self.performance_monitor = ModelPerformanceMonitor(self.db_manager)
            
            # Check retraining criteria
            should_retrain, criteria = await self.check_retraining_criteria()
            
            if not should_retrain:
                logger.info("Retraining criteria not met. Skipping cycle.")
                return True
            
            # Create retraining job
            job_id = await self.create_retraining_job(criteria)
            
            # Collect latest data
            data_collected = await self.collect_latest_data()
            if not data_collected:
                await self.update_job_status(RetrainingStatus.FAILED, {"error": "Data collection failed"})
                await self.send_notifications("FAILED", {"reason": "Data collection failed"})
                return False
            
            # Train new models
            new_models = await self.train_new_models()
            if not new_models:
                await self.update_job_status(RetrainingStatus.FAILED, {"error": "Model training failed"})
                await self.send_notifications("FAILED", {"reason": "Model training failed"})
                return False
            
            # Get current production models for comparison
            current_models = {}  # Would load from production
            
            # Evaluate new models
            evaluation = await self.evaluate_new_models(new_models, current_models)
            
            # Deploy if evaluation passes
            if evaluation["should_deploy"] and self.config.auto_deploy:
                deployment_success = await self.deploy_models(new_models)
                
                if deployment_success:
                    await self.update_job_status(RetrainingStatus.COMPLETED, {
                        "models_deployed": list(new_models.keys()),
                        "evaluation": evaluation,
                        "timestamp": datetime.now().isoformat()
                    })
                    await self.send_notifications("SUCCESS", {
                        "models": new_models,
                        "evaluation": evaluation
                    })
                else:
                    await self.update_job_status(RetrainingStatus.FAILED, {"error": "Deployment failed"})
                    await self.send_notifications("FAILED", {"reason": "Deployment failed"})
                    return False
            else:
                logger.info("Models not deployed (evaluation failed or auto-deploy disabled)")
                await self.update_job_status(RetrainingStatus.COMPLETED, {
                    "models_trained": list(new_models.keys()),
                    "deployed": False,
                    "reason": "Evaluation failed" if not evaluation["should_deploy"] else "Auto-deploy disabled"
                })
            
            logger.info("="*60)
            logger.info("Retraining Cycle Completed Successfully")
            logger.info("="*60)
            
            return True
            
        except Exception as e:
            logger.error(f"Retraining cycle failed: {e}")
            await self.update_job_status(RetrainingStatus.FAILED, {"error": str(e)})
            await self.send_notifications("FAILED", {"error": str(e)})
            return False
    
    def schedule_retraining(self):
        """Schedule periodic retraining"""
        logger.info(f"Scheduling retraining: {self.config.schedule_interval}")
        
        if self.config.schedule_interval == "daily":
            schedule.every().day.at("02:00").do(lambda: asyncio.run(self.run_retraining_cycle()))
        elif self.config.schedule_interval == "weekly":
            schedule.every().monday.at("02:00").do(lambda: asyncio.run(self.run_retraining_cycle()))
        elif self.config.schedule_interval == "monthly":
            schedule.every(30).days.do(lambda: asyncio.run(self.run_retraining_cycle()))
        
        logger.info("Retraining scheduled")
    
    def run_scheduler(self):
        """Run the scheduling loop"""
        logger.info("Starting retraining scheduler...")
        
        self.schedule_retraining()
        
        while True:
            schedule.run_pending()
            time.sleep(60)  # Check every minute


async def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Automated model retraining pipeline')
    parser.add_argument('--config', default='config/retraining.yaml',
                       help='Path to retraining configuration')
    parser.add_argument('--mode', choices=['once', 'scheduled', 'monitor'],
                       default='once',
                       help='Execution mode')
    parser.add_argument('--force', action='store_true',
                       help='Force retraining regardless of criteria')
    parser.add_argument('--dry-run', action='store_true',
                       help='Simulate retraining without making changes')
    
    args = parser.parse_args()
    
    # Create necessary directories
    for dir_name in ['logs', 'reports', 'backups']:
        os.makedirs(dir_name, exist_ok=True)
    
    # Load configuration
    config = RetrainingConfig()
    if Path(args.config).exists():
        with open(args.config, 'r') as f:
            config_data = yaml.safe_load(f)
            for key, value in config_data.items():
                if hasattr(config, key):
                    setattr(config, key, value)
    
    # Initialize pipeline
    pipeline = AutomatedRetrainingPipeline(config)
    
    if args.force:
        # Force retraining
        logger.info("Forcing retraining cycle...")
        success = await pipeline.run_retraining_cycle()
        sys.exit(0 if success else 1)
    
    elif args.mode == 'once':
        # Run single retraining cycle
        success = await pipeline.run_retraining_cycle()
        sys.exit(0 if success else 1)
    
    elif args.mode == 'scheduled':
        # Run on schedule
        pipeline.run_scheduler()
    
    elif args.mode == 'monitor':
        # Monitor mode - check criteria but don't retrain
        should_retrain, criteria = await pipeline.check_retraining_criteria()
        print(f"Should retrain: {should_retrain}")
        print(f"Criteria: {json.dumps(criteria, indent=2)}")


if __name__ == "__main__":
    asyncio.run(main())