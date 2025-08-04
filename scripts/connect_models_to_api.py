#!/usr/bin/env python3
"""
Connect Trained Models to FastAPI Endpoints
Integrates production models with API serving layer
"""
import os
import sys
import asyncio
import argparse
import logging
from pathlib import Path
from datetime import datetime
import json
import joblib
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
import shutil

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from database.connection import DatabaseManager, init_db
from database.models import MLModel, ModelVersion, Player
from api.models import PredictionRequest
import mlflow
from sqlalchemy import and_, func

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'logs/api_integration_{datetime.now():%Y%m%d_%H%M%S}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class ModelAPIConnector:
    """Connects trained models to API endpoints"""
    
    def __init__(self):
        """Initialize API connector"""
        self.db_manager = None
        self.models_dir = Path("models")
        self.api_models_dir = Path("api/ml_models")
        self.api_models_dir.mkdir(parents=True, exist_ok=True)
        
        # Model registry
        self.registered_models = {}
        self.active_models = {}
        
        logger.info("Model API Connector initialized")
    
    async def scan_trained_models(self) -> Dict[str, List[Dict]]:
        """
        Scan for trained models in the models directory
        
        Returns:
            Dictionary mapping targets to list of model files
        """
        logger.info("Scanning for trained models...")
        
        models_found = {}
        
        for target_dir in self.models_dir.iterdir():
            if target_dir.is_dir():
                target = target_dir.name.upper()
                models_found[target] = []
                
                # Find model files
                model_files = list(target_dir.glob("model_*.pkl"))
                scaler_files = list(target_dir.glob("scaler_*.pkl"))
                
                for model_file in model_files:
                    # Extract timestamp from filename
                    timestamp_str = model_file.stem.replace("model_", "")
                    
                    # Find matching scaler
                    scaler_file = None
                    for sf in scaler_files:
                        if timestamp_str in sf.stem:
                            scaler_file = sf
                            break
                    
                    # Get model stats
                    model_info = {
                        "model_path": str(model_file),
                        "scaler_path": str(scaler_file) if scaler_file else None,
                        "timestamp": timestamp_str,
                        "size_mb": model_file.stat().st_size / (1024 * 1024),
                        "created": datetime.fromtimestamp(model_file.stat().st_ctime)
                    }
                    
                    # Try to load and get model type
                    try:
                        model = joblib.load(model_file)
                        model_info["model_type"] = type(model).__name__
                        
                        # Get feature count if available
                        if hasattr(model, 'n_features_in_'):
                            model_info["n_features"] = model.n_features_in_
                    except Exception as e:
                        logger.warning(f"Could not load model {model_file}: {e}")
                        model_info["model_type"] = "unknown"
                    
                    models_found[target].append(model_info)
        
        # Sort by creation date (newest first)
        for target in models_found:
            models_found[target].sort(key=lambda x: x["created"], reverse=True)
        
        logger.info(f"Found models for targets: {list(models_found.keys())}")
        for target, models in models_found.items():
            logger.info(f"  {target}: {len(models)} model(s)")
        
        return models_found
    
    async def get_model_performance(self, model_path: str) -> Optional[Dict]:
        """
        Get performance metrics for a model from MLflow
        
        Args:
            model_path: Path to model file
            
        Returns:
            Dictionary with performance metrics
        """
        try:
            # Search MLflow for runs with this model
            mlflow.set_tracking_uri("sqlite:///mlflow.db")
            
            # Get all experiments
            experiments = mlflow.search_experiments()
            
            for exp in experiments:
                runs = mlflow.search_runs(
                    experiment_ids=[exp.experiment_id],
                    filter_string=f"params.model_path = '{model_path}'"
                )
                
                if not runs.empty:
                    # Get the most recent run
                    run = runs.iloc[0]
                    
                    metrics = {
                        "r2_score": run.get("metrics.ensemble_r2", run.get("metrics.test_r2", 0)),
                        "mae": run.get("metrics.ensemble_mae", run.get("metrics.test_mae", 0)),
                        "rmse": run.get("metrics.ensemble_rmse", run.get("metrics.test_rmse", 0)),
                        "cv_mean_r2": run.get("metrics.cv_mean_r2", 0),
                        "cv_std_r2": run.get("metrics.cv_std_r2", 0)
                    }
                    
                    return metrics
        except Exception as e:
            logger.warning(f"Could not get MLflow metrics: {e}")
        
        return None
    
    async def register_model_in_db(self, 
                                  target: str,
                                  model_info: Dict,
                                  performance: Optional[Dict] = None) -> int:
        """
        Register a model in the database
        
        Args:
            target: Target variable
            model_info: Model information
            performance: Performance metrics
            
        Returns:
            Model version ID
        """
        if not self.db_manager:
            self.db_manager = init_db()
        
        with self.db_manager.transaction() as session:
            # Check if ML model exists
            ml_model = session.query(MLModel).filter_by(
                name=f"{target.lower()}_predictor"
            ).first()
            
            if not ml_model:
                ml_model = MLModel(
                    name=f"{target.lower()}_predictor",
                    model_type=model_info.get("model_type", "ensemble"),
                    target_variable=target,
                    description=f"Production {target} prediction model",
                    is_active=True
                )
                session.add(ml_model)
                session.flush()
            
            # Create model version
            version_number = session.query(func.count(ModelVersion.id)).filter_by(
                model_id=ml_model.id
            ).scalar() + 1
            
            model_version = ModelVersion(
                model_id=ml_model.id,
                version_number=f"v{version_number}.0",
                model_path=model_info["model_path"],
                scaler_path=model_info.get("scaler_path"),
                metrics=performance or {},
                is_active=False,  # Will activate later
                hyperparameters={
                    "n_features": model_info.get("n_features", 0),
                    "model_type": model_info.get("model_type", "unknown")
                }
            )
            session.add(model_version)
            session.flush()
            
            logger.info(f"Registered {target} model version {version_number} in database")
            
            return model_version.id
    
    async def deploy_model_to_api(self,
                                 target: str,
                                 model_info: Dict,
                                 version_id: int):
        """
        Deploy a model to the API serving directory
        
        Args:
            target: Target variable
            model_info: Model information
            version_id: Model version ID
        """
        logger.info(f"Deploying {target} model to API...")
        
        # Create target directory
        target_dir = self.api_models_dir / target.lower()
        target_dir.mkdir(exist_ok=True)
        
        # Copy model file
        src_model = Path(model_info["model_path"])
        dst_model = target_dir / "model.pkl"
        shutil.copy2(src_model, dst_model)
        
        # Copy scaler file if exists
        if model_info.get("scaler_path"):
            src_scaler = Path(model_info["scaler_path"])
            dst_scaler = target_dir / "scaler.pkl"
            shutil.copy2(src_scaler, dst_scaler)
        
        # Create metadata file
        metadata = {
            "target": target,
            "model_type": model_info.get("model_type", "unknown"),
            "version_id": version_id,
            "deployed_at": datetime.now().isoformat(),
            "model_path": str(dst_model),
            "scaler_path": str(dst_scaler) if model_info.get("scaler_path") else None,
            "n_features": model_info.get("n_features", 0)
        }
        
        metadata_path = target_dir / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Update active model registry
        self.active_models[target] = {
            "model_path": str(dst_model),
            "scaler_path": str(dst_scaler) if model_info.get("scaler_path") else None,
            "metadata": metadata
        }
        
        logger.info(f"✓ Deployed {target} model to {target_dir}")
    
    async def activate_model_version(self, version_id: int, target: str):
        """
        Activate a model version in the database
        
        Args:
            version_id: Model version ID
            target: Target variable
        """
        if not self.db_manager:
            self.db_manager = init_db()
        
        with self.db_manager.transaction() as session:
            # Deactivate all other versions for this target
            ml_model = session.query(MLModel).filter_by(
                target_variable=target
            ).first()
            
            if ml_model:
                session.query(ModelVersion).filter_by(
                    model_id=ml_model.id
                ).update({"is_active": False})
            
            # Activate this version
            session.query(ModelVersion).filter_by(
                id=version_id
            ).update({"is_active": True})
            
            logger.info(f"Activated model version {version_id} for {target}")
    
    async def test_model_endpoint(self, target: str) -> bool:
        """
        Test that a model can be loaded and used for predictions
        
        Args:
            target: Target variable
            
        Returns:
            True if test successful
        """
        logger.info(f"Testing {target} model endpoint...")
        
        try:
            target_dir = self.api_models_dir / target.lower()
            
            # Load model
            model_path = target_dir / "model.pkl"
            model = joblib.load(model_path)
            
            # Load scaler if exists
            scaler = None
            scaler_path = target_dir / "scaler.pkl"
            if scaler_path.exists():
                scaler = joblib.load(scaler_path)
            
            # Load metadata
            metadata_path = target_dir / "metadata.json"
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            # Create dummy input
            n_features = metadata.get("n_features", 100)
            dummy_input = np.random.randn(1, n_features)
            
            # Apply scaler if exists
            if scaler:
                dummy_input = scaler.transform(dummy_input)
            
            # Make prediction
            prediction = model.predict(dummy_input)
            
            logger.info(f"✓ {target} model test successful. Prediction shape: {prediction.shape}")
            return True
            
        except Exception as e:
            logger.error(f"✗ {target} model test failed: {e}")
            return False
    
    async def create_api_config(self):
        """Create configuration file for API to load models"""
        
        config = {
            "models": {},
            "features": {},
            "last_updated": datetime.now().isoformat()
        }
        
        for target, model_info in self.active_models.items():
            config["models"][target.lower()] = {
                "path": model_info["model_path"],
                "scaler_path": model_info.get("scaler_path"),
                "metadata": model_info["metadata"]
            }
            
            # Add feature configuration
            config["features"][target.lower()] = {
                "n_features": model_info["metadata"].get("n_features", 0),
                "rolling_windows": [3, 5, 10, 20],
                "include_advanced_stats": True
            }
        
        config_path = self.api_models_dir / "config.json"
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"Created API configuration at {config_path}")
    
    async def connect_all_models(self,
                                targets: Optional[List[str]] = None,
                                use_latest: bool = True) -> Dict[str, Any]:
        """
        Connect all trained models to API endpoints
        
        Args:
            targets: Specific targets to connect (None for all)
            use_latest: Use the latest model for each target
            
        Returns:
            Dictionary with connection results
        """
        logger.info("="*60)
        logger.info("CONNECTING MODELS TO API")
        logger.info("="*60)
        
        # Scan for models
        models_found = await self.scan_trained_models()
        
        if not models_found:
            logger.error("No trained models found")
            return {}
        
        results = {}
        successful = 0
        failed = 0
        
        # Filter targets if specified
        if targets:
            models_found = {k: v for k, v in models_found.items() if k in targets}
        
        for target, model_list in models_found.items():
            if not model_list:
                logger.warning(f"No models found for {target}")
                failed += 1
                continue
            
            try:
                # Use latest model
                model_info = model_list[0] if use_latest else model_list[-1]
                
                logger.info(f"\nProcessing {target} model:")
                logger.info(f"  Model: {model_info['model_path']}")
                logger.info(f"  Type: {model_info['model_type']}")
                logger.info(f"  Created: {model_info['created']}")
                
                # Get performance metrics
                performance = await self.get_model_performance(model_info["model_path"])
                if performance:
                    logger.info(f"  R² Score: {performance.get('r2_score', 0):.4f}")
                    logger.info(f"  MAE: {performance.get('mae', 0):.2f}")
                
                # Register in database
                version_id = await self.register_model_in_db(target, model_info, performance)
                
                # Deploy to API
                await self.deploy_model_to_api(target, model_info, version_id)
                
                # Activate version
                await self.activate_model_version(version_id, target)
                
                # Test endpoint
                test_success = await self.test_model_endpoint(target)
                
                results[target] = {
                    "status": "success" if test_success else "partial",
                    "model_type": model_info["model_type"],
                    "version_id": version_id,
                    "performance": performance,
                    "test_passed": test_success
                }
                
                if test_success:
                    successful += 1
                    logger.info(f"✓ {target} model connected successfully")
                else:
                    failed += 1
                    logger.warning(f"⚠ {target} model connected but test failed")
                    
            except Exception as e:
                logger.error(f"✗ Failed to connect {target} model: {e}")
                results[target] = {"status": "failed", "error": str(e)}
                failed += 1
        
        # Create API configuration
        if self.active_models:
            await self.create_api_config()
        
        # Generate report
        await self.generate_connection_report(results, successful, failed)
        
        logger.info("\n" + "="*60)
        logger.info("CONNECTION SUMMARY")
        logger.info("="*60)
        logger.info(f"Total Models: {len(models_found)}")
        logger.info(f"Successfully Connected: {successful}")
        logger.info(f"Failed: {failed}")
        
        if successful > 0:
            logger.info("\n✓ Models are now available via API endpoints!")
            logger.info("  Start API server: python -m api.main")
            logger.info("  Documentation: http://localhost:8000/docs")
        
        return results
    
    async def generate_connection_report(self, 
                                        results: Dict,
                                        successful: int,
                                        failed: int):
        """Generate API connection report"""
        
        report_dir = Path("reports")
        report_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # JSON report
        json_report = {
            "timestamp": timestamp,
            "successful_connections": successful,
            "failed_connections": failed,
            "models": results,
            "api_config": {
                "models_directory": str(self.api_models_dir),
                "active_models": list(self.active_models.keys())
            }
        }
        
        json_path = report_dir / f"api_connection_{timestamp}.json"
        with open(json_path, 'w') as f:
            json.dump(json_report, f, indent=2, default=str)
        
        # Text report
        text_path = report_dir / f"api_connection_{timestamp}.txt"
        with open(text_path, 'w') as f:
            f.write("="*60 + "\n")
            f.write("MODEL API CONNECTION REPORT\n")
            f.write("="*60 + "\n\n")
            f.write(f"Generated: {timestamp}\n")
            f.write(f"Successful Connections: {successful}\n")
            f.write(f"Failed Connections: {failed}\n\n")
            
            f.write("MODEL DETAILS\n")
            f.write("-"*60 + "\n\n")
            
            for target, result in results.items():
                f.write(f"{target} MODEL\n")
                f.write(f"  Status: {result['status']}\n")
                
                if result['status'] != 'failed':
                    f.write(f"  Model Type: {result.get('model_type', 'unknown')}\n")
                    f.write(f"  Version ID: {result.get('version_id', 'N/A')}\n")
                    f.write(f"  Test Passed: {'Yes' if result.get('test_passed') else 'No'}\n")
                    
                    if result.get('performance'):
                        f.write(f"  R² Score: {result['performance'].get('r2_score', 0):.4f}\n")
                        f.write(f"  MAE: {result['performance'].get('mae', 0):.2f}\n")
                else:
                    f.write(f"  Error: {result.get('error', 'Unknown')}\n")
                f.write("\n")
            
            f.write("="*60 + "\n")
            f.write("API CONFIGURATION\n")
            f.write("="*60 + "\n")
            f.write(f"Models Directory: {self.api_models_dir}\n")
            f.write(f"Active Models: {', '.join(self.active_models.keys())}\n")
            f.write(f"Config File: {self.api_models_dir / 'config.json'}\n")
        
        logger.info(f"\nReports saved to:")
        logger.info(f"  - {json_path}")
        logger.info(f"  - {text_path}")


async def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Connect trained models to API endpoints')
    parser.add_argument('--targets', nargs='+', default=None,
                       help='Specific targets to connect (default: all)')
    parser.add_argument('--use-latest', action='store_true', default=True,
                       help='Use the latest model for each target')
    parser.add_argument('--test-only', action='store_true',
                       help='Only test existing connections')
    
    args = parser.parse_args()
    
    # Create necessary directories
    for dir_name in ['logs', 'reports', 'api/ml_models']:
        os.makedirs(dir_name, exist_ok=True)
    
    # Initialize connector
    connector = ModelAPIConnector()
    
    if args.test_only:
        # Test existing connections
        logger.info("Testing existing model connections...")
        
        for target in ['PTS', 'REB', 'AST']:
            success = await connector.test_model_endpoint(target)
            if success:
                logger.info(f"✓ {target} model connection test passed")
            else:
                logger.error(f"✗ {target} model connection test failed")
        
        return
    
    # Connect models
    results = await connector.connect_all_models(
        targets=args.targets,
        use_latest=args.use_latest
    )
    
    if results:
        successful = sum(1 for r in results.values() if r['status'] != 'failed')
        logger.info(f"\n✓ Connected {successful} model(s) to API successfully!")
    else:
        logger.error("\n✗ No models connected to API")


if __name__ == "__main__":
    asyncio.run(main())