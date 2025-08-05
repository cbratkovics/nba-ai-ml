"""
Production model registry with versioning and management
"""
import os
import json
import hashlib
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
import pickle
import joblib
import boto3
from botocore.exceptions import NoCredentialsError
import numpy as np
from sqlalchemy import create_engine, text
from dataclasses import dataclass, asdict
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class ModelMetadata:
    """Model metadata structure"""
    model_id: str
    model_name: str
    version: str
    target: str  # points, rebounds, assists
    algorithm: str
    created_at: str
    training_data: Dict[str, Any]
    hyperparameters: Dict[str, Any]
    metrics: Dict[str, float]
    feature_names: List[str]
    status: str  # active, inactive, testing
    deployment_info: Dict[str, Any]


class ModelRegistry:
    """
    Production model registry with versioning and cloud storage
    """
    
    def __init__(self, 
                 local_path: str = "./models",
                 s3_bucket: Optional[str] = None,
                 db_url: Optional[str] = None):
        """
        Initialize model registry
        
        Args:
            local_path: Local directory for model storage
            s3_bucket: S3 bucket for cloud storage (optional)
            db_url: Database URL for metadata storage
        """
        self.local_path = Path(local_path)
        self.local_path.mkdir(exist_ok=True)
        
        self.s3_bucket = s3_bucket
        self.s3_client = None
        if s3_bucket and os.getenv("AWS_ACCESS_KEY_ID"):
            try:
                self.s3_client = boto3.client('s3')
                logger.info(f"S3 client initialized for bucket: {s3_bucket}")
            except Exception as e:
                logger.warning(f"S3 initialization failed: {e}")
        
        self.db_url = db_url or os.getenv("DATABASE_URL")
        self.metadata_cache = {}
        
        # Create metadata directory
        self.metadata_path = self.local_path / "metadata"
        self.metadata_path.mkdir(exist_ok=True)
    
    def register_model(self,
                      model: Any,
                      model_name: str,
                      target: str,
                      algorithm: str,
                      training_data: Dict[str, Any],
                      hyperparameters: Dict[str, Any],
                      metrics: Dict[str, float],
                      feature_names: List[str]) -> str:
        """
        Register a new model with metadata
        
        Returns:
            model_id: Unique identifier for the model
        """
        # Generate model ID
        version = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_id = self._generate_model_id(model_name, target, version)
        
        # Create metadata
        metadata = ModelMetadata(
            model_id=model_id,
            model_name=model_name,
            version=version,
            target=target,
            algorithm=algorithm,
            created_at=datetime.now().isoformat(),
            training_data=training_data,
            hyperparameters=hyperparameters,
            metrics=metrics,
            feature_names=feature_names,
            status="testing",
            deployment_info={}
        )
        
        # Save model locally
        model_path = self.local_path / f"{model_id}.pkl"
        joblib.dump(model, model_path)
        logger.info(f"Model saved locally: {model_path}")
        
        # Save metadata
        self._save_metadata(metadata)
        
        # Upload to S3 if available
        if self.s3_client:
            self._upload_to_s3(model_path, model_id, metadata)
        
        # Store in database
        self._store_metadata_db(metadata)
        
        logger.info(f"Model registered: {model_id}")
        return model_id
    
    def get_model(self, model_id: str) -> Tuple[Any, ModelMetadata]:
        """
        Retrieve model and metadata by ID
        
        Returns:
            Tuple of (model, metadata)
        """
        # Check local cache first
        model_path = self.local_path / f"{model_id}.pkl"
        
        if not model_path.exists() and self.s3_client:
            # Try to download from S3
            self._download_from_s3(model_id)
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_id}")
        
        # Load model
        model = joblib.load(model_path)
        
        # Load metadata
        metadata = self._load_metadata(model_id)
        
        return model, metadata
    
    def get_active_model(self, target: str) -> Tuple[Any, ModelMetadata]:
        """
        Get the currently active model for a target
        
        Args:
            target: Prediction target (points, rebounds, assists)
            
        Returns:
            Tuple of (model, metadata)
        """
        # Query for active model
        active_models = self.list_models(target=target, status="active")
        
        if not active_models:
            # Fallback to latest model
            all_models = self.list_models(target=target)
            if not all_models:
                raise ValueError(f"No models found for target: {target}")
            
            # Sort by version (timestamp) and get latest
            latest = sorted(all_models, key=lambda x: x.version, reverse=True)[0]
            logger.warning(f"No active model for {target}, using latest: {latest.model_id}")
            return self.get_model(latest.model_id)
        
        # Get the most recent active model
        active_model = sorted(active_models, key=lambda x: x.version, reverse=True)[0]
        return self.get_model(active_model.model_id)
    
    def promote_model(self, model_id: str) -> bool:
        """
        Promote a model to active status
        
        Args:
            model_id: Model to promote
            
        Returns:
            Success status
        """
        try:
            # Load metadata
            metadata = self._load_metadata(model_id)
            
            # Deactivate current active model for this target
            active_models = self.list_models(target=metadata.target, status="active")
            for active_model in active_models:
                self._update_model_status(active_model.model_id, "inactive")
            
            # Promote new model
            metadata.status = "active"
            metadata.deployment_info = {
                "promoted_at": datetime.now().isoformat(),
                "promoted_by": "system"
            }
            
            self._save_metadata(metadata)
            self._store_metadata_db(metadata)
            
            logger.info(f"Model promoted to active: {model_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error promoting model {model_id}: {e}")
            return False
    
    def rollback_model(self, target: str) -> bool:
        """
        Rollback to previous active model
        
        Args:
            target: Prediction target
            
        Returns:
            Success status
        """
        try:
            # Get inactive models sorted by version
            inactive_models = self.list_models(target=target, status="inactive")
            if not inactive_models:
                logger.warning(f"No inactive models to rollback to for {target}")
                return False
            
            # Get the most recent inactive model
            previous_model = sorted(inactive_models, key=lambda x: x.version, reverse=True)[0]
            
            # Promote it
            return self.promote_model(previous_model.model_id)
            
        except Exception as e:
            logger.error(f"Error rolling back model for {target}: {e}")
            return False
    
    def compare_models(self, model_id1: str, model_id2: str) -> Dict[str, Any]:
        """
        Compare two models
        
        Returns:
            Comparison results
        """
        metadata1 = self._load_metadata(model_id1)
        metadata2 = self._load_metadata(model_id2)
        
        comparison = {
            "model_1": {
                "id": model_id1,
                "version": metadata1.version,
                "metrics": metadata1.metrics
            },
            "model_2": {
                "id": model_id2,
                "version": metadata2.version,
                "metrics": metadata2.metrics
            },
            "metric_differences": {}
        }
        
        # Calculate metric differences
        for metric in metadata1.metrics:
            if metric in metadata2.metrics:
                diff = metadata2.metrics[metric] - metadata1.metrics[metric]
                pct_change = (diff / metadata1.metrics[metric]) * 100 if metadata1.metrics[metric] != 0 else 0
                comparison["metric_differences"][metric] = {
                    "absolute": diff,
                    "percentage": pct_change
                }
        
        return comparison
    
    def list_models(self, 
                   target: Optional[str] = None,
                   status: Optional[str] = None,
                   limit: int = 50) -> List[ModelMetadata]:
        """
        List models with optional filtering
        
        Args:
            target: Filter by prediction target
            status: Filter by status
            limit: Maximum number of results
            
        Returns:
            List of model metadata
        """
        models = []
        
        # Load from metadata files
        for metadata_file in self.metadata_path.glob("*.json"):
            with open(metadata_file, 'r') as f:
                data = json.load(f)
                metadata = ModelMetadata(**data)
                
                # Apply filters
                if target and metadata.target != target:
                    continue
                if status and metadata.status != status:
                    continue
                
                models.append(metadata)
        
        # Sort by version (newest first)
        models.sort(key=lambda x: x.version, reverse=True)
        
        return models[:limit]
    
    def get_model_lineage(self, model_id: str) -> Dict[str, Any]:
        """
        Get complete lineage information for a model
        
        Returns:
            Lineage information including training data, features, etc.
        """
        metadata = self._load_metadata(model_id)
        
        lineage = {
            "model_id": model_id,
            "created_at": metadata.created_at,
            "algorithm": metadata.algorithm,
            "training_data": metadata.training_data,
            "feature_pipeline": {
                "features_used": metadata.feature_names,
                "feature_count": len(metadata.feature_names)
            },
            "hyperparameters": metadata.hyperparameters,
            "performance": metadata.metrics,
            "deployment_history": self._get_deployment_history(model_id)
        }
        
        return lineage
    
    def _generate_model_id(self, model_name: str, target: str, version: str) -> str:
        """Generate unique model ID"""
        content = f"{model_name}_{target}_{version}"
        return hashlib.md5(content.encode()).hexdigest()[:12]
    
    def _save_metadata(self, metadata: ModelMetadata):
        """Save metadata to file"""
        metadata_file = self.metadata_path / f"{metadata.model_id}.json"
        with open(metadata_file, 'w') as f:
            json.dump(asdict(metadata), f, indent=2)
        
        # Update cache
        self.metadata_cache[metadata.model_id] = metadata
    
    def _load_metadata(self, model_id: str) -> ModelMetadata:
        """Load metadata from file or cache"""
        # Check cache
        if model_id in self.metadata_cache:
            return self.metadata_cache[model_id]
        
        # Load from file
        metadata_file = self.metadata_path / f"{model_id}.json"
        if not metadata_file.exists():
            # Try to load from database
            metadata = self._load_metadata_db(model_id)
            if metadata:
                return metadata
            raise FileNotFoundError(f"Metadata not found for model: {model_id}")
        
        with open(metadata_file, 'r') as f:
            data = json.load(f)
            metadata = ModelMetadata(**data)
        
        # Cache it
        self.metadata_cache[model_id] = metadata
        return metadata
    
    def _update_model_status(self, model_id: str, status: str):
        """Update model status"""
        metadata = self._load_metadata(model_id)
        metadata.status = status
        self._save_metadata(metadata)
        self._store_metadata_db(metadata)
    
    def _upload_to_s3(self, model_path: Path, model_id: str, metadata: ModelMetadata):
        """Upload model to S3"""
        if not self.s3_client:
            return
        
        try:
            # Upload model
            s3_key = f"models/{model_id}.pkl"
            self.s3_client.upload_file(str(model_path), self.s3_bucket, s3_key)
            
            # Upload metadata
            metadata_key = f"models/metadata/{model_id}.json"
            self.s3_client.put_object(
                Bucket=self.s3_bucket,
                Key=metadata_key,
                Body=json.dumps(asdict(metadata))
            )
            
            logger.info(f"Model uploaded to S3: {s3_key}")
            
        except Exception as e:
            logger.error(f"S3 upload failed: {e}")
    
    def _download_from_s3(self, model_id: str):
        """Download model from S3"""
        if not self.s3_client:
            return
        
        try:
            # Download model
            s3_key = f"models/{model_id}.pkl"
            local_path = self.local_path / f"{model_id}.pkl"
            self.s3_client.download_file(self.s3_bucket, s3_key, str(local_path))
            
            # Download metadata
            metadata_key = f"models/metadata/{model_id}.json"
            metadata_path = self.metadata_path / f"{model_id}.json"
            self.s3_client.download_file(self.s3_bucket, metadata_key, str(metadata_path))
            
            logger.info(f"Model downloaded from S3: {model_id}")
            
        except Exception as e:
            logger.error(f"S3 download failed: {e}")
    
    def _store_metadata_db(self, metadata: ModelMetadata):
        """Store metadata in database"""
        if not self.db_url:
            return
        
        try:
            engine = create_engine(self.db_url)
            
            # Create table if not exists
            create_table_sql = """
            CREATE TABLE IF NOT EXISTS model_registry (
                model_id VARCHAR(50) PRIMARY KEY,
                model_name VARCHAR(100),
                version VARCHAR(50),
                target VARCHAR(50),
                algorithm VARCHAR(100),
                created_at TIMESTAMP,
                status VARCHAR(20),
                metrics JSONB,
                metadata JSONB
            )
            """
            
            with engine.connect() as conn:
                conn.execute(text(create_table_sql))
                conn.commit()
                
                # Upsert model metadata
                upsert_sql = """
                INSERT INTO model_registry 
                (model_id, model_name, version, target, algorithm, created_at, status, metrics, metadata)
                VALUES (:model_id, :model_name, :version, :target, :algorithm, :created_at, :status, :metrics, :metadata)
                ON CONFLICT (model_id) 
                DO UPDATE SET 
                    status = EXCLUDED.status,
                    metrics = EXCLUDED.metrics,
                    metadata = EXCLUDED.metadata
                """
                
                conn.execute(text(upsert_sql), {
                    "model_id": metadata.model_id,
                    "model_name": metadata.model_name,
                    "version": metadata.version,
                    "target": metadata.target,
                    "algorithm": metadata.algorithm,
                    "created_at": metadata.created_at,
                    "status": metadata.status,
                    "metrics": json.dumps(metadata.metrics),
                    "metadata": json.dumps(asdict(metadata))
                })
                conn.commit()
                
        except Exception as e:
            logger.error(f"Database storage failed: {e}")
    
    def _load_metadata_db(self, model_id: str) -> Optional[ModelMetadata]:
        """Load metadata from database"""
        if not self.db_url:
            return None
        
        try:
            engine = create_engine(self.db_url)
            
            with engine.connect() as conn:
                result = conn.execute(
                    text("SELECT metadata FROM model_registry WHERE model_id = :model_id"),
                    {"model_id": model_id}
                )
                row = result.fetchone()
                
                if row:
                    return ModelMetadata(**json.loads(row[0]))
                    
        except Exception as e:
            logger.error(f"Database load failed: {e}")
        
        return None
    
    def _get_deployment_history(self, model_id: str) -> List[Dict[str, str]]:
        """Get deployment history for a model"""
        # This would query from a deployment log table
        # For now, return from metadata
        metadata = self._load_metadata(model_id)
        
        history = []
        if metadata.deployment_info:
            history.append({
                "event": "promoted",
                "timestamp": metadata.deployment_info.get("promoted_at", ""),
                "details": metadata.deployment_info
            })
        
        return history