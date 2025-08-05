"""
A/B Testing framework for model experimentation
"""
import json
import logging
import random
import hashlib
import os
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, asdict
import numpy as np
from scipy import stats
from sqlalchemy import create_engine, text
import redis
from enum import Enum

logger = logging.getLogger(__name__)


class ExperimentStatus(Enum):
    """Experiment status enum"""
    DRAFT = "draft"
    RUNNING = "running"
    PAUSED = "paused"
    CONCLUDED = "concluded"
    ARCHIVED = "archived"


@dataclass
class ExperimentConfig:
    """Experiment configuration"""
    experiment_id: str
    name: str
    description: str
    variants: Dict[str, Dict[str, Any]]  # variant_id -> config
    traffic_allocation: Dict[str, float]  # variant_id -> percentage
    success_metrics: List[str]
    guardrail_metrics: List[str]
    minimum_sample_size: int
    created_at: str
    created_by: str
    status: ExperimentStatus


@dataclass
class ExperimentMetrics:
    """Metrics for an experiment variant"""
    variant_id: str
    sample_count: int
    metrics: Dict[str, Dict[str, float]]  # metric_name -> {mean, std, min, max}
    confidence_intervals: Dict[str, Tuple[float, float]]
    updated_at: str


class ExperimentManager:
    """
    A/B testing infrastructure for ML models
    """
    
    def __init__(self, 
                 db_url: Optional[str] = None,
                 redis_url: Optional[str] = None):
        """
        Initialize experiment manager
        
        Args:
            db_url: Database URL for experiment storage
            redis_url: Redis URL for assignment caching
        """
        self.db_url = db_url or os.getenv("DATABASE_URL")
        self.redis_url = redis_url or os.getenv("REDIS_URL")
        
        self.redis_client = None
        if self.redis_url:
            try:
                self.redis_client = redis.from_url(self.redis_url, decode_responses=True)
                self.redis_client.ping()
                logger.info("Experiment manager Redis initialized")
            except Exception as e:
                logger.error(f"Redis initialization failed: {e}")
        
        self.experiments: Dict[str, ExperimentConfig] = {}
        self._load_experiments()
    
    def create_experiment(self,
                         name: str,
                         description: str,
                         variants: Dict[str, Dict[str, Any]],
                         traffic_allocation: Dict[str, float],
                         success_metrics: List[str],
                         guardrail_metrics: Optional[List[str]] = None,
                         minimum_sample_size: int = 1000,
                         created_by: str = "system") -> str:
        """
        Create a new experiment
        
        Args:
            name: Experiment name
            description: Experiment description
            variants: Dictionary of variant configurations
            traffic_allocation: Traffic split between variants (must sum to 1.0)
            success_metrics: Metrics to optimize
            guardrail_metrics: Metrics that shouldn't degrade
            minimum_sample_size: Minimum samples per variant
            created_by: Creator identifier
            
        Returns:
            experiment_id
        """
        # Validate traffic allocation
        if abs(sum(traffic_allocation.values()) - 1.0) > 0.001:
            raise ValueError("Traffic allocation must sum to 1.0")
        
        # Generate experiment ID
        experiment_id = self._generate_experiment_id(name)
        
        # Create experiment config
        config = ExperimentConfig(
            experiment_id=experiment_id,
            name=name,
            description=description,
            variants=variants,
            traffic_allocation=traffic_allocation,
            success_metrics=success_metrics,
            guardrail_metrics=guardrail_metrics or [],
            minimum_sample_size=minimum_sample_size,
            created_at=datetime.now().isoformat(),
            created_by=created_by,
            status=ExperimentStatus.DRAFT
        )
        
        # Store experiment
        self.experiments[experiment_id] = config
        self._save_experiment(config)
        
        logger.info(f"Created experiment: {experiment_id}")
        return experiment_id
    
    def start_experiment(self, experiment_id: str) -> bool:
        """
        Start an experiment
        
        Args:
            experiment_id: Experiment to start
            
        Returns:
            Success status
        """
        if experiment_id not in self.experiments:
            raise ValueError(f"Experiment not found: {experiment_id}")
        
        experiment = self.experiments[experiment_id]
        
        if experiment.status != ExperimentStatus.DRAFT:
            raise ValueError(f"Cannot start experiment in status: {experiment.status}")
        
        experiment.status = ExperimentStatus.RUNNING
        self._save_experiment(experiment)
        
        logger.info(f"Started experiment: {experiment_id}")
        return True
    
    def get_variant_assignment(self, 
                             experiment_id: str, 
                             user_id: str,
                             attributes: Optional[Dict[str, Any]] = None) -> Optional[str]:
        """
        Get variant assignment for a user
        
        Args:
            experiment_id: Experiment ID
            user_id: User identifier
            attributes: User attributes for targeting
            
        Returns:
            Assigned variant ID or None if not in experiment
        """
        if experiment_id not in self.experiments:
            return None
        
        experiment = self.experiments[experiment_id]
        
        if experiment.status != ExperimentStatus.RUNNING:
            return None
        
        # Check cache first
        cached_assignment = self._get_cached_assignment(experiment_id, user_id)
        if cached_assignment:
            return cached_assignment
        
        # Generate assignment
        assignment = self._generate_assignment(experiment, user_id, attributes)
        
        # Cache assignment
        self._cache_assignment(experiment_id, user_id, assignment)
        
        return assignment
    
    def track_metric(self,
                    experiment_id: str,
                    variant_id: str,
                    metric_name: str,
                    value: float,
                    user_id: Optional[str] = None):
        """
        Track a metric value for an experiment
        
        Args:
            experiment_id: Experiment ID
            variant_id: Variant ID
            metric_name: Metric name
            value: Metric value
            user_id: Optional user ID for deduplication
        """
        if experiment_id not in self.experiments:
            return
        
        # Store metric in time series
        self._store_metric(experiment_id, variant_id, metric_name, value, user_id)
        
        # Update aggregated metrics
        self._update_aggregated_metrics(experiment_id, variant_id, metric_name, value)
    
    def get_experiment_results(self, experiment_id: str) -> Dict[str, Any]:
        """
        Get current results for an experiment
        
        Returns:
            Experiment results with statistical analysis
        """
        if experiment_id not in self.experiments:
            raise ValueError(f"Experiment not found: {experiment_id}")
        
        experiment = self.experiments[experiment_id]
        
        # Get metrics for each variant
        variant_metrics = {}
        for variant_id in experiment.variants:
            metrics = self._get_variant_metrics(experiment_id, variant_id)
            variant_metrics[variant_id] = metrics
        
        # Perform statistical analysis
        results = {
            "experiment_id": experiment_id,
            "name": experiment.name,
            "status": experiment.status.value,
            "variants": variant_metrics,
            "statistical_significance": {},
            "recommendations": []
        }
        
        # Calculate statistical significance for each success metric
        control_variant = self._get_control_variant(experiment)
        
        for metric in experiment.success_metrics:
            sig_results = self._calculate_significance(
                variant_metrics,
                control_variant,
                metric
            )
            results["statistical_significance"][metric] = sig_results
        
        # Generate recommendations
        results["recommendations"] = self._generate_recommendations(
            experiment,
            variant_metrics,
            results["statistical_significance"]
        )
        
        return results
    
    def conclude_experiment(self, 
                          experiment_id: str,
                          winning_variant: Optional[str] = None,
                          conclusion_notes: str = "") -> bool:
        """
        Conclude an experiment
        
        Args:
            experiment_id: Experiment to conclude
            winning_variant: Optional winning variant
            conclusion_notes: Notes about conclusion
            
        Returns:
            Success status
        """
        if experiment_id not in self.experiments:
            raise ValueError(f"Experiment not found: {experiment_id}")
        
        experiment = self.experiments[experiment_id]
        
        if experiment.status != ExperimentStatus.RUNNING:
            raise ValueError(f"Cannot conclude experiment in status: {experiment.status}")
        
        # Get final results
        results = self.get_experiment_results(experiment_id)
        
        # Store conclusion
        conclusion = {
            "concluded_at": datetime.now().isoformat(),
            "winning_variant": winning_variant,
            "conclusion_notes": conclusion_notes,
            "final_results": results
        }
        
        self._store_conclusion(experiment_id, conclusion)
        
        # Update status
        experiment.status = ExperimentStatus.CONCLUDED
        self._save_experiment(experiment)
        
        logger.info(f"Concluded experiment: {experiment_id}, winner: {winning_variant}")
        return True
    
    def get_active_experiments(self) -> List[ExperimentConfig]:
        """Get all active experiments"""
        return [
            exp for exp in self.experiments.values()
            if exp.status == ExperimentStatus.RUNNING
        ]
    
    def _generate_experiment_id(self, name: str) -> str:
        """Generate unique experiment ID"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        content = f"{name}_{timestamp}"
        return hashlib.md5(content.encode()).hexdigest()[:12]
    
    def _generate_assignment(self, 
                           experiment: ExperimentConfig,
                           user_id: str,
                           attributes: Optional[Dict[str, Any]]) -> str:
        """Generate variant assignment for user"""
        # Create deterministic hash for consistent assignment
        hash_input = f"{experiment.experiment_id}:{user_id}"
        hash_value = int(hashlib.md5(hash_input.encode()).hexdigest(), 16)
        
        # Map to variant based on traffic allocation
        random_value = (hash_value % 10000) / 10000.0
        
        cumulative = 0.0
        for variant_id, allocation in experiment.traffic_allocation.items():
            cumulative += allocation
            if random_value <= cumulative:
                return variant_id
        
        # Fallback to first variant
        return list(experiment.variants.keys())[0]
    
    def _get_control_variant(self, experiment: ExperimentConfig) -> str:
        """Get control variant (first variant by convention)"""
        return list(experiment.variants.keys())[0]
    
    def _calculate_significance(self,
                              variant_metrics: Dict[str, ExperimentMetrics],
                              control_variant: str,
                              metric_name: str) -> Dict[str, Any]:
        """Calculate statistical significance"""
        results = {}
        
        if control_variant not in variant_metrics:
            return results
        
        control_data = variant_metrics[control_variant]
        control_mean = control_data.metrics.get(metric_name, {}).get("mean", 0)
        control_std = control_data.metrics.get(metric_name, {}).get("std", 1)
        control_n = control_data.sample_count
        
        for variant_id, variant_data in variant_metrics.items():
            if variant_id == control_variant:
                continue
            
            variant_mean = variant_data.metrics.get(metric_name, {}).get("mean", 0)
            variant_std = variant_data.metrics.get(metric_name, {}).get("std", 1)
            variant_n = variant_data.sample_count
            
            # Perform t-test
            if control_n > 1 and variant_n > 1:
                # Calculate pooled standard deviation
                pooled_std = np.sqrt(
                    ((control_n - 1) * control_std**2 + (variant_n - 1) * variant_std**2) /
                    (control_n + variant_n - 2)
                )
                
                # Calculate t-statistic
                t_stat = (variant_mean - control_mean) / (pooled_std * np.sqrt(1/control_n + 1/variant_n))
                
                # Calculate p-value (two-tailed)
                df = control_n + variant_n - 2
                p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df))
                
                # Calculate relative improvement
                relative_improvement = ((variant_mean - control_mean) / control_mean) * 100 if control_mean != 0 else 0
                
                results[variant_id] = {
                    "control_mean": control_mean,
                    "variant_mean": variant_mean,
                    "relative_improvement": relative_improvement,
                    "p_value": p_value,
                    "significant": p_value < 0.05,
                    "confidence_level": 1 - p_value
                }
            else:
                results[variant_id] = {
                    "error": "Insufficient data for statistical test"
                }
        
        return results
    
    def _generate_recommendations(self,
                                experiment: ExperimentConfig,
                                variant_metrics: Dict[str, ExperimentMetrics],
                                significance_results: Dict[str, Dict[str, Any]]) -> List[str]:
        """Generate recommendations based on results"""
        recommendations = []
        
        # Check sample size
        for variant_id, metrics in variant_metrics.items():
            if metrics.sample_count < experiment.minimum_sample_size:
                recommendations.append(
                    f"Variant {variant_id} needs {experiment.minimum_sample_size - metrics.sample_count} "
                    f"more samples to reach minimum sample size"
                )
        
        # Check for significant winners
        for metric, results in significance_results.items():
            for variant_id, result in results.items():
                if result.get("significant") and result.get("relative_improvement", 0) > 0:
                    recommendations.append(
                        f"Variant {variant_id} shows significant improvement in {metric}: "
                        f"{result['relative_improvement']:.1f}% (p={result['p_value']:.3f})"
                    )
        
        # Check guardrail metrics
        # ... additional logic for guardrail metrics
        
        return recommendations
    
    def _get_cached_assignment(self, experiment_id: str, user_id: str) -> Optional[str]:
        """Get cached variant assignment"""
        if not self.redis_client:
            return None
        
        try:
            key = f"exp_assignment:{experiment_id}:{user_id}"
            return self.redis_client.get(key)
        except Exception as e:
            logger.debug(f"Cache get error: {e}")
            return None
    
    def _cache_assignment(self, experiment_id: str, user_id: str, variant_id: str):
        """Cache variant assignment"""
        if not self.redis_client:
            return
        
        try:
            key = f"exp_assignment:{experiment_id}:{user_id}"
            self.redis_client.setex(key, 86400 * 30, variant_id)  # 30 day TTL
        except Exception as e:
            logger.debug(f"Cache set error: {e}")
    
    def _store_metric(self, 
                     experiment_id: str,
                     variant_id: str,
                     metric_name: str,
                     value: float,
                     user_id: Optional[str]):
        """Store individual metric value"""
        # This would store in a time series database
        # For now, aggregate in Redis
        if not self.redis_client:
            return
        
        try:
            # Store in sorted set for percentile calculations
            key = f"exp_metrics:{experiment_id}:{variant_id}:{metric_name}"
            timestamp = datetime.now().timestamp()
            member = f"{timestamp}:{value}:{user_id or 'anonymous'}"
            self.redis_client.zadd(key, {member: timestamp})
            
            # Expire after 90 days
            self.redis_client.expire(key, 86400 * 90)
            
        except Exception as e:
            logger.error(f"Error storing metric: {e}")
    
    def _update_aggregated_metrics(self,
                                 experiment_id: str,
                                 variant_id: str,
                                 metric_name: str,
                                 value: float):
        """Update aggregated metrics"""
        # This would update running statistics
        # Implementation depends on storage backend
        pass
    
    def _get_variant_metrics(self, experiment_id: str, variant_id: str) -> ExperimentMetrics:
        """Get metrics for a variant"""
        # This would query from storage
        # For demo, return mock data
        return ExperimentMetrics(
            variant_id=variant_id,
            sample_count=random.randint(100, 10000),
            metrics={
                "mae": {
                    "mean": random.uniform(3.0, 5.0),
                    "std": random.uniform(0.5, 1.5),
                    "min": random.uniform(1.0, 2.0),
                    "max": random.uniform(8.0, 12.0)
                },
                "latency_ms": {
                    "mean": random.uniform(50, 150),
                    "std": random.uniform(10, 30),
                    "min": random.uniform(20, 40),
                    "max": random.uniform(200, 500)
                }
            },
            confidence_intervals={
                "mae": (3.5, 4.5),
                "latency_ms": (80, 120)
            },
            updated_at=datetime.now().isoformat()
        )
    
    def _save_experiment(self, experiment: ExperimentConfig):
        """Save experiment to storage"""
        # Store in database
        if self.db_url:
            try:
                engine = create_engine(self.db_url)
                
                # Create table if not exists
                create_table_sql = """
                CREATE TABLE IF NOT EXISTS experiments (
                    experiment_id VARCHAR(50) PRIMARY KEY,
                    name VARCHAR(200),
                    status VARCHAR(20),
                    config JSONB,
                    created_at TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT NOW()
                )
                """
                
                with engine.connect() as conn:
                    conn.execute(text(create_table_sql))
                    conn.commit()
                    
                    # Upsert experiment
                    upsert_sql = """
                    INSERT INTO experiments (experiment_id, name, status, config, created_at)
                    VALUES (:experiment_id, :name, :status, :config, :created_at)
                    ON CONFLICT (experiment_id)
                    DO UPDATE SET
                        status = EXCLUDED.status,
                        config = EXCLUDED.config,
                        updated_at = NOW()
                    """
                    
                    conn.execute(text(upsert_sql), {
                        "experiment_id": experiment.experiment_id,
                        "name": experiment.name,
                        "status": experiment.status.value,
                        "config": json.dumps(asdict(experiment)),
                        "created_at": experiment.created_at
                    })
                    conn.commit()
                    
            except Exception as e:
                logger.error(f"Error saving experiment: {e}")
    
    def _load_experiments(self):
        """Load experiments from storage"""
        if self.db_url:
            try:
                engine = create_engine(self.db_url)
                
                with engine.connect() as conn:
                    result = conn.execute(
                        text("SELECT config FROM experiments WHERE status != 'archived'")
                    )
                    
                    for row in result:
                        config_data = json.loads(row[0])
                        # Convert status string to enum
                        config_data['status'] = ExperimentStatus(config_data['status'])
                        experiment = ExperimentConfig(**config_data)
                        self.experiments[experiment.experiment_id] = experiment
                        
                logger.info(f"Loaded {len(self.experiments)} experiments")
                
            except Exception as e:
                logger.error(f"Error loading experiments: {e}")
    
    def _store_conclusion(self, experiment_id: str, conclusion: Dict[str, Any]):
        """Store experiment conclusion"""
        # This would store in a conclusions table
        # For now, just log
        logger.info(f"Experiment {experiment_id} concluded: {conclusion}")