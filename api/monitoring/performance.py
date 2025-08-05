"""
Performance monitoring for Railway environment
"""
import time
import asyncio
import logging
import json
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
import psutil
import os
from functools import wraps
from api.optimization.railway_optimizer import get_optimizer

logger = logging.getLogger(__name__)

# Safe import with fallback
try:
    from api.db.connection_pool import get_db_pool
    db_pool = get_db_pool()
except ImportError:
    logger.warning("Database pool not available for metrics")
    db_pool = None


class PerformanceMonitor:
    """Track performance metrics in Railway environment"""
    
    def __init__(self):
        self.metrics_buffer = []
        self.buffer_size = 100
        self.flush_interval = 60  # seconds
        self._last_flush = time.time()
        
        # Performance thresholds
        self.thresholds = {
            "memory_percent": 80,
            "cpu_percent": 70,
            "response_time_ms": 1000,
            "db_connection_usage": 80,
            "cache_hit_rate": 50
        }
        
        # Metric aggregators
        self.response_times = []
        self.error_count = 0
        self.request_count = 0
    
    def track_request(self, endpoint: str, method: str, duration_ms: float, status_code: int):
        """Track API request metrics"""
        self.request_count += 1
        self.response_times.append(duration_ms)
        
        # Keep only recent response times (last 1000)
        if len(self.response_times) > 1000:
            self.response_times = self.response_times[-1000:]
        
        if status_code >= 400:
            self.error_count += 1
        
        metric = {
            "timestamp": datetime.now().isoformat(),
            "type": "request",
            "endpoint": endpoint,
            "method": method,
            "duration_ms": duration_ms,
            "status_code": status_code
        }
        
        self._add_metric(metric)
        
        # Log slow requests
        if duration_ms > self.thresholds["response_time_ms"]:
            logger.warning(f"Slow request: {method} {endpoint} took {duration_ms:.0f}ms")
    
    async def track_metrics(self) -> Dict[str, Any]:
        """Collect and track system metrics"""
        metrics = {
            "timestamp": datetime.now().isoformat(),
            "system": await self._get_system_metrics(),
            "application": await self._get_application_metrics(),
            "database": await self._get_database_metrics(),
            "cache": await self._get_cache_metrics()
        }
        
        # Check for threshold violations
        violations = self._check_thresholds(metrics)
        if violations:
            metrics["alerts"] = violations
            for violation in violations:
                logger.warning(f"Performance threshold exceeded: {violation}")
        
        # Log to Railway (appears in logs)
        logger.info(f"METRICS: {json.dumps(metrics)}")
        
        return metrics
    
    def get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage"""
        process = psutil.Process()
        memory_info = process.memory_info()
        
        return {
            "rss_mb": memory_info.rss / 1024 / 1024,
            "vms_mb": memory_info.vms / 1024 / 1024,
            "percent": process.memory_percent(),
            "available_mb": psutil.virtual_memory().available / 1024 / 1024
        }
    
    def get_cpu_usage(self) -> float:
        """Get current CPU usage"""
        return psutil.Process().cpu_percent(interval=1)
    
    async def get_redis_stats(self) -> Dict[str, Any]:
        """Get Redis connection statistics"""
        try:
            optimizer = await get_optimizer()
            if optimizer.redis_pool:
                redis_client = optimizer.get_redis_client()
                info = redis_client.info()
                
                stats = {
                    "connected_clients": info.get("connected_clients", 0),
                    "used_memory_mb": info.get("used_memory", 0) / 1024 / 1024,
                    "total_commands": info.get("total_commands_processed", 0),
                    "keyspace_hits": info.get("keyspace_hits", 0),
                    "keyspace_misses": info.get("keyspace_misses", 0)
                }
                
                # Calculate hit rate
                total_ops = stats["keyspace_hits"] + stats["keyspace_misses"]
                if total_ops > 0:
                    stats["hit_rate"] = (stats["keyspace_hits"] / total_ops) * 100
                else:
                    stats["hit_rate"] = 0
                
                return stats
        except Exception as e:
            logger.error(f"Error getting Redis stats: {e}")
            return {}
    
    async def get_pool_stats(self) -> Dict[str, Any]:
        """Get database connection pool statistics"""
        try:
            if db_pool:
                return await db_pool.health_check()
            else:
                return {}
        except Exception as e:
            logger.error(f"Error getting pool stats: {e}")
            return {}
    
    def get_latency_stats(self) -> Dict[str, float]:
        """Get request latency statistics"""
        if not self.response_times:
            return {
                "mean": 0,
                "median": 0,
                "p95": 0,
                "p99": 0,
                "max": 0
            }
        
        sorted_times = sorted(self.response_times)
        n = len(sorted_times)
        
        return {
            "mean": sum(sorted_times) / n,
            "median": sorted_times[n // 2],
            "p95": sorted_times[int(n * 0.95)],
            "p99": sorted_times[int(n * 0.99)],
            "max": sorted_times[-1]
        }
    
    async def _get_system_metrics(self) -> Dict[str, Any]:
        """Get system-level metrics"""
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        return {
            "cpu_percent": cpu_percent,
            "memory_percent": memory.percent,
            "memory_available_mb": memory.available / 1024 / 1024,
            "disk_percent": disk.percent,
            "load_average": os.getloadavg()[0] if hasattr(os, 'getloadavg') else 0
        }
    
    async def _get_application_metrics(self) -> Dict[str, Any]:
        """Get application-level metrics"""
        process = psutil.Process()
        
        return {
            "uptime_seconds": time.time() - process.create_time(),
            "thread_count": process.num_threads(),
            "open_files": len(process.open_files()),
            "request_count": self.request_count,
            "error_rate": (self.error_count / self.request_count * 100) if self.request_count > 0 else 0,
            "latency": self.get_latency_stats()
        }
    
    async def _get_database_metrics(self) -> Dict[str, Any]:
        """Get database-related metrics"""
        pool_stats = await self.get_pool_stats()
        
        metrics = {
            "pool": pool_stats,
            "connection_usage_percent": 0
        }
        
        if pool_stats.get("total_connections", 0) > 0:
            metrics["connection_usage_percent"] = (
                pool_stats.get("used_connections", 0) / 
                pool_stats.get("total_connections", 1) * 100
            )
        
        return metrics
    
    async def _get_cache_metrics(self) -> Dict[str, Any]:
        """Get cache-related metrics"""
        redis_stats = await self.get_redis_stats()
        return {
            "redis": redis_stats,
            "hit_rate": redis_stats.get("hit_rate", 0)
        }
    
    def _check_thresholds(self, metrics: Dict[str, Any]) -> List[str]:
        """Check if any metrics exceed thresholds"""
        violations = []
        
        # System metrics
        if metrics["system"]["memory_percent"] > self.thresholds["memory_percent"]:
            violations.append(f"High memory usage: {metrics['system']['memory_percent']:.1f}%")
        
        if metrics["system"]["cpu_percent"] > self.thresholds["cpu_percent"]:
            violations.append(f"High CPU usage: {metrics['system']['cpu_percent']:.1f}%")
        
        # Database metrics
        db_usage = metrics["database"]["connection_usage_percent"]
        if db_usage > self.thresholds["db_connection_usage"]:
            violations.append(f"High database connection usage: {db_usage:.1f}%")
        
        # Cache metrics
        hit_rate = metrics["cache"]["hit_rate"]
        if hit_rate < self.thresholds["cache_hit_rate"]:
            violations.append(f"Low cache hit rate: {hit_rate:.1f}%")
        
        # Latency metrics
        p95_latency = metrics["application"]["latency"]["p95"]
        if p95_latency > self.thresholds["response_time_ms"]:
            violations.append(f"High P95 latency: {p95_latency:.0f}ms")
        
        return violations
    
    def _add_metric(self, metric: dict):
        """Add metric to buffer"""
        self.metrics_buffer.append(metric)
        
        # Flush if buffer is full or interval exceeded
        if (len(self.metrics_buffer) >= self.buffer_size or 
            time.time() - self._last_flush > self.flush_interval):
            self._flush_metrics()
    
    def _flush_metrics(self):
        """Flush metrics buffer"""
        if not self.metrics_buffer:
            return
        
        # In production, you might send these to a metrics service
        # For now, just log summary
        logger.info(f"Flushing {len(self.metrics_buffer)} metrics")
        
        self.metrics_buffer.clear()
        self._last_flush = time.time()


# Global performance monitor instance
performance_monitor = PerformanceMonitor()


def track_performance(func):
    """Decorator to track function performance"""
    @wraps(func)
    async def async_wrapper(*args, **kwargs):
        start_time = time.time()
        
        try:
            result = await func(*args, **kwargs)
            duration_ms = (time.time() - start_time) * 1000
            
            # Track successful execution
            performance_monitor._add_metric({
                "type": "function",
                "name": func.__name__,
                "duration_ms": duration_ms,
                "status": "success",
                "timestamp": datetime.now().isoformat()
            })
            
            return result
            
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            
            # Track failed execution
            performance_monitor._add_metric({
                "type": "function",
                "name": func.__name__,
                "duration_ms": duration_ms,
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            })
            
            raise
    
    @wraps(func)
    def sync_wrapper(*args, **kwargs):
        start_time = time.time()
        
        try:
            result = func(*args, **kwargs)
            duration_ms = (time.time() - start_time) * 1000
            
            performance_monitor._add_metric({
                "type": "function",
                "name": func.__name__,
                "duration_ms": duration_ms,
                "status": "success",
                "timestamp": datetime.now().isoformat()
            })
            
            return result
            
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            
            performance_monitor._add_metric({
                "type": "function",
                "name": func.__name__,
                "duration_ms": duration_ms,
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            })
            
            raise
    
    if asyncio.iscoroutinefunction(func):
        return async_wrapper
    else:
        return sync_wrapper


# Background task to collect metrics periodically
async def collect_metrics_loop():
    """Background task to collect system metrics"""
    while True:
        try:
            await performance_monitor.track_metrics()
        except Exception as e:
            logger.error(f"Error collecting metrics: {e}")
        
        # Collect every 60 seconds
        await asyncio.sleep(60)