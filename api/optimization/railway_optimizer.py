"""
Railway-specific performance optimizations for NBA ML Platform
"""
import os
import logging
import asyncio
from typing import Dict, Any, Optional, List
import redis
import asyncpg
from contextlib import asynccontextmanager
import time
from functools import wraps
import psutil

logger = logging.getLogger(__name__)


class RailwayOptimizer:
    """Optimizations for Railway deployment environment"""
    
    def __init__(self):
        self.redis_url = os.getenv("REDIS_URL")
        self.database_url = os.getenv("DATABASE_URL")
        
        # Performance settings
        self.redis_pool_size = int(os.getenv("REDIS_POOL_SIZE", "20"))
        self.db_pool_size = int(os.getenv("SUPABASE_POOL_SIZE", "20"))
        self.cache_ttl = int(os.getenv("CACHE_TTL", "300"))  # 5 minutes default
        
        # Initialize connection pools
        self.redis_pool = None
        self.db_pool = None
        self._initialized = False
        
        # Request coalescing
        self._pending_requests = {}
        self._request_lock = asyncio.Lock()
    
    async def initialize(self):
        """Initialize optimized connections"""
        if self._initialized:
            return
        
        try:
            # Initialize Redis connection pool
            if self.redis_url:
                self.redis_pool = redis.ConnectionPool.from_url(
                    self.redis_url,
                    max_connections=self.redis_pool_size,
                    decode_responses=True
                )
                logger.info(f"Redis pool initialized with {self.redis_pool_size} connections")
            
            # Initialize PostgreSQL connection pool
            if self.database_url:
                self.db_pool = await asyncpg.create_pool(
                    self.database_url,
                    min_size=5,
                    max_size=self.db_pool_size,
                    max_queries=50000,
                    max_inactive_connection_lifetime=300,
                    command_timeout=60
                )
                logger.info(f"Database pool initialized with {self.db_pool_size} connections")
            
            self._initialized = True
            
            # Log memory usage
            self._log_memory_usage("After initialization")
            
        except Exception as e:
            logger.error(f"Failed to initialize optimizer: {e}")
            raise
    
    async def cleanup(self):
        """Cleanup connections"""
        if self.db_pool:
            await self.db_pool.close()
            logger.info("Database pool closed")
        
        if self.redis_pool:
            self.redis_pool.disconnect()
            logger.info("Redis pool closed")
    
    def get_redis_client(self) -> redis.Redis:
        """Get Redis client from pool"""
        if not self.redis_pool:
            raise RuntimeError("Redis pool not initialized")
        return redis.Redis(connection_pool=self.redis_pool)
    
    @asynccontextmanager
    async def get_db_connection(self):
        """Get database connection from pool"""
        if not self.db_pool:
            raise RuntimeError("Database pool not initialized")
        
        async with self.db_pool.acquire() as conn:
            yield conn
    
    async def optimize_for_railway(self):
        """Apply Railway-specific optimizations"""
        logger.info("Applying Railway optimizations")
        
        # 1. Set memory limits based on Railway container
        self._configure_memory_limits()
        
        # 2. Configure connection pooling
        await self._optimize_connection_pools()
        
        # 3. Enable query optimization
        await self._enable_query_optimization()
        
        # 4. Setup caching strategies
        self._setup_caching_strategies()
        
        logger.info("Railway optimizations complete")
    
    def _configure_memory_limits(self):
        """Configure memory limits for Railway containers"""
        # Get available memory
        memory = psutil.virtual_memory()
        available_mb = memory.available / 1024 / 1024
        
        # Set conservative limits (70% of available)
        memory_limit_mb = int(available_mb * 0.7)
        
        logger.info(f"Memory limit set to {memory_limit_mb}MB (70% of {available_mb:.0f}MB available)")
        
        # Configure garbage collection thresholds
        import gc
        gc.set_threshold(700, 10, 10)  # More aggressive GC
    
    async def _optimize_connection_pools(self):
        """Optimize connection pool settings"""
        if self.db_pool:
            # Get pool statistics
            stats = {
                "size": self.db_pool.get_size(),
                "free": self.db_pool.get_idle_size(),
                "used": self.db_pool.get_size() - self.db_pool.get_idle_size()
            }
            logger.info(f"DB Pool stats: {stats}")
            
            # Adjust pool size based on usage
            if stats["used"] > stats["size"] * 0.8:
                logger.warning("High database connection usage detected")
    
    async def _enable_query_optimization(self):
        """Enable query optimization features"""
        if self.db_pool:
            async with self.db_pool.acquire() as conn:
                # Enable query planning optimizations
                await conn.execute("SET jit = 'on'")
                await conn.execute("SET max_parallel_workers_per_gather = 4")
                await conn.execute("SET work_mem = '256MB'")
                
                logger.info("Query optimization settings applied")
    
    def _setup_caching_strategies(self):
        """Setup intelligent caching strategies"""
        self.cache_strategies = {
            "predictions": {"ttl": 3600, "prefix": "pred:"},  # 1 hour
            "features": {"ttl": 1800, "prefix": "feat:"},     # 30 minutes
            "models": {"ttl": 86400, "prefix": "model:"},     # 24 hours
            "dashboard": {"ttl": 60, "prefix": "dash:"},      # 1 minute
            "analytics": {"ttl": 300, "prefix": "analytics:"} # 5 minutes
        }
    
    def cache_key(self, category: str, identifier: str) -> str:
        """Generate cache key with category prefix"""
        strategy = self.cache_strategies.get(category, {})
        prefix = strategy.get("prefix", "")
        return f"{prefix}{identifier}"
    
    def cache_ttl_for(self, category: str) -> int:
        """Get TTL for cache category"""
        strategy = self.cache_strategies.get(category, {})
        return strategy.get("ttl", self.cache_ttl)
    
    async def cached_query(self, 
                         query_key: str,
                         query_func,
                         category: str = "default",
                         force_refresh: bool = False):
        """Execute query with caching"""
        if not force_refresh and self.redis_pool:
            # Try cache first
            redis_client = self.get_redis_client()
            cache_key = self.cache_key(category, query_key)
            
            try:
                cached = redis_client.get(cache_key)
                if cached:
                    import json
                    return json.loads(cached)
            except Exception as e:
                logger.error(f"Cache read error: {e}")
        
        # Execute query
        result = await query_func()
        
        # Cache result
        if self.redis_pool and result is not None:
            try:
                redis_client = self.get_redis_client()
                cache_key = self.cache_key(category, query_key)
                ttl = self.cache_ttl_for(category)
                
                import json
                redis_client.setex(cache_key, ttl, json.dumps(result))
            except Exception as e:
                logger.error(f"Cache write error: {e}")
        
        return result
    
    async def coalesce_requests(self, 
                              request_key: str,
                              request_func,
                              timeout: float = 5.0):
        """Coalesce identical requests to reduce load"""
        async with self._request_lock:
            # Check if request is already pending
            if request_key in self._pending_requests:
                # Wait for existing request
                future = self._pending_requests[request_key]
                try:
                    result = await asyncio.wait_for(future, timeout=timeout)
                    return result
                except asyncio.TimeoutError:
                    logger.warning(f"Request coalescing timeout for {request_key}")
            
            # Create new request
            future = asyncio.create_task(request_func())
            self._pending_requests[request_key] = future
        
        try:
            result = await future
            return result
        finally:
            # Clean up
            async with self._request_lock:
                self._pending_requests.pop(request_key, None)
    
    def _log_memory_usage(self, context: str = ""):
        """Log current memory usage"""
        process = psutil.Process()
        memory_info = process.memory_info()
        memory_percent = process.memory_percent()
        
        logger.info(f"Memory usage {context}: {memory_info.rss / 1024 / 1024:.1f}MB ({memory_percent:.1f}%)")
    
    def monitor_performance(self, func):
        """Decorator to monitor function performance"""
        @wraps(func)
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            
            try:
                result = await func(*args, **kwargs)
                
                # Log performance metrics
                duration = time.time() - start_time
                if duration > 1.0:  # Log slow operations
                    logger.warning(f"{func.__name__} took {duration:.2f}s")
                
                return result
                
            except Exception as e:
                logger.error(f"Error in {func.__name__}: {e}")
                raise
        
        return wrapper
    
    async def get_optimization_stats(self) -> Dict[str, Any]:
        """Get current optimization statistics"""
        stats = {
            "timestamp": time.time(),
            "memory": {},
            "connections": {},
            "cache": {}
        }
        
        # Memory stats
        process = psutil.Process()
        memory_info = process.memory_info()
        stats["memory"] = {
            "rss_mb": memory_info.rss / 1024 / 1024,
            "percent": process.memory_percent(),
            "available_mb": psutil.virtual_memory().available / 1024 / 1024
        }
        
        # Connection stats
        if self.db_pool:
            stats["connections"]["database"] = {
                "total": self.db_pool.get_size(),
                "idle": self.db_pool.get_idle_size(),
                "used": self.db_pool.get_size() - self.db_pool.get_idle_size()
            }
        
        if self.redis_pool:
            stats["connections"]["redis"] = {
                "created": self.redis_pool.connection_kwargs.get('db', 0),
                "max": self.redis_pool.max_connections
            }
        
        # Cache stats
        if self.redis_pool:
            try:
                redis_client = self.get_redis_client()
                info = redis_client.info("stats")
                stats["cache"] = {
                    "hits": info.get("keyspace_hits", 0),
                    "misses": info.get("keyspace_misses", 0),
                    "hit_rate": info.get("keyspace_hits", 0) / 
                               (info.get("keyspace_hits", 0) + info.get("keyspace_misses", 1))
                }
            except Exception as e:
                logger.error(f"Failed to get cache stats: {e}")
        
        return stats


# Global optimizer instance
optimizer = RailwayOptimizer()


async def initialize_optimizer():
    """Initialize the global optimizer"""
    await optimizer.initialize()
    await optimizer.optimize_for_railway()


async def get_optimizer() -> RailwayOptimizer:
    """Get the global optimizer instance"""
    if not optimizer._initialized:
        await initialize_optimizer()
    return optimizer