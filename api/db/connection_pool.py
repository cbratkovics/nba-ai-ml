"""
Optimized connection pooling for Supabase PostgreSQL
"""
import os
import logging
from typing import Optional, AsyncGenerator
from contextlib import asynccontextmanager
import asyncpg
from asyncpg import Pool, Connection
import asyncio
from functools import wraps
import time

logger = logging.getLogger(__name__)


class SupabasePool:
    """Optimized connection pooling for Supabase"""
    
    _instance: Optional['SupabasePool'] = None
    _lock = asyncio.Lock()
    
    def __init__(self):
        self.pool: Optional[Pool] = None
        self.database_url = os.getenv("DATABASE_URL")
        
        # Pool configuration
        self.min_size = int(os.getenv("DB_POOL_MIN_SIZE", "5"))
        self.max_size = int(os.getenv("DB_POOL_MAX_SIZE", "20"))
        self.max_queries = int(os.getenv("DB_POOL_MAX_QUERIES", "50000"))
        self.max_inactive_lifetime = int(os.getenv("DB_POOL_MAX_INACTIVE_LIFETIME", "300"))
        self.timeout = int(os.getenv("DB_POOL_TIMEOUT", "60"))
        
        # Performance tracking
        self.query_count = 0
        self.slow_query_threshold = 1.0  # seconds
        self._initialized = False
    
    @classmethod
    async def get_instance(cls) -> 'SupabasePool':
        """Get singleton instance of connection pool"""
        if cls._instance is None:
            async with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
                    await cls._instance.init_pool()
        return cls._instance
    
    async def init_pool(self):
        """Initialize the connection pool"""
        if self._initialized:
            return
        
        if not self.database_url:
            raise ValueError("DATABASE_URL not set")
        
        try:
            logger.info(f"Initializing Supabase connection pool (min={self.min_size}, max={self.max_size})")
            
            self.pool = await asyncpg.create_pool(
                self.database_url,
                min_size=self.min_size,
                max_size=self.max_size,
                max_queries=self.max_queries,
                max_inactive_connection_lifetime=self.max_inactive_lifetime,
                command_timeout=self.timeout,
                # Performance optimizations
                server_settings={
                    'jit': 'on',
                    'max_parallel_workers_per_gather': '4',
                    'work_mem': '256MB',
                    'shared_preload_libraries': 'pg_stat_statements'
                },
                # Connection setup
                setup=self._setup_connection
            )
            
            self._initialized = True
            logger.info("Supabase connection pool initialized successfully")
            
            # Test the pool
            await self._test_pool()
            
        except Exception as e:
            logger.error(f"Failed to initialize connection pool: {e}")
            raise
    
    async def _setup_connection(self, connection: Connection):
        """Setup each new connection"""
        # Set application name for monitoring
        await connection.execute("SET application_name = 'nba_ml_platform'")
        
        # Set statement timeout to prevent long-running queries
        await connection.execute("SET statement_timeout = '30s'")
        
        # Enable query timing
        await connection.execute("SET log_min_duration_statement = 1000")  # Log queries over 1s
    
    async def _test_pool(self):
        """Test the connection pool"""
        try:
            async with self.acquire() as conn:
                result = await conn.fetchval("SELECT 1")
                if result != 1:
                    raise RuntimeError("Pool test query failed")
                logger.info("Connection pool test successful")
        except Exception as e:
            logger.error(f"Connection pool test failed: {e}")
            raise
    
    @asynccontextmanager
    async def acquire(self) -> AsyncGenerator[Connection, None]:
        """Acquire a connection from the pool"""
        if not self.pool:
            raise RuntimeError("Connection pool not initialized")
        
        start_time = time.time()
        connection = None
        
        try:
            # Acquire connection with timeout
            connection = await asyncio.wait_for(
                self.pool.acquire(),
                timeout=10.0
            )
            
            acquisition_time = time.time() - start_time
            if acquisition_time > 1.0:
                logger.warning(f"Slow connection acquisition: {acquisition_time:.2f}s")
            
            yield connection
            
        except asyncio.TimeoutError:
            logger.error("Timeout acquiring database connection")
            raise
        except Exception as e:
            logger.error(f"Error acquiring connection: {e}")
            raise
        finally:
            if connection:
                await self.pool.release(connection)
                self.query_count += 1
    
    async def execute(self, query: str, *args, timeout: Optional[float] = None):
        """Execute a query with automatic connection handling"""
        async with self.acquire() as conn:
            start_time = time.time()
            
            try:
                if timeout:
                    result = await asyncio.wait_for(
                        conn.execute(query, *args),
                        timeout=timeout
                    )
                else:
                    result = await conn.execute(query, *args)
                
                # Log slow queries
                duration = time.time() - start_time
                if duration > self.slow_query_threshold:
                    logger.warning(f"Slow query ({duration:.2f}s): {query[:100]}...")
                
                return result
                
            except asyncio.TimeoutError:
                logger.error(f"Query timeout after {timeout}s: {query[:100]}...")
                raise
            except Exception as e:
                logger.error(f"Query error: {e}")
                raise
    
    async def fetch(self, query: str, *args, timeout: Optional[float] = None):
        """Fetch multiple rows"""
        async with self.acquire() as conn:
            start_time = time.time()
            
            try:
                if timeout:
                    result = await asyncio.wait_for(
                        conn.fetch(query, *args),
                        timeout=timeout
                    )
                else:
                    result = await conn.fetch(query, *args)
                
                duration = time.time() - start_time
                if duration > self.slow_query_threshold:
                    logger.warning(f"Slow fetch ({duration:.2f}s): {query[:100]}...")
                
                return result
                
            except Exception as e:
                logger.error(f"Fetch error: {e}")
                raise
    
    async def fetchrow(self, query: str, *args, timeout: Optional[float] = None):
        """Fetch a single row"""
        async with self.acquire() as conn:
            try:
                if timeout:
                    result = await asyncio.wait_for(
                        conn.fetchrow(query, *args),
                        timeout=timeout
                    )
                else:
                    result = await conn.fetchrow(query, *args)
                
                return result
                
            except Exception as e:
                logger.error(f"Fetchrow error: {e}")
                raise
    
    async def fetchval(self, query: str, *args, timeout: Optional[float] = None):
        """Fetch a single value"""
        async with self.acquire() as conn:
            try:
                if timeout:
                    result = await asyncio.wait_for(
                        conn.fetchval(query, *args),
                        timeout=timeout
                    )
                else:
                    result = await conn.fetchval(query, *args)
                
                return result
                
            except Exception as e:
                logger.error(f"Fetchval error: {e}")
                raise
    
    async def get_pool_stats(self) -> dict:
        """Get current pool statistics"""
        if not self.pool:
            return {"status": "not_initialized"}
        
        stats = {
            "total_connections": self.pool.get_size(),
            "idle_connections": self.pool.get_idle_size(),
            "used_connections": self.pool.get_size() - self.pool.get_idle_size(),
            "min_size": self.pool.get_min_size(),
            "max_size": self.pool.get_max_size(),
            "query_count": self.query_count,
            "status": "healthy"
        }
        
        # Check pool health
        usage_ratio = stats["used_connections"] / stats["total_connections"]
        if usage_ratio > 0.8:
            stats["status"] = "high_usage"
            logger.warning(f"High connection pool usage: {usage_ratio:.1%}")
        elif usage_ratio < 0.1 and stats["total_connections"] > self.min_size:
            stats["status"] = "low_usage"
        
        return stats
    
    async def close(self):
        """Close the connection pool"""
        if self.pool:
            await self.pool.close()
            self._initialized = False
            logger.info("Connection pool closed")
    
    def transaction_wrapper(self, isolation_level: str = 'read_committed'):
        """Decorator for transactional operations"""
        def decorator(func):
            @wraps(func)
            async def wrapper(*args, **kwargs):
                async with self.acquire() as conn:
                    async with conn.transaction(isolation=isolation_level):
                        # Pass connection as first argument
                        return await func(conn, *args, **kwargs)
            return wrapper
        return decorator


# Convenience functions
async def get_db_pool() -> SupabasePool:
    """Get the database connection pool"""
    return await SupabasePool.get_instance()


@asynccontextmanager
async def get_db_connection() -> AsyncGenerator[Connection, None]:
    """Get a database connection from the pool"""
    pool = await get_db_pool()
    async with pool.acquire() as conn:
        yield conn


# Query helpers with automatic pooling
async def execute_query(query: str, *args, timeout: Optional[float] = None):
    """Execute a query using the pool"""
    pool = await get_db_pool()
    return await pool.execute(query, *args, timeout=timeout)


async def fetch_all(query: str, *args, timeout: Optional[float] = None):
    """Fetch all rows using the pool"""
    pool = await get_db_pool()
    return await pool.fetch(query, *args, timeout=timeout)


async def fetch_one(query: str, *args, timeout: Optional[float] = None):
    """Fetch one row using the pool"""
    pool = await get_db_pool()
    return await pool.fetchrow(query, *args, timeout=timeout)


async def fetch_value(query: str, *args, timeout: Optional[float] = None):
    """Fetch a single value using the pool"""
    pool = await get_db_pool()
    return await pool.fetchval(query, *args, timeout=timeout)