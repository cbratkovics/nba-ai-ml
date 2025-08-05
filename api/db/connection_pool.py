"""
Production-grade database connection pool with async support
"""
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.pool import NullPool, QueuePool
from sqlalchemy import event, pool
import asyncpg
import asyncio
import logging
import os
from typing import Optional, AsyncGenerator
from contextlib import asynccontextmanager
import time

logger = logging.getLogger(__name__)


class DatabasePool:
    """
    Robust database connection pool with retry logic and health monitoring
    """
    
    def __init__(self):
        self.engine: Optional[create_async_engine] = None
        self.session_factory: Optional[async_sessionmaker] = None
        self._initialized = False
        self._connection_count = 0
        self._error_count = 0
        self._last_error_time = None
        
    async def init(self, database_url: str, **kwargs):
        """
        Initialize connection pool with retry logic
        
        Args:
            database_url: Database connection URL
            **kwargs: Additional engine configuration
        """
        max_retries = kwargs.pop('max_retries', 3)
        retry_delay = kwargs.pop('retry_delay', 2)
        
        # Convert to async URL if needed
        if database_url.startswith("postgresql://"):
            database_url = database_url.replace("postgresql://", "postgresql+asyncpg://")
        
        for attempt in range(max_retries):
            try:
                logger.info(f"Initializing database pool (attempt {attempt + 1}/{max_retries})")
                
                # Create engine with production settings
                self.engine = create_async_engine(
                    database_url,
                    # Connection pool settings
                    pool_size=kwargs.get('pool_size', 20),
                    max_overflow=kwargs.get('max_overflow', 10),
                    pool_timeout=kwargs.get('pool_timeout', 30),
                    pool_recycle=kwargs.get('pool_recycle', 3600),  # Recycle connections after 1 hour
                    pool_pre_ping=True,  # Verify connections before use
                    
                    # Performance settings
                    echo=kwargs.get('echo', False),
                    echo_pool=kwargs.get('echo_pool', False),
                    
                    # Use QueuePool for better performance
                    poolclass=QueuePool,
                    
                    # Connection arguments for asyncpg
                    connect_args={
                        "server_settings": {
                            "application_name": "nba-ml-api",
                            "jit": "off"
                        },
                        "timeout": 10,
                        "command_timeout": 10,
                    }
                )
                
                # Set up event listeners
                self._setup_event_listeners()
                
                # Test the connection
                async with self.engine.connect() as conn:
                    await conn.execute("SELECT 1")
                
                # Create session factory
                self.session_factory = async_sessionmaker(
                    self.engine,
                    class_=AsyncSession,
                    expire_on_commit=False
                )
                
                self._initialized = True
                logger.info("Database pool initialized successfully")
                logger.info(f"Pool size: {self.engine.pool.size()}, Max overflow: {self.engine.pool.overflow()}")
                break
                
            except Exception as e:
                self._error_count += 1
                self._last_error_time = time.time()
                logger.error(f"Database connection attempt {attempt + 1} failed: {e}")
                
                if attempt == max_retries - 1:
                    raise RuntimeError(f"Failed to initialize database pool after {max_retries} attempts: {e}")
                
                await asyncio.sleep(retry_delay * (2 ** attempt))  # Exponential backoff
    
    def _setup_event_listeners(self):
        """Set up SQLAlchemy event listeners for monitoring"""
        
        @event.listens_for(self.engine.sync_engine, "connect")
        def receive_connect(dbapi_conn, connection_record):
            """Track new connections"""
            self._connection_count += 1
            logger.debug(f"New database connection established (total: {self._connection_count})")
        
        @event.listens_for(self.engine.sync_engine, "checkout")
        def receive_checkout(dbapi_conn, connection_record, connection_proxy):
            """Log when connections are checked out from pool"""
            logger.debug("Connection checked out from pool")
        
        @event.listens_for(self.engine.sync_engine, "checkin")
        def receive_checkin(dbapi_conn, connection_record):
            """Log when connections are returned to pool"""
            logger.debug("Connection returned to pool")
    
    @asynccontextmanager
    async def get_session(self) -> AsyncGenerator[AsyncSession, None]:
        """
        Get a database session with automatic error handling and cleanup
        
        Yields:
            AsyncSession: Database session
            
        Raises:
            RuntimeError: If pool is not initialized
        """
        if not self._initialized:
            raise RuntimeError("Database pool not initialized. Call init() first.")
        
        async with self.session_factory() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                self._error_count += 1
                self._last_error_time = time.time()
                raise
            finally:
                await session.close()
    
    async def execute_with_retry(self, query, max_retries: int = 3):
        """
        Execute a query with automatic retry logic
        
        Args:
            query: SQLAlchemy query to execute
            max_retries: Maximum number of retry attempts
            
        Returns:
            Query result
        """
        last_error = None
        
        for attempt in range(max_retries):
            try:
                async with self.get_session() as session:
                    result = await session.execute(query)
                    return result
            except Exception as e:
                last_error = e
                logger.warning(f"Query execution attempt {attempt + 1} failed: {e}")
                
                if attempt < max_retries - 1:
                    await asyncio.sleep(0.5 * (2 ** attempt))  # Exponential backoff
                    
        raise last_error
    
    async def health_check(self) -> dict:
        """
        Perform health check on database connection
        
        Returns:
            dict: Health status information
        """
        health_status = {
            "status": "unknown",
            "initialized": self._initialized,
            "connection_count": self._connection_count,
            "error_count": self._error_count,
            "last_error_time": self._last_error_time,
            "pool_size": None,
            "pool_checked_out": None,
            "response_time_ms": None,
        }
        
        if not self._initialized:
            health_status["status"] = "uninitialized"
            return health_status
        
        try:
            start_time = time.time()
            
            async with self.get_session() as session:
                result = await session.execute("SELECT 1")
                _ = result.scalar()
            
            response_time = (time.time() - start_time) * 1000
            
            # Get pool statistics
            pool_impl = self.engine.pool
            health_status.update({
                "status": "healthy",
                "pool_size": pool_impl.size(),
                "pool_checked_out": pool_impl.checked_out_connections(),
                "response_time_ms": round(response_time, 2),
            })
            
        except Exception as e:
            health_status["status"] = "unhealthy"
            health_status["error"] = str(e)
            logger.error(f"Database health check failed: {e}")
        
        return health_status
    
    async def close(self):
        """Close all database connections and cleanup"""
        if self.engine:
            await self.engine.dispose()
            self._initialized = False
            logger.info("Database pool closed")


# Global database pool instance
db_pool = DatabasePool()


async def init_database_pool():
    """Initialize the global database pool"""
    from database.connection import get_database_url
    
    database_url = get_database_url()
    if not database_url or "dummy" in database_url:
        logger.warning("No valid DATABASE_URL found, skipping pool initialization")
        return
    
    await db_pool.init(
        database_url,
        pool_size=int(os.getenv("DB_POOL_SIZE", "20")),
        max_overflow=int(os.getenv("DB_MAX_OVERFLOW", "10")),
        pool_timeout=int(os.getenv("DB_POOL_TIMEOUT", "30")),
        pool_recycle=int(os.getenv("DB_POOL_RECYCLE", "3600")),
        echo=os.getenv("DB_ECHO", "false").lower() == "true",
    )


async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    """
    Dependency to get database session
    
    Usage:
        @router.get("/users")
        async def get_users(db: AsyncSession = Depends(get_db_session)):
            result = await db.execute(select(User))
            return result.scalars().all()
    """
    async with db_pool.get_session() as session:
        yield session