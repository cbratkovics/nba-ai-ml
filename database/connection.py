import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from contextlib import asynccontextmanager
import logging

logger = logging.getLogger(__name__)

DATABASE_URL = os.getenv("DATABASE_URL", "")

if not DATABASE_URL:
    logger.error("DATABASE_URL not found in environment variables!")
    logger.error("Please set DATABASE_URL in Railway dashboard -> Variables")
    logger.error("Format: postgresql://user:password@host:port/database")
    # Use a dummy URL to prevent immediate crash, allowing health checks to work
    DATABASE_URL = "postgresql://dummy:dummy@localhost/dummy"
else:
    # Handle Heroku-style postgres:// URLs
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://")
    logger.info("Database URL configured successfully")

# Synchronous engine for compatibility
try:
    engine = create_engine(DATABASE_URL, pool_pre_ping=True)
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    logger.info("Database engine created successfully")
except Exception as e:
    logger.error(f"Failed to create database engine: {e}")
    # Create dummy engine to prevent crashes
    engine = create_engine("postgresql://dummy:dummy@localhost/dummy")
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Async engine for async operations
async_database_url = DATABASE_URL.replace("postgresql://", "postgresql+asyncpg://")
try:
    async_engine = create_async_engine(async_database_url, pool_pre_ping=True)
    AsyncSessionLocal = async_sessionmaker(async_engine, class_=AsyncSession, expire_on_commit=False)
except Exception as e:
    logger.error(f"Failed to create async engine: {e}")
    async_engine = None
    AsyncSessionLocal = None

def get_db():
    """Get synchronous database session"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@asynccontextmanager
async def get_db_session():
    """Get async database session"""
    if AsyncSessionLocal is None:
        raise RuntimeError("Async database not configured - check DATABASE_URL")
    
    async with AsyncSessionLocal() as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()

def test_db_connection():
    """Test database connection"""
    try:
        with engine.connect() as conn:
            result = conn.execute("SELECT 1")
            return True, "Database connection successful"
    except Exception as e:
        return False, f"Database connection failed: {str(e)}"