import os
import urllib.parse
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.engine.url import make_url
from contextlib import asynccontextmanager
import logging

logger = logging.getLogger(__name__)

def get_database_url():
    """Get properly formatted database URL with encoded special characters"""
    db_url = os.getenv("DATABASE_URL", "")
    
    if not db_url:
        logger.error("DATABASE_URL not found in environment variables!")
        logger.error("Please set DATABASE_URL in Railway dashboard -> Variables")
        logger.error("Format: postgresql://user:password@host:port/database")
        return None
    
    # Handle Heroku-style postgres:// URLs
    db_url = db_url.replace("postgres://", "postgresql://")
    
    # Parse and fix the URL for special characters
    try:
        url = make_url(db_url)
        # URL encode the password if it contains special characters
        if url.password and any(c in url.password for c in ['@', ':', '/', '?', '#', '[', ']', '!', '$', '&', "'", '(', ')', '*', '+', ',', ';', '=']):
            encoded_password = urllib.parse.quote(url.password, safe='')
            url = url.set(password=encoded_password)
            db_url = str(url)
            logger.info("Database password URL-encoded for special characters")
        
        logger.info("Database URL configured successfully")
        return db_url
    except Exception as e:
        logger.error(f"Failed to parse DATABASE_URL: {e}")
        logger.error("Please ensure your DATABASE_URL is properly formatted")
        return db_url

DATABASE_URL = get_database_url() or "postgresql://dummy:dummy@localhost/dummy"

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