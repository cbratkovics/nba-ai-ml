#!/usr/bin/env python
"""
Startup script for Railway deployment
Handles PORT environment variable properly
"""
import os
import sys
import uvicorn
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    # Get port from environment variable
    port = int(os.environ.get("PORT", 8000))
    host = "0.0.0.0"
    
    # Log startup information
    logger.info(f"Starting NBA ML API server")
    logger.info(f"Host: {host}")
    logger.info(f"Port: {port}")
    logger.info(f"Environment: {os.environ.get('ENVIRONMENT', 'production')}")
    
    # Check for critical environment variables
    if not os.environ.get("DATABASE_URL"):
        logger.warning("DATABASE_URL not set - database operations will fail")
    
    try:
        # Start the server
        uvicorn.run(
            "api.main:app",
            host=host,
            port=port,
            log_level="info",
            access_log=True
        )
    except Exception as e:
        logger.error(f"Failed to start server: {e}")
        sys.exit(1)