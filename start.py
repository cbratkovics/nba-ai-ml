#!/usr/bin/env python
"""
Production server starter for NBA ML API
Handles environment configuration and server initialization
"""

import os
import sys
import logging
import uvicorn

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Start the production server"""
    try:
        # Get configuration from environment
        host = os.getenv("HOST", "0.0.0.0")
        port = int(os.getenv("PORT", 8080))
        workers = int(os.getenv("WORKERS", 1))
        environment = os.getenv("ENVIRONMENT", "production")
        
        logger.info("Starting NBA ML API server")
        logger.info(f"Host: {host}")
        logger.info(f"Port: {port}")
        logger.info(f"Environment: {environment}")
        
        # Start the server
        if environment == "production":
            # Production configuration
            uvicorn.run(
                "api.main:app",
                host=host,
                port=port,
                workers=workers,
                loop="uvloop",  # Better performance
                access_log=False,  # Disable access logs for performance
                log_level="info"
            )
        else:
            # Development configuration
            uvicorn.run(
                "api.main:app",
                host=host,
                port=port,
                reload=True,
                log_level="debug"
            )
            
    except Exception as e:
        logger.error(f"Failed to start server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()