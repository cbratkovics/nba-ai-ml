#!/usr/bin/env python
"""
Production server starter for NBA ML API
Handles environment configuration and server initialization
"""

import os
import sys
import socket

# Force IPv4 for Railway (must be before any imports that use networking)
if os.getenv("ENVIRONMENT") == "production":
    original_getaddrinfo = socket.getaddrinfo
    def force_ipv4(host, port, family=0, type=0, proto=0, flags=0):
        return original_getaddrinfo(host, port, socket.AF_INET, type, proto, flags)
    socket.getaddrinfo = force_ipv4
    print("Forcing IPv4 connections for Railway deployment")

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
        logger.info(f"Database URL configured: {'DATABASE_URL' in os.environ}")
        
        # Start the server
        if environment == "production":
            # Production configuration - use asyncio instead of uvloop for better compatibility
            logger.info("Starting in production mode with asyncio loop")
            uvicorn.run(
                "api.main:app",
                host=host,
                port=port,
                workers=workers,
                loop="asyncio",  # Use asyncio for better Railway compatibility
                access_log=True,  # Enable access logs for debugging
                log_level="info"
            )
        else:
            # Development configuration
            logger.info("Starting in development mode")
            uvicorn.run(
                "api.main:app",
                host=host,
                port=port,
                reload=True,
                log_level="debug"
            )
            
    except Exception as e:
        logger.error(f"Failed to start server: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()