#!/usr/bin/env python
"""
Production server starter for NBA ML API
Handles environment configuration and server initialization
"""
import os
import sys

# Force IPv4 before any imports
import socket
_original = socket.getaddrinfo
def _ipv4_only(host, port, family=0, type=0, proto=0, flags=0):
    return _original(host, port, socket.AF_INET, type, proto, flags)
socket.getaddrinfo = _ipv4_only

import uvicorn
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8080))
    host = "0.0.0.0"
    
    logger.info(f"Starting NBA ML API on {host}:{port}")
    logger.info(f"Environment: {os.getenv('ENVIRONMENT', 'development')}")
    logger.info(f"IPv4 forcing: enabled")
    
    # Direct uvicorn run with minimal configuration
    uvicorn.run(
        "api.main:app",
        host=host,
        port=port,
        log_level="info",
        access_log=True
    )