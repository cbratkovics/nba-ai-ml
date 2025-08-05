"""
Production-grade error handling middleware
"""
from fastapi import Request, status
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
import traceback
import logging
import uuid
import os
from datetime import datetime
from typing import Union

logger = logging.getLogger(__name__)


async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """
    Global exception handler for all unhandled exceptions
    
    - Logs full error details for debugging
    - Returns sanitized error responses in production
    - Includes error tracking ID for support
    """
    error_id = str(uuid.uuid4())
    timestamp = datetime.utcnow().isoformat()
    
    # Log full error details
    logger.error(
        f"Unhandled exception {error_id}",
        extra={
            "error_id": error_id,
            "path": request.url.path,
            "method": request.method,
            "error_type": type(exc).__name__,
            "error_message": str(exc),
            "traceback": traceback.format_exc(),
            "headers": dict(request.headers),
        },
        exc_info=True
    )
    
    # Determine if we're in production
    is_production = os.getenv("ENVIRONMENT", "development").lower() == "production"
    
    # Prepare error response
    error_response = {
        "error": "Internal server error",
        "error_id": error_id,
        "timestamp": timestamp,
        "path": request.url.path,
    }
    
    # Include detailed error info in non-production environments
    if not is_production:
        error_response.update({
            "error_type": type(exc).__name__,
            "error_message": str(exc),
            "traceback": traceback.format_exc().split('\n')
        })
    
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=error_response
    )


async def http_exception_handler(request: Request, exc: StarletteHTTPException) -> JSONResponse:
    """
    Handler for HTTP exceptions (4xx, 5xx status codes)
    """
    error_id = str(uuid.uuid4())
    
    # Log 5xx errors as errors, 4xx as warnings
    if exc.status_code >= 500:
        logger.error(
            f"HTTP {exc.status_code} error {error_id}",
            extra={
                "error_id": error_id,
                "path": request.url.path,
                "method": request.method,
                "status_code": exc.status_code,
                "detail": exc.detail,
            }
        )
    else:
        logger.warning(
            f"HTTP {exc.status_code} error {error_id}",
            extra={
                "error_id": error_id,
                "path": request.url.path,
                "method": request.method,
                "status_code": exc.status_code,
                "detail": exc.detail,
            }
        )
    
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "error_id": error_id,
            "status_code": exc.status_code,
            "path": request.url.path,
            "timestamp": datetime.utcnow().isoformat(),
        }
    )


async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    """
    Handler for request validation errors (422)
    """
    error_id = str(uuid.uuid4())
    
    logger.warning(
        f"Validation error {error_id}",
        extra={
            "error_id": error_id,
            "path": request.url.path,
            "method": request.method,
            "errors": exc.errors(),
        }
    )
    
    # Format validation errors for better readability
    formatted_errors = []
    for error in exc.errors():
        field_path = " → ".join(str(loc) for loc in error["loc"])
        formatted_errors.append({
            "field": field_path,
            "message": error["msg"],
            "type": error["type"]
        })
    
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "error": "Validation failed",
            "error_id": error_id,
            "validation_errors": formatted_errors,
            "path": request.url.path,
            "timestamp": datetime.utcnow().isoformat(),
        }
    )


class DatabaseConnectionError(Exception):
    """Raised when database connection fails"""
    pass


class ModelLoadError(Exception):
    """Raised when ML model fails to load"""
    pass


class PredictionError(Exception):
    """Raised when prediction fails"""
    pass


class FeatureExtractionError(Exception):
    """Raised when feature extraction fails"""
    pass


async def database_exception_handler(request: Request, exc: DatabaseConnectionError) -> JSONResponse:
    """
    Handler for database connection errors
    """
    error_id = str(uuid.uuid4())
    
    logger.error(
        f"Database connection error {error_id}",
        extra={
            "error_id": error_id,
            "path": request.url.path,
            "error": str(exc),
        }
    )
    
    return JSONResponse(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        content={
            "error": "Database service temporarily unavailable",
            "error_id": error_id,
            "retry_after": 60,  # Suggest retry after 60 seconds
            "timestamp": datetime.utcnow().isoformat(),
        },
        headers={
            "Retry-After": "60"
        }
    )


async def model_exception_handler(request: Request, exc: Union[ModelLoadError, PredictionError]) -> JSONResponse:
    """
    Handler for ML model related errors
    """
    error_id = str(uuid.uuid4())
    
    logger.error(
        f"ML model error {error_id}",
        extra={
            "error_id": error_id,
            "path": request.url.path,
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
    )
    
    status_code = status.HTTP_503_SERVICE_UNAVAILABLE if isinstance(exc, ModelLoadError) else status.HTTP_500_INTERNAL_SERVER_ERROR
    
    return JSONResponse(
        status_code=status_code,
        content={
            "error": "Prediction service error",
            "error_id": error_id,
            "message": "Unable to process prediction request",
            "timestamp": datetime.utcnow().isoformat(),
        }
    )