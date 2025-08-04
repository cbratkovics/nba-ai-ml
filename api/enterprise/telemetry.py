"""
OpenTelemetry Monitoring and Observability
Enterprise-grade monitoring with distributed tracing, metrics, and logging
"""
import os
import time
import logging
from typing import Dict, Any, Optional, List, Callable
from datetime import datetime, timedelta
from contextlib import contextmanager
import asyncio
from functools import wraps

# OpenTelemetry imports
from opentelemetry import trace, metrics, baggage
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
from opentelemetry.exporter.prometheus import PrometheusMetricReader
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.sqlalchemy import SQLAlchemyInstrumentor
from opentelemetry.instrumentation.redis import RedisInstrumentor
from opentelemetry.instrumentation.requests import RequestsInstrumentor
from opentelemetry.instrumentation.aiohttp_client import AioHTTPClientInstrumentor
from opentelemetry.instrumentation.psycopg2 import Psycopg2Instrumentor
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader, ConsoleMetricExporter
from opentelemetry.sdk.resources import Resource, SERVICE_NAME, SERVICE_VERSION
from opentelemetry.trace import Status, StatusCode
from opentelemetry.metrics import get_meter_provider, set_meter_provider
from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator
from opentelemetry.sdk.trace.sampling import TraceIdRatioBa sedSampler, AlwaysOnSampler
from opentelemetry.semconv.trace import SpanAttributes
from opentelemetry.semconv.resource import ResourceAttributes

# Prometheus client for custom metrics
from prometheus_client import Counter, Histogram, Gauge, Summary, generate_latest
import prometheus_client

# FastAPI dependencies
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

logger = logging.getLogger(__name__)

# Configuration
OTLP_ENDPOINT = os.getenv("OTLP_ENDPOINT", "localhost:4317")
SERVICE_NAME_ENV = os.getenv("SERVICE_NAME", "nba-ml-api")
SERVICE_VERSION_ENV = os.getenv("SERVICE_VERSION", "1.0.0")
ENVIRONMENT = os.getenv("ENVIRONMENT", "production")
TRACE_SAMPLING_RATE = float(os.getenv("TRACE_SAMPLING_RATE", "0.1"))


# Custom Metrics
prediction_counter = Counter(
    'nba_ml_predictions_total',
    'Total number of predictions made',
    ['model', 'target', 'status']
)

prediction_latency = Histogram(
    'nba_ml_prediction_duration_seconds',
    'Prediction request duration in seconds',
    ['model', 'target'],
    buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]
)

model_accuracy_gauge = Gauge(
    'nba_ml_model_accuracy',
    'Current model accuracy (R² score)',
    ['model', 'target']
)

active_users_gauge = Gauge(
    'nba_ml_active_users',
    'Number of active users',
    ['organization', 'tier']
)

api_requests_counter = Counter(
    'nba_ml_api_requests_total',
    'Total API requests',
    ['method', 'endpoint', 'status_code']
)

cache_operations = Counter(
    'nba_ml_cache_operations_total',
    'Cache operations',
    ['operation', 'status']
)

database_operations = Histogram(
    'nba_ml_database_duration_seconds',
    'Database operation duration',
    ['operation', 'table'],
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0]
)

websocket_connections = Gauge(
    'nba_ml_websocket_connections',
    'Active WebSocket connections'
)

training_jobs = Counter(
    'nba_ml_training_jobs_total',
    'Model training jobs',
    ['status', 'model']
)

billing_events = Counter(
    'nba_ml_billing_events_total',
    'Billing events',
    ['event_type', 'plan']
)


class TelemetryManager:
    """Manages telemetry and observability"""
    
    def __init__(self):
        self.tracer_provider = None
        self.meter_provider = None
        self.tracer = None
        self.meter = None
        self._initialized = False
        
    def initialize(self):
        """Initialize OpenTelemetry"""
        if self._initialized:
            return
        
        # Create resource
        resource = Resource.create({
            ResourceAttributes.SERVICE_NAME: SERVICE_NAME_ENV,
            ResourceAttributes.SERVICE_VERSION: SERVICE_VERSION_ENV,
            ResourceAttributes.DEPLOYMENT_ENVIRONMENT: ENVIRONMENT,
            ResourceAttributes.SERVICE_INSTANCE_ID: os.getenv("HOSTNAME", "unknown"),
            ResourceAttributes.CLOUD_PROVIDER: "aws",
            ResourceAttributes.CLOUD_REGION: os.getenv("AWS_REGION", "us-west-2"),
        })
        
        # Setup tracing
        self._setup_tracing(resource)
        
        # Setup metrics
        self._setup_metrics(resource)
        
        # Instrument libraries
        self._instrument_libraries()
        
        self._initialized = True
        logger.info("OpenTelemetry initialized successfully")
    
    def _setup_tracing(self, resource: Resource):
        """Setup distributed tracing"""
        
        # Create tracer provider
        self.tracer_provider = TracerProvider(
            resource=resource,
            sampler=TraceIdRatioBa sedSampler(TRACE_SAMPLING_RATE)
        )
        
        # Add OTLP exporter
        otlp_exporter = OTLPSpanExporter(
            endpoint=OTLP_ENDPOINT,
            insecure=True
        )
        span_processor = BatchSpanProcessor(otlp_exporter)
        self.tracer_provider.add_span_processor(span_processor)
        
        # Add console exporter for debugging
        if ENVIRONMENT == "development":
            console_exporter = ConsoleSpanExporter()
            console_processor = BatchSpanProcessor(console_exporter)
            self.tracer_provider.add_span_processor(console_processor)
        
        # Set global tracer provider
        trace.set_tracer_provider(self.tracer_provider)
        
        # Get tracer
        self.tracer = trace.get_tracer(__name__, SERVICE_VERSION_ENV)
    
    def _setup_metrics(self, resource: Resource):
        """Setup metrics collection"""
        
        # Create OTLP metric exporter
        otlp_metric_exporter = OTLPMetricExporter(
            endpoint=OTLP_ENDPOINT,
            insecure=True
        )
        
        # Create metric readers
        otlp_reader = PeriodicExportingMetricReader(
            exporter=otlp_metric_exporter,
            export_interval_millis=60000  # Export every minute
        )
        
        # Add Prometheus exporter
        prometheus_reader = PrometheusMetricReader()
        
        # Create meter provider
        self.meter_provider = MeterProvider(
            resource=resource,
            metric_readers=[otlp_reader, prometheus_reader]
        )
        
        # Set global meter provider
        set_meter_provider(self.meter_provider)
        
        # Get meter
        self.meter = metrics.get_meter(__name__, SERVICE_VERSION_ENV)
        
        # Create custom metrics
        self._create_custom_metrics()
    
    def _create_custom_metrics(self):
        """Create custom OpenTelemetry metrics"""
        
        # Counter for predictions
        self.prediction_counter = self.meter.create_counter(
            name="nba_ml.predictions",
            description="Number of predictions made",
            unit="1"
        )
        
        # Histogram for latency
        self.prediction_latency = self.meter.create_histogram(
            name="nba_ml.prediction.latency",
            description="Prediction latency in milliseconds",
            unit="ms"
        )
        
        # Gauge for model accuracy
        self.model_accuracy = self.meter.create_observable_gauge(
            name="nba_ml.model.accuracy",
            description="Model accuracy (R² score)",
            callbacks=[self._get_model_accuracy]
        )
        
        # Up-down counter for active sessions
        self.active_sessions = self.meter.create_up_down_counter(
            name="nba_ml.sessions.active",
            description="Number of active sessions",
            unit="1"
        )
    
    def _instrument_libraries(self):
        """Auto-instrument common libraries"""
        
        # FastAPI
        FastAPIInstrumentor.instrument()
        
        # SQLAlchemy
        SQLAlchemyInstrumentor().instrument()
        
        # Redis
        RedisInstrumentor().instrument()
        
        # HTTP clients
        RequestsInstrumentor().instrument()
        AioHTTPClientInstrumentor().instrument()
        
        # PostgreSQL
        Psycopg2Instrumentor().instrument()
        
        logger.info("Libraries instrumented for telemetry")
    
    def _get_model_accuracy(self, options):
        """Callback to get model accuracy metrics"""
        # This would fetch actual model accuracy from database or cache
        observations = []
        
        for model in ["pts", "reb", "ast"]:
            observations.append(
                metrics.Observation(
                    value=0.94 + (hash(model) % 5) / 100,  # Mock value
                    attributes={"model": model}
                )
            )
        
        return observations
    
    @contextmanager
    def trace_span(self, name: str, attributes: Dict[str, Any] = None):
        """Context manager for creating trace spans"""
        with self.tracer.start_as_current_span(name) as span:
            if attributes:
                for key, value in attributes.items():
                    span.set_attribute(key, value)
            
            try:
                yield span
            except Exception as e:
                span.record_exception(e)
                span.set_status(Status(StatusCode.ERROR, str(e)))
                raise
            finally:
                span.set_status(Status(StatusCode.OK))
    
    def trace_async(self, name: str = None):
        """Decorator for tracing async functions"""
        def decorator(func):
            span_name = name or f"{func.__module__}.{func.__name__}"
            
            @wraps(func)
            async def wrapper(*args, **kwargs):
                with self.trace_span(span_name) as span:
                    # Add function arguments as span attributes
                    span.set_attribute("function.args", str(args[:3]))  # Limit to first 3 args
                    span.set_attribute("function.kwargs_keys", str(list(kwargs.keys())))
                    
                    start_time = time.time()
                    try:
                        result = await func(*args, **kwargs)
                        span.set_attribute("function.success", True)
                        return result
                    except Exception as e:
                        span.set_attribute("function.success", False)
                        raise
                    finally:
                        duration = (time.time() - start_time) * 1000
                        span.set_attribute("function.duration_ms", duration)
            
            return wrapper
        return decorator
    
    def record_prediction(self, model: str, target: str, latency: float, success: bool = True):
        """Record prediction metrics"""
        
        # Prometheus metrics
        prediction_counter.labels(
            model=model,
            target=target,
            status="success" if success else "failure"
        ).inc()
        
        prediction_latency.labels(
            model=model,
            target=target
        ).observe(latency)
        
        # OpenTelemetry metrics
        self.prediction_counter.add(
            1,
            {"model": model, "target": target, "status": "success" if success else "failure"}
        )
        
        self.prediction_latency.record(
            latency * 1000,  # Convert to milliseconds
            {"model": model, "target": target}
        )
    
    def record_api_request(self, method: str, endpoint: str, status_code: int, duration: float):
        """Record API request metrics"""
        
        api_requests_counter.labels(
            method=method,
            endpoint=endpoint,
            status_code=str(status_code)
        ).inc()
        
        # Add to trace span if exists
        span = trace.get_current_span()
        if span:
            span.set_attribute(SpanAttributes.HTTP_METHOD, method)
            span.set_attribute(SpanAttributes.HTTP_URL, endpoint)
            span.set_attribute(SpanAttributes.HTTP_STATUS_CODE, status_code)
            span.set_attribute("http.duration_ms", duration * 1000)
    
    def record_cache_operation(self, operation: str, hit: bool):
        """Record cache operation metrics"""
        
        cache_operations.labels(
            operation=operation,
            status="hit" if hit else "miss"
        ).inc()
    
    def record_database_operation(self, operation: str, table: str, duration: float):
        """Record database operation metrics"""
        
        database_operations.labels(
            operation=operation,
            table=table
        ).observe(duration)


class TelemetryMiddleware(BaseHTTPMiddleware):
    """Middleware for request telemetry"""
    
    def __init__(self, app: ASGIApp, telemetry_manager: TelemetryManager):
        super().__init__(app)
        self.telemetry = telemetry_manager
    
    async def dispatch(self, request: Request, call_next):
        """Process request with telemetry"""
        
        # Start span
        with self.telemetry.trace_span(
            f"{request.method} {request.url.path}",
            attributes={
                SpanAttributes.HTTP_METHOD: request.method,
                SpanAttributes.HTTP_URL: str(request.url),
                SpanAttributes.HTTP_SCHEME: request.url.scheme,
                SpanAttributes.HTTP_HOST: request.url.hostname,
                SpanAttributes.HTTP_TARGET: request.url.path,
                SpanAttributes.HTTP_USER_AGENT: request.headers.get("user-agent", ""),
                "client.address": request.client.host if request.client else "unknown",
                "organization.id": request.headers.get("x-organization-id", "unknown"),
            }
        ) as span:
            
            # Add baggage for distributed context
            baggage.set_baggage("request.id", span.get_span_context().trace_id)
            baggage.set_baggage("organization.id", request.headers.get("x-organization-id", "unknown"))
            
            start_time = time.time()
            
            try:
                # Process request
                response = await call_next(request)
                
                # Record metrics
                duration = time.time() - start_time
                
                span.set_attribute(SpanAttributes.HTTP_STATUS_CODE, response.status_code)
                
                self.telemetry.record_api_request(
                    method=request.method,
                    endpoint=request.url.path,
                    status_code=response.status_code,
                    duration=duration
                )
                
                # Add trace ID to response headers
                response.headers["x-trace-id"] = format(span.get_span_context().trace_id, "032x")
                
                return response
                
            except Exception as e:
                span.record_exception(e)
                span.set_status(Status(StatusCode.ERROR, str(e)))
                raise


class MetricsEndpoint:
    """Prometheus metrics endpoint"""
    
    async def __call__(self, request: Request) -> Response:
        """Return Prometheus metrics"""
        metrics_data = generate_latest()
        return Response(
            content=metrics_data,
            media_type="text/plain; version=0.0.4"
        )


# Global telemetry manager instance
telemetry_manager = TelemetryManager()


def setup_telemetry(app):
    """Setup telemetry for FastAPI app"""
    
    # Initialize telemetry
    telemetry_manager.initialize()
    
    # Add middleware
    app.add_middleware(TelemetryMiddleware, telemetry_manager=telemetry_manager)
    
    # Add metrics endpoint
    app.add_route("/metrics", MetricsEndpoint(), methods=["GET"])
    
    logger.info("Telemetry setup complete")


# Decorators for easy use
def trace_span(name: str = None):
    """Decorator to trace function execution"""
    def decorator(func):
        span_name = name or f"{func.__module__}.{func.__name__}"
        
        if asyncio.iscoroutinefunction(func):
            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                with telemetry_manager.trace_span(span_name):
                    return await func(*args, **kwargs)
            return async_wrapper
        else:
            @wraps(func)
            def sync_wrapper(*args, **kwargs):
                with telemetry_manager.trace_span(span_name):
                    return func(*args, **kwargs)
            return sync_wrapper
    
    return decorator


def measure_time(metric_name: str):
    """Decorator to measure function execution time"""
    def decorator(func):
        if asyncio.iscoroutinefunction(func):
            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                start = time.time()
                try:
                    result = await func(*args, **kwargs)
                    return result
                finally:
                    duration = time.time() - start
                    telemetry_manager.meter.create_histogram(
                        name=metric_name,
                        description=f"Execution time for {func.__name__}",
                        unit="s"
                    ).record(duration)
            return async_wrapper
        else:
            @wraps(func)
            def sync_wrapper(*args, **kwargs):
                start = time.time()
                try:
                    result = func(*args, **kwargs)
                    return result
                finally:
                    duration = time.time() - start
                    telemetry_manager.meter.create_histogram(
                        name=metric_name,
                        description=f"Execution time for {func.__name__}",
                        unit="s"
                    ).record(duration)
            return sync_wrapper
    
    return decorator