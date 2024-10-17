"""
Prometheus metrics module for AI Model Serving Platform
"""
import time
import functools
from typing import Dict, Any, Optional, Callable
from prometheus_client import (
    Counter, Histogram, Gauge, Info, 
    CollectorRegistry, generate_latest, 
    CONTENT_TYPE_LATEST, start_http_server
)
import logging

logger = logging.getLogger(__name__)


class ModelMetrics:
    """Prometheus metrics for model serving"""
    
    def __init__(self, model_name: str, model_version: str, registry: Optional[CollectorRegistry] = None):
        self.model_name = model_name
        self.model_version = model_version
        self.registry = registry or CollectorRegistry()
        
        # Request metrics
        self.request_count = Counter(
            'model_requests_total',
            'Total number of requests to the model',
            ['model_name', 'model_version', 'endpoint', 'method', 'status_code'],
            registry=self.registry
        )
        
        self.request_duration = Histogram(
            'model_request_duration_seconds',
            'Time spent processing model requests',
            ['model_name', 'model_version', 'endpoint', 'method'],
            buckets=(0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0, 7.5, 10.0),
            registry=self.registry
        )
        
        # Model inference metrics
        self.inference_duration = Histogram(
            'model_inference_duration_seconds',
            'Time spent on model inference',
            ['model_name', 'model_version'],
            buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0),
            registry=self.registry
        )
        
        self.batch_size = Histogram(
            'model_batch_size',
            'Batch size for model inference',
            ['model_name', 'model_version'],
            buckets=(1, 2, 4, 8, 16, 32, 64, 128, 256),
            registry=self.registry
        )
        
        # Model performance metrics
        self.prediction_confidence = Histogram(
            'model_prediction_confidence',
            'Confidence scores of model predictions',
            ['model_name', 'model_version'],
            buckets=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99),
            registry=self.registry
        )
        
        # Error metrics
        self.error_count = Counter(
            'model_errors_total',
            'Total number of errors in model serving',
            ['model_name', 'model_version', 'error_type'],
            registry=self.registry
        )
        
        # Resource metrics
        self.memory_usage = Gauge(
            'model_memory_usage_bytes',
            'Memory usage of the model server',
            ['model_name', 'model_version'],
            registry=self.registry
        )
        
        self.cpu_usage = Gauge(
            'model_cpu_usage_percent',
            'CPU usage of the model server',
            ['model_name', 'model_version'],
            registry=self.registry
        )
        
        self.gpu_usage = Gauge(
            'model_gpu_usage_percent',
            'GPU usage of the model server',
            ['model_name', 'model_version', 'gpu_id'],
            registry=self.registry
        )
        
        self.gpu_memory_usage = Gauge(
            'model_gpu_memory_usage_bytes',
            'GPU memory usage of the model server',
            ['model_name', 'model_version', 'gpu_id'],
            registry=self.registry
        )
        
        # Cache metrics
        self.cache_hits = Counter(
            'model_cache_hits_total',
            'Total number of cache hits',
            ['model_name', 'model_version'],
            registry=self.registry
        )
        
        self.cache_misses = Counter(
            'model_cache_misses_total',
            'Total number of cache misses',
            ['model_name', 'model_version'],
            registry=self.registry
        )
        
        # Model info
        self.model_info = Info(
            'model_info',
            'Information about the model',
            registry=self.registry
        )
        
        # Set model info
        self.model_info.info({
            'model_name': self.model_name,
            'model_version': self.model_version,
            'framework': 'pytorch',  # This could be dynamic
            'created_at': str(int(time.time()))
        })
        
        # Health metrics
        self.health_status = Gauge(
            'model_health_status',
            'Health status of the model (1=healthy, 0=unhealthy)',
            ['model_name', 'model_version'],
            registry=self.registry
        )
        
        # Set initial health status
        self.health_status.labels(
            model_name=self.model_name,
            model_version=self.model_version
        ).set(1)
    
    def record_request(self, endpoint: str, method: str, status_code: int, duration: float):
        """Record a request metric"""
        self.request_count.labels(
            model_name=self.model_name,
            model_version=self.model_version,
            endpoint=endpoint,
            method=method,
            status_code=status_code
        ).inc()
        
        self.request_duration.labels(
            model_name=self.model_name,
            model_version=self.model_version,
            endpoint=endpoint,
            method=method
        ).observe(duration)
    
    def record_inference(self, duration: float, batch_size: int, confidence_scores: Optional[list] = None):
        """Record inference metrics"""
        self.inference_duration.labels(
            model_name=self.model_name,
            model_version=self.model_version
        ).observe(duration)
        
        self.batch_size.labels(
            model_name=self.model_name,
            model_version=self.model_version
        ).observe(batch_size)
        
        if confidence_scores:
            for score in confidence_scores:
                self.prediction_confidence.labels(
                    model_name=self.model_name,
                    model_version=self.model_version
                ).observe(score)
    
    def record_error(self, error_type: str):
        """Record an error metric"""
        self.error_count.labels(
            model_name=self.model_name,
            model_version=self.model_version,
            error_type=error_type
        ).inc()
    
    def update_resource_usage(self, memory_bytes: int, cpu_percent: float, 
                            gpu_usage: Optional[Dict[str, float]] = None,
                            gpu_memory: Optional[Dict[str, int]] = None):
        """Update resource usage metrics"""
        self.memory_usage.labels(
            model_name=self.model_name,
            model_version=self.model_version
        ).set(memory_bytes)
        
        self.cpu_usage.labels(
            model_name=self.model_name,
            model_version=self.model_version
        ).set(cpu_percent)
        
        if gpu_usage:
            for gpu_id, usage in gpu_usage.items():
                self.gpu_usage.labels(
                    model_name=self.model_name,
                    model_version=self.model_version,
                    gpu_id=gpu_id
                ).set(usage)
        
        if gpu_memory:
            for gpu_id, memory in gpu_memory.items():
                self.gpu_memory_usage.labels(
                    model_name=self.model_name,
                    model_version=self.model_version,
                    gpu_id=gpu_id
                ).set(memory)
    
    def record_cache_hit(self):
        """Record a cache hit"""
        self.cache_hits.labels(
            model_name=self.model_name,
            model_version=self.model_version
        ).inc()
    
    def record_cache_miss(self):
        """Record a cache miss"""
        self.cache_misses.labels(
            model_name=self.model_name,
            model_version=self.model_version
        ).inc()
    
    def set_health_status(self, healthy: bool):
        """Set health status"""
        self.health_status.labels(
            model_name=self.model_name,
            model_version=self.model_version
        ).set(1 if healthy else 0)
    
    def get_metrics(self) -> str:
        """Get metrics in Prometheus format"""
        return generate_latest(self.registry).decode('utf-8')


def timing_decorator(metrics: ModelMetrics, endpoint: str):
    """Decorator to time function execution and record metrics"""
    def decorator(func: Callable):
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = await func(*args, **kwargs)
                duration = time.time() - start_time
                metrics.record_request(endpoint, "POST", 200, duration)
                return result
            except Exception as e:
                duration = time.time() - start_time
                metrics.record_request(endpoint, "POST", 500, duration)
                metrics.record_error(type(e).__name__)
                raise
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                metrics.record_request(endpoint, "POST", 200, duration)
                return result
            except Exception as e:
                duration = time.time() - start_time
                metrics.record_request(endpoint, "POST", 500, duration)
                metrics.record_error(type(e).__name__)
                raise
        
        # Return appropriate wrapper based on function type
        if hasattr(func, '__code__') and func.__code__.co_flags & 0x80:  # CO_COROUTINE
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


def start_metrics_server(port: int = 8080, registry: Optional[CollectorRegistry] = None):
    """Start Prometheus metrics HTTP server"""
    try:
        start_http_server(port, registry=registry)
        logger.info(f"Metrics server started on port {port}")
    except Exception as e:
        logger.error(f"Failed to start metrics server: {e}")
        raise

