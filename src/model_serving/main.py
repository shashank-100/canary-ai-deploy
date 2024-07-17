# ModelServeAI - Main Model Serving Application
# This file handles the core model serving functionality
"""
Main FastAPI application for AI Model Serving Platform
"""
import asyncio
import time
import uuid
from contextlib import asynccontextmanager
from typing import Dict, Any, List, Optional

import uvicorn
from fastapi import FastAPI, HTTPException, Depends, Request, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, Field

from ..common.config import config, load_model_endpoints_config
from ..common.metrics import ModelMetrics, start_metrics_server
from ..common.logging_config import setup_logging, get_logger, log_request, log_model_inference
from .model_manager import ModelManager
from .shadow_mode import ShadowModeManager
from .health_check import HealthChecker


# Setup logging
setup_logging(
    level=config.logging.level,
    format=config.logging.format,
    file_path=config.logging.file_path,
    max_file_size=config.logging.max_file_size,
    backup_count=config.logging.backup_count
)

logger = get_logger(__name__)

# Global instances
model_manager: Optional[ModelManager] = None
shadow_manager: Optional[ShadowModeManager] = None
health_checker: Optional[HealthChecker] = None
metrics: Optional[ModelMetrics] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    global model_manager, shadow_manager, health_checker, metrics
    
    logger.info("Starting AI Model Serving Platform")
    
    try:
        # Initialize metrics
        if config.metrics.enabled:
            metrics = ModelMetrics(config.model.name, config.model.version)
            start_metrics_server(config.metrics.port, metrics.registry)
            logger.info(f"Metrics server started on port {config.metrics.port}")
        
        # Initialize model manager
        model_manager = ModelManager(config.model.name, config.model.version)
        await model_manager.initialize()
        logger.info("Model manager initialized")
        
        # Initialize shadow mode manager
        if config.features.enable_shadow_mode:
            shadow_manager = ShadowModeManager(model_manager)
            await shadow_manager.initialize()
            logger.info("Shadow mode manager initialized")
        
        # Initialize health checker
        if config.features.enable_health_checks:
            health_checker = HealthChecker(model_manager)
            await health_checker.start()
            logger.info("Health checker started")
        
        logger.info("Application startup completed")
        yield
        
    except Exception as e:
        logger.error(f"Failed to start application: {e}", exc_info=True)
        raise
    finally:
        # Cleanup
        logger.info("Shutting down AI Model Serving Platform")
        
        if health_checker:
            await health_checker.stop()
        
        if shadow_manager:
            await shadow_manager.cleanup()
        
        if model_manager:
            await model_manager.cleanup()
        
        logger.info("Application shutdown completed")


# Create FastAPI app
app = FastAPI(
    title="AI Model Serving Platform",
    description="Scalable AI Model Serving Platform with Canary Deployments",
    version=config.model.version,
    lifespan=lifespan
)

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["*"]
)


# Request/Response models
class PredictionRequest(BaseModel):
    """Base prediction request model"""
    data: Dict[str, Any] = Field(..., description="Input data for prediction")
    model_version: Optional[str] = Field(None, description="Specific model version to use")
    shadow_mode: Optional[bool] = Field(False, description="Enable shadow mode for this request")


class PredictionResponse(BaseModel):
    """Base prediction response model"""
    predictions: List[Dict[str, Any]] = Field(..., description="Model predictions")
    model_name: str = Field(..., description="Name of the model used")
    model_version: str = Field(..., description="Version of the model used")
    request_id: str = Field(..., description="Unique request identifier")
    inference_time: float = Field(..., description="Inference time in seconds")
    confidence_scores: Optional[List[float]] = Field(None, description="Confidence scores")


class BatchPredictionRequest(BaseModel):
    """Batch prediction request model"""
    batch_data: List[Dict[str, Any]] = Field(..., description="Batch input data for prediction")
    model_version: Optional[str] = Field(None, description="Specific model version to use")
    batch_size: Optional[int] = Field(None, description="Batch size for processing")


class BatchPredictionResponse(BaseModel):
    """Batch prediction response model"""
    predictions: List[List[Dict[str, Any]]] = Field(..., description="Batch model predictions")
    model_name: str = Field(..., description="Name of the model used")
    model_version: str = Field(..., description="Version of the model used")
    request_id: str = Field(..., description="Unique request identifier")
    total_inference_time: float = Field(..., description="Total inference time in seconds")
    batch_size: int = Field(..., description="Actual batch size used")


class HealthResponse(BaseModel):
    """Health check response model"""
    status: str = Field(..., description="Health status")
    model_name: str = Field(..., description="Model name")
    model_version: str = Field(..., description="Model version")
    timestamp: str = Field(..., description="Timestamp of health check")
    details: Dict[str, Any] = Field(default_factory=dict, description="Additional health details")


# Dependency functions
async def get_request_id() -> str:
    """Generate unique request ID"""
    return str(uuid.uuid4())


async def log_request_middleware(request: Request, call_next):
    """Middleware to log requests"""
    start_time = time.time()
    request_id = str(uuid.uuid4())
    
    # Add request ID to request state
    request.state.request_id = request_id
    
    response = await call_next(request)
    
    duration = time.time() - start_time
    
    # Log request
    log_request(
        logger,
        request.method,
        str(request.url.path),
        response.status_code,
        duration,
        request_id
    )
    
    # Record metrics
    if metrics:
        metrics.record_request(
            str(request.url.path),
            request.method,
            response.status_code,
            duration
        )
    
    return response


# Add request logging middleware
app.middleware("http")(log_request_middleware)


# API Routes
@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint"""
    return {
        "message": "AI Model Serving Platform",
        "model_name": config.model.name,
        "model_version": config.model.version,
        "status": "running"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    if not health_checker:
        raise HTTPException(status_code=503, detail="Health checker not initialized")
    
    health_status = await health_checker.check_health()
    
    return HealthResponse(
        status="healthy" if health_status["healthy"] else "unhealthy",
        model_name=config.model.name,
        model_version=config.model.version,
        timestamp=health_status["timestamp"],
        details=health_status.get("details", {})
    )


@app.get("/ready", response_model=Dict[str, str])
async def readiness_check():
    """Readiness check endpoint"""
    if not model_manager or not model_manager.is_ready():
        raise HTTPException(status_code=503, detail="Model not ready")
    
    return {
        "status": "ready",
        "model_name": config.model.name,
        "model_version": config.model.version
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(
    request: PredictionRequest,
    background_tasks: BackgroundTasks,
    request_id: str = Depends(get_request_id)
):
    """Single prediction endpoint"""
    if not model_manager:
        raise HTTPException(status_code=503, detail="Model manager not initialized")
    
    start_time = time.time()
    
    try:
        # Perform prediction
        result = await model_manager.predict(
            request.data,
            model_version=request.model_version
        )
        
        inference_time = time.time() - start_time
        
        # Log inference
        log_model_inference(
            logger,
            config.model.name,
            config.model.version,
            inference_time,
            1,
            success=True
        )
        
        # Record metrics
        if metrics:
            confidence_scores = [pred.get("confidence", 0.0) for pred in result["predictions"]]
            metrics.record_inference(inference_time, 1, confidence_scores)
        
        # Handle shadow mode
        if request.shadow_mode and shadow_manager:
            background_tasks.add_task(
                shadow_manager.process_shadow_request,
                request.data,
                result["predictions"]
            )
        
        return PredictionResponse(
            predictions=result["predictions"],
            model_name=config.model.name,
            model_version=result["model_version"],
            request_id=request_id,
            inference_time=inference_time,
            confidence_scores=[pred.get("confidence") for pred in result["predictions"]]
        )
        
    except Exception as e:
        inference_time = time.time() - start_time
        
        # Log error
        log_model_inference(
            logger,
            config.model.name,
            config.model.version,
            inference_time,
            1,
            success=False,
            error=e
        )
        
        # Record error metrics
        if metrics:
            metrics.record_error(type(e).__name__)
        
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/batch-predict", response_model=BatchPredictionResponse)
async def batch_predict(
    request: BatchPredictionRequest,
    request_id: str = Depends(get_request_id)
):
    """Batch prediction endpoint"""
    if not model_manager:
        raise HTTPException(status_code=503, detail="Model manager not initialized")
    
    start_time = time.time()
    batch_size = request.batch_size or config.model.batch_size
    
    try:
        # Perform batch prediction
        result = await model_manager.batch_predict(
            request.batch_data,
            batch_size=batch_size,
            model_version=request.model_version
        )
        
        total_inference_time = time.time() - start_time
        actual_batch_size = len(request.batch_data)
        
        # Log inference
        log_model_inference(
            logger,
            config.model.name,
            config.model.version,
            total_inference_time,
            actual_batch_size,
            success=True
        )
        
        # Record metrics
        if metrics:
            all_confidence_scores = []
            for batch_predictions in result["predictions"]:
                for pred in batch_predictions:
                    if "confidence" in pred:
                        all_confidence_scores.append(pred["confidence"])
            
            metrics.record_inference(total_inference_time, actual_batch_size, all_confidence_scores)
        
        return BatchPredictionResponse(
            predictions=result["predictions"],
            model_name=config.model.name,
            model_version=result["model_version"],
            request_id=request_id,
            total_inference_time=total_inference_time,
            batch_size=actual_batch_size
        )
        
    except Exception as e:
        total_inference_time = time.time() - start_time
        
        # Log error
        log_model_inference(
            logger,
            config.model.name,
            config.model.version,
            total_inference_time,
            len(request.batch_data),
            success=False,
            error=e
        )
        
        # Record error metrics
        if metrics:
            metrics.record_error(type(e).__name__)
        
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")


@app.get("/metrics", response_class=PlainTextResponse)
async def get_metrics():
    """Prometheus metrics endpoint"""
    if not metrics:
        raise HTTPException(status_code=503, detail="Metrics not enabled")
    
    return metrics.get_metrics()


@app.get("/info", response_model=Dict[str, Any])
async def get_model_info():
    """Get model information"""
    if not model_manager:
        raise HTTPException(status_code=503, detail="Model manager not initialized")
    
    return await model_manager.get_model_info()


# Debug endpoints (only enabled in debug mode)
if config.features.enable_debug_endpoints:
    @app.get("/debug/config", response_model=Dict[str, Any])
    async def get_config():
        """Get current configuration (debug only)"""
        return config.dict()
    
    @app.get("/debug/endpoints", response_model=Dict[str, Any])
    async def get_endpoints_config():
        """Get model endpoints configuration (debug only)"""
        return load_model_endpoints_config()


def main():
    """Main entry point"""
    uvicorn.run(
        "src.model_serving.main:app",
        host=config.host,
        port=config.port,
        reload=config.reload,
        workers=1,  # Use 1 worker for now, can be increased
        log_config=None,  # Use our custom logging
        access_log=False  # Disable uvicorn access log, we handle it ourselves
    )


if __name__ == "__main__":
    main()

