"""
Pipeline Orchestrator for AI Model Serving Platform
Manages multi-model pipelines and workflow execution
"""
import asyncio
import json
import logging
import time
import uuid
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from enum import Enum
import httpx
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
import redis
from kubernetes import client, config
from kubernetes.client.rest import ApiException

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PipelineStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class ModelStep:
    name: str
    endpoint: str
    input_field: str
    output_field: str
    timeout: int = 60
    retry_count: int = 3


@dataclass
class PipelineConfig:
    name: str
    description: str
    steps: List[ModelStep]
    timeout: int = 300


class PipelineRequest(BaseModel):
    pipeline_name: str
    input_data: Dict[str, Any]
    priority: int = 1
    metadata: Optional[Dict[str, Any]] = None


class PipelineResponse(BaseModel):
    pipeline_id: str
    status: PipelineStatus
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    metadata: Dict[str, Any]


class PipelineOrchestrator:
    """Main pipeline orchestrator class"""
    
    def __init__(self, redis_url: str = "redis://redis:6379"):
        self.redis_client = redis.from_url(redis_url)
        self.http_client = httpx.AsyncClient(timeout=60.0)
        
        # Load Kubernetes config
        try:
            config.load_incluster_config()
        except:
            config.load_kube_config()
        
        self.k8s_client = client.CustomObjectsApi()
        
        # Pipeline configurations
        self.pipelines: Dict[str, PipelineConfig] = {}
        self._load_pipeline_configs()
        
        # Active executions
        self.active_executions: Dict[str, Dict[str, Any]] = {}
    
    def _load_pipeline_configs(self):
        """Load pipeline configurations"""
        # OCR + NER Pipeline
        ocr_ner_pipeline = PipelineConfig(
            name="ocr-ner",
            description="OCR text extraction followed by Named Entity Recognition",
            steps=[
                ModelStep(
                    name="ocr-model",
                    endpoint="http://ocr-model-service:8000",
                    input_field="image",
                    output_field="text"
                ),
                ModelStep(
                    name="bert-ner",
                    endpoint="http://bert-ner-model-service:8000",
                    input_field="text",
                    output_field="entities"
                )
            ]
        )
        
        # Image Classification + Object Detection Pipeline
        vision_pipeline = PipelineConfig(
            name="vision-analysis",
            description="Image classification followed by object detection",
            steps=[
                ModelStep(
                    name="resnet-classifier",
                    endpoint="http://resnet-classifier-service:8000",
                    input_field="image",
                    output_field="classification"
                ),
                ModelStep(
                    name="object-detector",
                    endpoint="http://object-detector-service:8000",
                    input_field="image",
                    output_field="objects"
                )
            ]
        )
        
        # Text Processing Pipeline
        text_pipeline = PipelineConfig(
            name="text-analysis",
            description="Sentiment analysis followed by topic classification",
            steps=[
                ModelStep(
                    name="sentiment-analyzer",
                    endpoint="http://sentiment-analyzer-service:8000",
                    input_field="text",
                    output_field="sentiment"
                ),
                ModelStep(
                    name="topic-classifier",
                    endpoint="http://topic-classifier-service:8000",
                    input_field="text",
                    output_field="topics"
                )
            ]
        )
        
        self.pipelines = {
            "ocr-ner": ocr_ner_pipeline,
            "vision-analysis": vision_pipeline,
            "text-analysis": text_pipeline
        }
    
    async def execute_pipeline(self, request: PipelineRequest) -> str:
        """Execute a pipeline asynchronously"""
        pipeline_id = str(uuid.uuid4())
        
        # Validate pipeline
        if request.pipeline_name not in self.pipelines:
            raise HTTPException(
                status_code=404,
                detail=f"Pipeline '{request.pipeline_name}' not found"
            )
        
        pipeline_config = self.pipelines[request.pipeline_name]
        
        # Create execution record
        execution = {
            "pipeline_id": pipeline_id,
            "pipeline_name": request.pipeline_name,
            "status": PipelineStatus.PENDING,
            "input_data": request.input_data,
            "priority": request.priority,
            "metadata": request.metadata or {},
            "created_at": time.time(),
            "started_at": None,
            "completed_at": None,
            "steps_completed": 0,
            "total_steps": len(pipeline_config.steps),
            "current_step": None,
            "result": None,
            "error": None
        }
        
        # Store in Redis
        await self._store_execution(pipeline_id, execution)
        
        # Add to active executions
        self.active_executions[pipeline_id] = execution
        
        logger.info(f"Pipeline {pipeline_id} queued for execution")
        return pipeline_id
    
    async def _execute_pipeline_steps(self, pipeline_id: str):
        """Execute pipeline steps sequentially"""
        execution = self.active_executions.get(pipeline_id)
        if not execution:
            logger.error(f"Execution {pipeline_id} not found")
            return
        
        try:
            # Update status to running
            execution["status"] = PipelineStatus.RUNNING
            execution["started_at"] = time.time()
            await self._store_execution(pipeline_id, execution)
            
            pipeline_config = self.pipelines[execution["pipeline_name"]]
            current_data = execution["input_data"]
            
            # Execute each step
            for i, step in enumerate(pipeline_config.steps):
                execution["current_step"] = step.name
                await self._store_execution(pipeline_id, execution)
                
                logger.info(f"Executing step {i+1}/{len(pipeline_config.steps)}: {step.name}")
                
                # Execute model inference
                step_result = await self._execute_model_step(step, current_data)
                
                if "error" in step_result:
                    # Step failed
                    execution["status"] = PipelineStatus.FAILED
                    execution["error"] = step_result["error"]
                    execution["completed_at"] = time.time()
                    await self._store_execution(pipeline_id, execution)
                    logger.error(f"Pipeline {pipeline_id} failed at step {step.name}: {step_result['error']}")
                    return
                
                # Update current data for next step
                current_data = {step.output_field: step_result["result"]}
                execution["steps_completed"] = i + 1
                await self._store_execution(pipeline_id, execution)
            
            # Pipeline completed successfully
            execution["status"] = PipelineStatus.COMPLETED
            execution["result"] = current_data
            execution["completed_at"] = time.time()
            execution["current_step"] = None
            await self._store_execution(pipeline_id, execution)
            
            logger.info(f"Pipeline {pipeline_id} completed successfully")
            
        except Exception as e:
            # Unexpected error
            execution["status"] = PipelineStatus.FAILED
            execution["error"] = f"Unexpected error: {str(e)}"
            execution["completed_at"] = time.time()
            await self._store_execution(pipeline_id, execution)
            logger.error(f"Pipeline {pipeline_id} failed with unexpected error: {e}")
        
        finally:
            # Remove from active executions
            if pipeline_id in self.active_executions:
                del self.active_executions[pipeline_id]
    
    async def _execute_model_step(self, step: ModelStep, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a single model inference step"""
        for attempt in range(step.retry_count):
            try:
                # Prepare request data
                request_data = {"data": {step.input_field: input_data.get(step.input_field, input_data)}}
                
                # Make inference request
                start_time = time.time()
                response = await self.http_client.post(
                    f"{step.endpoint}/predict",
                    json=request_data,
                    timeout=step.timeout
                )
                inference_time = time.time() - start_time
                
                if response.status_code == 200:
                    result = response.json()
                    
                    # Extract predictions
                    if "predictions" in result:
                        predictions = result["predictions"]
                    else:
                        predictions = result
                    
                    return {
                        "result": predictions,
                        "inference_time": inference_time,
                        "attempt": attempt + 1
                    }
                else:
                    error_msg = f"HTTP {response.status_code}: {response.text}"
                    if attempt == step.retry_count - 1:
                        return {"error": error_msg}
                    else:
                        logger.warning(f"Step {step.name} attempt {attempt + 1} failed: {error_msg}")
                        await asyncio.sleep(2 ** attempt)  # Exponential backoff
                        
            except Exception as e:
                error_msg = f"Exception: {str(e)}"
                if attempt == step.retry_count - 1:
                    return {"error": error_msg}
                else:
                    logger.warning(f"Step {step.name} attempt {attempt + 1} failed: {error_msg}")
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff
        
        return {"error": "All retry attempts failed"}
    
    async def get_pipeline_status(self, pipeline_id: str) -> Optional[Dict[str, Any]]:
        """Get pipeline execution status"""
        # Check active executions first
        if pipeline_id in self.active_executions:
            return self.active_executions[pipeline_id]
        
        # Check Redis storage
        return await self._get_execution(pipeline_id)
    
    async def cancel_pipeline(self, pipeline_id: str) -> bool:
        """Cancel a running pipeline"""
        if pipeline_id in self.active_executions:
            execution = self.active_executions[pipeline_id]
            execution["status"] = PipelineStatus.CANCELLED
            execution["completed_at"] = time.time()
            await self._store_execution(pipeline_id, execution)
            del self.active_executions[pipeline_id]
            logger.info(f"Pipeline {pipeline_id} cancelled")
            return True
        
        return False
    
    async def list_pipelines(self) -> List[Dict[str, Any]]:
        """List available pipeline configurations"""
        return [
            {
                "name": name,
                "description": config.description,
                "steps": [asdict(step) for step in config.steps],
                "timeout": config.timeout
            }
            for name, config in self.pipelines.items()
        ]
    
    async def get_pipeline_metrics(self) -> Dict[str, Any]:
        """Get pipeline execution metrics"""
        # Get recent executions from Redis
        executions = await self._get_recent_executions(limit=100)
        
        total_executions = len(executions)
        completed = sum(1 for e in executions if e.get("status") == PipelineStatus.COMPLETED)
        failed = sum(1 for e in executions if e.get("status") == PipelineStatus.FAILED)
        
        # Calculate average execution time
        completed_executions = [e for e in executions if e.get("status") == PipelineStatus.COMPLETED]
        avg_execution_time = 0
        if completed_executions:
            total_time = sum(
                e.get("completed_at", 0) - e.get("started_at", 0)
                for e in completed_executions
                if e.get("started_at") and e.get("completed_at")
            )
            avg_execution_time = total_time / len(completed_executions)
        
        return {
            "total_executions": total_executions,
            "completed": completed,
            "failed": failed,
            "success_rate": completed / total_executions if total_executions > 0 else 0,
            "active_executions": len(self.active_executions),
            "avg_execution_time": avg_execution_time,
            "pipeline_configs": len(self.pipelines)
        }
    
    async def _store_execution(self, pipeline_id: str, execution: Dict[str, Any]):
        """Store execution data in Redis"""
        try:
            self.redis_client.setex(
                f"pipeline:{pipeline_id}",
                86400,  # 24 hours TTL
                json.dumps(execution, default=str)
            )
        except Exception as e:
            logger.error(f"Failed to store execution {pipeline_id}: {e}")
    
    async def _get_execution(self, pipeline_id: str) -> Optional[Dict[str, Any]]:
        """Get execution data from Redis"""
        try:
            data = self.redis_client.get(f"pipeline:{pipeline_id}")
            if data:
                return json.loads(data)
        except Exception as e:
            logger.error(f"Failed to get execution {pipeline_id}: {e}")
        
        return None
    
    async def _get_recent_executions(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent executions from Redis"""
        try:
            keys = self.redis_client.keys("pipeline:*")
            executions = []
            
            for key in keys[:limit]:
                data = self.redis_client.get(key)
                if data:
                    execution = json.loads(data)
                    executions.append(execution)
            
            # Sort by creation time
            executions.sort(key=lambda x: x.get("created_at", 0), reverse=True)
            return executions[:limit]
            
        except Exception as e:
            logger.error(f"Failed to get recent executions: {e}")
            return []
    
    async def cleanup(self):
        """Cleanup resources"""
        await self.http_client.aclose()


# FastAPI application
app = FastAPI(
    title="Pipeline Orchestrator",
    description="Multi-model pipeline orchestration service",
    version="1.0.0"
)

# Global orchestrator instance
orchestrator = PipelineOrchestrator()


@app.on_event("startup")
async def startup_event():
    """Application startup"""
    logger.info("Pipeline Orchestrator starting up...")


@app.on_event("shutdown")
async def shutdown_event():
    """Application shutdown"""
    logger.info("Pipeline Orchestrator shutting down...")
    await orchestrator.cleanup()


@app.post("/pipelines/execute", response_model=Dict[str, str])
async def execute_pipeline(request: PipelineRequest, background_tasks: BackgroundTasks):
    """Execute a pipeline"""
    pipeline_id = await orchestrator.execute_pipeline(request)
    
    # Start execution in background
    background_tasks.add_task(orchestrator._execute_pipeline_steps, pipeline_id)
    
    return {"pipeline_id": pipeline_id, "status": "queued"}


@app.get("/pipelines/{pipeline_id}/status", response_model=PipelineResponse)
async def get_pipeline_status(pipeline_id: str):
    """Get pipeline execution status"""
    execution = await orchestrator.get_pipeline_status(pipeline_id)
    
    if not execution:
        raise HTTPException(status_code=404, detail="Pipeline not found")
    
    return PipelineResponse(
        pipeline_id=execution["pipeline_id"],
        status=execution["status"],
        result=execution.get("result"),
        error=execution.get("error"),
        metadata={
            "created_at": execution.get("created_at"),
            "started_at": execution.get("started_at"),
            "completed_at": execution.get("completed_at"),
            "steps_completed": execution.get("steps_completed", 0),
            "total_steps": execution.get("total_steps", 0),
            "current_step": execution.get("current_step")
        }
    )


@app.delete("/pipelines/{pipeline_id}")
async def cancel_pipeline(pipeline_id: str):
    """Cancel a running pipeline"""
    success = await orchestrator.cancel_pipeline(pipeline_id)
    
    if not success:
        raise HTTPException(status_code=404, detail="Pipeline not found or not running")
    
    return {"message": "Pipeline cancelled"}


@app.get("/pipelines")
async def list_pipelines():
    """List available pipeline configurations"""
    return await orchestrator.list_pipelines()


@app.get("/metrics")
async def get_metrics():
    """Get pipeline execution metrics"""
    return await orchestrator.get_pipeline_metrics()


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "pipeline-orchestrator"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

