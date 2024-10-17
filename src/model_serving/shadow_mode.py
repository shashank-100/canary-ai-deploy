"""
Shadow Mode Manager for AI Model Serving Platform
Handles A/B testing and shadow deployments
"""
import asyncio
import time
import json
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import aioredis
import aiofiles

from ..common.config import config
from ..common.logging_config import get_logger


@dataclass
class ShadowRequest:
    """Shadow request data structure"""
    request_id: str
    timestamp: datetime
    input_data: Dict[str, Any]
    production_result: List[Dict[str, Any]]
    shadow_result: Optional[List[Dict[str, Any]]] = None
    comparison_metrics: Optional[Dict[str, Any]] = None
    processing_time_ms: Optional[float] = None


@dataclass
class ShadowMetrics:
    """Shadow mode metrics"""
    total_requests: int = 0
    successful_comparisons: int = 0
    failed_comparisons: int = 0
    accuracy_matches: int = 0
    confidence_differences: List[float] = None
    latency_differences: List[float] = None
    error_rate: float = 0.0
    
    def __post_init__(self):
        if self.confidence_differences is None:
            self.confidence_differences = []
        if self.latency_differences is None:
            self.latency_differences = []


class ShadowModeManager:
    """Manages shadow mode deployments and A/B testing"""
    
    def __init__(self, model_manager):
        self.model_manager = model_manager
        self.logger = get_logger(__name__)
        
        # Shadow model configuration
        self.shadow_model_name = f"{config.model.name}-shadow"
        self.shadow_model_version = "v2.0.0"  # Shadow version
        
        # Redis client for storing shadow results
        self.redis_client: Optional[aioredis.Redis] = None
        
        # Shadow mode configuration
        self.shadow_percentage = 5.0  # Percentage of traffic to shadow
        self.comparison_enabled = True
        self.store_results = True
        
        # Metrics
        self.metrics = ShadowMetrics()
        
        # Background tasks
        self._background_tasks = set()
        self._cleanup_task: Optional[asyncio.Task] = None
        
        # Comparison functions
        self.comparison_functions = {
            "bert-ner": self._compare_ner_results,
            "resnet-classifier": self._compare_classification_results
        }
    
    async def initialize(self) -> None:
        """Initialize shadow mode manager"""
        try:
            self.logger.info("Initializing shadow mode manager")
            
            # Initialize Redis connection
            if config.redis.url:
                self.redis_client = aioredis.from_url(
                    config.redis.url,
                    max_connections=config.redis.max_connections,
                    retry_on_timeout=config.redis.retry_on_timeout,
                    socket_timeout=config.redis.socket_timeout
                )
                await self.redis_client.ping()
                self.logger.info("Redis connection established for shadow mode")
            
            # Start background cleanup task
            self._cleanup_task = asyncio.create_task(self._cleanup_old_results())
            
            self.logger.info("Shadow mode manager initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize shadow mode manager: {e}", exc_info=True)
            raise
    
    async def process_shadow_request(self, input_data: Dict[str, Any], 
                                   production_result: List[Dict[str, Any]]) -> None:
        """Process a shadow request asynchronously"""
        request_id = f"shadow_{int(time.time() * 1000)}"
        
        # Create shadow request
        shadow_request = ShadowRequest(
            request_id=request_id,
            timestamp=datetime.utcnow(),
            input_data=input_data,
            production_result=production_result
        )
        
        # Process in background
        task = asyncio.create_task(self._process_shadow_request_async(shadow_request))
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)
    
    async def _process_shadow_request_async(self, shadow_request: ShadowRequest) -> None:
        """Process shadow request asynchronously"""
        try:
            start_time = time.time()
            
            # Make shadow prediction (simulate with production model for now)
            # In a real implementation, this would use a different model version
            shadow_result = await self._make_shadow_prediction(shadow_request.input_data)
            
            processing_time = (time.time() - start_time) * 1000  # Convert to ms
            
            # Update shadow request
            shadow_request.shadow_result = shadow_result
            shadow_request.processing_time_ms = processing_time
            
            # Compare results
            if self.comparison_enabled:
                comparison_metrics = await self._compare_results(
                    shadow_request.production_result,
                    shadow_request.shadow_result
                )
                shadow_request.comparison_metrics = comparison_metrics
                
                # Update metrics
                await self._update_metrics(comparison_metrics, processing_time)
            
            # Store results
            if self.store_results and self.redis_client:
                await self._store_shadow_result(shadow_request)
            
            self.logger.debug(f"Shadow request {shadow_request.request_id} processed successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to process shadow request {shadow_request.request_id}: {e}", exc_info=True)
            self.metrics.failed_comparisons += 1
    
    async def _make_shadow_prediction(self, input_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Make prediction using shadow model"""
        # For demonstration, we'll simulate a shadow model by adding some variation
        # In a real implementation, this would use a different model version
        
        try:
            # Use the same model but simulate different behavior
            result = await self.model_manager.predict(input_data)
            shadow_result = result["predictions"]
            
            # Simulate shadow model differences
            for prediction in shadow_result:
                if "confidence" in prediction:
                    # Add small random variation to confidence
                    import random
                    variation = random.uniform(-0.05, 0.05)
                    prediction["confidence"] = max(0.0, min(1.0, prediction["confidence"] + variation))
                    prediction["shadow_model"] = True
            
            return shadow_result
            
        except Exception as e:
            self.logger.error(f"Shadow prediction failed: {e}", exc_info=True)
            raise
    
    async def _compare_results(self, production_result: List[Dict[str, Any]], 
                             shadow_result: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Compare production and shadow results"""
        model_name = config.model.name
        
        if model_name in self.comparison_functions:
            return await self.comparison_functions[model_name](production_result, shadow_result)
        else:
            return await self._generic_comparison(production_result, shadow_result)
    
    async def _compare_ner_results(self, production_result: List[Dict[str, Any]], 
                                 shadow_result: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Compare NER results"""
        metrics = {
            "entity_count_match": len(production_result) == len(shadow_result),
            "entity_count_diff": abs(len(production_result) - len(shadow_result)),
            "label_matches": 0,
            "confidence_differences": [],
            "exact_match": False
        }
        
        # Compare entities
        prod_labels = [entity.get("label", "") for entity in production_result]
        shadow_labels = [entity.get("label", "") for entity in shadow_result]
        
        # Count label matches
        for prod_label, shadow_label in zip(prod_labels, shadow_labels):
            if prod_label == shadow_label:
                metrics["label_matches"] += 1
        
        # Compare confidences
        for prod_entity, shadow_entity in zip(production_result, shadow_result):
            prod_conf = prod_entity.get("confidence", 0.0)
            shadow_conf = shadow_entity.get("confidence", 0.0)
            metrics["confidence_differences"].append(abs(prod_conf - shadow_conf))
        
        # Check exact match
        metrics["exact_match"] = (
            len(production_result) == len(shadow_result) and
            metrics["label_matches"] == len(production_result)
        )
        
        return metrics
    
    async def _compare_classification_results(self, production_result: List[Dict[str, Any]], 
                                            shadow_result: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Compare classification results"""
        metrics = {
            "top1_match": False,
            "top5_overlap": 0,
            "confidence_differences": [],
            "class_differences": []
        }
        
        if production_result and shadow_result:
            # Compare top-1 predictions
            prod_top1 = production_result[0].get("class", "")
            shadow_top1 = shadow_result[0].get("class", "")
            metrics["top1_match"] = prod_top1 == shadow_top1
            
            # Compare top-5 overlap
            prod_classes = set(pred.get("class", "") for pred in production_result[:5])
            shadow_classes = set(pred.get("class", "") for pred in shadow_result[:5])
            metrics["top5_overlap"] = len(prod_classes.intersection(shadow_classes))
            
            # Compare confidences
            for prod_pred, shadow_pred in zip(production_result, shadow_result):
                prod_conf = prod_pred.get("confidence", 0.0)
                shadow_conf = shadow_pred.get("confidence", 0.0)
                metrics["confidence_differences"].append(abs(prod_conf - shadow_conf))
                
                prod_class = prod_pred.get("class", "")
                shadow_class = shadow_pred.get("class", "")
                if prod_class != shadow_class:
                    metrics["class_differences"].append({
                        "production": prod_class,
                        "shadow": shadow_class
                    })
        
        return metrics
    
    async def _generic_comparison(self, production_result: List[Dict[str, Any]], 
                                shadow_result: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generic comparison for unknown model types"""
        return {
            "result_count_match": len(production_result) == len(shadow_result),
            "result_count_diff": abs(len(production_result) - len(shadow_result)),
            "exact_match": production_result == shadow_result
        }
    
    async def _update_metrics(self, comparison_metrics: Dict[str, Any], processing_time: float) -> None:
        """Update shadow mode metrics"""
        self.metrics.total_requests += 1
        self.metrics.successful_comparisons += 1
        
        # Update accuracy metrics based on model type
        if config.model.name == "bert-ner":
            if comparison_metrics.get("exact_match", False):
                self.metrics.accuracy_matches += 1
        elif config.model.name == "resnet-classifier":
            if comparison_metrics.get("top1_match", False):
                self.metrics.accuracy_matches += 1
        
        # Update confidence differences
        conf_diffs = comparison_metrics.get("confidence_differences", [])
        if conf_diffs:
            self.metrics.confidence_differences.extend(conf_diffs)
        
        # Update latency differences (assuming production latency is baseline)
        # This is a simplified calculation
        baseline_latency = 100.0  # ms
        latency_diff = processing_time - baseline_latency
        self.metrics.latency_differences.append(latency_diff)
        
        # Calculate error rate
        if self.metrics.total_requests > 0:
            self.metrics.error_rate = self.metrics.failed_comparisons / self.metrics.total_requests
    
    async def _store_shadow_result(self, shadow_request: ShadowRequest) -> None:
        """Store shadow result in Redis"""
        try:
            if not self.redis_client:
                return
            
            # Convert to JSON
            data = asdict(shadow_request)
            data["timestamp"] = shadow_request.timestamp.isoformat()
            
            # Store with expiration (7 days)
            key = f"shadow_result:{shadow_request.request_id}"
            await self.redis_client.setex(
                key,
                timedelta(days=7).total_seconds(),
                json.dumps(data, default=str)
            )
            
            # Add to index for querying
            index_key = f"shadow_index:{datetime.utcnow().strftime('%Y-%m-%d')}"
            await self.redis_client.sadd(index_key, shadow_request.request_id)
            await self.redis_client.expire(index_key, timedelta(days=7).total_seconds())
            
        except Exception as e:
            self.logger.error(f"Failed to store shadow result: {e}", exc_info=True)
    
    async def _cleanup_old_results(self) -> None:
        """Background task to cleanup old shadow results"""
        while True:
            try:
                await asyncio.sleep(3600)  # Run every hour
                
                if not self.redis_client:
                    continue
                
                # Cleanup results older than 7 days
                cutoff_date = datetime.utcnow() - timedelta(days=7)
                cutoff_key = f"shadow_index:{cutoff_date.strftime('%Y-%m-%d')}"
                
                # Get old request IDs
                old_request_ids = await self.redis_client.smembers(cutoff_key)
                
                if old_request_ids:
                    # Delete old results
                    keys_to_delete = [f"shadow_result:{req_id.decode()}" for req_id in old_request_ids]
                    keys_to_delete.append(cutoff_key)
                    
                    await self.redis_client.delete(*keys_to_delete)
                    self.logger.info(f"Cleaned up {len(old_request_ids)} old shadow results")
                
            except Exception as e:
                self.logger.error(f"Error during shadow results cleanup: {e}", exc_info=True)
    
    async def get_shadow_metrics(self) -> Dict[str, Any]:
        """Get current shadow mode metrics"""
        metrics_dict = asdict(self.metrics)
        
        # Calculate additional statistics
        if self.metrics.confidence_differences:
            metrics_dict["avg_confidence_diff"] = sum(self.metrics.confidence_differences) / len(self.metrics.confidence_differences)
            metrics_dict["max_confidence_diff"] = max(self.metrics.confidence_differences)
        
        if self.metrics.latency_differences:
            metrics_dict["avg_latency_diff"] = sum(self.metrics.latency_differences) / len(self.metrics.latency_differences)
            metrics_dict["max_latency_diff"] = max(self.metrics.latency_differences)
        
        if self.metrics.total_requests > 0:
            metrics_dict["accuracy_rate"] = self.metrics.accuracy_matches / self.metrics.total_requests
            metrics_dict["success_rate"] = self.metrics.successful_comparisons / self.metrics.total_requests
        
        return metrics_dict
    
    async def get_shadow_results(self, date: Optional[str] = None, limit: int = 100) -> List[Dict[str, Any]]:
        """Get shadow results for a specific date"""
        if not self.redis_client:
            return []
        
        try:
            # Use today's date if not specified
            if not date:
                date = datetime.utcnow().strftime('%Y-%m-%d')
            
            # Get request IDs for the date
            index_key = f"shadow_index:{date}"
            request_ids = await self.redis_client.smembers(index_key)
            
            # Limit results
            request_ids = list(request_ids)[:limit]
            
            # Get shadow results
            results = []
            for request_id in request_ids:
                key = f"shadow_result:{request_id.decode()}"
                data = await self.redis_client.get(key)
                if data:
                    results.append(json.loads(data))
            
            return results
            
        except Exception as e:
            self.logger.error(f"Failed to get shadow results: {e}", exc_info=True)
            return []
    
    async def cleanup(self) -> None:
        """Cleanup shadow mode manager"""
        try:
            self.logger.info("Cleaning up shadow mode manager")
            
            # Cancel background tasks
            if self._cleanup_task:
                self._cleanup_task.cancel()
                try:
                    await self._cleanup_task
                except asyncio.CancelledError:
                    pass
            
            # Cancel all background tasks
            for task in self._background_tasks:
                task.cancel()
            
            # Wait for tasks to complete
            if self._background_tasks:
                await asyncio.gather(*self._background_tasks, return_exceptions=True)
            
            # Close Redis connection
            if self.redis_client:
                await self.redis_client.close()
            
            self.logger.info("Shadow mode manager cleanup completed")
            
        except Exception as e:
            self.logger.error(f"Error during shadow mode cleanup: {e}", exc_info=True)

