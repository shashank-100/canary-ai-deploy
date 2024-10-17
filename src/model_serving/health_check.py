"""
Health Check module for AI Model Serving Platform
Monitors model health and system resources
"""
import asyncio
import time
import psutil
import torch
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from dataclasses import dataclass

from ..common.config import config
from ..common.logging_config import get_logger


@dataclass
class HealthStatus:
    """Health status data structure"""
    healthy: bool
    timestamp: str
    details: Dict[str, Any]
    checks: Dict[str, bool]


class HealthChecker:
    """Health checker for model serving"""
    
    def __init__(self, model_manager):
        self.model_manager = model_manager
        self.logger = get_logger(__name__)
        
        # Health check configuration
        self.check_interval = config.features.health_check_interval if hasattr(config.features, 'health_check_interval') else 30
        self.memory_threshold = 0.9  # 90% memory usage threshold
        self.cpu_threshold = 0.95    # 95% CPU usage threshold
        self.gpu_memory_threshold = 0.9  # 90% GPU memory threshold
        
        # Health history
        self.health_history: List[HealthStatus] = []
        self.max_history_size = 100
        
        # Background task
        self._health_check_task: Optional[asyncio.Task] = None
        self._running = False
        
        # Current health status
        self._current_status = HealthStatus(
            healthy=True,
            timestamp=datetime.utcnow().isoformat(),
            details={},
            checks={}
        )
    
    async def start(self) -> None:
        """Start health checking"""
        try:
            self.logger.info("Starting health checker")
            self._running = True
            self._health_check_task = asyncio.create_task(self._health_check_loop())
            self.logger.info("Health checker started")
        except Exception as e:
            self.logger.error(f"Failed to start health checker: {e}", exc_info=True)
            raise
    
    async def stop(self) -> None:
        """Stop health checking"""
        try:
            self.logger.info("Stopping health checker")
            self._running = False
            
            if self._health_check_task:
                self._health_check_task.cancel()
                try:
                    await self._health_check_task
                except asyncio.CancelledError:
                    pass
            
            self.logger.info("Health checker stopped")
        except Exception as e:
            self.logger.error(f"Error stopping health checker: {e}", exc_info=True)
    
    async def _health_check_loop(self) -> None:
        """Main health check loop"""
        while self._running:
            try:
                await self._perform_health_check()
                await asyncio.sleep(self.check_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in health check loop: {e}", exc_info=True)
                await asyncio.sleep(self.check_interval)
    
    async def _perform_health_check(self) -> None:
        """Perform comprehensive health check"""
        try:
            checks = {}
            details = {}
            
            # Check model status
            model_healthy, model_details = await self._check_model_health()
            checks["model"] = model_healthy
            details["model"] = model_details
            
            # Check system resources
            system_healthy, system_details = await self._check_system_health()
            checks["system"] = system_healthy
            details["system"] = system_details
            
            # Check GPU resources (if available)
            if torch.cuda.is_available():
                gpu_healthy, gpu_details = await self._check_gpu_health()
                checks["gpu"] = gpu_healthy
                details["gpu"] = gpu_details
            
            # Check dependencies
            deps_healthy, deps_details = await self._check_dependencies()
            checks["dependencies"] = deps_healthy
            details["dependencies"] = deps_details
            
            # Overall health status
            overall_healthy = all(checks.values())
            
            # Create health status
            status = HealthStatus(
                healthy=overall_healthy,
                timestamp=datetime.utcnow().isoformat(),
                details=details,
                checks=checks
            )
            
            # Update current status
            self._current_status = status
            
            # Add to history
            self._add_to_history(status)
            
            # Log health status changes
            if not overall_healthy:
                failed_checks = [check for check, result in checks.items() if not result]
                self.logger.warning(f"Health check failed: {failed_checks}")
            
        except Exception as e:
            self.logger.error(f"Health check failed: {e}", exc_info=True)
            
            # Create unhealthy status
            self._current_status = HealthStatus(
                healthy=False,
                timestamp=datetime.utcnow().isoformat(),
                details={"error": str(e)},
                checks={"health_check": False}
            )
    
    async def _check_model_health(self) -> tuple[bool, Dict[str, Any]]:
        """Check model health"""
        details = {}
        
        try:
            # Check if model manager is ready
            if not self.model_manager.is_ready():
                details["status"] = "not_ready"
                details["error"] = "Model manager not ready"
                return False, details
            
            # Get model info
            model_info = await self.model_manager.get_model_info()
            details.update(model_info)
            
            # Perform a simple inference test
            test_successful = await self._test_model_inference()
            details["inference_test"] = test_successful
            
            if not test_successful:
                details["error"] = "Model inference test failed"
                return False, details
            
            details["status"] = "healthy"
            return True, details
            
        except Exception as e:
            details["status"] = "error"
            details["error"] = str(e)
            return False, details
    
    async def _test_model_inference(self) -> bool:
        """Test model inference with dummy data"""
        try:
            # Create test data based on model type
            if config.model.name == "bert-ner":
                test_data = {"text": "Hello world"}
            elif config.model.name == "resnet-classifier":
                # Create a small dummy base64 image
                import base64
                from PIL import Image
                import io
                
                # Create a small test image
                img = Image.new('RGB', (32, 32), color='red')
                buffer = io.BytesIO()
                img.save(buffer, format='JPEG')
                img_str = base64.b64encode(buffer.getvalue()).decode()
                test_data = {"image": img_str}
            else:
                test_data = {"test": "data"}
            
            # Perform inference with timeout
            result = await asyncio.wait_for(
                self.model_manager.predict(test_data),
                timeout=10.0
            )
            
            return result is not None and "predictions" in result
            
        except Exception as e:
            self.logger.debug(f"Model inference test failed: {e}")
            return False
    
    async def _check_system_health(self) -> tuple[bool, Dict[str, Any]]:
        """Check system resource health"""
        details = {}
        healthy = True
        
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            details["cpu_percent"] = cpu_percent
            if cpu_percent > self.cpu_threshold * 100:
                healthy = False
                details["cpu_warning"] = f"High CPU usage: {cpu_percent}%"
            
            # Memory usage
            memory = psutil.virtual_memory()
            details["memory_percent"] = memory.percent
            details["memory_available_gb"] = memory.available / (1024**3)
            details["memory_total_gb"] = memory.total / (1024**3)
            
            if memory.percent > self.memory_threshold * 100:
                healthy = False
                details["memory_warning"] = f"High memory usage: {memory.percent}%"
            
            # Disk usage
            disk = psutil.disk_usage('/')
            details["disk_percent"] = (disk.used / disk.total) * 100
            details["disk_free_gb"] = disk.free / (1024**3)
            details["disk_total_gb"] = disk.total / (1024**3)
            
            if (disk.used / disk.total) > 0.9:  # 90% disk usage
                healthy = False
                details["disk_warning"] = f"High disk usage: {(disk.used / disk.total) * 100:.1f}%"
            
            # Load average (Unix systems)
            try:
                load_avg = psutil.getloadavg()
                details["load_average"] = {
                    "1min": load_avg[0],
                    "5min": load_avg[1],
                    "15min": load_avg[2]
                }
            except AttributeError:
                # getloadavg not available on Windows
                pass
            
            # Process information
            process = psutil.Process()
            details["process"] = {
                "pid": process.pid,
                "memory_mb": process.memory_info().rss / (1024**2),
                "cpu_percent": process.cpu_percent(),
                "num_threads": process.num_threads(),
                "create_time": process.create_time()
            }
            
            return healthy, details
            
        except Exception as e:
            details["error"] = str(e)
            return False, details
    
    async def _check_gpu_health(self) -> tuple[bool, Dict[str, Any]]:
        """Check GPU health"""
        details = {}
        healthy = True
        
        try:
            if not torch.cuda.is_available():
                details["status"] = "not_available"
                return True, details  # Not having GPU is not unhealthy
            
            # GPU count and current device
            gpu_count = torch.cuda.device_count()
            current_device = torch.cuda.current_device()
            
            details["gpu_count"] = gpu_count
            details["current_device"] = current_device
            details["gpus"] = {}
            
            # Check each GPU
            for i in range(gpu_count):
                gpu_details = {}
                
                # GPU properties
                props = torch.cuda.get_device_properties(i)
                gpu_details["name"] = props.name
                gpu_details["total_memory_gb"] = props.total_memory / (1024**3)
                
                # Memory usage
                torch.cuda.set_device(i)
                memory_allocated = torch.cuda.memory_allocated(i) / (1024**3)
                memory_reserved = torch.cuda.memory_reserved(i) / (1024**3)
                memory_total = props.total_memory / (1024**3)
                
                gpu_details["memory_allocated_gb"] = memory_allocated
                gpu_details["memory_reserved_gb"] = memory_reserved
                gpu_details["memory_utilization"] = memory_allocated / memory_total
                
                # Check memory threshold
                if memory_allocated / memory_total > self.gpu_memory_threshold:
                    healthy = False
                    gpu_details["warning"] = f"High GPU memory usage: {(memory_allocated / memory_total) * 100:.1f}%"
                
                # GPU utilization (if nvidia-ml-py is available)
                try:
                    import pynvml
                    pynvml.nvmlInit()
                    handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                    utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    gpu_details["gpu_utilization"] = utilization.gpu
                    gpu_details["memory_utilization_nvml"] = utilization.memory
                    
                    # Temperature
                    temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                    gpu_details["temperature_c"] = temp
                    
                except ImportError:
                    gpu_details["nvml_available"] = False
                except Exception as e:
                    gpu_details["nvml_error"] = str(e)
                
                details["gpus"][f"gpu_{i}"] = gpu_details
            
            return healthy, details
            
        except Exception as e:
            details["error"] = str(e)
            return False, details
    
    async def _check_dependencies(self) -> tuple[bool, Dict[str, Any]]:
        """Check external dependencies"""
        details = {}
        healthy = True
        
        try:
            # Check Redis connection (if configured)
            if config.redis.url:
                redis_healthy = await self._check_redis_connection()
                details["redis"] = {"healthy": redis_healthy}
                if not redis_healthy:
                    healthy = False
            
            # Check database connection (if configured)
            if config.database.url:
                db_healthy = await self._check_database_connection()
                details["database"] = {"healthy": db_healthy}
                if not db_healthy:
                    healthy = False
            
            # Check S3 connection (if configured)
            if config.s3.bucket_name:
                s3_healthy = await self._check_s3_connection()
                details["s3"] = {"healthy": s3_healthy}
                if not s3_healthy:
                    healthy = False
            
            return healthy, details
            
        except Exception as e:
            details["error"] = str(e)
            return False, details
    
    async def _check_redis_connection(self) -> bool:
        """Check Redis connection"""
        try:
            import aioredis
            redis = aioredis.from_url(config.redis.url, socket_timeout=5)
            await redis.ping()
            await redis.close()
            return True
        except Exception:
            return False
    
    async def _check_database_connection(self) -> bool:
        """Check database connection"""
        try:
            # This is a simplified check - in production, use proper connection pooling
            import asyncpg
            conn = await asyncpg.connect(config.database.url, timeout=5)
            await conn.execute("SELECT 1")
            await conn.close()
            return True
        except Exception:
            return False
    
    async def _check_s3_connection(self) -> bool:
        """Check S3 connection"""
        try:
            import boto3
            from botocore.exceptions import ClientError
            
            s3_client = boto3.client(
                's3',
                region_name=config.s3.region,
                aws_access_key_id=config.s3.access_key_id,
                aws_secret_access_key=config.s3.secret_access_key
            )
            
            # Try to list objects (with timeout)
            s3_client.head_bucket(Bucket=config.s3.bucket_name)
            return True
        except Exception:
            return False
    
    def _add_to_history(self, status: HealthStatus) -> None:
        """Add health status to history"""
        self.health_history.append(status)
        
        # Keep only recent history
        if len(self.health_history) > self.max_history_size:
            self.health_history = self.health_history[-self.max_history_size:]
    
    async def check_health(self) -> Dict[str, Any]:
        """Get current health status"""
        return {
            "healthy": self._current_status.healthy,
            "timestamp": self._current_status.timestamp,
            "details": self._current_status.details,
            "checks": self._current_status.checks
        }
    
    async def get_health_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get health history"""
        recent_history = self.health_history[-limit:] if limit > 0 else self.health_history
        return [
            {
                "healthy": status.healthy,
                "timestamp": status.timestamp,
                "checks": status.checks
            }
            for status in recent_history
        ]
    
    async def get_health_summary(self) -> Dict[str, Any]:
        """Get health summary statistics"""
        if not self.health_history:
            return {"status": "no_data"}
        
        # Calculate uptime percentage
        total_checks = len(self.health_history)
        healthy_checks = sum(1 for status in self.health_history if status.healthy)
        uptime_percentage = (healthy_checks / total_checks) * 100 if total_checks > 0 else 0
        
        # Get recent status
        recent_statuses = self.health_history[-10:]  # Last 10 checks
        recent_healthy = sum(1 for status in recent_statuses if status.healthy)
        recent_uptime = (recent_healthy / len(recent_statuses)) * 100 if recent_statuses else 0
        
        return {
            "overall_uptime_percentage": uptime_percentage,
            "recent_uptime_percentage": recent_uptime,
            "total_checks": total_checks,
            "healthy_checks": healthy_checks,
            "current_status": self._current_status.healthy,
            "last_check": self._current_status.timestamp
        }

