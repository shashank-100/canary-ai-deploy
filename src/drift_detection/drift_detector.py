# Drift Detection Module
# Handles data drift detection for ML models
"""
Data Drift Detection module for AI Model Serving Platform
Monitors input data distribution shifts and model performance drift
"""
import asyncio
import time
import json
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from abc import ABC, abstractmethod
import yaml
import aiofiles
import aioredis
import boto3
from scipy import stats
from sklearn.metrics import wasserstein_distance
import logging

from ..common.config import config
from ..common.logging_config import get_logger


@dataclass
class DriftResult:
    """Drift detection result"""
    model_name: str
    timestamp: datetime
    drift_detected: bool
    drift_score: float
    drift_method: str
    threshold: float
    feature_name: str
    reference_period: str
    detection_period: str
    details: Dict[str, Any]


@dataclass
class DriftAlert:
    """Drift alert data structure"""
    model_name: str
    timestamp: datetime
    alert_type: str
    severity: str
    message: str
    drift_results: List[DriftResult]
    recommended_actions: List[str]


class BaseDriftDetector(ABC):
    """Abstract base class for drift detectors"""
    
    def __init__(self, name: str, threshold: float = 0.05):
        self.name = name
        self.threshold = threshold
        self.logger = get_logger(f"drift.{name}")
    
    @abstractmethod
    async def detect_drift(self, reference_data: np.ndarray, 
                          detection_data: np.ndarray) -> Tuple[bool, float, Dict[str, Any]]:
        """Detect drift between reference and detection data"""
        pass


class KolmogorovSmirnovDetector(BaseDriftDetector):
    """Kolmogorov-Smirnov test for drift detection"""
    
    def __init__(self, threshold: float = 0.05):
        super().__init__("kolmogorov_smirnov", threshold)
    
    async def detect_drift(self, reference_data: np.ndarray, 
                          detection_data: np.ndarray) -> Tuple[bool, float, Dict[str, Any]]:
        """Perform KS test for drift detection"""
        try:
            # Perform KS test
            ks_statistic, p_value = stats.ks_2samp(reference_data, detection_data)
            
            # Drift detected if p-value is below threshold
            drift_detected = p_value < self.threshold
            
            details = {
                "ks_statistic": float(ks_statistic),
                "p_value": float(p_value),
                "reference_size": len(reference_data),
                "detection_size": len(detection_data),
                "reference_mean": float(np.mean(reference_data)),
                "detection_mean": float(np.mean(detection_data)),
                "reference_std": float(np.std(reference_data)),
                "detection_std": float(np.std(detection_data))
            }
            
            return drift_detected, float(p_value), details
            
        except Exception as e:
            self.logger.error(f"KS test failed: {e}", exc_info=True)
            return False, 1.0, {"error": str(e)}


class PopulationStabilityIndexDetector(BaseDriftDetector):
    """Population Stability Index (PSI) for drift detection"""
    
    def __init__(self, threshold: float = 0.1, bins: int = 10):
        super().__init__("population_stability_index", threshold)
        self.bins = bins
    
    async def detect_drift(self, reference_data: np.ndarray, 
                          detection_data: np.ndarray) -> Tuple[bool, float, Dict[str, Any]]:
        """Calculate PSI for drift detection"""
        try:
            # Create bins based on reference data
            _, bin_edges = np.histogram(reference_data, bins=self.bins)
            
            # Calculate distributions
            ref_counts, _ = np.histogram(reference_data, bins=bin_edges)
            det_counts, _ = np.histogram(detection_data, bins=bin_edges)
            
            # Convert to proportions
            ref_props = ref_counts / len(reference_data)
            det_props = det_counts / len(detection_data)
            
            # Avoid division by zero
            ref_props = np.where(ref_props == 0, 0.0001, ref_props)
            det_props = np.where(det_props == 0, 0.0001, det_props)
            
            # Calculate PSI
            psi = np.sum((det_props - ref_props) * np.log(det_props / ref_props))
            
            # Drift detected if PSI is above threshold
            drift_detected = psi > self.threshold
            
            details = {
                "psi_score": float(psi),
                "bins": self.bins,
                "reference_distribution": ref_props.tolist(),
                "detection_distribution": det_props.tolist(),
                "bin_edges": bin_edges.tolist()
            }
            
            return drift_detected, float(psi), details
            
        except Exception as e:
            self.logger.error(f"PSI calculation failed: {e}", exc_info=True)
            return False, 0.0, {"error": str(e)}


class WassersteinDistanceDetector(BaseDriftDetector):
    """Wasserstein distance for drift detection"""
    
    def __init__(self, threshold: float = 0.1):
        super().__init__("wasserstein_distance", threshold)
    
    async def detect_drift(self, reference_data: np.ndarray, 
                          detection_data: np.ndarray) -> Tuple[bool, float, Dict[str, Any]]:
        """Calculate Wasserstein distance for drift detection"""
        try:
            # Calculate Wasserstein distance
            distance = wasserstein_distance(reference_data, detection_data)
            
            # Drift detected if distance is above threshold
            drift_detected = distance > self.threshold
            
            details = {
                "wasserstein_distance": float(distance),
                "reference_size": len(reference_data),
                "detection_size": len(detection_data),
                "reference_quantiles": np.percentile(reference_data, [25, 50, 75]).tolist(),
                "detection_quantiles": np.percentile(detection_data, [25, 50, 75]).tolist()
            }
            
            return drift_detected, float(distance), details
            
        except Exception as e:
            self.logger.error(f"Wasserstein distance calculation failed: {e}", exc_info=True)
            return False, 0.0, {"error": str(e)}


class MaximumMeanDiscrepancyDetector(BaseDriftDetector):
    """Maximum Mean Discrepancy (MMD) for drift detection"""
    
    def __init__(self, threshold: float = 0.05, gamma: float = 1.0):
        super().__init__("maximum_mean_discrepancy", threshold)
        self.gamma = gamma
    
    def _rbf_kernel(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """RBF kernel for MMD calculation"""
        X = X.reshape(-1, 1) if X.ndim == 1 else X
        Y = Y.reshape(-1, 1) if Y.ndim == 1 else Y
        
        # Calculate pairwise squared distances
        XX = np.sum(X**2, axis=1, keepdims=True)
        YY = np.sum(Y**2, axis=1, keepdims=True)
        XY = np.dot(X, Y.T)
        
        distances = XX - 2*XY + YY.T
        
        # Apply RBF kernel
        return np.exp(-self.gamma * distances)
    
    async def detect_drift(self, reference_data: np.ndarray, 
                          detection_data: np.ndarray) -> Tuple[bool, float, Dict[str, Any]]:
        """Calculate MMD for drift detection"""
        try:
            # Calculate kernel matrices
            K_XX = self._rbf_kernel(reference_data, reference_data)
            K_YY = self._rbf_kernel(detection_data, detection_data)
            K_XY = self._rbf_kernel(reference_data, detection_data)
            
            # Calculate MMD
            m, n = len(reference_data), len(detection_data)
            mmd = (np.sum(K_XX) / (m * m) + 
                   np.sum(K_YY) / (n * n) - 
                   2 * np.sum(K_XY) / (m * n))
            
            # Drift detected if MMD is above threshold
            drift_detected = mmd > self.threshold
            
            details = {
                "mmd_score": float(mmd),
                "gamma": self.gamma,
                "reference_size": m,
                "detection_size": n
            }
            
            return drift_detected, float(mmd), details
            
        except Exception as e:
            self.logger.error(f"MMD calculation failed: {e}", exc_info=True)
            return False, 0.0, {"error": str(e)}


class FeatureExtractor:
    """Extract features from model inputs for drift detection"""
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.logger = get_logger(f"feature_extractor.{model_name}")
    
    async def extract_features(self, data: List[Dict[str, Any]]) -> Dict[str, np.ndarray]:
        """Extract features from input data"""
        if self.model_name == "bert-ner":
            return await self._extract_text_features(data)
        elif self.model_name == "resnet-classifier":
            return await self._extract_image_features(data)
        else:
            return await self._extract_generic_features(data)
    
    async def _extract_text_features(self, data: List[Dict[str, Any]]) -> Dict[str, np.ndarray]:
        """Extract features from text data"""
        features = {
            "text_length": [],
            "token_count": [],
            "special_chars_ratio": []
        }
        
        for item in data:
            text = item.get("text", "")
            
            # Text length
            features["text_length"].append(len(text))
            
            # Token count (simple whitespace split)
            features["token_count"].append(len(text.split()))
            
            # Special characters ratio
            special_chars = sum(1 for c in text if not c.isalnum() and not c.isspace())
            features["special_chars_ratio"].append(special_chars / len(text) if text else 0)
        
        return {k: np.array(v) for k, v in features.items()}
    
    async def _extract_image_features(self, data: List[Dict[str, Any]]) -> Dict[str, np.ndarray]:
        """Extract features from image data"""
        features = {
            "image_brightness": [],
            "image_contrast": [],
            "image_size": [],
            "color_distribution": []
        }
        
        for item in data:
            try:
                # Decode image
                image_data = item.get("image", "")
                if image_data.startswith("data:image"):
                    image_data = image_data.split(",")[1]
                
                import base64
                import io
                from PIL import Image
                
                image_bytes = base64.b64decode(image_data)
                image = Image.open(io.BytesIO(image_bytes))
                
                if image.mode != "RGB":
                    image = image.convert("RGB")
                
                # Convert to numpy array
                img_array = np.array(image)
                
                # Extract features
                features["image_brightness"].append(np.mean(img_array))
                features["image_contrast"].append(np.std(img_array))
                features["image_size"].append(img_array.size)
                
                # Color distribution (mean of each channel)
                color_dist = np.mean(img_array, axis=(0, 1))
                features["color_distribution"].append(np.mean(color_dist))
                
            except Exception as e:
                self.logger.warning(f"Failed to extract features from image: {e}")
                # Use default values
                features["image_brightness"].append(0)
                features["image_contrast"].append(0)
                features["image_size"].append(0)
                features["color_distribution"].append(0)
        
        return {k: np.array(v) for k, v in features.items()}
    
    async def _extract_generic_features(self, data: List[Dict[str, Any]]) -> Dict[str, np.ndarray]:
        """Extract generic features from data"""
        features = {
            "data_size": [],
            "numeric_fields": [],
            "string_fields": []
        }
        
        for item in data:
            # Data size (number of fields)
            features["data_size"].append(len(item))
            
            # Count numeric and string fields
            numeric_count = sum(1 for v in item.values() if isinstance(v, (int, float)))
            string_count = sum(1 for v in item.values() if isinstance(v, str))
            
            features["numeric_fields"].append(numeric_count)
            features["string_fields"].append(string_count)
        
        return {k: np.array(v) for k, v in features.items()}


class DriftDetectionManager:
    """Main drift detection manager"""
    
    def __init__(self):
        self.logger = get_logger(__name__)
        self.config_path = "/app/config/drift_config.yaml"
        
        # Load configuration
        self.drift_config = {}
        
        # Initialize detectors
        self.detectors = {
            "kolmogorov_smirnov": KolmogorovSmirnovDetector(),
            "population_stability_index": PopulationStabilityIndexDetector(),
            "wasserstein_distance": WassersteinDistanceDetector(),
            "maximum_mean_discrepancy": MaximumMeanDiscrepancyDetector()
        }
        
        # Feature extractors
        self.feature_extractors = {}
        
        # Storage clients
        self.redis_client: Optional[aioredis.Redis] = None
        self.s3_client = None
        
        # Results storage
        self.drift_results: List[DriftResult] = []
    
    async def initialize(self) -> None:
        """Initialize drift detection manager"""
        try:
            self.logger.info("Initializing drift detection manager")
            
            # Load configuration
            await self._load_config()
            
            # Initialize storage clients
            await self._initialize_storage()
            
            # Initialize feature extractors
            self._initialize_feature_extractors()
            
            self.logger.info("Drift detection manager initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize drift detection manager: {e}", exc_info=True)
            raise
    
    async def _load_config(self) -> None:
        """Load drift detection configuration"""
        try:
            if os.path.exists(self.config_path):
                async with aiofiles.open(self.config_path, 'r') as f:
                    content = await f.read()
                    self.drift_config = yaml.safe_load(content)
            else:
                # Use default configuration
                self.drift_config = {
                    "models": {
                        config.model.name: {
                            "input_features": ["text_length"] if config.model.name == "bert-ner" else ["image_brightness"],
                            "drift_methods": ["kolmogorov_smirnov"],
                            "thresholds": {"ks_threshold": 0.05},
                            "reference_window_days": 30,
                            "detection_window_days": 7
                        }
                    },
                    "alerts": {
                        "slack_webhook": True,
                        "email_notifications": True,
                        "prometheus_alerts": True
                    },
                    "storage": {
                        "reference_data_path": "s3://model-artifacts/reference-data/",
                        "drift_reports_path": "s3://model-artifacts/drift-reports/"
                    }
                }
            
            self.logger.info("Drift detection configuration loaded")
            
        except Exception as e:
            self.logger.error(f"Failed to load drift config: {e}", exc_info=True)
            raise
    
    async def _initialize_storage(self) -> None:
        """Initialize storage clients"""
        try:
            # Initialize Redis client
            if config.redis.url:
                self.redis_client = aioredis.from_url(
                    config.redis.url,
                    max_connections=config.redis.max_connections
                )
                await self.redis_client.ping()
                self.logger.info("Redis client initialized for drift detection")
            
            # Initialize S3 client
            if config.s3.bucket_name:
                self.s3_client = boto3.client(
                    's3',
                    region_name=config.s3.region,
                    aws_access_key_id=config.s3.access_key_id,
                    aws_secret_access_key=config.s3.secret_access_key
                )
                self.logger.info("S3 client initialized for drift detection")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize storage clients: {e}", exc_info=True)
            raise
    
    def _initialize_feature_extractors(self) -> None:
        """Initialize feature extractors for each model"""
        for model_name in self.drift_config.get("models", {}):
            self.feature_extractors[model_name] = FeatureExtractor(model_name)
    
    async def detect_drift_for_model(self, model_name: str) -> List[DriftResult]:
        """Detect drift for a specific model"""
        if model_name not in self.drift_config.get("models", {}):
            self.logger.warning(f"No drift configuration found for model: {model_name}")
            return []
        
        model_config = self.drift_config["models"][model_name]
        results = []
        
        try:
            # Get reference and detection data
            reference_data = await self._get_reference_data(model_name, model_config)
            detection_data = await self._get_detection_data(model_name, model_config)
            
            if not reference_data or not detection_data:
                self.logger.warning(f"Insufficient data for drift detection: {model_name}")
                return []
            
            # Extract features
            feature_extractor = self.feature_extractors[model_name]
            ref_features = await feature_extractor.extract_features(reference_data)
            det_features = await feature_extractor.extract_features(detection_data)
            
            # Perform drift detection for each feature
            for feature_name in model_config.get("input_features", []):
                if feature_name not in ref_features or feature_name not in det_features:
                    continue
                
                # Run each configured drift method
                for method_name in model_config.get("drift_methods", []):
                    if method_name not in self.detectors:
                        continue
                    
                    detector = self.detectors[method_name]
                    
                    # Update detector threshold if specified
                    threshold_key = f"{method_name.replace('_', '')}_threshold"
                    if threshold_key in model_config.get("thresholds", {}):
                        detector.threshold = model_config["thresholds"][threshold_key]
                    
                    # Detect drift
                    drift_detected, drift_score, details = await detector.detect_drift(
                        ref_features[feature_name],
                        det_features[feature_name]
                    )
                    
                    # Create drift result
                    result = DriftResult(
                        model_name=model_name,
                        timestamp=datetime.utcnow(),
                        drift_detected=drift_detected,
                        drift_score=drift_score,
                        drift_method=method_name,
                        threshold=detector.threshold,
                        feature_name=feature_name,
                        reference_period=f"{model_config['reference_window_days']} days",
                        detection_period=f"{model_config['detection_window_days']} days",
                        details=details
                    )
                    
                    results.append(result)
                    
                    # Log drift detection
                    if drift_detected:
                        self.logger.warning(
                            f"Drift detected for {model_name}.{feature_name} "
                            f"using {method_name}: score={drift_score:.4f}, threshold={detector.threshold}"
                        )
                    else:
                        self.logger.info(
                            f"No drift detected for {model_name}.{feature_name} "
                            f"using {method_name}: score={drift_score:.4f}"
                        )
            
            # Store results
            await self._store_drift_results(results)
            
            return results
            
        except Exception as e:
            self.logger.error(f"Drift detection failed for model {model_name}: {e}", exc_info=True)
            return []
    
    async def _get_reference_data(self, model_name: str, model_config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get reference data for drift detection"""
        # This is a simplified implementation
        # In production, you would fetch actual historical data
        try:
            if self.redis_client:
                # Try to get from Redis cache
                key = f"reference_data:{model_name}"
                data = await self.redis_client.get(key)
                if data:
                    return json.loads(data)
            
            # Generate sample reference data for demonstration
            if model_name == "bert-ner":
                return [
                    {"text": "Hello world"},
                    {"text": "This is a test"},
                    {"text": "Natural language processing"},
                    {"text": "Machine learning model"},
                    {"text": "Artificial intelligence"}
                ] * 20  # Simulate more data
            elif model_name == "resnet-classifier":
                # Generate dummy image data
                import base64
                from PIL import Image
                import io
                
                dummy_images = []
                for i in range(100):
                    img = Image.new('RGB', (64, 64), color=(i*2, i*3, i*4))
                    buffer = io.BytesIO()
                    img.save(buffer, format='JPEG')
                    img_str = base64.b64encode(buffer.getvalue()).decode()
                    dummy_images.append({"image": img_str})
                
                return dummy_images
            
            return []
            
        except Exception as e:
            self.logger.error(f"Failed to get reference data: {e}", exc_info=True)
            return []
    
    async def _get_detection_data(self, model_name: str, model_config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get recent data for drift detection"""
        # This is a simplified implementation
        # In production, you would fetch recent request data
        try:
            if self.redis_client:
                # Try to get recent requests from Redis
                key = f"recent_requests:{model_name}"
                data = await self.redis_client.get(key)
                if data:
                    return json.loads(data)
            
            # Generate sample detection data for demonstration
            if model_name == "bert-ner":
                return [
                    {"text": "Hello there"},
                    {"text": "This is another test"},
                    {"text": "Deep learning algorithms"},
                    {"text": "Neural network models"},
                    {"text": "Computer vision tasks"}
                ] * 15  # Simulate slightly different data
            elif model_name == "resnet-classifier":
                # Generate dummy image data with slight variations
                import base64
                from PIL import Image
                import io
                
                dummy_images = []
                for i in range(80):
                    # Slightly different color distribution
                    img = Image.new('RGB', (64, 64), color=(i*3, i*2, i*5))
                    buffer = io.BytesIO()
                    img.save(buffer, format='JPEG')
                    img_str = base64.b64encode(buffer.getvalue()).decode()
                    dummy_images.append({"image": img_str})
                
                return dummy_images
            
            return []
            
        except Exception as e:
            self.logger.error(f"Failed to get detection data: {e}", exc_info=True)
            return []
    
    async def _store_drift_results(self, results: List[DriftResult]) -> None:
        """Store drift detection results"""
        try:
            # Store in memory
            self.drift_results.extend(results)
            
            # Store in Redis
            if self.redis_client:
                for result in results:
                    key = f"drift_result:{result.model_name}:{int(result.timestamp.timestamp())}"
                    data = asdict(result)
                    data["timestamp"] = result.timestamp.isoformat()
                    
                    await self.redis_client.setex(
                        key,
                        timedelta(days=30).total_seconds(),
                        json.dumps(data, default=str)
                    )
            
            # Store in S3 (if configured)
            if self.s3_client and results:
                await self._store_results_to_s3(results)
            
        except Exception as e:
            self.logger.error(f"Failed to store drift results: {e}", exc_info=True)
    
    async def _store_results_to_s3(self, results: List[DriftResult]) -> None:
        """Store drift results to S3"""
        try:
            # Create report
            report = {
                "timestamp": datetime.utcnow().isoformat(),
                "results": [asdict(result) for result in results]
            }
            
            # Convert timestamps to strings
            for result_data in report["results"]:
                result_data["timestamp"] = result_data["timestamp"].isoformat() if isinstance(result_data["timestamp"], datetime) else result_data["timestamp"]
            
            # Upload to S3
            key = f"drift-reports/{datetime.utcnow().strftime('%Y/%m/%d')}/drift_report_{int(time.time())}.json"
            
            self.s3_client.put_object(
                Bucket=config.s3.bucket_name,
                Key=key,
                Body=json.dumps(report, indent=2),
                ContentType='application/json'
            )
            
            self.logger.info(f"Drift report uploaded to S3: {key}")
            
        except Exception as e:
            self.logger.error(f"Failed to upload drift report to S3: {e}", exc_info=True)
    
    async def get_drift_summary(self, model_name: str = None) -> Dict[str, Any]:
        """Get drift detection summary"""
        try:
            # Filter results by model if specified
            if model_name:
                filtered_results = [r for r in self.drift_results if r.model_name == model_name]
            else:
                filtered_results = self.drift_results
            
            if not filtered_results:
                return {"status": "no_data"}
            
            # Calculate summary statistics
            total_checks = len(filtered_results)
            drift_detected_count = sum(1 for r in filtered_results if r.drift_detected)
            drift_rate = drift_detected_count / total_checks if total_checks > 0 else 0
            
            # Group by model and feature
            by_model = {}
            for result in filtered_results:
                if result.model_name not in by_model:
                    by_model[result.model_name] = {}
                
                feature_key = f"{result.feature_name}_{result.drift_method}"
                if feature_key not in by_model[result.model_name]:
                    by_model[result.model_name][feature_key] = {
                        "total_checks": 0,
                        "drift_detected": 0,
                        "latest_score": 0,
                        "latest_timestamp": None
                    }
                
                feature_stats = by_model[result.model_name][feature_key]
                feature_stats["total_checks"] += 1
                if result.drift_detected:
                    feature_stats["drift_detected"] += 1
                
                # Update latest
                if (feature_stats["latest_timestamp"] is None or 
                    result.timestamp > datetime.fromisoformat(feature_stats["latest_timestamp"])):
                    feature_stats["latest_score"] = result.drift_score
                    feature_stats["latest_timestamp"] = result.timestamp.isoformat()
            
            return {
                "total_checks": total_checks,
                "drift_detected_count": drift_detected_count,
                "drift_rate": drift_rate,
                "by_model": by_model,
                "latest_check": max(r.timestamp for r in filtered_results).isoformat() if filtered_results else None
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get drift summary: {e}", exc_info=True)
            return {"error": str(e)}
    
    async def cleanup(self) -> None:
        """Cleanup drift detection manager"""
        try:
            if self.redis_client:
                await self.redis_client.close()
            
            self.logger.info("Drift detection manager cleanup completed")
            
        except Exception as e:
            self.logger.error(f"Error during drift detection cleanup: {e}", exc_info=True)


# Main function for running as a standalone script
async def main():
    """Main function for drift detection job"""
    logger = get_logger(__name__)
    
    try:
        logger.info("Starting drift detection job")
        
        # Initialize drift detection manager
        drift_manager = DriftDetectionManager()
        await drift_manager.initialize()
        
        # Get models to check from environment
        models_to_check = os.getenv("MODELS_TO_CHECK", config.model.name).split(",")
        
        # Perform drift detection for each model
        all_results = []
        for model_name in models_to_check:
            model_name = model_name.strip()
            logger.info(f"Checking drift for model: {model_name}")
            
            results = await drift_manager.detect_drift_for_model(model_name)
            all_results.extend(results)
        
        # Generate summary
        summary = await drift_manager.get_drift_summary()
        logger.info(f"Drift detection completed. Summary: {summary}")
        
        # Check if any drift was detected and send alerts
        drift_detected = any(r.drift_detected for r in all_results)
        if drift_detected:
            logger.warning("Drift detected! Consider retraining models or investigating data sources.")
            # Here you would send alerts (Slack, email, etc.)
        
        # Cleanup
        await drift_manager.cleanup()
        
        logger.info("Drift detection job completed successfully")
        
    except Exception as e:
        logger.error(f"Drift detection job failed: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    import os
    asyncio.run(main())

