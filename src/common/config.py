"""
Common configuration module for ModelServeAI
"""
import os
from typing import Optional, Dict, Any
from pydantic import BaseSettings, Field
import yaml
import json


class DatabaseConfig(BaseSettings):
    """Database configuration"""
    url: str = Field(default="postgresql://localhost:5432/modelmetadata", env="DATABASE_URL")
    pool_size: int = Field(default=10, env="DB_POOL_SIZE")
    max_overflow: int = Field(default=20, env="DB_MAX_OVERFLOW")
    pool_timeout: int = Field(default=30, env="DB_POOL_TIMEOUT")
    pool_recycle: int = Field(default=3600, env="DB_POOL_RECYCLE")


class RedisConfig(BaseSettings):
    """Redis configuration"""
    url: str = Field(default="redis://localhost:6379", env="REDIS_URL")
    max_connections: int = Field(default=20, env="REDIS_MAX_CONNECTIONS")
    retry_on_timeout: bool = Field(default=True, env="REDIS_RETRY_ON_TIMEOUT")
    socket_timeout: int = Field(default=5, env="REDIS_SOCKET_TIMEOUT")


class S3Config(BaseSettings):
    """S3 configuration"""
    bucket_name: str = Field(default="model-artifacts", env="AWS_S3_BUCKET")
    region: str = Field(default="us-west-2", env="AWS_REGION")
    access_key_id: Optional[str] = Field(default=None, env="AWS_ACCESS_KEY_ID")
    secret_access_key: Optional[str] = Field(default=None, env="AWS_SECRET_ACCESS_KEY")
    endpoint_url: Optional[str] = Field(default=None, env="AWS_S3_ENDPOINT_URL")


class ModelConfig(BaseSettings):
    """Model serving configuration"""
    name: str = Field(default="unknown", env="MODEL_NAME")
    version: str = Field(default="v1.0.0", env="MODEL_VERSION")
    cache_size: int = Field(default=1000, env="MODEL_CACHE_SIZE")
    batch_size: int = Field(default=32, env="MODEL_BATCH_SIZE")
    timeout: int = Field(default=30, env="MODEL_TIMEOUT")
    max_requests: int = Field(default=1000, env="MAX_REQUESTS")
    max_requests_jitter: int = Field(default=100, env="MAX_REQUESTS_JITTER")
    workers: int = Field(default=4, env="WORKERS")


class MetricsConfig(BaseSettings):
    """Metrics and monitoring configuration"""
    enabled: bool = Field(default=True, env="ENABLE_METRICS")
    port: int = Field(default=8080, env="PROMETHEUS_PORT")
    path: str = Field(default="/metrics", env="PROMETHEUS_PATH")
    push_gateway_url: Optional[str] = Field(default=None, env="PROMETHEUS_PUSH_GATEWAY_URL")


class LoggingConfig(BaseSettings):
    """Logging configuration"""
    level: str = Field(default="INFO", env="LOG_LEVEL")
    format: str = Field(default="json", env="LOG_FORMAT")
    file_path: Optional[str] = Field(default=None, env="LOG_FILE_PATH")
    max_file_size: int = Field(default=100, env="LOG_MAX_FILE_SIZE_MB")
    backup_count: int = Field(default=5, env="LOG_BACKUP_COUNT")


class FeatureFlags(BaseSettings):
    """Feature flags configuration"""
    enable_shadow_mode: bool = Field(default=True, env="ENABLE_SHADOW_MODE")
    enable_drift_detection: bool = Field(default=True, env="ENABLE_DRIFT_DETECTION")
    enable_debug_endpoints: bool = Field(default=False, env="ENABLE_DEBUG_ENDPOINTS")
    enable_health_checks: bool = Field(default=True, env="ENABLE_HEALTH_CHECKS")


class AppConfig(BaseSettings):
    """Main application configuration"""
    environment: str = Field(default="development", env="ENVIRONMENT")
    debug: bool = Field(default=False, env="DEBUG")
    host: str = Field(default="0.0.0.0", env="HOST")
    port: int = Field(default=8000, env="PORT")
    reload: bool = Field(default=False, env="RELOAD")
    
    # Sub-configurations
    database: DatabaseConfig = DatabaseConfig()
    redis: RedisConfig = RedisConfig()
    s3: S3Config = S3Config()
    model: ModelConfig = ModelConfig()
    metrics: MetricsConfig = MetricsConfig()
    logging: LoggingConfig = LoggingConfig()
    features: FeatureFlags = FeatureFlags()
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


def load_model_endpoints_config(config_path: str = "/app/config/endpoints.yaml") -> Dict[str, Any]:
    """Load model endpoints configuration from YAML file"""
    try:
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        else:
            # Return default configuration if file doesn't exist
            return {
                "models": {
                    "bert-ner": {
                        "name": "BERT Named Entity Recognition",
                        "version": "v1.0.0",
                        "path": "/models/bert-ner",
                        "framework": "pytorch",
                        "resources": {
                            "cpu": "500m",
                            "memory": "1Gi",
                            "gpu": 0
                        }
                    }
                }
            }
    except Exception as e:
        print(f"Error loading model endpoints config: {e}")
        return {"models": {}}


def get_config() -> AppConfig:
    """Get application configuration instance"""
    return AppConfig()


# Global configuration instance
config = get_config()

