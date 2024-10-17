"""
Logging configuration module for AI Model Serving Platform
"""
import logging
import logging.handlers
import sys
import json
from typing import Dict, Any
from datetime import datetime
import traceback


class JSONFormatter(logging.Formatter):
    """Custom JSON formatter for structured logging"""
    
    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            'timestamp': datetime.utcnow().isoformat() + 'Z',
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
        }
        
        # Add extra fields if present
        if hasattr(record, 'model_name'):
            log_entry['model_name'] = record.model_name
        if hasattr(record, 'model_version'):
            log_entry['model_version'] = record.model_version
        if hasattr(record, 'request_id'):
            log_entry['request_id'] = record.request_id
        if hasattr(record, 'user_id'):
            log_entry['user_id'] = record.user_id
        if hasattr(record, 'duration'):
            log_entry['duration'] = record.duration
        if hasattr(record, 'status_code'):
            log_entry['status_code'] = record.status_code
        
        # Add exception information if present
        if record.exc_info:
            log_entry['exception'] = {
                'type': record.exc_info[0].__name__,
                'message': str(record.exc_info[1]),
                'traceback': traceback.format_exception(*record.exc_info)
            }
        
        return json.dumps(log_entry, ensure_ascii=False)


class PlainFormatter(logging.Formatter):
    """Plain text formatter for human-readable logs"""
    
    def __init__(self):
        super().__init__(
            fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )


def setup_logging(
    level: str = "INFO",
    format_type: str = "json",
    file_path: str = None,
    max_file_size: int = 100,
    backup_count: int = 5
) -> None:
    """
    Setup logging configuration
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        format_type: Format type ('json' or 'plain')
        file_path: Path to log file (optional)
        max_file_size: Maximum file size in MB
        backup_count: Number of backup files to keep
    """
    # Convert string level to logging level
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    
    # Choose formatter
    if format_type.lower() == "json":
        formatter = JSONFormatter()
    else:
        formatter = PlainFormatter()
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)
    
    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(numeric_level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # File handler (if file_path is provided)
    if file_path:
        file_handler = logging.handlers.RotatingFileHandler(
            file_path,
            maxBytes=max_file_size * 1024 * 1024,  # Convert MB to bytes
            backupCount=backup_count,
            encoding='utf-8'
        )
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
    
    # Set specific logger levels
    logging.getLogger("uvicorn").setLevel(logging.INFO)
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("fastapi").setLevel(logging.INFO)
    logging.getLogger("sqlalchemy").setLevel(logging.WARNING)
    logging.getLogger("redis").setLevel(logging.WARNING)
    logging.getLogger("boto3").setLevel(logging.WARNING)
    logging.getLogger("botocore").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance with the given name"""
    return logging.getLogger(name)


class LoggerAdapter(logging.LoggerAdapter):
    """Custom logger adapter for adding context to log messages"""
    
    def __init__(self, logger: logging.Logger, extra: Dict[str, Any]):
        super().__init__(logger, extra)
    
    def process(self, msg: str, kwargs: Dict[str, Any]) -> tuple:
        """Process log message and add extra context"""
        # Merge extra context
        if 'extra' in kwargs:
            kwargs['extra'].update(self.extra)
        else:
            kwargs['extra'] = self.extra.copy()
        
        return msg, kwargs


def get_model_logger(model_name: str, model_version: str) -> LoggerAdapter:
    """Get a logger adapter with model context"""
    logger = get_logger(f"model.{model_name}")
    return LoggerAdapter(logger, {
        'model_name': model_name,
        'model_version': model_version
    })


def get_request_logger(request_id: str, user_id: str = None) -> LoggerAdapter:
    """Get a logger adapter with request context"""
    logger = get_logger("request")
    extra = {'request_id': request_id}
    if user_id:
        extra['user_id'] = user_id
    return LoggerAdapter(logger, extra)


# Utility functions for common logging patterns
def log_model_inference(
    logger: logging.Logger,
    model_name: str,
    model_version: str,
    duration: float,
    batch_size: int,
    success: bool = True,
    error: Exception = None
):
    """Log model inference information"""
    extra = {
        'model_name': model_name,
        'model_version': model_version,
        'duration': duration,
        'batch_size': batch_size
    }
    
    if success:
        logger.info(
            f"Model inference completed successfully in {duration:.3f}s for batch size {batch_size}",
            extra=extra
        )
    else:
        extra['error_type'] = type(error).__name__ if error else 'Unknown'
        logger.error(
            f"Model inference failed after {duration:.3f}s: {error}",
            extra=extra,
            exc_info=error
        )


def log_request(
    logger: logging.Logger,
    method: str,
    path: str,
    status_code: int,
    duration: float,
    request_id: str = None,
    user_id: str = None
):
    """Log HTTP request information"""
    extra = {
        'method': method,
        'path': path,
        'status_code': status_code,
        'duration': duration
    }
    
    if request_id:
        extra['request_id'] = request_id
    if user_id:
        extra['user_id'] = user_id
    
    level = logging.INFO if status_code < 400 else logging.ERROR
    logger.log(
        level,
        f"{method} {path} - {status_code} - {duration:.3f}s",
        extra=extra
    )

