# ModelServeAI - API Documentation

## Overview

The ModelServeAI provides RESTful APIs for machine learning model inference, pipeline orchestration, and system management. This documentation covers all available endpoints, request/response formats, authentication methods, and usage examples.

## Base URLs

- **Development**: `https://dev-modelserve.com`
- **Production**: `https://modelserve.com`

## Authentication

### API Key Authentication

All API requests require authentication using API keys passed in the `Authorization` header:

```http
Authorization: Bearer <your-api-key>
```

### Rate Limiting

API requests are rate-limited to prevent abuse:

- **Free Tier**: 100 requests per minute
- **Standard Tier**: 1,000 requests per minute
- **Premium Tier**: 10,000 requests per minute

Rate limit headers are included in all responses:

```http
X-RateLimit-Limit: 1000
X-RateLimit-Remaining: 999
X-RateLimit-Reset: 1640995200
```

## Model Inference APIs

### BERT Named Entity Recognition

Extract named entities from text using BERT-based models.

#### Endpoint
```http
POST /v1/models/bert-ner/predict
```

#### Request Body
```json
{
  "text": "Apple Inc. is planning to open a new store in New York City.",
  "model_version": "v1.2.0",
  "options": {
    "confidence_threshold": 0.8,
    "return_probabilities": true,
    "max_entities": 50
  }
}
```

#### Response
```json
{
  "request_id": "req_123456789",
  "model_version": "v1.2.0",
  "processing_time_ms": 45,
  "entities": [
    {
      "text": "Apple Inc.",
      "label": "ORG",
      "confidence": 0.99,
      "start": 0,
      "end": 10
    },
    {
      "text": "New York City",
      "label": "LOC", 
      "confidence": 0.95,
      "start": 45,
      "end": 58
    }
  ],
  "metadata": {
    "input_length": 59,
    "entities_found": 2,
    "model_load_time_ms": 12
  }
}
```

#### Error Response
```json
{
  "error": {
    "code": "INVALID_INPUT",
    "message": "Text input is required and cannot be empty",
    "details": {
      "field": "text",
      "provided": "",
      "expected": "non-empty string"
    }
  },
  "request_id": "req_123456789"
}
```

### ResNet Image Classification

Classify images using ResNet-based convolutional neural networks.

#### Endpoint
```http
POST /v1/models/resnet-classifier/predict
```

#### Request Body (Multipart Form)
```http
Content-Type: multipart/form-data

image: <binary image data>
model_version: v2.1.0
options: {
  "top_k": 5,
  "confidence_threshold": 0.1,
  "return_probabilities": true
}
```

#### Request Body (Base64 JSON)
```json
{
  "image": "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQ...",
  "model_version": "v2.1.0",
  "options": {
    "top_k": 5,
    "confidence_threshold": 0.1,
    "return_probabilities": true,
    "image_preprocessing": {
      "resize": [224, 224],
      "normalize": true,
      "center_crop": true
    }
  }
}
```

#### Response
```json
{
  "request_id": "req_987654321",
  "model_version": "v2.1.0",
  "processing_time_ms": 78,
  "predictions": [
    {
      "class": "golden_retriever",
      "confidence": 0.89,
      "class_id": 207
    },
    {
      "class": "labrador_retriever", 
      "confidence": 0.08,
      "class_id": 208
    },
    {
      "class": "nova_scotia_duck_tolling_retriever",
      "confidence": 0.02,
      "class_id": 209
    }
  ],
  "metadata": {
    "image_size": [640, 480],
    "preprocessing_time_ms": 15,
    "inference_time_ms": 63,
    "total_classes": 1000
  }
}
```

### Batch Prediction

Process multiple inputs in a single request for improved efficiency.

#### Endpoint
```http
POST /v1/models/{model_name}/batch-predict
```

#### Request Body
```json
{
  "inputs": [
    {
      "id": "input_1",
      "data": {
        "text": "First text to analyze"
      }
    },
    {
      "id": "input_2", 
      "data": {
        "text": "Second text to analyze"
      }
    }
  ],
  "model_version": "v1.2.0",
  "options": {
    "batch_size": 32,
    "timeout_seconds": 300
  }
}
```

#### Response
```json
{
  "request_id": "req_batch_123",
  "batch_id": "batch_456789",
  "status": "completed",
  "processing_time_ms": 234,
  "results": [
    {
      "id": "input_1",
      "status": "success",
      "prediction": {
        "entities": [...]
      }
    },
    {
      "id": "input_2",
      "status": "success", 
      "prediction": {
        "entities": [...]
      }
    }
  ],
  "metadata": {
    "total_inputs": 2,
    "successful": 2,
    "failed": 0,
    "batch_processing_time_ms": 234
  }
}
```

## Pipeline Orchestration APIs

### Pipeline Definition

Define multi-model workflows that chain different models together.

#### Endpoint
```http
POST /v1/pipelines
```

#### Request Body
```json
{
  "name": "document_analysis_pipeline",
  "description": "OCR followed by NER for document processing",
  "version": "1.0.0",
  "stages": [
    {
      "id": "ocr_stage",
      "model": "tesseract-ocr",
      "model_version": "v4.1.0",
      "inputs": ["image"],
      "outputs": ["text"],
      "options": {
        "language": "eng",
        "psm": 6
      }
    },
    {
      "id": "ner_stage", 
      "model": "bert-ner",
      "model_version": "v1.2.0",
      "inputs": ["text"],
      "outputs": ["entities"],
      "depends_on": ["ocr_stage"],
      "options": {
        "confidence_threshold": 0.8
      }
    }
  ],
  "error_handling": {
    "retry_policy": {
      "max_retries": 3,
      "backoff_strategy": "exponential"
    },
    "failure_action": "continue"
  }
}
```

#### Response
```json
{
  "pipeline_id": "pipe_123456789",
  "name": "document_analysis_pipeline",
  "version": "1.0.0",
  "status": "active",
  "created_at": "2024-01-15T10:30:00Z",
  "stages": [
    {
      "id": "ocr_stage",
      "status": "ready",
      "model_endpoint": "/v1/models/tesseract-ocr/predict"
    },
    {
      "id": "ner_stage",
      "status": "ready", 
      "model_endpoint": "/v1/models/bert-ner/predict"
    }
  ]
}
```

### Pipeline Execution

Execute a defined pipeline with input data.

#### Endpoint
```http
POST /v1/pipelines/{pipeline_id}/execute
```

#### Request Body
```json
{
  "inputs": {
    "image": "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQ..."
  },
  "execution_options": {
    "timeout_seconds": 600,
    "priority": "normal",
    "async": false
  },
  "metadata": {
    "user_id": "user_123",
    "session_id": "session_456"
  }
}
```

#### Response (Synchronous)
```json
{
  "execution_id": "exec_789012345",
  "pipeline_id": "pipe_123456789", 
  "status": "completed",
  "started_at": "2024-01-15T10:35:00Z",
  "completed_at": "2024-01-15T10:35:45Z",
  "total_time_ms": 45000,
  "results": {
    "ocr_stage": {
      "status": "success",
      "output": {
        "text": "This is the extracted text from the image."
      },
      "processing_time_ms": 15000
    },
    "ner_stage": {
      "status": "success",
      "output": {
        "entities": [
          {
            "text": "extracted text",
            "label": "MISC",
            "confidence": 0.85
          }
        ]
      },
      "processing_time_ms": 30000
    }
  }
}
```

#### Response (Asynchronous)
```json
{
  "execution_id": "exec_789012345",
  "pipeline_id": "pipe_123456789",
  "status": "running",
  "started_at": "2024-01-15T10:35:00Z",
  "estimated_completion": "2024-01-15T10:36:00Z",
  "status_url": "/v1/pipelines/executions/exec_789012345/status",
  "webhook_url": "https://your-app.com/pipeline-webhook"
}
```

### Pipeline Status

Check the status of a running pipeline execution.

#### Endpoint
```http
GET /v1/pipelines/executions/{execution_id}/status
```

#### Response
```json
{
  "execution_id": "exec_789012345",
  "pipeline_id": "pipe_123456789",
  "status": "running",
  "started_at": "2024-01-15T10:35:00Z",
  "current_stage": "ner_stage",
  "progress": {
    "completed_stages": 1,
    "total_stages": 2,
    "percentage": 50
  },
  "stage_status": [
    {
      "id": "ocr_stage",
      "status": "completed",
      "started_at": "2024-01-15T10:35:00Z",
      "completed_at": "2024-01-15T10:35:15Z"
    },
    {
      "id": "ner_stage", 
      "status": "running",
      "started_at": "2024-01-15T10:35:15Z",
      "estimated_completion": "2024-01-15T10:35:45Z"
    }
  ]
}
```

## Model Management APIs

### List Available Models

Get information about all available models and their versions.

#### Endpoint
```http
GET /v1/models
```

#### Query Parameters
- `category` (optional): Filter by model category (nlp, cv, audio)
- `status` (optional): Filter by model status (active, deprecated, beta)
- `limit` (optional): Maximum number of results (default: 50)
- `offset` (optional): Pagination offset (default: 0)

#### Response
```json
{
  "models": [
    {
      "name": "bert-ner",
      "display_name": "BERT Named Entity Recognition",
      "category": "nlp",
      "description": "Extract named entities from text using BERT",
      "status": "active",
      "versions": [
        {
          "version": "v1.2.0",
          "status": "active",
          "created_at": "2024-01-10T09:00:00Z",
          "performance": {
            "accuracy": 0.94,
            "f1_score": 0.92,
            "avg_latency_ms": 45
          }
        },
        {
          "version": "v1.1.0",
          "status": "deprecated", 
          "created_at": "2023-12-15T14:30:00Z"
        }
      ],
      "endpoints": {
        "predict": "/v1/models/bert-ner/predict",
        "batch_predict": "/v1/models/bert-ner/batch-predict"
      }
    }
  ],
  "pagination": {
    "total": 5,
    "limit": 50,
    "offset": 0,
    "has_more": false
  }
}
```

### Model Information

Get detailed information about a specific model.

#### Endpoint
```http
GET /v1/models/{model_name}
```

#### Response
```json
{
  "name": "bert-ner",
  "display_name": "BERT Named Entity Recognition",
  "category": "nlp",
  "description": "Extract named entities from text using BERT-based neural networks",
  "status": "active",
  "current_version": "v1.2.0",
  "created_at": "2023-10-01T12:00:00Z",
  "updated_at": "2024-01-10T09:00:00Z",
  "versions": [
    {
      "version": "v1.2.0",
      "status": "active",
      "created_at": "2024-01-10T09:00:00Z",
      "model_size_mb": 438,
      "performance_metrics": {
        "accuracy": 0.94,
        "precision": 0.93,
        "recall": 0.91,
        "f1_score": 0.92,
        "avg_latency_ms": 45,
        "throughput_rps": 22
      },
      "supported_languages": ["en", "es", "fr", "de"],
      "entity_types": ["PERSON", "ORG", "LOC", "MISC"]
    }
  ],
  "usage_stats": {
    "total_requests": 1250000,
    "requests_last_30_days": 45000,
    "avg_daily_requests": 1500
  },
  "pricing": {
    "per_request": 0.001,
    "currency": "USD",
    "billing_unit": "request"
  }
}
```

### Model Health Check

Check the health and availability of a specific model.

#### Endpoint
```http
GET /v1/models/{model_name}/health
```

#### Response
```json
{
  "model_name": "bert-ner",
  "version": "v1.2.0",
  "status": "healthy",
  "last_check": "2024-01-15T10:45:00Z",
  "response_time_ms": 12,
  "availability": {
    "uptime_percentage": 99.95,
    "last_24h": 99.98,
    "last_7d": 99.92
  },
  "performance": {
    "avg_latency_ms": 45,
    "p95_latency_ms": 78,
    "p99_latency_ms": 120,
    "error_rate": 0.001
  },
  "resources": {
    "cpu_usage": 0.65,
    "memory_usage": 0.78,
    "gpu_usage": 0.45,
    "active_connections": 23
  }
}
```

## System Management APIs

### System Status

Get overall system health and status information.

#### Endpoint
```http
GET /v1/system/status
```

#### Response
```json
{
  "status": "healthy",
  "version": "2.1.0",
  "uptime_seconds": 2592000,
  "last_check": "2024-01-15T10:50:00Z",
  "components": {
    "api_gateway": {
      "status": "healthy",
      "response_time_ms": 5,
      "last_check": "2024-01-15T10:50:00Z"
    },
    "model_servers": {
      "status": "healthy",
      "active_models": 3,
      "total_capacity": 100,
      "used_capacity": 67
    },
    "database": {
      "status": "healthy",
      "connection_pool": {
        "active": 15,
        "idle": 35,
        "max": 50
      }
    },
    "cache": {
      "status": "healthy",
      "memory_usage": 0.72,
      "hit_rate": 0.89
    },
    "message_queue": {
      "status": "healthy",
      "pending_jobs": 12,
      "processing_rate": 150
    }
  },
  "metrics": {
    "requests_per_second": 45,
    "avg_response_time_ms": 78,
    "error_rate": 0.002,
    "active_users": 234
  }
}
```

### Metrics

Get detailed system metrics for monitoring and analysis.

#### Endpoint
```http
GET /v1/system/metrics
```

#### Query Parameters
- `start_time` (optional): Start time for metrics (ISO 8601)
- `end_time` (optional): End time for metrics (ISO 8601)
- `granularity` (optional): Metrics granularity (1m, 5m, 1h, 1d)
- `metrics` (optional): Comma-separated list of specific metrics

#### Response
```json
{
  "time_range": {
    "start": "2024-01-15T09:00:00Z",
    "end": "2024-01-15T10:00:00Z",
    "granularity": "5m"
  },
  "metrics": {
    "request_rate": [
      {
        "timestamp": "2024-01-15T09:00:00Z",
        "value": 42.5
      },
      {
        "timestamp": "2024-01-15T09:05:00Z", 
        "value": 45.2
      }
    ],
    "error_rate": [
      {
        "timestamp": "2024-01-15T09:00:00Z",
        "value": 0.001
      },
      {
        "timestamp": "2024-01-15T09:05:00Z",
        "value": 0.002
      }
    ],
    "avg_latency": [
      {
        "timestamp": "2024-01-15T09:00:00Z",
        "value": 78.5
      },
      {
        "timestamp": "2024-01-15T09:05:00Z",
        "value": 82.1
      }
    ]
  }
}
```

## Error Handling

### Error Response Format

All API errors follow a consistent format:

```json
{
  "error": {
    "code": "ERROR_CODE",
    "message": "Human-readable error message",
    "details": {
      "field": "specific_field",
      "provided": "invalid_value",
      "expected": "expected_format"
    }
  },
  "request_id": "req_123456789",
  "timestamp": "2024-01-15T10:30:00Z"
}
```

### Common Error Codes

| Code | HTTP Status | Description |
|------|-------------|-------------|
| `INVALID_INPUT` | 400 | Request input validation failed |
| `UNAUTHORIZED` | 401 | Authentication required or invalid |
| `FORBIDDEN` | 403 | Insufficient permissions |
| `NOT_FOUND` | 404 | Requested resource not found |
| `RATE_LIMITED` | 429 | Rate limit exceeded |
| `MODEL_UNAVAILABLE` | 503 | Model temporarily unavailable |
| `INTERNAL_ERROR` | 500 | Internal server error |
| `TIMEOUT` | 504 | Request timeout |

### Retry Guidelines

- **Rate Limiting (429)**: Wait for the time specified in `Retry-After` header
- **Server Errors (5xx)**: Use exponential backoff with jitter
- **Timeout Errors**: Retry with longer timeout values
- **Model Unavailable**: Wait 30-60 seconds before retrying

## SDKs and Libraries

### Python SDK

```python
from ai_model_serving import Client

# Initialize client
client = Client(
    api_key="your-api-key",
    base_url="https://api.ai-model-serving.com"
)

# BERT NER prediction
result = client.bert_ner.predict(
    text="Apple Inc. is based in Cupertino, California.",
    confidence_threshold=0.8
)

# Image classification
with open("image.jpg", "rb") as f:
    result = client.resnet_classifier.predict(
        image=f,
        top_k=5
    )

# Pipeline execution
pipeline = client.pipelines.get("document_analysis_pipeline")
result = pipeline.execute(
    inputs={"image": image_data},
    async=True
)
```

### JavaScript SDK

```javascript
import { AIModelServingClient } from '@ai-model-serving/client';

// Initialize client
const client = new AIModelServingClient({
  apiKey: 'your-api-key',
  baseUrl: 'https://api.ai-model-serving.com'
});

// BERT NER prediction
const result = await client.bertNer.predict({
  text: 'Apple Inc. is based in Cupertino, California.',
  confidenceThreshold: 0.8
});

// Batch prediction
const batchResult = await client.bertNer.batchPredict({
  inputs: [
    { id: '1', data: { text: 'First text' } },
    { id: '2', data: { text: 'Second text' } }
  ]
});
```

### cURL Examples

#### BERT NER Prediction
```bash
curl -X POST https://api.ai-model-serving.com/v1/models/bert-ner/predict \
  -H "Authorization: Bearer your-api-key" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Apple Inc. is planning to open a new store in New York City.",
    "options": {
      "confidence_threshold": 0.8
    }
  }'
```

#### Image Classification
```bash
curl -X POST https://api.ai-model-serving.com/v1/models/resnet-classifier/predict \
  -H "Authorization: Bearer your-api-key" \
  -F "image=@image.jpg" \
  -F 'options={"top_k": 5}'
```

#### Pipeline Execution
```bash
curl -X POST https://api.ai-model-serving.com/v1/pipelines/pipe_123456789/execute \
  -H "Authorization: Bearer your-api-key" \
  -H "Content-Type: application/json" \
  -d '{
    "inputs": {
      "image": "data:image/jpeg;base64,..."
    },
    "execution_options": {
      "async": false
    }
  }'
```

## Webhooks

### Pipeline Completion Webhook

Receive notifications when pipeline executions complete.

#### Webhook Payload
```json
{
  "event": "pipeline.execution.completed",
  "execution_id": "exec_789012345",
  "pipeline_id": "pipe_123456789",
  "status": "completed",
  "started_at": "2024-01-15T10:35:00Z",
  "completed_at": "2024-01-15T10:35:45Z",
  "results": {
    "ocr_stage": {
      "status": "success",
      "output": {...}
    },
    "ner_stage": {
      "status": "success", 
      "output": {...}
    }
  },
  "metadata": {
    "user_id": "user_123",
    "session_id": "session_456"
  }
}
```

### Model Status Webhook

Receive notifications about model status changes.

#### Webhook Payload
```json
{
  "event": "model.status.changed",
  "model_name": "bert-ner",
  "version": "v1.2.0",
  "old_status": "healthy",
  "new_status": "degraded",
  "timestamp": "2024-01-15T10:45:00Z",
  "details": {
    "reason": "high_error_rate",
    "error_rate": 0.05,
    "threshold": 0.01
  }
}
```

## Rate Limits and Quotas

### Request Rate Limits

| Tier | Requests/Minute | Burst Limit |
|------|-----------------|-------------|
| Free | 100 | 200 |
| Standard | 1,000 | 2,000 |
| Premium | 10,000 | 20,000 |
| Enterprise | Custom | Custom |

### Resource Quotas

| Resource | Free | Standard | Premium | Enterprise |
|----------|------|----------|---------|------------|
| Models | 2 | 10 | 50 | Unlimited |
| Pipelines | 1 | 5 | 25 | Unlimited |
| Storage (GB) | 1 | 10 | 100 | Custom |
| Bandwidth (GB/month) | 10 | 100 | 1,000 | Custom |

## Support and Resources

- **API Status**: https://status.modelserveai.com
- **Documentation**: https://docs.modelserveai.com
- **Support**: support@ai-model-serving.com
- **Community**: https://community.modelserveai.com
- **GitHub**: https://github.com/modelserveai/platform

---

*Last updated: January 15, 2024*

