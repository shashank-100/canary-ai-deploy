#!/bin/bash

# ResNet Classifier Model Entrypoint Script

set -e

echo "Starting ResNet Classifier Model Server..."

# Set default values if not provided
export MODEL_NAME=${MODEL_NAME:-resnet-classifier}
export MODEL_VERSION=${MODEL_VERSION:-v1.0.0}
export PORT=${PORT:-8000}
export METRICS_PORT=${METRICS_PORT:-8080}
export HOST=${HOST:-0.0.0.0}
export WORKERS=${WORKERS:-1}
export LOG_LEVEL=${LOG_LEVEL:-INFO}

# Print configuration
echo "Configuration:"
echo "  Model Name: $MODEL_NAME"
echo "  Model Version: $MODEL_VERSION"
echo "  Port: $PORT"
echo "  Metrics Port: $METRICS_PORT"
echo "  Host: $HOST"
echo "  Workers: $WORKERS"
echo "  Log Level: $LOG_LEVEL"

# Check GPU availability
echo "Checking GPU availability..."
python -c "
import torch
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU count: {torch.cuda.device_count()}')
    for i in range(torch.cuda.device_count()):
        print(f'GPU {i}: {torch.cuda.get_device_name(i)}')
        print(f'  Memory: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.1f} GB')
else:
    print('No GPU available, using CPU')
"

# Wait for dependencies (if needed)
if [ ! -z "$DATABASE_URL" ]; then
    echo "Waiting for database..."
    python -c "
import asyncio
import asyncpg
import sys
import os

async def wait_for_db():
    for i in range(30):
        try:
            conn = await asyncpg.connect(os.environ['DATABASE_URL'], timeout=5)
            await conn.execute('SELECT 1')
            await conn.close()
            print('Database is ready!')
            return
        except Exception as e:
            print(f'Database not ready (attempt {i+1}/30): {e}')
            await asyncio.sleep(2)
    print('Database connection failed after 30 attempts')
    sys.exit(1)

asyncio.run(wait_for_db())
"
fi

if [ ! -z "$REDIS_URL" ]; then
    echo "Waiting for Redis..."
    python -c "
import asyncio
import aioredis
import sys
import os

async def wait_for_redis():
    for i in range(30):
        try:
            redis = aioredis.from_url(os.environ['REDIS_URL'], socket_timeout=5)
            await redis.ping()
            await redis.close()
            print('Redis is ready!')
            return
        except Exception as e:
            print(f'Redis not ready (attempt {i+1}/30): {e}')
            await asyncio.sleep(2)
    print('Redis connection failed after 30 attempts')
    sys.exit(1)

asyncio.run(wait_for_redis())
"
fi

# Download model if needed (in production, models would be pre-downloaded)
echo "Checking model availability..."
python -c "
import torch
import torchvision.models as models

print('Loading ResNet model...')
try:
    model = models.resnet50(pretrained=True)
    model.eval()
    print('ResNet model loaded successfully!')
    
    # Test GPU if available
    if torch.cuda.is_available():
        model = model.cuda()
        print('Model moved to GPU successfully!')
        
        # Test inference
        dummy_input = torch.randn(1, 3, 224, 224).cuda()
        with torch.no_grad():
            output = model(dummy_input)
        print('GPU inference test successful!')
    else:
        # Test CPU inference
        dummy_input = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            output = model(dummy_input)
        print('CPU inference test successful!')
        
except Exception as e:
    print(f'Failed to load or test model: {e}')
    exit(1)
"

# Set memory fraction for GPU (if available)
if [ ! -z "$NVIDIA_VISIBLE_DEVICES" ] && [ "$NVIDIA_VISIBLE_DEVICES" != "" ]; then
    echo "Setting GPU memory configuration..."
    export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
fi

# Start the application
echo "Starting FastAPI application..."
exec python -m uvicorn src.model_serving.main:app \
    --host $HOST \
    --port $PORT \
    --workers $WORKERS \
    --log-level $(echo $LOG_LEVEL | tr '[:upper:]' '[:lower:]') \
    --no-access-log

