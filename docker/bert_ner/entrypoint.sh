#!/bin/bash

# BERT NER Model Entrypoint Script

set -e

echo "Starting BERT NER Model Server..."

# Set default values if not provided
export MODEL_NAME=${MODEL_NAME:-bert-ner}
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
import os
from transformers import AutoTokenizer, AutoModelForTokenClassification

model_path = 'dbmdz/bert-large-cased-finetuned-conll03-english'
print(f'Loading model: {model_path}')

try:
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForTokenClassification.from_pretrained(model_path)
    print('Model loaded successfully!')
except Exception as e:
    print(f'Failed to load model: {e}')
    exit(1)
"

# Start the application
echo "Starting FastAPI application..."
exec python -m uvicorn src.model_serving.main:app \
    --host $HOST \
    --port $PORT \
    --workers $WORKERS \
    --log-level $(echo $LOG_LEVEL | tr '[:upper:]' '[:lower:]') \
    --no-access-log

