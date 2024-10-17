#!/bin/bash

# Drift Detector Entrypoint Script

set -e

echo "Starting Drift Detection Job..."

# Set default values if not provided
export MODELS_TO_CHECK=${MODELS_TO_CHECK:-bert-ner,resnet-classifier}
export DRIFT_THRESHOLD=${DRIFT_THRESHOLD:-0.1}
export LOG_LEVEL=${LOG_LEVEL:-INFO}

# Print configuration
echo "Configuration:"
echo "  Models to check: $MODELS_TO_CHECK"
echo "  Drift threshold: $DRIFT_THRESHOLD"
echo "  Log level: $LOG_LEVEL"

# Wait for dependencies
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

# Check S3 access (if configured)
if [ ! -z "$AWS_S3_BUCKET" ]; then
    echo "Checking S3 access..."
    python -c "
import boto3
import os
from botocore.exceptions import ClientError

try:
    s3_client = boto3.client('s3')
    s3_client.head_bucket(Bucket=os.environ['AWS_S3_BUCKET'])
    print('S3 access verified!')
except Exception as e:
    print(f'S3 access check failed: {e}')
    print('Continuing without S3...')
"
fi

# Run drift detection
echo "Running drift detection..."
python -m src.drift_detection.drift_detector

echo "Drift detection job completed."

