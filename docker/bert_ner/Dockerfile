# BERT NER Model Dockerfile
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ ./src/
COPY docker/bert_ner/entrypoint.sh ./entrypoint.sh

# Create necessary directories
RUN mkdir -p /app/config /app/logs /app/models

# Set environment variables
ENV MODEL_NAME=bert-ner
ENV MODEL_VERSION=v1.0.0
ENV PORT=8000
ENV METRICS_PORT=8080
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Make entrypoint executable
RUN chmod +x ./entrypoint.sh

# Create non-root user
RUN useradd --create-home --shell /bin/bash app && \
    chown -R app:app /app
USER app

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:${PORT}/health || exit 1

# Expose ports
EXPOSE 8000 8080

# Set entrypoint
ENTRYPOINT ["./entrypoint.sh"]

