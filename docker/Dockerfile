# Use Python 3.11 slim image for modern ML models
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libc6-dev \
    libffi-dev \
    libssl-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy Python 3.11 requirements first (for better Docker caching)
COPY requirements_py311_gcp.txt .
RUN pip install --no-cache-dir -r requirements_py311_gcp.txt

# Copy application code
COPY *.py ./
COPY models/ ./models/

# Create necessary directories
RUN mkdir -p /app/logs /app/results /app/data

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Default command
CMD ["python", "cloud_experiment_runner.py"] 