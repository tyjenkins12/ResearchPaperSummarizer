# Multi-stage Docker build for Research Paper Summarizer
FROM python:3.11-slim as base

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH="/app/src" \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Copy dependency files
COPY pyproject.toml ./
COPY requirements.txt* ./

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install -e .

# Development stage
FROM base as development

# Install development dependencies
RUN pip install -e .[dev]

# Copy source code
COPY src/ src/
COPY tests/ tests/
COPY *.py ./

# Expose port
EXPOSE 8000

# Command for development
CMD ["uvicorn", "papersum.web.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]

# Production stage
FROM base as production

# Copy only necessary files
COPY src/ src/
COPY data/ data/
COPY *.py ./

# Create non-root user
RUN adduser --disabled-password --gecos "" appuser && \
    chown -R appuser:appuser /app
USER appuser

# Create directories for uploads and data
RUN mkdir -p /app/data/uploads /app/data/outputs

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Expose port
EXPOSE 8000

# Command for production
CMD ["uvicorn", "papersum.web.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]