# Multi-stage build for smaller image size
FROM python:3.11-slim AS builder

WORKDIR /app

# Install only essential build dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential && \
    rm -rf /var/lib/apt/lists/*

# Copy app requirements and install Python dependencies
COPY app/requirements.txt /app/app/requirements.txt
RUN pip install --no-cache-dir --user -r /app/app/requirements.txt

# Final stage - minimal runtime image
FROM python:3.11-slim

WORKDIR /app

# Copy Python packages from builder stage
COPY --from=builder /root/.local /root/.local

# Copy application code
COPY . /app/

# Add .local/bin to PATH
ENV PATH=/root/.local/bin:$PATH \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

EXPOSE 8501

# Runtime env vars expected:
# - Dagshub_movie (required)
# - TMDB_API (optional)
CMD ["streamlit", "run", "app/app.py", "--server.address=0.0.0.0", "--server.port=8501"]
