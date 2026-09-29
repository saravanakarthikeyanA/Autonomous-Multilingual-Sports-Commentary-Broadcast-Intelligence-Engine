# Production Dockerfile for Autonomous Sports Commentary Platform on AWS
FROM python:3.11-slim

WORKDIR /app

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install system dependencies: Audio tools & networking
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    ffmpeg \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Create necessary directories and configure Streamlit headless credentials
RUN mkdir -p /app/data/mlflow /app/data/audio_cache /root/.streamlit /app/.streamlit \
    && echo '[general]\nemail = ""' > /root/.streamlit/credentials.toml \
    && cp /root/.streamlit/credentials.toml /app/.streamlit/credentials.toml \
    && echo '[server]\nheadless = true\nenableCORS = false\nenableXsrfProtection = false\nfileWatcherType = "none"\n[browser]\ngatherUsageStats = false' > /root/.streamlit/config.toml \
    && cp /root/.streamlit/config.toml /app/.streamlit/config.toml

# Copy source code and application assets
COPY . .

# Copy and prepare entrypoint
COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

# Expose ports: 8501 for Streamlit Broadcast UI, 8000 for Prometheus Metrics Exporter
EXPOSE 8501 8000

ENTRYPOINT ["/app/entrypoint.sh"]
