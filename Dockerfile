# Production Dockerfile for Autonomous Sports Commentary System (Lean API-First)
FROM python:3.11-slim

WORKDIR /app

# Install lightweight system audio dependencies and curl for healthcheck
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    ffmpeg \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code and data
COPY . .

# Expose Streamlit UI port & Prometheus metrics port
EXPOSE 8501 8000

# Launch Streamlit Application (dynamically adapts to $PORT on Render, defaults to 8501 on AWS/Local)
CMD ["sh", "-c", "streamlit run app.py --server.port=${PORT:-8501} --server.address=0.0.0.0 --server.enableCORS=false --server.enableXsrfProtection=false"]
