# Production Dockerfile for Autonomous Sports Commentary & Observability System
FROM python:3.11-slim

WORKDIR /app

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install system dependencies: Audio tools, Nginx, Supervisor, Prometheus, Gettext, Fontconfig
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    ffmpeg \
    curl \
    gettext-base \
    nginx \
    supervisor \
    prometheus \
    adduser \
    libfontconfig1 \
    && rm -rf /var/lib/apt/lists/*

# Install lightweight standalone Grafana (Multi-Arch: Apple Silicon arm64 & Cloud amd64)
RUN ARCH=$(dpkg --print-architecture) \
    && curl -fsSL "https://dl.grafana.com/oss/release/grafana_10.0.3_${ARCH}.deb" -o /tmp/grafana.deb \
    && (dpkg -i /tmp/grafana.deb || apt-get install -fy) \
    && rm -f /tmp/grafana.deb

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Create necessary directories
RUN mkdir -p /app/data/mlflow /app/data/audio_cache /var/log /tmp/prometheus /var/lib/grafana/dashboards /etc/grafana/provisioning/dashboards /etc/grafana/provisioning/datasources /etc/prometheus

# Copy service configs
COPY nginx.conf.template /etc/nginx/nginx.conf.template
COPY supervisord.conf /etc/supervisor/conf.d/supervisord.conf
COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

# Copy Prometheus and Grafana provisioning assets
COPY monitoring/prometheus.yml /etc/prometheus/prometheus.yml
COPY monitoring/grafana_dashboard.json /var/lib/grafana/dashboards/dashboard.json
COPY monitoring/grafana_provisioning/dashboards/dashboards.yml /etc/grafana/provisioning/dashboards/dashboards.yml
COPY monitoring/grafana_provisioning/datasources/datasource.yml /etc/grafana/provisioning/datasources/datasource.yml

# Copy source code and application assets
COPY . .

# Expose default port
EXPOSE 8501

ENTRYPOINT ["/app/entrypoint.sh"]
