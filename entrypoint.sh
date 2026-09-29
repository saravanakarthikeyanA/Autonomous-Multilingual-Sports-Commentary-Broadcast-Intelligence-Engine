#!/bin/bash
set -e

echo "================================================================="
echo "🏏 Starting Autonomous Sports Commentary Platform on AWS"
echo "🌐 Streamlit Broadcast Hub: Port 8501"
echo "📊 Prometheus Metrics Exporter: Port 8000"
echo "================================================================="

# Ensure runtime directories exist
mkdir -p /app/data/mlflow /app/data/audio_cache /root/.streamlit /app/.streamlit

# Guarantee Streamlit headless configuration & suppress interactive prompt
echo '[general]' > /root/.streamlit/credentials.toml
echo 'email = ""' >> /root/.streamlit/credentials.toml
cp /root/.streamlit/credentials.toml /app/.streamlit/credentials.toml

echo '[server]' > /root/.streamlit/config.toml
echo 'headless = true' >> /root/.streamlit/config.toml
echo 'enableCORS = false' >> /root/.streamlit/config.toml
echo 'enableXsrfProtection = false' >> /root/.streamlit/config.toml
echo 'fileWatcherType = "none"' >> /root/.streamlit/config.toml
echo '[browser]' >> /root/.streamlit/config.toml
echo 'gatherUsageStats = false' >> /root/.streamlit/config.toml
cp /root/.streamlit/config.toml /app/.streamlit/config.toml

# Ensure match statistics database is populated
if [ ! -f /app/data/match_stats.db ] || [ ! -s /app/data/match_stats.db ]; then
    echo "[Entrypoint] Initializing match statistics database..."
    python /app/database/db_loader.py || true
fi

# Launch Streamlit Application directly on Port 8501
exec streamlit run app.py \
    --server.port=8501 \
    --server.address=0.0.0.0 \
    --server.enableCORS=false \
    --server.enableXsrfProtection=false \
    --server.headless=true \
    --browser.gatherUsageStats=false \
    --server.fileWatcherType=none
