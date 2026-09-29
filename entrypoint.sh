#!/bin/bash
set -e

# Default to port 8501 or 80 if $PORT is not set
export PORT=${PORT:-8501}
export PATH=$PATH:/usr/sbin:/usr/bin:/usr/local/bin:/usr/share/grafana/bin

# Grafana Subpath & Security Environment Variables
export GF_SECURITY_ADMIN_USER=${GF_SECURITY_ADMIN_USER:-admin}
export GF_SECURITY_ADMIN_PASSWORD=${GF_SECURITY_ADMIN_PASSWORD:-admin}
export GF_SERVER_SERVE_FROM_SUB_PATH=true
export GF_SERVER_ROOT_URL="%(protocol)s://%(domain)s/grafana/"

# Internal Service URIs
export MLFLOW_TRACKING_URI="http://127.0.0.1:5000/mlflow"

echo "================================================================="
echo "🏏 Starting Autonomous Sports Commentary & Intelligence Stack"
echo "🌐 Unified Web Entrypoint on Port: $PORT"
echo "================================================================="

# Inject $PORT into nginx configuration
envsubst '$PORT' < /etc/nginx/nginx.conf.template > /etc/nginx/nginx.conf

# Ensure all runtime directories exist
mkdir -p /var/log/nginx /var/log/supervisor /run /tmp/prometheus /app/data/mlflow /app/data/audio_cache /var/lib/grafana/dashboards /var/lib/grafana/data /var/log/grafana

# Ensure match statistics database is populated
if [ ! -f /app/data/match_stats.db ] || [ ! -s /app/data/match_stats.db ]; then
    echo "[Entrypoint] Initializing match statistics database..."
    python /app/database/db_loader.py || true
fi

# Start all daemons via supervisor
exec /usr/bin/supervisord -c /etc/supervisor/conf.d/supervisord.conf
