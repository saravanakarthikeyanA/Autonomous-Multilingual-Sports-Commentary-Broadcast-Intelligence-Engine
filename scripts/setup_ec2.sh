#!/usr/bin/env bash
# ==============================================================================
# AWS EC2 Setup & Bootstrap Script
# Autonomous Multilingual Sports Commentary & Broadcast Intelligence Engine
# ==============================================================================
set -euo pipefail

echo "================================================================="
echo "🏏 Initializing Autonomous Sports Commentary Stack on AWS EC2"
echo "================================================================="

# 1. Update and install core dependencies
echo "[1/7] 📦 Updating APT packages and installing system prerequisites..."
sudo apt-get update -y
sudo apt-get install -y --no-install-recommends \
    apt-transport-https \
    ca-certificates \
    curl \
    gnupg \
    lsb-release \
    git \
    ufw \
    ffmpeg \
    libsndfile1 \
    python3-pip \
    python3-venv

# 2. Install Docker Engine & Docker Compose Plugin (Official Docker Repo)
echo "[2/7] 🐳 Installing Docker Engine and Docker Compose plugin..."
if ! command -v docker &> /dev/null; then
    sudo install -m 0755 -d /etc/apt/keyrings
    curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg --yes
    sudo chmod a+r /etc/apt/keyrings/docker.gpg

    echo \
      "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
      $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

    sudo apt-get update -y
    sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
    
    # Enable and start Docker daemon
    sudo systemctl enable docker
    sudo systemctl start docker
    echo "✅ Docker installed successfully."
else
    echo "✅ Docker already installed."
fi

# Add current user and ubuntu to docker group
sudo usermod -aG docker "$USER" || true
if id "ubuntu" &>/dev/null; then
    sudo usermod -aG docker ubuntu || true
fi

# 3. Configure 2GB Swap Memory (Protects against OOM spikes during heavy LLM/TTS bursts)
echo "[3/7] 🧠 Configuring 2GB Swap Memory for EC2 instance stability..."
if [ ! -f /swapfile ]; then
    sudo fallocate -l 2G /swapfile || sudo dd if=/dev/zero of=/swapfile bs=1M count=2048
    sudo chmod 600 /swapfile
    sudo mkswap /swapfile
    sudo swapon /swapfile
    if ! grep -q '/swapfile' /etc/fstab; then
        echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
    fi
    sudo sysctl vm.swappiness=10
    echo "✅ 2GB Swap enabled."
else
    echo "✅ Swap already active."
fi

# 4. Configure UFW Firewall rules
echo "[4/7] 🛡️  Configuring UFW Firewall..."
sudo ufw allow 22/tcp comment 'SSH' || true
sudo ufw allow 80/tcp comment 'HTTP' || true
sudo ufw allow 443/tcp comment 'HTTPS' || true
sudo ufw allow 8501/tcp comment 'Streamlit Broadcast Hub' || true
sudo ufw allow 3000/tcp comment 'Grafana Dashboards' || true
sudo ufw allow 5001/tcp comment 'MLflow Tracking' || true
sudo ufw allow 9090/tcp comment 'Prometheus Metrics' || true
sudo ufw --force enable || true
echo "✅ Firewall rules configured."

# 5. Setup Project Environment Variables (.env)
echo "[5/7] ⚙️  Configuring Application Environment (.env)..."
if [ ! -f .env ]; then
    if [ -f .env.example ]; then
        cp .env.example .env
        echo "⚠️  Created .env from .env.example. Please ensure GROQ_API_KEY is set in .env"
    else
        cat << 'EOF' > .env
GROQ_API_KEY=your_groq_api_key_here
GROQ_MODEL=qwen/qwen3.8-27b
GROQ_FAST_MODEL=qwen/qwen3.8-27b
HF_TOKEN=
DB_PATH=data/match_stats.db
MLFLOW_TRACKING_URI=http://mlflow:5000
MLFLOW_EXPERIMENT_NAME=Cricket_Commentary_Evaluation
EOF
        echo "⚠️  Created default .env file."
    fi
else
    echo "✅ Existing .env file found."
fi

# 6. Ensure data directory & local database initialization
echo "[6/7] 📊 Initializing match data and storage directories..."
mkdir -p data/mlflow/artifacts data/audio_cache

# 7. Build and start Multi-Container Stack via Docker Compose
echo "[7/7] 🚀 Building and starting Docker Compose containers..."
sudo docker compose down --remove-orphans || true
sudo docker compose up -d --build

# Health Status & URLs
PUBLIC_IP=$(curl -s http://checkip.amazonaws.com || curl -s https://api.ipify.org || echo "YOUR_EC2_PUBLIC_IP")

echo ""
echo "================================================================="
echo "🎉 Autonomous Multilingual Commentary Stack is LIVE on AWS EC2!"
echo "================================================================="
echo "🏏 Streamlit Broadcast UI : http://${PUBLIC_IP}:8501"
echo "📈 Grafana Dashboards     : http://${PUBLIC_IP}:3000 (admin / admin)"
echo "🧪 MLflow Evaluation Hub  : http://${PUBLIC_IP}:5001"
echo "📊 Prometheus Metrics UI  : http://${PUBLIC_IP}:9090"
echo "================================================================="
echo ""
echo "Container Status:"
sudo docker compose ps
