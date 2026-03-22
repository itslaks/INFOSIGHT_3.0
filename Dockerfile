# ============================================================================
# INFOSIGHT 3.0 - Production Dockerfile for NVIDIA GPU-enabled Deployment
# ============================================================================
# This Dockerfile is optimized for JarvisLabs.ai and cloud GPU environments
# Base: Ubuntu 22.04 with CUDA 11.8 runtime for GPU acceleration
# ============================================================================

FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04

# Set environment variables for non-interactive installation
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONIOENCODING=utf-8

# ============================================================================
# SYSTEM DEPENDENCIES - Layer 1: Critical System Packages
# ============================================================================
RUN apt-get update && apt-get install -y --no-install-recommends \
    # Core utilities
    build-essential \
    cmake \
    curl \
    wget \
    git \
    ca-certificates \
    # Python runtime
    python3.10 \
    python3.10-venv \
    python3.10-dev \
    python3-pip \
    # Security tools for network scanning
    nmap \
    # Audio processing dependencies
    ffmpeg \
    libsndfile1 \
    sox \
    libsox-dev \
    libopus-dev \
    # Image processing
    tesseract-ocr \
    libtesseract-dev \
    # SSL/TLS
    openssl \
    libssl-dev \
    # Database
    sqlite3 \
    libsqlite3-dev \
    # Compression
    unzip \
    # Additional required libs
    libffi-dev \
    libopenblas-dev \
    liblapack-dev \
    gfortran \
    libblas-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# ============================================================================
# SYSTEM DEPENDENCIES - Layer 2: Optional but Recommended
# ============================================================================
RUN apt-get update && apt-get install -y --no-install-recommends \
    # TOR support (for DONNA AI dark web module)
    tor \
    privoxy \
    # Additional security tools
    whois \
    traceroute \
    # Monitoring and debugging
    htop \
    nano \
    vim \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# ============================================================================
# PYTHON SETUP - Layer 3: Python 3.10 Configuration
# ============================================================================
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 && \
    python -m pip install --upgrade pip setuptools wheel

# ============================================================================
# WORKING DIRECTORY SETUP
# ============================================================================
WORKDIR /app

# ============================================================================
# COPY APPLICATION CODE
# ============================================================================
# Copy requirements first for better Docker caching
COPY requirements.txt requirements.txt
COPY requirements_frozen.txt requirements_frozen.txt

# Copy entire project
COPY . .

# ============================================================================
# PYTHON DEPENDENCIES - Layer 4: Main Python Packages
# ============================================================================
# Install frozen dependencies directly to ensure consistency
RUN pip install --upgrade pip setuptools wheel && \
    pip install -r requirements_frozen.txt 2>&1 | grep -v "already satisfied" || true

# ============================================================================
# DIRECTORY AND PERMISSIONS - Layer 5: Setup Runtime Directories
# ============================================================================
RUN mkdir -p /app/data \
    /app/logs \
    /app/audio/cache \
    /app/audio/temp \
    /app/temp \
    /app/static/generated_images \
    /app/models/runs/detect \
    /app/.cache \
    /root/.cache \
    /root/.ollama && \
    chmod -R 755 /app

# ============================================================================
# ENVIRONMENT VARIABLES - Layer 6: Runtime Configuration
# ============================================================================
# Python optimization
ENV TF_CPP_MIN_LOG_LEVEL=3 \
    TF_ENABLE_ONEDNN_OPTS=0 \
    PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512 \
    CUDA_VISIBLE_DEVICES=0 \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

# Hugging Face configuration (optimized for JarvisLabs)
ENV HF_HOME=/app/.cache/huggingface \
    TORCH_HOME=/app/.cache/torch \
    TRANSFORMERS_CACHE=/app/.cache/transformers \
    OLLAMA_MODELS=/app/llama/models \
    OLLAMA_HOST=127.0.0.1:11434

# Application configuration
ENV FLASK_APP=server.py \
    FLASK_ENV=production \
    PYTHONPATH=/app

# ============================================================================
# EXPOSE PORTS
# ============================================================================
# Main Flask application
EXPOSE 5000

# Ollama local LLM server (fallback)
EXPOSE 11434

# Individual blueprint ports (optional for direct access)
EXPOSE 5001 5002 5003 5004 5005 5006 5007 5008 5009 5010 5011 5012 5013

# ============================================================================
# HEALTH CHECK - Layer 7: Application Health Verification
# ============================================================================
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:5000/ || exit 1

# ============================================================================
# MODEL PREPARATION - Layer 8: Pre-download Critical Models
# ============================================================================
# Note: Models are lazy-loaded, but we can pre-download to speed up startup
# Uncomment if you want to pre-load models during build:
# RUN python -c "from transformers import AutoModel, AutoTokenizer; AutoModel.from_pretrained('Salesforce/blip-image-captioning-large')" && \
#     python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}')"

# ============================================================================
# STARTUP SCRIPT - Layer 9: Container Entrypoint
# ============================================================================
# Create a startup script that handles initialization
RUN cat > /app/start.sh << 'EOF'
#!/bin/bash
set -e

echo "=========================================="
echo "INFOSIGHT 3.0 - Container Startup"
echo "=========================================="

# Display system info
echo "Environment Info:"
echo "  Python: $(python --version)"
echo "  CUDA: $CUDA_VISIBLE_DEVICES"
echo "  Working Dir: $(pwd)"
echo ""

# Check API tokens
echo "Checking API Configuration..."
python check_token.py 2>/dev/null || echo "  ⚠️ HF_API_TOKEN validation skipped (optional)"
echo ""

# Optional: Start Ollama service (local LLM fallback) in background
if command -v ollama &> /dev/null; then
    echo "Starting Ollama service..."
    ollama serve &
    OLLAMA_PID=$!
    sleep 3
    echo "  ✓ Ollama started (PID: $OLLAMA_PID)"
else
    echo "  ℹ️ Ollama not available (will use cloud LLM)"
fi

# Optional: Start TOR service for DONNA AI
if command -v tor &> /dev/null; then
    echo "Starting TOR service..."
    service tor start 2>/dev/null || echo "  ℹ️ TOR not available"
fi

echo ""
echo "=========================================="
echo "Starting INFOSIGHT 3.0 Flask Server"
echo "=========================================="

# Start the main application
exec python server.py
EOF

chmod +x /app/start.sh

# ============================================================================
# ENTRYPOINT CONFIGURATION
# ============================================================================
ENTRYPOINT ["/bin/bash"]
CMD ["/app/start.sh"]

# ============================================================================
# BUILD METADATA - Labels for container tracking
# ============================================================================
LABEL maintainer="INFOSIGHT Development Team" \
      version="3.0" \
      description="INFOSIGHT 3.0 - Advanced Cybersecurity & AI Intelligence Suite" \
      platform="JarvisLabs.ai / NVIDIA GPU" \
      python="3.10" \
      cuda="11.8"
