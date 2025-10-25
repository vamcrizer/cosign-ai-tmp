# Use NVIDIA CUDA 12.8 base image with Ubuntu 22.04
FROM nvidia/cuda:12.8.0-runtime-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    CUDA_HOME=/usr/local/cuda \
    PATH=/usr/local/cuda/bin:$PATH \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    # Python and build essentials
    python3.9 \
    python3.9-dev \
    python3-pip \
    # Build tools
    build-essential \
    cmake \
    git \
    wget \
    curl \
    ca-certificates \
    # OpenCV dependencies
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    # Video codec libraries
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libv4l-dev \
    libxvidcore-dev \
    libx264-dev \
    # Image libraries
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    # Additional utilities
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Create symbolic link for python
RUN ln -sf /usr/bin/python3.10 /usr/bin/python && \
    ln -sf /usr/bin/pip3 /usr/bin/pip

# Upgrade pip
RUN pip install --upgrade pip setuptools wheel

# Install uv (fast Python package installer)
RUN pip install uv

# Install PyTorch and torchvision with CUDA 12.8 support using uv
RUN uv pip install --system torch torchvision --index-url https://download.pytorch.org/whl/cu128

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies from requirements.txt
RUN uv pip install --system -r requirements.txt

# Copy project files
COPY . .

# Create necessary directories
RUN mkdir -p /app/test \
    /app/logs \
    /app/temp

# Set permissions
RUN chmod -R 755 /app

# Expose port for FastAPI
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default environment variables (can be overridden)
ENV UNISIGN_CHECKPOINT=/app/pretrained_weight/best_checkpoint.pth \
    UNISIGN_RGB_SUPPORT=true \
    CUDA_VISIBLE_DEVICES=0

# Default command: run FastAPI server
CMD ["python", "demo/api_server.py"]
