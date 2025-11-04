# Base CUDA image
FROM nvidia/cuda:12.8.0-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    CUDA_HOME=/usr/local/cuda \
    PATH=/opt/conda/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# System deps (remove build-essential & cmake - not needed for runtime)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl wget git ca-certificates \
    libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 libxrender-dev \
    libgomp1 libavcodec-dev libavformat-dev libswscale-dev \
    libv4l-dev libxvidcore-dev libx264-dev \
    libjpeg-dev libpng-dev libtiff-dev ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Install Miniconda
RUN wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh \
    && bash miniconda.sh -b -p /opt/conda \
    && rm miniconda.sh \
    && conda clean -afy

# Accept Anaconda ToS and create conda env python 3.9
RUN conda config --set channel_priority flexible && \
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main && \
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r && \
    conda create -y -n app python=3.9 && \
    conda clean -afy

ENV CONDA_DEFAULT_ENV=app

# Upgrade pip and install uv in conda environment
RUN /opt/conda/envs/app/bin/pip install --upgrade pip setuptools wheel && \
    /opt/conda/envs/app/bin/pip install uv

COPY requirements.txt .

# Activate conda env and install PyTorch with CUDA 12.8 support using uv
RUN /bin/bash -c "source /opt/conda/etc/profile.d/conda.sh && \
    conda activate app && \
    uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128 --no-cache-dir"

# Activate conda env and install other dependencies using uv
RUN /bin/bash -c "source /opt/conda/etc/profile.d/conda.sh && \
    conda activate app && \
    uv pip install -r requirements.txt --no-cache-dir"

# Copy project
COPY . .

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

ENV UNISIGN_CHECKPOINT=/app/pretrained_weight/best_checkpoint.pth \
    UNISIGN_RGB_SUPPORT=true \
    CUDA_VISIBLE_DEVICES=0

# Use direct path instead of conda run for better performance
CMD ["/opt/conda/envs/app/bin/python", "demo/api_server.py"]
