#!/bin/bash

# Uni-Sign Docker Container Runner
# This script runs the Uni-Sign container with GPU support

set -e  # Exit on error

# Configuration
IMAGE_NAME="unisign:latest"
CONTAINER_NAME="unisign-server"
HOST_PORT=6336
CONTAINER_PORT=8000

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo "================================================"
echo "  Starting Uni-Sign Container"
echo "================================================"

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo -e "${RED}Error: Docker is not running!${NC}"
    exit 1
fi

# Check if NVIDIA GPU is available
if ! docker run --rm --gpus all nvidia/cuda:12.8.0-runtime-ubuntu22.04 nvidia-smi > /dev/null 2>&1; then
    echo -e "${YELLOW}Warning: GPU not detected or NVIDIA Container Toolkit not installed${NC}"
    echo -e "${YELLOW}Container will run but may not use GPU acceleration${NC}"
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Check if image exists
if ! docker image inspect ${IMAGE_NAME} > /dev/null 2>&1; then
    echo -e "${RED}Error: Image '${IMAGE_NAME}' not found!${NC}"
    echo "Please build the image first with: docker build -t ${IMAGE_NAME} ."
    exit 1
fi

# Check if pretrained_weight directory exists
if [ ! -d "$(pwd)/pretrained_weight" ]; then
    echo -e "${YELLOW}Warning: pretrained_weight directory not found!${NC}"
    echo "Creating directory: $(pwd)/pretrained_weight"
    mkdir -p "$(pwd)/pretrained_weight"
fi

# Check if port is already in use
if lsof -Pi :${HOST_PORT} -sTCP:LISTEN -t >/dev/null 2>&1 ; then
    echo -e "${RED}Error: Port ${HOST_PORT} is already in use!${NC}"
    echo "Please stop the service using this port or change HOST_PORT in this script"
    exit 1
fi

# Stop and remove existing container if running
if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    echo -e "${YELLOW}Stopping and removing existing container...${NC}"
    docker stop ${CONTAINER_NAME} > /dev/null 2>&1 || true
    docker rm ${CONTAINER_NAME} > /dev/null 2>&1 || true
fi

echo -e "${GREEN}Starting container...${NC}"
echo "  Image: ${IMAGE_NAME}"
echo "  Port: http://localhost:${HOST_PORT}"
echo "  API Docs: http://localhost:${HOST_PORT}/docs"
echo "  Health: http://localhost:${HOST_PORT}/health"
echo ""
echo -e "${YELLOW}Press Ctrl+C to stop the container${NC}"
echo "================================================"
echo ""

# Run container
docker run --rm \
  --name ${CONTAINER_NAME} \
  --gpus all \
  -p ${HOST_PORT}:${CONTAINER_PORT} \
  -v "$(pwd)/pretrained_weight:/app/pretrained_weight" \
  ${IMAGE_NAME}
