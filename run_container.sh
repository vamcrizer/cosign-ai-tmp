#!/bin/bash

# Uni-Sign Docker Container Runner (Persistent Version)
# Giữ lại container sau khi dừng, để có thể restart lại mà không mất dữ liệu

set -e  # Dừng nếu có lỗi

# Cấu hình
IMAGE_NAME="vamcrizer/cosign:v1"
CONTAINER_NAME="cosignai-server"
HOST_PORT=6336
CONTAINER_PORT=8000

# Màu cho output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo "================================================"
echo "  Starting Uni-Sign Container (Persistent Mode)"
echo "================================================"

# Kiểm tra Docker đang chạy
if ! docker info > /dev/null 2>&1; then
    echo -e "${RED}Error: Docker chưa khởi động!${NC}"
    exit 1
fi

# Kiểm tra GPU
if ! docker run --rm --gpus all nvidia/cuda:12.8.0-runtime-ubuntu22.04 nvidia-smi > /dev/null 2>&1; then
    echo -e "${YELLOW}Warning: Không phát hiện GPU hoặc chưa cài NVIDIA Container Toolkit${NC}"
    read -p "Tiếp tục chạy CPU-only? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
    GPU_FLAG=""
else
    GPU_FLAG="--gpus all"
fi

# Kiểm tra image
if ! docker image inspect ${IMAGE_NAME} > /dev/null 2>&1; then
    echo -e "${RED}Error: Image '${IMAGE_NAME}' chưa tồn tại!${NC}"
    echo "Hãy build image trước với: docker build -t ${IMAGE_NAME} ."
    exit 1
fi

# Kiểm tra thư mục pretrained_weight
if [ ! -d "$(pwd)/pretrained_weight" ]; then
    echo -e "${YELLOW}Warning: Không tìm thấy thư mục pretrained_weight${NC}"
    mkdir -p "$(pwd)/pretrained_weight"
fi

# Kiểm tra port
if lsof -Pi :${HOST_PORT} -sTCP:LISTEN -t >/dev/null 2>&1 ; then
    echo -e "${RED}Error: Port ${HOST_PORT} đang được sử dụng!${NC}"
    exit 1
fi

# Nếu container đã tồn tại -> hỏi khởi động lại hay tạo mới
if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    echo -e "${YELLOW}Container '${CONTAINER_NAME}' đã tồn tại.${NC}"
    read -p "Bạn muốn khởi động lại container cũ? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo -e "${GREEN}Đang khởi động lại container...${NC}"
        docker start -ai ${CONTAINER_NAME}
        exit 0
    else
        echo -e "${YELLOW}Xóa container cũ...${NC}"
        docker rm -f ${CONTAINER_NAME} > /dev/null 2>&1
    fi
fi

# Chạy container (giữ lại sau khi dừng)
echo -e "${GREEN}Đang khởi động container...${NC}"
docker run -it \
  --name ${CONTAINER_NAME} \
  ${GPU_FLAG} \
  -p ${HOST_PORT}:${CONTAINER_PORT} \
  -v "$(pwd)/pretrained_weight:/app/pretrained_weight" \
  ${IMAGE_NAME}

echo -e "${GREEN}Container đã dừng nhưng vẫn được giữ lại.${NC}"
echo "Bạn có thể chạy lại bằng: docker start -ai ${CONTAINER_NAME}"
