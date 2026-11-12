# YaYan-AI v4.5 — RTX6000 Server Edition
# Base: NVIDIA CUDA 12.1 + cuDNN + Ubuntu 22.04 + Python 3.10
# 用法（連網機器）：
#   cd <repo-root>
#   docker build -t yayan-ai:4.5 -f docker/Dockerfile .
#   docker save yayan-ai:4.5 -o yayan-ai-4.5.tar
#   # 把 tar 與 ai_models/ 一起搬到離線伺服器
# 離線伺服器：
#   docker load -i yayan-ai-4.5.tar
#   docker compose -f docker/docker-compose.yml up -d

FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    HF_DATASETS_OFFLINE=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.10 python3-pip python3.10-venv \
        ffmpeg libsndfile1 sox libsox-fmt-all \
        git curl ca-certificates tzdata locales \
    && ln -sf /usr/bin/python3.10 /usr/bin/python \
    && locale-gen zh_TW.UTF-8 \
    && rm -rf /var/lib/apt/lists/*

ENV LANG=zh_TW.UTF-8 LC_ALL=zh_TW.UTF-8

WORKDIR /opt/yayan

COPY requirements.txt /opt/yayan/requirements.txt
RUN python -m pip install --upgrade pip && \
    python -m pip install \
      --extra-index-url https://download.pytorch.org/whl/cu121 \
      -r /opt/yayan/requirements.txt

COPY yayan/        /opt/yayan/yayan/
COPY configs/      /opt/yayan/configs/
COPY scripts/      /opt/yayan/scripts/
COPY app_rtx6000.py auto_batch_rtx6000.py /opt/yayan/

RUN chmod +x /opt/yayan/scripts/*.sh

EXPOSE 7860

CMD ["python", "app_rtx6000.py"]
