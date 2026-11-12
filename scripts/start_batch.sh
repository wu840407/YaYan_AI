#!/usr/bin/env bash
# 啟動批次處理（持續監看新檔）
# 用法: bash scripts/start_batch.sh
export GRADIO_ANALYTICS_ENABLED=False
export GRADIO_DO_NOT_TRACK=1
set -euo pipefail

cd "$(dirname "$0")/.."

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export YAYAN_MODELS_ROOT="${YAYAN_MODELS_ROOT:-/data/ai_models}"
export YAYAN_INPUT_DIR="${YAYAN_INPUT_DIR:-/data/input_audio}"
export YAYAN_OUTPUT_DIR="${YAYAN_OUTPUT_DIR:-/data/output_text}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p "$YAYAN_INPUT_DIR" "$YAYAN_OUTPUT_DIR"

exec python auto_batch_rtx6000.py --watch
