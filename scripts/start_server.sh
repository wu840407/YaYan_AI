#!/usr/bin/env bash
# 啟動 YaYan-AI v4.5 伺服器（離線模式）
# 用法: bash scripts/start_server.sh
set -euo pipefail

cd "$(dirname "$0")/.."

# ===== 全域離線旗標 =====
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export MODELSCOPE_OFFLINE=1
export DO_NOT_TRACK=1
export VLLM_NO_USAGE_STATS=1
export AWQ_USE_TRITON=0
export YAYAN_MODELS_ROOT="${YAYAN_MODELS_ROOT:-/data/ai_models}"
export YAYAN_INPUT_DIR="${YAYAN_INPUT_DIR:-/data/input_audio}"
export YAYAN_OUTPUT_DIR="${YAYAN_OUTPUT_DIR:-/data/output_text}"

# 注意：不要設 CUDA_HOME 指向 conda env，那裡沒有完整 CUDA Toolkit。
# 系統若有真正的 CUDA Toolkit (例 /usr/local/cuda)，可在啟動前手動 export。

mkdir -p "$YAYAN_INPUT_DIR" "$YAYAN_OUTPUT_DIR"

echo "============================================"
echo " YaYan-AI v4.5  Server (Offline Mode)"
echo " Models: $YAYAN_MODELS_ROOT"
echo " Input : $YAYAN_INPUT_DIR"
echo " Output: $YAYAN_OUTPUT_DIR"
echo "============================================"

# 模型完整性檢查
python scripts/verify_models.py

# 偵測 backend
LLM_BACKEND="$(python -c "from yayan.config import CONFIG; print(CONFIG['llm'].get('backend','transformers').lower())")"
echo "ℹ️  LLM backend = $LLM_BACKEND"

VLLM_PID=""
cleanup() {
  if [ -n "$VLLM_PID" ] && kill -0 "$VLLM_PID" 2>/dev/null; then
    echo "🛑 停止 vllm serve (PID=$VLLM_PID)"
    kill "$VLLM_PID" 2>/dev/null || true
    wait "$VLLM_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

if [ "$LLM_BACKEND" = "openai" ]; then
  LLM_DIR="$YAYAN_MODELS_ROOT/YaYan_Reasoner"
  LLM_PORT="$(python -c "from yayan.config import CONFIG; import urllib.parse; u=urllib.parse.urlparse(CONFIG['llm'].get('openai_base_url','http://127.0.0.1:8001/v1')); print(u.port or 8001)")"
  SERVED_NAME="$(python -c "from yayan.config import CONFIG; print(CONFIG['llm'].get('served_model_name','YaYan_Reasoner'))")"
  MAX_LEN="$(python -c "from yayan.config import CONFIG; print(CONFIG['llm'].get('max_model_len',4096))")"
  GPU_UTIL="$(python -c "from yayan.config import CONFIG; print(CONFIG['llm'].get('gpu_memory_utilization',0.85))")"

  if ! command -v vllm >/dev/null 2>&1; then
    echo "❌ 設定為 backend: openai 但找不到 vllm 指令。"
    echo "   請：pip install 'vllm==0.6.4.post1' 'openai>=1.40.0' 'numpy<2.0'"
    exit 1
  fi

  # 檢查 vllm 版本，避免拿到 v1 引擎
  VLLM_VER="$(python -c 'import vllm; print(vllm.__version__)' 2>/dev/null || echo "unknown")"
  echo "ℹ️  偵測到 vllm 版本: $VLLM_VER"
  case "$VLLM_VER" in
    0.6.*) ;;  # OK
    *)
      echo "❌ vllm 版本 ($VLLM_VER) 不是 0.6.x，無法在 Turing 上穩定執行。"
      echo "   請：pip install --force-reinstall 'vllm==0.6.4.post1' 'numpy<2.0'"
      exit 1
      ;;
  esac

  VLLM_LOG="$YAYAN_OUTPUT_DIR/../vllm_serve.log"
  echo "🚀 啟動 vllm serve sidecar 於 GPU 1，port=$LLM_PORT"
  # 注意：CUDA_VISIBLE_DEVICES=1 限定 vLLM 只看到 GPU 1（ASR 用 GPU 0 不受影響）
  CUDA_VISIBLE_DEVICES=1 \
  VLLM_ATTENTION_BACKEND=XFORMERS \
  HF_HUB_OFFLINE=1 \
  TRANSFORMERS_OFFLINE=1 \
  DO_NOT_TRACK=1 \
  VLLM_NO_USAGE_STATS=1 \
  vllm serve "$LLM_DIR" \
      --quantization awq \
      --dtype float16 \
      --max-model-len "$MAX_LEN" \
      --gpu-memory-utilization "$GPU_UTIL" \
      --enforce-eager \
      --tensor-parallel-size 1 \
      --served-model-name "$SERVED_NAME" \
      --host 127.0.0.1 \
      --port "$LLM_PORT" \
      --trust-remote-code \
      > "$VLLM_LOG" 2>&1 &
  VLLM_PID=$!
  echo "   vllm PID=$VLLM_PID, log=$VLLM_LOG"

  echo "⏳ 等待 vllm serve 健康檢查 (最多 180 秒)…"
  for i in $(seq 1 60); do
    if curl -fsS "http://127.0.0.1:$LLM_PORT/v1/models" >/dev/null 2>&1; then
      echo "✅ vllm serve 已就緒"
      break
    fi
    if ! kill -0 "$VLLM_PID" 2>/dev/null; then
      echo "❌ vllm serve 已退出，最後 50 行 log："
      tail -n 50 "$VLLM_LOG" || true
      exit 1
    fi
    sleep 3
  done
fi

# 啟動主 app（Gradio）
exec python app_rtx6000.py
