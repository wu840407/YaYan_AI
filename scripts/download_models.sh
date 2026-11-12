#!/usr/bin/env bash
# 在「連網機器」執行，會把所有 YaYan-AI v4.5 需要的模型下載到 $MODELS_ROOT
#
# 用法:
#   bash download_models.sh                    # 下載核心模型
#   bash download_models.sh --with-diarize     # 額外下載 pyannote 三件套（需 HF token）
#   MODELS_ROOT=/data/ai_models bash download_models.sh

set -euo pipefail

MODELS_ROOT="${MODELS_ROOT:-/data/ai_models}"
WITH_DIARIZE=0
for arg in "$@"; do
  case "$arg" in
    --with-diarize) WITH_DIARIZE=1 ;;
  esac
done

mkdir -p "$MODELS_ROOT"

echo "============================================"
echo " YaYan-AI v4.5  模型下載"
echo " 目標路徑：$MODELS_ROOT"
echo "============================================"

if command -v hf >/dev/null 2>&1; then
  HF_CLI="hf"
elif command -v huggingface-cli >/dev/null 2>&1; then
  HF_CLI="huggingface-cli"
else
  echo "❌ 找不到 hf 或 huggingface-cli"
  exit 1
fi
echo "ℹ️  使用 $HF_CLI 下載"

download_to() {
  local repo="$1"
  local dest="$MODELS_ROOT/$2"
  if [ -d "$dest" ] && [ -n "$(ls -A "$dest" 2>/dev/null || true)" ]; then
    echo "⏩ 已存在，略過: $dest"
    return
  fi
  echo "⬇️  下載 $repo → $dest"
  if [ "$HF_CLI" = "hf" ]; then
    hf download "$repo" --local-dir "$dest"
  else
    huggingface-cli download "$repo" --local-dir "$dest"
  fi
}

# === LLM ===
download_to "Qwen/Qwen3-32B-AWQ" "YaYan_Reasoner"

# === ASR ===
download_to "FunAudioLLM/SenseVoiceSmall" "YaYan_ASR_Mandarin"
download_to "DataoceanAI/dolphin-base"    "YaYan_ASR_Eastern"
download_to "openai/whisper-large-v3"     "YaYan_ASR_Global"

# === LID ===
download_to "speechbrain/lang-id-voxlingua107-ecapa" "YaYan_LID"

# === silero-vad: pip 內建權重，不在此下載 ===

# === Diarization 三件套 ===
if [ "$WITH_DIARIZE" = "1" ]; then
  if [ -z "${HF_TOKEN:-}" ]; then
    echo "❌ 需要 HF_TOKEN 才能下載 pyannote。請先："
    echo "    export HF_TOKEN=hf_xxx"
    echo "    並在瀏覽器到下列三個 repo 各按一次 Agree to license："
    echo "      https://huggingface.co/pyannote/speaker-diarization-3.1"
    echo "      https://huggingface.co/pyannote/segmentation-3.0"
    echo "      https://huggingface.co/pyannote/wespeaker-voxceleb-resnet34-LM"
    exit 1
  fi
  export HUGGINGFACE_HUB_TOKEN="$HF_TOKEN"

  # Pipeline 設定（YAML 裡引用下面兩個）
  download_to "pyannote/speaker-diarization-3.1"          "YaYan_Diarize"
  # 語音分段（5MB）
  download_to "pyannote/segmentation-3.0"                 "YaYan_Diarize_Seg"
  # 說話人 embedding（28MB）
  download_to "pyannote/wespeaker-voxceleb-resnet34-LM"   "YaYan_Diarize_Embed"
fi

echo "============================================"
echo " ✅ 模型下載完成"
echo "    總大小："
du -sh "$MODELS_ROOT" || true
echo "============================================"
