#!/usr/bin/env bash
# 在「連網機器」先打包 wheels：
#   bash scripts/install_offline.sh package
# 把 offline_packages/ 整個搬到伺服器後，於伺服器執行：
#   bash scripts/install_offline.sh install

set -euo pipefail

MODE="${1:-install}"
WHEEL_DIR="$(pwd)/offline_packages"
REQ="requirements.txt"

case "$MODE" in
  package)
    echo "📦 打包 wheels 到 $WHEEL_DIR"
    mkdir -p "$WHEEL_DIR"
    pip download -r "$REQ" -d "$WHEEL_DIR" \
      --extra-index-url https://download.pytorch.org/whl/cu121
    echo "✅ 打包完成。大小："
    du -sh "$WHEEL_DIR"
    echo ""
    echo "下一步："
    echo "  rsync -av $WHEEL_DIR/ user@server:$WHEEL_DIR/"
    echo "  rsync -av <repo>/  user@server:/opt/yayan/"
    ;;
  install)
    if [ ! -d "$WHEEL_DIR" ]; then
      echo "❌ 找不到 $WHEEL_DIR，請先在連網機器跑 package 模式。"
      exit 1
    fi
    echo "📥 離線安裝中…"
    pip install --no-index --find-links "$WHEEL_DIR" -r "$REQ"
    echo "✅ 安裝完成"
    ;;
  *)
    echo "用法: $0 [package|install]"
    exit 1
    ;;
esac
