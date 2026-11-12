#!/usr/bin/env bash
# YaYan-AI 生產部署腳本 — v4.6
# 用法：bash deploy/install_production.sh

set -euo pipefail

PROJECT_DIR="/data/AI_Project"
LOGS_DIR="/data/yayan_logs"

# ── 部署前改這兩項（內部主機資訊，勿提交真值）──
SERVER_FQDN="your-server.example.com"   # 對外存取網域名
SERVER_HOSTNAME="your-server"           # 短主機名（也是憑證檔名前綴）
# ─────────────────────────────────────────
CERT_FILE="/etc/nginx/ssl/${SERVER_HOSTNAME}-fullchain.crt"

echo "============================================="
echo "  YaYan-AI 生產部署"
echo "============================================="

# ── Step 1: 建立 log 目錄 ──
sudo mkdir -p "$LOGS_DIR"
sudo chown administrator:administrator "$LOGS_DIR"
echo "✅ Log dir: $LOGS_DIR"

# ── Step 2: 安裝 systemd 服務 ──
echo ""
echo "▶ 安裝 systemd service..."
sudo cp "$PROJECT_DIR/deploy/yayan.service" /etc/systemd/system/yayan.service
sudo systemctl daemon-reload

if systemctl is-active --quiet yayan; then
    sudo systemctl stop yayan
fi

sudo systemctl enable yayan
sudo systemctl start yayan

echo "  等 60 秒讓 LLM 載入..."
sleep 60

if systemctl is-active --quiet yayan; then
    echo "✅ yayan.service 啟動成功"
else
    echo "❌ yayan.service 啟動失敗，看 log: sudo journalctl -u yayan -e"
    sudo systemctl status yayan --no-pager
    exit 1
fi

# ── Step 3: 安裝 Nginx ──
echo ""
echo "▶ 安裝 Nginx..."
if ! command -v nginx >/dev/null 2>&1; then
    sudo apt update
    sudo apt install -y nginx
fi

# ── Step 4: 配 Nginx 反向代理 ──
echo ""
echo "▶ 配置 Nginx 反向代理..."
sudo cp "$PROJECT_DIR/deploy/nginx-yayan.conf" /etc/nginx/sites-available/yayan
sudo ln -sf /etc/nginx/sites-available/yayan /etc/nginx/sites-enabled/yayan
sudo rm -f /etc/nginx/sites-enabled/default

# 檢查憑證是否存在
if [ ! -f "$CERT_FILE" ]; then
    echo ""
    echo "⚠️  Nginx 設定已就位，但 $CERT_FILE 不存在"
    echo "   請先放好 AD CA 憑證再 reload Nginx："
    echo "     deploy/install_ad_cert.sh /path/to/server.pfx"
    echo ""
    exit 0
fi

# ── Step 5: 防火牆 ──
echo ""
echo "▶ 配置防火牆..."
sudo ufw allow 80/tcp comment "HTTP redirect"
sudo ufw allow 443/tcp comment "HTTPS YaYan"
sudo ufw deny 7860/tcp comment "Block direct Gradio access"

# ── Step 6: 測試 + reload Nginx ──
echo ""
echo "▶ 測試 Nginx 設定..."
sudo nginx -t

echo ""
echo "▶ 啟動 Nginx..."
sudo systemctl enable nginx
sudo systemctl reload nginx || sudo systemctl start nginx

echo ""
echo "============================================="
echo "✅ 部署完成"
echo "============================================="
echo ""
echo "服務管理："
echo "  sudo systemctl status yayan"
echo "  sudo systemctl restart yayan"
echo "  sudo journalctl -u yayan -f         # 即時 log"
echo "  tail -f /data/yayan_logs/yayan.log  # 應用 log"
echo ""
echo "存取："
echo "  https://$SERVER_FQDN/"
echo "  https://$SERVER_HOSTNAME/"
echo ""
