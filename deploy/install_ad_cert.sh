#!/usr/bin/env bash
# 把 Windows AD CA 匯出的 PFX 憑證轉成 Nginx 可用的 PEM
# 用法：bash deploy/install_ad_cert.sh /path/to/server.pfx

set -euo pipefail

PFX_PATH="${1:-}"
SSL_DIR="/etc/nginx/ssl"

# ── 部署前改這幾項（內部主機資訊，勿提交真值）──
SERVER_FQDN="your-server.example.com"   # 伺服器完整網域名
SERVER_HOSTNAME="your-server"           # 短主機名（也當作 cert 檔名前綴）
SERVER_IP="192.0.2.10"                  # 伺服器實際 IP
ORG_NAME="Your-Org"                     # 組織代號（憑證 O=）
# ─────────────────────────────────────────

CN="$SERVER_HOSTNAME"                   # cert 檔名前綴

if [ -z "$PFX_PATH" ] || [ ! -f "$PFX_PATH" ]; then
    echo "用法：bash $0 /path/to/server.pfx"
    echo ""
    echo "如何取得 PFX："
    echo "  1. AD CS 主機建立 .inf 檔，內容範例："
    cat <<INF

  [Version]
  Signature="\$Windows NT\$"

  [NewRequest]
  Subject = "CN=$SERVER_FQDN, O=$ORG_NAME, C=TW"
  KeyLength = 2048
  KeyAlgorithm = RSA
  MachineKeySet = TRUE
  RequestType = PKCS10
  KeyUsage = 0xa0
  HashAlgorithm = SHA256
  ProviderName = "Microsoft RSA SChannel Cryptographic Provider"

  [Extensions]
  2.5.29.17 = "{text}"
  _continue_ = "DNS=$SERVER_FQDN&"
  _continue_ = "DNS=$SERVER_HOSTNAME&"
  _continue_ = "IPAddress=$SERVER_IP"

  [RequestAttributes]
  CertificateTemplate = WebServer

INF
    echo "  2. 在 AD CS server 跑："
    echo "       certreq -new server.inf server.csr"
    echo "       certreq -submit -config \"<CA-server>\\<CA-name>\" server.csr server.cer"
    echo "       certreq -accept server.cer"
    echo "  3. 開 certlm.msc → 個人 → 憑證 → 匯出（含私鑰）→ PFX"
    echo "  4. 把 PFX scp 到此 server，傳給本腳本"
    exit 1
fi

echo "============================================="
echo "  AD CA 憑證安裝"
echo "  Source: $PFX_PATH"
echo "  Target: $SSL_DIR"
echo "============================================="

sudo mkdir -p "$SSL_DIR"
sudo chmod 700 "$SSL_DIR"

echo ""
echo "▶ 轉換 PFX → PEM（會問你 PFX 匯出時設的密碼）..."
echo ""

# server cert (no key, no chain)
sudo openssl pkcs12 -in "$PFX_PATH" \
    -clcerts -nokeys \
    -out "$SSL_DIR/${CN}.crt"

# private key (unlocked)
sudo openssl pkcs12 -in "$PFX_PATH" \
    -nocerts -nodes \
    -out "$SSL_DIR/${CN}.key"

# CA chain (intermediate + root)
sudo openssl pkcs12 -in "$PFX_PATH" \
    -cacerts -nokeys -chain \
    -out "$SSL_DIR/${CN}-chain.crt" 2>/dev/null || true

# Combine fullchain（Nginx 需要 server cert + intermediate）
if [ -s "$SSL_DIR/${CN}-chain.crt" ]; then
    sudo bash -c "cat $SSL_DIR/${CN}.crt $SSL_DIR/${CN}-chain.crt > $SSL_DIR/${CN}-fullchain.crt"
else
    sudo cp "$SSL_DIR/${CN}.crt" "$SSL_DIR/${CN}-fullchain.crt"
fi

# 權限
sudo chmod 600 "$SSL_DIR"/*.key
sudo chmod 644 "$SSL_DIR"/*.crt
sudo chown root:root "$SSL_DIR"/*

echo ""
echo "✅ 憑證安裝完成"
ls -la "$SSL_DIR"

echo ""
echo "▶ 驗證憑證..."
echo ""
sudo openssl x509 -in "$SSL_DIR/${CN}-fullchain.crt" -noout \
    -subject -issuer -dates

echo ""
echo "▶ Reload Nginx..."
sudo nginx -t && sudo systemctl reload nginx

echo ""
echo "✅ 完成。請用瀏覽器測試："
echo "   https://$SERVER_FQDN/"
echo ""
echo "如果瀏覽器跳警告，確認 AD CA 已透過 GPO 派發到客戶端的："
echo "   控制台 → 憑證管理 → 受信任的根憑證授權單位"
