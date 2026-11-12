#!/usr/bin/env bash
# 在 Linux 上自簽 Root CA + Server Cert
# Root CA → 透過 AD GPO 派發到所有客戶端
# Server Cert → 給 Nginx 用
#
# 用法：bash deploy/generate_cert.sh

set -euo pipefail

SSL_DIR="/etc/nginx/ssl"

# ── 部署前改這幾項（內部主機資訊，勿提交真值）──
SERVER_FQDN="your-server.example.com"   # 伺服器完整網域名（憑證 CN / SAN）
SERVER_HOSTNAME="your-server"           # 短主機名（SAN，也當作 cert 檔名前綴）
SERVER_IP="192.0.2.10"                  # ← 改成你 server 的實際 IP
ORG_NAME="Your-Org"                     # 組織代號（憑證 O=）
CA_NAME="Your Internal CA"              # Root CA 名稱
# ─────────────────────────────────────────

CN="$SERVER_HOSTNAME"                   # server cert 檔名前綴
CA_FILE="internal-ca"                   # CA 檔名前綴
CA_DAYS=3650                            # CA 10 年
SERVER_DAYS=825                         # Server cert 最多 825 天（瀏覽器規範）

echo "============================================="
echo "  自簽憑證生成器"
echo "  CA: $CA_NAME"
echo "  Server CN: $SERVER_FQDN"
echo "  Server IP: $SERVER_IP"
echo "============================================="

# 確認 IP 對不對
read -p "Server IP 是 $SERVER_IP 嗎？(y/N) " yn
if [[ ! "$yn" =~ ^[Yy]$ ]]; then
    echo "請編輯本腳本，改 SERVER_IP 變數後重跑"
    exit 1
fi

sudo mkdir -p "$SSL_DIR"
sudo chmod 700 "$SSL_DIR"

cd /tmp

# ── 1. 建 Root CA ──
echo ""
echo "▶ Step 1: 建立 Root CA（10 年效期）"

if [ ! -f "$SSL_DIR/${CA_FILE}.crt" ]; then
    sudo openssl genrsa -out "${CA_FILE}.key" 4096

    sudo openssl req -x509 -new -nodes -key "${CA_FILE}.key" \
        -sha256 -days "$CA_DAYS" \
        -out "${CA_FILE}.crt" \
        -subj "/C=TW/O=$ORG_NAME/CN=$CA_NAME"

    sudo mv "${CA_FILE}.key" "$SSL_DIR/"
    sudo mv "${CA_FILE}.crt" "$SSL_DIR/"
    echo "  ✅ Root CA 生成完成"
else
    echo "  ⚠️  Root CA 已存在，跳過生成（避免覆蓋）"
fi

# ── 2. 建 Server Cert ──
echo ""
echo "▶ Step 2: 簽 Server Cert"

# 2a. server private key
sudo openssl genrsa -out "${CN}.key" 2048

# 2b. CSR
sudo openssl req -new -key "${CN}.key" \
    -out "${CN}.csr" \
    -subj "/C=TW/O=$ORG_NAME/CN=$SERVER_FQDN"

# 2c. SAN extension（瀏覽器要 SAN 才認）
sudo tee "${CN}.ext" > /dev/null <<EOF
authorityKeyIdentifier=keyid,issuer
basicConstraints=CA:FALSE
keyUsage = digitalSignature, nonRepudiation, keyEncipherment, dataEncipherment
extendedKeyUsage = serverAuth
subjectAltName = @alt_names

[alt_names]
DNS.1 = $SERVER_FQDN
DNS.2 = $SERVER_HOSTNAME
IP.1 = $SERVER_IP
EOF

# 2d. 用 CA 簽
sudo openssl x509 -req -in "${CN}.csr" \
    -CA "$SSL_DIR/${CA_FILE}.crt" \
    -CAkey "$SSL_DIR/${CA_FILE}.key" \
    -CAcreateserial \
    -out "${CN}.crt" \
    -days "$SERVER_DAYS" -sha256 \
    -extfile "${CN}.ext"

# 2e. 組 fullchain (server + CA)
sudo bash -c "cat ${CN}.crt $SSL_DIR/${CA_FILE}.crt > ${CN}-fullchain.crt"

# 2f. 搬到 SSL_DIR
sudo mv "${CN}.key" "$SSL_DIR/"
sudo mv "${CN}.crt" "$SSL_DIR/"
sudo mv "${CN}-fullchain.crt" "$SSL_DIR/"
sudo rm -f "${CN}.csr" "${CN}.ext"

# 2g. 權限
sudo chmod 600 "$SSL_DIR"/*.key
sudo chmod 644 "$SSL_DIR"/*.crt
sudo chown root:root "$SSL_DIR"/*

echo "  ✅ Server Cert 簽好了"

# ── 3. 匯出 CA 給 AD GPO ──
echo ""
echo "▶ Step 3: 匯出 CA 給 AD GPO 派發"

CA_EXPORT_DIR="/tmp/${CA_FILE}-export"
sudo mkdir -p "$CA_EXPORT_DIR"

# .crt（給 GPO 派發用，二進位 DER 或 PEM 都行，Windows 兩個都吃）
sudo cp "$SSL_DIR/${CA_FILE}.crt" "$CA_EXPORT_DIR/${CA_FILE}.crt"

# 轉成 DER 格式（部分 Windows 工具偏好）
sudo openssl x509 -in "$SSL_DIR/${CA_FILE}.crt" -outform DER \
    -out "$CA_EXPORT_DIR/${CA_FILE}.der"

# 轉 P7B（Windows 慣用格式）
sudo openssl crl2pkcs7 -nocrl -certfile "$SSL_DIR/${CA_FILE}.crt" \
    -out "$CA_EXPORT_DIR/${CA_FILE}.p7b" -outform DER

sudo chown -R "$(id -un):$(id -gn)" "$CA_EXPORT_DIR"

# ── 4. 驗證 ──
echo ""
echo "▶ Step 4: 驗證憑證"

sudo openssl x509 -in "$SSL_DIR/${CN}-fullchain.crt" -noout \
    -subject -issuer -dates

echo ""
sudo openssl verify -CAfile "$SSL_DIR/${CA_FILE}.crt" \
    "$SSL_DIR/${CN}.crt"

# ── 5. 完成提示 ──
echo ""
echo "============================================="
echo "✅ 憑證生成完成"
echo "============================================="
echo ""
echo "Server 端檔案（給 Nginx 用）："
echo "  $SSL_DIR/${CN}-fullchain.crt"
echo "  $SSL_DIR/${CN}.key"
echo ""
echo "CA 根憑證（給 AD GPO 派發）："
ls -la "$CA_EXPORT_DIR"
echo ""
echo "============================================="
echo "  AD GPO 派發步驟（在 AD 主機操作）"
echo "============================================="
echo ""
echo "1. SCP 取 CA 根憑證到 AD server："
echo "     scp $CA_EXPORT_DIR/${CA_FILE}.crt admin@<AD-server>:C:/temp/"
echo ""
echo "2. 在 AD 主機開 Group Policy Management Console (gpmc.msc)"
echo ""
echo "3. 編輯網域的 Group Policy Object：右鍵 → Edit"
echo ""
echo "4. 路徑："
echo "     Computer Configuration"
echo "     → Policies"
echo "     → Windows Settings"
echo "     → Security Settings"
echo "     → Public Key Policies"
echo "     → Trusted Root Certification Authorities"
echo ""
echo "5. 右鍵 → Import → 選 C:/temp/${CA_FILE}.crt"
echo ""
echo "6. 客戶端套用 GPO（自動或手動）："
echo "     gpupdate /force"
echo ""
echo "7. 客戶端驗證："
echo "     certmgr.msc → 信任的根憑證授權單位 → 找 '$CA_NAME'"
echo ""
echo "============================================="
echo "  接著請 reload Nginx："
echo "     sudo nginx -t && sudo systemctl reload nginx"
echo "============================================="
