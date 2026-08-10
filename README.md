# 🏺 YaYan-AI (雅言) - Chinese Dialect Speech-to-Text System

![Python](https://img.shields.io/badge/Python-3.10-blue)
![Architecture](https://img.shields.io/badge/Architecture-Dual%20RTX%204000-purple)
![License](https://img.shields.io/badge/License-MIT-orange)
![Version](https://img.shields.io/badge/Version-V4.1-green)

> **Full offline, privacy-first Chinese dialect speech transcription with speaker diarization**
> **全本地化、隱私優先的中文方言語音轉寫系統，支援說話者分離與對話格式輸出**

---

## 📖 Introduction (專案簡介)

**[English]**
**YaYan-AI** is a fully offline AI system that converts Chinese dialectal speech into structured Traditional Chinese transcripts with speaker diarization. It identifies who said what, when, and outputs clean dialogue like:

```
[00:00]A方：你好
[00:03]B方：你好啊，最近怎樣？
```

V4 introduces a complete pipeline redesign: **Pyannote 3.1** for speaker separation, **FunASR SenseVoice** for dialect ASR, **Whisper Large V3** for international languages, and **Qwen2.5-7B** for two-stage LLM correction and analysis — all running locally on dual RTX 4000 Ada GPUs.

**[中文]**
**雅言 (YaYan-AI)** 是一套完全本地化的 AI 系統，將多種中文方言語音轉換為帶說話者標記的正體中文對話逐字稿。

V4 完整重構核心管線：使用 **Pyannote 3.1** 進行說話者分離、**FunASR SenseVoice** 處理漢語方言、**Whisper Large V3** 處理國際語言、**Qwen2.5-7B** 執行兩階段文字校正與內容分析，全部在雙 RTX 4000 Ada 伺服器上離線運行。

---

## 🌟 Version History (版本演進)

| 版本 | 硬體 | 核心改進 |
| :--- | :--- | :--- |
| **V1** | RTX 3090 × 1 | 原型，Whisper + Llama，無說話者分離 |
| **V2~V3** | RTX 4000 × 2 | 雙卡分流，多方言支援，信心度儀表板 |
| **V4.1** | RTX 4000 × 2 | **Pyannote 說話者分離、FunASR 方言 ASR、Qwen2.5-7B、LLM 兩階段校正、分批處理長音訊** |
| **計畫中** | H200 × 2 | Qwen2.5-72B 全精度、LoRA 方言微調 |

---

## 🚀 Key Features V4.1 (核心功能)

### 1. 👥 Speaker Diarization (說話者分離)
自動偵測音訊中有幾個說話者，輸出帶時間戳記的對話格式。
支援手動指定說話者人數（2~5人）以提升準確率。
```
[00:00]說話者1：你好
[00:03]說話者2：你好啊
```

### 2. 🎙️ Dual ASR Engine (雙引擎 ASR)
- **FunASR SenseVoice（cuda:0）**：處理漢語方言（台語、粵語、四川話、吳語、山東話、普通話）
- **Whisper Large V3（cuda:1）**：處理國際語言（英、日、韓、俄、藏、維吾爾語）
- 支援自動混合語言偵測

### 3. 🧠 Two-Stage LLM Correction (兩階段 LLM 校正)
- **第一階段**：格式保護校正 — 時間戳記與說話者標記完全不動，只校正文字內容，簡體轉繁體
- **第二階段**：內容分析 — 重點摘要、說話意圖分析、全文翻譯、情境研判報告
- 分批處理長音訊，避免 VRAM OOM

### 4. 📊 Confidence Dashboard (信心度儀表板)
即時顯示 ASR 信心度（%）、說話者數量、語音段數、引擎狀態。

### 5. 💬 AI Correction Chat (AI 對話修正框)
轉寫完成後可直接對 AI 下指令修正輸出，例如：
- 「把說話者1改成張先生」
- 「這段對話的主題是什麼？」

---

## 🏗️ Architecture (系統架構)

```
音訊輸入
    │
    ▼
[音訊預處理] librosa — 重採樣 16kHz、正規化音量
    │
    ▼
[說話者分離] Pyannote 3.1 on cuda:0
    │  → [(0.0, 3.2, SPEAKER_00), (3.5, 7.1, SPEAKER_01), ...]
    ▼
[逐段 ASR]
  方言 → FunASR SenseVoice on cuda:0
  外語 → Whisper Large V3 on cuda:1
    │  → [00:00]說話者1：... \n [00:03]說話者2：...
    ▼
[LLM 第一階段] Qwen2.5-7B on cuda:1 — 分批文字校正 + 簡繁轉換
    │
    ▼
[LLM 第二階段] Qwen2.5-7B on cuda:1 — 內容分析
    │
    ▼
Gradio WebUI 輸出
```

---

## 📦 Models Required (需要下載的模型)

| 模型 | 用途 | 大小 | 來源 |
| :--- | :--- | :--- | :--- |
| `pyannote/speaker-diarization-3.1` | 說話者分離 | ~300MB | HuggingFace（需申請授權）|
| `pyannote/segmentation-3.0` | 分離底層模型 | ~26MB | HuggingFace（需申請授權）|
| `pyannote/wespeaker-voxceleb-resnet34-LM` | Embedding | ~400MB | HuggingFace（需申請授權）|
| `iic/SenseVoiceSmall` | 方言 ASR | ~300MB | ModelScope（自動快取）|
| `openai/whisper-large-v3` | 通用 ASR | ~3GB | HuggingFace |
| `Qwen/Qwen2.5-7B-Instruct` | LLM 校正+分析 | ~15GB | HuggingFace |

> Pyannote 系列模型需先至 HuggingFace 申請使用授權（免費，審核約數分鐘）。

```bash
# 登入 HuggingFace
huggingface-cli login

# 下載模型
hf download pyannote/speaker-diarization-3.1 \
  --local-dir /data/ai_models/pyannote-speaker-diarization-3.1
hf download pyannote/segmentation-3.0 \
  --local-dir /data/ai_models/pyannote-segmentation-3.0
hf download pyannote/wespeaker-voxceleb-resnet34-LM \
  --local-dir /data/ai_models/wespeaker-voxceleb-resnet34-LM
hf download openai/whisper-large-v3 \
  --local-dir /data/ai_models/whisper-large-v3
hf download Qwen/Qwen2.5-7B-Instruct \
  --local-dir /data/ai_models/Qwen2.5-7B-Instruct
```

---

## 🛠️ Installation (安裝步驟)

### 1. 系統需求
- OS：Ubuntu 22.04 / 24.04
- GPU：NVIDIA RTX 4000 Ada × 2（各 20GB VRAM）或同等級雙卡
- NVIDIA Driver 535+，CUDA 12.1+
- Python 3.10

### 2. 建立環境

```bash
git clone https://github.com/wu840407/YaYan-AI.git
cd YaYan-AI

conda create -n yayan_ai python=3.10 -y
conda activate yayan_ai
```

### 3. 安裝套件

```bash
# PyTorch（CUDA 12.1）
pip install torch==2.5.1+cu121 torchaudio==2.5.1+cu121 torchvision==0.20.1+cu121 \
  --index-url https://download.pytorch.org/whl/cu121

# 其他依賴
pip install transformers>=4.45.0 bitsandbytes>=0.43.0 accelerate>=0.26.0 \
  funasr==1.3.1 "pyannote.audio==3.3.2" onnxruntime-gpu \
  librosa soundfile gradio>=4.0.0 numpy scipy
```

### 4. 設定模型路徑

下載完模型後，確認 `app_rtx4000_V4.py` 裡的 `MODEL_PATHS` 對應正確路徑：

```python
MODEL_PATHS = {
    "pyannote_diarization": "/data/ai_models/pyannote-speaker-diarization-3.1",
    "pyannote_segmentation": "/data/ai_models/pyannote-segmentation-3.0",
    "sensevoice":            "/data/ai_models/iic/SenseVoiceSmall",
    "whisper":               "/data/ai_models/whisper-large-v3",
    "whisper_tw":            "",   # 台語微調版（選用，留空則使用通用版）
    "qwen":                  "/data/ai_models/Qwen2.5-7B-Instruct",
}
```

同時修改 `pyannote-speaker-diarization-3.1/config.yaml`，將 embedding 路徑指向本地：

```yaml
embedding: /data/ai_models/wespeaker-voxceleb-resnet34-LM
segmentation: /data/ai_models/pyannote-segmentation-3.0/pytorch_model.bin
```

---

## ▶️ Usage (使用方法)

```bash
conda activate yayan_ai
cd /data/AI_Project
python app_rtx4000_V4.py
```

開啟瀏覽器訪問 `http://localhost:7860`

1. 上傳音訊檔案或直接錄音
2. 選擇來源語言
3. 指定說話者人數（或自動偵測）
4. 選擇分析模式
5. 點擊「開始轉寫分析」

---

## 🔧 VRAM Usage (顯存配置)

| GPU | 載入內容 | 佔用估算 |
| :--- | :--- | :--- |
| cuda:0 | Pyannote 3.1 + FunASR SenseVoice | ~6~7 GB |
| cuda:1 | Whisper Large V3 (fp16) + Qwen2.5-7B (4-bit) | ~8~9 GB |

> 兩者交替使用，不會同時佔滿。RTX 4000 Ada 20GB 可正常運行。

---

## 📝 License

This project is open-source and available under the MIT License.
