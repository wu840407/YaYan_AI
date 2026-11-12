# YaYan-AI v4.5 — Third-Party Component Notices

本文件列出 YaYan-AI 系統內使用的第三方模型與函式庫之原始作者、授權，以履行 Apache 2.0 / MIT 等授權條款的署名要求。系統 UI 與面向使用者文件中以 `YaYan_*` 別名稱呼這些組件，但其原始來源、授權合規責任於本文件中完整保留。

---

## Models

| 內部別名 | 原始模型 | 作者 | 授權 |
|---|---|---|---|
| YaYan_Reasoner / YaYan_Translator（openai backend，主要部署） | Qwen3.8-27B（GGUF Q4_K_M，Unsloth 量化） | Qwen Team, Alibaba Cloud | Apache-2.0 |
| YaYan_Reasoner / YaYan_Translator（transformers backend，內建備援） | Qwen3-14B | Qwen Team, Alibaba Cloud | Apache-2.0 |
| YaYan_ASR_Mandarin | SenseVoiceSmall | FunAudioLLM (Tongyi Lab) | Apache-2.0 |
| YaYan_ASR_Eastern | Dolphin-base | Dataocean AI × Tsinghua University | Apache-2.0 |
| YaYan_ASR_Global | Whisper-large-v3 | OpenAI | MIT |
| YaYan_VAD | Silero-VAD | Snakers4 | MIT |
| YaYan_LID | VoxLingua107 ECAPA-TDNN | SpeechBrain | Apache-2.0 |
| YaYan_Diarize (optional) | speaker-diarization-3.1 | pyannote.audio | MIT |
| YaYan_TWConv | OpenCC dictionaries | BYVoid et al. | Apache-2.0 |

各模型授權全文請見對應上游 repository。本系統不修改原始權重；僅以本地別名管理部署目錄。

## Libraries

依賴之 Python 套件詳見 `requirements.txt`。重要組件：

- PyTorch (BSD-3-Clause) — Meta AI
- Transformers / Accelerate / BitsAndBytes — Hugging Face (Apache-2.0)
- Gradio (Apache-2.0) — Hugging Face
- vLLM (Apache-2.0, optional)
- librosa (ISC) / soundfile (BSD)
- watchdog (Apache-2.0)

---

YaYan-AI 系統本體之程式碼授權見 `LICENSE`。
