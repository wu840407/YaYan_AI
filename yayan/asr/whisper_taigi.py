"""YaYan_ASR_Taigi — v4.7-C：台語(閩南語)專用 Whisper-medium 微調。

設計重點：
- 沿用 whisper_global 的 HF pipeline 載入樣式（fp16、cuda:0、local_files_only）。
- **離線環境**：權重需手動放入 /data/ai_models/YaYan_ASR_Taigi/，否則 _load()
  會丟 FileNotFoundError；router 會捕捉並自動退回 Dolphin（不中斷該段）。
- 預設停用：由 config asr.enable_taigi_asr 控制，router 才會路由到這裡。
- 為什麼是獨立模型：使用者回報濃厚台語鄉音時 Dolphin nan 抓字不穩，
  這顆是針對台灣腔調微調的 Whisper-medium，作為 nan 路由的替代引擎。
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import torch

from ..config import CONFIG, model_path

logger = logging.getLogger("YaYan.ASR.Taigi")

_PIPE = None


def _load():
    global _PIPE
    if _PIPE is not None:
        return _PIPE
    from transformers import pipeline

    local_dir = model_path("YaYan_ASR_Taigi")
    if not local_dir.exists():
        # 離線環境權重未放入：明確報錯，交由 router 退回 Dolphin
        raise FileNotFoundError(
            f"YaYan_ASR_Taigi 不存在: {local_dir}（台語專用模型需離線手動放入）"
        )
    logger.info("載入 YaYan_ASR_Taigi …")
    device = CONFIG["devices"]["asr_gpu"]
    _PIPE = pipeline(
        "automatic-speech-recognition",
        model=str(local_dir),
        torch_dtype=torch.float16,
        device=device,
        model_kwargs={"local_files_only": True},
    )
    return _PIPE


def transcribe(audio: np.ndarray, language_hint: Optional[str] = None) -> str:
    """台語 ASR；language_hint 保留介面一致性，實際固定走台語/中文解碼。"""
    pipe = _load()
    sample_rate = CONFIG["audio"]["sample_rate"]
    # 這顆是 zh-tw 微調模型，固定中文轉錄任務即可
    gen_kwargs = {"task": "transcribe", "language": "chinese"}

    try:
        out = pipe(
            {"raw": audio.astype(np.float32), "sampling_rate": sample_rate},
            generate_kwargs=gen_kwargs,
            return_timestamps=True,
        )
    except Exception as e:
        logger.warning(f"YaYan_ASR_Taigi 識別失敗，回退無語言提示: {e}")
        out = pipe(
            {"raw": audio.astype(np.float32), "sampling_rate": sample_rate},
            generate_kwargs={"task": "transcribe"},
            return_timestamps=True,
        )
    return out["text"].strip() if isinstance(out, dict) else str(out)
