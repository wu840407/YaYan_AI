"""YaYan_LID_Dialect — 漢語方言語種識別（FireRedLID 包裝）。

與 `yayan/lid.py`（VoxLingua107）介面相同（`detect(audio, sample_rate) -> (routing, conf)`），
可由 config 的 `asr.lid_backend` 切換。差別在於 VoxLingua107 對整個漢語只有一個
"Chinese" 標籤，本模組能分出閩/粵/吳/湘/西南官話，台語與粵語因此才走得到專用 ASR。

⚠️ 標籤集限制（模型本身沒有這些標籤，不是判斷失誤）：
維吾爾語、贛語、客家話**不在** FireRedLID 的標籤表內，永遠不會被判出來。
"""
from __future__ import annotations

import logging
import os
import sys
from typing import Optional, Tuple

import numpy as np

from .config import CONFIG

logger = logging.getLogger("YaYan.LID.Dialect")

_MODEL = None

# FireRedLID 輸出（漢語為 "<lang> <dialect>" 兩層）→ default.yaml 的 routing key
LABEL_TO_ROUTING = {
    # ---- 漢語方言（本模組存在的理由）----
    "zh min": "nan",        # 閩語 → 閩南/台語 ASR（Breeze-ASR-Taigi）
    "zh yue": "yue",        # 粵語 → Whisper（必須指定 cantonese）
    "zh wu": "wuu",         # 吳語
    "zh xiang": "hsn",      # 湘語
    "zh xinan": "cmn-sw",   # 西南官話（四川話為代表）
    "zh mandarin": "zh",
    "zh north": "zh",       # 北方官話
    "zh": "zh",
    # ---- 其他語言（沿用 lid.py 的就近映射）----
    "bo": "bo", "ja": "ja", "ko": "ko",
    "en": "en", "fr": "fr", "de": "de", "ru": "ru", "es": "es",
    "it": "fr", "pt": "es", "nl": "de", "pl": "ru", "uk": "ru",
    "th": "th", "ms": "ms", "vi": "vi", "id": "id", "tl": "ms",
    "fa": "fa", "ur": "ur", "ar": "ar", "hi": "hi", "bn": "hi",
    "kk": "ug", "mn": "ug", "uz": "ug", "tr": "ug", "tt": "ug",
}

_DEFAULT_MODEL_DIR = "/data/ai_models/YaYan_LID_Dialect"
_DEFAULT_REPO_DIR = "/data/AI_Project/third_party/FireRedASR2S"


def _load() -> None:
    """延後載入：只有真的切到本 backend 才吃記憶體。"""
    global _MODEL
    if _MODEL is not None:
        return

    asr_cfg = CONFIG.get("asr", {})
    repo_dir = asr_cfg.get("firered_repo_dir", _DEFAULT_REPO_DIR)
    model_dir = asr_cfg.get("firered_lid_dir", _DEFAULT_MODEL_DIR)

    if not os.path.isdir(model_dir):
        raise FileNotFoundError(f"FireRedLID 模型目錄不存在: {model_dir}")
    if repo_dir not in sys.path:
        sys.path.insert(0, repo_dir)

    from fireredasr2s.fireredlid import FireRedLid, FireRedLidConfig

    use_half = bool(asr_cfg.get("firered_lid_half", False))
    _MODEL = FireRedLid.from_pretrained(
        model_dir, FireRedLidConfig(use_gpu=True, use_half=use_half)
    )
    logger.info(f"FireRedLID 載入完成（half={use_half}）")


def detect(audio: np.ndarray, sample_rate: int = 16000) -> Tuple[str, float]:
    """回傳 (routing_key, confidence)。判不出來時回 ("auto", 0.0)。"""
    _load()

    # FeatExtractor 走的是 kaldiio.load_mat 的資料型態（int16 尺度），
    # 但 pipeline 傳進來的是 float32 [-1, 1]，要先換算。
    if audio.dtype != np.int16:
        audio = np.clip(audio, -1.0, 1.0)
        audio = (audio * 32767.0).astype(np.int16)

    results = _MODEL.process(["seg"], [(sample_rate, audio)])
    if not results:
        return "auto", 0.0

    label = (results[0].get("lang") or "").strip()
    conf = float(results[0].get("confidence") or 0.0)
    if not label:
        return "auto", 0.0

    routing = _label_to_routing(label)
    if routing is None:
        logger.debug(f"FireRedLID 標籤 {label!r} 無對應 routing，退回 auto")
        return "auto", conf
    return routing, conf


def _label_to_routing(label: str) -> Optional[str]:
    if label in LABEL_TO_ROUTING:
        return LABEL_TO_ROUTING[label]
    # 兩層標籤但方言部分不認識時（例如未來新增的方言），退回語言層
    head = label.split()[0]
    return LABEL_TO_ROUTING.get(head)
