"""YaYan_LID — 語種識別（VoxLingua107 ECAPA-TDNN 包裝）。"""
from __future__ import annotations

import logging
from typing import Tuple

import numpy as np
import torch

from .config import CONFIG, model_path

logger = logging.getLogger("YaYan.LID")

_LID_MODEL = None

# VoxLingua107 偵測結果（ISO 639-1） → default.yaml 的 routing key
LANG_TO_ROUTING = {
    # ---- 漢語系 ----
    "zh": "zh", "cmn": "zh", "nan": "zh",
    "yue": "yue", "wuu": "wuu", "cdo": "cdo",
    # ---- 東亞 ----
    "ja": "ja", "ko": "ko",
    # ---- 中亞 ----
    "bo": "bo", "ug": "ug",
    "kk": "ug",   # 哈薩克語就近走 Eastern
    "mn": "ug",   # 蒙古語就近走 Eastern
    # ---- 中東 / 南亞 ----
    "fa": "fa", "ur": "ur", "ar": "ar", "hi": "hi",
    "bn": "hi",   # 孟加拉語近似
    # ---- 歐洲 ----
    "en": "en", "fr": "fr", "de": "de", "ru": "ru", "es": "es",
    "it": "fr",   # 沒專屬路由就近走法語（都歐洲）
    "pt": "es",   # 葡萄牙語近似西班牙語
    "nl": "de",   # 荷蘭語近似德語
    "pl": "ru",   # 波蘭語近似俄語
    "uk": "ru",   # 烏克蘭語走俄語
    # ---- 東南亞 ----
    "th": "th", "ms": "ms", "vi": "vi", "id": "id",
    "tl": "ms",   # 菲律賓語就近走馬來語
}


def _load() -> None:
    global _LID_MODEL
    if _LID_MODEL is not None:
        return
    try:
        from speechbrain.inference.classifiers import EncoderClassifier
    except ImportError:
        from speechbrain.pretrained import EncoderClassifier  # type: ignore

    local_dir = model_path("YaYan_LID")
    if not local_dir.exists():
        raise FileNotFoundError(f"YaYan_LID 不存在: {local_dir}")
    logger.info("載入 YaYan_LID …")
    device = CONFIG["devices"]["asr_gpu"]
    _LID_MODEL = EncoderClassifier.from_hparams(
        source=str(local_dir),
        savedir=str(local_dir),
        run_opts={"device": device},
    )


def detect(audio: np.ndarray, sample_rate: int = 16000) -> Tuple[str, float]:
    """回傳 (routing_code, confidence)。routing_code 為 default.yaml 的 asr.routing 鍵。"""
    _load()
    audio_t = torch.from_numpy(audio).float().unsqueeze(0)
    out = _LID_MODEL.classify_batch(audio_t)
    score = float(out[1].exp().max().item())
    label = out[3][0]
    iso639 = label.split(":")[0].strip().lower() if isinstance(label, str) else "auto"
    routing = LANG_TO_ROUTING.get(iso639, "auto")
    logger.info(f"YaYan_LID 偵測: {iso639} → routing={routing} (conf={score:.2f})")
    return routing, score
