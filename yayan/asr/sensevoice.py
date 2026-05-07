"""YaYan_ASR_Mandarin — 漢語方言 ASR 包裝（SenseVoiceSmall）。

兩段切片策略：
1. yayan/vad.py 負責主切片（VAD-based）
2. 本模組額外保險：若單段仍 > MAX_CHUNK_SEC，做 fixed-window 二次切片
"""
from __future__ import annotations

import logging
import re
from typing import List, Optional

import numpy as np

from ..config import CONFIG, model_path

logger = logging.getLogger("YaYan.ASR.Mandarin")

_MODEL = None
MAX_CHUNK_SEC = 30  # SenseVoice 在 24GB Turing 上的安全上限


def _load():
    global _MODEL
    if _MODEL is not None:
        return _MODEL
    from funasr import AutoModel

    local_dir = model_path("YaYan_ASR_Mandarin")
    if not local_dir.exists():
        raise FileNotFoundError(f"YaYan_ASR_Mandarin 不存在: {local_dir}")

    logger.info("載入 YaYan_ASR_Mandarin …")
    device = CONFIG["devices"]["asr_gpu"]
    _MODEL = AutoModel(
        model=str(local_dir),
        trust_remote_code=True,
        disable_update=True,
        device=device,
    )
    return _MODEL


def _run_one(audio: np.ndarray, sv_lang: str) -> str:
    """執行一次 SenseVoice 推論並抽出文字。"""
    model = _load()
    res = model.generate(
        input=audio,
        cache={},
        language=sv_lang,
        use_itn=True,
        batch_size_s=60,
    )
    if not res:
        return ""
    text = res[0].get("text", "") if isinstance(res[0], dict) else str(res[0])
    return _strip_tags(text)


def transcribe(audio: np.ndarray, language_hint: Optional[str] = None) -> str:
    """language_hint: zh/yue/wuu/cmn/auto

    若 audio 過長，自動做 fixed-window 二次切片避免 OOM。
    """
    sample_rate = CONFIG["audio"]["sample_rate"]

    sv_lang = {
        "zh": "zh", "cmn": "zh", "yue": "yue", "wuu": "zh", "cdo": "zh"
    }.get((language_hint or "auto").lower(), "auto")

    duration = len(audio) / sample_rate

    # 短段直接送
    if duration <= MAX_CHUNK_SEC:
        return _run_one(audio, sv_lang)

    # 長段二次切片
    logger.warning(
        f"chunk {duration:.1f}s 超過 {MAX_CHUNK_SEC}s，二次切片送 SenseVoice"
    )
    step = MAX_CHUNK_SEC * sample_rate
    pieces: List[str] = []
    for i in range(0, len(audio), step):
        sub = audio[i:i + step]
        if len(sub) / sample_rate < 0.5:
            continue  # 太短的尾巴跳過
        try:
            text = _run_one(sub, sv_lang)
            if text:
                pieces.append(text)
        except Exception as e:
            logger.error(f"二次切片 [{i/sample_rate:.1f}s] 失敗: {e}")
            continue
    return " ".join(pieces)


def _strip_tags(text: str) -> str:
    """移除 SenseVoice 的 <|...|> 控制標籤。"""
    return re.sub(r"<\|[^|]+\|>", "", text).strip()
