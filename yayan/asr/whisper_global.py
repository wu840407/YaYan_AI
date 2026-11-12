"""YaYan_ASR_Global — 通用多語 ASR（波斯語、烏爾都語、英語等）。"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import torch

from ..config import CONFIG, model_path

logger = logging.getLogger("YaYan.ASR.Global")

_PIPE = None


def _load():
    global _PIPE
    if _PIPE is not None:
        return _PIPE
    from transformers import pipeline

    local_dir = model_path("YaYan_ASR_Global")
    if not local_dir.exists():
        raise FileNotFoundError(f"YaYan_ASR_Global 不存在: {local_dir}")
    logger.info("載入 YaYan_ASR_Global …")
    device = CONFIG["devices"]["asr_gpu"]
    _PIPE = pipeline(
        "automatic-speech-recognition",
        model=str(local_dir),
        torch_dtype=torch.float16,
        device=device,
        model_kwargs={"local_files_only": True},
    )
    return _PIPE


# routing key → Whisper 語言名稱。不給提示時 Whisper 會自己判，但實測誤判代價很大：
# 粵語不指定 language=cantonese 會被轉寫成國語（「我哋嘅」變「我们的」），
# 指定後才輸出真正的粵文。名稱必須是 Whisper 認得的英文全名，不是 ISO 代碼。
_LANG_HINT = {
    "fa": "persian",
    "ur": "urdu",
    "en": "english",
    "ar": "arabic",
    # 粵語：Whisper large-v3 才有 yue，是目前唯一能產出粵文的路徑
    "yue": "cantonese",
    "yue-hk": "cantonese",
    "yue-gz": "cantonese",
    # 藏語：Dolphin 把 bo 映射成 zh-CN（當中文解碼），Whisper 有原生藏語
    "bo": "tibetan",
    # 其餘 config routing 表裡既有的語種，補上提示減少自動偵測誤判
    "ja": "japanese",
    "ko": "korean",
    "hi": "hindi",
    "fr": "french",
    "de": "german",
    "ru": "russian",
    "es": "spanish",
    "th": "thai",
    "ms": "malay",
    "vi": "vietnamese",
    "id": "indonesian",
    # ⚠️ 緬甸語(my)、寮語(lo)、高棉語(km)刻意不列也不開放選擇：模型名義上支援，
    #    但用公開語料實測，指定緬甸語時輸出的文字系統五花八門、寮語輸出的是泰文、
    #    高棉語九成片段陷入重複迴圈，產不出可用的逐字稿。
    # ⚠️ ug（維吾爾語）刻意不列：Whisper 100 種語言裡沒有維語，
    #    指定會直接拋 Unsupported language。維語目前無可用引擎，見 PROJECT-HISTORY。
}


def transcribe(audio: np.ndarray, language_hint: Optional[str] = None) -> str:
    pipe = _load()
    sample_rate = CONFIG["audio"]["sample_rate"]
    gen_kwargs = {"task": "transcribe"}
    lang = _LANG_HINT.get((language_hint or "").lower())
    if lang:
        gen_kwargs["language"] = lang

    try:
        out = pipe(
            {"raw": audio.astype(np.float32), "sampling_rate": sample_rate},
            generate_kwargs=gen_kwargs,
            return_timestamps=True,
        )
    except Exception as e:
        logger.warning(f"YaYan_ASR_Global 識別失敗，回退無語言提示: {e}")
        out = pipe(
            {"raw": audio.astype(np.float32), "sampling_rate": sample_rate},
            generate_kwargs={"task": "transcribe"},
            return_timestamps=True,
        )
    return out["text"].strip() if isinstance(out, dict) else str(out)
