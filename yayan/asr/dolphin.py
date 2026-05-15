"""YaYan_ASR_Dialect — Dolphin-CN-Dialect 包裝。

對應 Dolphin SDK 真實接受的 lang_sym + region_sym（從 languages.md 修正）。
注意：Dolphin 用自己的命名（zh-SICHUAN、ct-HK 等），不是 ISO 639。
"""
from __future__ import annotations

import logging
import os
import re
import tempfile
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import soundfile as sf
import torch

from ..config import CONFIG, model_path

logger = logging.getLogger("YaYan.ASR.Dolphin")

_MODEL = None

# ★ 修正：使用 Dolphin 官方 languages.md 的 region 命名
# (lang_sym, region_sym) - region 可為 None（讓模型自動判）
ROUTING_TO_DOLPHIN: dict = {
    # ── 漢語方言（lang_sym=zh，region 是省份）──
    "zh":     ("zh", "CN"),
    "cmn":    ("zh", "CN"),
    "cmn-tw": ("zh", "TW"),
    # 北方官話
    "cmn-ne": ("zh", "LIAONING"),    # 東北話（用遼寧代表）
    "cmn-sd": ("zh", "SHANDONG"),    # 山東話
    "cmn-zy": ("zh", "HENAN"),       # 河南話
    "cmn-xa": ("zh", "SHAANXI"),     # 西安話（陝西）
    "cmn-lz": ("zh", "GANSU"),       # 蘭州話（甘肅）
    "cmn-tj": ("zh", "TIANJIN"),     # 天津話
    "cmn-hb": ("zh", "HEBEI"),       # 河北
    "cmn-nx": ("zh", "NINGXIA"),     # 寧夏
    "cmn-sx": ("zh", "SHANXI"),      # 山西
    "cmn-ah": ("zh", "ANHUI"),       # 安徽
    # 西南官話
    "cmn-sw": ("zh", "SICHUAN"),     # 四川話
    "cmn-yn": ("zh", "YUNNAN"),      # 雲南話
    "cmn-hb2":("zh", "HUBEI"),       # 湖北話（武漢）
    "cmn-wh": ("zh", "HUBEI"),       # 武漢話 → 湖北
    # 江淮官話（南京話沒專屬 → 走 zh-CN）
    "cmn-jh": ("zh", "CN"),
    # 吳語
    "wuu":    ("zh", "WU"),          # 吳語通用
    "wuu-sz": ("zh", "WU"),          # 蘇州話 → 吳語
    "wuu-nb": ("zh", "WU"),          # 寧波話 → 吳語
    "wuu-wz": ("zh", "WENZHOU"),     # 溫州話有專屬
    "wuu-sh": ("zh", "SHANGHAI"),    # 上海話
    # 粵語（lang_sym=ct，不是 yue）
    "yue":    ("ct", "NULL"),        # 粵語通用
    "yue-hk": ("ct", "HK"),          # 香港粵語
    "yue-gz": ("ct", "GZ"),          # 廣州粵語
    # 閩語
    "nan":    ("zh", "MINNAN"),      # 閩南語/台語
    "nan-tw": ("zh", "MINNAN"),
    "nan-cs": ("zh", "MINNAN"),      # 潮汕話 → 閩南
    "nan-hn": ("zh", "MINNAN"),      # 海南話 → 閩南
    "cdo":    ("zh", "FUJIAN"),      # 福州話（閩東）→ 福建
    "min":    ("zh", "MINNAN"),
    "hokkien":("zh", "MINNAN"),
    # 客家、湘、贛、晉（沒專屬 region → 走 zh-CN 讓模型自動判）
    "hak":    ("zh", "GUANGDONG"),   # 客家就近用廣東（部分客家在廣東）
    "hsn":    ("zh", "HUNAN"),       # 湘語 → 湖南
    "gan":    ("zh", "CN"),          # 贛語沒專屬
    "cjy":    ("zh", "SHANXI"),      # 晉語 → 山西
    # 中亞
    "bo":     ("zh", "CN"),          # 藏語在 zh family
    "ug":     ("ug", "CN"),          # 維吾爾語
}


@dataclass
class WordTimestamp:
    word: str
    start: float
    end: float


@dataclass
class DolphinResult:
    text: str
    words: List[WordTimestamp]
    detected_lang: str = ""
    detected_region: str = ""


def _load():
    global _MODEL
    if _MODEL is not None:
        return _MODEL
    try:
        import dolphin
    except ImportError as e:
        raise ImportError(
            "dolphin SDK 未安裝。請：pip install dataoceanai-dolphin"
        ) from e

    local_dir = model_path("YaYan_ASR_Dialect")
    if not local_dir.exists():
        raise FileNotFoundError(f"YaYan_ASR_Dialect 不存在: {local_dir}")

    device = CONFIG["devices"]["asr_gpu"]

    pt_files = list(local_dir.glob("*.pt"))
    if not pt_files:
        raise FileNotFoundError(f"{local_dir} 找不到 .pt 模型權重")
    pt_files.sort(key=lambda p: p.stat().st_size, reverse=True)
    model_name = pt_files[0].stem
    logger.info(
        f"載入 YaYan_ASR_Dialect (model_name={model_name}, file={pt_files[0].name}) on {device}"
    )

    _MODEL = dolphin.load_model(model_name, str(local_dir), device)
    return _MODEL


def transcribe(
    audio: np.ndarray,
    language_hint: Optional[str] = None,
    enable_word_timestamp: bool = True,
    sample_rate: int = 16000,
) -> DolphinResult:
    import dolphin

    model = _load()
    tmp_path = None
    try:
        if isinstance(audio, np.ndarray):
            if audio.ndim > 1:
                audio = audio.mean(axis=0)
            audio = audio.astype(np.float32)
            tmp_fd, tmp_path = tempfile.mkstemp(suffix=".wav", prefix="yayan_dolphin_")
            os.close(tmp_fd)
            sf.write(tmp_path, audio, sample_rate, subtype="PCM_16")
            audio_input = tmp_path
        else:
            audio_input = audio

        # 取 lang_sym + region_sym（unknown routing → 不傳，讓 Dolphin 自動判）
        mapping = ROUTING_TO_DOLPHIN.get((language_hint or "auto").lower())
        kwargs = {}
        if mapping:
            lang_sym, region_sym = mapping
            kwargs["lang_sym"] = lang_sym
            if region_sym and region_sym not in ("NULL", "AUTO"):
                kwargs["region_sym"] = region_sym
        # else: 不傳 lang_sym/region_sym，Dolphin 走全自動

        if enable_word_timestamp:
            kwargs["predict_time"] = True

        # 退化策略：region 不認 → 去掉 region 重試
        for retry in range(3):
            try:
                result = dolphin.transcribe(model, audio_input, **kwargs)
                break
            except TypeError:
                kwargs.pop("predict_time", None)
            except Exception as e:
                emsg = str(e)
                if "Unsupported language or region" in emsg and "region_sym" in kwargs:
                    logger.warning(
                        f"Dolphin 不支援 region={kwargs['region_sym']}，去掉 region 重試"
                    )
                    kwargs.pop("region_sym", None)
                    continue
                elif "Unsupported language or region" in emsg and "lang_sym" in kwargs:
                    logger.warning(
                        f"Dolphin 不支援 lang={kwargs['lang_sym']}，全自動模式重試"
                    )
                    kwargs.pop("lang_sym", None)
                    continue
                else:
                    raise
        else:
            logger.error(f"Dolphin 連續失敗")
            return DolphinResult(text="", words=[])

    except Exception as e:
        logger.error(f"Dolphin 推論失敗 (lang={language_hint}): {e}")
        return DolphinResult(text="", words=[])
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except Exception:
                pass

    text = (getattr(result, "text", "") or "").strip()
    text = _strip_special_tokens(text)

    words: List[WordTimestamp] = []
    raw_words = (
        getattr(result, "words", None)
        or getattr(result, "word_timestamps", None)
        or getattr(result, "tokens", None)
    )
    if raw_words:
        for w in raw_words:
            try:
                if isinstance(w, dict):
                    words.append(WordTimestamp(
                        word=_strip_special_tokens(str(w.get("word", w.get("text", "")))),
                        start=float(w.get("start", w.get("start_time", 0))),
                        end=float(w.get("end", w.get("end_time", 0))),
                    ))
                else:
                    words.append(WordTimestamp(
                        word=_strip_special_tokens(str(getattr(w, "word", getattr(w, "text", "")))),
                        start=float(getattr(w, "start", getattr(w, "start_time", 0))),
                        end=float(getattr(w, "end", getattr(w, "end_time", 0))),
                    ))
            except Exception:
                continue

    detected_lang = (
        getattr(result, "language", "")
        or getattr(result, "lang", "")
        or getattr(result, "lang_sym", "")
        or ""
    )
    detected_region = (
        getattr(result, "region", "")
        or getattr(result, "region_sym", "")
        or ""
    )

    return DolphinResult(
        text=text,
        words=words,
        detected_lang=str(detected_lang),
        detected_region=str(detected_region),
    )


def _strip_special_tokens(text: str) -> str:
    """移除 Dolphin 各種特殊 token：<|...|>, <CN>, <notimestamp>, <0.50> 等。"""
    text = re.sub(r"<\|[^|]*\|>", "", text)
    text = re.sub(r"<[A-Za-z][A-Za-z0-9_-]*>", "", text)
    text = re.sub(r"<\d+(?:\.\d+)?>", "", text)
    return text.strip()
