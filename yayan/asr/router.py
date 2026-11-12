"""YaYan ASR 路由器：依 routing key 決定走哪個 ASR 引擎。

歷史變動（v4.6）：
  - 漢語方言（22 種）→ YaYan_ASR_Dialect (Dolphin-CN-Dialect-Small)
  - 日韓 → YaYan_ASR_Global (Whisper-large-v3，比 SenseVoice 在日韓更強)
  - 中亞 (bo/ug) → YaYan_ASR_Dialect (Dolphin 也支援)
  - 其他 → YaYan_ASR_Global
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from ..config import CONFIG

logger = logging.getLogger("YaYan.ASR.Router")

# 漢語方言 + Dolphin 直接支援的非中文
DOLPHIN_ROUTINGS = {
    # 漢語方言
    "zh", "cmn", "yue", "wuu", "nan", "cdo", "hak", "hsn", "gan", "cjy",
    "wuu-sz", "wuu-nb", "wuu-wz",
    "nan-cs", "nan-hn",
    "cmn-sw", "cmn-sd", "cmn-ne", "cmn-zy", "cmn-wh", "cmn-xa", "cmn-lz", "cmn-jh",
    # 中亞
    "bo", "ug",
    "min", "hokkien",  # 別名
}

# v4.7-C：台語專用 ASR 觸發的 routing key（預設值，可由 config 覆寫）
_DEFAULT_TAIGI_ROUTINGS = {"nan", "nan-cs", "nan-hn", "min", "hokkien"}


def _force_whisper_routings() -> set:
    """強制改走 Whisper 的 routing key（config asr.force_whisper_routings）。

    用途：Dolphin-small 對某些語種覆蓋不佳時的逃生門。實測 yue（粵語）在
    Dolphin 只吐國語轉寫，Whisper 指定 language=cantonese 才會產出真正的粵文；
    bo（藏語）在 Dolphin 被映射成 zh-CN，等於當中文解碼。
    repo 預設空集合＝行為與既有完全相同。
    """
    v = CONFIG["asr"].get("force_whisper_routings") or []
    if isinstance(v, str):
        v = [v]
    return {str(x).strip().lower() for x in v if str(x).strip()}


def _taigi_routings() -> set:
    cfg = CONFIG["asr"].get("taigi_routings")
    return {str(r).lower() for r in cfg} if cfg else _DEFAULT_TAIGI_ROUTINGS


@dataclass
class WordTS:
    word: str
    start: float  # 全域時間（chunk_start_in_audio + word_offset_in_chunk）
    end: float


@dataclass
class ASRResult:
    text: str
    asr_alias: str
    routing: str
    words: List[WordTS] = None  # type: ignore
    error: str = ""   # 引擎推論失敗的原因；空字串＝沒有失敗


def transcribe(
    audio: np.ndarray,
    routing: str = "auto",
    language_hint: Optional[str] = None,
    chunk_start_sec: float = 0.0,
) -> ASRResult:
    """依 routing key 選 ASR 引擎。

    chunk_start_sec: 用於把 word timestamp 從 chunk-local 轉成 audio-global
    """
    routing = (routing or "auto").lower()
    logger.info(f"路由 → routing={routing}")

    # v4.7-C：台語(nan)專用 ASR（預設關）。載入/識別失敗自動退回 Dolphin。
    if CONFIG["asr"].get("enable_taigi_asr", False) and routing in _taigi_routings():
        try:
            return _taigi_transcribe(audio, routing, chunk_start_sec)
        except Exception as e:
            logger.warning(f"台語專用 ASR 失敗，退回 Dolphin（routing={routing}）: {e}")

    # 有些語種 Dolphin-small 覆蓋不佳，改交給 Whisper（config 列表，repo 預設空＝行為不變）
    if routing in _force_whisper_routings():
        logger.info(f"routing={routing} 在 force_whisper 清單中，改走 Whisper")
        return _whisper_transcribe(audio, routing, chunk_start_sec)

    if routing in DOLPHIN_ROUTINGS or routing == "auto":
        # 包含 auto → Dolphin 自己會做 LID
        return _dolphin_transcribe(audio, routing, chunk_start_sec)
    else:
        # 日韓 / 歐美 / 中東 → Whisper
        return _whisper_transcribe(audio, routing, chunk_start_sec)


def _asr_hotwords(routing: str) -> Optional[list]:
    """術語庫 → ASR hotword 偏置詞表（開關 asr.enable_asr_hotwords，預設關）。

    任何失敗都回 None＝不做偏置：術語庫掛掉不該讓轉錄跟著掛。
    """
    if not CONFIG["asr"].get("enable_asr_hotwords", False):
        return None
    try:
        from ..glossary import hotwords_for_asr
        return hotwords_for_asr(source_lang=routing or "any") or None
    except Exception as e:
        logger.warning(f"ASR hotword 取詞失敗，本段不偏置：{e}")
        return None


def _dolphin_transcribe(
    audio: np.ndarray, routing: str, chunk_start_sec: float,
) -> ASRResult:
    from . import dolphin as _dolphin_mod
    res = _dolphin_mod.transcribe(
        audio, language_hint=routing, hotwords=_asr_hotwords(routing)
    )

    # VoxLingua107 的標籤集裡沒有閩南語（只有 "Chinese"），Whisper LID 也沒有 nan，
    # 所以自動模式永遠判不出台語，台語專用 ASR 形同虛設。
    # Dolphin 自己有 zh-MINNAN 判定，拿它當觸發器把該段改交台語專用 ASR 重跑。
    if (
        CONFIG["asr"].get("enable_minnan_redirect", False)
        and CONFIG["asr"].get("enable_taigi_asr", False)
        and (res.detected_region or "").strip().upper() == "MINNAN"
    ):
        try:
            logger.info("Dolphin 判定 region=MINNAN → 改走台語專用 ASR 重跑")
            return _taigi_transcribe(audio, "nan", chunk_start_sec)
        except Exception as e:
            logger.warning(f"台語專用 ASR 失敗，沿用 Dolphin 結果：{e}")
    words = [
        WordTS(
            word=w.word,
            start=w.start + chunk_start_sec,
            end=w.end + chunk_start_sec,
        )
        for w in (res.words or [])
    ]
    return ASRResult(
        text=res.text,
        asr_alias="YaYan_ASR_Dialect",
        routing=routing,
        words=words,
        error=getattr(res, "error", "") or "",
    )


def _taigi_transcribe(
    audio: np.ndarray, routing: str, chunk_start_sec: float,
) -> ASRResult:
    """台語專用 ASR（Breeze-ASR-26）。回傳純文字，無 word timestamp。"""
    from . import whisper_taigi as _taigi
    text = _taigi.transcribe(audio, language_hint=routing)
    return ASRResult(
        text=text,
        asr_alias="YaYan_ASR_Taigi",
        routing=routing,
        words=[],
    )


def _whisper_transcribe(
    audio: np.ndarray, routing: str, chunk_start_sec: float,
) -> ASRResult:
    """Whisper 路徑（日韓、波斯、英、法、德、俄、泰、馬來等）。"""
    from . import whisper_global as _whisper
    res = _whisper.transcribe(audio, language_hint=routing)

    # 嘗試從 Whisper 取 word timestamp
    words = []
    if hasattr(res, "words") and res.words:
        for w in res.words:
            try:
                words.append(WordTS(
                    word=str(w["word"]) if isinstance(w, dict) else str(w.word),
                    start=float(w["start"] if isinstance(w, dict) else w.start) + chunk_start_sec,
                    end=float(w["end"] if isinstance(w, dict) else w.end) + chunk_start_sec,
                ))
            except Exception:
                continue
    text = getattr(res, "text", "") if hasattr(res, "text") else str(res)

    return ASRResult(
        text=text,
        asr_alias="YaYan_ASR_Global",
        routing=routing,
        words=words,
    )

# 向後相容
AsrResult = ASRResult

