"""YaYan_TWConv — 強制台灣正體後處理（OpenCC s2twp）+ 清理 LLM 殘留。"""
from __future__ import annotations

import logging
import re
from typing import Optional

from ..config import CONFIG

logger = logging.getLogger("YaYan.TWConv")

_CONVERTER = None

# Qwen3 / DeepSeek-R1 系列的 thinking 標記
_THINK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL | re.IGNORECASE)
# 模型常見的開頭贅詞
_PREFIX_RE = re.compile(
    r"^(?:好的[，,。.]?|當然[，,。.]?|以下是[^：:]*[：:]?\s*|"
    r"翻譯[後如下]*[：:]?\s*|譯文[：:]?\s*|這裡是[^：:]*[：:]?\s*)",
    re.IGNORECASE,
)
# 常被夾帶的 markdown 程式碼框
_CODEFENCE_RE = re.compile(r"^```[a-zA-Z]*\n?|\n?```$", re.MULTILINE)


def _get_converter():
    global _CONVERTER
    if _CONVERTER is not None:
        return _CONVERTER
    if not CONFIG["postprocess"]["use_opencc"]:
        return None
    try:
        from opencc import OpenCC
    except ImportError:
        logger.warning("opencc 未安裝，略過台灣正體轉換。")
        return None
    cfg_name = CONFIG["postprocess"]["opencc_config"]
    _CONVERTER = OpenCC(cfg_name.replace(".json", ""))
    return _CONVERTER


def strip_llm_artifacts(text: str) -> str:
    """剝除 <think>、贅詞前綴、未閉合 markdown 碼框等。"""
    if not text:
        return text
    text = _THINK_RE.sub("", text)
    text = _CODEFENCE_RE.sub("", text)
    text = _PREFIX_RE.sub("", text.lstrip())
    return text.strip()


def to_taiwan_traditional(text: str) -> str:
    """LLM 殘留剝除 → 簡轉台灣正體（含慣用詞）。"""
    if not text:
        return text
    text = strip_llm_artifacts(text)
    conv = _get_converter()
    if conv is None:
        return text
    try:
        return conv.convert(text)
    except Exception as e:
        logger.warning(f"OpenCC 轉換失敗: {e}")
        return text
