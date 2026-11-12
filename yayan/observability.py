"""可觀測性：讓雅言自己的 log 不被第三方套件關掉，並提供用量紀錄與輸出檢查的小工具。

背景：第一次轉錄時，某個語音套件會把 root logger 的等級調成 ERROR，之後所有經由 root
輸出的 INFO / WARNING（包含 YaYan.* 的路由、LID、回退訊息）全部消失，事後無從查證
某條路徑有沒有被走到。這裡讓 "YaYan" 這棵 logger 自帶 handler 與等級，不再依賴 root。
"""
from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
import unicodedata
from typing import Any, Dict, Tuple

from .config import CONFIG

_LOG_FORMAT = "[%(asctime)s] [%(name)s] %(levelname)s: %(message)s"
_usage_lock = threading.Lock()


def setup_logging() -> None:
    """讓 YaYan.* 的 log 自成一棵：自帶 handler、固定 INFO、不往 root 傳。可重複呼叫。"""
    lg = logging.getLogger("YaYan")
    if not any(getattr(h, "_yayan", False) for h in lg.handlers):
        h = logging.StreamHandler()
        h.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt="%H:%M:%S"))
        h._yayan = True  # type: ignore[attr-defined]
        lg.addHandler(h)
    lg.setLevel(logging.INFO)
    lg.propagate = False


def log_usage(kind: str, **fields: Any) -> None:
    """把一次操作的摘要寫成一行 JSON（logging.usage_log_path，預設空＝不寫）。

    只記統計數字（長度、語種、段數、耗時、失敗數），**不記檔名與任何文字內容**。
    任何寫入失敗都只記 log，不影響主流程。
    """
    path = (CONFIG.get("logging", {}) or {}).get("usage_log_path", "")
    if not path:
        return
    try:
        rec: Dict[str, Any] = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "kind": kind}
        rec.update(fields)
        line = json.dumps(rec, ensure_ascii=False, default=str)
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with _usage_lock, open(path, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception as e:
        logging.getLogger("YaYan.Usage").warning(f"用量紀錄寫入失敗：{e}")


# 辨識模型陷入迴圈時會把同一片段重複幾十上百次。判定刻意保守——寧可漏截，不可誤截
# （逐字稿裡的數字報讀與連續應答都是正常內容）：
#   - 片段的實質內容有一半以上是數字（含中文數字、報讀用字、英文數字單字）就完全不動。
#   - 其餘片段要連續重複 5 次以上，且重複區段的實質內容合計 50 字以上，才視為迴圈
#     （例：2 個字的片段要連續 25 次、10 個字的句子要連續 5 次）。真正的迴圈動輒數百字。
#   - 截斷後保留一次；單一字元的疊字保留兩個（「哈哈」「謝謝」）。
_LOOP_UNIT = re.compile(r"(.{1,40}?)\1{4,}", re.S)
_NUMERIC_EXTRA = set("零〇兩两洞么幺拐勾點点廿卅壹貳參肆伍陸柒捌玖拾佰仟")
_NUMBER_WORDS = {"zero", "oh", "one", "two", "three", "four", "five", "six", "seven", "eight",
                 "nine", "ten", "double", "triple", "hundred", "thousand", "point"}
_NON_CONTENT = re.compile(r"[\s，。、！？；：,.!?;:…·\-—~～「」『』（）()\[\]\"']+")
_WORD = re.compile(r"[A-Za-z]+")


def _is_numeric_char(ch: str) -> bool:
    return ch in _NUMERIC_EXTRA or unicodedata.numeric(ch, None) is not None


def _mostly_numeric(unit: str, core: str) -> bool:
    words = _WORD.findall(unit)
    if words and all(w.lower() in _NUMBER_WORDS for w in words):
        return True
    return 2 * sum(_is_numeric_char(c) for c in core) >= len(core)


def _collapse(m: "re.Match") -> str:
    unit, whole = m.group(1), m.group(0)
    core = _NON_CONTENT.sub("", unit)
    if not core or _mostly_numeric(unit, core):
        return whole
    total = (len(whole) // len(unit)) * len(core)
    if total >= 50:
        return unit * 2 if len(unit) == 1 else unit
    return whole


def collapse_repetition(text: str) -> Tuple[str, bool]:
    """把辨識模型的重複迴圈截短（保留一次；單字疊字保留兩個）。回傳 (處理後文字, 是否有截斷)。"""
    if not text or len(text) < 20:
        return text, False
    out = _LOOP_UNIT.sub(_collapse, text)
    return out, out != text
