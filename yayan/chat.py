"""AI 問答分頁的後端（tools.enable_ai_chat，預設關）。

與翻譯共用同一個 LLM（`pipeline._get_llm()`），不另外載模型。
設計重點：
- 模型的 context 有限（server 端 n_ctx），所以歷史對話與帶入的逐字稿都依字數裁切，
  寧可丟掉最舊的對話，也不讓請求超長而整個失敗。
- 預設「一次回完」（非串流）。llama.cpp 的 server 預設在收到新請求時會中斷正在串流的
  請求（interrupt_requests），而翻譯每送一批就是一個新請求，串流中的回覆會被靜默截斷。
  只有確定 server 端已關閉該行為時，才把 tools.ai_chat_stream 設為 true。
- 串流模式下若回覆沒有正常結束（finish_reason 為空），會明確標示「回覆被中斷」。
"""
from __future__ import annotations

import logging
from typing import Dict, Iterator, List, Optional

from .config import CONFIG

logger = logging.getLogger("YaYan.Chat")

SYSTEM_PROMPT = (
    "你是「雅言 YaYan」語音轉錄系統內建的 AI 助理，部署在離線環境，無法連網，"
    "也查不到即時資訊。請一律使用台灣正體中文與台灣慣用語回答，內容力求準確、簡潔。"
    "不確定或不知道的事請直接說不知道，不要編造。"
    "不要提及底層模型或供應商的名稱；被問到時，回答你是雅言 YaYan 自主研發模型。"
)

_CONTEXT_HEADER = (
    "以下是使用者目前在系統中的轉錄結果，供回答時參考。"
    "它是語音辨識與機器翻譯的產物，可能有聽錯或譯錯之處；"
    "若使用者的問題與它無關，請忽略。\n"
)


# 以這些字樣開頭/結尾的 assistant 訊息是介面自己產生的提示，不是模型的回答，不送回模型
ERROR_PREFIX = "⚠️"
INTERRUPTED_SUFFIX = "\n\n⚠️（回覆被中斷，內容不完整；請再問一次。）"
EMPTY_REPLY = "⚠️ 模型沒有產生內容。可能是提問或帶入的內容過長，請縮短後再試一次。"
# 用完整字樣比對，避免模型正常回答裡剛好有 ⚠️ 就被整輪丟掉；"…" 是等待中的佔位字
_UI_NOTICE_MARKERS = ("⚠️ 目前無法取得回覆", "⚠️（回覆被中斷", "⚠️ 模型沒有產生內容")


def _cfg() -> Dict:
    return CONFIG.get("tools", {}) or {}


def _clip(text: str, limit: int) -> str:
    text = (text or "").strip()
    if len(text) <= limit:
        return text
    # 頭尾各留一半：錄音內容的開場與結尾（結論）通常都重要
    half = limit // 2
    return text[:half] + "\n…（中間略）…\n" + text[-half:]


def build_messages(
    history: List[Dict[str, str]],
    user_message: str,
    transcript: str = "",
    translation: str = "",
) -> List[Dict[str, str]]:
    """組出送給 LLM 的 messages。history 為 [{role, content}, ...]（不含本次提問）。"""
    cfg = _cfg()
    ctx_limit = int(cfg.get("ai_chat_context_chars", 3000))
    hist_limit = int(cfg.get("ai_chat_history_chars", 2500))
    q_limit = int(cfg.get("ai_chat_question_chars", 2000))
    transcript = transcript if isinstance(transcript, str) else ""
    translation = translation if isinstance(translation, str) else ""

    system = SYSTEM_PROMPT
    blocks = []
    if (transcript or "").strip():
        blocks.append("【辨識原文】\n" + _clip(transcript, ctx_limit // 2))
    if (translation or "").strip():
        blocks.append("【譯文】\n" + _clip(translation, ctx_limit // 2))
    if blocks:
        system += "\n\n" + _CONTEXT_HEADER + "\n\n".join(blocks)

    # 先整理成乾淨的「一問一答」：缺任一方、或回答是介面提示（錯誤/被中斷）的輪次整輪捨棄，
    # 確保送出的 messages 一定是 user/assistant 交替。
    pairs: List[List[Dict[str, str]]] = []
    pending: Optional[str] = None
    for m in history or []:
        if not isinstance(m, dict):
            continue
        role, content = m.get("role"), m.get("content")
        text = content.strip() if isinstance(content, str) else ""
        if role == "user":
            pending = text or None
        elif role == "assistant":
            if pending and text and text != "…" and not any(mk in text for mk in _UI_NOTICE_MARKERS):
                pairs.append([
                    {"role": "user", "content": pending},
                    {"role": "assistant", "content": text},
                ])
            pending = None

    # 由新到舊保留，超過字數上限就捨棄更舊的
    kept: List[Dict[str, str]] = []
    used = 0
    for pair in reversed(pairs):
        size = sum(len(m["content"]) for m in pair)
        if used + size > hist_limit:
            break
        kept = pair + kept
        used += size

    return [{"role": "system", "content": system}] + kept + [
        {"role": "user", "content": _clip(str(user_message), q_limit)}
    ]


def stream_reply(messages: List[Dict[str, str]]) -> Iterator[str]:
    """逐步產出「到目前為止的完整回覆」。"""
    from . import pipeline  # 延後 import：避免問答分頁關閉時多載一層

    cfg = _cfg()
    max_tokens = int(cfg.get("ai_chat_max_new_tokens", 1024))
    temperature = float(cfg.get("ai_chat_temperature", CONFIG["llm"].get("temperature", 0.6)))
    llm = pipeline._get_llm()

    use_stream = llm.backend == "openai" and bool(cfg.get("ai_chat_stream", False))
    if not use_stream:
        text = llm.chat(messages, max_new_tokens=max_tokens, temperature=temperature)
        yield text or EMPTY_REPLY
        return

    stream = llm._oai.chat.completions.create(
        model=llm._served_name,
        messages=messages,
        temperature=temperature,
        top_p=CONFIG["llm"].get("top_p", 0.9),
        max_tokens=max_tokens,
        extra_body=llm._openai_extra_body(),
        stream=True,
    )
    acc = ""
    finish_reason = None
    for chunk in stream:
        if not chunk.choices:
            continue
        choice = chunk.choices[0]
        finish_reason = getattr(choice, "finish_reason", None) or finish_reason
        delta: Optional[str] = getattr(choice.delta, "content", None)
        if delta:
            acc += delta
            yield acc
    if not acc:
        yield EMPTY_REPLY
    elif finish_reason is None:
        # server 因為收到別的請求而提前結束這個串流：不可當成完整回答
        logger.warning("AI 問答串流未正常結束（finish_reason 為空），標示為被中斷")
        yield acc + INTERRUPTED_SUFFIX
