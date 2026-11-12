"""YaYan_LID_Whisper — v4.7-B：複用既有 Whisper-large-v3 做第二個 LID。

設計重點：
- **不載入新模型、不額外吃 VRAM**：直接借用 yayan.asr.whisper_global 已載入的
  HF pipeline（YaYan_ASR_Global）的 model / feature_extractor / tokenizer。
  用的是同一份權重，不會產生第二份。
- **版本相容**：鎖定 transformers==4.51.3。取語言的方式採用「forced-decoder
  第一個 token 的 logits」——這是 Whisper 自誕生以來就有的內建語言偵測機制，
  只用 tokenization_whisper.LANGUAGES 與標準 forward，不依賴新版才有的 API
  （例如不使用 model.detect_language，避免版本耦合）。
- Whisper large-v3 對中文方言（zh / yue）與外語的區分比 VoxLingua107 穩，
  用來與 VoxLingua107 投票（見 pipeline._ensemble_lid）。
"""
from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import numpy as np
import torch

from .config import CONFIG
from .lid import LANG_TO_ROUTING  # 沿用同一份 ISO639 → routing 對照，單一真相來源

logger = logging.getLogger("YaYan.LID.Whisper")

# 語言 token id 快取：list[(iso639_code, token_id)]，第一次用時建立
_LANG_TOKEN_IDS: Optional[List[Tuple[str, int]]] = None


def _components():
    """借用 whisper_global 已載入的 pipeline，回傳 (model, feature_extractor, tokenizer)。

    呼叫 whisper_global._load()：若 YaYan_ASR_Global 已載入則直接回傳既有實例，
    不會重複載入；用的是與 Global ASR 完全相同的那一份權重。
    """
    from .asr import whisper_global

    pipe = whisper_global._load()
    return pipe.model, pipe.feature_extractor, pipe.tokenizer


def _lang_token_ids(tokenizer) -> List[Tuple[str, int]]:
    """建立 Whisper 語言 token（<|en|>, <|zh|>, <|yue|>...）→ token id 對照並快取。"""
    global _LANG_TOKEN_IDS
    if _LANG_TOKEN_IDS is not None:
        return _LANG_TOKEN_IDS

    # transformers 4.51.3 既有：100 種語言的 iso639 → 全名對照
    from transformers.models.whisper.tokenization_whisper import LANGUAGES

    unk = getattr(tokenizer, "unk_token_id", None)
    ids: List[Tuple[str, int]] = []
    for code in LANGUAGES:  # e.g. 'en', 'zh', 'yue', 'ja'...
        tid = tokenizer.convert_tokens_to_ids(f"<|{code}|>")
        if tid is None or tid == unk:
            continue
        ids.append((code, int(tid)))
    if not ids:
        raise RuntimeError("Whisper-LID：找不到任何語言 token，tokenizer 不符預期")
    _LANG_TOKEN_IDS = ids
    logger.info(f"Whisper-LID：建立語言 token 對照 {len(ids)} 種")
    return ids


def detect(audio: np.ndarray, sample_rate: int = 16000) -> Tuple[str, float]:
    """用 Whisper 內建語言偵測回傳 (routing_code, confidence)。

    機制：把音訊轉成 mel features，decoder 只餵 <|startoftranscript|>，
    取下一個 token 的 logits，在「語言 token」集合上做 softmax，
    argmax 即偵測語言，對應機率即信心值。
    """
    model, feature_extractor, tokenizer = _components()
    device = model.device

    feats = feature_extractor(
        audio.astype(np.float32), sampling_rate=sample_rate, return_tensors="pt"
    ).input_features
    feats = feats.to(device=device, dtype=model.dtype)

    sot = tokenizer.convert_tokens_to_ids("<|startoftranscript|>")
    decoder_input_ids = torch.tensor([[sot]], device=device)

    with torch.no_grad():
        # forward 回傳 [batch, dec_len, vocab]；取 SOT 位置預測的下一個 token
        logits = model(feats, decoder_input_ids=decoder_input_ids).logits[0, -1]

    lang_ids = _lang_token_ids(tokenizer)
    idx = torch.tensor([tid for _, tid in lang_ids], device=device)
    probs = torch.softmax(logits[idx].float(), dim=-1)
    best = int(probs.argmax().item())

    iso639 = lang_ids[best][0]
    conf = float(probs[best].item())
    routing = LANG_TO_ROUTING.get(iso639, "auto")
    logger.info(f"Whisper-LID 偵測: {iso639} → routing={routing} (conf={conf:.2f})")
    return routing, conf
