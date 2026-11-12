"""YaYan 聲紋抽取（v5.0 M2）。

用 wespeaker resnet34 embedding 抽 256 維聲紋向量，並 L2 normalize。
embedding 物件複用 diarization pipeline（見 diarize.get_embedding_model），
不重載、不額外吃 VRAM。

向量維度：實測 256（wespeaker-voxceleb-resnet34-LM）。
"""
from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import numpy as np

logger = logging.getLogger("YaYan.Voiceprint")

EMBEDDING_DIM = 256
_MIN_SECONDS = 0.5   # 短於此聲紋不可靠


def l2_normalize(vec: np.ndarray) -> np.ndarray:
    """L2 正規化（零向量原樣回傳）。"""
    vec = np.asarray(vec, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(vec))
    if norm < 1e-12:
        return vec
    return vec / norm


def _to_batch(audio: np.ndarray):
    """轉成 wespeaker 期望的 (batch=1, channel=1, samples) torch tensor。"""
    import torch
    a = np.asarray(audio, dtype=np.float32).reshape(-1)
    return torch.from_numpy(a).reshape(1, 1, -1)


def extract(audio: np.ndarray, sample_rate: int = 16000) -> Optional[np.ndarray]:
    """從單段音訊抽 1 個 L2-normalized 256 維聲紋向量。

    太短（< 0.5s）、含 NaN 或失敗回 None。
    """
    a = np.asarray(audio, dtype=np.float32).reshape(-1)
    if a.size < int(sample_rate * _MIN_SECONDS):
        return None

    from . import diarize
    import torch

    model = diarize.get_embedding_model()
    try:
        with torch.no_grad():
            emb = model(_to_batch(a))
    except Exception as e:
        logger.warning(f"聲紋抽取失敗: {e}")
        return None

    emb = np.asarray(emb, dtype=np.float32).reshape(-1)
    if emb.size != EMBEDDING_DIM:
        logger.warning(f"聲紋維度異常: {emb.shape}（預期 {EMBEDDING_DIM}）")
    if not np.all(np.isfinite(emb)):
        logger.warning("聲紋向量含 NaN/Inf，捨棄。")
        return None
    return l2_normalize(emb)


def extract_from_segments(
    audio: np.ndarray,
    segments: List[Tuple[float, float]],
    sample_rate: int = 16000,
    max_seconds: float = 30.0,
) -> Optional[np.ndarray]:
    """把同一語者的多個 (start, end) 區段音訊串接後抽 1 個代表向量。

    最多取 max_seconds 秒（夠抽穩定聲紋，且避免超長串接拖慢）。
    """
    if not segments:
        return None
    sr = sample_rate
    pieces: List[np.ndarray] = []
    total = 0.0
    for s, e in segments:
        if e <= s:
            continue
        i0 = max(0, int(round(s * sr)))
        i1 = min(len(audio), int(round(e * sr)))
        if i1 <= i0:
            continue
        pieces.append(audio[i0:i1])
        total += (i1 - i0) / sr
        if total >= max_seconds:
            break
    if not pieces:
        return None
    concat = np.concatenate(pieces).astype(np.float32)
    return extract(concat, sample_rate=sr)
