"""YaYan_VAD — silero-vad 5.x+ 改用 pip 內建權重，不再從 HF 下載。"""
from __future__ import annotations

import logging
from typing import List, Tuple

import numpy as np
import torch

from .config import CONFIG

logger = logging.getLogger("YaYan.VAD")

_MODEL = None


def _get_model():
    global _MODEL
    if _MODEL is not None:
        return _MODEL
    from silero_vad import load_silero_vad
    _MODEL = load_silero_vad()  # ← pip 內建權重，不打網路
    logger.info("silero-vad 載入完成（pip-bundled）")
    return _MODEL


def split_speech(
    audio: np.ndarray,
    sample_rate: int = 16000,
) -> List[Tuple[float, float, np.ndarray]]:
    """切出語音段。回傳 [(start_sec, end_sec, chunk_audio), ...]。"""
    from silero_vad import get_speech_timestamps

    cfg = CONFIG["audio"]
    threshold = cfg.get("vad_threshold", 0.5)
    min_dur = cfg.get("min_chunk_seconds", 0.5)
    max_dur = cfg.get("max_chunk_seconds", 30)
    pad = cfg.get("pad_seconds", 0.2)

    model = _get_model()
    waveform = torch.from_numpy(audio).float()

    timestamps = get_speech_timestamps(
        waveform,
        model,
        sampling_rate=sample_rate,
        threshold=threshold,
        min_speech_duration_ms=int(min_dur * 1000),
        max_speech_duration_s=max_dur,
        speech_pad_ms=int(pad * 1000),
        return_seconds=False,  # 用 sample 為單位
    )

    chunks: List[Tuple[float, float, np.ndarray]] = []
    for ts in timestamps:
        s, e = int(ts["start"]), int(ts["end"])
        chunks.append((s / sample_rate, e / sample_rate, audio[s:e]))

    if not chunks:
        # 整段都沒檢測到語音，當作一整段送
        chunks = [(0.0, len(audio) / sample_rate, audio)]
    
    logger.info(f"VAD 切出 {len(chunks)} 段")
    return chunks