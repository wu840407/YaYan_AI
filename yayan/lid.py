"""YaYan_LID — 語種識別（VoxLingua107 ECAPA-TDNN 包裝）。"""
from __future__ import annotations

import logging
import shutil
from typing import Tuple

import numpy as np
import torch

from .config import CONFIG, model_path

logger = logging.getLogger("YaYan.LID")

_LID_MODEL = None

# VoxLingua107 偵測結果（ISO 639-1） → default.yaml 的 routing key
LANG_TO_ROUTING = {
    # ---- 漢語系 ----
    "zh": "zh", "cmn": "zh", "nan": "zh",
    "yue": "yue", "wuu": "wuu", "cdo": "cdo",
    # ---- 東亞 ----
    "ja": "ja", "ko": "ko",
    # ---- 中亞 ----
    "bo": "bo", "ug": "ug",
    "kk": "ug",   # 哈薩克語就近走 Eastern
    "mn": "ug",   # 蒙古語就近走 Eastern
    # ---- 中東 / 南亞 ----
    "fa": "fa", "ur": "ur", "ar": "ar", "hi": "hi",
    "bn": "hi",   # 孟加拉語近似
    # ---- 歐洲 ----
    "en": "en", "fr": "fr", "de": "de", "ru": "ru", "es": "es",
    "it": "fr",   # 沒專屬路由就近走法語（都歐洲）
    "pt": "es",   # 葡萄牙語近似西班牙語
    "nl": "de",   # 荷蘭語近似德語
    "pl": "ru",   # 波蘭語近似俄語
    "uk": "ru",   # 烏克蘭語走俄語
    # ---- 東南亞 ----
    "th": "th", "ms": "ms", "vi": "vi", "id": "id",
    "tl": "ms",   # 菲律賓語就近走馬來語
}


def _patch_torch_amp() -> None:
    """把 torch.cuda.amp.custom_fwd/bwd 別名到 torch.amp 下。

    SpeechBrain 1.1 用的是 `torch.amp.custom_fwd`，那個位置 torch 2.4+ 才有；
    本專案鎖 torch 2.3.1，同樣的函式在 `torch.cuda.amp` 底下。補別名即可，
    不需要（也不可以）動版本鎖。

    ⚠️ 不補的後果不是報錯而是**靜默失效**：LID 每一段都拋
    AttributeError，被 pipeline 吞成 lid_error → 所有語段退回預設路由，
    方言路由等於不存在，enable_lid_context / enable_lid_ensemble 也一併失效。
    2026-08-05 查出時已經這樣壞了一段時間，症狀是「所有方言都被判成通用 zh」。
    """
    if hasattr(torch.amp, "custom_fwd") or not hasattr(torch.cuda.amp, "custom_fwd"):
        return

    # 不能直接別名：新版簽名多了 device_type=，torch 2.3.1 的版本不吃這個關鍵字，
    # 直接指過去會變成 TypeError。包一層把 device_type 吃掉再轉呼叫。
    def custom_fwd(fwd=None, *, device_type=None, cast_inputs=None):
        if fwd is None:
            def deco(f):
                if cast_inputs is None:
                    return torch.cuda.amp.custom_fwd(f)
                return torch.cuda.amp.custom_fwd(f, cast_inputs=cast_inputs)
            return deco
        return torch.cuda.amp.custom_fwd(fwd)

    def custom_bwd(bwd=None, *, device_type=None):
        if bwd is None:
            return lambda f: torch.cuda.amp.custom_bwd(f)
        return torch.cuda.amp.custom_bwd(bwd)

    torch.amp.custom_fwd = custom_fwd  # type: ignore[attr-defined]
    torch.amp.custom_bwd = custom_bwd  # type: ignore[attr-defined]
    logger.info("已補 torch.amp.custom_fwd/bwd 相容包裝（SpeechBrain 1.1 × torch 2.3.1）")


def _ensure_label_encoder(local_dir) -> None:
    """確保 label_encoder.ckpt 是可讀的真檔案。

    SpeechBrain 的 fetch 機制在 source 是本機路徑時，會在 savedir 建一個指向
    「工作目錄 + 檔名」的 symlink（例：/data/AI_Project/label_encoder.txt），
    那個位置根本沒有檔案 → 斷鏈 → 載入時 KeyError（標籤表是空的，模型輸出的
    類別索引查不到）。真正的表就在模型目錄的 label_encoder.txt。
    """
    ckpt = local_dir / "label_encoder.ckpt"
    txt = local_dir / "label_encoder.txt"
    if ckpt.exists() and ckpt.stat().st_size > 0:
        return
    if not txt.exists():
        logger.warning("label_encoder.txt 不存在，LID 標籤可能無法解碼：%s", txt)
        return
    try:
        if ckpt.is_symlink() or ckpt.exists():
            ckpt.unlink()
        shutil.copyfile(txt, ckpt)
        logger.info("已修復 label_encoder.ckpt（原為斷鏈 symlink）")
    except OSError as e:
        logger.warning("修復 label_encoder.ckpt 失敗：%s", e)


def _load() -> None:
    global _LID_MODEL
    if _LID_MODEL is not None:
        return
    _patch_torch_amp()
    try:
        from speechbrain.inference.classifiers import EncoderClassifier
    except ImportError:
        from speechbrain.pretrained import EncoderClassifier  # type: ignore

    local_dir = model_path("YaYan_LID")
    if not local_dir.exists():
        raise FileNotFoundError(f"YaYan_LID 不存在: {local_dir}")
    _ensure_label_encoder(local_dir)
    logger.info("載入 YaYan_LID …")
    device = CONFIG["devices"]["asr_gpu"]
    _LID_MODEL = EncoderClassifier.from_hparams(
        source=str(local_dir),
        savedir=str(local_dir),
        run_opts={"device": device},
    )


def detect(audio: np.ndarray, sample_rate: int = 16000) -> Tuple[str, float]:
    """回傳 (routing_code, confidence)。routing_code 為 default.yaml 的 asr.routing 鍵。"""
    _load()
    audio_t = torch.from_numpy(audio).float().unsqueeze(0)
    out = _LID_MODEL.classify_batch(audio_t)
    score = float(out[1].exp().max().item())
    label = out[3][0]
    iso639 = label.split(":")[0].strip().lower() if isinstance(label, str) else "auto"
    routing = LANG_TO_ROUTING.get(iso639, "auto")
    logger.info(f"YaYan_LID 偵測: {iso639} → routing={routing} (conf={score:.2f})")
    return routing, score
