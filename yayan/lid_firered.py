"""YaYan_LID_Dialect — 漢語方言語種識別（FireRedLID 包裝）。

與 `yayan/lid.py`（VoxLingua107）介面相同（`detect(audio, sample_rate) -> (routing, conf)`），
可由 config 的 `asr.lid_backend` 切換。差別在於 VoxLingua107 對整個漢語只有一個
"Chinese" 標籤，本模組能分出閩/粵/吳/湘/西南官話，台語與粵語因此才走得到專用 ASR。

⚠️ 標籤集限制（模型本身沒有這些標籤，不是判斷失誤）：
維吾爾語、贛語、客家話**不在** FireRedLID 的標籤表內，永遠不會被判出來。
"""
from __future__ import annotations

import logging
import os
import sys
from typing import Optional, Tuple

import numpy as np

from .config import CONFIG

logger = logging.getLogger("YaYan.LID.Dialect")

_MODEL = None

# FireRedLID 輸出（漢語為 "<lang> <dialect>" 兩層）→ default.yaml 的 routing key
LABEL_TO_ROUTING = {
    # ---- 漢語方言（本模組存在的理由）----
    "zh min": "nan",        # 閩語 → 閩南/台語 ASR（Breeze-ASR-Taigi）
    "zh yue": "yue",        # 粵語 → Whisper（必須指定 cantonese）
    "zh wu": "wuu",         # 吳語
    "zh xiang": "hsn",      # 湘語
    "zh xinan": "cmn-sw",   # 西南官話（四川話為代表）
    "zh mandarin": "zh",
    "zh north": "zh",       # 北方官話
    "zh": "zh",
    # ---- 其他語言（沿用 lid.py 的就近映射）----
    "bo": "bo", "ja": "ja", "ko": "ko",
    "en": "en", "fr": "fr", "de": "de", "ru": "ru", "es": "es",
    "it": "fr", "pt": "es", "nl": "de", "pl": "ru", "uk": "ru",
    "th": "th", "ms": "ms", "vi": "vi", "id": "id", "tl": "ms",
    "fa": "fa", "ur": "ur", "ar": "ar", "hi": "hi", "bn": "hi",
    "kk": "ug", "mn": "ug", "uz": "ug", "tr": "ug", "tt": "ug",
    # ⚠️ 緬甸語(my)、寮語(lo)、高棉語(km)刻意不列，判出來時退回 auto（通用中文引擎）：
    #    (1) 低取樣率語音上的閩南語很常被整段判成 my/lo，若直接送去
    #        當緬甸語辨識，這些錄音會從「辨識出一部分」變成完全不可用；
    #    (2) 辨識引擎對這三種語言的輸出本身不可用（見 asr/whisper_global.py 的註解）。
}

_DEFAULT_MODEL_DIR = "/data/ai_models/YaYan_LID_Dialect"
_DEFAULT_REPO_DIR = "/data/AI_Project/third_party/FireRedASR2S"


def _load() -> None:
    """延後載入：只有真的切到本 backend 才吃記憶體。"""
    global _MODEL
    if _MODEL is not None:
        return

    asr_cfg = CONFIG.get("asr", {})
    repo_dir = asr_cfg.get("firered_repo_dir", _DEFAULT_REPO_DIR)
    model_dir = asr_cfg.get("firered_lid_dir", _DEFAULT_MODEL_DIR)

    if not os.path.isdir(model_dir):
        raise FileNotFoundError(f"FireRedLID 模型目錄不存在: {model_dir}")
    if repo_dir not in sys.path:
        sys.path.insert(0, repo_dir)

    from fireredasr2s.fireredlid import FireRedLid, FireRedLidConfig

    use_half = bool(asr_cfg.get("firered_lid_half", False))
    _MODEL = FireRedLid.from_pretrained(
        model_dir, FireRedLidConfig(use_gpu=True, use_half=use_half)
    )
    logger.info(f"FireRedLID 載入完成（half={use_half}）")


def detect(audio: np.ndarray, sample_rate: int = 16000) -> Tuple[str, float]:
    """回傳 (routing_key, confidence)。判不出來時回 ("auto", 0.0)。"""
    _load()

    # FeatExtractor 走的是 kaldiio.load_mat 的資料型態（int16 尺度），
    # 但 pipeline 傳進來的是 float32 [-1, 1]，要先換算。
    if audio.dtype != np.int16:
        audio = np.clip(audio, -1.0, 1.0)
        audio = (audio * 32767.0).astype(np.int16)

    if CONFIG.get("asr", {}).get("firered_lid_rescue", False):
        try:
            return _detect_with_rescue(audio, sample_rate)
        except Exception as e:  # 救回路徑出錯時退回原本的 1-best，不影響既有行為
            logger.warning(f"FireRedLID 兩步判定失敗，退回 1-best: {e}")

    results = _MODEL.process(["seg"], [(sample_rate, audio)])
    if not results:
        return "auto", 0.0

    label = (results[0].get("lang") or "").strip()
    conf = float(results[0].get("confidence") or 0.0)
    if not label:
        return "auto", 0.0

    routing = _label_to_routing(label)
    if routing is None:
        logger.debug(f"FireRedLID 標籤 {label!r} 無對應 routing，退回 auto")
        return "auto", conf
    return routing, conf


def _label_to_routing(label: str) -> Optional[str]:
    if label in LABEL_TO_ROUTING:
        return LABEL_TO_ROUTING[label]
    # 兩層標籤但方言部分不認識時（例如未來新增的方言），退回語言層
    head = label.split()[0]
    return LABEL_TO_ROUTING.get(head)


# ---------------------------------------------------------------------------
# 兩步判定（asr.firered_lid_rescue，預設關）
#
# 低取樣率語音上，1-best 常把粵語/閩南語判成越南語、緬甸語、泰語等；
# 但同一段音訊「若強制第一個 token 為 zh，再看方言分佈」大多判得對。
# 因此：第一步判成容易混淆的外語、而強制漢語後方言分佈又很篤定時，改採方言結果。
#
# ⚠️ 不建議開啟：用公開的 FLEURS 越南、寮、泰、緬甸語實測，開啟後有 54～87% 的
# 真外語被誤判成漢語方言。只對 rescue_from 清單內的語言、且方言機率夠高時才救，
# 並且預設關閉。
# ---------------------------------------------------------------------------
_DEFAULT_RESCUE_FROM = ("vi", "my", "th", "lo", "yo", "haw", "so", "jw", "su", "war")
_DIALECTS = ("mandarin", "north", "xinan", "min", "wu", "xiang", "yue")
_TOKENS = None


def _token_table():
    global _TOKENS
    if _TOKENS is None:
        model_dir = CONFIG.get("asr", {}).get("firered_lid_dir", _DEFAULT_MODEL_DIR)
        with open(os.path.join(model_dir, "dict.txt"), encoding="utf-8") as f:
            _TOKENS = [line.split()[0] for line in f if line.strip()]
    return _TOKENS


def _posteriors(audio_int16: np.ndarray, sample_rate: int):
    """回傳 (語言分佈, 強制 zh 後的方言分佈)，皆為對整個 token 表的機率向量。"""
    import torch

    tok = _token_table()
    feats, lengths, _durs, _, _ = _MODEL.feat_extractor([(sample_rate, audio_int16)], ["seg"])
    if feats is None:
        return None, None
    cfg = _MODEL.config
    if cfg.use_gpu:
        feats, lengths = feats.cuda(), lengths.cuda()
        if cfg.use_half:
            feats = feats.half()
    dec = _MODEL.model.lid_decoder

    def step(ys, enc, mask):
        tgt_mask = dec.ignored_target_position_is_0(ys, dec.pad_id)
        x = dec.dropout(dec.tgt_word_emb(ys) * dec.scale + dec.positional_encoding(ys))
        for layer in dec.layer_stack:
            x = layer.forward(x, enc, tgt_mask, mask, cache=None)
        return torch.softmax(dec.tgt_word_prj(dec.layer_norm_out(x)[:, -1]).float(), -1)[0]

    with torch.no_grad():
        enc, _, mask = _MODEL.model.encoder(feats, lengths)
        sos = torch.full((1, 1), dec.sos_id, dtype=torch.long, device=enc.device)
        p_lang = step(sos, enc, mask)
        zh = torch.full((1, 1), tok.index("zh"), dtype=torch.long, device=enc.device)
        p_dia = step(torch.cat([sos, zh], 1), enc, mask)
    return p_lang.cpu().numpy(), p_dia.cpu().numpy()


def _detect_with_rescue(audio_int16: np.ndarray, sample_rate: int) -> Tuple[str, float]:
    asr_cfg = CONFIG.get("asr", {})
    rescue_from = set(asr_cfg.get("firered_lid_rescue_from", _DEFAULT_RESCUE_FROM))
    min_prob = float(asr_cfg.get("firered_lid_rescue_min_prob", 0.8))
    tok = _token_table()

    p_lang, p_dia = _posteriors(audio_int16, sample_rate)
    if p_lang is None:
        return "auto", 0.0
    top = int(p_lang.argmax())
    lang, lang_p = tok[top], float(p_lang[top])
    dia_idx = [tok.index(d) for d in _DIALECTS]
    best = max(dia_idx, key=lambda i: p_dia[i])
    dia, dia_p = tok[best], float(p_dia[best])

    if lang == "zh":
        label, conf = f"zh {dia}", lang_p * dia_p
    elif lang in rescue_from and dia_p >= min_prob:
        logger.debug(f"FireRedLID 救回：1-best={lang}({lang_p:.2f}) → zh {dia}({dia_p:.2f})")
        label, conf = f"zh {dia}", dia_p
    else:
        label, conf = lang, lang_p

    routing = _label_to_routing(label)
    if routing is None:
        return "auto", conf
    return routing, conf


# ---------------------------------------------------------------------------
# 整段先驗（asr.lid_call_prior，預設關）用的逐段明細
#
# 逐段判定在低取樣率語音的短片段上很不穩（幾秒鐘的閩南語常被判成緬甸語、越南語），
# 而且單看一段無從分辨「被誤判的方言」與「真的外語」。但以整段錄音來看兩者差很多：
# 真外語的錄音幾乎沒有片段會被判成漢語，漢語方言的錄音則總有一部分片段判得對。
# pipeline 會先收集每一段的分佈，再以整段錄音為單位決定（見 pipeline._apply_call_prior）。
# ---------------------------------------------------------------------------
def detect_detail(audio: np.ndarray, sample_rate: int = 16000) -> Optional[dict]:
    """回傳 {lang, lang_p, p_zh, dia(方言機率向量，順序同 DIALECTS)}；抽不到特徵時回 None。"""
    _load()
    if audio.dtype != np.int16:
        audio = (np.clip(audio, -1.0, 1.0) * 32767.0).astype(np.int16)
    tok = _token_table()
    p_lang, p_dia = _posteriors(audio, sample_rate)
    if p_lang is None:
        return None
    top = int(p_lang.argmax())
    dia = np.array([float(p_dia[tok.index(d)]) for d in _DIALECTS], dtype=np.float64)
    return {"lang": tok[top], "lang_p": float(p_lang[top]),
            "p_zh": float(p_lang[tok.index("zh")]), "dia": dia}


DIALECTS = _DIALECTS


def routing_of(label: str) -> Optional[str]:
    """FireRedLID 標籤（如 "zh min"、"vi"）→ routing key；不認識回 None。"""
    return _label_to_routing(label)
