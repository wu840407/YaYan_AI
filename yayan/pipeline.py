"""YaYan-AI 端到端流程（含分批翻譯）。"""
from __future__ import annotations

import logging
import re
import time
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import librosa
import numpy as np

from . import __version__
from . import observability
from .config import CONFIG
from .llm import LlmClient, to_taiwan_traditional

observability.setup_logging()   # YaYan.* 的 log 不依賴 root（第三方套件會把 root 調成 ERROR）
logger = logging.getLogger("YaYan.Pipeline")

SPEAKER_LABELS = ["A", "B", "C", "D", "E"]
TRANSLATE_BATCH_LINES = 30   # ★ 每批翻譯 30 行（防 LLM 截斷）
# enroll_unknown 自動建檔的語者名前綴；這類項目只是佔位，不可當識別結果顯示
UNNAMED_SPEAKER_PREFIX = "未知語者_"


@dataclass
class Segment:
    start: float
    end: float
    speaker: str = "A"
    raw_text: str = ""
    asr_alias: str = ""
    routing: str = ""
    lid_conf: float = 0.0
    lid_method: str = "default"


@dataclass
class TranscriptionResult:
    audio_path: str
    detected_language: str
    routing: str
    segments: List[Segment] = field(default_factory=list)
    raw_text: str = ""
    translated_text: str = ""
    language_breakdown: Dict[str, int] = field(default_factory=dict)
    quality_hint: str = ""   # 逐字稿明顯不成句時的建議文字；空字串＝沒有可提示的問題
    # V5.3：本次轉錄偵測到的語者與其聲紋向量，供 UI 直接建檔（不進 to_dict，含 ndarray）。
    # 只有 speaker_id.enable_speaker_id 開啟時才有內容。
    speaker_candidates: List[dict] = field(default_factory=list)
    # 辨識失敗、輸出過少或重複輸出被截斷時給使用者看的提醒；空字串＝沒有問題。
    # 只有 asr.enable_failure_notice 開啟時才有內容。
    failure_notice: str = ""
    # 本次轉錄的統計數字（段數、失敗段數、各階段耗時…），不含任何文字內容。
    stats: Dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "audio_path": self.audio_path,
            "detected_language": self.detected_language,
            "routing": self.routing,
            "language_breakdown": self.language_breakdown,
            "raw_text": self.raw_text,
            "translated_text": self.translated_text,
            "quality_hint": self.quality_hint,
            "failure_notice": self.failure_notice,
            "stats": self.stats,
            "segments": [
                {
                    "start": s.start, "end": s.end, "speaker": s.speaker,
                    "raw_text": s.raw_text, "asr_alias": s.asr_alias,
                    "routing": s.routing, "lid_conf": s.lid_conf,
                    "lid_method": s.lid_method,
                }
                for s in self.segments
            ],
        }


_LLM: Optional[LlmClient] = None


def _get_llm() -> LlmClient:
    global _LLM
    if _LLM is None:
        _LLM = LlmClient(alias=CONFIG["llm"]["alias"])
    return _LLM


def warmup() -> None:
    _get_llm()
    logger.info(f"YaYan-AI v{__version__} warmup 完成。")


def _audio_channels(path: str) -> int:
    """原始音檔的聲道數（只供統計；讀不到回 0）。"""
    try:
        import soundfile as sf
        return int(sf.info(path).channels)
    except Exception:
        return 0


def _load_audio(path: str) -> np.ndarray:
    sr = CONFIG["audio"]["sample_rate"]
    y, _ = librosa.load(path, sr=sr, mono=True)
    peak = float(np.max(np.abs(y))) if y.size else 0.0
    if peak > 0:
        y = y / peak
    return y.astype(np.float32)


def _normalize_speakers(
    raw_segments: List[Tuple[float, float, str]],
) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    next_idx = 0
    for _, _, label in raw_segments:
        if label not in mapping:
            if next_idx < len(SPEAKER_LABELS):
                mapping[label] = SPEAKER_LABELS[next_idx]
                next_idx += 1
            else:
                mapping[label] = SPEAKER_LABELS[-1]
    return mapping


def _label_speaker_at(
    t: float,
    raw_segments: List[Tuple[float, float, str]],
    mapping: Dict[str, str],
) -> str:
    if not raw_segments:
        return "A"
    for s, e, label in raw_segments:
        if s <= t <= e:
            return mapping.get(label, "A")
    return "A"


def _fmt_speaker_prefix(speaker: str, t1: str, t2: str) -> str:
    """A~E 匿名標籤加「方」（[A方 ...]）；已識別姓名/疑似不加（[張三 ...]）。"""
    if speaker in SPEAKER_LABELS:
        return f"[{speaker}方 {t1}-{t2}]"
    return f"[{speaker} {t1}-{t2}]"


def _identify_speakers(
    audio: np.ndarray,
    sample_rate: int,
    raw_speakers: List[Tuple[float, float, str]],
    fallback_mapping: Dict[str, str],
) -> Tuple[Dict[str, str], List[dict]]:
    """V5.0 M2：對每個 diarization 原始 speaker 抽聲紋比對。

    回傳 (mapping, candidates)：
    - mapping：raw_label → 顯示名稱
    - candidates：V5.3 新增，每位語者的比對結果與**聲紋向量**，供 UI 直接建檔。
      向量留在記憶體不落 DB，使用者按下建檔才寫入。

    - similarity ≥ threshold_high → 姓名（高信心）
    - threshold_mid ≤ similarity < high → 「疑似_姓名(分數)」（中信心）
    - 以下或無候選 → 退回 A/B/C（fallback_mapping），可選自動建「未知語者_時間戳」待命名

    任一步失敗都退回該 label 的 A/B/C，絕不讓識別拖垮轉錄。
    """
    from . import speaker_db, voiceprint

    cfg = CONFIG["speaker_id"]
    th_high = float(cfg.get("threshold_high", 0.70))
    th_mid = float(cfg.get("threshold_mid", 0.55))
    top_k = int(cfg.get("top_k", 5))
    ef = int(cfg.get("hnsw_ef_search", 100))
    enroll_unknown = bool(cfg.get("enroll_unknown", True))

    # 聚合每個 raw_label 的所有區段
    segs_by_label: Dict[str, List[Tuple[float, float]]] = {}
    for s, e, label in raw_speakers:
        segs_by_label.setdefault(label, []).append((s, e))

    mapping: Dict[str, str] = {}
    candidates: List[dict] = []

    def _record(label, fallback, segs, vec, status, matched="", sim=0.0):
        """把該語者的比對結果與聲紋收進 candidates，供 UI 建檔用。"""
        candidates.append({
            "raw_label": label,
            "display": mapping.get(label, fallback),
            "status": status,           # identified / suspected / unknown
            "matched": matched,
            "similarity": round(float(sim), 3),
            "speak_sec": round(sum(e - s for s, e in segs), 1),
            "vec": vec,                 # ndarray；不進 to_dict
        })

    for label, segs in segs_by_label.items():
        fallback = fallback_mapping.get(label, "A")
        try:
            vec = voiceprint.extract_from_segments(audio, segs, sample_rate=sample_rate)
        except Exception as e:
            logger.debug(f"語者 {label} 抽聲紋失敗: {e}")
            vec = None
        if vec is None:
            mapping[label] = fallback
            continue

        try:
            hits = speaker_db.search(vec, top_k=top_k, ef_search=ef)
        except Exception as e:
            logger.warning(f"聲紋搜尋失敗，退回 {fallback}: {e}")
            mapping[label] = fallback
            # 搜尋失敗但聲紋抽到了，仍可建檔
            _record(label, fallback, segs, vec, "unknown")
            continue

        best = hits[0] if hits else None
        # enroll_unknown 自動建的「未知語者_時間戳」只是待命名的佔位項，不是識別結果。
        # 讓它參與比對是對的（同一人跨檔案能連起來），但**不可以拿它的名字當顯示名**——
        # 否則逐字稿會出現 [未知語者_20260722_002834_SPEAKER_00 00:05-00:06] 這種標籤。
        if best and str(best.get("name", "")).startswith(UNNAMED_SPEAKER_PREFIX):
            mapping[label] = fallback
            logger.info(
                f"語者 {label} → 命中未命名語者 {best['name']}"
                f"（sim={best['similarity']:.2f}），顯示仍用 {fallback}"
            )
            _record(label, fallback, segs, vec, "unknown", sim=best["similarity"])
            continue

        if best and best["similarity"] >= th_high:
            mapping[label] = best["name"]
            logger.info(
                f"語者 {label} → {best['name']}（sim={best['similarity']:.2f} 高信心）"
            )
            _record(label, fallback, segs, vec, "identified",
                    matched=best["name"], sim=best["similarity"])
        elif best and best["similarity"] >= th_mid:
            mapping[label] = f"疑似_{best['name']}({best['similarity']:.2f})"
            logger.info(
                f"語者 {label} → 疑似_{best['name']}（sim={best['similarity']:.2f} 中信心）"
            )
            _record(label, fallback, segs, vec, "suspected",
                    matched=best["name"], sim=best["similarity"])
        else:
            mapping[label] = fallback
            sim_txt = f"{best['similarity']:.2f}" if best else "無候選"
            logger.info(f"語者 {label} → 辨識不出（best={sim_txt}），退回 {fallback}")
            _record(label, fallback, segs, vec, "unknown",
                    sim=best["similarity"] if best else 0.0)
            if enroll_unknown:
                try:
                    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                    speaker_db.add_speaker(
                        f"{UNNAMED_SPEAKER_PREFIX}{ts}_{label}", vec,
                        note="自動建檔待命名", source="auto",
                    )
                except Exception as e:
                    logger.debug(f"未知語者自動建檔失敗: {e}")

    # display 要等 mapping 填完才正確（_record 呼叫當下可能還沒寫入）
    for c in candidates:
        c["display"] = mapping.get(c["raw_label"], c["display"])
    return mapping, candidates


def _fmt_time(sec: float) -> str:
    sec = int(round(sec))
    h, remain = divmod(sec, 3600)
    m, s = divmod(remain, 60)
    if h > 0:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def _slice_with_context(
    audio: np.ndarray,
    start: float,
    end: float,
    sample_rate: int,
    context_sec: float,
) -> np.ndarray:
    """v4.7-A：取 [start, end] 並向前後各借 context_sec 秒的音訊。

    只用於 LID 語言判斷：短段借鄰近 context 後，VoxLingua107 的信心更穩，
    減少混合語音逐段判錯。**不影響 ASR 切段**（ASR 仍用原本的 VAD chunk）。
    邊界自動夾在 [0, 全長] 內。
    """
    total_sec = len(audio) / sample_rate
    ctx_start = max(0.0, start - context_sec)
    ctx_end = min(total_sec, end + context_sec)
    i0 = int(round(ctx_start * sample_rate))
    i1 = int(round(ctx_end * sample_rate))
    return audio[i0:i1]


def _dynamic_lid_threshold(seg_dur_sec: float) -> float:
    if seg_dur_sec < 2.0:
        return 0.40
    elif seg_dur_sec < 5.0:
        return 0.50
    else:
        return 0.60


def _vote_routing(
    candidates: List[Tuple[str, float]],
    fallback: str,
) -> str:
    valid = [(r, c) for r, c in candidates if c > 0.3 and r != "auto"]
    if not valid:
        return fallback
    weighted: Counter = Counter()
    for r, c in valid:
        weighted[r] += c
    return weighted.most_common(1)[0][0]


# v4.7-B：被視為「LID 確實判出語言」的 method（作為 speaker_inherit 的 anchor）
_LID_ANCHOR_METHODS = (
    "lid", "ensemble_agree", "ensemble_vox", "ensemble_whisper",
)


def _ensemble_lid(
    r1: str, c1: float, r2: str, c2: float,
) -> Tuple[str, float, str]:
    """v4.7-B：VoxLingua107(r1,c1) 與 Whisper(r2,c2) 兩個 LID 投票。

    - 一致：取兩者較高的信心並再加成（封頂 1.0），method=ensemble_agree。
    - 不一致：取信心較高的那一個 routing，但信心打折（兩模型分歧 → 降低可信度），
      method 標明採用了哪個來源（ensemble_vox / ensemble_whisper）。
    """
    AGREE_BOOST = 1.15
    DISAGREE_PENALTY = 0.7
    if r1 == r2:
        return r1, min(1.0, max(c1, c2) * AGREE_BOOST), "ensemble_agree"
    if c1 >= c2:
        return r1, c1 * DISAGREE_PENALTY, "ensemble_vox"
    return r2, c2 * DISAGREE_PENALTY, "ensemble_whisper"


def _batched_translate(
    text_lines: List[str],
    source_language: str,
    batch_size: int = TRANSLATE_BATCH_LINES,
) -> str:
    """把長文字分批送 LLM，避免單次 max_new_tokens 截斷。"""
    if not text_lines:
        return ""

    llm = _get_llm()
    translated_chunks: List[str] = []

    n_batches = (len(text_lines) + batch_size - 1) // batch_size
    logger.info(f"LLM 分批翻譯：{len(text_lines)} 行 → {n_batches} 批，每批 {batch_size} 行")

    for batch_idx in range(n_batches):
        start = batch_idx * batch_size
        end = min(start + batch_size, len(text_lines))
        chunk_lines = text_lines[start:end]
        chunk_text = "\n".join(chunk_lines)

        # V5.0 M3：RAG 術語庫（開關預設關）。只注入「本批文字實際命中」的術語，
        # 空命中時 glossary_block="" → prompt 與既有逐字相同。檢索失敗不影響翻譯。
        glossary_block = ""
        if CONFIG.get("rag", {}).get("enable_rag", False):
            try:
                from . import glossary
                glossary_block = glossary.glossary_for_text(chunk_text, source_language)
                if glossary_block:
                    logger.info(f"  批 {batch_idx + 1}: 注入術語區塊（{glossary_block.count(chr(10)) - 1} 條）")
            except Exception as e:
                logger.warning(f"  批 {batch_idx + 1}: 術語檢索失敗，跳過注入：{e}")

        try:
            result = llm.translate(
                chunk_text, source_language=source_language, glossary_block=glossary_block
            )
            translated_chunks.append(result.strip())
            logger.info(
                f"  批 {batch_idx + 1}/{n_batches}: "
                f"{len(chunk_text)} 字 → {len(result)} 字"
            )
        except Exception as e:
            logger.exception(f"批 {batch_idx + 1} 翻譯失敗，回退原文: {e}")
            translated_chunks.append(chunk_text)

    return "\n".join(translated_chunks)


_HINT_ZH_ROUTINGS = {
    "zh", "cmn", "auto", "wuu", "hsn", "gan", "cjy", "hak", "cdo",
    "cmn-sw", "cmn-sd", "cmn-ne", "cmn-zy", "cmn-wh", "cmn-xa", "cmn-lz", "cmn-jh",
    "wuu-sz", "wuu-nb", "wuu-wz",
}


def _quality_hint(raw_text: str, user_hint: str, lang_counter) -> str:
    """逐字稿可讀性守門：明顯不成句時回傳一句建議，**不自動切換引擎**。

    存在理由：VoxLingua107 分不出漢語方言，自動模式一律走通用 Dolphin；
    台語、福州話等在 Dolphin 下會崩成不成詞的字串（「控把繃」「經典博案」），
    但使用者不見得看得出那是引擎選錯而不是原音就這樣。

    刻意只提示不切換：可讀性判斷會把「本來就講得破碎」的錄音誤判成辨識失敗，
    自動換引擎會在使用者不知情下改變結果；提示則讓人保有判斷權。

    任何失敗都回空字串——這是加分提示，不能讓它擋掉轉錄。
    """
    if not CONFIG["asr"].get("enable_quality_hint", False):
        return ""
    # 使用者已明確指定語言 → 他已經做過選擇，再建議「改選語言」只是雜訊
    if (user_hint or "auto") != "auto":
        return ""
    # 只對走漢語方言引擎的結果提示；日韓歐美走 Whisper，不適用
    dominant = lang_counter.most_common(1)[0][0] if lang_counter else "auto"
    if dominant not in _HINT_ZH_ROUTINGS:
        return ""

    # 去掉時間戳前綴再取樣，避免 [A方 00:01-00:05] 干擾判斷
    body = re.sub(r"\[[^\]]*\]", "", raw_text or "").strip()
    body = re.sub(r"\s+", "", body)
    if len(body) < 40:
        return ""          # 太短，判不準
    sample = body[:400]

    try:
        prompt = (
            "下面是一段語音辨識的逐字稿。請判斷它是否為可理解的自然語句。"
            "如果出現明顯不成詞、語義斷裂的片段，視為辨識失敗。"
            "只回答『通順』或『不通順』，不要解釋。\n\n逐字稿：" + sample
        )
        verdict = _get_llm().chat(
            [{"role": "user", "content": prompt}], max_new_tokens=20
        )
    except Exception as e:
        logger.warning(f"可讀性判斷失敗，略過提示：{e}")
        return ""

    if "不通順" not in (verdict or ""):
        return ""
    if str(CONFIG["asr"].get("lid_backend", "voxlingua")).lower() == "firered":
        # 方言 LID 已能分出閩/粵/吳/湘/西南官話，但短片段仍可能判錯
        return (
            "⚠️ 逐字稿可讀性偏低，可能是自動偵測判錯了語言或方言。"
            "若您知道這段錄音的語言，請在「來源語言」直接指定後重新辨識。"
        )
    return (
        "⚠️ 逐字稿可讀性偏低，可能是語言選擇不符。"
        "若這段錄音是閩南語／台語、粵語或其他特定方言，"
        "請在「來源語言」改選對應項目後重新辨識——自動偵測分不出漢語方言。"
    )


def _apply_asr_corrections(text: str, routing: str) -> str:
    """ASR 後處理：套用術語庫的錯字對照（開關 rag.enable_asr_correction，預設關）。

    在 OpenCC 轉正體之後才做，字集才與術語庫一致；在加時間戳前綴之前做，
    替換規則不會誤傷 [A方 00:01-00:05] 這類標記。
    任何失敗都回原文——術語庫掛掉不該讓轉錄跟著掛。
    """
    if not CONFIG.get("rag", {}).get("enable_asr_correction", False):
        return text
    try:
        from .glossary import apply_corrections
        fixed, n = apply_corrections(text, source_lang=routing or "any")
        if n:
            logger.info(f"ASR 後處理術語替換 {n} 處")
        return fixed
    except Exception as e:
        logger.warning(f"ASR 術語後處理失敗，保留原文：{e}")
        return text


_WARMED_UP = False

# 整段先驗不會動到的外語：系統有對應引擎、而且實務上真的會出現的語言。
# 其餘被判出的語言（多半是短片段被誤判成的低資源語言）在整段錄音明顯是漢語時會被改回漢語。
_CALL_PRIOR_KEEP_DEFAULT = ("en", "ja", "ko", "fr", "de", "ru", "es", "ar", "fa", "ur", "hi", "bo", "ug")


def _apply_call_prior(chunks, seg_routings, details, lid_mod, asr_cfg) -> None:
    """以整段錄音為單位修正逐段語種（開關 asr.lid_call_prior，預設關）。就地修改 seg_routings。

    兩件事：
    1. 這段錄音已有足夠多的片段被判成漢語（秒數、時長占比、段數三個門檻都過）時，其餘被判成
       「非常見外語」的片段一律視為漢語——那些多半是短片段的誤判。真外語的錄音
       幾乎沒有片段會被判成漢語，所以不會被牽連。
    2. 漢語片段的方言，由「該段自己的方言分佈」與「整段錄音的方言分佈」合起來決定，
       避免同一段錄音裡零星幾段被判成別的方言而走錯引擎。
       逐段信心不足（原本要靠鄰段投票）但分佈上是漢語的片段，也在這裡直接決定方言。
    """

    min_share = float(asr_cfg.get("lid_call_prior_min_share", 0.15))
    min_sec = float(asr_cfg.get("lid_call_prior_min_sec", 8.0))
    min_segments = int(asr_cfg.get("lid_call_prior_min_segments", 2))
    beta = float(asr_cfg.get("lid_call_prior_beta", 1.0))
    # keep 清單以 routing 比對：LID 會把義大利語併到法語、哈薩克語併到維吾爾語等，
    # 用 routing 才能一併保護這些就近對應的語言。
    keep = set(asr_cfg.get("lid_call_prior_keep", _CALL_PRIOR_KEEP_DEFAULT))

    def _kept(lang: str) -> bool:
        return lang in keep or (lid_mod.routing_of(lang) or "") in keep

    idx = [i for i, d in enumerate(details) if d is not None]
    if len(idx) < 2:
        return
    dur = {i: chunks[i][1] - chunks[i][0] for i in idx}
    total = sum(dur.values())
    zh_idx = [i for i in idx if details[i]["lang"] == "zh"]
    zh_sec = sum(dur[i] for i in zh_idx)
    if total <= 0 or zh_sec <= 0:
        return
    # 三個條件都要過：短錄音裡單獨一個片段被誤判成漢語，不足以把整段外語翻成漢語
    force = (zh_sec >= min_sec and zh_sec / total >= min_share and len(zh_idx) >= min_segments)

    call_dia = sum(details[i]["dia"] * details[i]["p_zh"] * dur[i] for i in idx)
    if float(call_dia.sum()) <= 0:
        return
    call_dia = call_dia / call_dia.sum()

    eps = 1e-6
    n_lang = n_dia = 0
    for i in idx:
        d = details[i]
        is_zh = d["lang"] == "zh"
        if not is_zh and not (force and not _kept(d["lang"])):
            continue
        k = int(np.argmax(np.log(d["dia"] + eps) + beta * np.log(call_dia + eps)))
        rt = lid_mod.routing_of(f"zh {lid_mod.DIALECTS[k]}")
        if not rt:
            continue
        old_rt = seg_routings[i][0]
        if rt != old_rt:
            n_lang += (not is_zh)
            n_dia += is_zh
            # 信心取整段方言分佈的機率：這個決定是整段錄音的證據撐起來的，不是單段
            seg_routings[i] = (rt, float(call_dia[k]), "call_prior")
    if n_lang or n_dia:
        logger.info(
            f"整段先驗：漢語片段 {zh_sec:.0f}/{total:.0f} 秒"
            f"（{'已' if force else '未'}達門檻），改判外語→漢語 {n_lang} 段、方言調整 {n_dia} 段"
        )


def _failure_notice(stats: dict) -> str:
    """把「逐字稿可能不完整」的狀況講給使用者聽（開關 asr.enable_failure_notice，預設關）。

    存在理由：引擎推論失敗（例如 GPU 記憶體不足）時，該段只會默默消失，
    使用者拿到的是看似正常、其實缺段甚至空白的逐字稿。
    """
    if not CONFIG["asr"].get("enable_failure_notice", False):
        return ""
    notes = []
    if stats.get("n_failed"):
        notes.append(
            f"⚠️ 有 {stats['n_failed']} 段（約 {stats['failed_sec']:.0f} 秒語音）辨識失敗，"
            "逐字稿不完整；請稍後重新轉錄，仍失敗請洽系統管理者。"
        )
    elif stats.get("no_speech"):
        # 靜音或雜訊也可能讓辨識模型吐出幾個字，所以有文字時要講明那些字不可靠
        notes.append(
            "ℹ️ 這個音檔沒有偵測到明顯的語音（可能是靜音、雜訊或無人接聽）"
            + ("，以下文字可能不可靠。" if stats.get("chars_raw") else "。")
        )
    elif stats.get("speech_sec", 0) >= 5 and stats.get("chars_raw", 0) < 0.5 * stats["speech_sec"]:
        notes.append(
            f"⚠️ 偵測到約 {stats['speech_sec']:.0f} 秒語音，但幾乎沒有辨識出文字；"
            "可能是來源語言選錯或辨識失敗，請確認後重新轉錄。"
        )
    if stats.get("n_repetition"):
        notes.append(f"⚠️ 有 {stats['n_repetition']} 段出現重複輸出，已自動截斷，該段內容可能不完整。")
    if stats.get("translate_failed"):
        notes.append("⚠️ 譯文產生失敗，目前顯示的是辨識原文；請稍後按重新翻譯。")
    return "\n".join(notes)


def transcribe_audio(
    audio_path: str,
    language: str = "auto",
    use_vad: Optional[bool] = None,
    use_diarize: Optional[bool] = None,
) -> TranscriptionResult:
    global _WARMED_UP
    t_start = time.time()
    # 行程內第一次轉錄：耗時含模型載入，不可拿來比速度。
    # （之後第一次用到別的引擎時同樣會含載入時間，這個旗標涵蓋不到。）
    cold_start = not _WARMED_UP
    _WARMED_UP = True
    audio_cfg = CONFIG["audio"]
    asr_cfg = CONFIG["asr"]
    use_vad = asr_cfg["enable_vad"] if use_vad is None else use_vad
    use_diarize = CONFIG["diarize"]["enabled"] if use_diarize is None else use_diarize

    audio = _load_audio(audio_path)
    sr = audio_cfg["sample_rate"]

    if audio.size == 0:
        logger.warning(f"音檔為空: {audio_path}")
        return TranscriptionResult(
            audio_path=audio_path,
            detected_language=language, routing=language,
        )

    user_hint = language
    no_speech = False

    if use_vad:
        from . import vad
        try:
            chunks = vad.split_speech(audio, sample_rate=sr)
            no_speech = bool(getattr(vad, "LAST_NO_SPEECH", False))
        except Exception as e:
            logger.warning(f"VAD 失敗，整段直送: {e}")
            chunks = [(0.0, len(audio) / sr, audio)]
    else:
        chunks = [(0.0, len(audio) / sr, audio)]

    raw_speakers: List[Tuple[float, float, str]] = []
    speaker_mapping: Dict[str, str] = {}
    if use_diarize:
        from . import diarize
        try:
            raw_speakers = diarize.diarize(audio, sample_rate=sr)
            speaker_mapping = _normalize_speakers(raw_speakers)
            logger.info(
                f"Diarize 偵測 {len(speaker_mapping)} 位 → "
                f"{', '.join(sorted(set(speaker_mapping.values())))}"
            )
        except Exception as e:
            logger.warning(f"Diarization 失敗，略過: {e}")

    # V5.0 M2：聲紋語者識別（diarization 之後加掛；enable_speaker_id 預設 false
    #          → 不進這個分支，speaker_mapping 維持 A/B/C，行為與 v4.7 完全相同）
    speaker_candidates: List[dict] = []
    if use_diarize and raw_speakers and CONFIG["speaker_id"]["enable_speaker_id"]:
        try:
            speaker_mapping, speaker_candidates = _identify_speakers(
                audio, sr, raw_speakers, speaker_mapping
            )
            logger.info(
                f"聲紋識別 → {', '.join(sorted(set(speaker_mapping.values())))}"
            )
        except Exception as e:
            logger.warning(f"聲紋識別失敗，退回 A/B/C: {e}")

    # LID backend：voxlingua（預設，維持既有行為）或 firered（能分漢語方言）
    _lid_backend = str(asr_cfg.get("lid_backend", "voxlingua")).lower()
    if _lid_backend == "firered":
        from . import lid_firered as _lid_mod
    else:
        from . import lid as _lid_mod
    enable_lid = asr_cfg["enable_lid"]
    seg_routings: List[Tuple[str, float, str]] = []

    if enable_lid and user_hint == "auto":
        # v4.7-A：逐段 LID 借前後 context（可由 config 關閉）
        use_lid_context = asr_cfg.get("enable_lid_context", False)
        lid_context_sec = float(asr_cfg.get("lid_context_sec", 1.5))
        # v4.7-B：LID Ensemble（VoxLingua107 + Whisper 投票，可由 config 關閉）
        use_ensemble = asr_cfg.get("enable_lid_ensemble", False)
        if use_ensemble and _lid_backend == "firered":
            # Whisper LID 對漢語只回 zh，會把 firered 判出的方言票投回通用中文，
            # 等於抵消本 backend 的唯一價值。
            logger.info("lid_backend=firered，強制關閉 LID ensemble")
            use_ensemble = False
        # 整段先驗（預設關）：只有方言 LID backend 提供逐段分佈時才能用。
        # 逐段判定本身不變，先驗只在其後做修正。
        use_call_prior = bool(asr_cfg.get("lid_call_prior", False)) and _lid_backend == "firered"
        lid_details: List[Optional[dict]] = [None] * len(chunks)
        _lid_whisper = None
        if use_ensemble:
            from . import lid_whisper as _lid_whisper
        logger.info(
            f"逐段 LID（{len(chunks)} 段）... "
            f"context={'±%.1fs' % lid_context_sec if use_lid_context else 'off'}, "
            f"ensemble={'on' if use_ensemble else 'off'}"
        )
        for i, (start, end, chunk) in enumerate(chunks):
            seg_dur = end - start
            if seg_dur < 0.5:
                seg_routings.append(("auto", 0.0, "too_short"))
                continue
            try:
                if use_lid_context:
                    lid_audio = _slice_with_context(
                        audio, start, end, sr, lid_context_sec
                    )
                else:
                    lid_audio = chunk
                rt, conf = _lid_mod.detect(lid_audio, sample_rate=sr)
                if use_call_prior:
                    # 逐段判定仍用原本的 detect()（結果與關閉時完全相同），
                    # 另外取完整分佈只供整段先驗使用；取不到就當這一段沒有明細。
                    try:
                        lid_details[i] = _lid_mod.detect_detail(lid_audio, sample_rate=sr)
                    except Exception as e:
                        logger.warning(f"段 {i} 取 LID 分佈失敗，該段不參與整段先驗：{e}")
                sub_method = "lid"
                if use_ensemble:
                    try:
                        w_rt, w_conf = _lid_whisper.detect(lid_audio, sample_rate=sr)
                        rt, conf, sub_method = _ensemble_lid(rt, conf, w_rt, w_conf)
                    except Exception as e:
                        logger.debug(f"段 {i} Whisper-LID 失敗，退回 VoxLingua: {e}")
                threshold = _dynamic_lid_threshold(seg_dur)
                if conf >= threshold:
                    seg_routings.append((rt, conf, sub_method))
                else:
                    seg_routings.append(("auto", conf, "low_conf"))
            except Exception as e:
                logger.debug(f"段 {i} LID 失敗: {e}")
                seg_routings.append(("auto", 0.0, "lid_error"))
        if use_call_prior:
            try:
                _apply_call_prior(chunks, seg_routings, lid_details, _lid_mod, asr_cfg)
            except Exception as e:      # 先驗只是修正，失敗時保留逐段結果
                logger.warning(f"整段先驗失敗，沿用逐段判定：{e}")
    else:
        # 使用者明確指定語言（非 auto）時，指定值優先於逐段 LID：
        # VoxLingua107 對漢語只有一個 "Chinese" 標籤，會高信心地把台語/粵語等
        # 一律判成 zh，蓋掉使用者的選擇，專用 ASR（如台語 Breeze）就永遠走不到。
        seg_routings = [(user_hint, 1.0, "user_hint")] * len(chunks)

    fallback_routing = user_hint if user_hint != "auto" else "zh"
    for i, (rt, conf, method) in enumerate(seg_routings):
        if rt != "auto":
            continue
        window = []
        for j in range(max(0, i - 3), min(len(seg_routings), i + 4)):
            if j == i:
                continue
            r, c, _ = seg_routings[j]
            window.append((r, c))
        voted = _vote_routing(window, fallback_routing)
        seg_routings[i] = (voted, 0.0,
                          "neighbor_vote" if voted != fallback_routing else "fallback")

    if use_diarize and raw_speakers:
        last_speaker = None
        last_routing = None
        for i, (start, end, _chunk) in enumerate(chunks):
            mid_t = (start + end) / 2
            speaker = _label_speaker_at(mid_t, raw_speakers, speaker_mapping)
            current_routing, conf, method = seg_routings[i]
            if speaker == last_speaker and method in ("low_conf", "lid_error", "fallback"):
                if last_routing:
                    seg_routings[i] = (last_routing, conf, "speaker_inherit")
            if method in _LID_ANCHOR_METHODS and conf >= 0.6:
                last_speaker = speaker
                last_routing = current_routing

    from .asr import transcribe as asr_transcribe

    segments: List[Segment] = []
    lang_counter: Counter = Counter()
    n_failed = n_empty = n_repetition = 0
    failed_sec = 0.0
    use_rep_guard = asr_cfg.get("enable_repetition_guard", False)
    t_asr = time.time()

    for i, (start, end, chunk) in enumerate(chunks):
        routing, conf, method = seg_routings[i]
        try:
            r = asr_transcribe(
                chunk, routing=routing, language_hint=routing,
                chunk_start_sec=start,
            )
        except Exception as e:
            logger.error(f"ASR 失敗 [{start:.1f}s-{end:.1f}s] routing={routing}: {e}")
            n_failed += 1
            failed_sec += end - start
            continue

        if getattr(r, "error", ""):
            # 引擎內部已吞掉例外並回空字串：仍要記成失敗，否則這段會無聲無息地消失
            n_failed += 1
            failed_sec += end - start
        elif not r.text:
            n_empty += 1
        elif use_rep_guard:
            cut_text, was_cut = observability.collapse_repetition(r.text)
            if was_cut:
                logger.warning(
                    f"重複輸出已截斷 [{start:.1f}s-{end:.1f}s] routing={routing}："
                    f"{len(r.text)} 字 → {len(cut_text)} 字"
                )
                r.text = cut_text
                n_repetition += 1

        speaker_label = _label_speaker_at(
            (start + end) / 2, raw_speakers, speaker_mapping
        ) if use_diarize else "A"

        if r.text:
            text_tw = to_taiwan_traditional(r.text)
            text_tw = _apply_asr_corrections(text_tw, routing)
            segments.append(Segment(
                start=start, end=end,
                speaker=speaker_label,
                raw_text=text_tw,
                asr_alias=r.asr_alias,
                routing=routing,
                lid_conf=conf,
                lid_method=method,
            ))
            lang_counter[routing] += 1

    t_asr = time.time() - t_asr
    if lang_counter:
        dominant_lang = lang_counter.most_common(1)[0][0]
    else:
        dominant_lang = user_hint if user_hint != "auto" else "zh"
    logger.info(f"語言分布: {dict(lang_counter)} | 主要: {dominant_lang}")
    if n_failed:
        logger.error(f"本次轉錄有 {n_failed}/{len(chunks)} 段辨識失敗（約 {failed_sec:.0f} 秒語音）")

    # 組合 raw_text（含時間戳）
    raw_text_lines: List[str] = []
    for s in segments:
        t1 = _fmt_time(s.start)
        t2 = _fmt_time(s.end)
        if use_diarize:
            prefix = _fmt_speaker_prefix(s.speaker, t1, t2)
        else:
            prefix = f"[{t1}-{t2}]"
        raw_text_lines.append(f"{prefix} {s.raw_text}")
    raw_text = "\n".join(raw_text_lines).strip()

    # ★ LLM 翻譯改用分批
    translated = ""
    translate_failed = False
    t_llm = time.time()
    if raw_text_lines:
        try:
            lang_summary = ", ".join(f"{k}({v}段)" for k, v in lang_counter.most_common(3))
            source_desc = f"{dominant_lang} (混合: {lang_summary})"
            translated = _batched_translate(
                raw_text_lines,
                source_language=source_desc,
                batch_size=TRANSLATE_BATCH_LINES,
            )
        except Exception as e:
            logger.exception(f"LLM 分批翻譯失敗，回退原文: {e}")
            translated = raw_text
            translate_failed = True
        translated = to_taiwan_traditional(translated)
    t_llm = time.time() - t_llm

    stats = {
        "audio_sec": round(len(audio) / sr, 1),
        # VAD 完全沒偵測到語音時會整段直送；那不算「有語音」
        "speech_sec": 0.0 if no_speech else round(sum(e - s for s, e, _ in chunks), 1),
        "no_speech": no_speech,
        "cold_start": cold_start,
        "user_hint": user_hint,
        "routing": dominant_lang,
        "breakdown": dict(lang_counter),
        "lid_methods": dict(Counter(m for _, _, m in seg_routings)),
        "n_chunks": len(chunks),
        "n_segments": len(segments),
        "n_failed": n_failed,
        "failed_sec": round(failed_sec, 1),
        "n_empty": n_empty,
        "n_repetition": n_repetition,
        "chars_raw": sum(len(s.raw_text) for s in segments),
        "chars_out": len(translated),
        "translate_failed": translate_failed,
        "diarize": bool(use_diarize),
        "n_speakers": len({s.speaker for s in segments}),
        "audio_channels": _audio_channels(audio_path),
        "t_asr": round(t_asr, 1),
        "t_llm": round(t_llm, 1),
        "t_total": round(time.time() - t_start, 1),
    }
    observability.log_usage("transcribe", **stats)

    return TranscriptionResult(
        audio_path=audio_path,
        detected_language=dominant_lang,
        routing=dominant_lang,
        segments=segments,
        raw_text=raw_text,
        translated_text=translated,
        language_breakdown=dict(lang_counter),
        quality_hint=_quality_hint(raw_text, user_hint, lang_counter),
        speaker_candidates=speaker_candidates,
        failure_notice=_failure_notice(stats),
        stats=stats,
    )


def refine_with_user_edit(
    raw_text: str, user_edit: str, source_language: str = "zh",
) -> str:
    if not user_edit or not user_edit.strip():
        return ""
    raw_text = (raw_text or user_edit).strip()
    user_edit = user_edit.strip()

    # refine 也要分批（如果輸入很長）
    lines = user_edit.split("\n")
    if len(lines) > TRANSLATE_BATCH_LINES * 2:
        logger.info(f"refine 輸入 {len(lines)} 行，分批處理")
        translated = _batched_translate(
            lines, source_language=source_language,
            batch_size=TRANSLATE_BATCH_LINES,
        )
        return to_taiwan_traditional(translated)

    try:
        llm = _get_llm()
        refined = llm.refine(
            raw_text=raw_text, user_edit=user_edit,
            source_language=source_language,
        )
    except Exception as e:
        logger.exception(f"LLM refine 失敗: {e}")
        return to_taiwan_traditional(user_edit)

    return to_taiwan_traditional(refined)
