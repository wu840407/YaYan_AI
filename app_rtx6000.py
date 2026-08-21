"""
YaYan-AI — Quadro RTX 6000 (Turing) x 2 主介面
"""
from __future__ import annotations

import os
import sys
import json
import logging
import statistics
from datetime import datetime
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
os.environ.setdefault("GRADIO_DO_NOT_TRACK", "1")

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)

import gradio as gr
import numpy as np
import librosa

from yayan import __version__
from yayan.config import CONFIG
from yayan.pipeline import (
    transcribe_audio,
    refine_with_user_edit,
    warmup,
    TranscriptionResult,
)
# V5.0 M2：聲紋語者識別（底層模組；UI「語者管理」分頁用）
from yayan import speaker_db, voiceprint
# V5.0 M5：多聲道分離（純獨立工具，不依賴任何模型）
from yayan import channel_split
# V5.0 M3：特殊字詞 RAG 術語庫（DB 存取模組；翻譯注入在 pipeline 內）
from yayan import glossary


# 22 中文方言全部展開
DIALECT_TO_ROUTING = {
    "🔍 自動偵測（建議：逐段 LID）": "auto",
    # ⚠️ 這兩項都走 auto。自動偵測分不出漢語方言，**台語請直接選下方的
    #    「閩南語 / 台語」**，選這裡不會啟用台語專用引擎。
    "🔀 多方言混雜（自動偵測，不指定方言）": "auto",
    # ── 北方官話 ──
    "🇨🇳 北京話 / 普通話": "zh",
    "🇨🇳 東北話": "cmn-ne",
    "🇨🇳 山東話": "cmn-sd",
    "🇨🇳 河南話": "cmn-zy",
    "🇨🇳 西安話": "cmn-xa",
    "🇨🇳 蘭州話": "cmn-lz",
    # ── 西南官話 ──
    "🇨🇳 四川話 (西南官話)": "cmn-sw",
    "🇨🇳 武漢話": "cmn-wh",
    # ── 江淮官話 ──
    "🇨🇳 南京話": "cmn-jh",
    # ── 吳語 ──
    "🇨🇳 上海話 (吳語)": "wuu",
    "🇨🇳 蘇州話": "wuu-sz",
    "🇨🇳 寧波話": "wuu-nb",
    "🇨🇳 溫州話": "wuu-wz",
    # ── 粵語 ──
    "🇭🇰 廣東話 (粵語)": "yue",
    # ── 閩語 ──
    "🇹🇼 閩南語 / 台語": "nan",
    "🇨🇳 潮汕話 / 潮州話": "nan-cs",
    "🇨🇳 海南話": "nan-hn",
    "🇨🇳 福州話 (閩東語)": "cdo",
    # ── 客贛湘晉 ──
    "🇨🇳 客家話": "hak",
    "🇨🇳 湖南話 (湘語)": "hsn",
    "🇨🇳 江西話 (贛語)": "gan",
    "🇨🇳 山西話 (晉語)": "cjy",
    # ── 中國少數民族 ──
    "🏔️ 藏語 (Tibetan)": "bo",
    "🌙 維吾爾語 (Uyghur)": "ug",
    # ── 東亞 ──
    "🇯🇵 日文 (Japanese)": "ja",
    "🇰🇷 韓文 (Korean)": "ko",
    # ── 中東 / 南亞 ──
    "🇮🇷 波斯語 (Farsi)": "fa",
    "🇵🇰 烏爾都語 (Urdu)": "ur",
    "🇸🇦 阿拉伯語 (Arabic)": "ar",
    "🇮🇳 印地語 (Hindi)": "hi",
    # ── 歐洲 ──
    "🇬🇧 英語 (English)": "en",
    "🇫🇷 法語 (French)": "fr",
    "🇩🇪 德語 (German)": "de",
    "🇷🇺 俄語 (Russian)": "ru",
    "🇪🇸 西班牙語 (Spanish)": "es",
    # ── 東南亞 ──
    "🇹🇭 泰語 (Thai)": "th",
    "🇲🇾 馬來語 (Malay)": "ms",
    "🇻🇳 越南語 (Vietnamese)": "vi",
    "🇮🇩 印尼語 (Indonesian)": "id",
}
DIALECT_CHOICES = list(DIALECT_TO_ROUTING.keys())

ROUTING_DISPLAY = {
    "zh": "普通話", "cmn": "普通話",
    "cmn-ne": "東北話", "cmn-sd": "山東話", "cmn-zy": "河南話",
    "cmn-xa": "西安話", "cmn-lz": "蘭州話",
    "cmn-sw": "四川話", "cmn-wh": "武漢話", "cmn-jh": "南京話",
    "yue": "粵語",
    "wuu": "上海話", "wuu-sz": "蘇州話", "wuu-nb": "寧波話", "wuu-wz": "溫州話",
    "nan": "閩南語/台語", "nan-cs": "潮汕話", "nan-hn": "海南話",
    "cdo": "福州話",
    "hak": "客家話", "hsn": "湘語", "gan": "贛語", "cjy": "晉語",
    "bo": "藏語", "ug": "維吾爾語",
    "ja": "日文", "ko": "韓文",
    "fa": "波斯語", "ur": "烏爾都語", "ar": "阿拉伯語", "hi": "印地語",
    "en": "英語", "fr": "法語", "de": "德語", "ru": "俄語", "es": "西班牙語",
    "th": "泰語", "ms": "馬來語", "vi": "越南語", "id": "印尼語",
    "auto": "未識別",
}


def _calc_confidence(result: TranscriptionResult) -> tuple[float, str]:
    segments = result.segments
    raw_text = result.raw_text or ""
    translated = result.translated_text or ""
    if not segments:
        return 0.0, "無段落"

    notes = []
    score = 100.0
    durations = [s.end - s.start for s in segments]
    total_dur = sum(durations)

    if total_dur > 0:
        seg_per_min = len(segments) / (total_dur / 60)
        if seg_per_min > 80:
            score -= 15; notes.append("VAD 切片過密")
        elif seg_per_min < 5:
            score -= 10; notes.append("VAD 切片過疏")

    avg_dur = statistics.mean(durations)
    if avg_dur < 0.8:
        score -= 15; notes.append("段落過短")
    elif avg_dur > 20:
        score -= 10; notes.append("段落過長")

    if len(durations) > 1:
        std = statistics.stdev(durations)
        cv = std / avg_dur if avg_dur > 0 else 0
        if cv > 1.5:
            score -= 10; notes.append("段長不穩")

    seg_texts = [s.raw_text.strip() for s in segments if s.raw_text.strip()]
    if seg_texts:
        unique_ratio = len(set(seg_texts)) / len(seg_texts)
        if unique_ratio < 0.6:
            score -= 20; notes.append("ASR 重複嚴重")
        elif unique_ratio < 0.8:
            score -= 10; notes.append("ASR 些許重複")

    if raw_text and translated:
        ratio = len(translated) / max(len(raw_text), 1)
        if ratio < 0.5:
            score -= 15; notes.append("譯文偏短")
        elif ratio > 2.0:
            score -= 10; notes.append("譯文偏長")
    elif raw_text and not translated:
        score -= 30; notes.append("LLM 翻譯失敗")

    # LID 只計「真的由模型判出來」的段落。user_hint 的 conf 固定是 1.0，
    # 那代表使用者手動指定、不是模型信心，計入會讓分數虛胖。
    model_lid = [
        s.lid_conf for s in segments
        if s.lid_conf > 0 and s.lid_method not in ("user_hint", "fallback", "default")
    ]
    if model_lid:
        avg_lid = statistics.mean(model_lid)
        # 連續扣分取代級距：0.9 以上幾乎不扣，0.4 附近扣滿 20
        penalty = max(0.0, min(20.0, (0.90 - avg_lid) * 40))
        if penalty >= 1.0:
            score -= penalty
            notes.append(f"LID 平均信心 {avg_lid:.2f}")
    elif any(s.lid_method == "user_hint" for s in segments):
        notes.append("語言為手動指定")

    err_count = sum(1 for s in segments if s.lid_method == "lid_error")
    if err_count:
        score -= 15
        notes.append(f"LID 異常 {err_count}/{len(segments)} 段")
    fb_count = sum(1 for s in segments if s.lid_method == "fallback")
    if segments and fb_count / len(segments) > 0.3:
        score -= 10
        notes.append("LID 判不出的段落偏多")

    # 逐字稿可讀性（enable_quality_hint 開啟時才有值）
    if getattr(result, "quality_hint", ""):
        score -= 20
        notes.append("逐字稿可讀性偏低，可能語言選錯")

    score = max(0.0, min(100.0, score))
    note_text = "、".join(notes) if notes else "品質良好"
    return round(score, 1), note_text


def _format_lang_breakdown(breakdown: dict) -> str:
    if not breakdown:
        return ""
    parts = []
    total = sum(breakdown.values())
    for code, count in sorted(breakdown.items(), key=lambda x: -x[1])[:5]:
        name = ROUTING_DISPLAY.get(code, code)
        pct = count / total * 100
        parts.append(f"{name} {count}({pct:.0f}%)")
    return "｜".join(parts)


def fn_transcribe(audio_path, dialect_label, enable_diarize):
    # 末位回傳值為「原始譯文」存入隱藏 State，供 M3 校正學習 diff 用（rag 關閉時不使用）
    if audio_path is None:
        return "請先上傳或錄製音檔。", "", "", "", "—", ""

    routing = DIALECT_TO_ROUTING.get(dialect_label, "auto")
    try:
        result: TranscriptionResult = transcribe_audio(
            audio_path,
            language=routing,
            use_diarize=enable_diarize,
        )
    except Exception as e:
        logging.exception("transcribe 失敗")
        return f"識別失敗：{e}", "", "", "", "—", ""

    breakdown_text = _format_lang_breakdown(result.language_breakdown)
    n_speakers = len(set(s.speaker for s in result.segments)) if result.segments else 0
    info = (
        f"📊 語言分布：{breakdown_text}\n"
        f"👥 說話人：{n_speakers} 位｜段數：{len(result.segments)}"
    )
    # 可讀性偏低時附上建議（只提示，不自動改引擎；開關 asr.enable_quality_hint）
    if getattr(result, "quality_hint", ""):
        info += f"\n{result.quality_hint}"
    confidence, note = _calc_confidence(result)
    score_text = f"{confidence:.1f} / 100\n{note}"
    return (
        info, result.raw_text, result.raw_text, result.translated_text,
        score_text, result.translated_text,
    )


def fn_refine(raw_text_original, edited_text, dialect_label):
    if not edited_text or not edited_text.strip():
        return gr.update()
    routing = DIALECT_TO_ROUTING.get(dialect_label, "auto")
    try:
        refined = refine_with_user_edit(
            raw_text=raw_text_original or edited_text,
            user_edit=edited_text,
            source_language=routing,
        )
    except Exception as e:
        logging.exception("refine 失敗")
        return f"重新潤飾失敗：{e}"
    return refined


def fn_save_as(translated_text, source_audio):
    if not translated_text or not translated_text.strip():
        return None
    out_dir = Path(CONFIG["paths"]["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    base = Path(source_audio).stem if source_audio else "manual"
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = out_dir / f"{base}_{ts}.txt"
    out.write_text(translated_text, encoding="utf-8")
    return str(out)


# ─────────────────────── V5.0 M2：語者管理分頁 callbacks ───────────────────────
# 這些函式只操作 speaker_db / voiceprint，完全不碰上面的轉錄 callback。

SPEAKER_PAGE_SIZE = 20
SPEAKER_LIST_HEADERS = ["ID", "姓名/代號", "樣本數", "備註", "建檔時間", "更新時間"]


def _load_sample_audio(path):
    """讀取單一語音樣本檔，回傳 (mono float32 audio, sample_rate)。"""
    sr = CONFIG["audio"]["sample_rate"]
    y, _ = librosa.load(path, sr=sr, mono=True)
    peak = float(np.max(np.abs(y))) if y.size else 0.0
    if peak > 0:
        y = y / peak
    return y.astype(np.float32), sr


def _fmt_dt(dt):
    return dt.strftime("%Y-%m-%d %H:%M") if dt else ""


def _render_speaker_list(page, keyword):
    """回傳 (dataframe 列資料, 頁碼資訊文字, 修正後頁碼)；DB 失敗不讓整個 app 掛掉。"""
    try:
        try:
            page = int(page)
        except (TypeError, ValueError):
            page = 1
        kw = (keyword or "").strip()
        rows, total = speaker_db.list_speakers(
            page=page, page_size=SPEAKER_PAGE_SIZE, keyword=kw
        )
        max_page = max(1, (total + SPEAKER_PAGE_SIZE - 1) // SPEAKER_PAGE_SIZE)
        if page > max_page:
            page = max_page
            rows, total = speaker_db.list_speakers(
                page=page, page_size=SPEAKER_PAGE_SIZE, keyword=kw
            )
        data = [
            [r["id"], r["name"], r["sample_count"], r["note"],
             _fmt_dt(r["created_at"]), _fmt_dt(r["updated_at"])]
            for r in rows
        ]
        info = f"第 {page} / {max_page} 頁　共 {total} 位語者"
        return data, info, page
    except Exception as e:
        logging.exception("語者列表讀取失敗")
        return [], f"⚠️ 資料庫讀取失敗：{e}", 1


def fn_speaker_enroll(name, note, files, page, keyword):
    name = (name or "").strip()
    if not name:
        data, info, page = _render_speaker_list(page, keyword)
        return "⚠️ 請先輸入姓名/代號。", data, info, page
    if not files:
        data, info, page = _render_speaker_list(page, keyword)
        return "⚠️ 請至少上傳一段語音樣本。", data, info, page

    paths = [getattr(f, "name", f) for f in files]
    vecs = []
    for p in paths:
        try:
            y, sr = _load_sample_audio(p)
            v = voiceprint.extract(y, sample_rate=sr)
            if v is not None:
                vecs.append(v)
        except Exception as e:
            logging.warning(f"樣本抽聲紋失敗 {p}: {e}")

    if not vecs:
        data, info, page = _render_speaker_list(page, keyword)
        return "❌ 所有樣本都抽不出有效聲紋（太短或無語音）。", data, info, page

    try:
        sid = speaker_db.add_speaker(
            name, vecs[0], note=(note or "").strip(), source="enroll"
        )
        for v in vecs[1:]:
            speaker_db.add_sample(sid, v, source="enroll")
    except Exception as e:
        logging.exception("建檔失敗")
        data, info, page = _render_speaker_list(page, keyword)
        return f"❌ 建檔失敗：{e}", data, info, page

    data, info, page = _render_speaker_list(page, keyword)
    return f"✅ 已建檔語者 #{sid}「{name}」，採用 {len(vecs)} 段樣本。", data, info, page


def fn_speaker_delete(speaker_id, page, keyword):
    try:
        sid = int(speaker_id)
    except (TypeError, ValueError):
        data, info, page = _render_speaker_list(page, keyword)
        return "⚠️ 請輸入有效的語者 ID。", data, info, page
    sp = speaker_db.get_speaker(sid)
    if not sp:
        data, info, page = _render_speaker_list(page, keyword)
        return f"⚠️ 找不到語者 #{sid}。", data, info, page
    speaker_db.delete_speaker(sid)
    data, info, page = _render_speaker_list(page, keyword)
    return f"🗑️ 已刪除語者 #{sid}「{sp['name']}」。", data, info, page


def fn_speaker_search(keyword):
    return _render_speaker_list(1, keyword)


def fn_speaker_page(delta, page, keyword):
    try:
        page = int(page)
    except (TypeError, ValueError):
        page = 1
    return _render_speaker_list(max(1, page + delta), keyword)


# ─────────────────────── V5.0 M5：聲道分離分頁 callbacks ───────────────────────
# 這些函式只呼叫 channel_split（純獨立工具），完全不碰轉錄/聲紋/LLM 任何邏輯。
CHANNEL_INFO_HEADERS = ["聲道", "取樣率(Hz)", "長度(秒)", "檔名", "路徑"]


def fn_detect_channels(audio_path):
    """上傳後偵測聲道數與基本資訊。"""
    if not audio_path:
        return "請先上傳音檔。", gr.update(interactive=False)
    try:
        info = channel_split.probe(audio_path)
    except Exception as e:
        logging.exception("聲道偵測失敗")
        return f"⚠️ 偵測失敗：{e}", gr.update(interactive=False)
    msg = (
        f"🎚️ 聲道數：**{info['channels']}**　｜　取樣率：{info['samplerate']} Hz　｜　"
        f"長度：{info['duration']:.1f} 秒　｜　格式：{info['format']}/{info['subtype']}"
    )
    if info["channels"] < 2:
        msg += "\n\nℹ️ 此檔為單聲道，無需分離（仍可按下方按鈕輸出一份複本）。"
    return msg, gr.update(interactive=True)


def fn_split_channels(audio_path):
    """執行分離，回傳下載檔清單與每聲道資訊表。"""
    if not audio_path:
        return "請先上傳音檔。", None, []
    out_root = CONFIG["paths"]["output_dir"]
    try:
        out_dir, results = channel_split.split_channels(audio_path, out_root)
    except Exception as e:
        logging.exception("聲道分離失敗")
        return f"⚠️ 分離失敗：{e}", None, []
    files = [r["path"] for r in results]
    rows = [
        [r["channel"], r["samplerate"], r["duration"], Path(r["path"]).name, r["path"]]
        for r in results
    ]
    status = f"✅ 已分離為 {len(results)} 個單聲道檔，存於：{out_dir}"
    return status, files, rows


# ─────────────────────── V5.0 M3：術語庫 RAG 分頁 callbacks ───────────────────────
# 這些函式只操作 glossary（DB + 記憶體快取），完全不碰轉錄/聲紋/LLM。
GLOSSARY_LIST_HEADERS = ["ID", "術語", "類型", "正確譯法", "備註", "語言", "命中"]
GLOSSARY_PAGE_SIZE = 20
# UI 顯示標籤 → DB 類型鍵
GLOSSARY_TYPE_CHOICES = {
    "專有名詞": "proper_noun",
    "同音錯字校正": "typo_fix",
    "使用者校正": "correction",
}


def _render_glossary_list(page, keyword):
    try:
        page = max(1, int(page or 1))
    except (TypeError, ValueError):
        page = 1
    try:
        rows, total = glossary.list_terms(page, GLOSSARY_PAGE_SIZE, keyword or "")
    except Exception as e:
        logging.exception("術語列表讀取失敗")
        return [], f"讀取失敗：{e}", page
    max_page = max(1, (total + GLOSSARY_PAGE_SIZE - 1) // GLOSSARY_PAGE_SIZE)
    page = min(page, max_page)
    data = [
        [
            r["id"], r["term"],
            glossary.TYPE_LABEL.get(r["term_type"], r["term_type"]),
            r["correct"], r["note"], r["source_lang"], r["hit_count"],
        ]
        for r in rows
    ]
    info = f"第 {page} / {max_page} 頁　共 {total} 條術語"
    return data, info, page


def fn_glossary_add(term, correct, type_label, note, source_lang, page, keyword):
    if not term or not term.strip() or not correct or not correct.strip():
        data, info, page = _render_glossary_list(page, keyword)
        return "⚠️ 術語與正確譯法皆不可為空。", data, info, page
    ttype = GLOSSARY_TYPE_CHOICES.get(type_label, "proper_noun")
    try:
        tid = glossary.add_term(
            term=term, correct=correct, term_type=ttype,
            note=note or "", source_lang=(source_lang or "any").strip() or "any",
        )
    except Exception as e:
        logging.exception("術語新增失敗")
        data, info, page = _render_glossary_list(page, keyword)
        return f"⚠️ 新增失敗：{e}", data, info, page
    data, info, page = _render_glossary_list(page, keyword)
    return f"✅ 已儲存術語 #{tid}「{term}」→「{correct}」。", data, info, page


def fn_glossary_delete(term_id, page, keyword):
    try:
        tid = int(term_id)
    except (TypeError, ValueError):
        data, info, page = _render_glossary_list(page, keyword)
        return "⚠️ 請輸入有效的術語 ID。", data, info, page
    ok = glossary.delete_term(tid)
    data, info, page = _render_glossary_list(page, keyword)
    msg = f"🗑️ 已刪除術語 #{tid}。" if ok else f"⚠️ 找不到術語 #{tid}。"
    return msg, data, info, page


def fn_glossary_search(keyword):
    return _render_glossary_list(1, keyword)


def fn_glossary_page(delta, page, keyword):
    try:
        page = int(page)
    except (TypeError, ValueError):
        page = 1
    return _render_glossary_list(max(1, page + delta), keyword)


def fn_glossary_import(file_path, page, keyword):
    import csv
    if not file_path:
        data, info, page = _render_glossary_list(page, keyword)
        return "⚠️ 請先選擇 CSV/TSV 檔。", data, info, page
    try:
        with open(file_path, "r", encoding="utf-8-sig", newline="") as f:
            sample = f.read(4096)
            f.seek(0)
            delim = "\t" if sample.count("\t") > sample.count(",") else ","
            reader = csv.reader(f, delimiter=delim)
            rows = [r for r in reader if any((c or "").strip() for c in r)]
    except Exception as e:
        logging.exception("術語匯入讀檔失敗")
        data, info, page = _render_glossary_list(page, keyword)
        return f"⚠️ 讀檔失敗：{e}", data, info, page
    # 若首列像表頭（含「術語/term」等字樣）則略過
    if rows and any(h in (rows[0][0] or "").lower() for h in ("term", "術語", "詞")):
        rows = rows[1:]
    ok, skip, errs = glossary.bulk_import([tuple(r) for r in rows])
    data, info, page = _render_glossary_list(1, keyword)
    msg = f"✅ 匯入完成：成功 {ok} 筆，略過 {skip} 筆。"
    if errs:
        msg += f"　⚠️ {len(errs)} 筆錯誤（前 3）：" + "；".join(errs[:3])
    return msg, data, info, page


# ── 校正學習（半自動）：擷取候選 → 使用者勾選 → 入庫 ──
def fn_extract_corrections(orig_translated, edited_translated):
    """比對原譯與使用者改後譯文，列出候選詞對供勾選。"""
    if not orig_translated or not edited_translated:
        return gr.update(choices=[], value=[]), "（尚無可擷取的修正；請先轉錄並編輯譯文）"
    try:
        cands = glossary.extract_corrections(orig_translated, edited_translated)
    except Exception as e:
        logging.exception("校正候選擷取失敗")
        return gr.update(choices=[], value=[]), f"⚠️ 擷取失敗：{e}"
    if not cands:
        return gr.update(choices=[], value=[]), "（未偵測到詞級修正；整句改寫不會被當作術語）"
    choices = [f"{c['wrong']} → {c['correct']}" for c in cands]
    return gr.update(choices=choices, value=choices), f"偵測到 {len(choices)} 條候選，請勾選要存入術語庫的項目。"


def fn_save_corrections(selected):
    if not selected:
        return "⚠️ 未勾選任何候選。"
    saved = 0
    for item in selected:
        if " → " not in item:
            continue
        wrong, correct = item.split(" → ", 1)
        wrong, correct = wrong.strip(), correct.strip()
        if not wrong or not correct:
            continue
        try:
            glossary.add_term(
                term=wrong, correct=correct, term_type="correction", source="learned"
            )
            saved += 1
        except Exception:
            logging.exception("校正入庫失敗：%s → %s", wrong, correct)
    return f"✅ 已將 {saved} 條校正存入術語庫（可至「術語庫管理」分頁查看/重新整理）。"


CSS = """
.yayan-title { font-size: 1.5em; font-weight: 600; }
.yayan-sub   { color: #888; font-size: 0.9em; }
.confidence-box textarea {
    text-align: center !important;
    font-size: 1.4em !important;
    font-weight: 700 !important;
    color: #2563eb !important;
    line-height: 1.3 !important;
}
"""


# 離線內網部署：主題字型一律用系統內建字型，不可用 GoogleFont。
# Gradio 預設主題的 Montserrat / IBM Plex Mono 會讓前端動態注入
# fonts.googleapis.com 的 stylesheet，外網不通時要等逾時才放棄，
# 表現為「第一次進站一直轉圈圈、之後就正常」。
THEME = gr.themes.Soft(
    font=["system-ui", "-apple-system", "Segoe UI", "Microsoft JhengHei",
          "PingFang TC", "Noto Sans TC", "sans-serif"],
    font_mono=["ui-monospace", "Consolas", "Menlo", "monospace"],
)


def build_ui() -> gr.Blocks:
    rag_on = CONFIG.get("rag", {}).get("enable_rag", False)  # M3：術語庫總開關
    with gr.Blocks(title=f"YaYan-AI v{__version__}", css=CSS, theme=THEME) as demo:
        gr.Markdown(
            f"""
            # 🏺 YaYan-AI **v{__version__}**　— 多語言語音轉錄系統
            <p class="yayan-sub">Edition: RTX6000-Server　|　雅言 YaYan 自主研發語音辨識模型　|　逐段 LID + 5 人說話人分離 + 字級時間戳</p>
            """
        )

        with gr.Tabs():
            # ───────── Tab 1：轉錄翻譯（v4.7 原樣，僅多包一層 Tab）─────────
            with gr.Tab("🎙️ 轉錄翻譯"):
                with gr.Row():
                    with gr.Column(scale=1):
                        audio_input = gr.Audio(
                            sources=["upload", "microphone"],
                            type="filepath",
                            label="🎤 上傳或錄製音檔",
                        )
                        dialect = gr.Dropdown(
                            choices=DIALECT_CHOICES,
                            value="🔍 自動偵測（建議：逐段 LID）",
                            label="來源語言",
                            info="台語／粵語等特定方言請直接指定——自動偵測分不出漢語方言，"
                                 "指定後才會啟用該語言的專用辨識引擎。",
                        )
                        enable_diarize = gr.Checkbox(
                            label="啟用說話人分離（A方/B方/C方/D方/E方）",
                            value=True,
                        )
                        transcribe_btn = gr.Button("🚀 開始轉錄翻譯", variant="primary", size="lg")
                        info_box = gr.Textbox(label="識別資訊", interactive=False, lines=3)

                        confidence_box = gr.Textbox(
                            label="🎯 識別精準度",
                            value="—",
                            interactive=False,
                            lines=2,
                            elem_classes=["confidence-box"],
                        )

                    with gr.Column(scale=2):
                        gr.Markdown("### 📜 識別原文（可編輯，含時間戳）")
                        raw_text_display = gr.State("")
                        orig_translated = gr.State("")   # M3：原始譯文（校正學習 diff 用）
                        raw_text_box = gr.Textbox(
                            label="ASR 原文",
                            lines=10,
                            interactive=True,
                            placeholder="格式：[A方 00:01-00:05] 你好",
                        )

                        gr.Markdown("### 🇹🇼 台灣正體中文譯文（可編輯）")
                        translated_box = gr.Textbox(
                            label="譯文",
                            lines=10,
                            interactive=True,
                            placeholder="翻譯結果會保留 [A方 00:01-00:05] 標籤",
                        )

                        with gr.Row():
                            refine_raw_btn = gr.Button("🔄 依【編輯後原文】重新翻譯潤飾", variant="secondary")
                            refine_translated_btn = gr.Button("✨ 依【編輯後譯文】重新潤飾", variant="secondary")

                        # ── M3 校正學習（半自動，rag 開啟時才出現）──
                        if rag_on:
                            with gr.Accordion("📖 把本次修正存進術語庫（校正學習）", open=False):
                                corr_extract_btn = gr.Button("🔍 擷取本次修正候選", size="sm")
                                corr_info = gr.Markdown("編輯上方譯文後，按此擷取詞級修正候選。")
                                corr_choices = gr.CheckboxGroup(
                                    label="候選詞對（勾選要存入的項目）", choices=[]
                                )
                                corr_save_btn = gr.Button("💾 存入勾選的校正", variant="primary", size="sm")
                                corr_status = gr.Textbox(label="校正結果", interactive=False, lines=1)

                        save_btn = gr.Button("💾 另存新檔", variant="primary")
                        save_file = gr.File(
                            label="📁 點擊下方檔案連結即可選擇儲存位置",
                            interactive=False,
                        )

                # 吃 GPU 的重事件（ASR+LLM）綁同一 concurrency_id 且限 1：彼此排隊、
                # 全速跑、避免兩份同時搶 cuda:0/cuda:1 而互相拖慢或 OOM。輕操作不受此限。
                transcribe_btn.click(
                    fn=fn_transcribe,
                    inputs=[audio_input, dialect, enable_diarize],
                    outputs=[info_box, raw_text_display, raw_text_box, translated_box,
                             confidence_box, orig_translated],
                    concurrency_id="gpu",
                    concurrency_limit=1,
                )
                refine_raw_btn.click(
                    fn=fn_refine,
                    inputs=[raw_text_display, raw_text_box, dialect],
                    outputs=[translated_box],
                    concurrency_id="gpu",
                    concurrency_limit=1,
                )
                refine_translated_btn.click(
                    fn=fn_refine,
                    inputs=[raw_text_display, translated_box, dialect],
                    outputs=[translated_box],
                    concurrency_id="gpu",
                    concurrency_limit=1,
                )
                save_btn.click(
                    fn=fn_save_as,
                    inputs=[translated_box, audio_input],
                    outputs=[save_file],
                )
                if rag_on:
                    corr_extract_btn.click(
                        fn=fn_extract_corrections,
                        inputs=[orig_translated, translated_box],
                        outputs=[corr_choices, corr_info],
                    )
                    corr_save_btn.click(
                        fn=fn_save_corrections,
                        inputs=[corr_choices],
                        outputs=[corr_status],
                    )

                gr.Markdown(
                    """
                    ---
                    **本版新增（v5.1）：**
                    - **台語專用辨識引擎**：來源語言選「閩南語 / 台語」即啟用，
                      辨識品質明顯優於通用引擎（自動偵測分不出台語，必須手動指定）
                    - **語言選擇提示**：逐字稿明顯不成句時，於識別資訊提醒可能選錯語言
                    - **術語庫**：專名統一譯法、同音錯字自動校正（見「術語庫管理」分頁）
                    - **聲紋語者識別**：建檔後自動標註說話人姓名（見「語者管理」分頁）
                    - **多聲道分離**：立體聲／多軌音檔拆成單聲道（見「聲道分離」分頁）
                    - **識別精準度**：改用連續評分，並納入語言判定信心與逐字稿可讀性

                    **既有功能：**
                    - **22 種中文方言**：含福州話、客家話、湘贛晉、潮汕、海南、蘇杭吳語細分等
                    - **字級時間戳**：每段顯示 `[A方 00:01-00:05] 內容`
                    - **逐段語言識別**：混合語音每段獨立判斷
                    - **5 人說話人分離**：A方 / B方 / C方 / D方 / E方
                    - **語言分布統計**：左上顯示音檔內各語言比例

                    **語音辨識引擎：**
                    - 漢語方言（22 種） + 藏維 → **雅言 YaYan 自主研發方言語音模型**
                    - 日韓 + 歐洲 + 中東 + 東南亞 → **雅言 YaYan 自主研發全球語音模型**
                    """
                )

            # ───────── Tab 2：語者管理（V5.0 M2 新增）─────────
            with gr.Tab("👤 語者管理"):
                gr.Markdown(
                    "### 👤 聲紋語者管理　"
                    "<span class='yayan-sub'>上傳語音樣本建檔聲紋，供轉錄時自動識別說話人</span>"
                )
                sp_page = gr.State(1)

                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("#### ➕ 建檔新語者")
                        sp_name = gr.Textbox(label="姓名 / 代號", placeholder="例：張三 或 目標A")
                        sp_note = gr.Textbox(label="備註（單位/番號等，可空）", placeholder="可留空")
                        sp_files = gr.File(
                            label="🎤 語音樣本（可多段，建議每段 ≥ 3 秒乾淨語音）",
                            file_count="multiple",
                            type="filepath",
                            file_types=["audio"],
                        )
                        sp_enroll_btn = gr.Button("➕ 建檔聲紋", variant="primary")
                        sp_status = gr.Textbox(label="操作結果", interactive=False, lines=2)

                        gr.Markdown("#### 🗑️ 刪除語者")
                        with gr.Row():
                            sp_del_id = gr.Number(label="語者 ID", precision=0, value=None)
                            sp_del_btn = gr.Button("🗑️ 刪除", variant="stop")

                    with gr.Column(scale=2):
                        gr.Markdown("#### 📋 已建檔語者")
                        with gr.Row():
                            sp_search = gr.Textbox(
                                label="🔍 關鍵字搜尋（姓名/代號）",
                                placeholder="輸入後按搜尋",
                                scale=3,
                            )
                            sp_search_btn = gr.Button("🔍 搜尋", scale=1)
                            sp_refresh_btn = gr.Button("🔄 重新整理", scale=1)
                        sp_list = gr.Dataframe(
                            headers=SPEAKER_LIST_HEADERS,
                            datatype=["number", "str", "number", "str", "str", "str"],
                            interactive=False,
                            wrap=True,
                            row_count=(1, "dynamic"),
                        )
                        with gr.Row():
                            sp_prev_btn = gr.Button("⬅️ 上一頁")
                            sp_pageinfo = gr.Textbox(
                                label="", interactive=False, lines=1, scale=3
                            )
                            sp_next_btn = gr.Button("下一頁 ➡️")

                # ── 事件綁定（全部只動 speaker 區塊，不影響 Tab 1）──
                sp_enroll_btn.click(
                    fn=fn_speaker_enroll,
                    inputs=[sp_name, sp_note, sp_files, sp_page, sp_search],
                    outputs=[sp_status, sp_list, sp_pageinfo, sp_page],
                )
                sp_del_btn.click(
                    fn=fn_speaker_delete,
                    inputs=[sp_del_id, sp_page, sp_search],
                    outputs=[sp_status, sp_list, sp_pageinfo, sp_page],
                )
                sp_search_btn.click(
                    fn=fn_speaker_search,
                    inputs=[sp_search],
                    outputs=[sp_list, sp_pageinfo, sp_page],
                )
                sp_refresh_btn.click(
                    fn=fn_speaker_search,
                    inputs=[sp_search],
                    outputs=[sp_list, sp_pageinfo, sp_page],
                )
                sp_prev_btn.click(
                    fn=lambda pg, kw: fn_speaker_page(-1, pg, kw),
                    inputs=[sp_page, sp_search],
                    outputs=[sp_list, sp_pageinfo, sp_page],
                )
                sp_next_btn.click(
                    fn=lambda pg, kw: fn_speaker_page(1, pg, kw),
                    inputs=[sp_page, sp_search],
                    outputs=[sp_list, sp_pageinfo, sp_page],
                )
                # 開啟頁面時自動載入第一頁
                demo.load(
                    fn=lambda: _render_speaker_list(1, ""),
                    outputs=[sp_list, sp_pageinfo, sp_page],
                )

            # ───────── Tab 3：聲道分離（V5.0 M5 新增，config 開關預設關）─────────
            if CONFIG.get("tools", {}).get("enable_channel_split", False):
              with gr.Tab("🎚️ 聲道分離"):
                gr.Markdown(
                    "### 🎚️ 多聲道分離　"
                    "<span class='yayan-sub'>把多聲道（立體聲/多軌）音檔拆成多個單聲道 wav，"
                    "供個別轉錄或處理</span>"
                )
                with gr.Row():
                    with gr.Column(scale=1):
                        ch_audio = gr.Audio(
                            label="🎵 上傳音檔",
                            type="filepath",
                        )
                        ch_detect_info = gr.Markdown("上傳後將自動偵測聲道數。")
                        ch_split_btn = gr.Button(
                            "✂️ 分離聲道", variant="primary", interactive=False
                        )
                        ch_status = gr.Textbox(label="操作結果", interactive=False, lines=2)
                    with gr.Column(scale=2):
                        gr.Markdown("#### 📋 各聲道資訊")
                        ch_table = gr.Dataframe(
                            headers=CHANNEL_INFO_HEADERS,
                            datatype=["number", "number", "number", "str", "str"],
                            interactive=False,
                            wrap=True,
                            row_count=(1, "dynamic"),
                        )
                        gr.Markdown("#### ⬇️ 下載單聲道檔")
                        ch_files = gr.File(
                            label="分離結果（點擊下載）",
                            file_count="multiple",
                            interactive=False,
                        )

                # ── 事件綁定（全部只動聲道分離區塊，不影響其他 Tab）──
                ch_audio.change(
                    fn=fn_detect_channels,
                    inputs=[ch_audio],
                    outputs=[ch_detect_info, ch_split_btn],
                )
                ch_split_btn.click(
                    fn=fn_split_channels,
                    inputs=[ch_audio],
                    outputs=[ch_status, ch_files, ch_table],
                )

            # ───────── Tab 4：術語庫管理（V5.0 M3 新增，config 開關預設關）─────────
            if rag_on:
                with gr.Tab("📖 術語庫管理"):
                    gr.Markdown(
                        "### 📖 特殊字詞術語庫　"
                        "<span class='yayan-sub'>專名統一譯法、同音錯字校正、使用者校正；"
                        "翻譯時自動檢索比對並注入</span>"
                    )
                    gl_page = gr.State(1)
                    with gr.Row():
                        with gr.Column(scale=1):
                            gr.Markdown("#### ➕ 新增 / 更新術語")
                            gl_term = gr.Textbox(label="術語（來源詞/錯字/原文）", placeholder="例：產稅")
                            gl_correct = gr.Textbox(label="正確譯法 / 正確詞", placeholder="例：殘缺")
                            gl_type = gr.Dropdown(
                                label="類型",
                                choices=list(GLOSSARY_TYPE_CHOICES.keys()),
                                value="專有名詞",
                            )
                            gl_note = gr.Textbox(label="備註（番號/單位/來源，可空）", placeholder="可留空")
                            gl_lang = gr.Textbox(label="適用來源語言", value="any", placeholder="any 或 zh/yue …")
                            gl_add_btn = gr.Button("➕ 儲存術語", variant="primary")
                            gl_status = gr.Textbox(label="操作結果", interactive=False, lines=2)

                            gr.Markdown("#### 🗑️ 刪除術語")
                            with gr.Row():
                                gl_del_id = gr.Number(label="術語 ID", precision=0, value=None)
                                gl_del_btn = gr.Button("🗑️ 刪除", variant="stop")

                            gr.Markdown("#### 📥 批次匯入（CSV/TSV）")
                            gr.Markdown(
                                "<span class='yayan-sub'>欄位順序：術語, 正確譯法, 類型"
                                "(proper_noun/typo_fix/correction), 備註, 語言（後三欄可省）</span>"
                            )
                            gl_import_file = gr.File(label="選擇 CSV/TSV", type="filepath")
                            gl_import_btn = gr.Button("📥 匯入")

                        with gr.Column(scale=2):
                            gr.Markdown("#### 📋 術語清單")
                            with gr.Row():
                                gl_search = gr.Textbox(
                                    label="🔍 關鍵字（術語/譯法/備註）",
                                    placeholder="輸入後按搜尋", scale=3,
                                )
                                gl_search_btn = gr.Button("🔍 搜尋", scale=1)
                                gl_refresh_btn = gr.Button("🔄 重新整理", scale=1)
                            gl_list = gr.Dataframe(
                                headers=GLOSSARY_LIST_HEADERS,
                                datatype=["number", "str", "str", "str", "str", "str", "number"],
                                interactive=False, wrap=True, row_count=(1, "dynamic"),
                            )
                            with gr.Row():
                                gl_prev_btn = gr.Button("⬅️ 上一頁")
                                gl_pageinfo = gr.Textbox(label="", interactive=False, lines=1, scale=3)
                                gl_next_btn = gr.Button("下一頁 ➡️")

                    # ── 事件綁定（只動術語區塊）──
                    gl_add_btn.click(
                        fn=fn_glossary_add,
                        inputs=[gl_term, gl_correct, gl_type, gl_note, gl_lang, gl_page, gl_search],
                        outputs=[gl_status, gl_list, gl_pageinfo, gl_page],
                    )
                    gl_del_btn.click(
                        fn=fn_glossary_delete,
                        inputs=[gl_del_id, gl_page, gl_search],
                        outputs=[gl_status, gl_list, gl_pageinfo, gl_page],
                    )
                    gl_import_btn.click(
                        fn=fn_glossary_import,
                        inputs=[gl_import_file, gl_page, gl_search],
                        outputs=[gl_status, gl_list, gl_pageinfo, gl_page],
                    )
                    gl_search_btn.click(
                        fn=fn_glossary_search, inputs=[gl_search],
                        outputs=[gl_list, gl_pageinfo, gl_page],
                    )
                    gl_refresh_btn.click(
                        fn=fn_glossary_search, inputs=[gl_search],
                        outputs=[gl_list, gl_pageinfo, gl_page],
                    )
                    gl_prev_btn.click(
                        fn=lambda pg, kw: fn_glossary_page(-1, pg, kw),
                        inputs=[gl_page, gl_search],
                        outputs=[gl_list, gl_pageinfo, gl_page],
                    )
                    gl_next_btn.click(
                        fn=lambda pg, kw: fn_glossary_page(1, pg, kw),
                        inputs=[gl_page, gl_search],
                        outputs=[gl_list, gl_pageinfo, gl_page],
                    )
                    demo.load(
                        fn=lambda: _render_glossary_list(1, ""),
                        outputs=[gl_list, gl_pageinfo, gl_page],
                    )
    return demo


def main():
    server_cfg = CONFIG["server"]
    print("=" * 60)
    print(f"  YaYan-AI v{__version__}  |  Edition: RTX6000-Server")
    print(f"  Models root: {CONFIG['paths']['models_root']}")
    print(f"  ASR GPU: {CONFIG['devices']['asr_gpu']}  |  LLM GPU: {CONFIG['devices']['llm_gpu']}")
    print(f"  LLM backend: {CONFIG['llm'].get('backend')}  |  quant: {CONFIG['llm'].get('quantization')}")
    print(f"  支援語言/方言數: {len(DIALECT_TO_ROUTING)}")
    print("=" * 60)

    print("⏳ 預載模型 …")
    try:
        warmup()
    except Exception as e:
        print(f"⚠️ Warmup 失敗（仍可啟動）：{e}")

    # V5.0 M2：聲紋資料庫初始化（建表 IF NOT EXISTS；失敗不影響轉錄功能）
    try:
        speaker_db.init_db()
        print(f"✅ 聲紋資料庫就緒（目前 {speaker_db.count_speakers()} 位語者）")
    except Exception as e:
        print(f"⚠️ 聲紋資料庫初始化失敗（語者管理分頁不可用，轉錄不受影響）：{e}")

    # V5.0 M3：術語庫初始化（僅在 enable_rag 時；失敗不影響轉錄/翻譯）
    if CONFIG.get("rag", {}).get("enable_rag", False):
        try:
            glossary.init_db()
            print(f"✅ 術語庫就緒（目前 {glossary.count_terms()} 條術語）")
        except Exception as e:
            print(f"⚠️ 術語庫初始化失敗（術語庫分頁不可用，翻譯不受影響）：{e}")

    demo = build_ui()
    # allowed_paths：Gradio 4.x 只允許下載 cwd/暫存目錄內的檔案；輸出檔寫在 output_dir
    # （絕對路徑，服務 cwd 之外），未列入就會下載失敗。涵蓋「另存新檔」與「聲道分離」兩處下載。
    allowed = [str(Path(CONFIG["paths"]["output_dir"]).resolve())]
    demo.queue(default_concurrency_limit=2).launch(
        server_name=server_cfg["host"],
        server_port=server_cfg["port"],
        share=server_cfg["share"],
        allowed_paths=allowed,
    )


if __name__ == "__main__":
    main()
