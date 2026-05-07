"""
YaYan-AI v4.5 — Quadro RTX 6000 (Turing) x 2 主介面
"""
from __future__ import annotations

import os
import sys
import json
import logging
from datetime import datetime
from pathlib import Path

# 必須在 import gradio / huggingface_hub 之前
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
os.environ.setdefault("GRADIO_DO_NOT_TRACK", "1")
os.environ.setdefault("AWQ_USE_TRITON", "0")

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)

import gradio as gr

from yayan import __version__
from yayan.config import CONFIG
from yayan.pipeline import (
    transcribe_audio,
    refine_with_user_edit,
    warmup,
    TranscriptionResult,
)


# ---------------- 語言選單 ---------------- #
DIALECT_TO_ROUTING = {
    "🔍 自動偵測": "auto",
    # ── 漢語系 ──
    "🇨🇳 北京話 / 普通話": "zh",
    "🇨🇳 山東話": "zh",
    "🇨🇳 上海話 (吳語)": "wuu",
    "🇨🇳 四川話": "zh",
    "🇭🇰 廣東話 (粵語)": "yue",
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
ROUTING_TO_DIALECT = {v: k for k, v in DIALECT_TO_ROUTING.items()}


# ---------------- 業務邏輯 ---------------- #

def _calc_confidence(result: TranscriptionResult) -> float:
    """估算識別精準度分數（0-100）。"""
    score = 0.0
    if result.segments:
        score += 60.0
        avg_dur = sum(s.end - s.start for s in result.segments) / len(result.segments)
        if 1.0 <= avg_dur <= 15.0:
            score += 20.0
        elif 0.5 <= avg_dur <= 25.0:
            score += 10.0
    if result.routing and result.routing != "auto":
        score += 10.0
    if result.translated_text and result.translated_text.strip():
        score += 10.0
    return round(min(100.0, score), 1)


def fn_transcribe(audio_path, dialect_label, enable_diarize):
    if audio_path is None:
        return "請先上傳或錄製音檔。", "", "", "", "—"

    routing = DIALECT_TO_ROUTING.get(dialect_label, "auto")
    try:
        result: TranscriptionResult = transcribe_audio(
            audio_path,
            language=routing,
            use_diarize=enable_diarize,
        )
    except Exception as e:
        logging.exception("transcribe 失敗")
        return f"識別失敗：{e}", "", "", "", "—"

    detected = ROUTING_TO_DIALECT.get(result.routing, result.routing)
    info = f"偵測語言：{detected}（{result.routing}）｜段數：{len(result.segments)}"
    confidence = _calc_confidence(result)
    score_text = f"{confidence:.1f} / 100"
    return info, result.raw_text, result.raw_text, result.translated_text, score_text


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
    """產生下載檔；前端會跳『另存新檔』對話框。"""
    if not translated_text or not translated_text.strip():
        return None
    out_dir = Path(CONFIG["paths"]["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    base = Path(source_audio).stem if source_audio else "manual"
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = out_dir / f"{base}_{ts}.txt"
    out.write_text(translated_text, encoding="utf-8")
    return str(out)


# ---------------- Gradio 介面 ---------------- #

CSS = """
.yayan-title { font-size: 1.5em; font-weight: 600; }
.yayan-sub   { color: #888; font-size: 0.9em; }
.confidence-box textarea {
    text-align: center !important;
    font-size: 1.6em !important;
    font-weight: 700 !important;
    color: #2563eb !important;
}
"""


def build_ui() -> gr.Blocks:
    with gr.Blocks(title=f"YaYan-AI v{__version__}", css=CSS, theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            f"""
            # 🏺 YaYan-AI **v{__version__}**　— 多語言情報系統
            <p class="yayan-sub">Edition: RTX6000-Server　|　ASR: SenseVoice / Dolphin / Whisper-large-v3　|　LLM: Qwen3-32B-AWQ</p>
            """
        )

        with gr.Row():
            with gr.Column(scale=1):
                audio_input = gr.Audio(
                    sources=["upload", "microphone"],
                    type="filepath",
                    label="🎤 上傳或錄製音檔",
                )
                dialect = gr.Dropdown(
                    choices=DIALECT_CHOICES,
                    value="🔍 自動偵測",
                    label="來源語言（建議用自動偵測）",
                )
                enable_diarize = gr.Checkbox(
                    label="啟用說話人分離（雙人通話建議）",
                    value=False,
                )
                transcribe_btn = gr.Button("🚀 開始轉錄翻譯", variant="primary", size="lg")
                info_box = gr.Textbox(label="識別資訊", interactive=False, lines=2)

                # ★ 左下角：精準度分數
                confidence_box = gr.Textbox(
                    label="🎯 識別精準度分數",
                    value="—",
                    interactive=False,
                    elem_classes=["confidence-box"],
                )

            with gr.Column(scale=2):
                gr.Markdown("### 📜 識別原文（可編輯）")
                raw_text_display = gr.State("")
                raw_text_box = gr.Textbox(
                    label="ASR 原文",
                    lines=6,
                    interactive=True,
                    placeholder="識別結果會顯示在此處，您可直接修改後按「依編輯重新潤飾」。",
                )

                gr.Markdown("### 🇹🇼 台灣正體中文譯文（可編輯）")
                translated_box = gr.Textbox(
                    label="譯文",
                    lines=8,
                    interactive=True,
                    placeholder="翻譯結果會顯示在此處，您可直接修改後按「依編輯重新潤飾」。",
                )

                with gr.Row():
                    refine_raw_btn = gr.Button("🔄 依【編輯後原文】重新翻譯潤飾", variant="secondary")
                    refine_translated_btn = gr.Button("✨ 依【編輯後譯文】重新潤飾", variant="secondary")

                save_btn = gr.Button("💾 另存新檔", variant="primary")
                save_file = gr.File(
                    label="📁 點擊下方檔案連結即可選擇儲存位置",
                    interactive=False,
                )

        # ---------- 事件 ---------- #
        transcribe_btn.click(
            fn=fn_transcribe,
            inputs=[audio_input, dialect, enable_diarize],
            outputs=[info_box, raw_text_display, raw_text_box, translated_box, confidence_box],
        )

        refine_raw_btn.click(
            fn=fn_refine,
            inputs=[raw_text_display, raw_text_box, dialect],
            outputs=[translated_box],
        )

        refine_translated_btn.click(
            fn=fn_refine,
            inputs=[raw_text_display, translated_box, dialect],
            outputs=[translated_box],
        )

        save_btn.click(
            fn=fn_save_as,
            inputs=[translated_box, audio_input],
            outputs=[save_file],
        )

        gr.Markdown(
            """
            ---
            **使用提示：**
            1. 上傳音檔 → 點「開始轉錄翻譯」 → 取得原文與譯文。
            2. 若 ASR 有錯，**直接在「ASR 原文」框修改** → 點「依編輯後原文重新翻譯潤飾」。
            3. 若譯文要微調，**直接在「譯文」框修改** → 點「依編輯後譯文重新潤飾」。
            4. 滿意後按「另存新檔」，可選擇儲存位置（瀏覽器原生對話框）。

            **支援語言：** 漢語方言 9 種、藏維 2 種、東亞 2 種、中東南亞 4 種、歐洲 5 種、東南亞 4 種，共 26 種。
            """
        )
    return demo


def main():
    server_cfg = CONFIG["server"]
    print("=" * 60)
    print(f"  YaYan-AI v{__version__}  |  Edition: RTX6000-Server")
    print(f"  Models root: {CONFIG['paths']['models_root']}")
    print(f"  ASR GPU: {CONFIG['devices']['asr_gpu']}  |  LLM GPU: {CONFIG['devices']['llm_gpu']}")
    print(f"  LLM backend: {CONFIG['llm'].get('backend')}  |  quant: {CONFIG['llm'].get('quantization')}")
    print(f"  支援語言數: {len(DIALECT_TO_ROUTING)}")
    print("=" * 60)

    print("⏳ 預載模型 …")
    try:
        warmup()
    except Exception as e:
        print(f"⚠️ Warmup 失敗（仍可啟動，將在首次請求時載入）：{e}")

    demo = build_ui()
    demo.queue(default_concurrency_limit=2).launch(
        server_name=server_cfg["host"],
        server_port=server_cfg["port"],
        share=server_cfg["share"],
    )


if __name__ == "__main__":
    main()
