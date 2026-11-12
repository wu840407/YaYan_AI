"""
YaYan-AI v4.5 批次處理（含 watchdog 自動監看）
使用：
  python auto_batch_rtx6000.py            # 一次性處理 input_dir 所有檔案
  python auto_batch_rtx6000.py --watch    # 持續監看新檔，邊放邊處理
  python auto_batch_rtx6000.py --diarize  # 啟用說話人分離
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Set

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("YaYan.Batch")

from yayan import __version__
from yayan.config import CONFIG
from yayan.pipeline import transcribe_audio, warmup

SUPPORTED_EXT = {".mp3", ".wav", ".m4a", ".flac", ".ogg", ".aac", ".opus"}


def _is_audio(p: Path) -> bool:
    return p.is_file() and p.suffix.lower() in SUPPORTED_EXT


def process_one(audio_path: Path, output_dir: Path, language: str, use_diarize: bool) -> bool:
    out_txt = output_dir / f"{audio_path.stem}_translated.txt"
    out_json = output_dir / f"{audio_path.stem}_meta.json"

    if out_txt.exists():
        logger.info(f"⏩ 已處理過，略過: {audio_path.name}")
        return True

    t0 = time.time()
    try:
        result = transcribe_audio(
            str(audio_path),
            language=language,
            use_diarize=use_diarize,
        )
    except Exception as e:
        logger.exception(f"❌ 處理失敗: {audio_path.name} | {e}")
        return False

    output_dir.mkdir(parents=True, exist_ok=True)
    out_txt.write_text(result.translated_text or "", encoding="utf-8")
    out_json.write_text(
        json.dumps(result.to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    dt = time.time() - t0
    logger.info(f"✅ {audio_path.name} → {out_txt.name}  ({dt:.1f}s)")
    return True


def run_once(input_dir: Path, output_dir: Path, language: str, use_diarize: bool) -> int:
    files = sorted(p for p in input_dir.iterdir() if _is_audio(p))
    if not files:
        logger.info(f"📂 {input_dir} 無音檔。")
        return 0
    logger.info(f"📊 找到 {len(files)} 個檔案。")
    ok = 0
    for p in files:
        if process_one(p, output_dir, language, use_diarize):
            ok += 1
    logger.info(f"🎉 完成 {ok}/{len(files)}")
    return ok


def run_watch(input_dir: Path, output_dir: Path, language: str, use_diarize: bool) -> None:
    try:
        from watchdog.events import FileSystemEventHandler
        from watchdog.observers import Observer
    except ImportError:
        logger.warning("watchdog 未安裝，回退為 polling 模式（每 5 秒掃描）。")
        _run_polling(input_dir, output_dir, language, use_diarize)
        return

    seen: Set[str] = set()
    for p in input_dir.iterdir():
        if _is_audio(p):
            seen.add(p.name)

    logger.info(f"👀 開始監看 {input_dir}（已忽略現有 {len(seen)} 個既有檔案）")

    class Handler(FileSystemEventHandler):
        def on_created(self, event):
            if event.is_directory:
                return
            p = Path(event.src_path)
            if not _is_audio(p):
                return
            time.sleep(1.0)
            logger.info(f"📥 新檔: {p.name}")
            process_one(p, output_dir, language, use_diarize)

    observer = Observer()
    observer.schedule(Handler(), str(input_dir), recursive=False)
    observer.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()


def _run_polling(input_dir: Path, output_dir: Path, language: str, use_diarize: bool) -> None:
    seen: Set[str] = set(p.name for p in input_dir.iterdir() if _is_audio(p))
    interval = CONFIG["batch"]["poll_interval_seconds"]
    logger.info(f"👀 polling 模式（{interval}s/次）")
    while True:
        try:
            for p in input_dir.iterdir():
                if _is_audio(p) and p.name not in seen:
                    logger.info(f"📥 新檔: {p.name}")
                    process_one(p, output_dir, language, use_diarize)
                    seen.add(p.name)
        except KeyboardInterrupt:
            return
        time.sleep(interval)


def main():
    parser = argparse.ArgumentParser(description=f"YaYan-AI v{__version__} 批次處理")
    parser.add_argument("--input", default=CONFIG["paths"]["input_dir"], help="輸入資料夾")
    parser.add_argument("--output", default=CONFIG["paths"]["output_dir"], help="輸出資料夾")
    parser.add_argument("--language", default="auto", help="auto/zh/yue/wuu/bo/ug/fa/ur/en")
    parser.add_argument("--diarize", action="store_true", help="啟用說話人分離")
    parser.add_argument("--watch", action="store_true", help="持續監看新檔")
    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)

    if not input_dir.exists():
        input_dir.mkdir(parents=True, exist_ok=True)
        logger.warning(f"建立空輸入資料夾: {input_dir}")
        if not args.watch:
            logger.info("無檔可處理，結束。")
            return

    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(f"  YaYan-AI v{__version__} Batch")
    print(f"  Input : {input_dir}")
    print(f"  Output: {output_dir}")
    print(f"  Mode  : {'watch' if args.watch else 'one-shot'}")
    print("=" * 60)

    logger.info("⏳ 預載模型 …")
    warmup()

    if args.watch:
        run_watch(input_dir, output_dir, args.language, args.diarize)
    else:
        run_once(input_dir, output_dir, args.language, args.diarize)


if __name__ == "__main__":
    main()
