"""
YaYan-AI V5.0 M5 — 多聲道分離工具（純獨立模組）

把一個多聲道音檔拆成多個單聲道 wav，每聲道一檔。
- 偵測：soundfile.info（不解碼整檔即可取得聲道數/取樣率/長度）
- 分離：soundfile 串流分塊（sf.blocks），記憶體只佔一個 block，與總時長無關
- 輸出：保留原始取樣率，固定 16-bit PCM（PCM_16），對下游 ASR 最通用

⚠️ 此模組完全不依賴 pipeline / ASR / LID / 翻譯 / 聲紋 / LLM，
   import 它不會載入任何模型。請勿在此引入重模組。
"""
from __future__ import annotations

import logging
import shutil
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path

import numpy as np
import soundfile as sf

logger = logging.getLogger("yayan.channel_split")

# 保護上限：聲道數異常多通常代表檔案解析錯誤，擋下避免產生大量垃圾檔
MAX_CHANNELS = 32
# 串流分塊大小（frames）。~1M frames * 32ch * 4byte ≈ 128MB 峰值上限，安全。
BLOCK_FRAMES = 1_048_576


class ChannelSplitError(Exception):
    """聲道分離相關錯誤（給 UI 顯示友善訊息用）。"""


def _ffmpeg_to_wav(src: Path) -> Path:
    """soundfile 無法讀的容器（部分 mp3/m4a 等）→ 用 ffmpeg 轉成臨時 wav。

    只做容器轉換，保留原聲道數與取樣率（不混音、不重採樣）。
    """
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise ChannelSplitError("此音檔格式 soundfile 無法直接讀取，且系統找不到 ffmpeg 可轉換。")
    tmp = Path(tempfile.mkdtemp(prefix="yayan_chsplit_")) / (src.stem + ".wav")
    cmd = [ffmpeg, "-y", "-i", str(src), "-c:a", "pcm_s16le", str(tmp)]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
    except subprocess.CalledProcessError as e:
        err = e.stderr.decode("utf-8", "ignore")[-500:] if e.stderr else str(e)
        raise ChannelSplitError(f"ffmpeg 轉檔失敗：{err}") from e
    return tmp


def _open_info(path: str):
    """回傳 (sf.info 結果, 實際可讀路徑, 是否為 ffmpeg 臨時檔)。

    優先直接讀；失敗則 fallback ffmpeg 轉 wav 再讀。
    """
    src = Path(path)
    if not src.exists():
        raise ChannelSplitError(f"找不到檔案：{path}")
    try:
        return sf.info(str(src)), str(src), False
    except Exception as e:
        logger.info("soundfile 無法直接讀 %s（%s），改用 ffmpeg fallback", src.name, e)
        wav = _ffmpeg_to_wav(src)
        return sf.info(str(wav)), str(wav), True


def probe(path: str) -> dict:
    """偵測音檔聲道資訊（不解碼整檔）。供 UI 上傳後即時顯示。

    回傳 dict：channels, samplerate, frames, duration, format, subtype。
    """
    info, _real, is_tmp = _open_info(path)
    if is_tmp:
        # probe 階段用完即清掉臨時 wav（分離時會再轉一次；probe 通常只跑一次成本可接受）
        try:
            shutil.rmtree(Path(_real).parent, ignore_errors=True)
        except Exception:
            pass
    return {
        "channels": int(info.channels),
        "samplerate": int(info.samplerate),
        "frames": int(info.frames),
        "duration": float(info.duration),
        "format": info.format,
        "subtype": info.subtype,
    }


def split_channels(
    path: str,
    out_root: str,
    block_frames: int = BLOCK_FRAMES,
) -> tuple[str, list[dict]]:
    """把多聲道音檔拆成多個單聲道 16-bit PCM wav。

    串流分塊處理，記憶體與總時長無關。

    回傳 (輸出資料夾路徑, 每聲道資訊 list)。
    每聲道資訊：channel(1-based), path, samplerate, frames, duration。

    單聲道檔不重複產檔，仍會複製一份到輸出夾以維持一致行為。
    """
    info, real_path, is_tmp = _open_info(path)
    tmp_dir = Path(real_path).parent if is_tmp else None
    try:
        n_ch = int(info.channels)
        sr = int(info.samplerate)
        if n_ch < 1:
            raise ChannelSplitError("偵測不到有效聲道。")
        if n_ch > MAX_CHANNELS:
            raise ChannelSplitError(
                f"聲道數為 {n_ch}，超過上限 {MAX_CHANNELS}，疑似檔案解析錯誤，已中止。"
            )

        base = Path(path).stem
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = Path(out_root) / "channel_split" / f"{base}_{ts}"
        out_dir.mkdir(parents=True, exist_ok=True)

        out_paths = [out_dir / f"ch{ch + 1:02d}.wav" for ch in range(n_ch)]
        writers: list[sf.SoundFile] = []
        written_frames = 0
        try:
            for op in out_paths:
                writers.append(
                    sf.SoundFile(str(op), mode="w", samplerate=sr, channels=1, subtype="PCM_16")
                )
            for block in sf.blocks(real_path, blocksize=block_frames, dtype="float32", always_2d=True):
                # block 形狀 (frames, channels)；切第二維取單聲道
                for ch in range(n_ch):
                    writers[ch].write(block[:, ch])
                written_frames += block.shape[0]
        finally:
            for w in writers:
                try:
                    w.close()
                except Exception:
                    pass

        duration = written_frames / sr if sr else 0.0
        results = [
            {
                "channel": ch + 1,
                "path": str(out_paths[ch]),
                "samplerate": sr,
                "frames": written_frames,
                "duration": round(duration, 3),
            }
            for ch in range(n_ch)
        ]
        logger.info("聲道分離完成：%s → %d 檔（%d frames/ch）", base, n_ch, written_frames)
        return str(out_dir), results
    finally:
        if tmp_dir is not None:
            shutil.rmtree(tmp_dir, ignore_errors=True)
