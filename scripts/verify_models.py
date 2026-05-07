#!/usr/bin/env python3
"""驗證所有 YaYan-AI v4.5 模型已正確放置在 models_root，可離線載入。"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from yayan.config import CONFIG, ALIASES, model_path  # noqa: E402

REQUIRED = [
    ("YaYan_Reasoner", True, 20_000, "LLM (Qwen3-14B BF16 ~28GB)"),
    ("YaYan_ASR_Mandarin",  True,     200,  "SenseVoiceSmall"),
    ("YaYan_ASR_Eastern",   True,     100,  "Dolphin-base"),
    ("YaYan_ASR_Global",    True,   1_000,  "Whisper-large-v3"),
    ("YaYan_LID",           True,      10,  "VoxLingua107 ECAPA"),
]
OPTIONAL = [
    ("YaYan_Diarize",        False,    1,  "pyannote speaker-diarization-3.1 (pipeline)"),
    ("YaYan_Diarize_Seg",    False,    1,  "pyannote segmentation-3.0"),
    ("YaYan_Diarize_Embed",  False,    5,  "pyannote wespeaker-voxceleb-resnet34-LM"),
]


def _dir_size_mb(p: Path) -> float:
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) / 1024 / 1024


def _check_llm_quant_config(p: Path) -> str:
    cfg_path = p / "config.json"
    if not cfg_path.exists():
        return ""
    try:
        with cfg_path.open("r", encoding="utf-8") as f:
            cfg = json.load(f)
    except Exception:
        return ""
    qc = cfg.get("quantization_config") or {}
    quant_method = qc.get("quant_method") or qc.get("method") or ""
    bits = qc.get("bits") or qc.get("w_bit") or ""
    if quant_method:
        return f" [quant={quant_method}{f'/{bits}-bit' if bits else ''}]"
    if cfg.get("torch_dtype"):
        return f" [dtype={cfg['torch_dtype']}]"
    return ""


def check(alias: str, required: bool, min_mb: float, hint: str) -> bool:
    try:
        p = model_path(alias)
    except KeyError:
        if required:
            print(f"  ❌ {alias} 未註冊於 yayan/config.py 的 ALIASES，請補上對應路徑")
            return False
        print(f"  ⚠️  {alias} 未註冊（可選，跳過）  ({hint})")
        return True

    if not p.exists():
        msg = "❌ 缺失" if required else "⚠️  未下載（可選）"
        print(f"  {msg}: {alias}  -> {p}    ({hint})")
        return not required

    files = list(p.iterdir())
    if not files:
        print(f"  ❌ 空資料夾: {alias} -> {p}")
        return not required

    size_mb = _dir_size_mb(p)
    if size_mb < min_mb:
        print(
            f"  ⚠️  {alias} 大小可疑 ({size_mb:.1f} MB < 預期 {min_mb} MB)，"
            f"可能下載不完整 -> {p}"
        )
        return not required

    extra = _check_llm_quant_config(p) if alias == "YaYan_Reasoner" else ""
    print(f"  ✅ {alias}  ({size_mb:,.1f} MB){extra}  {hint}")
    return True


def _check_silero_vad():
    try:
        import silero_vad
        ver = getattr(silero_vad, "__version__", "?")
        print(f"  ✅ silero-vad（pip 內建權重） v{ver}")
        return True
    except ImportError:
        print(f"  ❌ silero-vad 未安裝，請：pip install 'silero-vad>=5.1'")
        return False


def main():
    print("=" * 64)
    print(f"  YaYan-AI v4.5 模型檢查")
    print(f"  models_root: {CONFIG['paths']['models_root']}")
    print(f"  llm.backend: {CONFIG['llm'].get('backend')}  "
          f"quant: {CONFIG['llm'].get('quantization')}")
    print("=" * 64)

    print("\n[必要 — 從 HF 下載到 models_root]")
    ok = all(check(a, r, m, h) for a, r, m, h in REQUIRED)

    print("\n[必要 — pip 套件內建]")
    ok = _check_silero_vad() and ok

    print("\n[可選 — Diarization 三件套]")
    for a, r, m, h in OPTIONAL:
        check(a, r, m, h)

    print("\n" + "=" * 64)
    if ok:
        print("✅ 所有必要模型已就緒，可啟動 app_rtx6000.py")
        sys.exit(0)
    else:
        print("❌ 有必要模型缺失或不完整")
        sys.exit(1)


if __name__ == "__main__":
    main()
