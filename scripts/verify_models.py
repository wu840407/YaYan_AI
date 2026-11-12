#!/usr/bin/env python3
"""模型驗證腳本。"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from yayan.config import CONFIG, ALIASES, model_path  # noqa: E402

REQUIRED = [
    ("YaYan_Reasoner",      True,  20_000,  "LLM (Qwen3-14B BF16 ~28GB)"),
    ("YaYan_ASR_Dialect",   True,   1_200,  "Dolphin-CN-Dialect-Small (22 方言 ~1.6GB)"),
    ("YaYan_ASR_Global",    True,   1_000,  "Whisper-large-v3"),
    ("YaYan_LID",           True,      10,  "VoxLingua107 ECAPA"),
]
OPTIONAL = [
    ("YaYan_Diarize",        False,    1,  "pyannote speaker-diarization-3.1"),
    ("YaYan_Diarize_Seg",    False,    1,  "pyannote segmentation-3.0"),
    ("YaYan_Diarize_Embed",  False,    5,  "pyannote wespeaker"),
    ("YaYan_ASR_Taigi",      False, 1500,  "v4.7-C 台語專用 Breeze-ASR-26（預設停用，缺失退回 Dolphin）"),
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
    if qc.get("quant_method"):
        return f" [quant={qc['quant_method']}]"
    if cfg.get("torch_dtype"):
        return f" [dtype={cfg['torch_dtype']}]"
    return ""


def check(alias: str, required: bool, min_mb: float, hint: str) -> bool:
    try:
        p = model_path(alias)
    except KeyError:
        if required:
            print(f"  ❌ {alias} 未註冊於 model_aliases.yaml")
            return False
        print(f"  ⚠️  {alias} 未註冊（可選，跳過）  ({hint})")
        return True

    if not p.exists():
        msg = "❌ 缺失" if required else "⚠️  未下載（可選）"
        print(f"  {msg}: {alias}  -> {p}    ({hint})")
        return not required

    if not any(p.iterdir()):
        print(f"  ❌ 空資料夾: {alias} -> {p}")
        return not required

    size_mb = _dir_size_mb(p)
    if size_mb < min_mb:
        print(f"  ⚠️  {alias} 大小可疑 ({size_mb:.1f} MB < 預期 {min_mb} MB)")
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
        print(f"  ❌ silero-vad 未安裝")
        return False


def _check_dolphin_sdk():
    try:
        import dolphin
        print(f"  ✅ dolphin SDK 已安裝")
        return True
    except ImportError:
        print(f"  ❌ dolphin SDK 未安裝，請：pip install dataoceanai-dolphin")
        return False


def main():
    print("=" * 64)
    print(f"  YaYan-AI 模型檢查")
    print(f"  models_root: {CONFIG['paths']['models_root']}")
    print(f"  llm.backend: {CONFIG['llm'].get('backend')} | quant: {CONFIG['llm'].get('quantization')}")
    print("=" * 64)

    print("\n[必要 — 從 HF 下載]")
    ok = all(check(a, r, m, h) for a, r, m, h in REQUIRED)

    print("\n[必要 — pip 套件]")
    ok = _check_silero_vad() and ok
    ok = _check_dolphin_sdk() and ok

    print("\n[可選 — Diarization]")
    for a, r, m, h in OPTIONAL:
        check(a, r, m, h)

    print("\n" + "=" * 64)
    if ok:
        print("✅ 所有必要模型已就緒")
        sys.exit(0)
    else:
        print("❌ 有必要模型/套件缺失")
        sys.exit(1)


if __name__ == "__main__":
    main()
