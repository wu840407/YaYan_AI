#!/usr/bin/env python3
"""驗證 YaYan-AI 在離線模式下能正常運作。

使用方式：
  python scripts/verify_offline.py            # 模擬離線（socket guard）
  python scripts/verify_offline.py --real     # 真的拔網路線後再跑
  python scripts/verify_offline.py --audio a.wav --audio b.wav --language auto
      # 另外在離線狀態下實際轉錄音檔（會載入所有用到的模型、呼叫本機 LLM）
"""
from __future__ import annotations

import argparse
import os
import re
import sys
import socket
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def block_external_network():
    real_create_connection = socket.create_connection

    def guarded_create_connection(addr, *args, **kwargs):
        host = addr[0] if isinstance(addr, tuple) else addr
        if host in ("127.0.0.1", "localhost", "::1"):
            return real_create_connection(addr, *args, **kwargs)
        BLOCKED.append(str(addr))
        raise OSError(f"❌ 偵測到對外連線嘗試: {addr} —— 不是純離線！")

    socket.create_connection = guarded_create_connection

    # 有些套件直接呼叫 socket.connect，不經過 create_connection，一併攔截
    real_connect = socket.socket.connect

    def guarded_connect(self, address):
        if self.family in (socket.AF_INET, socket.AF_INET6):
            host = address[0] if isinstance(address, tuple) else address
            if host not in ("127.0.0.1", "localhost", "::1"):
                BLOCKED.append(str(address))
                raise OSError(f"❌ 偵測到對外連線嘗試: {address} —— 不是純離線！")
        return real_connect(self, address)

    socket.socket.connect = guarded_connect


BLOCKED: list = []   # 被攔下的對外連線（即使被套件吞掉例外也會留紀錄）


def set_offline_env():
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    os.environ["MODELSCOPE_OFFLINE"] = "1"
    os.environ["DO_NOT_TRACK"] = "1"
    os.environ["VLLM_NO_USAGE_STATS"] = "1"


CHECKS = []


def check(name):
    def deco(fn):
        CHECKS.append((name, fn))
        return fn
    return deco


@check("環境變數")
def chk_env():
    keys = ["HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"]
    missing = [k for k in keys if os.environ.get(k) != "1"]
    if missing:
        return False, f"未設: {missing}"
    return True, "全部 = 1"


@check("模型目錄（HF 來源）")
def chk_models():
    from yayan.config import model_path
    # silero-vad 不在這裡查（走 pip 內建權重）
    from yayan.config import CONFIG
    asr = CONFIG["asr"]
    needed = ["YaYan_ASR_Dialect", "YaYan_ASR_Global", "YaYan_LID",
              "YaYan_Diarize", "YaYan_Diarize_Seg", "YaYan_Diarize_Embed"]
    if asr.get("enable_taigi_asr"):
        needed.append("YaYan_ASR_Taigi")
    if CONFIG["llm"].get("backend", "transformers") == "transformers":
        needed.append("YaYan_Reasoner")
    paths = {a: model_path(a) for a in needed}
    if str(asr.get("lid_backend", "voxlingua")).lower() == "firered":
        paths["firered_lid_dir"] = Path(asr.get("firered_lid_dir", ""))
    missing = [a for a, p in paths.items() if not p.exists() or not any(p.iterdir())]
    if missing:
        return False, f"缺: {missing}"
    return True, f"齊（{len(paths)} 項）"


@check("LLM 服務（本機）")
def chk_llm_endpoint():
    from yayan.config import CONFIG
    llm = CONFIG["llm"]
    if llm.get("backend", "transformers") != "openai":
        return True, "backend=transformers，行程內載入，略過"
    import urllib.request
    from urllib.parse import urlparse
    base = str(llm.get("openai_base_url") or llm.get("base_url") or "").rstrip("/")
    if not base:
        return False, "找不到 openai base_url 設定"
    host = urlparse(base).hostname
    if host not in ("127.0.0.1", "localhost", "::1"):
        return False, f"LLM 端點不是本機：{host}"
    with urllib.request.urlopen(f"{base}/models", timeout=10) as r:
        return r.status == 200, f"/models HTTP {r.status}"


@check("transformers 載 LLM tokenizer")
def chk_tokenizer():
    from transformers import AutoTokenizer
    from yayan.config import model_path
    tok = AutoTokenizer.from_pretrained(
        str(model_path("YaYan_Reasoner")),
        trust_remote_code=True,
        local_files_only=True,
    )
    return True, f"vocab={len(tok)}"


@check("Qwen3 chat template 含 enable_thinking")
def chk_template():
    from transformers import AutoTokenizer
    from yayan.config import model_path
    tok = AutoTokenizer.from_pretrained(
        str(model_path("YaYan_Reasoner")),
        trust_remote_code=True,
        local_files_only=True,
    )
    try:
        out = tok.apply_chat_template(
            [{"role": "user", "content": "test"}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        # Qwen3 在 enable_thinking=False 時會輸出「空的」<think></think> 區塊，這是正常的
        if "<think>" in out and not re.search(r"<think>\s*</think>", out):
            return False, "template 仍含未關閉的 <think>，可能不是 Qwen3"
        return True, "OK（無 <think> 起手）"
    except TypeError:
        return False, "tokenizer 不支援 enable_thinking 參數（非 Qwen3?）"


@check("silero-vad 內建權重可載入")
def chk_silero_load():
    import silero_vad
    ver = getattr(silero_vad, "__version__", "?")
    # 真的去呼叫一次，不只是 import 而已
    from silero_vad import load_silero_vad
    m = load_silero_vad()
    return True, f"v{ver} 載入 OK（{type(m).__name__}）"


@check("OpenCC 可載 s2twp")
def chk_opencc():
    from opencc import OpenCC
    cc = OpenCC("s2twp")
    out = cc.convert("信息处理")
    return ("資訊" in out or "訊息" in out), out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", action="store_true",
                    help="不啟用 socket guard，假設你已經真的拔網路線")
    ap.add_argument("--audio", action="append", default=[],
                    help="離線狀態下實際轉錄這個音檔（可重複）")
    ap.add_argument("--language", default="auto")
    args = ap.parse_args()

    if args.audio:
        def chk_transcribe():
            from yayan import pipeline
            outs = []
            for a in args.audio:
                res = pipeline.transcribe_audio(a, language=args.language)
                routings = sorted({s.routing for s in res.segments})
                if not res.raw_text or not res.translated_text:
                    return False, f"{Path(a).name}: 轉錄或翻譯為空"
                outs.append(f"{Path(a).name}→{routings}")
            return True, "; ".join(outs)
        CHECKS.append(("實際轉錄（含 LLM）", chk_transcribe))

    set_offline_env()
    if not args.real:
        block_external_network()
        print("🔒 已啟用 socket guard（任何對外連線會丟例外）")
    else:
        print("📡 真離線模式（請確認網路線拔掉或防火牆封死）")

    print("=" * 60)
    failed = 0
    for name, fn in CHECKS:
        try:
            ok, msg = fn()
            mark = "✅" if ok else "❌"
            print(f"  {mark} {name:35s}  {msg}")
            if not ok:
                failed += 1
        except Exception as e:
            print(f"  ❌ {name:35s}  {type(e).__name__}: {e}")
            failed += 1
    if BLOCKED:
        print(f"  ❌ 有 {len(BLOCKED)} 次對外連線被攔下（即使上面都打勾）：{sorted(set(BLOCKED))[:5]}")
        failed += 1
    print("=" * 60)
    if failed:
        print(f"❌ 失敗 {failed} 項")
        sys.exit(1)
    print("✅ 全部離線檢查通過，可以放心拔網路線")


if __name__ == "__main__":
    main()
