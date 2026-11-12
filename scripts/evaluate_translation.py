#!/usr/bin/env python
"""YaYan 端到端評測 — 語音 → 華語文字，對照人工華語參考稿。

`evaluate_asr.py` 量的是「聽寫對不對」（CER，需要逐字參考稿）；本腳本量的是系統最終產出
「語音 → 華語文字」與人工華語參考稿的接近程度，適用於參考稿是譯文而非逐字稿的評測集。

用法（兩階段，皆可中斷續跑）：
    # 1) 轉錄：每個單位在每種模式下各跑一次 ASR，結果累加寫入 --work
    conda run --no-capture-output -n yayan python scripts/evaluate_translation.py asr \\
        --manifest eval.jsonl --work run.jsonl --modes auto,label
    # 2) 翻譯＋計分：相同的原文只翻一次；輸出彙總 JSON
    conda run --no-capture-output -n yayan python scripts/evaluate_translation.py score \\
        --manifest eval.jsonl --work run.jsonl --out result.json

manifest 格式（每行一個 JSON）：
    {"id": "x_0001",                 # 必填，唯一
     "group": "nan",                 # 彙總分組用（通常是語種）
     "audio": ["/abs/a.wav", ...],   # 必填；多個檔會各自轉錄後串接
     "ref": "人工華語參考稿（純內容）",  # 必填
     "label": "nan"}                 # 人工語種（routing key）；modes 含 label 時必填

模式：auto＝來源語言自動偵測（系統預設）；label＝直接指定 manifest 的 label（等於使用者選對語言）。
兩者的差距就是「語種判定還能改善多少」的上限。

指標（皆只比對內容字元，忽略標點、空白、[A方 00:01-00:05] 這類標籤，並先做簡繁正規化）：
    chrF   字元 n-gram（1~4）F2 分數，對語序與措辭差異較寬容，適合比對譯文
    recall 參考稿的字有多少比例能在系統輸出中依序對上（輸出多講不扣分，適合參考稿有省略的情況）
"""
from __future__ import annotations

import argparse
import difflib
import json
import os
import re
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

_TAG = re.compile(r"\[[^\]]*\]")
_CONTENT = re.compile(r"[㐀-䶿一-鿿豈-﫿0-9A-Za-z]")
try:
    import opencc
    _T2S = opencc.OpenCC("t2s").convert
except Exception:  # 沒有 opencc 時退化為不做簡繁正規化
    _T2S = lambda s: s


def normalize(text: str) -> str:
    return "".join(_CONTENT.findall(_T2S(_TAG.sub("", text or "")))).lower()


def chrf(ref: str, hyp: str, max_n: int = 4, beta: float = 2.0) -> float:
    if not ref or not hyp:
        return 0.0
    ps, rs = [], []
    for n in range(1, max_n + 1):
        r = Counter(ref[i:i + n] for i in range(len(ref) - n + 1))
        h = Counter(hyp[i:i + n] for i in range(len(hyp) - n + 1))
        if not r or not h:
            continue
        hit = sum((r & h).values())
        ps.append(hit / sum(h.values()))
        rs.append(hit / sum(r.values()))
    if not ps:
        return 0.0
    p, r = sum(ps) / len(ps), sum(rs) / len(rs)
    if p + r == 0:
        return 0.0
    return (1 + beta ** 2) * p * r / (beta ** 2 * p + r)


def recall(ref: str, hyp: str) -> float:
    if not ref:
        return 0.0
    m = difflib.SequenceMatcher(None, ref, hyp, autojunk=False)
    return sum(b.size for b in m.get_matching_blocks()) / len(ref)


def _load(path: str) -> list:
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def _gpu_free_mib(index: int = 0) -> int:
    try:
        out = subprocess.run(
            ["nvidia-smi", f"--id={index}", "--query-gpu=memory.total,memory.used",
             "--format=csv,noheader,nounits"], capture_output=True, text=True, check=True).stdout
        total, used = (int(x) for x in out.strip().split(","))
        return total - used
    except Exception:
        return 1 << 30


def cmd_asr(args) -> int:
    units = _load(args.manifest)
    modes = [m for m in args.modes.split(",") if m]
    done = set()
    if os.path.exists(args.work):
        done = {(r["id"], r["mode"]) for r in _load(args.work)}
    jobs = [(u, m) for m in modes for u in units if (u["id"], m) not in done]
    print(f"units={len(units)} modes={modes} todo={len(jobs)}", flush=True)
    if not jobs:
        return 0

    import torch
    if args.gpu_cap_gb:
        total = torch.cuda.get_device_properties(0).total_memory / 2 ** 30
        torch.cuda.set_per_process_memory_fraction(min(1.0, args.gpu_cap_gb / total), device=0)
    from yayan import pipeline

    captured: list = []

    def _stub(text_lines, source_language="", **kw):   # 轉錄階段不打 LLM，翻譯留到 score 階段
        captured.append((list(text_lines), source_language))
        return "\n".join(text_lines)

    pipeline._batched_translate = _stub
    t0 = time.time()
    with open(args.work, "a", encoding="utf-8") as fo:
        for n, (u, mode) in enumerate(jobs):
            if _gpu_free_mib() < args.min_free_mib:
                print(f"GPU 剩餘記憶體低於 {args.min_free_mib} MiB，先停（可續跑）", flush=True)
                return 3
            lang = "auto" if mode == "auto" else u["label"]
            rec = dict(id=u["id"], mode=mode, language=lang, parts=[])
            for path in u["audio"]:
                captured.clear()
                part = dict(audio=path)
                try:
                    res = pipeline.transcribe_audio(path, language=lang, use_diarize=args.diarize)
                    part.update(routing=res.routing, breakdown=res.language_breakdown,
                                speech_sec=round(sum(s.end - s.start for s in res.segments), 1),
                                raw="\n".join(s.raw_text for s in res.segments),
                                to_translate=[dict(lines=l, src=s) for l, s in captured])
                except Exception as e:
                    part.update(error=f"{type(e).__name__}: {e}")
                rec["parts"].append(part)
            fo.write(json.dumps(rec, ensure_ascii=False) + "\n"); fo.flush()
            if n % 10 == 0:
                print(f"{n}/{len(jobs)} {time.time() - t0:.0f}s", flush=True)
    print(f"ASR done; peak GPU {torch.cuda.max_memory_allocated() / 1e9:.1f} GB", flush=True)
    return 0


def cmd_score(args) -> int:
    units = {u["id"]: u for u in _load(args.manifest)}
    runs = [r for r in _load(args.work) if r["id"] in units]
    cache_path = args.work + ".translations.json"
    cache = json.load(open(cache_path, encoding="utf-8")) if os.path.exists(cache_path) else {}

    if not args.no_translate:
        from yayan import pipeline
        todo = []
        for r in runs:
            for p in r["parts"]:
                for t in p.get("to_translate", []):
                    key = json.dumps([t["src"], t["lines"]], ensure_ascii=False)
                    if key not in cache and key not in todo and any(l.strip() for l in t["lines"]):
                        todo.append(key)
        print(f"翻譯：{len(todo)} 批待送（已快取 {len(cache)}）", flush=True)
        for n, key in enumerate(todo):
            src, lines = json.loads(key)
            cache[key] = pipeline._batched_translate(lines, source_language=src)
            if n % 10 == 0 or n == len(todo) - 1:
                json.dump(cache, open(cache_path, "w", encoding="utf-8"), ensure_ascii=False)
                print(f"  {n + 1}/{len(todo)}", flush=True)

    rows = []
    for r in runs:
        u = units[r["id"]]
        raw = "\n".join(p.get("raw", "") for p in r["parts"])
        trans = []
        for p in r["parts"]:
            for t in p.get("to_translate", []):
                key = json.dumps([t["src"], t["lines"]], ensure_ascii=False)
                trans.append(cache.get(key, "\n".join(t["lines"])))
        ref, hr, ht = normalize(u["ref"]), normalize(raw), normalize("\n".join(trans))
        speech = sum(p.get("speech_sec", 0) for p in r["parts"])
        rows.append(dict(
            id=r["id"], group=u.get("group", ""), mode=r["mode"], label=u.get("label"),
            routing=[p.get("routing") for p in r["parts"]],
            errors=[p["error"] for p in r["parts"] if p.get("error")],
            ref_chars=len(ref), raw_chars=len(hr), out_chars=len(ht), speech_sec=round(speech, 1),
            # 語音不少卻幾乎沒有輸出：多半是推論失敗被吞掉，不是真的沒內容
            suspect_empty=bool(speech >= 5 and len(hr) < 0.5 * speech),
            chrf_out=round(chrf(ref, ht), 4), recall_out=round(recall(ref, ht), 4),
            chrf_raw=round(chrf(ref, hr), 4), recall_raw=round(recall(ref, hr), 4)))

    summary = {}
    for mode in sorted({x["mode"] for x in rows}):
        by = defaultdict(list)
        for x in rows:
            if x["mode"] == mode:
                by[x["group"]].append(x); by["ALL"].append(x)
        summary[mode] = {}
        for g, L in sorted(by.items()):
            w = sum(x["ref_chars"] for x in L) or 1
            avg = lambda k: round(sum(x[k] * x["ref_chars"] for x in L) / w, 4)
            lab = [x for x in L if x.get("label")]
            summary[mode][g] = dict(
                n=len(L), ref_chars=w, chrf_out=avg("chrf_out"), recall_out=avg("recall_out"),
                chrf_raw=avg("chrf_raw"), recall_raw=avg("recall_raw"),
                routing_match=round(sum(all(rt == x["label"] for rt in x["routing"]) for x in lab) / len(lab), 4) if lab else None,
                suspect_empty=sum(x["suspect_empty"] for x in L), errors=sum(bool(x["errors"]) for x in L))
    print("\n模式   分組   筆數 | 輸出 chrF / recall | 原文 chrF / recall | 路由=標籤 | 疑似空輸出")
    for mode, G in summary.items():
        for g, s in G.items():
            rm = "  -  " if s["routing_match"] is None else f"{s['routing_match']:.0%}"
            print(f"{mode:6s} {g:5s} {s['n']:4d} | {s['chrf_out']:.3f} / {s['recall_out']:.3f}     | "
                  f"{s['chrf_raw']:.3f} / {s['recall_raw']:.3f}     | {rm:>5s} | {s['suspect_empty']}")
    if args.out:
        json.dump(dict(summary=summary, rows=rows), open(args.out, "w", encoding="utf-8"),
                  ensure_ascii=False, indent=1)
        print(f"\n結果已寫入 {args.out}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="YaYan 端到端評測（語音 → 華語文字）")
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("asr", help="轉錄階段")
    a.add_argument("--manifest", required=True)
    a.add_argument("--work", required=True, help="轉錄結果（jsonl，可續跑）")
    a.add_argument("--modes", default="auto,label")
    a.add_argument("--diarize", action="store_true", help="開啟說話人分離（預設關：只評內容，省記憶體）")
    a.add_argument("--gpu-cap-gb", type=float, default=0, help="本行程的 GPU 記憶體上限（0＝不設）")
    a.add_argument("--min-free-mib", type=int, default=0, help="GPU 剩餘記憶體低於此值就先停（與其他服務共卡時用）")
    a.set_defaults(fn=cmd_asr)
    s = sub.add_parser("score", help="翻譯＋計分階段")
    s.add_argument("--manifest", required=True)
    s.add_argument("--work", required=True)
    s.add_argument("--out")
    s.add_argument("--no-translate", action="store_true", help="不呼叫 LLM，只用已快取的翻譯（沒有就以原文代替）")
    s.set_defaults(fn=cmd_score)
    args = ap.parse_args()
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
