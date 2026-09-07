#!/usr/bin/env python
"""YaYan 評測腳本 — 吃「評測集 manifest」吐 CER / LID 混淆矩陣 / 時間戳誤差。

這是所有模型升級決策的對照組產生器：任何「新模型比較好」的說法，
都要能在同一份 manifest 上跑出更好的數字才算數。

用法：
    conda run --no-capture-output -n yayan python scripts/evaluate_asr.py \\
        --manifest /data/ai_datasets/eval/manifest.jsonl \\
        --language auto \\
        --out .claude/scratch/$(date +%F)-eval.json

manifest 格式（每行一個 JSON）：
    {"audio": "/abs/path.wav",       # 必填
     "text": "參考逐字稿",             # 算 CER 用；沒有就只跑 LID
     "dialect": "nan",               # 人工方言標籤（routing key）；沒有就不做混淆矩陣
     "words": [{"start": 0.5, "end": 0.7, "text": "你"}]}   # 選填，算時間戳誤差

⚠️ 標籤品質決定一切：2026-08-24 的教訓是 /data/Sample 有 3 個檔內容與檔名不符，
   一度得出「台語只有 20%」的錯誤結論。用新評測集前先 --verify-labels 抽驗。
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# 正規化：CER 只比對「內容字元」，標點/空白/拼音註記不計入
_PUNCT = re.compile(r"[\s，。、！？；：「」『』（）()\[\]【】《》〈〉…—－\-~～·・.,!?;:\"'`]+")
_PAREN = re.compile(r"[（(]([^（()）]*)[）)]")
_CJK = re.compile(r"[㐀-䶿一-鿿豈-﫿]")


def _drop_non_cjk_parens(m: re.Match) -> str:
    """括號內完全沒有漢字就整段剝除（台語羅馬字註記、英文釋義都適用）。

    不用列舉帶調字母——那樣會漏掉大寫變體（Ài vs ài），實測踩過這個坑。
    """
    return "" if not _CJK.search(m.group(1)) else m.group(0)


def normalize(text: str, strip_romanization: bool = True) -> str:
    """把參考與辨識結果化到同一個比較基準。"""
    if not text:
        return ""
    if strip_romanization:
        # Common Voice 台語逐字稿是「漢字（羅馬字）」格式，羅馬字不列入 CER
        text = _PAREN.sub(_drop_non_cjk_parens, text)
    text = _PUNCT.sub("", text)
    return text.strip()


def edit_distance(ref: str, hyp: str) -> tuple[int, int, int]:
    """回傳 (替換, 刪除, 插入)。O(min(m,n)) 空間的 Levenshtein + 回溯計數。"""
    m, n = len(ref), len(hyp)
    if m == 0:
        return 0, 0, n
    if n == 0:
        return 0, m, 0

    # 完整 DP 表（評測集規模不大，直接存表以便回溯分類錯誤型態）
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if ref[i - 1] == hyp[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j - 1], dp[i - 1][j], dp[i][j - 1])

    sub = dele = ins = 0
    i, j = m, n
    while i > 0 or j > 0:
        if i > 0 and j > 0 and ref[i - 1] == hyp[j - 1] and dp[i][j] == dp[i - 1][j - 1]:
            i, j = i - 1, j - 1
        elif i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + 1:
            sub += 1
            i, j = i - 1, j - 1
        elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            dele += 1
            i -= 1
        else:
            ins += 1
            j -= 1
    return sub, dele, ins


def timestamp_error(ref_words: list, hyp_words: list) -> dict | None:
    """字級時間戳誤差：只比對「文字相同且順序對得上」的字，避免拿錯位的字比。"""
    if not ref_words or not hyp_words:
        return None
    errs = []
    hi = 0
    for rw in ref_words:
        rt = normalize(rw.get("text", ""))
        if not rt:
            continue
        for k in range(hi, len(hyp_words)):
            if normalize(hyp_words[k].get("text", "")) == rt:
                errs.append(abs(float(hyp_words[k].get("start", 0)) - float(rw.get("start", 0))))
                hi = k + 1
                break
    if not errs:
        return None
    errs.sort()
    return {
        "matched_words": len(errs),
        "mean_abs_err_s": round(sum(errs) / len(errs), 3),
        "median_abs_err_s": round(errs[len(errs) // 2], 3),
        "p90_abs_err_s": round(errs[int(len(errs) * 0.9)], 3),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="YaYan ASR / LID 評測")
    ap.add_argument("--manifest", required=True, help="評測集 jsonl")
    ap.add_argument("--language", default="auto",
                    help="送給 pipeline 的來源語言；auto 才會觸發逐段 LID")
    ap.add_argument("--out", help="結果寫成 JSON 的路徑")
    ap.add_argument("--limit", type=int, help="只跑前 N 筆（除錯用）")
    ap.add_argument("--no-diarize", action="store_true", help="關閉說話人分離（加快）")
    ap.add_argument("--keep-romanization", action="store_true",
                    help="不剝除參考文字括號內的羅馬字（預設剝除）")
    args = ap.parse_args()

    items = []
    with open(args.manifest, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    if args.limit:
        items = items[:args.limit]
    if not items:
        print("manifest 是空的", file=sys.stderr)
        return 1

    from yayan import pipeline
    from yayan.config import CONFIG

    backend = CONFIG["asr"].get("lid_backend", "voxlingua")
    print(f"評測 {len(items)} 筆 | language={args.language} | lid_backend={backend}",
          flush=True)

    tot_ref = tot_sub = tot_del = tot_ins = 0
    per_dialect = defaultdict(lambda: {"ref": 0, "sub": 0, "del": 0, "ins": 0, "n": 0})
    confusion = defaultdict(Counter)
    ts_stats = []
    failures = []
    t0 = time.time()

    for idx, it in enumerate(items, 1):
        audio = it["audio"]
        try:
            res = pipeline.transcribe_audio(
                audio, language=args.language,
                use_diarize=False if args.no_diarize else None,
            )
        except Exception as e:
            failures.append({"audio": audio, "error": f"{type(e).__name__}: {e}"})
            continue

        truth = it.get("dialect")
        if truth:
            confusion[truth][res.routing] += 1

        # ⚠️ 用逐段的 raw_text 拼，不要用 res.raw_text——後者含
        # 「[A方 00:01-00:05]」前綴，正規化剝掉標點後會變成 000105 混進 CER。
        hyp_text = "".join(s.raw_text or "" for s in res.segments) or res.raw_text

        ref_raw = it.get("text")
        if ref_raw:
            ref = normalize(ref_raw, strip_romanization=not args.keep_romanization)
            hyp = normalize(hyp_text, strip_romanization=False)
            sub, dele, ins = edit_distance(ref, hyp)
            tot_ref += len(ref); tot_sub += sub; tot_del += dele; tot_ins += ins
            if truth:
                d = per_dialect[truth]
                d["ref"] += len(ref); d["sub"] += sub
                d["del"] += dele; d["ins"] += ins; d["n"] += 1

        if it.get("words"):
            hyp_words = [
                {"start": w.start, "end": w.end, "text": w.text}
                for s in res.segments for w in (getattr(s, "words", None) or [])
            ]
            te = timestamp_error(it["words"], hyp_words)
            if te:
                ts_stats.append(te)

        if idx % 10 == 0 or idx == len(items):
            print(f"  {idx}/{len(items)}  ({time.time()-t0:.0f}s)", flush=True)

    def cer(ref_n, s, d, i):
        return round((s + d + i) / ref_n * 100, 2) if ref_n else None

    report = {
        "manifest": args.manifest,
        "language": args.language,
        "lid_backend": backend,
        "n_items": len(items),
        "n_failed": len(failures),
        "elapsed_s": round(time.time() - t0, 1),
        "overall_cer_pct": cer(tot_ref, tot_sub, tot_del, tot_ins),
        "overall_counts": {"ref_chars": tot_ref, "sub": tot_sub,
                           "del": tot_del, "ins": tot_ins},
        "per_dialect_cer_pct": {
            k: {"cer_pct": cer(v["ref"], v["sub"], v["del"], v["ins"]),
                "n_items": v["n"], "ref_chars": v["ref"]}
            for k, v in sorted(per_dialect.items())
        },
        "lid_confusion": {k: dict(v.most_common()) for k, v in sorted(confusion.items())},
        "lid_recall_pct": {
            k: round(v.get(k, 0) / sum(v.values()) * 100, 1)
            for k, v in sorted(confusion.items()) if sum(v.values())
        },
        "timestamp": ({
            "n_files": len(ts_stats),
            "mean_abs_err_s": round(sum(t["mean_abs_err_s"] for t in ts_stats) / len(ts_stats), 3),
        } if ts_stats else None),
        "failures": failures[:20],
    }

    print("\n===== 結果 =====")
    print(f"整體 CER: {report['overall_cer_pct']}%   （失敗 {len(failures)} 筆）")
    if report["per_dialect_cer_pct"]:
        print("\n逐方言 CER:")
        for k, v in report["per_dialect_cer_pct"].items():
            print(f"  {k:<10} {v['cer_pct']:>6}%   ({v['n_items']} 筆 / {v['ref_chars']} 字)")
    if report["lid_recall_pct"]:
        print("\nLID 召回率（判對自己的比例）:")
        for k, v in report["lid_recall_pct"].items():
            top = list(confusion[k].most_common(3))
            print(f"  {k:<10} {v:>5}%   最常判成: {top}")
    if report["timestamp"]:
        print(f"\n時間戳平均絕對誤差: {report['timestamp']['mean_abs_err_s']}s")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=2),
                                  encoding="utf-8")
        print(f"\n完整結果 → {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
