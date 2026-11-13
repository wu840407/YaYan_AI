#!/usr/bin/env python
"""YaYan LLM 評測 — 比較兩個 LLM 在「逐字稿校正／正規化」這個任務上的表現。

`scripts/evaluate_asr.py` 測的是 ASR，補不上 LLM 這塊。本腳本的存在理由：
換 LLM 時若沒有可機械計分的指標，就只能憑感覺，而「感覺比較好」在
長批次格式遵循這種事情上非常不可靠。

⚠️ 本專案的 LLM 做的是**校正／正規化**，不是自由翻譯——輸出長度應該 ≈ 輸入長度。
   因此「改寫幅度大」是缺點不是優點，這與一般翻譯評測的直覺相反。

四個指標（全部可機械計分，不需參考譯文）：
  1. 格式違規率    時間戳標籤 [A方 00:01-00:05] 的保留率、行數是否對得上
  2. 過度改寫率    與原文的字元級編輯距離（校正任務應該小幅修改）
  3. 術語命中率    指定術語有沒有被正確保留／替換
  4. 中國用語殘留  OpenCC 之前就該由 LLM 處理掉的簡體與中國慣用詞

用法（兩個 endpoint 都要是 OpenAI 相容）：
    python scripts/evaluate_llm.py \\
        --endpoint http://127.0.0.1:8001/v1 --name qwen3.6 \\
        --manifest /path/to/llm_manifest.jsonl \\
        --out results/$(date +%F)-llm-qwen36.json

manifest 每行：{"raw_text": "[A方 00:01-00:05] 逐字稿…", "source_language": "zh",
               "terms": ["應保留的術語"]}   # terms 選填
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

_TAG_RE = re.compile(r"\[[^\]]{2,40}\]")

# 中國慣用詞 → 台灣慣用詞。只收「OpenCC s2twp 之前就該由 LLM 處理掉」且
# 不會誤傷的詞；模稜兩可的（如「質量」在物理語境是正確的）不收。
_CN_TERMS = {
    "信息": "資訊", "軟件": "軟體", "硬件": "硬體", "網絡": "網路",
    "視頻": "影片", "音頻": "音訊", "數據庫": "資料庫", "服務器": "伺服器",
    "激活": "啟用", "默認": "預設", "屏幕": "螢幕", "打印": "列印",
    "拷貝": "複製", "菜單": "選單", "回車": "換行", "內存": "記憶體",
}
_SIMPLIFIED_ONLY = set(
    "这个们说为时过国东华实现发经济团员关联报应该议观点电语种认识"
    "讲请谢铁钟银错长门问间闻开买卖车马鸟鱼龙见贝页风飞习乡书写"
)


def edit_distance(a: str, b: str) -> int:
    """字元級 Levenshtein（只要距離，用 O(min) 空間）。"""
    if len(a) < len(b):
        a, b = b, a
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def score_one(src: str, out: str, terms: list[str]) -> dict:
    res = {}

    # 1. 格式：標籤保留率 + 行數一致
    n_src_tag = len(_TAG_RE.findall(src))
    n_out_tag = len(_TAG_RE.findall(out))
    res["tag_keep"] = (n_out_tag / n_src_tag) if n_src_tag else 1.0
    src_lines = [l for l in src.splitlines() if l.strip()]
    out_lines = [l for l in out.splitlines() if l.strip()]
    res["line_match"] = (
        min(len(out_lines), len(src_lines)) / max(len(src_lines), 1)
        if src_lines else 1.0
    )

    # 2. 過度改寫：校正任務的編輯距離應該偏小
    res["edit_ratio"] = edit_distance(src, out) / max(len(src), 1)

    # 3. 術語命中
    if terms:
        hit = sum(1 for t in terms if t and t in out)
        res["term_hit"] = hit / len(terms)
    else:
        res["term_hit"] = None

    # 4. 中國用語殘留（OpenCC 之前）
    res["simplified_chars"] = sum(1 for ch in out if ch in _SIMPLIFIED_ONLY)
    res["cn_terms"] = [k for k in _CN_TERMS if k in out]

    # LLM 贅述外洩（prompt 要求只輸出結果）
    res["has_preamble"] = any(
        m in out for m in ("以下是", "翻譯如下", "Here is", "Translation:", "<think>")
    )
    return res


def main() -> int:
    ap = argparse.ArgumentParser(description="YaYan LLM 校正品質評測")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--endpoint", required=True, help="OpenAI 相容 base_url")
    ap.add_argument("--name", required=True, help="這個模型的代號，用於報告")
    ap.add_argument("--model", default=None, help="served model name（預設自動取第一個）")
    ap.add_argument("--out")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--timeout", type=int, default=600)
    ap.add_argument("--temperature", type=float, default=0.6,
                    help="取樣溫度（預設 0.6 同正式設定；評測比較建議 0）")
    ap.add_argument("--outputs-out", help="逐筆模型輸出寫成 jsonl（比對改前改後用）")
    args = ap.parse_args()

    import requests
    from yayan.config import load_prompt

    items = [json.loads(l) for l in open(args.manifest, encoding="utf-8") if l.strip()]
    if args.limit:
        items = items[:args.limit]

    model = args.model
    if not model:
        r = requests.get(f"{args.endpoint}/models", timeout=30)
        model = r.json()["data"][0]["id"]

    prompt_tpl = load_prompt("translate")
    print(f"評測 {len(items)} 筆 | {args.name} | model={model} | {args.endpoint}",
          flush=True)

    rows = []
    t0 = time.time()
    for i, it in enumerate(items, 1):
        src = it["raw_text"]
        sys_prompt = (prompt_tpl.replace("{glossary}", it.get("glossary", ""))
                      .replace("{source_language}", it.get("source_language", "zh")))
        try:
            t1 = time.time()
            resp = requests.post(
                f"{args.endpoint}/chat/completions",
                json={"model": model, "messages": [
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": src},
                ], "temperature": args.temperature, "max_tokens": 2048},
                timeout=args.timeout,
            )
            resp.raise_for_status()
            out = resp.json()["choices"][0]["message"]["content"]
            elapsed = time.time() - t1
        except Exception as e:
            print(f"  [{i}] 失敗: {type(e).__name__}: {e}", flush=True)
            rows.append({"error": f"{type(e).__name__}: {e}"})
            continue

        s = score_one(src, out, it.get("terms") or [])
        if args.outputs_out:
            with open(args.outputs_out, "a", encoding="utf-8") as fo:
                fo.write(json.dumps({"i": i, "out": out}, ensure_ascii=False) + "\n")
        s["elapsed_s"] = round(elapsed, 1)
        s["out_chars"] = len(out)
        rows.append(s)
        print(f"  [{i}/{len(items)}] tag={s['tag_keep']:.0%} "
              f"edit={s['edit_ratio']:.2f} {elapsed:.0f}s", flush=True)

    ok = [r for r in rows if "error" not in r]
    n = len(ok) or 1

    def avg(k):
        vals = [r[k] for r in ok if r.get(k) is not None]
        return round(sum(vals) / len(vals), 3) if vals else None

    report = {
        "name": args.name, "model": model, "endpoint": args.endpoint, "temperature": args.temperature,
        "n_items": len(items), "n_failed": len(rows) - len(ok),
        "elapsed_s": round(time.time() - t0, 1),
        "tag_keep_avg": avg("tag_keep"),
        "line_match_avg": avg("line_match"),
        "edit_ratio_avg": avg("edit_ratio"),
        "term_hit_avg": avg("term_hit"),
        "simplified_chars_total": sum(r.get("simplified_chars", 0) for r in ok),
        "cn_term_hits": sorted({t for r in ok for t in r.get("cn_terms", [])}),
        "preamble_leak_count": sum(1 for r in ok if r.get("has_preamble")),
        "avg_latency_s": avg("elapsed_s"),
        "per_item": rows,
    }

    print(f"\n===== {args.name} =====")
    print(f"標籤保留率    : {report['tag_keep_avg']}")
    print(f"行數一致率    : {report['line_match_avg']}")
    print(f"編輯幅度      : {report['edit_ratio_avg']}  （校正任務應偏小）")
    print(f"術語命中率    : {report['term_hit_avg']}")
    print(f"簡體殘留字數  : {report['simplified_chars_total']}")
    print(f"中國用語      : {report['cn_term_hits'] or '無'}")
    print(f"贅述外洩      : {report['preamble_leak_count']} / {len(ok)}")
    print(f"平均延遲      : {report['avg_latency_s']}s  （失敗 {report['n_failed']}）")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=2),
                                  encoding="utf-8")
        print(f"\n完整結果 → {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
