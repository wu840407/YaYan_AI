#!/usr/bin/env python
"""YaYan 評測腳本 — 吃「評測集 manifest」吐 CER / LID 混淆矩陣 / 時間戳誤差。

這是所有模型升級決策的對照組產生器：任何「新模型比較好」的說法，
都要能在同一份 manifest 上跑出更好的數字才算數。

用法：
    conda run --no-capture-output -n yayan python scripts/evaluate_asr.py \\
        --manifest /path/to/manifest.jsonl \\
        --language auto \\
        --out results/$(date +%F)-eval.json

manifest 格式（每行一個 JSON）：
    {"audio": "/abs/path.wav",       # 必填
     "text": "參考逐字稿",             # 算 CER 用；沒有就只跑 LID
     "dialect": "nan",               # 人工方言標籤（routing key）；沒有就不做混淆矩陣
     "words": [{"start": 0.5, "end": 0.7, "text": "你"}],   # 選填，算時間戳誤差
     "group": "cmn-fleurs"}          # 選填，報表分組名稱（沒有就用 dialect）

⚠️ 標籤品質決定一切：曾有樣本的實際內容與檔名標示的語言不符，差點因此得出錯誤結論。
   用新評測集前先用 scripts/review_annotations.py 抽聽。

評測用選項（皆選填；不加時行為與舊版相同）：
    --set asr.lid_backend=voxlingua   只在本行程內覆寫設定，不改設定檔（可重複）
    --skip-llm                        不呼叫 LLM（純 ASR 評測，不打正式翻譯服務）
    --t2s                             參考與辨識結果都先轉簡體再比（消除簡繁差異）
    --corrected                       另算 LLM 校正後文字的 CER（需呼叫 LLM）
    --hyps-out x.jsonl                逐筆輸出辨識結果，供事後重算
    --gpu-cap-gb / --max-other-mib / --busy-log  與正式服務共卡時的保護
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
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


_LINE_PREFIX = re.compile(r"^\s*\[[^\]]*\]\s*", re.M)


def strip_line_prefix(text: str) -> str:
    """去掉每行開頭的「[A方 00:01-00:05]」或「[00:01-00:05]」前綴。"""
    return _LINE_PREFIX.sub("", text or "")


def _parse_set(items: list) -> list:
    """'asr.lid_backend=voxlingua' → (['asr','lid_backend'], 'voxlingua')；值用 YAML 解析。"""
    import yaml
    out = []
    for it in items or []:
        if "=" not in it:
            raise SystemExit(f"--set 格式錯誤（要 a.b=值）：{it}")
        k, v = it.split("=", 1)
        out.append((k.strip().split("."), yaml.safe_load(v)))
    return out


def _apply_set(config: dict, pairs: list) -> None:
    for keys, val in pairs:
        d = config
        for k in keys[:-1]:
            if k not in d or not isinstance(d[k], dict):
                raise SystemExit(f"--set 找不到設定區段：{'.'.join(keys)}")
            d = d[k]
        if keys[-1] not in d:
            raise SystemExit(f"--set 設定檔裡沒有這個鍵（避免打錯字靜默無效）：{'.'.join(keys)}")
        d[keys[-1]] = val


def _gpu_other_mib(index: int = 0) -> int:
    """這張卡上「其他行程」（正式服務等）用了多少記憶體；不含本行程自己。

    不能用整張卡的剩餘量判斷：本行程自己載的模型也算在 used 裡，而且等待時不會釋放，
    一旦開始等就永遠恢復不了（自我死鎖）。
    """
    try:
        out = subprocess.run(
            ["nvidia-smi", f"--id={index}", "--query-compute-apps=pid,used_memory",
             "--format=csv,noheader,nounits"], capture_output=True, text=True, check=True).stdout
        me = os.getpid()
        tot = 0
        for line in out.strip().splitlines():
            pid, mem = (x.strip() for x in line.split(","))
            if pid.isdigit() and int(pid) != me:
                tot += int(mem)
        return tot
    except Exception:
        return 0


def _wait_until_safe(args) -> None:
    """與正式服務共卡時：GPU 剩餘不足、或正式服務 log 剛有動靜，就等。"""
    waited = 0
    while True:
        reasons = []
        if args.max_other_mib:
            other = _gpu_other_mib(args.gpu_index)
            if other > args.max_other_mib:
                reasons.append(f"其他行程用了 {other} MiB > {args.max_other_mib}")
        for p in args.busy_log or []:
            try:
                age = time.time() - os.path.getmtime(p)
                if age < args.busy_quiet_sec:
                    reasons.append(f"{os.path.basename(p)} {age:.0f} 秒前有動靜")
            except OSError:
                pass
        if not reasons:
            if waited:
                print(f"  恢復（等了 {waited} 秒）", flush=True)
            return
        if waited == 0:
            try:   # 暫停期間把本行程沒在用的快取還給正式服務
                import torch
                torch.cuda.empty_cache()
            except Exception:
                pass
        if waited % 300 == 0:
            print(f"  暫停：{'；'.join(reasons)}", flush=True)
        time.sleep(30)
        waited += 30


def _audio_sec(path: str) -> float:
    try:
        import soundfile as sf
        return float(sf.info(path).duration)
    except Exception:
        return 0.0


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
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VAL",
                    help="只在本行程覆寫設定，例：--set asr.lid_backend=voxlingua（可重複）")
    ap.add_argument("--skip-llm", action="store_true",
                    help="不呼叫 LLM 翻譯／校正（純 ASR 評測）")
    ap.add_argument("--t2s", action="store_true",
                    help="參考與辨識結果都先用 OpenCC 轉簡體再比，消除簡繁差異")
    ap.add_argument("--corrected", action="store_true",
                    help="另算 LLM 校正後文字（translated_text）的 CER")
    ap.add_argument("--hyps-out", help="逐筆結果寫成 jsonl")
    ap.add_argument("--gpu-cap-gb", type=float, default=0,
                    help="本行程在 ASR GPU 上的記憶體上限（0＝不設）")
    ap.add_argument("--gpu-index", type=int, default=0,
                    help="--max-other-mib 監看哪張卡（nvidia-smi 編號）")
    ap.add_argument("--max-other-mib", type=int, default=0,
                    help="同卡其他行程（正式服務）用量超過此值就暫停等待（表示它正在工作）")
    ap.add_argument("--busy-log", action="append", default=[],
                    help="正式服務的 log 檔；最近有寫入就暫停（可重複）")
    ap.add_argument("--busy-quiet-sec", type=int, default=180,
                    help="--busy-log 要安靜多久才繼續（秒）")
    args = ap.parse_args()
    if args.corrected and args.skip_llm:
        print("--corrected 需要 LLM，不能和 --skip-llm 一起用", file=sys.stderr)
        return 2

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

    set_pairs = _parse_set(args.set)
    from yayan.config import CONFIG
    _apply_set(CONFIG, set_pairs)

    if args.gpu_cap_gb:
        import torch
        dev = torch.device(CONFIG.get("devices", {}).get("asr_gpu", "cuda:0"))
        total = torch.cuda.get_device_properties(dev).total_memory / 2 ** 30
        torch.cuda.set_per_process_memory_fraction(min(1.0, args.gpu_cap_gb / total), device=dev)

    from yayan import pipeline

    if args.skip_llm:
        def _no_llm(text_lines, source_language="", **kw):
            return "\n".join(text_lines)
        pipeline._batched_translate = _no_llm
        pipeline._quality_hint = lambda *a, **k: ""   # 這條也會呼叫 LLM

    t2s = None
    if args.t2s:
        import opencc
        t2s = opencc.OpenCC("t2s").convert

    def norm(text: str, strip_rom: bool) -> str:
        t = normalize(text, strip_romanization=strip_rom)
        return t2s(t) if t2s else t

    backend = CONFIG["asr"].get("lid_backend", "voxlingua")
    overrides = {".".join(k): v for k, v in set_pairs}
    print(f"評測 {len(items)} 筆 | language={args.language} | lid_backend={backend}"
          f" | skip_llm={args.skip_llm} | t2s={args.t2s} | 覆寫={overrides}", flush=True)

    tot_ref = tot_sub = tot_del = tot_ins = 0
    per_dialect = defaultdict(lambda: {"ref": 0, "sub": 0, "del": 0, "ins": 0, "n": 0})
    per_group = defaultdict(lambda: {"ref": 0, "sub": 0, "del": 0, "ins": 0, "n": 0,
                                     "c_ref": 0, "c_err": 0})
    tot_c_ref = tot_c_err = 0
    confusion = defaultdict(Counter)
    ts_stats = []
    failures = []
    audio_sec_total = proc_sec_total = 0.0   # 不含第一筆（含模型載入）
    seg_routings = Counter()
    seen_engines: set = set()   # 模型用到才載：第一次用到某引擎的那筆不算 RTF
    engine_failed = []   # pipeline 吞掉的引擎失敗（OOM 等）：該段會變空字串，CER 會被灌高
    hyp_f = open(args.hyps_out, "w", encoding="utf-8") if args.hyps_out else None
    t0 = time.time()

    for idx, it in enumerate(items, 1):
        audio = it["audio"]
        if args.max_other_mib or args.busy_log:
            _wait_until_safe(args)
        t_item = time.time()
        try:
            res = pipeline.transcribe_audio(
                audio, language=args.language,
                use_diarize=False if args.no_diarize else None,
            )
        except Exception as e:
            failures.append({"audio": audio, "error": f"{type(e).__name__}: {e}"})
            continue
        dt_item = time.time() - t_item
        dur = _audio_sec(audio)
        engines = {getattr(s, "asr_alias", "") for s in res.segments}
        cold = bool(engines - seen_engines) or not seen_engines
        seen_engines |= engines
        if not cold and dur > 0:
            audio_sec_total += dur
            proc_sec_total += dt_item
        for s in res.segments:
            seg_routings[getattr(s, "routing", "") or "?"] += 1
        n_eng_fail = int((getattr(res, "stats", None) or {}).get("n_failed") or 0)
        if n_eng_fail:
            engine_failed.append(audio)

        truth = it.get("dialect")
        if truth:
            confusion[truth][res.routing] += 1

        # ⚠️ 用逐段的 raw_text 拼，不要用 res.raw_text——後者含
        # 「[A方 00:01-00:05]」前綴，正規化剝掉標點後會變成 000105 混進 CER。
        hyp_text = "".join(s.raw_text or "" for s in res.segments) or res.raw_text

        ref_raw = it.get("text")
        rec = {"audio": audio, "group": it.get("group") or truth, "dialect": truth,
               "routing": res.routing,
               "seg_routings": [getattr(s, "routing", "") for s in res.segments],
               "hyp": hyp_text, "audio_sec": round(dur, 2), "proc_sec": round(dt_item, 2),
               "engine_failed_segments": n_eng_fail}
        if ref_raw:
            ref = norm(ref_raw, not args.keep_romanization)
            hyp = norm(hyp_text, False)
            sub, dele, ins = edit_distance(ref, hyp)
            tot_ref += len(ref); tot_sub += sub; tot_del += dele; tot_ins += ins
            if truth:
                d = per_dialect[truth]
                d["ref"] += len(ref); d["sub"] += sub
                d["del"] += dele; d["ins"] += ins; d["n"] += 1
            g = per_group[it.get("group") or truth or "-"]
            g["ref"] += len(ref); g["sub"] += sub; g["del"] += dele; g["ins"] += ins; g["n"] += 1
            rec.update(ref=ref_raw, ref_norm=ref, hyp_norm=hyp, errors=sub + dele + ins)
            if args.corrected:
                corr_text = strip_line_prefix(res.translated_text)
                corr = norm(corr_text, False)
                c_err = sum(edit_distance(ref, corr))
                tot_c_ref += len(ref); tot_c_err += c_err
                g["c_ref"] += len(ref); g["c_err"] += c_err
                rec.update(hyp_corrected=corr_text, errors_corrected=c_err)
        if hyp_f:
            hyp_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            hyp_f.flush()

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

    if hyp_f:
        hyp_f.close()

    gpu_peak = None
    try:
        import torch
        if torch.cuda.is_available():
            gpu_peak = {f"cuda:{i}": round(torch.cuda.max_memory_reserved(i) / 2 ** 20)
                        for i in range(torch.cuda.device_count())}
    except Exception:
        pass

    report = {
        "manifest": args.manifest,
        "language": args.language,
        "lid_backend": backend,
        "overrides": overrides,
        "skip_llm": args.skip_llm,
        "t2s": args.t2s,
        "no_diarize": args.no_diarize,
        "n_items": len(items),
        "n_failed": len(failures),
        "n_items_with_engine_failure": len(engine_failed),
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
        "per_group_cer_pct": {
            k: {"cer_pct": cer(v["ref"], v["sub"], v["del"], v["ins"]),
                "n_items": v["n"], "ref_chars": v["ref"],
                **({"corrected_cer_pct": cer(v["c_ref"], v["c_err"], 0, 0)} if v["c_ref"] else {})}
            for k, v in sorted(per_group.items())
        },
        "overall_corrected_cer_pct": cer(tot_c_ref, tot_c_err, 0, 0) if tot_c_ref else None,
        "segment_routings": dict(seg_routings.most_common()),
        "rtf_excl_cold": (round(proc_sec_total / audio_sec_total, 3) if audio_sec_total else None),
        "audio_sec_excl_cold": round(audio_sec_total, 1),
        "gpu_peak_reserved_mib_this_process": gpu_peak,
        "timestamp": ({
            "n_files": len(ts_stats),
            "mean_abs_err_s": round(sum(t["mean_abs_err_s"] for t in ts_stats) / len(ts_stats), 3),
        } if ts_stats else None),
        "failures": failures[:20],
    }

    print("\n===== 結果 =====")
    print(f"整體 CER: {report['overall_cer_pct']}%   （失敗 {len(failures)} 筆）")
    if engine_failed:
        print(f"⚠️ {len(engine_failed)} 筆有引擎失敗（OOM 等），該段為空字串、CER 偏高；"
              f"逐筆見 --hyps-out 的 engine_failed_segments")
    if report["overall_corrected_cer_pct"] is not None:
        print(f"校正後 CER: {report['overall_corrected_cer_pct']}%")
    if report["per_group_cer_pct"]:
        print("\n逐組 CER:")
        for k, v in report["per_group_cer_pct"].items():
            extra = f"  校正後 {v['corrected_cer_pct']}%" if "corrected_cer_pct" in v else ""
            print(f"  {k:<14} {v['cer_pct']:>6}%   ({v['n_items']} 筆 / {v['ref_chars']} 字){extra}")
    print(f"RTF（不含載入模型的那幾筆）: {report['rtf_excl_cold']}")
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
