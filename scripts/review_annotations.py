#!/usr/bin/env python
"""標註抽聽工具 — 音檔與逐字稿並排，逐筆記錄人耳判定。

評測集的參考稿若沒人實際聽過，量出來的分數就不可信。本工具讓人用最少的操作抽聽：
左邊播音檔，右邊看逐字稿，回答三個問題後按「儲存並下一筆」。結果累加寫入 TSV，可隨時中斷再續。

用法：
    conda run --no-capture-output -n yayan python scripts/review_annotations.py \\
        --manifest /path/manifest.jsonl --root /path/dataset --out /path/抽聽結果.tsv \\
        [--ids ids.txt] [--port 7862]

manifest 每行一個 JSON，需要的欄位：id、text（逐字稿檔，相對 --root）、
audio（[{"audio": 相對路徑, "channel": null|0|1, "speakers": [...]}, ...]）；
其餘欄位（lang、grade、pairing、needs_listening、unresolved…）有就顯示。

⚠️ 預設只綁 127.0.0.1（音檔與逐字稿不對區域網路公開）。從自己的電腦使用請先建 SSH 通道：
    ssh -L 7862:127.0.0.1:7862 <使用者>@<主機>     然後瀏覽 http://127.0.0.1:7862
若無法建通道，可加 --host 0.0.0.0 --auth 帳號:密碼 直接對區域網路開放（明文 HTTP，用完即關）。
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from datetime import datetime

import gradio as gr

FIELDS = ["id", "語種判定", "實際語種", "文字相符", "另一方有無未抄語音", "備註", "時間"]
Q_LANG = ["與標示相同", "不同", "聽不出來"]
Q_TEXT = ["相符", "大致相符（小出入）", "明顯不符", "無法判斷"]
Q_SIDE = ["沒有（或本來就只有一方）", "有，另一方有講話但沒抄", "不確定"]


def load_results(path: str) -> dict:
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8", newline="") as f:
        return {r["id"]: r for r in csv.DictReader(f, delimiter="\t")}


def save_results(path: str, results: dict) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, delimiter="\t")
        w.writeheader()
        for r in results.values():
            w.writerow({k: r.get(k, "") for k in FIELDS})
    os.replace(tmp, path)


def main() -> int:
    ap = argparse.ArgumentParser(description="標註抽聽工具")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--root", required=True, help="資料集根目錄（manifest 內的相對路徑以此為準）")
    ap.add_argument("--out", required=True, help="抽聽結果 TSV（累加、可續）")
    ap.add_argument("--ids", help="只抽聽這份清單內的編號（一行一個，依清單順序）")
    ap.add_argument("--port", type=int, default=7862)
    ap.add_argument("--host", default="127.0.0.1",
                    help="綁定位址。預設只開本機；要讓區域網路連線請設 0.0.0.0，且必須同時給 --auth")
    ap.add_argument("--auth", help="登入帳密，格式 帳號:密碼（--host 不是本機時必填）")
    args = ap.parse_args()

    auth = None
    if args.auth:
        if ":" not in args.auth:
            print("--auth 格式應為 帳號:密碼"); return 1
        auth = tuple(args.auth.split(":", 1))
    if args.host not in ("127.0.0.1", "localhost") and not auth:
        print("對區域網路開放時必須設定 --auth 帳號:密碼（音檔與逐字稿不可無保護公開）"); return 1

    items = {}
    for line in open(args.manifest, encoding="utf-8"):
        if line.strip():
            r = json.loads(line); items[r["id"]] = r
    order = [l.strip() for l in open(args.ids, encoding="utf-8") if l.strip()] if args.ids else list(items)
    order = [i for i in order if i in items]
    if not order:
        print("沒有可抽聽的項目"); return 1
    root = os.path.abspath(args.root)
    max_audio = max(len(items[i]["audio"]) for i in order)

    def render(pos: int):
        pos = max(0, min(len(order) - 1, int(pos)))
        it = items[order[pos]]
        results = load_results(args.out)
        prev = results.get(it["id"], {})
        text = open(os.path.join(root, it["text"]), encoding="utf-8").read()
        info = [f"### {it['id']}　（第 {pos + 1} / {len(order)} 筆，已完成 {sum(i in results for i in order)} 筆）",
                f"- 標示語種：**{it.get('lang', '?')}**　等級：{it.get('grade', '?')}　類型：{it.get('tx_type', '?')}",
                f"- 文字涵蓋：{it.get('covered', '?')}　配對方式：{it.get('pairing', '?')}"
                + ("　⚠️ 稿內有 [省略] 段" if it.get("has_omission") else "")]
        if it.get("needs_listening"):
            info.append("- **請特別留意**：" + "；".join(it["needs_listening"]))
        if it.get("unresolved"):
            info.append("- 清洗時的疑點：" + "；".join(str(x) for x in it["unresolved"][:4]))
        audio_updates = []
        for k in range(max_audio):
            if k < len(it["audio"]):
                a = it["audio"][k]
                ch = "" if a.get("channel") is None else f"（請聽第 {a['channel'] + 1} 聲道：{'左' if a['channel'] == 0 else '右'}）"
                who = "、".join(a.get("speakers") or []) or "無對應文字"
                audio_updates.append(gr.update(value=os.path.join(root, a["audio"]), visible=True,
                                               label=f"音檔 {k + 1}：{os.path.basename(a['audio'])}　對應說話人 {who} {ch}"))
            else:
                audio_updates.append(gr.update(value=None, visible=False))
        return (pos, "\n".join(info), text, *audio_updates,
                prev.get("語種判定") or None, prev.get("實際語種", ""), prev.get("文字相符") or None,
                prev.get("另一方有無未抄語音") or None, prev.get("備註", ""), "")

    def save(pos, q_lang, real_lang, q_text, q_side, note, step):
        it = items[order[int(pos)]]
        if not (q_lang and q_text and q_side):
            out = list(render(pos)); out[-1] = "⚠️ 三個問題都要回答才能儲存"
            out[-6:-1] = [q_lang, real_lang, q_text, q_side, note]
            return out
        results = load_results(args.out)
        results[it["id"]] = {"id": it["id"], "語種判定": q_lang, "實際語種": real_lang or "", "文字相符": q_text,
                             "另一方有無未抄語音": q_side, "備註": note or "",
                             "時間": datetime.now().strftime("%Y-%m-%d %H:%M")}
        save_results(args.out, results)
        out = list(render(int(pos) + step))
        out[-1] = f"✅ 已儲存 {it['id']}" + ("　🎉 全部完成" if all(i in results for i in order) else "")
        return out

    with gr.Blocks(title="標註抽聽") as demo:
        pos = gr.State(0)
        gr.Markdown("# 🎧 標註抽聽　<small>聽音檔、看逐字稿、回答三個問題</small>")
        with gr.Row():
            with gr.Column(scale=1):
                info = gr.Markdown()
                audios = [gr.Audio(type="filepath", interactive=False, visible=False) for _ in range(max_audio)]
                q_lang = gr.Radio(Q_LANG, label="① 聽起來的語種與標示相同嗎？")
                real_lang = gr.Textbox(label="若不同，實際是什麼語種（可空）")
                q_text = gr.Radio(Q_TEXT, label="② 逐字稿的意思與語音相符嗎？（稿是華語譯文，不必逐字相同）")
                q_side = gr.Radio(Q_SIDE, label="③ 有沒有另一方講了話但稿裡沒抄？")
                note = gr.Textbox(label="備註（聽到的具體問題，可空）", lines=2)
                with gr.Row():
                    b_prev = gr.Button("⬅️ 上一筆（不儲存）")
                    b_save = gr.Button("💾 儲存並下一筆", variant="primary")
                    b_next = gr.Button("跳過 ➡️")
                status = gr.Markdown()
            with gr.Column(scale=1):
                text = gr.Textbox(label="逐字稿", lines=32, interactive=False)
        outs = [pos, info, text, *audios, q_lang, real_lang, q_text, q_side, note, status]
        demo.load(lambda: render(next((k for k, i in enumerate(order) if i not in load_results(args.out)), 0)), outputs=outs)
        b_prev.click(lambda p: render(int(p) - 1), inputs=[pos], outputs=outs)
        b_next.click(lambda p: render(int(p) + 1), inputs=[pos], outputs=outs)
        b_save.click(lambda *a: save(*a, 1), inputs=[pos, q_lang, real_lang, q_text, q_side, note], outputs=outs)

    demo.queue().launch(server_name=args.host, server_port=args.port, share=False,
                        allowed_paths=[root], auth=auth)
    return 0


if __name__ == "__main__":
    sys.exit(main())
