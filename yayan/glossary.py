"""YaYan 術語庫 RAG（v5.0 M3）：特殊字詞 / 同音錯字 / 校正學習。

設計重點：
- 比對：精確（正規化字串）為主 + difflib 模糊為輔，**不依賴 pgvector / pg_trgm extension**。
- 檢索：術語表載入記憶體掃描，O(文本長度)，翻譯熱路徑不查 DB（毫秒級、零感知）。
- 注入：只把「本批文字實際命中」的術語注入 system prompt，並有硬上限避免稀釋。
- 開關 rag.enable_rag 預設 false：不檢索、不注入，翻譯 prompt 與既有逐字相同。

driver = psycopg2，沿用 M2 連線 pattern；DB 同 yayan_voiceprint，新增 glossary_terms 表。
⚠️ 本模組不依賴 pipeline / LLM / ASR，import 不會載入任何模型。
"""
from __future__ import annotations

import logging
import os
import re
from contextlib import closing
from difflib import SequenceMatcher
from typing import Dict, List, Optional, Tuple

import psycopg2

from .config import CONFIG

logger = logging.getLogger("YaYan.Glossary")

# 術語類型與顯示標籤、注入排序權重
TERM_TYPES = ("typo_fix", "proper_noun", "correction")
TYPE_LABEL = {
    "typo_fix": "同音錯字校正",
    "proper_noun": "專有名詞",
    "correction": "校正",
}
TYPE_PRIORITY = {"proper_noun": 30, "typo_fix": 20, "correction": 10}

# 時間戳 / 說話人標籤 [A方 00:01-00:05]，比對前剝除避免誤配到時間
_TAG_RE = re.compile(r"\[[^\]]*\]")
_PUNCT_RE = re.compile(r"^[\s\W_0-9:：，。、,.!?！？～\-]+$")


# ──────────────────────────── 連線 / 設定 ────────────────────────────

def _rag_cfg() -> dict:
    return CONFIG.get("rag", {}) or {}


def _db_cfg() -> dict:
    return _rag_cfg().get("db", {}) or {}


def _password() -> str:
    cfg = _db_cfg()
    env_key = cfg.get("password_env", "")
    if env_key and os.environ.get(env_key):
        return os.environ[env_key]
    return cfg.get("password_default", "")


def _connect():
    cfg = _db_cfg()
    return psycopg2.connect(
        host=cfg["host"],
        port=cfg.get("port", 5432),
        dbname=cfg["dbname"],
        user=cfg["user"],
        password=_password(),
    )


def _norm(s: str) -> str:
    """正規化比對鍵：去頭尾空白、casefold、移除所有空白。"""
    return re.sub(r"\s+", "", (s or "").strip().casefold())


# ──────────────────────────── 初始化 ────────────────────────────

def init_db() -> None:
    """建立 glossary_terms 表 / 索引（IF NOT EXISTS，可重複執行）。不需任何 extension。"""
    with closing(_connect()) as conn, conn.cursor() as cur:
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS glossary_terms (
                id          BIGSERIAL   PRIMARY KEY,
                term        TEXT        NOT NULL,
                term_norm   TEXT        NOT NULL,
                term_type   TEXT        NOT NULL DEFAULT 'proper_noun',
                correct     TEXT        NOT NULL,
                note        TEXT        NOT NULL DEFAULT '',
                source_lang TEXT        NOT NULL DEFAULT 'any',
                priority    INT         NOT NULL DEFAULT 10,
                hit_count   INT         NOT NULL DEFAULT 0,
                source      TEXT        NOT NULL DEFAULT 'manual',
                created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
                updated_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
                UNIQUE (term_norm, term_type, source_lang)
            );
            """
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_glossary_type ON glossary_terms(term_type);"
        )
        conn.commit()
    invalidate_cache()
    logger.info("術語庫初始化完成（glossary_terms）")


# ──────────────────────────── 寫入 / 編輯 ────────────────────────────

def add_term(
    term: str,
    correct: str,
    term_type: str = "proper_noun",
    note: str = "",
    source_lang: str = "any",
    priority: Optional[int] = None,
    source: str = "manual",
) -> int:
    """新增術語；若 (正規化詞, 類型, 語言) 已存在則更新譯法/備註。回傳 id。"""
    term = (term or "").strip()
    correct = (correct or "").strip()
    if not term or not correct:
        raise ValueError("術語與正確譯法皆不可為空。")
    if term_type not in TERM_TYPES:
        raise ValueError(f"未知術語類型：{term_type}")
    tn = _norm(term)
    if priority is None:
        priority = TYPE_PRIORITY.get(term_type, 10)
    with closing(_connect()) as conn, conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO glossary_terms
                (term, term_norm, term_type, correct, note, source_lang, priority, source)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (term_norm, term_type, source_lang) DO UPDATE
                SET term = EXCLUDED.term,
                    correct = EXCLUDED.correct,
                    note = EXCLUDED.note,
                    priority = EXCLUDED.priority,
                    updated_at = now()
            RETURNING id;
            """,
            (term, tn, term_type, correct, note, source_lang or "any", priority, source),
        )
        tid = cur.fetchone()[0]
        conn.commit()
    invalidate_cache()
    return int(tid)


def update_term(
    term_id: int,
    term: Optional[str] = None,
    correct: Optional[str] = None,
    term_type: Optional[str] = None,
    note: Optional[str] = None,
    source_lang: Optional[str] = None,
    priority: Optional[int] = None,
) -> bool:
    sets, vals = [], []
    if term is not None:
        t = term.strip()
        sets += ["term = %s", "term_norm = %s"]
        vals += [t, _norm(t)]
    if correct is not None:
        sets.append("correct = %s"); vals.append(correct.strip())
    if term_type is not None:
        if term_type not in TERM_TYPES:
            raise ValueError(f"未知術語類型：{term_type}")
        sets.append("term_type = %s"); vals.append(term_type)
    if note is not None:
        sets.append("note = %s"); vals.append(note)
    if source_lang is not None:
        sets.append("source_lang = %s"); vals.append(source_lang or "any")
    if priority is not None:
        sets.append("priority = %s"); vals.append(int(priority))
    if not sets:
        return False
    sets.append("updated_at = now()")
    vals.append(int(term_id))
    with closing(_connect()) as conn, conn.cursor() as cur:
        cur.execute(
            f"UPDATE glossary_terms SET {', '.join(sets)} WHERE id = %s;", vals
        )
        changed = cur.rowcount
        conn.commit()
    invalidate_cache()
    return changed > 0


def delete_term(term_id: int) -> bool:
    with closing(_connect()) as conn, conn.cursor() as cur:
        cur.execute("DELETE FROM glossary_terms WHERE id = %s;", (int(term_id),))
        changed = cur.rowcount
        conn.commit()
    invalidate_cache()
    return changed > 0


def get_term(term_id: int) -> Optional[Dict]:
    with closing(_connect()) as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT id, term, term_type, correct, note, source_lang, priority, hit_count "
            "FROM glossary_terms WHERE id = %s;",
            (int(term_id),),
        )
        r = cur.fetchone()
    if not r:
        return None
    return {
        "id": r[0], "term": r[1], "term_type": r[2], "correct": r[3],
        "note": r[4], "source_lang": r[5], "priority": r[6], "hit_count": r[7],
    }


def bulk_import(rows: List[Tuple]) -> Tuple[int, int, List[str]]:
    """批次匯入。rows = [(term, correct, term_type, note, source_lang), ...]（後三項可省）。

    回傳 (成功筆數, 略過筆數, 錯誤訊息清單)。
    """
    ok, skip, errs = 0, 0, []
    for idx, row in enumerate(rows, 1):
        try:
            term = (row[0] if len(row) > 0 else "").strip()
            correct = (row[1] if len(row) > 1 else "").strip()
            ttype = (row[2].strip() if len(row) > 2 and row[2] else "proper_noun")
            note = (row[3].strip() if len(row) > 3 and row[3] else "")
            slang = (row[4].strip() if len(row) > 4 and row[4] else "any")
            if not term or not correct:
                skip += 1
                continue
            if ttype not in TERM_TYPES:
                ttype = "proper_noun"
            add_term(term, correct, ttype, note, slang, source="import")
            ok += 1
        except Exception as e:  # 單列失敗不中斷整批
            errs.append(f"第 {idx} 列：{e}")
    invalidate_cache()
    return ok, skip, errs


# ──────────────────────────── 查詢 / 分頁（管理頁籤用）────────────────────────────

def list_terms(
    page: int = 1, page_size: int = 20, keyword: str = "", term_type: str = ""
) -> Tuple[List[Dict], int]:
    """分頁列表（關鍵字 ILIKE term/correct/note；可選類型過濾）。用 ILIKE，不依賴 trgm。"""
    page = max(1, int(page)); page_size = max(1, int(page_size))
    offset = (page - 1) * page_size
    kw = (keyword or "").strip()
    where, params = [], []
    if kw:
        where.append("(term ILIKE %s OR correct ILIKE %s OR note ILIKE %s)")
        like = f"%{kw}%"; params += [like, like, like]
    if term_type in TERM_TYPES:
        where.append("term_type = %s"); params.append(term_type)
    clause = ("WHERE " + " AND ".join(where)) if where else ""
    with closing(_connect()) as conn, conn.cursor() as cur:
        cur.execute(f"SELECT count(*) FROM glossary_terms {clause};", params)
        total = cur.fetchone()[0]
        cur.execute(
            f"SELECT id, term, term_type, correct, note, source_lang, priority, hit_count "
            f"FROM glossary_terms {clause} ORDER BY updated_at DESC LIMIT %s OFFSET %s;",
            params + [page_size, offset],
        )
        rows = cur.fetchall()
    terms = [
        {
            "id": r[0], "term": r[1], "term_type": r[2], "correct": r[3],
            "note": r[4], "source_lang": r[5], "priority": r[6], "hit_count": r[7],
        }
        for r in rows
    ]
    return terms, int(total)


def count_terms() -> int:
    with closing(_connect()) as conn, conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM glossary_terms;")
        return int(cur.fetchone()[0])


# ──────────────────────────── 記憶體快取 + 檢索（翻譯熱路徑）────────────────────────────

_CACHE: Optional[List[Dict]] = None


def invalidate_cache() -> None:
    """術語表異動後呼叫，下次 lookup 會重載。"""
    global _CACHE
    _CACHE = None


def _load_cache() -> List[Dict]:
    global _CACHE
    if _CACHE is not None:
        return _CACHE
    rows: List[Dict] = []
    try:
        with closing(_connect()) as conn, conn.cursor() as cur:
            cur.execute(
                "SELECT id, term, term_norm, term_type, correct, note, source_lang, "
                "priority, hit_count FROM glossary_terms;"
            )
            for r in cur.fetchall():
                rows.append({
                    "id": r[0], "term": r[1], "term_norm": r[2], "term_type": r[3],
                    "correct": r[4], "note": r[5], "source_lang": r[6],
                    "priority": r[7], "hit_count": r[8],
                })
    except Exception as e:
        logger.warning("術語快取載入失敗（視為空庫，不影響翻譯）：%s", e)
        rows = []
    _CACHE = rows
    return rows


def _lang_ok(term_lang: str, source_lang: str) -> bool:
    tl = (term_lang or "any").strip()
    if tl == "any":
        return True
    return tl in (source_lang or "")


def _best_window_ratio(text: str, term: str) -> float:
    """在 text 中以 len(term) 視窗滑動，回傳最大 difflib 相似度。含字元交集預過濾。"""
    L = len(term)
    if L == 0:
        return 0.0
    if len(text) < L:
        return SequenceMatcher(None, text, term).ratio() if text else 0.0
    tset = set(term)
    best = 0.0
    sm = SequenceMatcher()
    sm.set_seq2(term)
    for i in range(len(text) - L + 1):
        w = text[i:i + L]
        if not (set(w) & tset):  # 無共同字元，跳過
            continue
        sm.set_seq1(w)
        r = sm.ratio()
        if r > best:
            best = r
            if best >= 0.999:
                break
    return best


def _best_window(text: str, term: str) -> Tuple[float, str]:
    """同 _best_window_ratio，但一併回傳命中的原文片段（供提示顯示用）。"""
    L = len(term)
    if L == 0 or len(text) < L:
        return 0.0, ""
    tset = set(term)
    best, best_w, best_aligned = 0.0, "", -1
    sm = SequenceMatcher()
    sm.set_seq2(term)
    for i in range(len(text) - L + 1):
        w = text[i:i + L]
        if not (set(w) & tset):
            continue
        sm.set_seq1(w)
        r = sm.ratio()
        # 同分時用「逐位對齊字數」決勝：「藍海濤」對上「藍海滔」與「是藍海」的
        # difflib 相似度都是 0.67，但前者前兩字同位相同、後者完全錯位。
        # 不做這個決勝會提示出「文中的『是藍海』可能是『藍海濤』」這種看不懂的話。
        aligned = sum(1 for a, b in zip(w, term) if a == b)
        if r > best or (r == best and aligned > best_aligned):
            best, best_w, best_aligned = r, w, aligned
            if best >= 0.999:
                break
    return best, best_w


def name_hints(text: str, source_lang: str = "any") -> List[Dict]:
    """找出文中「疑似專名誤植」的片段（開關 rag.enable_name_fuzzy_hint，預設關）。

    為什麼要獨立一套門檻：三字人名差一個字的相似度是 0.67，遠低於一般術語用的
    fuzzy_threshold（0.86），所以人名的誤植永遠比對不到。這裡只對 proper_noun
    套用較鬆的 name_fuzzy_threshold，不動一般術語的門檻。

    ⚠️ 放鬆門檻的代價是會把「不同的人」配在一起——「王建華」與「王建國」相似度
    同樣是 0.67。因此本函式**只產生『可能是』的提示交給 LLM 依語境判斷**，
    絕不做替換，也不進 correction_pairs。人員身分的最終判斷仍在人。
    """
    cfg = _rag_cfg()
    if not cfg.get("enable_name_fuzzy_hint", False) or not text:
        return []
    th = float(cfg.get("name_fuzzy_threshold", 0.60))
    limit = int(cfg.get("max_name_hints", 10))

    clean = _TAG_RE.sub(" ", text)
    norm_text = _norm(clean)
    out: List[Dict] = []
    seen: set = set()
    for t in _load_cache():
        if (t.get("term_type") or "") != "proper_noun":
            continue
        if not _lang_ok(t.get("source_lang"), source_lang):
            continue
        term = (t.get("term") or "").strip()
        canonical = (t.get("correct") or term).strip()
        if len(term) < 2:
            continue
        if term in clean:
            continue                      # 已經是正確寫法，不需要提示
        r, w = _best_window(norm_text, _norm(term))
        if r < th or r >= 0.999 or not w or w in seen:
            continue
        seen.add(w)
        out.append({"found": w, "canonical": canonical, "score": round(r, 2)})
    out.sort(key=lambda h: -h["score"])
    return out[:limit]


def format_name_hints(hints: List[Dict]) -> str:
    if not hints:
        return ""
    lines = ["【疑似專名誤植，請依語境判斷是否為同一對象；不確定就保留原文】"]
    for h in hints:
        lines.append(
            f"- 文中的「{h['found']}」可能是「{h['canonical']}」（相似度 {h['score']}）"
        )
    return "\n".join(lines) + "\n\n"


def lookup(text: str, source_lang: str = "any") -> List[Dict]:
    """檢索本段文字命中的術語，已排序並套用注入上限。開關關閉時呼叫端不應呼叫本函式。"""
    if not text:
        return []
    cfg = _rag_cfg()
    max_terms = int(cfg.get("max_inject_terms", 20))
    enable_fuzzy = bool(cfg.get("enable_fuzzy", True))
    fuzzy_threshold = float(cfg.get("fuzzy_threshold", 0.86))
    fuzzy_min_len = int(cfg.get("fuzzy_min_len", 2))

    clean = _TAG_RE.sub(" ", text)        # 去時間戳/說話人標籤
    norm_text = _norm(clean)
    hits: List[Dict] = []
    for t in _load_cache():
        if not _lang_ok(t["source_lang"], source_lang):
            continue
        term, tn = t["term"], t["term_norm"]
        mode, score = None, 1.0
        if term and term in clean:
            mode = "exact"
        elif tn and tn in norm_text:
            mode = "exact"
        elif enable_fuzzy and len(tn) >= fuzzy_min_len:
            r = _best_window_ratio(norm_text, tn)
            if r >= fuzzy_threshold:
                mode, score = "fuzzy", r
        if mode:
            hits.append({**t, "match_mode": mode, "match_score": round(score, 3)})
    # 精確優先 → priority 高優先 → 長詞優先 → hit_count 高優先
    hits.sort(key=lambda h: (
        h["match_mode"] == "fuzzy", -h["priority"], -len(h["term"]), -h["hit_count"]
    ))
    return hits[:max_terms]


def format_glossary_block(hits: List[Dict]) -> str:
    """把命中術語格式化為注入 system prompt 的區塊；空則回傳空字串（prompt 不變）。"""
    if not hits:
        return ""
    lines = ["【術語對照表，翻譯時務必遵守】"]
    for h in hits:
        note = f"（{h['note']}）" if h.get("note") else ""
        if h["term_type"] == "proper_noun":
            # term 與 correct 相同的專名（例如語者姓名）產生的是「X 一律譯為 X」，
            # 對 LLM 沒有任何指示作用，只會吃掉 max_inject_terms 的額度。
            if (h.get("term") or "").strip() == (h.get("correct") or "").strip():
                continue
            lines.append(f"- 「{h['term']}」一律譯為「{h['correct']}」{note}")
        else:
            label = TYPE_LABEL.get(h["term_type"], "校正")
            lines.append(f"- 「{h['term']}」實際應為「{h['correct']}」（{label}）{note}")
    return "\n".join(lines) + "\n\n"


def glossary_for_text(text: str, source_lang: str = "any") -> str:
    """便捷組合：檢索 + 格式化，回傳可直接填入 {glossary} 的字串。

    含兩段：術語對照表（確定的譯法）＋疑似專名誤植提示（不確定，交 LLM 判斷）。
    後者需 rag.enable_name_fuzzy_hint 開啟，預設關＝與既有行為逐字相同。
    """
    block = format_glossary_block(lookup(text, source_lang))
    try:
        block += format_name_hints(name_hints(text, source_lang))
    except Exception as e:
        logger.warning("專名提示產生失敗，略過：%s", e)
    return block


# ─────────────────────── ASR 後處理：確定性術語替換 ───────────────────────

_CORRECTION_TYPES = ("typo_fix", "correction")


def correction_pairs(source_lang: str = "any") -> List[Tuple[str, str]]:
    """取「錯字 → 正確字」對照表，長詞優先。

    只用 typo_fix / correction：proper_noun 的 term 本身就是正確寫法，
    term→correct 等於 no-op，放進來只會拖慢替換。
    """
    try:
        rows = _load_cache()
    except Exception as e:
        logger.warning("術語校正表載入失敗（略過後處理）：%s", e)
        return []

    all_langs = (source_lang or "any").strip().lower() in ("", "any", "auto")
    pairs: List[Tuple[str, str]] = []
    seen: set = set()
    for r in rows:
        if (r.get("term_type") or "") not in _CORRECTION_TYPES:
            continue
        if not all_langs and not _lang_ok(r.get("source_lang"), source_lang):
            continue
        wrong = _TAG_RE.sub("", (r.get("term") or "")).strip()
        right = _TAG_RE.sub("", (r.get("correct") or "")).strip()
        if not wrong or not right or wrong == right:
            continue
        if wrong in seen:
            continue
        seen.add(wrong)
        pairs.append((wrong, right))

    # 長詞優先：否則短詞會先吃掉長詞的一部分（「文超」先替換掉「陳文超」的後半）
    pairs.sort(key=lambda t: -len(t[0]))
    return pairs


def apply_corrections(text: str, source_lang: str = "any") -> Tuple[str, int]:
    """把術語庫的錯字對照直接套用在 ASR 逐字稿上，回傳 (新文字, 替換次數)。

    這是 hotword 偏置的替代方案：偏置要看模型臉色（實測 dolphin-small 上無效），
    字串替換則是確定性的——只要詞條建得對，一定會被修正。

    用單次 regex alternation 掃描，不做連鎖替換：若同時有 A→B 與 B→C 兩條規則，
    A 只會變成 B，不會再被推成 C（避免規則互相串接產生意料外結果）。
    """
    if not text:
        return text, 0
    pairs = correction_pairs(source_lang)
    if not pairs:
        return text, 0

    mapping = {w: r for w, r in pairs}
    pattern = re.compile("|".join(re.escape(w) for w, _ in pairs))
    count = 0

    def _sub(mo):
        nonlocal count
        count += 1
        return mapping[mo.group(0)]

    return pattern.sub(_sub, text), count


# ──────────────────────────── ASR hotword 偏置 ────────────────────────────

def hotwords_for_asr(source_lang: str = "any", limit: Optional[int] = None) -> List[str]:
    """取術語庫詞條，供 ASR 解碼階段做 hotword 偏置。

    與 lookup() 的差別：lookup 是拿「已經產生的逐字稿」去比對命中；這裡是逐字稿
    還不存在的階段，沒有東西可比對，所以不做檢索，直接把整份詞表依優先度截斷後
    交給解碼器偏置。

    回傳的是**希望 ASR 輸出的字面**——correct 優先，缺才退回 term
    （typo_fix 的 term 是錯字，拿去偏置等於教模型犯錯）。

    任何失敗都回空 list：hotword 是加分項，絕不能讓它擋掉轉錄。
    """
    if limit is None:
        try:
            limit = int(_rag_cfg().get("max_asr_hotwords", 50) or 50)
        except (TypeError, ValueError):
            limit = 50
    if limit <= 0:
        return []

    try:
        rows = _load_cache()
    except Exception as e:  # _load_cache 自己已吞例外，這裡是雙保險
        logger.warning("ASR hotword 取詞失敗（本次不做偏置）：%s", e)
        return []

    # source_lang 未知/auto 時不做語言過濾——寧可多給幾個詞，也不要漏掉該偏置的
    all_langs = (source_lang or "any").strip().lower() in ("", "any", "auto")

    scored: List[Tuple[int, str]] = []
    for r in rows:
        if not all_langs and not _lang_ok(r.get("source_lang"), source_lang):
            continue
        w = (r.get("correct") or r.get("term") or "").strip()
        w = _TAG_RE.sub("", w).strip()
        if not w or _PUNCT_RE.match(w):
            continue
        if len(w) > 20:
            # 過長的是整句改寫型 correction，不是詞彙，拿去偏置只會稀釋效果
            continue
        rank = TYPE_PRIORITY.get(r.get("term_type") or "", 0) + int(r.get("priority") or 0)
        scored.append((rank, w))

    scored.sort(key=lambda t: -t[0])
    out: List[str] = []
    seen: set = set()
    for _, w in scored:
        if w in seen:
            continue
        seen.add(w)
        out.append(w)
        if len(out) >= limit:
            break
    return out


# ──────────────────────────── 校正學習（半自動）────────────────────────────

def _good_pair(old: str, new: str) -> bool:
    old, new = old.strip(), new.strip()
    if not old or not new or old == new:
        return False
    if len(old) > 12 or len(new) > 12:        # 過長視為整句改寫，非術語
        return False
    if _PUNCT_RE.match(old) or _PUNCT_RE.match(new):  # 純標點/數字/時間
        return False
    return True


def extract_corrections(
    raw_text: str, user_edit: str, max_candidates: int = 30
) -> List[Dict]:
    """比對原譯與使用者改後文字，抽出候選詞對（供 UI 勾選確認後才入庫）。

    回傳 [{'wrong': 原詞, 'correct': 改後}, ...]，已去重。
    """
    def strip_tag(line: str) -> str:
        return _TAG_RE.sub("", line).strip()

    raw_lines = [strip_tag(l) for l in (raw_text or "").split("\n")]
    new_lines = [strip_tag(l) for l in (user_edit or "").split("\n")]

    cands: List[Dict] = []
    seen = set()
    char_sm = SequenceMatcher()
    line_sm = SequenceMatcher(None, raw_lines, new_lines)
    for tag, i1, i2, j1, j2 in line_sm.get_opcodes():
        if tag != "replace":
            continue
        for k in range(max(i2 - i1, j2 - j1)):
            a = raw_lines[i1 + k] if i1 + k < i2 else ""
            b = new_lines[j1 + k] if j1 + k < j2 else ""
            if not a or not b or a == b:
                continue
            char_sm.set_seqs(a, b)
            for op, a1, a2, b1, b2 in char_sm.get_opcodes():
                if op != "replace":
                    continue
                old, new = a[a1:a2], b[b1:b2]
                if _good_pair(old, new) and (old, new) not in seen:
                    seen.add((old, new))
                    cands.append({"wrong": old.strip(), "correct": new.strip()})
                    if len(cands) >= max_candidates:
                        return cands
    return cands
