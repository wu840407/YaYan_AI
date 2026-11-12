"""YaYan 聲紋資料庫（v5.0 M2）：PostgreSQL + pgvector。

driver = psycopg2（依賴最少）。向量以字串 '[..]' + ::vector 轉型存取，
不依賴 pgvector-python。相似度搜尋走 HNSW + cosine 距離。

設計（雙軌 centroid）：
  - speakers：每人一筆，embedding = 該人所有樣本向量的平均（centroid）。
  - speaker_samples：保留每段原始向量，加樣本時用 AVG 重算 centroid。
  - cosine 距離對向量長度不變，故 centroid 不必額外正規化即可正確搜尋。
"""
from __future__ import annotations

import logging
import os
from contextlib import closing
from typing import Dict, List, Optional, Tuple

import numpy as np
import psycopg2

from .config import CONFIG

logger = logging.getLogger("YaYan.SpeakerDB")


# ──────────────────────────── 連線 / 設定 ────────────────────────────

def _sid_cfg() -> dict:
    return CONFIG["speaker_id"]


def _db_cfg() -> dict:
    return _sid_cfg()["db"]


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


def _dim() -> int:
    return int(_sid_cfg().get("embedding_dim", 256))


def _vec_to_str(vec) -> str:
    """np 向量 → pgvector 文字字面值 '[x,y,...]'。"""
    arr = np.asarray(vec, dtype=np.float32).reshape(-1)
    return "[" + ",".join(f"{x:.8f}" for x in arr) + "]"


# ──────────────────────────── 初始化 ────────────────────────────

def init_db() -> None:
    """建立 extension / 表 / 索引（全部 IF NOT EXISTS，可重複執行不報錯）。"""
    dim = _dim()
    with closing(_connect()) as conn, conn.cursor() as cur:
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        cur.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm;")

        cur.execute(
            f"""
            CREATE TABLE IF NOT EXISTS speakers (
                id           BIGSERIAL PRIMARY KEY,
                name         TEXT          NOT NULL,
                note         TEXT          DEFAULT '',
                embedding    vector({dim}) NOT NULL,
                sample_count INT           NOT NULL DEFAULT 1,
                created_at   TIMESTAMPTZ   NOT NULL DEFAULT now(),
                updated_at   TIMESTAMPTZ   NOT NULL DEFAULT now()
            );
            """
        )
        cur.execute(
            f"""
            CREATE TABLE IF NOT EXISTS speaker_samples (
                id          BIGSERIAL PRIMARY KEY,
                speaker_id  BIGINT        NOT NULL REFERENCES speakers(id) ON DELETE CASCADE,
                embedding   vector({dim}) NOT NULL,
                source      TEXT          DEFAULT '',
                created_at  TIMESTAMPTZ   NOT NULL DEFAULT now()
            );
            """
        )

        # 姓名/代號關鍵字搜尋（trigram，模糊 ILIKE）
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_speakers_name_trgm "
            "ON speakers USING gin (name gin_trgm_ops);"
        )
        # HNSW + cosine：20000 筆毫秒級相似度搜尋
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_speakers_embedding_hnsw "
            "ON speakers USING hnsw (embedding vector_cosine_ops) "
            "WITH (m = 16, ef_construction = 200);"
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_samples_speaker "
            "ON speaker_samples(speaker_id);"
        )
        conn.commit()
    logger.info("聲紋資料庫初始化完成（speakers / speaker_samples / 索引）")


# ──────────────────────────── 寫入 ────────────────────────────

def add_speaker(name: str, embedding, note: str = "", source: str = "") -> int:
    """新增語者（含第一段樣本），回傳語者 id。"""
    vec = _vec_to_str(embedding)
    with closing(_connect()) as conn, conn.cursor() as cur:
        cur.execute(
            "INSERT INTO speakers (name, note, embedding, sample_count) "
            "VALUES (%s, %s, %s::vector, 1) RETURNING id;",
            (name, note, vec),
        )
        sid = cur.fetchone()[0]
        cur.execute(
            "INSERT INTO speaker_samples (speaker_id, embedding, source) "
            "VALUES (%s, %s::vector, %s);",
            (sid, vec, source),
        )
        conn.commit()
    logger.info(f"新增語者 #{sid}「{name}」")
    return int(sid)


def add_sample(speaker_id: int, embedding, source: str = "") -> int:
    """為既有語者加一段樣本，AVG 重算 centroid，回傳更新後 sample_count。"""
    vec = _vec_to_str(embedding)
    with closing(_connect()) as conn, conn.cursor() as cur:
        cur.execute(
            "INSERT INTO speaker_samples (speaker_id, embedding, source) "
            "VALUES (%s, %s::vector, %s);",
            (speaker_id, vec, source),
        )
        cur.execute(
            "UPDATE speakers SET "
            "  embedding = (SELECT avg(embedding) FROM speaker_samples WHERE speaker_id = %s), "
            "  sample_count = (SELECT count(*) FROM speaker_samples WHERE speaker_id = %s), "
            "  updated_at = now() "
            "WHERE id = %s RETURNING sample_count;",
            (speaker_id, speaker_id, speaker_id),
        )
        row = cur.fetchone()
        conn.commit()
    return int(row[0]) if row else 0


def delete_speaker(speaker_id: int) -> None:
    """刪除語者（樣本經 ON DELETE CASCADE 連動刪除）。"""
    with closing(_connect()) as conn, conn.cursor() as cur:
        cur.execute("DELETE FROM speakers WHERE id = %s;", (speaker_id,))
        conn.commit()


def rename_speaker(speaker_id: int, name: str, note: Optional[str] = None) -> None:
    """改名（待命名的未知語者命名也走這裡）。"""
    with closing(_connect()) as conn, conn.cursor() as cur:
        if note is None:
            cur.execute(
                "UPDATE speakers SET name = %s, updated_at = now() WHERE id = %s;",
                (name, speaker_id),
            )
        else:
            cur.execute(
                "UPDATE speakers SET name = %s, note = %s, updated_at = now() WHERE id = %s;",
                (name, note, speaker_id),
            )
        conn.commit()


# ──────────────────────────── 查詢 ────────────────────────────

def search(embedding, top_k: int = 5, ef_search: int = 100) -> List[Dict]:
    """相似度搜尋，回傳 [{id, name, sample_count, similarity}, ...]（依相似度遞減）。

    similarity = 1 - cosine_distance，範圍約 [-1, 1]，越大越像。
    """
    vec = _vec_to_str(embedding)
    ef = int(ef_search)
    with closing(_connect()) as conn, conn.cursor() as cur:
        # SET LOCAL 需在同一交易內，下面 SELECT 緊接，commit 前有效
        cur.execute(f"SET LOCAL hnsw.ef_search = {ef};")
        cur.execute(
            "SELECT id, name, sample_count, 1 - (embedding <=> %s::vector) AS similarity "
            "FROM speakers ORDER BY embedding <=> %s::vector LIMIT %s;",
            (vec, vec, int(top_k)),
        )
        rows = cur.fetchall()
        conn.commit()
    return [
        {"id": r[0], "name": r[1], "sample_count": r[2], "similarity": float(r[3])}
        for r in rows
    ]


def list_speakers(
    page: int = 1, page_size: int = 20, keyword: str = ""
) -> Tuple[List[Dict], int]:
    """分頁列表（關鍵字過濾姓名；純數字關鍵字同時比對語者 id），回傳 (這頁的列表, 總筆數)。"""
    page = max(1, int(page))
    page_size = max(1, int(page_size))
    offset = (page - 1) * page_size
    kw = (keyword or "").strip()

    with closing(_connect()) as conn, conn.cursor() as cur:
        if kw:
            like = f"%{kw}%"
            # 純數字關鍵字同時比對 id（精確）與姓名（模糊）；非數字只比對姓名。
            if kw.isdigit():
                where = "(name ILIKE %s OR id = %s)"
                filter_params: tuple = (like, int(kw))
            else:
                where = "name ILIKE %s"
                filter_params = (like,)
            cur.execute(f"SELECT count(*) FROM speakers WHERE {where};", filter_params)
            total = cur.fetchone()[0]
            cur.execute(
                "SELECT id, name, note, sample_count, created_at, updated_at "
                f"FROM speakers WHERE {where} "
                "ORDER BY updated_at DESC LIMIT %s OFFSET %s;",
                (*filter_params, page_size, offset),
            )
        else:
            cur.execute("SELECT count(*) FROM speakers;")
            total = cur.fetchone()[0]
            cur.execute(
                "SELECT id, name, note, sample_count, created_at, updated_at "
                "FROM speakers ORDER BY updated_at DESC LIMIT %s OFFSET %s;",
                (page_size, offset),
            )
        rows = cur.fetchall()

    speakers = [
        {
            "id": r[0], "name": r[1], "note": r[2], "sample_count": r[3],
            "created_at": r[4], "updated_at": r[5],
        }
        for r in rows
    ]
    return speakers, int(total)


def get_speaker(speaker_id: int) -> Optional[Dict]:
    with closing(_connect()) as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT id, name, note, sample_count, created_at, updated_at "
            "FROM speakers WHERE id = %s;",
            (speaker_id,),
        )
        r = cur.fetchone()
    if not r:
        return None
    return {
        "id": r[0], "name": r[1], "note": r[2], "sample_count": r[3],
        "created_at": r[4], "updated_at": r[5],
    }


def count_speakers() -> int:
    with closing(_connect()) as conn, conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM speakers;")
        return int(cur.fetchone()[0])
