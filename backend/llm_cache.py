"""LLM 调用磁盘缓存（SQLite，借鉴 KG_RAG 的 joblib.Memory / kg-rag SQLiteCache）。

作用域限制：只接抽取（extractor）与社区摘要（community）等开发/构建场景。
禁止接入评测答案生成链——评测要求每次全量重跑（regression R10），
缓存会拿到旧答案污染结果新鲜性。

env 开关: ENABLE_LLM_CACHE=1 启用（默认关闭，不改变现有行为）。
"""
import hashlib
import json
import os
import sqlite3
import threading
import time

_DEFAULT_DB = os.path.join(os.path.dirname(__file__), "..", ".sisyphus", "cache", "llm_cache.sqlite")


def _cache_key(model: str, prompt: str, temperature: float, max_tokens: int) -> str:
    payload = json.dumps({
        "model": model, "prompt": prompt,
        "temperature": float(temperature), "max_tokens": int(max_tokens),
    }, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class LLMCache:
    """SQLite 实现的 LLM 响应缓存（线程安全，幂等写入）。"""

    def __init__(self, db_path: str = _DEFAULT_DB):
        self.db_path = db_path
        self._lock = threading.Lock()
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS llm_cache ("
            "key TEXT PRIMARY KEY, model TEXT, response TEXT, created_at REAL)"
        )
        self._conn.commit()

    def get(self, model: str, prompt: str, temperature: float, max_tokens: int) -> str | None:
        k = _cache_key(model, prompt, temperature, max_tokens)
        with self._lock:
            row = self._conn.execute(
                "SELECT response FROM llm_cache WHERE key=?", (k,)).fetchone()
        return row[0] if row else None

    def put(self, model: str, prompt: str, temperature: float, max_tokens: int, response: str) -> None:
        if not response:
            return
        k = _cache_key(model, prompt, temperature, max_tokens)
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO llm_cache (key, model, response, created_at) VALUES (?,?,?,?)",
                (k, model, response, time.time()))
            self._conn.commit()

    def close(self):
        with self._lock:
            self._conn.close()


_cache = None


def get_cache() -> LLMCache:
    """全局单例；未启用时返回 None（调用方自行回退）。"""
    global _cache
    if os.getenv("ENABLE_LLM_CACHE", "0") != "1":
        return None
    if _cache is None:
        _cache = LLMCache()
    return _cache
