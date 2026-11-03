# -*- coding: utf-8 -*-
"""运行时模型选择（embedding / 答案 LLM）的持久化。

落盘在 data/runtime/model_prefs.json，不写 .env：
- .env 仍是"出厂默认"（LOCAL_EMBEDDING_MODEL / LOCAL_LLM_MODEL），
- 这里的值是运维在「模型设置」页显式切换后的选择，重启后继续生效。
"""
import json
import os
import tempfile
from datetime import datetime

from config import PROJECT_ROOT

PREFS_PATH = os.path.join(PROJECT_ROOT, "data", "runtime", "model_prefs.json")


def load_prefs(path: str = None) -> dict:
    p = path or PREFS_PATH
    try:
        with open(p, "r", encoding="utf-8") as f:
            d = json.load(f)
        return d if isinstance(d, dict) else {}
    except Exception:
        return {}


def save_prefs(embedding_model: str = None, llm_model: str = None,
               path: str = None) -> dict:
    """合并写入；只覆盖显式给出的字段。原子替换，避免半截文件。"""
    p = path or PREFS_PATH
    d = load_prefs(p)
    if embedding_model:
        d["embedding_model"] = str(embedding_model)
    if llm_model:
        d["llm_model"] = str(llm_model)
    d["updated_at"] = datetime.now().isoformat(timespec="seconds")
    os.makedirs(os.path.dirname(p), exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(p), suffix=".json")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(d, f, ensure_ascii=False, indent=1)
        os.replace(tmp, p)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
    return d


def clear_prefs(path: str = None) -> None:
    p = path or PREFS_PATH
    try:
        os.remove(p)
    except FileNotFoundError:
        pass
