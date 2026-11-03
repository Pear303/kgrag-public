"""手册注册表：管理所有上传手册的元数据与构建产物路径。

存储：JSON 文件（单用户场景足够）。所有写操作持进程内 threading.Lock，
保证 current 唯一性与并发安全。
"""
import os
import re
import json
import threading
import uuid

from config import PROJECT_ROOT

REGISTRY_DIR = os.path.join(PROJECT_ROOT, ".sisyphus", "registry")
REGISTRY_PATH = os.path.join(REGISTRY_DIR, "manuals.json")

_lock = threading.Lock()


def _slugify(name: str) -> str:
    base = os.path.splitext(os.path.basename(name))[0]
    s = re.sub(r"[^\w\u4e00-\u9fff-]+", "_", base).strip("_")
    return s or uuid.uuid4().hex[:8]


def _default_manual(source_manual: str, manual_id: str = None) -> dict:
    return {
        "manual_id": manual_id or _slugify(source_manual),
        "display_name": os.path.splitext(os.path.basename(source_manual))[0],
        "file_path": "",
        "uploaded_at": "",
        "status": "uploading",
        "current": False,
        "source_manual": source_manual,
        "build": {},
        "task_id": None,
        "error": None,
    }


class ManualRegistry:
    def __init__(self, path: str = None):
        self.path = path or REGISTRY_PATH

    def _load(self) -> dict:
        if not os.path.exists(self.path):
            return {"manuals": []}
        with open(self.path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _save(self, data: dict) -> None:
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        tmp = self.path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        os.replace(tmp, self.path)

    def list_manuals(self) -> list:
        with _lock:
            return self._load().get("manuals", [])

    def get_manual(self, manual_id: str) -> dict:
        for m in self.list_manuals():
            if m["manual_id"] == manual_id:
                return m
        return None

    def add_manual(self, manual: dict) -> dict:
        with _lock:
            data = self._load()
            m = dict(_default_manual(manual["source_manual"], manual.get("manual_id")))
            m.update(manual)
            data.setdefault("manuals", []).append(m)
            self._save(data)
            return m

    def update_manual(self, manual_id: str, **fields) -> dict:
        with _lock:
            data = self._load()
            for m in data.get("manuals", []):
                if m["manual_id"] == manual_id:
                    m.update(fields)
                    self._save(data)
                    return m
            raise KeyError(f"手册不存在: {manual_id}")

    def set_current(self, manual_id: str) -> dict:
        with _lock:
            data = self._load()
            target = None
            for m in data.get("manuals", []):
                if m["manual_id"] == manual_id:
                    target = m
                    m["current"] = True
                else:
                    m["current"] = False
            if target is None:
                raise KeyError(f"手册不存在: {manual_id}")
            self._save(data)
            return target

    def remove_manual(self, manual_id: str) -> None:
        with _lock:
            data = self._load()
            data["manuals"] = [m for m in data.get("manuals", [])
                               if m["manual_id"] != manual_id]
            self._save(data)

    def get_current(self) -> dict:
        for m in self.list_manuals():
            if m.get("current"):
                return m
        return None
