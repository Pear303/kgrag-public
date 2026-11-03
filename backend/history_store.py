"""历史会话持久化：把问答记录保存到 JSON，重启不丢。

组织：会话（session）分组。每个会话含多轮 message（user/assistant）。
assistant 消息额外携带该轮的检索详情（paths / path_detail / refusal /
vector_candidates 等结构化字段），供前端"点选历史会话"时完整回看——
包括答案、KG 多跳路径、Vector 候选片段，与发生时所见一致。

存储：.sisyphus/registry/history.json。所有写操作持进程内 threading.Lock，
原子写（tmp + os.replace），与 manual_registry 风格一致。
"""
import os
import json
import threading
import uuid
from datetime import datetime

from config import PROJECT_ROOT

HISTORY_PATH = os.path.join(PROJECT_ROOT, ".sisyphus", "registry", "history.json")

_lock = threading.Lock()


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


class HistoryStore:
    def __init__(self, path: str = None):
        self.path = path or HISTORY_PATH

    # ---------- 底层读写 ----------
    def _load(self) -> dict:
        if not os.path.exists(self.path):
            return {"sessions": []}
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            # 损坏时保留原始文件，返回空（避免写操作把坏文件覆盖）
            return {"sessions": []}

    def _save(self, data: dict) -> None:
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        tmp = self.path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        os.replace(tmp, self.path)

    # ---------- 会话级操作 ----------
    def create_session(self, title: str = "新对话") -> dict:
        with _lock:
            data = self._load()
            sess = {
                "session_id": uuid.uuid4().hex[:12],
                "title": (title or "新对话")[:40],
                "created_at": _now(),
                "updated_at": _now(),
                "messages": [],
            }
            data.setdefault("sessions", []).append(sess)
            self._save(data)
            return sess

    def list_sessions(self, limit: int = 200) -> list:
        """返回会话元信息（不含 messages，列表用），按更新时间倒序。"""
        with _lock:
            data = self._load()
            out = []
            for s in data.get("sessions", []):
                out.append({
                    "session_id": s["session_id"],
                    "title": s.get("title", ""),
                    "created_at": s.get("created_at", ""),
                    "updated_at": s.get("updated_at", ""),
                    "n_messages": len(s.get("messages", [])),
                })
            out.sort(key=lambda x: x.get("updated_at") or "", reverse=True)
            return out[:limit]

    def get_session(self, session_id: str) -> dict:
        with _lock:
            data = self._load()
            for s in data.get("sessions", []):
                if s["session_id"] == session_id:
                    return s
            return None

    def rename_session(self, session_id: str, new_title: str) -> bool:
        with _lock:
            data = self._load()
            for s in data.get("sessions", []):
                if s["session_id"] == session_id:
                    s["title"] = (new_title or "未命名")[:40]
                    s["updated_at"] = _now()
                    self._save(data)
                    return True
            return False

    def delete_session(self, session_id: str) -> bool:
        with _lock:
            data = self._load()
            before = len(data.get("sessions", []))
            data["sessions"] = [s for s in data.get("sessions", [])
                                if s["session_id"] != session_id]
            if len(data["sessions"]) == before:
                return False
            self._save(data)
            return True

    # ---------- 消息级操作 ----------
    def append_message(self, session_id: str, role: str, content: str,
                       **detail) -> dict:
        """向会话追加一条消息。role: user / assistant。

        detail 携带该轮检索详情（KG paths/path_detail、refusal、
        vector candidates 等），assistant 回看时据此还原完整展示。
        """
        with _lock:
            data = self._load()
            for s in data.get("sessions", []):
                if s["session_id"] == session_id:
                    msg = {"role": role, "content": content}
                    if role == "assistant" and detail:
                        msg["detail"] = detail
                    s.setdefault("messages", []).append(msg)
                    s["updated_at"] = _now()
                    # 标题用首条用户消息前 20 字
                    if not s.get("title") or s["title"] == "新对话":
                        first_user = next(
                            (m for m in s["messages"]
                             if m.get("role") == "user"), None)
                        if first_user:
                            s["title"] = first_user["content"][:20] + \
                                ("…" if len(first_user["content"]) > 20 else "")
                    self._save(data)
                    return msg
            raise KeyError(f"会话不存在: {session_id}")

    def clear_all(self) -> None:
        with _lock:
            self._save({"sessions": []})
