# -*- coding: utf-8 -*-
"""backend/history_store.py 单元测试（不依赖网络/Neo4j）。"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from backend.history_store import HistoryStore


@pytest.fixture
def store(tmp_path):
    return HistoryStore(path=str(tmp_path / "history.json"))


def test_create_and_get_session(store):
    s = store.create_session()
    assert s["session_id"]
    assert s["messages"] == []
    got = store.get_session(s["session_id"])
    assert got is not None


def test_append_message_and_retrieve(store):
    s = store.create_session()
    store.append_message(s["session_id"], "user", "问题A")
    store.append_message(s["session_id"], "assistant", "答案A",
                         paths=["p1"], path_detail=[{"seq": []}],
                         refusal=False)
    got = store.get_session(s["session_id"])
    assert len(got["messages"]) == 2
    assert got["messages"][0]["role"] == "user"
    assert got["messages"][1]["detail"]["paths"] == ["p1"]
    assert got["messages"][1]["detail"]["refusal"] is False


def test_title_from_first_user_message(store):
    s = store.create_session()
    store.append_message(s["session_id"], "user", "这是一段很长的用户问题用于测试标题截断行为是否正确生效的一整句话")
    got = store.get_session(s["session_id"])
    assert got["title"].startswith("这是一段很长的用户问题")


def test_delete_session(store):
    s = store.create_session()
    assert store.delete_session(s["session_id"]) is True
    assert store.get_session(s["session_id"]) is None
    assert store.delete_session("nonexistent") is False


def test_persistence_across_reload(store):
    """持久化：重新 new 一个实例指向同一文件，数据仍在。"""
    s = store.create_session()
    store.append_message(s["session_id"], "user", "问题")
    store.append_message(s["session_id"], "assistant", "答案", paths=["x"])
    store2 = HistoryStore(path=store.path)
    got = store2.get_session(s["session_id"])
    assert got is not None
    assert len(got["messages"]) == 2


def test_append_to_missing_session_raises(store):
    with pytest.raises(KeyError):
        store.append_message("nope", "user", "hi")
