import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from backend import task_manager


@pytest.fixture
def tm(tmp_path, monkeypatch):
    monkeypatch.setattr(task_manager, "TASK_DIR", str(tmp_path / "tasks"))
    monkeypatch.setattr("backend.manual_registry.REGISTRY_PATH",
                        str(tmp_path / "manuals.json"))
    tm = task_manager.TaskManager()
    for mid in ("m1", "a", "b"):
        tm.registry.add_manual({"manual_id": mid, "source_manual": f"{mid}.md",
                                "status": "uploading"})
    return tm


def test_submit_creates_task_and_completes(tm, monkeypatch):
    calls = []

    def fake_pipeline(manual_id, source_manual, md_path, progress_cb=None,
                        extraction_prompt=None):
        calls.append((manual_id, source_manual, md_path))
        if progress_cb:
            progress_cb(7, 100, "完成")
        return {"triples": 5}

    monkeypatch.setattr(task_manager, "build_pipeline", fake_pipeline)
    tid = tm.submit("m1", "m1.md", "/tmp/m1.md")
    tm.wait(tid, timeout=5)
    t = tm.get_task(tid)
    assert t["status"] == "done"
    assert t["result"]["triples"] == 5
    assert calls == [("m1", "m1.md", "/tmp/m1.md")]


def test_submit_records_progress(tm, monkeypatch):
    def fake_pipeline(manual_id, source_manual, md_path, progress_cb=None,
                        extraction_prompt=None):
        progress_cb(1, 5, "chunk 化")
        progress_cb(3, 45, "导入")
        return {}

    monkeypatch.setattr(task_manager, "build_pipeline", fake_pipeline)
    tid = tm.submit("m1", "m1.md", "/tmp/m1.md")
    tm.wait(tid, timeout=5)
    t = tm.get_task(tid)
    assert t["step"] == 3
    assert t["pct"] == 100  # 完成态 pct 归一为 100


def test_failure_records_error(tm, monkeypatch):
    def boom(manual_id, source_manual, md_path, progress_cb=None,
                        extraction_prompt=None):
        raise RuntimeError("LLM 超时")

    monkeypatch.setattr(task_manager, "build_pipeline", boom)
    tid = tm.submit("m1", "m1.md", "/tmp/m1.md")
    tm.wait(tid, timeout=5)
    t = tm.get_task(tid)
    assert t["status"] == "failed"
    assert "LLM 超时" in t["error"]


def test_mutual_exclusion(tm, monkeypatch):
    # 单 build 互斥：第二个任务排队，第一个完成后才执行
    order = []
    import threading
    gate = threading.Event()

    def slow(manual_id, source_manual, md_path, progress_cb=None,
                        extraction_prompt=None):
        order.append(manual_id)
        gate.wait(timeout=5)
        return {}

    monkeypatch.setattr(task_manager, "build_pipeline", slow)
    t1 = tm.submit("a", "a.md", "/tmp/a.md")
    t2 = tm.submit("b", "b.md", "/tmp/b.md")
    time.sleep(0.3)
    gate.set()
    tm.wait(t1, timeout=5)
    tm.wait(t2, timeout=5)
    assert order == ["a", "b"]


def test_list_tasks(tm, monkeypatch):
    def fake(manual_id, source_manual, md_path, progress_cb=None,
                        extraction_prompt=None):
        return {}

    monkeypatch.setattr(task_manager, "build_pipeline", fake)
    tid = tm.submit("m1", "m1.md", "/tmp/m1.md")
    tm.wait(tid, timeout=5)
    assert len(tm.list_tasks()) >= 1
