"""手册管理页面功能可用性冒烟测试（不触发真实 LLM/构建/删除生产数据）。

覆盖手册管理页可见按钮对应的前端处理函数：
- 上传并构建 -> _api_upload
- 刷新任务 -> _render_tasks
- 刷新手册 -> _render_manuals
- 设为当前 -> _api_activate
- 删除 -> _api_delete

以及对应 API 端点的隔离测试（临时 registry + mock 服务构建/图删除）。
所有后端构建均被 mock，绝不消耗 LLM / Ollama embedding / 在线额度。
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient


# ============ 前端按钮处理函数 ============

def test_upload_and_build_button_success(tmp_path):
    """『上传并构建』按钮在 API 成功时返回成功提示。"""
    from frontend import app as front
    md = tmp_path / "manual.md"
    md.write_text("# 测试手册\n", encoding="utf-8")

    class _FakeFile:
        name = str(md)

    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = {"task_id": "task123"}

    with patch("frontend.app.requests.post", return_value=resp) as mock_post:
        out = front._api_upload(_FakeFile())

    assert "上传成功" in out
    assert "task123" in out
    assert mock_post.call_args.args[0].endswith("/manuals")


def test_upload_and_build_button_failure(tmp_path):
    """『上传并构建』按钮在 API 返回失败时给出明确失败信息。"""
    from frontend import app as front
    md = tmp_path / "manual.md"
    md.write_text("# 测试手册\n", encoding="utf-8")

    class _FakeFile:
        name = str(md)

    resp = MagicMock()
    resp.status_code = 400
    resp.text = "unsupported format"

    with patch("frontend.app.requests.post", return_value=resp):
        out = front._api_upload(_FakeFile())

    assert "上传失败" in out
    assert "unsupported format" in out


def test_refresh_manuals_button():
    """『刷新手册』按钮渲染手册列表表格。"""
    from frontend import app as front
    samples = [
        {
            "manual_id": "manual_a",
            "status": "ready",
            "build": {"triples": 10, "communities": 2},
            "current": True,
        },
        {
            "manual_id": "manual_b",
            "status": "processing",
            "build": {},
            "current": False,
        },
    ]
    with patch("frontend.app._api_list_manuals", return_value=samples):
        out = front._render_manuals()

    assert "手册ID" in out
    assert "manual_a" in out
    assert "manual_b" in out
    assert "★" in out
    assert "10" in out


def test_refresh_tasks_button():
    """『刷新任务』按钮渲染任务进度表格。"""
    from frontend import app as front
    samples = [
        {"task_id": "t1", "status": "done", "step": 7, "pct": 100},
        {"task_id": "t2", "status": "queued", "step": 0, "pct": 0},
    ]
    with patch("frontend.app._api_list_tasks", return_value=samples):
        out = front._render_tasks()

    assert "任务ID" in out
    assert "t1" in out
    assert "t2" in out
    assert "100%" in out


def test_activate_button_success():
    """『设为当前』按钮在 API 成功时返回切换成功。"""
    from frontend import app as front
    resp = MagicMock()
    resp.status_code = 200
    resp.text = ""
    with patch("frontend.app.requests.post", return_value=resp) as mock_post:
        out = front._api_activate("manual_a")

    assert "已切换到 manual_a" in out
    assert "/manual_a/activate" in mock_post.call_args.args[0]


def test_activate_button_failure():
    """『设为当前』按钮在 API 失败时返回后端错误文本。"""
    from frontend import app as front
    resp = MagicMock()
    resp.status_code = 500
    resp.text = '{"detail":"build failed"}'
    with patch("frontend.app.requests.post", return_value=resp):
        out = front._api_activate("manual_a")

    assert "切换失败" in out
    assert "build failed" in out


def test_delete_button_success():
    """『删除』按钮在 API 成功时返回删除成功。"""
    from frontend import app as front
    resp = MagicMock()
    resp.status_code = 200
    resp.text = ""
    with patch("frontend.app.requests.delete", return_value=resp) as mock_delete:
        out = front._api_delete("manual_b")

    assert "已删除 manual_b" in out
    assert "/manual_b" in mock_delete.call_args.args[0]


def test_delete_button_failure():
    """『删除』按钮在 API 失败时返回后端错误。"""
    from frontend import app as front
    resp = MagicMock()
    resp.status_code = 400
    resp.text = "当前激活手册不可删除"
    with patch("frontend.app.requests.delete", return_value=resp):
        out = front._api_delete("manual_b")

    assert "删除失败" in out
    assert "当前激活手册不可删除" in out


# ============ 对应 API 端点（隔离临时 registry，mock 副作用）============

def test_activate_endpoint_switches_current(tmp_path, monkeypatch):
    """『设为当前』API：切换到指定手册，且仅保留一个 current。"""
    from backend import manual_registry as mr
    monkeypatch.setattr(mr, "REGISTRY_PATH", str(tmp_path / "manuals.json"))

    import api_server
    api_server._registry = mr.ManualRegistry()
    api_server._registry.add_manual({"manual_id": "m1", "source_manual": "m1.md",
                                     "status": "ready", "current": True})
    api_server._registry.add_manual({"manual_id": "m2", "source_manual": "m2.md",
                                     "status": "ready", "current": False})

    monkeypatch.setattr(api_server, "_build_service", lambda manual: MagicMock())

    client = TestClient(api_server.app)
    r = client.post("/manuals/m2/activate")
    assert r.status_code == 200
    assert r.json()["current"] is True

    current = [m for m in api_server._registry.list_manuals() if m.get("current")]
    assert [m["manual_id"] for m in current] == ["m2"]


def test_delete_endpoint_removes_manual(tmp_path, monkeypatch):
    """『删除』API：删除非当前非预构建手册，且不触碰真实 Neo4j。"""
    from backend import manual_registry as mr
    monkeypatch.setattr(mr, "REGISTRY_PATH", str(tmp_path / "manuals.json"))

    import api_server
    api_server._registry = mr.ManualRegistry()
    api_server._registry.add_manual({"manual_id": "keep", "source_manual": "keep.md",
                                     "status": "ready", "current": False})
    api_server._registry.add_manual({"manual_id": "del", "source_manual": "del.md",
                                     "status": "ready", "current": False})

    # 阻止真实 GraphStore/删除图谱副作用
    monkeypatch.setattr("backend.graph_store.GraphStore", lambda: MagicMock())
    monkeypatch.setattr("backend.build_pipeline._delete_manual_graph",
                        lambda store, src: None)

    client = TestClient(api_server.app)
    r = client.delete("/manuals/del")
    assert r.status_code == 200
    assert r.json()["deleted"] == "del"
    assert api_server._registry.get_manual("del") is None
    assert api_server._registry.get_manual("keep") is not None


def test_task_list_endpoint_returns_tasks(tmp_path, monkeypatch):
    """任务进度数据源可用（列表接口）。"""
    from backend import task_manager as tm
    monkeypatch.setattr(tm, "TASK_DIR", str(tmp_path / "tasks"))
    monkeypatch.setattr("backend.manual_registry.REGISTRY_PATH",
                        str(tmp_path / "manuals.json"))

    import api_server
    api_server._task_manager = tm.TaskManager()

    client = TestClient(api_server.app)
    r = client.get("/tasks")
    assert r.status_code == 200
    assert isinstance(r.json(), list)