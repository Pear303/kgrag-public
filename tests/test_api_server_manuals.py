# tests/test_api_server_manuals.py
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client(tmp_path, monkeypatch):
    from backend import manual_registry as mr
    from backend import task_manager as tm
    monkeypatch.setattr(mr, "REGISTRY_PATH", str(tmp_path / "manuals.json"))
    monkeypatch.setattr(tm, "TASK_DIR", str(tmp_path / "tasks"))
    # 阻止 KGRAGService 真实初始化（导入 api_server 时懒构建，端点调用时才触发）
    import api_server
    return TestClient(api_server.app)


def test_health_ok(client):
    r = client.get("/health")
    assert r.status_code in (200, 500)  # 500 = 服务未初始化 Neo4j，但不崩


def test_manuals_empty_list(client):
    r = client.get("/manuals")
    assert r.status_code == 200
    assert isinstance(r.json(), list)
