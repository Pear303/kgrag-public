import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from backend.manual_registry import ManualRegistry


@pytest.fixture
def reg(tmp_path, monkeypatch):
    monkeypatch.setattr("backend.manual_registry.REGISTRY_PATH", str(tmp_path / "manuals.json"))
    return ManualRegistry()


def test_add_and_get(reg):
    reg.add_manual({"manual_id": "m1", "source_manual": "m1.md", "status": "ready"})
    assert reg.get_manual("m1")["manual_id"] == "m1"


def test_list_manuals(reg):
    reg.add_manual({"manual_id": "m1", "source_manual": "m1.md", "status": "ready"})
    reg.add_manual({"manual_id": "m2", "source_manual": "m2.md", "status": "processing"})
    assert len(reg.list_manuals()) == 2


def test_set_current_unique(reg):
    reg.add_manual({"manual_id": "m1", "source_manual": "m1.md", "status": "ready"})
    reg.add_manual({"manual_id": "m2", "source_manual": "m2.md", "status": "ready"})
    reg.set_current("m1")
    reg.set_current("m2")
    currents = [m for m in reg.list_manuals() if m.get("current")]
    assert len(currents) == 1
    assert currents[0]["manual_id"] == "m2"


def test_get_current_none_when_empty(reg):
    assert reg.get_current() is None


def test_remove_manual(reg):
    reg.add_manual({"manual_id": "m1", "source_manual": "m1.md", "status": "ready"})
    reg.remove_manual("m1")
    assert reg.get_manual("m1") is None


def test_update_manual(reg):
    reg.add_manual({"manual_id": "m1", "source_manual": "m1.md", "status": "processing"})
    reg.update_manual("m1", status="ready", build={"triples": 10})
    assert reg.get_manual("m1")["status"] == "ready"
    assert reg.get_manual("m1")["build"]["triples"] == 10


def test_add_manual_generates_id_when_missing(reg):
    m = reg.add_manual({"source_manual": "auto.md", "status": "ready"})
    assert m["manual_id"]
