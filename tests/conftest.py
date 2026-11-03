import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ---- Neo4j marker & auto-skip ----
def pytest_configure(config):
    config.addinivalue_line(
        "markers", "neo4j: mark test as requiring Neo4j database"
    )


def pytest_collection_modifyitems(config, items):
    """Auto-skip neo4j tests if Neo4j is unreachable."""
    try:
        from neo4j import GraphDatabase
        from config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD
        driver = GraphDatabase.driver(
            NEO4J_URI,
            auth=(NEO4J_USER, NEO4J_PASSWORD)
        )
        driver.verify_connectivity()
        driver.close()
        neo4j_available = True
    except Exception:
        neo4j_available = False

    skip_neo4j = pytest.mark.skip(reason="Neo4j not available at configured NEO4J_URI")
    for item in items:
        if "neo4j" in item.keywords and not neo4j_available:
            item.add_marker(skip_neo4j)


@pytest.fixture(scope="session")
def test_manual_path():
    return os.path.join(os.path.dirname(__file__), "..", "data", "manuals", "pump_manual.md")


@pytest.fixture(scope="session")
def sample_md_content():
    return """## 测试设备
| 故障现象 | 故障关系 | 可能原因 |
|---|---|---|
| 温度过高 | 可能原因 | 润滑不足 |
| 温度过高 | 可能原因 | 过载运行 |
"""
