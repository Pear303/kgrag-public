"""Tests for backend/graph_store.py —— requires Neo4j running.

All tests are marked @pytest.mark.neo4j and auto-skip when Neo4j is unreachable.
"""
import pytest
import uuid

from backend.graph_store import get_property_graph_store

# Unique prefix for test data —— ensures isolation from production data
TEST_PREFIX = "test_" + uuid.uuid4().hex[:8]


@pytest.fixture(scope="module")
def graph_store():
    """Module-scoped GraphStore fixture."""
    return get_property_graph_store()


@pytest.fixture
def cleanup(graph_store):
    """Cleanup fixture —— removes all test data after each test."""
    yield
    cypher_entity = "MATCH (n:Entity) WHERE n.name STARTS WITH $prefix DETACH DELETE n"
    cypher_auto = "MATCH (n:__Entity__) WHERE n.id STARTS WITH $prefix DETACH DELETE n"
    try:
        graph_store.structured_query(cypher_entity, {"prefix": TEST_PREFIX})
        graph_store.structured_query(cypher_auto, {"prefix": TEST_PREFIX})
    except Exception as e:
        print(f"Cleanup warning: {e}")


@pytest.mark.neo4j
def test_structured_query_basic(graph_store, cleanup):
    """Test 1: structured_query returns list of dicts."""
    result = graph_store.structured_query("RETURN 1 AS test_value, 'hello' AS msg")
    assert isinstance(result, list)
    assert len(result) == 1
    assert result[0]["test_value"] == 1
    assert result[0]["msg"] == "hello"


@pytest.mark.neo4j
def test_structured_query_with_params(graph_store, cleanup):
    """Test 2: structured_query accepts param_map."""
    result = graph_store.structured_query(
        "RETURN $name AS name, $age AS age",
        {"name": "Alice", "age": 30}
    )
    assert len(result) == 1
    assert result[0]["name"] == "Alice"
    assert result[0]["age"] == 30


@pytest.mark.neo4j
def test_seed_triples_writes_entities(graph_store, cleanup):
    """Test 3: seed_triples creates :Entity nodes with :REL relationships."""
    triples = [
        (f"{TEST_PREFIX}_source", "可能原因", f"{TEST_PREFIX}_target1"),
        (f"{TEST_PREFIX}_source", "解决方法", f"{TEST_PREFIX}_target2"),
    ]
    graph_store.seed_triples(triples)

    # Verify nodes were created
    nodes = graph_store.structured_query(
        "MATCH (n:Entity) WHERE n.name STARTS WITH $prefix "
        "RETURN n.name AS name ORDER BY name",
        {"prefix": TEST_PREFIX},
    )
    names = [n["name"] for n in nodes]
    assert f"{TEST_PREFIX}_source" in names
    assert f"{TEST_PREFIX}_target1" in names
    assert f"{TEST_PREFIX}_target2" in names

    # Verify relationships
    rels = graph_store.structured_query(
        "MATCH (s:Entity)-[r:REL]->(o:Entity) "
        "WHERE s.name STARTS WITH $prefix "
        "RETURN s.name AS src, r.type AS rel, o.name AS dst ORDER BY src, dst",
        {"prefix": TEST_PREFIX},
    )
    assert len(rels) == 2
    assert rels[0]["rel"] in ("可能原因", "解决方法")


@pytest.mark.neo4j
def test_upsert_auto_triples(graph_store, cleanup):
    """Test 4: upsert_auto_triples writes to :__Entity__ namespace with :REL edges."""
    triples = [
        (f"{TEST_PREFIX}_auto_src", "包含", f"{TEST_PREFIX}_auto_dst1"),
        (f"{TEST_PREFIX}_auto_src", "表现为", f"{TEST_PREFIX}_auto_dst2"),
    ]
    graph_store.upsert_auto_triples(triples)

    # Verify :__Entity__ nodes
    nodes = graph_store.structured_query(
        "MATCH (n:__Entity__) WHERE n.id STARTS WITH $prefix "
        "RETURN n.id AS id ORDER BY id",
        {"prefix": f"{TEST_PREFIX}_auto"},
    )
    ids = [n["id"] for n in nodes]
    assert f"{TEST_PREFIX}_auto_src" in ids
    assert f"{TEST_PREFIX}_auto_dst1" in ids
    assert f"{TEST_PREFIX}_auto_dst2" in ids

    # Verify relationships are :REL type (critical — must not be per-type labels)
    rels = graph_store.structured_query(
        "MATCH (s:__Entity__)-[r]->(o:__Entity__) "
        "WHERE s.id STARTS WITH $prefix "
        "RETURN type(r) AS rt, count(*) AS c",
        {"prefix": f"{TEST_PREFIX}_auto"},
    )
    rel_types = [r["rt"] for r in rels]
    assert "REL" in rel_types, f"Expected :REL edges, got {rel_types}"


@pytest.mark.neo4j
def test_get_all_entities(graph_store, cleanup):
    """Test 5: get_all_entities returns entity list containing our test data."""
    # Seed some test data first
    triples = [(f"{TEST_PREFIX}_list_a", "可能原因", f"{TEST_PREFIX}_list_b")]
    graph_store.seed_triples(triples)

    entities = graph_store.get_all_entities()
    assert isinstance(entities, (list, set, tuple))
    assert len(entities) > 0
    # Test entities should appear in the list
    entity_names = [str(e) for e in entities]
    assert f"{TEST_PREFIX}_list_a" in entity_names


@pytest.mark.neo4j
def test_idempotent_seed(graph_store, cleanup):
    """Test 6: seed_triples is idempotent —— re-seeding same data shouldn't duplicate."""
    triples = [(f"{TEST_PREFIX}_idem", "可能原因", f"{TEST_PREFIX}_idem_target")]
    graph_store.seed_triples(triples)
    graph_store.seed_triples(triples)  # Seed same data twice

    # Count matching nodes —— should be exactly 2 (not 4)
    count = graph_store.structured_query(
        "MATCH (n:Entity) WHERE n.name STARTS WITH $prefix "
        "RETURN count(n) AS c",
        {"prefix": f"{TEST_PREFIX}_idem"},
    )
    assert count[0]["c"] == 2


@pytest.mark.neo4j
def test_seed_isolates_entities_by_source_manual(graph_store, cleanup):
    """Test A6: 多手册同名实体隔离——不同 source_manual 的同名实体互不覆盖。

    回归：实体唯一键从 name 改为 (name, source_manual)，同名实体后导入者
    不再覆盖先导入者的 source_manual，多手册隔离失效问题被根治。
    """
    triples = [(f"{TEST_PREFIX}_iso", "可能原因", f"{TEST_PREFIX}_iso_target")]
    graph_store.seed_triples(triples, source_manual="manual_a", source_line=1)
    # 同三元组、不同手册名重 seed：应创建独立实体，而非覆盖 manual_a
    graph_store.seed_triples(triples, source_manual="manual_b", source_line=2)
    rows = graph_store.structured_query(
        "MATCH (n:Entity) WHERE n.name STARTS WITH $prefix "
        "RETURN n.name AS name, n.source_manual AS sm, n.source_line AS sl "
        "ORDER BY name, sm",
        {"prefix": f"{TEST_PREFIX}_iso"},
    )
    # 两个手册各保留 2 个独立节点（source + target），共 4 个，互不覆盖
    assert len(rows) == 4
    assert {r["sm"] for r in rows} == {"manual_a", "manual_b"}
    # manual_a 的 source_line 未被覆盖，manual_b 的为新值
    assert {r["sl"] for r in rows if r["sm"] == "manual_a"} == {1}
    assert {r["sl"] for r in rows if r["sm"] == "manual_b"} == {2}


@pytest.mark.neo4j
def test_seed_triples_no_parallel_edge_on_existing_rel(graph_store, cleanup):
    """回归（2026-10-20 双写修复）：已有同型边（无 meta 的历史建库形态）时，
    seed_triples 按 (a, type, b) 命中复用该边，不再造平行边。

    旧代码 MERGE {type, meta} 在 MERGE 属性子集匹配语义下对无 meta 同型边
    不互命中，7690 曾积累 2,146 组孪生边（10-20 已清理）。
    """
    src_manual = TEST_PREFIX + "_dw_manual"
    s, r, o = f"{TEST_PREFIX}_dw_s", "可能原因", f"{TEST_PREFIX}_dw_o"
    # 1) 预置一条"裸边"（只有 type，无 meta）——模拟旧版建库形态
    graph_store.structured_query(
        "MERGE (a:Entity {name:$s, source_manual:$src}) "
        "MERGE (b:Entity {name:$o, source_manual:$src}) "
        "MERGE (a)-[:REL {type:$r}]->(b)",
        {"s": s, "o": o, "r": r, "src": src_manual},
    )
    # 2) seed_triples 同三元组（v4 行为：命中裸边，不新建）
    graph_store.seed_triples([(s, r, o, "none")], source_manual=src_manual)
    edges = graph_store.structured_query(
        "MATCH (a:Entity)-[e:REL]->(b:Entity) WHERE a.name = $s "
        "RETURN count(e) AS c",
        {"s": s},
    )
    assert edges[0]["c"] == 1, "同型边已存在时 seed_triples 不得产生平行边"

    # 3) 干净图首 seed：meta 必须在 ON CREATE 时写入（最终态与 v3 一致）
    s2, o2 = f"{TEST_PREFIX}_dw_s2", f"{TEST_PREFIX}_dw_o2"
    graph_store.seed_triples([(s2, "解决方法", o2, "none")], source_manual=src_manual)
    meta_rows = graph_store.structured_query(
        "MATCH (a:Entity)-[e:REL]->(b:Entity) WHERE a.name = $s "
        "RETURN e.meta AS meta",
        {"s": s2},
    )
    assert len(meta_rows) == 1 and meta_rows[0]["meta"] == "none"


@pytest.mark.neo4j
def test_expand_paths_distinct_collapses_parallel_paths(graph_store, cleanup):
    """回归（2026-10-20）：expand_paths 加 DISTINCT——平行边产出的同投影路径行
    只保留一条，不挤占 LIMIT 路径槽。"""
    s, o = f"{TEST_PREFIX}_pe_a", f"{TEST_PREFIX}_pe_b"
    # 人造两条平行边（同 a,b,type；CREATE 两次绕过 MERGE 去重）
    graph_store.structured_query(
        "MERGE (a:Entity {name:$s}) "
        "MERGE (b:Entity {name:$o}) "
        "CREATE (a)-[:REL {type:$r}]->(b) "
        "CREATE (a)-[:REL {type:$r}]->(b)",
        {"s": s, "o": o, "r": "可能原因"},
    )
    paths = graph_store.expand_paths([s], max_hop=1)
    hits = [p for p in paths if p[0] == [s, o]]
    assert len(hits) == 1, f"平行边的同投影路径应被 DISTINCT 折叠为 1 条，got {len(hits)}"
