"""Neo4j 图存储封装（自封装 Cypher，文章 2 / 4.1 节）。

本模块不使用 LlamaIndex 的 Neo4jPropertyGraphStore，直接用 neo4j 官方驱动
（requirements.txt 钉 5.28.4）+ Cypher 读写，统一支撑：
    - 人工图谱写入（:Entity + :REL）与自动图谱写入（:__Entity__）
    - 场景A/场景B 多跳查询
    - 向量策略自动降级（>=5.23 用原生向量索引，否则内存余弦）

LlamaIndex 在本项目只承担模型接入（LLM/Embedding 适配）与向量基线；三元
组抽取与查询路由均为本仓库自写实现，不经 LlamaIndex。

原注释中的 Neo4j 4.2.1 + 4.4.x 驱动环境是复现初期状态，现为 5.26.0
Community，该段选型理由已失效故删去。
"""
import os
import re

from config import (NEO4J_URI, NEO4J_BUILD_URI, NEO4J_USER, NEO4J_PASSWORD,
                    NEO4J_DATABASE, VECTOR_MODE)

import neo4j
from neo4j import GraphDatabase

NATIVE_VECTOR_MIN_VERSION = (5, 23, 0)


def _norm_uri(u):
    return (u or "").strip().rstrip("/").lower()


def _eval_uri():
    return os.getenv("NEO4J_EVAL_URI") or "bolt://localhost:7690"


def assert_build_uri(uri):
    """建图/删除只许落在主库；禁止静默写冻结评测库。"""
    if _norm_uri(uri) == _norm_uri(_eval_uri()) and os.getenv("ALLOW_BUILD_ON_EVAL") != "1":
        raise PermissionError(
            "拒绝写入评测库 %s：建图应使用 NEO4J_BUILD_URI（.env 的 NEO4J_URI=7688）" % uri)

# ---- Cypher 注入防护：标签/关系名白名单 ----
# Cypher 原生不支持参数化标签名和关系类型名，因此必须验证。
# 仅允许已知标签 + 符合 Neo4j 标识符规范的字符串（字母/数字/下划线）。
_VALID_LABELS = frozenset({"Entity", "__Entity__", "__Entity__v2", "Community"})
_VALID_REL_TYPES = frozenset({"REL"})

_VALID_CYPHER_ID = re.compile(r'^[A-Za-z_][A-Za-z0-9_]*\Z')


def _validate_cypher_identifier(value: str, kind: str) -> str:
    """校验 Cypher 标识符（标签名或关系类型名），防止注入。

    Neo4j 不支持在 MATCH/CREATE 中参数化标签名或关系类型名，
    因此必须用白名单 + 正则严格校验，拒绝非法字符。
    """
    if not isinstance(value, str):
        raise ValueError(f"非法 {kind} 类型: 期望字符串，实际 {type(value).__name__}")
    if not _VALID_CYPHER_ID.match(value):
        raise ValueError(f"非法 {kind}: '{value}' 包含不允许的字符（仅允许字母、数字、下划线）")
    return value


def _validate_label(label: str) -> str:
    """校验并返回合法的 Neo4j 标签名。"""
    if label not in _VALID_LABELS:
        raise ValueError(
            f"未知标签 '{label}'，仅支持: {', '.join(sorted(_VALID_LABELS))}。"
            f"如需添加新标签，请在 graph_store._VALID_LABELS 中注册。"
        )
    return _validate_cypher_identifier(label, "标签")


def _validate_rel_type(rel_type: str) -> str:
    """校验并返回合法的关系类型名。"""
    if rel_type not in _VALID_REL_TYPES:
        raise ValueError(
            f"未知关系类型 '{rel_type}'，仅支持: {', '.join(sorted(_VALID_REL_TYPES))}。"
            f"如需添加新关系类型，请在 graph_store._VALID_REL_TYPES 中注册。"
        )
    return _validate_cypher_identifier(rel_type, "关系类型")


def detect_neo4j_version() -> tuple:
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    try:
        with driver.session() as s:
            rec = s.run("CALL dbms.components() YIELD versions RETURN versions").single()
            ver_str = rec["versions"][0].split("-")[0]
            return tuple(int(x) for x in ver_str.split("."))
    finally:
        driver.close()


def resolve_vector_mode() -> str:
    if VECTOR_MODE in ("native", "memory"):
        return VECTOR_MODE
    try:
        v = detect_neo4j_version()
        return "native" if v >= NATIVE_VECTOR_MIN_VERSION else "memory"
    except Exception:
        return "memory"


def assert_writable():
    """只读保护（2026-09-15 新增）：SERVING_READONLY=1 时拒绝一切写入。

    背景：本机 Neo4j 为 **5.26.0 Community**，实测不支持 RBAC
    （`GRANT ROLE ...` 与 `SHOW ROLES` 均报 UnsupportedAdministrationCommand，
    新建用户同样具备写权限）→ 无法用"只读账号"隔离生产服务与冻结评测库。
    因此改在**应用层设闸**：服务进程即使被触发了手册构建（上传/激活/重建），
    也会在真正写库前被此处拒绝，从而保证评测库 7690 不被写入。

    访问方式保持公开（非 GraphStore 实例方法），便于 build_pipeline 等
    在构造 store 之前先做前置校验。
    """
    if os.getenv("SERVING_READONLY", "0") == "1":
        raise PermissionError(
            "SERVING_READONLY=1：当前进程为只读模式，拒绝写入图库"
            "（生产服务不得修改冻结评测库；如需构建请以非只读进程执行）")


class GraphStore:
    """轻量图存储封装，统一驱动连接与查询。"""

    def __init__(self, uri=None):
        self._uri = uri or NEO4J_URI
        self._driver = GraphDatabase.driver(
            self._uri, auth=(NEO4J_USER, NEO4J_PASSWORD)
        )
        self.vector_mode = resolve_vector_mode()

    def assert_write_target(self):
        assert_writable()
        assert_build_uri(self._uri)

    @property
    def driver(self):
        return self._driver

    def close(self):
        self._driver.close()

    # ---- 写入 ----
    def seed_triples(self, triples, source_manual: str = None, source_line: int = None):
        """写入人工规则三元组，标签 :Entity，关系 :REL（文章 3.2 节）。

        v2: 支持 provenance——每个节点写入 source_manual + source_line 属性。
        v3: 支持关系元类型 meta——关系上写入 meta 属性（parallel/sequence/conditional/none）。
        v4（2026-10-20 双写修复）: 关系 MERGE 键从 {type, meta} 收窄为 {type}，meta 改为
            ON CREATE SET（仅创建时写入）。原因：MERGE 属性子集匹配语义下，对已存在的
            无 meta 同型边（历史建库形态/其他写库路径）带 meta 重 seed 会造出平行边
            （7690 曾因此积累 2,146 组孪生边，10-20 已清理）。干净图最终态与 v3 逐字节
            一致；脏图不再双写。meta 全库无读取方，first-writer-wins 语义安全。
        """
        self.assert_write_target()
        with self._driver.session() as s:
            for row in triples:
                if isinstance(row, dict):
                    subj, rel, obj = row.get("s"), row.get("r"), row.get("o")
                    meta = row.get("meta", "none")
                elif len(row) >= 4:
                    subj, rel, obj, meta = row[0], row[1], row[2], row[3]
                else:
                    # 兼容旧调用：3 元组 (s, r, o) → meta 默认 "none"
                    subj, rel, obj = row[0], row[1], row[2]
                    meta = "none"
                if subj is None or rel is None or obj is None:
                    continue
                if meta is None:
                    meta = "none"
                if source_manual:
                    s.run(
                        "MERGE (a:Entity {name:$s, source_manual:$src}) "
                        "ON CREATE SET a.source_line = $ln "
                        "ON MATCH SET a.source_line = $ln "
                        "MERGE (b:Entity {name:$o, source_manual:$src}) "
                        "ON CREATE SET b.source_line = $ln "
                        "ON MATCH SET b.source_line = $ln "
                        "MERGE (a)-[r:REL {type:$r}]->(b) "
                        "ON CREATE SET r.meta = $m",
                        s=subj, o=obj, r=rel, m=meta, src=source_manual, ln=source_line,
                    )
                else:
                    s.run(
                        "MERGE (a:Entity {name:$s}) "
                        "MERGE (b:Entity {name:$o}) "
                        "MERGE (a)-[r:REL {type:$r}]->(b) "
                        "ON CREATE SET r.meta = $m",
                        s=subj, o=obj, r=rel, m=meta,
                    )

    def upsert_auto_triples(self, triples):
        """写入 LLM 自动抽取三元组，标签 :__Entity__，关系按类型（场景A）。"""
        self.assert_write_target()
        with self._driver.session() as s:
            for subj, rel, obj in triples:
                s.run(
                    "MERGE (a:__Entity__ {id:$s}) "
                    "MERGE (b:__Entity__ {id:$o}) "
                    "MERGE (a)-[:REL {type:$r}]->(b)",
                    s=subj, o=obj, r=rel,
                )

    def upsert_triples_with_provenance(self, triple_records: list, label: str = "__Entity__v2"):
        """写入带溯源信息的三元组（v2 新增）。

        参数:
            triple_records: [{"s":..., "r":..., "o":..., "source":..., "chunk_id":...}, ...]
            label: 节点标签（默认 :__Entity__v2，避免覆盖旧数据）
        """
        _validate_label(label)
        self.assert_write_target()
        with self._driver.session() as s:
            for rec in triple_records:
                s.run(
                    f"MERGE (a:{label} {{id:$s}}) "  # label 经白名单验证，安全
                    f"ON CREATE SET a.source_manual = $src, a.source_chunk = $cid "
                    f"MERGE (b:{label} {{id:$o}}) "  # label 经白名单验证，安全
                    f"ON CREATE SET b.source_manual = $src, b.source_chunk = $cid "
                    f"MERGE (a)-[:REL {{type:$r}}]->(b)",
                    s=rec["s"], o=rec["o"], r=rec["r"],
                    src=rec.get("source", ""), cid=rec.get("chunk_id", -1),
                )

    # ---- 读取 ----
    def get_all_entities(self, label="Entity"):
        _validate_label(label)
        with self._driver.session() as s:
            return [r["name"] for r in s.run(f"MATCH (n:{label}) RETURN n.name AS name")]

    def get_graph(self, label="Entity"):
        """返回 (nodes, edges) 供 NetworkX 社区发现。"""
        _validate_label(label)
        with self._driver.session() as s:
            nodes = [r["name"] for r in s.run(f"MATCH (n:{label}) RETURN n.name AS name")]
            edges = [(r["s"], r["r"], r["o"]) for r in
                     s.run(f"MATCH (n:{label})-[r:REL]->(m:{label}) "
                           f"RETURN n.name AS s, r.type AS r, m.name AS o")]
        return nodes, edges

    def expand_paths(self, matched, label="Entity", rel_label="REL", max_hop=4, limit=10):
        """以匹配实体为起点展开 1~max_hop 跳路径（文章 4.1/4.2 Cypher）。"""
        if not matched:
            return []
        _validate_label(label)
        _validate_rel_type(rel_label)
        max_hop = int(max_hop)        # 防注入：显式转整数
        # DISTINCT（2026-10-20）：平行边（同 a,b,type 的重复 :REL）会产出投影完全相同
        # 的路径行，挤占 LIMIT 路径槽；DISTINCT 按 (node_names, rel_types) 折叠。
        # 注：本方法当前无调用方（生产路径走 engines._expand_paths_per_depth），
        # 此为防御性修复，保持与文章 4.1/4.2 Cypher 对应。
        cypher = (f"MATCH path = (n:{label})-[:{rel_label}*1..{max_hop}]->(m:{label}) "
                  f"WHERE n.name IN $matched "
                  f"RETURN DISTINCT [node in nodes(path) | node.name] AS node_names, "
                  f"[rel in relationships(path) | rel.type] AS rel_types "
                  f"LIMIT $limit")
        with self._driver.session() as s:
            rows = s.run(cypher, matched=matched, limit=int(limit))
            return [(r["node_names"], r["rel_types"]) for r in rows]

    def load_entity_embeddings(self, label="Entity"):
        """场景B：预计算 :Entity.name 的 embedding（文章 4.2 节，内存余弦）。"""
        from config import get_embed_model
        embed = get_embed_model()
        names = self.get_all_entities(label)
        embs = [embed.get_text_embedding(n) for n in names]
        return names, embs

    def structured_query(self, cypher: str, param_map: dict = None) -> list[dict]:
        """执行参数化 Cypher 查询，返回记录列表。"""
        with self._driver.session() as session:
            result = session.run(cypher, param_map or {})
            return [dict(record) for record in result]

    def upsert_community(self, community_id: str, label: str, summary: str,
                         report: str, size: int, chunk_count: int, chunk_ids: list,
                         source_manual: str):
        """创建或更新 :Community 节点，存储社区摘要信息。

        社区身份是 (id, source_manual) 而不是 id：只按 id MERGE 时，第二本手册的同号社区
        会覆盖第一本手册的那一个，摘要复用也会跨手册抄串（2026-09-06 的 IS620F 缓存实测
        13/63 条摘要来自另一本手册）。source_manual 必填且不允许多个调用点各猜一份，
        因此这里不设默认值——漏传即在写边界上响亮地失败，而不是静默落进错误命名空间。
        """
        self.assert_write_target()
        if not source_manual:
            raise ValueError(
                "upsert_community 需要非空 source_manual：社区层按 (id, source_manual) 定身份")
        with self._driver.session() as s:
            s.run(
                "MERGE (c:Community {id:$id, source_manual:$sm}) "
                "ON CREATE SET c.label = $label, c.summary = $summary, "
                "c.report = $report, c.size = $size, c.chunk_count = $chunk_count, "
                "c.chunk_ids = $chunk_ids "
                "ON MATCH SET c.label = $label, c.summary = $summary, "
                "c.report = $report, c.size = $size, c.chunk_count = $chunk_count, "
                "c.chunk_ids = $chunk_ids",
                id=community_id, label=label, summary=summary,
                report=report, size=size, chunk_count=chunk_count,
                chunk_ids=chunk_ids, sm=source_manual,
            )

    def get_community_summaries(self, source_manual: str = None) -> list[dict]:
        """获取 :Community 摘要；给 source_manual 时只取该本手册的。"""
        with self._driver.session() as s:
            if source_manual:
                recs = s.run(
                    "MATCH (c:Community {source_manual: $sm}) RETURN c.id AS id, "
                    "c.label AS label, c.source_manual AS source_manual, "
                    "c.summary AS summary, c.report AS report, c.size AS size, "
                    "c.chunk_count AS chunk_count, c.chunk_ids AS chunk_ids", sm=source_manual)
            else:
                recs = s.run(
                    "MATCH (c:Community) RETURN c.id AS id, c.label AS label, "
                    "c.source_manual AS source_manual, "
                    "c.summary AS summary, c.report AS report, c.size AS size, "
                    "c.chunk_count AS chunk_count, c.chunk_ids AS chunk_ids")
            return [dict(r) for r in recs]


# 兼容旧调用名
get_property_graph_store = GraphStore
def get_raw_driver(uri=None):
    return GraphDatabase.driver(uri or NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
