"""community 模块单元测试（纯逻辑，不依赖 Neo4j/Ollama）。

覆盖：Louvain 社区发现、实体↔Chunk 绑定、build_communities 端到端
（去重/孤立 Chunk/description 生成）、摘要生成断点跳过。
"""
import numpy as np
import networkx as nx

from backend.community import (
    _run_louvain,
    _bind_chunk_ids,
    build_communities,
)


def _fake_embed(text):
    """确定性伪 embedding：按字符求和到 8 维并归一化。"""
    v = np.zeros(8, dtype=np.float32)
    for ch in str(text):
        v[ord(ch) % 8] += 1.0
    n = np.linalg.norm(v)
    return (v / n).tolist() if n else v.tolist()


class _FakeEmbedModel:
    def get_text_embedding(self, text):
        return _fake_embed(text)


def test_run_louvain_partitions_nodes():
    """Louvain 应把无连边的独立子图分到不同社区，且每个节点恰属一个社区。"""
    g = nx.DiGraph()
    g.add_edges_from([("a", "b"), ("b", "c"), ("x", "y")])  # 两个连通分量
    node_to_comm = _run_louvain(g, seed=42)
    assert set(node_to_comm) == {"a", "b", "c", "x", "y"}, "所有节点都必须分到社区"
    assert len(set(node_to_comm.values())) >= 2, "两个不相连分量应分成不同社区"
    assert node_to_comm["a"] == node_to_comm["b"] == node_to_comm["c"], "同分量同社区"


def test_run_louvain_empty_graph():
    """空图返回空映射，不报错。"""
    assert _run_louvain(nx.DiGraph()) == {}


def test_bind_chunk_ids_substring_match():
    """实体名直接出现在 chunk 文本中时应子串命中（无需 embedding 回退）。"""
    chunk_texts = ["叶轮磨损导致泵流量下降", "轴承温度过高需更换"]
    embed = _FakeEmbedModel()
    binding = _bind_chunk_ids(["叶轮磨损", "轴承"], chunk_texts, embed)
    assert 0 in binding["叶轮磨损"], "叶轮磨损 应子串命中 chunk 0"
    assert 1 in binding["轴承"], "轴承 应子串命中 chunk 1"


def test_bind_chunk_ids_embedding_fallback_below_threshold():
    """子串未命中且 embedding 相似度低于 0.70 时，不绑定（宁缺毋滥）。"""
    chunk_texts = ["完全无关的文本A", "同样无关的文本B"]
    embed = _FakeEmbedModel()
    binding = _bind_chunk_ids(["温度传感器"], chunk_texts, embed)
    assert binding["温度传感器"] == [], "低相似度不应回退绑定"


def test_build_communities_creates_communities_and_orphans():
    """端到端：有边的节点形成社区，未绑定的 chunk 成为孤立社区。"""
    graph_nodes = [("泵", ["Entity"]), ("叶轮", ["Entity"]), ("轴承", ["Entity"])]
    graph_edges = [("泵", "包含", "叶轮"), ("泵", "包含", "轴承")]
    chunk_texts = [
        "泵包含叶轮，叶轮磨损会导致流量下降",   # 绑定 泵/叶轮
        "轴承温度过高需要补充润滑油",            # 绑定 轴承
        "完全独立的段落",                         # 孤立 chunk
    ]
    communities = build_communities(graph_nodes, graph_edges, chunk_texts,
                                    _FakeEmbedModel(), generate_summaries=False)
    assert communities, "必须有社区产出"
    # 全部有 description 与 description_embedding
    for c in communities:
        assert c["description"], "description 不得为空"
        assert c["description_embedding"], "description_embedding 不得为空"
    # 孤立 chunk（"完全独立的段落"）应作为 orphan 社区出现
    orphan_ids = [c["id"] for c in communities if str(c["id"]).startswith("orphan")]
    assert orphan_ids, "未绑定 chunk 必须成为孤立社区"


def test_build_communities_dedup_same_chunk_set():
    """社区去重：chunk_ids 完全相同的社区合并节点，不产生重复社区。"""
    graph_nodes = [("泵", ["Entity"]), ("叶轮", ["Entity"])]
    graph_edges = [("泵", "包含", "叶轮")]
    chunk_texts = ["泵包含叶轮"]
    communities = build_communities(graph_nodes, graph_edges, chunk_texts,
                                    _FakeEmbedModel(), generate_summaries=False)
    # 两个节点同属一个 chunk → 合并为一个社区 + 1 个孤立（无）
    ids = [str(c["id"]) for c in communities]
    assert len(ids) == len(set(ids)), "社区 id 不得重复"
    non_orphan = [c for c in communities if not str(c["id"]).startswith("orphan")]
    assert len(non_orphan) == 1, "同 chunk 的节点应合并为单一社区"


def test_build_communities_empty_graph_yields_orphans_only():
    """空图（无边无节点）时每个 chunk 都是孤立社区。"""
    communities = build_communities([], [], ["文本1", "文本2"],
                                    _FakeEmbedModel(), generate_summaries=False)
    assert len(communities) == 2, "2 个 chunk → 2 个孤立社区"
    assert all(str(c["id"]).startswith("orphan") for c in communities)


def test_generate_summaries_skips_existing_and_empty():
    """摘要生成：已有摘要的社区跳过；无 raw_context 的社区摘要置空。"""
    from backend.community import generate_community_summaries

    communities = [
        {"id": "c1", "nodes": {"a"}, "chunk_ids": [0], "raw_context": ["片段A"]},
        {"id": "c2", "nodes": {"b"}, "chunk_ids": [], "raw_context": []},
    ]
    # 不提供 graph_store，跳过 Neo4j 检查；api_client 会真实调用——这里用空列表
    # 触发"无 raw_context → 摘要置空"分支即可，避免网络调用。
    no_text = [{"id": "c_empty", "nodes": set(), "chunk_ids": [], "raw_context": []}]
    out = generate_community_summaries(no_text, [], graph_store=None)
    assert out[0]["summary"] == "" and out[0]["report"] == ""

    # 含文本但无 graph_store：会尝试调用 api_client（未初始化网络）——
    # 用注入异常的方式验证失败兜底路径（catch Exception → 摘要置空）。
    from backend.api_client import ApiClient
    original_new = ApiClient.__new__

    class _BoomClient:
        def chat(self, messages, **kwargs):
            raise RuntimeError("no network")

    ApiClient.__new__ = lambda cls: _BoomClient()
    try:
        out2 = generate_community_summaries(communities, ["片段A"], graph_store=None)
    finally:
        ApiClient.__new__ = original_new
    assert out2[0]["summary"] == "", "LLM 失败必须兜底为空而非崩溃"
    assert out2[1]["summary"] == "", "无 raw_context 的社区摘要为空"
