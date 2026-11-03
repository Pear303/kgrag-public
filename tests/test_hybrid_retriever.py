"""hybrid_retriever 模块单元测试（fake 向量/BM25/LLM 依赖，无真实索引）。

覆盖三种检索模式（vector / bm25 / hybrid）的 query 与 retrieve_only、
_init 的索引构建分支、_normalize 静态方法、_get_text 的 corpus 兜底，
以及 build_hybrid_retriever 工厂。
"""
from types import SimpleNamespace

import pytest

from backend.hybrid_retriever import HybridRetriever, build_hybrid_retriever


# ── fakes ────────────────────────────────────────────────────────────

class _FakeNode:
    """同时模拟 NodeWithScore（retrieve 返回，含 .node.*）与 docstore 里的 TextNode（.text/.node_id）。"""

    def __init__(self, node_id, text, score=0.9):
        self.node_id = node_id
        self.text = text
        self.node = SimpleNamespace(node_id=node_id, text=text)
        self.score = score

    def get_score(self):
        return self.score


class _FakeVec:
    def __init__(self, nodes, similarity_threshold=0.0):
        self.similarity_threshold = similarity_threshold
        self.retriever = SimpleNamespace(
            _index=SimpleNamespace(
                docstore=SimpleNamespace(
                    docs={n.node.node_id: n for n in nodes},
                ),
            ),
            retrieve=lambda q: nodes,
        )


class _FakeBM25:
    def __init__(self, corpus, indexed=True):
        self._corpus = corpus
        self.is_indexed = indexed
        self.built_texts = None
        self.search_results = []

    def build_index(self, texts):
        self.built_texts = texts
        self.is_indexed = True

    def search(self, q, top_k=10):
        return self.search_results


class _FakeLLM:
    def __init__(self, resp="合成答案"):
        self._resp = resp

    def complete(self, prompt):
        return self._resp


# ── _normalize ───────────────────────────────────────────────────────

def test_normalize_empty():
    assert HybridRetriever._normalize([]) == []


def test_normalize_all_zero():
    assert HybridRetriever._normalize([0.0, 0.0]) == [0.0, 0.0]


def test_normalize_all_same_positive():
    assert HybridRetriever._normalize([5.0, 5.0]) == [1.0, 1.0]


def test_normalize_range():
    assert HybridRetriever._normalize([1.0, 3.0, 5.0]) == [0.0, 0.5, 1.0]


# ── __init__ 索引构建 ────────────────────────────────────────────────

def test_init_builds_index_when_not_indexed():
    """texts 传入且 BM25 未索引 → 调用 build_index。"""
    bm = _FakeBM25(["a", "b"], indexed=False)
    hr = HybridRetriever(_FakeVec([]), bm, _FakeLLM(), texts=["a", "b"])
    assert bm.built_texts == ["a", "b"]
    assert bm.is_indexed is True


def test_init_skips_build_when_indexed():
    """BM25 已索引 → 不再 build_index。"""
    bm = _FakeBM25(["a", "b"], indexed=True)
    HybridRetriever(_FakeVec([]), bm, _FakeLLM(), texts=["a", "b"])
    assert bm.built_texts is None


# ── query：hybrid 融合 ───────────────────────────────────────────────

def test_query_hybrid_merges_vec_and_bm25():
    """hybrid：向量与 BM25 命中同一 corpus 索引 → 加权融合 + source 标记。"""
    nodes = [
        _FakeNode("n0", "轴承温度过高原因", score=0.9),
        _FakeNode("n1", "润滑不良导致振动", score=0.7),
    ]
    corpus = ["轴承温度过高原因", "润滑不良导致振动", "叶轮流道磨损"]
    bm = _FakeBM25(corpus)
    bm.search_results = [(0, 0.8), (1, 0.6)]
    hr = HybridRetriever(_FakeVec(nodes), bm, _FakeLLM(), top_k=2, mode="hybrid")

    out = hr.query("轴承温度过高")
    assert out["answer"] == "合成答案"
    assert out["contexts"] == ["轴承温度过高原因", "润滑不良导致振动"]
    assert out["mode"] == "hybrid"
    # 0 号双路归一化后均为 1.0 → hybrid；1 号两路归一化均为 0 → bm25（v=0 兜底标记）
    assert out["source"] == ["hybrid", "bm25"]
    assert out["scores"][0] == pytest.approx(1.0)


# ── query：vector 模式 ───────────────────────────────────────────────

def test_query_vector_mode_uses_threshold():
    """vector：低于 similarity_threshold 的节点被过滤。"""
    nodes = [
        _FakeNode("n0", "轴承温度过高原因", score=0.9),
        _FakeNode("n1", "无关片段", score=0.2),
    ]
    corpus = ["轴承温度过高原因", "无关片段"]
    bm = _FakeBM25(corpus)
    hr = HybridRetriever(
        _FakeVec(nodes, similarity_threshold=0.5), bm, _FakeLLM(),
        top_k=5, mode="vector",
    )
    out = hr.query("轴承温度过高")
    assert out["contexts"] == ["轴承温度过高原因"], "低于阈值的节点应被过滤"
    assert out["source"] == ["vec"]


def test_query_vector_mode_returns_text_by_node_id():
    """vector：_get_text 按字符串 node_id 直取 docstore 文本（回归：此前按位置索引会 TypeError）。"""
    nodes = [
        _FakeNode("n0", "轴承温度过高原因", score=0.9),
        _FakeNode("n1", "润滑不良导致振动", score=0.7),
    ]
    bm = _FakeBM25(["轴承温度过高原因", "润滑不良导致振动"])
    hr = HybridRetriever(_FakeVec(nodes), bm, _FakeLLM(), top_k=5, mode="vector")
    out = hr.query("轴承")
    assert out["contexts"] == ["轴承温度过高原因", "润滑不良导致振动"]
    assert out["scores"] == [0.9, 0.7]


# ── query：bm25 模式 + corpus 兜底 ───────────────────────────────────

def test_query_bm25_mode_falls_back_to_corpus():
    """bm25：越界索引从 _corpus 兜底取文本。"""
    nodes = [
        _FakeNode("n0", "轴承温度过高原因", score=0.9),
        _FakeNode("n1", "润滑不良导致振动", score=0.7),
    ]
    corpus = ["轴承温度过高原因", "润滑不良导致振动", "叶轮流道磨损"]
    bm = _FakeBM25(corpus)
    bm.search_results = [(0, 0.9), (2, 0.5)]
    hr = HybridRetriever(_FakeVec(nodes), bm, _FakeLLM(), top_k=5, mode="bm25")
    out = hr.query("磨损")
    assert out["contexts"] == ["轴承温度过高原因", "叶轮流道磨损"]
    assert out["source"] == ["bm25", "bm25"]


def test_query_empty_returns_placeholder():
    """无任何命中 → 返回占位文案与空列表。"""
    bm = _FakeBM25([])
    hr = HybridRetriever(_FakeVec([]), bm, _FakeLLM(), mode="vector")
    out = hr.query("完全无关")
    assert out["answer"] == "未检索到相关内容。"
    assert out["contexts"] == []
    assert out["scores"] == []


# ── retrieve_only ────────────────────────────────────────────────────

def test_retrieve_only_returns_triples():
    """retrieve_only 返回 (chunk_id, score, source) 三元组列表。"""
    bm = _FakeBM25(["a", "b"])
    bm.search_results = [(0, 0.8), (1, 0.6)]
    hr = HybridRetriever(_FakeVec([]), bm, _FakeLLM(), top_k=5, mode="bm25")
    triples = hr.retrieve_only("问题")
    assert triples == [(0, 0.8, "bm25"), (1, 0.6, "bm25")]


# ── 工厂 ─────────────────────────────────────────────────────────────

def test_build_hybrid_retriever_factory():
    eng = build_hybrid_retriever(_FakeVec([]), _FakeBM25([]), _FakeLLM(), mode="bm25")
    assert isinstance(eng, HybridRetriever)
    assert eng.mode == "bm25"
