"""router 模块单元测试（fake 依赖，不触发真实 LLM/Neo4j）。

覆盖 SimpleRouter 微观/宏观分支、HybridQueryEngine 的 chunk 检索两种路径
（bm25 混合 vs 纯向量回退）、两个 build_* 工厂函数。
"""
from unittest.mock import patch

from backend.router import (
    HybridQueryEngine,
    SimpleRouter,
    build_hybrid_engine,
    build_router,
)


# ── fakes ────────────────────────────────────────────────────────────

class _FakeLLM:
    """模拟 LlamaIndex LLM：complete() 返回预设文本。"""

    def __init__(self, response="A"):
        self._resp = response

    def complete(self, prompt):
        return self._resp


class _FakeGraph:
    """模拟图引擎：query 返回 [(names, rels), ...]。"""

    def __init__(self, paths=None):
        self.paths = paths or []

    def query(self, q):
        return self.paths


class _FakeComm:
    """模拟社区引擎：query 返回 (matched, contexts)。"""

    def __init__(self, matched=None, contexts=None):
        self.matched = matched or []
        self.contexts = contexts or []

    def query(self, q):
        return self.matched, self.contexts


class _FakeVec:
    """模拟向量引擎：query 返回 contexts + scores。"""

    def __init__(self, contexts=None, scores=None):
        self.contexts = contexts or []
        self.scores = scores or []

    def query(self, q):
        return {"contexts": self.contexts, "scores": self.scores}


# ── SimpleRouter ─────────────────────────────────────────────────────

def test_simple_router_macro_branch():
    """LLM 返回 B → 宏观分支，走社区引擎并统计命中数。"""
    comm = _FakeComm(["c1"], ["ctx1"])
    router = SimpleRouter(_FakeLLM("B"), _FakeGraph(), comm)
    out = router.query("如何安排设备维护？")
    assert out["mode"] == "macro"
    assert out["communities_matched"] == 1
    assert out["chunks_recalled"] == 1
    assert "社区分析完成" in out["answer"]


def test_simple_router_macro_contains_keyword():
    """LLM 输出含"宏观"字样（即使无 B）仍判宏观。"""
    comm = _FakeComm(["c1"], ["ctx1"])
    router = SimpleRouter(_FakeLLM("应该走宏观引擎"), _FakeGraph(), comm)
    assert router.query("维护建议")["mode"] == "macro"


def test_simple_router_micro_branch_builds_chains():
    """LLM 返回 A → 微观分支，图谱路径格式化为因果链。"""
    graph = _FakeGraph([
        (["泵", "叶轮"], ["包含"]),
        (["叶轮", "磨损"], ["可能原因"]),
    ])
    router = SimpleRouter(_FakeLLM("A"), graph, _FakeComm())
    out = router.query("叶轮磨损原因？")
    assert out["mode"] == "micro"
    assert out["path_count"] == 2
    assert "泵 --[包含]--> 叶轮" in out["answer"]
    assert "叶轮 --[可能原因]--> 磨损" in out["answer"]


def test_simple_router_micro_empty_paths_placeholder():
    """微观分支无路径 → 返回占位提示而非空串。"""
    router = SimpleRouter(_FakeLLM("A"), _FakeGraph([]), _FakeComm())
    out = router.query("完全无关问题")
    assert out["mode"] == "micro"
    assert out["path_count"] == 0
    assert "未找到相关图谱路径" in out["answer"]


def test_build_router_returns_simple_router():
    """build_router 工厂返回 SimpleRouter 实例。"""
    r = build_router(_FakeLLM("A"), _FakeGraph(), _FakeComm())
    assert isinstance(r, SimpleRouter)


# ── HybridQueryEngine ────────────────────────────────────────────────

def test_hybrid_engine_without_bm25_falls_back_to_vec():
    """无 bm25_retriever → chunk_retriever=None，走 vec_eng.query 回退。"""
    vec = _FakeVec(["片段A", "片段B"], [0.9, 0.8])
    graph = _FakeGraph([(["泵", "叶轮"], ["包含"])])
    eng = HybridQueryEngine(vec, graph, _FakeLLM("融合答案"), bm25_retriever=None)
    assert eng.chunk_retriever is None
    out = eng.query("泵叶轮问题")
    assert out["vec_contexts"] == ["片段A", "片段B"]
    assert out["vec_scores"] == [0.9, 0.8]
    assert out["graph_path_count"] == 1
    assert out["retriever_mode"] == eng.retriever_mode


def test_hybrid_engine_with_bm25_uses_chunk_retriever():
    """有 bm25 → 构造 HybridRetriever 并用其 query 结果作为向量侧。"""
    with patch("backend.hybrid_retriever.HybridRetriever") as HR:
        hr_inst = HR.return_value
        hr_inst.query.return_value = {"contexts": ["chunk1"], "scores": [0.7]}
        vec = _FakeVec(["x"], [1.0])
        eng = HybridQueryEngine(
            vec, _FakeGraph(), _FakeLLM("答案"),
            bm25_retriever=object(), texts=["文本"],
        )
        assert eng.chunk_retriever is not None
        out = eng.query("问题")
        assert out["vec_contexts"] == ["chunk1"], "应使用 chunk_retriever 结果"
        assert out["vec_scores"] == [0.7]


def test_hybrid_engine_empty_contexts_placeholder():
    """双来源均空 → prompt 使用占位文案，仍返回答案。"""
    eng = HybridQueryEngine(_FakeVec([], []), _FakeGraph([]), _FakeLLM("无法确定"),
                            bm25_retriever=None)
    out = eng.query("无关问题")
    assert out["graph_path_count"] == 0
    assert isinstance(out["answer"], str)


def test_build_hybrid_engine():
    """build_hybrid_engine 工厂返回 HybridQueryEngine 实例。"""
    eng = build_hybrid_engine(_FakeLLM("x"), _FakeVec(["c"], [0.5]), _FakeGraph())
    assert isinstance(eng, HybridQueryEngine)
