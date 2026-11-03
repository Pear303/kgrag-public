"""rag_protocol 单元测试：协议字段与三引擎适配器。"""
import pytest

from backend.rag_protocol import RetrievalResult, RagEngine


def test_retrieval_result_defaults():
    r = RetrievalResult()
    assert r.answer == ""
    assert r.contexts == []
    assert r.paths == []
    assert r.scores == []
    assert r.matched_communities == 0
    assert r.metadata == {}


def test_manual_engine_query_structured():
    from backend.engines import ManualGraphQueryEngine

    class _Manual:
        def query(self, q, **kwargs):
            return [(["泵", "叶轮"], ["包含"], ["+"])]
    engine = ManualGraphQueryEngine.__new__(ManualGraphQueryEngine)
    engine.query = _Manual().query
    r = engine.query_structured("泵的结构")
    assert isinstance(r, RetrievalResult)
    assert r.paths == [(["泵", "叶轮"], ["包含"], ["+"])]


def test_community_engine_query_structured():
    from backend.engines import CommunityQueryEngine

    class _Comm:
        def query(self, q, **kwargs):
            return ["comm1"], ["正文1", "正文2"]
    engine = CommunityQueryEngine.__new__(CommunityQueryEngine)
    engine.query = _Comm().query
    r = engine.query_structured("密封系统")
    assert isinstance(r, RetrievalResult)
    assert r.contexts == ["正文1", "正文2"]
    assert r.matched_communities == 1


def test_vector_engine_query_structured():
    from backend.vector_rag import VectorRAGQueryEngine

    class _Vec:
        def query(self, q, **kwargs):
            return {"answer": "答案", "contexts": ["c1"], "scores": [0.9]}
    engine = VectorRAGQueryEngine.__new__(VectorRAGQueryEngine)
    engine.query = _Vec().query
    r = engine.query_structured("问题")
    assert r.answer == "答案"
    assert r.contexts == ["c1"]
    assert r.scores == [0.9]


def test_rag_engine_protocol_runtime_check():
    """验证实现类满足 RagEngine Protocol（不影响运行，仅结构校验）。"""
    from typing import runtime_checkable, Protocol
    from backend.engines import ManualGraphQueryEngine
    from backend.vector_rag import VectorRAGQueryEngine

    @runtime_checkable
    class _Proto(Protocol):
        def query_structured(self, question: str, **kwargs) -> RetrievalResult: ...

    for cls in (ManualGraphQueryEngine, VectorRAGQueryEngine):
        assert hasattr(cls, "query_structured"), f"{cls.__name__} 缺少 query_structured"
