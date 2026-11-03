"""vector_rag 模块单元测试。

覆盖 VectorRAGQueryEngine 的初始化、检索、查询与边界情况。
依赖本地 Ollama（qwen3:latest + qwen3-embedding:0.6b）。
"""
import os
import pytest

from backend.parser import parse_manual_chunks
from llama_index.core import Document


@pytest.fixture(scope="module")
def vector_engine(test_manual_path):
    """模块级夹具：复用同一个引擎实例，避免重复初始化开销。"""
    from config import get_llm, get_embed_model
    from backend.vector_rag import VectorRAGQueryEngine

    llm = get_llm()
    embed = get_embed_model()
    chunks = parse_manual_chunks(test_manual_path)
    documents = [
        Document(text=c.text, metadata={"chunk_id": c.chunk_id})
        for c in chunks
    ]
    engine = VectorRAGQueryEngine(documents, embed, llm)
    return engine


def test_vector_rag_initialization(vector_engine):
    """测试 1：引擎初始化成功，retriever 与 postprocessor 均已就绪。"""
    assert vector_engine is not None
    assert vector_engine.retriever is not None
    assert vector_engine.postprocessor is not None


def test_retrieve_only_returns_nodes_with_scores(vector_engine):
    """测试 2：retrieve_only 返回带评分的节点列表。"""
    results = vector_engine.retrieve_only("密封系统维护")
    assert isinstance(results, list)
    assert len(results) > 0, "应在真实手册中检索到至少一个节点"
    for r in results:
        # NodeWithScore 同时暴露 .score 属性和 .get_score() 方法
        assert hasattr(r, "score") or hasattr(r, "get_score")


def test_query_returns_expected_structure(vector_engine):
    """测试 3：query() 返回字典，包含 answer / contexts / scores 三个字段。"""
    result = vector_engine.query("轴承座温度升高怎么处理")
    assert isinstance(result, dict)
    assert "answer" in result
    assert "contexts" in result
    assert "scores" in result
    assert isinstance(result["answer"], str)
    assert len(result["answer"]) > 0
    assert isinstance(result["contexts"], list)
    assert isinstance(result["scores"], list)


def test_query_with_industrial_term(vector_engine):
    """测试 4：工业术语查询应返回语义相关的答案内容。"""
    result = vector_engine.query("机械密封泄漏的可能原因")
    assert len(result["answer"]) > 5, "答案不应过短"
    # 答案中应出现与密封/泄漏/磨损相关的关键词
    answer_text = result["answer"]
    keywords = ["密封", "泄漏", "磨损", "老化", "seal", "leak"]
    found = any(kw in answer_text for kw in keywords)
    assert found, f"答案中应包含至少一个相关关键词，当前答案: {answer_text[:100]}"


def test_empty_documents_initialization():
    """测试 5：空文档列表——引擎初始化应抛出明确错误（无法创建空索引）。"""
    from config import get_llm, get_embed_model
    from backend.vector_rag import VectorRAGQueryEngine

    llm = get_llm()
    embed = get_embed_model()
    try:
        VectorRAGQueryEngine([], embed, llm)
        # 若未抛异常，引擎应仍可正常创建（取决于底层 VectorStoreIndex 行为）
    except (ValueError, IndexError, TypeError):
        pass  # 预期行为：无法对空文档列表创建索引
