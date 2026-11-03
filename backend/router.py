"""智能路由 + 混合检索引擎（修复版）。

1. SimpleRouter: LLM 直接分类（替换 LLMSingleSelector，修复 embedding 类型 bug）
2. HybridQueryEngine: 向量 RAG + KG-RAG 混合检索与答案融合（自学文档 6.6-2）
3. 支持 --retrieval-mode 切换向量/BM25/混合检索
"""
from typing import List, Tuple, Optional

from config import VECTOR_WEIGHT, BM25_WEIGHT, RETRIEVAL_MODE


def _use_structured(engine, question, **kw):
    """优先调用统一协议 query_structured；无协议引擎回退旧 query()。

    返回 (RetrievalResult, True) 或 (None, False)。新引擎接入时路由代码
    无需按引擎类型分叉。
    """
    if hasattr(engine, "query_structured"):
        return engine.query_structured(question, **kw), True
    return None, False


class SimpleRouter:
    """轻量级路由：用 LLM 直接做微观/宏观分类，避免 LLMSingleSelector 的 embedding bug。"""

    MICRO_DESC = "用于查询具体部件的故障原因、维修步骤、特定参数等微观问题"
    MACRO_DESC = "用于总结、概括、系统级维护建议等宏观问题"

    def __init__(self, llm, graph_engine, community_engine):
        self.llm = llm
        self.graph_eng = graph_engine
        self.comm_eng = community_engine

    def query(self, question: str) -> dict:
        """路由查询：LLM 判断微观/宏观 → 执行对应引擎"""
        prompt = (
            "你是一个查询路由器。判断以下用户问题应该使用哪种引擎：\n\n"
            f"引擎A（微观）: {self.MICRO_DESC}\n"
            f"引擎B（宏观）: {self.MACRO_DESC}\n\n"
            f"用户问题: {question}\n\n"
            "只输出一个字母 A 或 B，不要输出其他内容："
        )
        response = str(self.llm.complete(prompt)).strip().upper()

        if "B" in response or "宏观" in response:
            r, _ = _use_structured(self.comm_eng, question)
            if r is not None:
                return {
                    "mode": "macro",
                    "communities_matched": r.matched_communities,
                    "chunks_recalled": len(r.contexts),
                    "answer": f"社区分析完成: 命中 {r.matched_communities} 个社区, 召回 {len(r.contexts)} 段原始文档"
                }
            matched, contexts = self.comm_eng.query(question)
            return {
                "mode": "macro",
                "communities_matched": len(matched),
                "chunks_recalled": len(contexts),
                "answer": f"社区分析完成: 命中 {len(matched)} 个社区, 召回 {len(contexts)} 段原始文档"
            }
        else:
            r, _ = _use_structured(self.graph_eng, question)
            paths = r.paths if r is not None else self.graph_eng.query(question)
            chains = []
            for path in paths:
                names, rels = path[0], path[1]
                dirs = path[2] if len(path) >= 3 else []
                chain = str(names[0])
                for i, rel in enumerate(rels):
                    if i + 1 < len(names):
                        d = dirs[i] if i < len(dirs) else "?"
                        if d == "-":
                            chain += f" <--[{rel}]-- {str(names[i + 1])}"
                        else:
                            chain += f" --[{rel}]--> {str(names[i + 1])}"
                chains.append(chain)
            return {
                "mode": "micro",
                "paths": chains,
                "path_count": len(chains),
                "answer": "\n".join(chains[:5]) if chains else "未找到相关图谱路径"
            }


class HybridQueryEngine:
    """混合检索：向量 RAG 检索 + KG-RAG 路径推理 + LLM 融合。

    利用向量 RAG 的「覆盖面广」和 KG-RAG 的「因果推理强」，
    将两种检索结果拼接为统一 context，交给 LLM 综合回答。

    支持三种 chunk 检索模式（通过 retriever_mode 控制）：
    - "vector": 纯向量余弦相似度
    - "bm25": 纯 BM25 关键词检索
    - "hybrid": 向量 + BM25 加权融合（默认）
    """

    def __init__(self, vector_engine, graph_engine, llm,
                 bm25_retriever=None, texts=None,
                 vector_weight=VECTOR_WEIGHT, bm25_weight=BM25_WEIGHT,
                 retriever_mode=RETRIEVAL_MODE):
        """
        Args:
            vector_engine: VectorRAGQueryEngine 实例
            graph_engine: ManualGraphQueryEngine 或 AutoGraphQueryEngine 实例
            llm: LlamaIndex LLM 实例
            bm25_retriever: 可选，BM25Retriever 实例（启用混合检索时需要）
            texts: 可选，原始 chunk 文本列表（用于 BM25 索引）
            vector_weight: 混合检索中向量权重，默认 0.7
            bm25_weight: 混合检索中 BM25 权重，默认 0.3
            retriever_mode: chunk 检索模式——"vector" / "bm25" / "hybrid"
        """
        self.vec_eng = vector_engine
        self.graph_eng = graph_engine
        self.llm = llm
        self.retriever_mode = retriever_mode

        if bm25_retriever is not None:
            from config import TOP_K
            from backend.hybrid_retriever import HybridRetriever
            self.chunk_retriever = HybridRetriever(
                vector_engine=vector_engine,
                bm25_retriever=bm25_retriever,
                llm=llm,
                texts=texts,
                top_k=TOP_K,
                vector_weight=vector_weight,
                bm25_weight=bm25_weight,
                mode=retriever_mode,
            )
        else:
            self.chunk_retriever = None

    def query(self, question: str) -> dict:
        """混合查询：chunk检索（向量/BM25/混合） + 图谱 + LLM 融合"""
        if self.chunk_retriever is not None:
            chunk_result = self.chunk_retriever.query(question)
            vec_context_text = "\n---\n".join(chunk_result.get("contexts", [])[:5])
            vec_contexts = chunk_result.get("contexts", [])
            vec_scores = chunk_result.get("scores", [])
        else:
            vr, _ = _use_structured(self.vec_eng, question)
            if vr is not None:
                vec_contexts = vr.contexts
                vec_scores = vr.scores
            else:
                vec_result = self.vec_eng.query(question)
                vec_contexts = vec_result.get("contexts", [])
                vec_scores = vec_result.get("scores", [])
            vec_context_text = "\n---\n".join(vec_contexts[:5])

        graph_paths = None
        r, _ = _use_structured(self.graph_eng, question)
        if r is not None:
            graph_paths = r.paths
        else:
            graph_paths = self.graph_eng.query(question)
        graph_context = ""
        for path in graph_paths[:5]:
            names, rels = path[0], path[1]
            dirs = path[2] if len(path) >= 3 else []
            chain = str(names[0])
            for i, rel in enumerate(rels):
                if i + 1 < len(names):
                    d = dirs[i] if i < len(dirs) else "?"
                    if d == "-":
                        chain += f" <--[{rel}]-- {str(names[i + 1])}"
                    else:
                        chain += f" --[{rel}]--> {str(names[i + 1])}"
            graph_context += chain + "\n"

        # 组装混合 prompt
        prompt = (
            "你是一名工业设备故障诊断专家。你有两个知识来源：\n\n"
            "【来源1：文档片段（向量检索）】\n"
            f"{vec_context_text if vec_context_text else '（未检索到相关片段）'}\n\n"
            "【来源2：知识图谱路径（因果关系链）】\n"
            f"{graph_context if graph_context else '（未找到相关图谱路径）'}\n\n"
            "约束：\n"
            "1. 优先使用「知识图谱路径」中的因果关系做推理判断\n"
            "2. 用「文档片段」补充具体检修标准和数值参数\n"
            "3. 如果两个来源信息不一致，优先信任图谱路径的结构化推理\n"
            "4. 每条结论需标注信息来源（【向量】或【图谱】）\n"
            "5. 如果都无法确证，明确说明「根据当前知识库无法确定」\n\n"
            f"【问题】{question}\n\n【答案】"
        )
        answer = str(self.llm.complete(prompt))

        return {
            "answer": answer,
            "vec_contexts": vec_contexts,
            "vec_scores": vec_scores,
            "graph_paths": [str(p[0][0]) + "..." for p in graph_paths[:3]],
            "graph_path_count": len(graph_paths),
            "retriever_mode": self.retriever_mode,
        }


def build_router(llm, graph_engine, community_engine):
    """构造 SimpleRouter（修复版，替换原 LLMSingleSelector 方案）。"""
    return SimpleRouter(llm, graph_engine, community_engine)


def build_hybrid_engine(llm, vector_engine, graph_engine,
                       bm25_retriever=None, texts=None,
                       vector_weight=VECTOR_WEIGHT, bm25_weight=BM25_WEIGHT,
                       retriever_mode=RETRIEVAL_MODE):
    """构造混合检索引擎（支持向量/BM25/混合 chunk 检索）。"""
    return HybridQueryEngine(
        vector_engine, graph_engine, llm,
        bm25_retriever=bm25_retriever,
        texts=texts,
        vector_weight=vector_weight,
        bm25_weight=bm25_weight,
        retriever_mode=retriever_mode,
    )
