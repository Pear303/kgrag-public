"""混合检索引擎（向量 + BM25 关键词）。

将向量相似度检索与 BM25 关键词检索融合，通过加权求和重新排序，
同时保留 LLM 合成能力，完全兼容现有 VectorRAGQueryEngine 接口。
"""
from typing import List, Optional
from collections import OrderedDict

from config import TOP_K


class HybridRetriever:
    """混合检索：向量语义 + BM25 关键词加权融合。

    工作流：
        1. 对同一 query 分别执行向量检索和 BM25 检索
        2. 将两路分数归一化到 [0, 1]
        3. final_score = vector_weight * vec_norm_score + bm25_weight * bm25_norm_score
        4. 去重、重新排序、取 top-k
        5. 将召回片段交给 LLM 合成答案（复用 VectorRAGQueryEngine 的 prompt 模板）
    """

    def __init__(
        self,
        vector_engine,
        bm25_retriever,
        llm,
        texts: Optional[List[str]] = None,
        top_k: int = TOP_K,
        vector_weight: float = 0.7,
        bm25_weight: float = 0.3,
        mode: str = "hybrid",
    ):
        """
        Args:
            vector_engine: VectorRAGQueryEngine 实例（提供 query/retrieve_only）
            bm25_retriever: BM25Retriever 实例
            llm: LLM 实例
            texts: 原始 chunk 文本列表（用于 BM25 索引，也可直接复用 bm25_retriever._corpus）
            top_k: 最终返回数量
            vector_weight: 向量检索权重，默认 0.7
            bm25_weight: BM25 检索权重，默认 0.3
            mode: 检索模式——"vector" / "bm25" / "hybrid"
        """
        self.vec_eng = vector_engine
        self.bm25 = bm25_retriever
        self.llm = llm
        self.top_k = top_k
        self.vector_weight = vector_weight
        self.bm25_weight = bm25_weight
        self.mode = mode

        if texts and not bm25_retriever.is_indexed:
            bm25_retriever.build_index(texts)

    def query(self, question: str) -> dict:
        """执行混合检索并合成答案，接口与 VectorRAGQueryEngine.query() 完全兼容。

        Returns:
            {
                "answer": str,
                "contexts": [str, ...],
                "scores": [float, ...],
                "mode": str,           # 本次使用的检索模式
                "source": [str, ...],  # 每个 chunk 的来源标记（vec/bm25/hybrid）
            }
        """
        if self.mode == "vector":
            merged, sources, scores = self._vector_only(question)
        elif self.mode == "bm25":
            merged, sources, scores = self._bm25_only(question)
        else:
            merged, sources, scores = self._hybrid_merge(question)

        contexts = [self._get_text(idx) for idx in merged]

        if not contexts:
            return {
                "answer": "未检索到相关内容。",
                "contexts": [],
                "scores": [],
                "mode": self.mode,
                "source": [],
            }

        prompt = (
            "你是一名工业设备故障诊断专家。严格遵守以下约束：\n"
            "1. 仅使用下方【手册片段】中的信息回答，禁止编造、扩展或引入外部知识。\n"
            "2. 每条结论必须能在【手册片段】中找到直接依据，无法确证的条目不得输出。\n"
            "3. 如果片段信息不足以完整回答，明确说明「根据手册未找到相关答案」，不要猜测。\n"
            "4. 答案需结构化：先列出故障现象→可能原因→解决方法，逐条给出原文依据。\n\n"
            "【手册片段】\n"
            + "\n---\n".join(contexts[:3])
            + f"\n\n【问题】{question}\n\n【答案】"
        )
        answer = str(self.llm.complete(prompt))

        return {
            "answer": answer,
            "contexts": contexts,
            "scores": scores,
            "mode": self.mode,
            "source": sources,
        }

    def retrieve_only(self, question: str) -> list:
        """仅检索不合成，返回 (chunk_id, score, source) 列表。"""
        if self.mode == "vector":
            merged, sources, scores = self._vector_only(question)
        elif self.mode == "bm25":
            merged, sources, scores = self._bm25_only(question)
        else:
            merged, sources, scores = self._hybrid_merge(question)

        return [(idx, scores[i], sources[i]) for i, idx in enumerate(merged)]

    def _get_text(self, idx) -> str:
        """从向量引擎的 retriever 中按索引或 node_id 获取原始文本。

        两种调用方：
        - bm25/hybrid 模式：整数 corpus 索引 → 按位置取 docstore；
        - vector 模式：字符串 node_id → 直接按 dict 键取。
        越界/缺失时回退到 bm25._corpus，仍缺失返回空串。
        """
        try:
            nodes = self.vec_eng.retriever._index.docstore.docs
            if isinstance(idx, str):
                node = nodes.get(idx)
                if node is not None:
                    return node.text
            else:
                node_id = list(nodes.keys())[idx]
                return nodes[node_id].text
        except (IndexError, KeyError, AttributeError, TypeError):
            if isinstance(idx, int) and idx < len(self.bm25._corpus):
                return self.bm25._corpus[idx]
            return ""

    def _vector_only(self, question: str):
        """纯向量检索。"""
        nodes = self.vec_eng.retriever.retrieve(question)
        # 用 postprocessor 过滤
        filtered = []
        for n in nodes:
            try:
                score = float(getattr(n, "score", 0) or n.get_score())
            except Exception:
                score = 0.0
            if score >= getattr(self.vec_eng, "similarity_threshold", 0.0):
                filtered.append(n)

        if not filtered:
            filtered = nodes[:self.top_k]

        idx_list = []
        sources = []
        scores = []
        for n in filtered[:self.top_k]:
            try:
                node_id = n.node.node_id if hasattr(n, "node") else str(n.node_id)
            except AttributeError:
                node_id = str(id(n))
            try:
                doc_id = node_id
                idx_list.append(doc_id)
            except Exception:
                idx_list.append(str(id(n)))
            try:
                score = float(getattr(n, "score", 0) or n.get_score())
            except Exception:
                score = 0.0
            sources.append("vec")
            scores.append(score)

        return idx_list, sources, scores

    def _bm25_only(self, question: str):
        """纯 BM25 检索。"""
        results = self.bm25.search(question, top_k=self.top_k)
        idx_list = [r[0] for r in results]
        scores = [r[1] for r in results]
        sources = ["bm25"] * len(idx_list)

        return idx_list, sources, scores

    def _hybrid_merge(self, question: str):
        """向量 + BM25 两路融合。

        1. 各取 top_k 结果
        2. 分别归一化到 [0, 1]
        3. 按 chunk_id 合并，加权求和
        4. 重排序取 top_k
        """
        # --- 向量路 ---
        vec_nodes = self.vec_eng.retriever.retrieve(question)
        vec_items = []  # [(doc_node_id, score)]
        for n in vec_nodes[:self.top_k]:
            try:
                score = float(getattr(n, "score", 0) or n.get_score())
            except Exception:
                score = 0.0
            # 找到此 node 在索引中的序号
            try:
                node_id = n.node.node_id if hasattr(n, "node") else str(n.node_id)
                # 从 text 反查 idx
                node_text = n.node.text if hasattr(n, "node") else n.text
                for i, t in enumerate(self.bm25._corpus):
                    if t[:50] == node_text[:50]:
                        vec_items.append((i, score))
                        break
                else:
                    vec_items.append((str(node_id), score))
            except Exception:
                vec_items.append((str(id(n)), score))

        # --- BM25 路 ---
        bm25_results = self.bm25.search(question, top_k=self.top_k)
        bm25_items = [(r[0], r[1]) for r in bm25_results]

        # --- 归一化 ---
        vec_scores = [s for _, s in vec_items]
        bm25_scores = [s for _, s in bm25_items]
        vec_norm = self._normalize(vec_scores)
        bm25_norm = self._normalize(bm25_scores)

        # --- 合并 ---
        combined = OrderedDict()
        for (idx, _), norm_s in zip(vec_items, vec_norm):
            key = str(idx)
            combined[key] = combined.get(key, {"vec": 0, "bm25": 0})
            combined[key]["vec"] = max(combined[key]["vec"], norm_s)

        for (idx, _), norm_s in zip(bm25_items, bm25_norm):
            key = str(idx)
            combined[key] = combined.get(key, {"vec": 0, "bm25": 0})
            combined[key]["bm25"] = max(combined[key]["bm25"], norm_s)

        # --- 加权求和 ---
        merged = []
        for doc_id, scores_map in combined.items():
            v = scores_map["vec"]
            b = scores_map["bm25"]
            final = self.vector_weight * v + self.bm25_weight * b
            is_both = v > 0 and b > 0
            src = "hybrid" if is_both else ("vec" if v > 0 else "bm25")
            # 尝试恢复整数索引
            try:
                idx = int(doc_id)
            except (ValueError, TypeError):
                idx = doc_id
            merged.append((idx, final, src))

        merged.sort(key=lambda x: x[1], reverse=True)
        merged = merged[:self.top_k]

        idx_list = [m[0] for m in merged]
        scores = [m[1] for m in merged]
        sources = [m[2] for m in merged]

        return idx_list, sources, scores

    @staticmethod
    def _normalize(scores: List[float]) -> List[float]:
        """Min-Max 归一化到 [0, 1]，全零时返回全零。"""
        if not scores:
            return []
        mn = min(scores)
        mx = max(scores)
        if mx == mn:
            return [1.0] * len(scores) if mx > 0 else [0.0] * len(scores)
        return [(s - mn) / (mx - mn) for s in scores]


def build_hybrid_retriever(
    vector_engine,
    bm25_retriever,
    llm,
    texts: Optional[List[str]] = None,
    top_k: int = TOP_K,
    vector_weight: float = 0.7,
    bm25_weight: float = 0.3,
    mode: str = "hybrid",
):
    """构造混合检索器的工厂函数。"""
    return HybridRetriever(
        vector_engine=vector_engine,
        bm25_retriever=bm25_retriever,
        llm=llm,
        texts=texts,
        top_k=top_k,
        vector_weight=vector_weight,
        bm25_weight=bm25_weight,
        mode=mode,
    )
