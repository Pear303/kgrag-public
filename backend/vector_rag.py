"""纯向量 RAG 查询引擎（作为 KG-RAG 的对比基线）。

支持混合检索：精确 ID 匹配 + 语义检索兜底，解决 GSK983 等结构化手册的精确查询问题。
"""
from typing import Callable, Optional, List
import hashlib
import json
import os
import re

from llama_index.core import (StorageContext, VectorStoreIndex,
                              load_index_from_storage)
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.core.schema import NodeWithScore

from config import SIMILARITY_THRESHOLD, TOP_K
from backend.rag_protocol import RetrievalResult

# 索引缓存的代际标记：改分块/改 embedding 调用方式时 bump 它，旧缓存自动失效。
_VECTOR_INDEX_SCHEMA = "v1"


def _doc_fingerprint(documents: list) -> str:
    """文档集指纹：只按文本内容算，不需要 embedding，用来发现"书换了/分块变了"。

    用排序后的逐篇哈希聚合，这样节点存储顺序（uuid）不影响指纹。
    """
    per_doc = sorted(hashlib.sha256((d.text or "").encode("utf-8")).hexdigest()
                     for d in documents)
    h = hashlib.sha256()
    for x in per_doc:
        h.update(x.encode("ascii"))
    return h.hexdigest()


def _embed_signature(embed_model) -> str:
    """embedding 身份：模型名 + 向量维度。换模型或维度变了就必须重建。"""
    name = (getattr(embed_model, "model_id", None)
            or getattr(embed_model, "model_name", None)
            or getattr(embed_model, "model", None)
            or type(embed_model).__name__)
    dim = getattr(embed_model, "dimensions", None) or getattr(
        embed_model, "embed_dim", None)
    return f"{name}|{dim}"


class VectorRAGQueryEngine:
    """纯向量 RAG：embed → top-k 检索 → 相似度过滤 → LLM 合成答案。

    支持混合检索：精确 ID 匹配 + 语义检索兜底，解决 GSK983 等结构化手册的精确查询问题。

    与 KG-RAG 使用完全相同的 embedding 模型和文档切分，
    确保对比的公平性（唯一差异是「是否使用知识图谱」）。

    answer_generator: 可选，与 KG-RAG 共用同一答案生成器（默认 self.llm.complete）。
        评测公平性要求两系统用同一生成器，避免小模型 vs 大模型造成分数假象。

    修复 A1：相似度阈值默认取 config.SIMILARITY_THRESHOLD（与 KG-RAG 检索阈值
    同口径，此前固定 0.45 与 KG 的 0.5 不一致，导致 Vector 召回偏多）。
    """

    def __init__(
        self,
        documents: list,
        embed_model,
        llm,
        top_k: int = TOP_K,
        similarity_threshold: float = SIMILARITY_THRESHOLD,
        answer_generator: Optional[Callable[[str], str]] = None,
        context_truncator: Optional[Callable[[list], list]] = None,
        prompt_template: Optional[str] = None,
        index_cache_dir: Optional[str] = None,
        progress: Optional[Callable[[str, int, int], None]] = None,
    ):
        self.llm = llm
        self.top_k = top_k
        self.similarity_threshold = similarity_threshold
        self.answer_generator = answer_generator
        # 修复 KG-1：答案生成前对 contexts 做与 RAGAS 评分相同的截断
        # （默认 None = 只取前 3 条旧行为；评测脚本注入 _truncate_contexts 后，
        # 生成与评分看到同一份 context，消除两侧不对称）。
        self.context_truncator = context_truncator
        # 修复 KG-11：prompt 模板注入（评测脚本传入 evaluator._COT_PROMPT_TEMPLATE，
        # 与 KG-RAG 完全同一模板，消除"KG 走 CoT、Vector 走普通模板"的措辞不对称）。
        self.prompt_template = prompt_template

        # 修复 9.13-⑤：不再改全局 Settings（此前直接
        # Settings.embed_model = ... 污染进程级单例，注释却写"避免污染全局"，
        # 顺序变化会悄悄影响后续所有用 Settings 的模块）。embed_model/llm
        # 显式传给 from_documents，查询路径用 self.llm，均不依赖全局状态。
        #
        # index_cache_dir 给了就走"落盘一次、之后直接加载"：服务每次激活都要
        # 把整本书重新 embedding（实测 120 块 × ~131ms ≈ 16s），而同一轮次下
        # 文本和模型都不变，重算没有新信息。默认 None＝行为与改动前完全一致，
        # 评测链路不传这个参数，因此不受影响。
        self.index = self._build_or_load_index(
            documents, embed_model, llm, index_cache_dir, progress,
        )
        self.retriever = VectorIndexRetriever(
            index=self.index, similarity_top_k=top_k
        )
        self.postprocessor = SimilarityPostprocessor(
            similarity_cutoff=similarity_threshold
        )

    # ---------- Vector 索引落盘/加载 ----------
    def _build_or_load_index(self, documents, embed_model, llm, cache_dir,
                             progress=None):
        self.index_source = "off"
        self.index_reason = "未启用缓存"
        if not cache_dir:
            if progress:
                progress("vector", 0, 1)
            idx = VectorStoreIndex.from_documents(
                documents, embed_model=embed_model, llm=llm)
            if progress:
                progress("vector", 1, 1)
            return idx

        if progress:
            progress("vector", 0, 1)
        meta_path = os.path.join(cache_dir, "_meta.json")
        doc_fp = _doc_fingerprint(documents)
        embed_sig = _embed_signature(embed_model)

        loaded = None
        if not os.path.exists(meta_path):
            self.index_reason = "还没有落盘缓存，现建"
        if os.path.exists(meta_path):
            try:
                with open(meta_path, encoding="utf-8") as f:
                    meta = json.load(f)
                # 四道门任一不过都要重建，顺序固定（先便宜后贵）：
                # schema→文档指纹→模型身份→抽样向量值尺。最后那道是"值尺"，
                # 专防"模型名没变但换模型了"和跨进程向量不可复现。
                if meta.get("schema") != _VECTOR_INDEX_SCHEMA:
                    self.index_reason = (f"schema {meta.get('schema')}"
                                         f"!={_VECTOR_INDEX_SCHEMA}")
                elif meta.get("doc_fp") != doc_fp:
                    self.index_reason = "文档指纹变了"
                elif meta.get("embed_sig") != embed_sig:
                    self.index_reason = (f"embedding 身份变了 "
                                         f"{meta.get('embed_sig')} -> {embed_sig}")
                elif not self._probe_embedding(embed_model, documents, meta):
                    self.index_reason = "抽样向量对不上"
                else:
                    # 注意：embed_model 只能传给 load_index_from_storage（它进
                    # 索引构造参数）；StorageContext.from_defaults 不接这个参数，
                    # 不传的话索引会去实例化默认 OpenAI embedding 并报错。
                    sc = StorageContext.from_defaults(persist_dir=cache_dir)
                    loaded = load_index_from_storage(sc, embed_model=embed_model)
                    self.index_source = "cache"
                    self.index_reason = "四门全过"
            except Exception as e:
                # 缓存坏掉不许把服务起不来：退回去现场重建
                loaded = None
                self.index_source = "built"
                self.index_reason = f"加载失败 {type(e).__name__}: {e}"

        if loaded is not None:
            print(f"  [vec] 索引复用: {cache_dir}"
                  f"（{len(documents)} 块，跳过 embedding）", flush=True)
            if progress:
                progress("vector", 1, 1)
            return loaded

        index = VectorStoreIndex.from_documents(
            documents, embed_model=embed_model, llm=llm)
        self.index_source = "built"
        if self.index_reason != "未启用缓存":
            print(f"  [vec] 索引重建（{self.index_reason}）", flush=True)
        try:
            os.makedirs(cache_dir, exist_ok=True)
            index.storage_context.persist(persist_dir=cache_dir)
            sample = self._sample_vector(embed_model, documents)
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump({"schema": _VECTOR_INDEX_SCHEMA, "doc_fp": doc_fp,
                           "embed_sig": embed_sig, "n_docs": len(documents),
                           "sample_idx": 0, "sample_dim": len(sample),
                           "sample_embedding": sample},
                          f, ensure_ascii=False)
            print(f"  [vec] 索引落盘: {cache_dir}（{len(documents)} 块）", flush=True)
        except Exception as e:
            print(f"  [vec] 索引落盘失败（下次仍会重建）: {type(e).__name__}: {e}",
                  flush=True)
        if progress:
            progress("vector", 1, 1)
        return index

    @staticmethod
    def _sample_vector(embed_model, documents) -> list:
        """取第一篇的文本现场算一次向量，作为跨进程可比性的值尺样本。"""
        vec = embed_model.get_text_embedding(documents[0].text or "")
        return [float(x) for x in vec]

    def _probe_embedding(self, embed_model, documents, meta) -> bool:
        """把存下来的样本向量与现算的比一次；对不上就判定缓存不可复用。"""
        stored = meta.get("sample_embedding") or []
        if not stored:
            return False
        now = self._sample_vector(embed_model, documents)
        if len(now) != len(stored):
            return False
        return all(abs(a - b) <= 1e-6 for a, b in zip(now, stored))


    def _extract_identifiers(self, question: str) -> dict:
        """从问题中提取结构化标识符：报警号、参数号、DGN号等。
        
        Returns:
            dict: {
                'alarm_numbers': [...],  # 报警号
                'param_numbers': [...],  # 参数号
                'dgn_numbers': [...],    # DGN号
            }
        """
        patterns = {
            'alarm_numbers': [
                r'GSK983报警(\d{1,3})',
                r'报警(\d{3})',
                r'报警号?\s*(\d{1,3})',
            ],
            'param_numbers': [
                r'参数(\d{4}~?\d*)',
                r'参数(\d{4})',
                r'GSK983参数(\d{4}~?\d*)',
            ],
            'dgn_numbers': [
                r'DGN#?(\d+~?\d*)',
                r'DGN#?(\d+)',
                r'诊断DGN#?(\d+)',
            ],
        }
        
        identifiers = {}
        for key, patterns in patterns.items():
            found = []
            for pattern in patterns:
                matches = re.findall(pattern, question)
                found.extend(matches)
            if found:
                identifiers[key] = list(set(found))  # 去重
        return identifiers

    def _exact_match_retrieve(self, identifiers: dict) -> list:
        """基于精确标识符进行精确匹配检索。
        
        从索引中检索包含指定标识符的节点。
        """
        exact_nodes = []
        
        # 构建查询条件
        all_ids = []
        for ids in identifiers.values():
            all_ids.extend(ids)
        
        if not all_ids:
            return []
        
        # 使用 retriever 检索更多节点，然后过滤
        # 扩大 top_k 以确保召回包含目标标识符的节点
        original_top_k = self.retriever.similarity_top_k
        self.retriever.similarity_top_k = max(self.top_k * 3, 30)
        
        try:
            # 用标识符本身构造检索 query，语义+标识符双重召回
            query_str = " ".join(all_ids[:3])
            nodes = self.retriever.retrieve(query_str)
            for node in nodes:
                text = node.get_text()
                for id_list in identifiers.values():
                    for ident in id_list:
                        if ident in text:
                            exact_nodes.append(node)
                            break
        finally:
            self.retriever.similarity_top_k = original_top_k
        
        # 去重
        seen = set()
        unique_nodes = []
        for node in exact_nodes:
            node_id = getattr(node, 'node_id', id(node))
            if node_id not in seen:
                seen.add(node_id)
                unique_nodes.append(node)
        
        return unique_nodes

    def _merge_prioritize(self, exact_nodes: list, semantic_nodes: list) -> list:
        """合并精确匹配和语义检索结果，精确匹配优先，去重。"""
        seen = set()
        merged = []
        
        # 精确匹配优先
        for node in exact_nodes:
            node_id = getattr(node, 'node_id', id(node))
            if node_id not in seen:
                seen.add(node_id)
                merged.append(node)
        
        # 语义检索补充
        for node in semantic_nodes:
            node_id = getattr(node, 'node_id', id(node))
            if node_id not in seen:
                seen.add(node_id)
                merged.append(node)
        
        return merged[:self.top_k]

    def retrieve(self, question: str):
        """混合检索：精确 ID 匹配 + 语义检索兜底。"""
        # 1. 精确匹配：提取标识符
        identifiers = self._extract_identifiers(question)
        
        # 2. 精确匹配检索（传 identifiers 字典）
        exact_nodes = self._exact_match_retrieve(identifiers) if any(identifiers.values()) else []
        
        # 3. 语义检索兜底
        semantic_nodes = self.retriever.retrieve(question)
        filtered = self.postprocessor.postprocess_nodes(semantic_nodes)
        
        # 4. 合并去重，精确匹配优先
        return self._merge_prioritize(exact_nodes, filtered)[:self.top_k]

    def query(self, question: str) -> dict:
        """端到端查询：混合检索 + 过滤 + LLM 合成。"""
        # 使用混合检索
        nodes = self.retrieve(question)
        filtered = self.postprocessor.postprocess_nodes(nodes)
        contexts = [n.get_text() for n in filtered]

        # 无召回时返回兜底
        if not contexts:
            return {"answer": "未检索到相关内容。", "contexts": [], "scores": []}

        # 修复 KG-1：注入 context_truncator 时，生成用与 RAGAS 评分相同的截断结果
        gen_ctxs = self.context_truncator(contexts) if self.context_truncator else contexts[:3]
        if self.prompt_template:
            # 修复 KG-11：与 KG-RAG 使用同一 prompt 模板（占位 {context}/{question}）
            prompt = self.prompt_template.format(
                context="\n".join(gen_ctxs), question=question
            )
        else:
            prompt = (
                "你是一名工业设备故障诊断专家。严格遵守以下约束：\n"
                "1. 仅使用下方【手册片段】中的信息回答，禁止编造、扩展或引入外部知识。\n"
                "2. 每条结论必须能在【手册片段】中找到直接依据，无法确证的条目不得输出。\n"
                "3. 如果片段信息不足以完整回答，明确说明「根据手册未找到相关答案」，不要猜测。\n"
                "4. 答案需结构化：先列出故障现象→可能原因→解决方法，逐条给出原文依据。\n\n"
                "【手册片段】\n"
                + "\n---\n".join(gen_ctxs)
                + f"\n\n【问题】{question}\n\n【答案】"
            )
        if self.answer_generator is not None:
            response = self.answer_generator(prompt)
        else:
            response = self.llm.complete(prompt)
        return {
            "answer": str(response),
            "contexts": contexts,
            "scores": [n.get_score() for n in filtered],
        }

    def retrieve_only(self, question: str) -> list:
        """仅检索不合成，用于对比召回质量。"""
        return self.retrieve(question)

    def query_structured(self, question: str, **kwargs) -> RetrievalResult:
        """统一入口：Vector 引擎端到端返回 answer/contexts/scores。"""
        d = self.query(question)
        return RetrievalResult(
            answer=d.get("answer", ""),
            contexts=d.get("contexts", []),
            scores=d.get("scores", []),
        )
