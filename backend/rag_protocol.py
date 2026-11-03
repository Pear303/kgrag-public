"""统一检索结果与引擎协议（借鉴 kg-rag 的 RagSystem Protocol）。

设计动机：三引擎返回结构各异（Manual: [(names,rels,dirs)]、Community:
(matched,contexts)、Vector: {answer,contexts,scores}），评估器靠
isinstance(engine, dict) 分流。本模块定义统一结果与入口，评估器只依赖
query_structured；旧 query() 保留兼容，不破坏现有调用方。
"""
from dataclasses import dataclass, field
from typing import List, Optional, Protocol, Tuple


@dataclass
class RetrievalResult:
    answer: str = ""                                    # 仅端到端引擎（Vector）有
    contexts: List[str] = field(default_factory=list)   # 社区引擎 comm_contexts
    paths: List[Tuple] = field(default_factory=list)    # 图引擎 [(names, rels, dirs)]
    scores: List[float] = field(default_factory=list)   # 向量引擎 scores
    matched_communities: int = 0
    metadata: dict = field(default_factory=dict)


class RagEngine(Protocol):
    def query_structured(self, question: str, **kwargs) -> RetrievalResult:
        """统一检索入口。评估器只依赖此方法。"""
