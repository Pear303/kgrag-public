"""本地 Ollama Embedding 适配器。

背景：llama-index 自带的 OllamaEmbedding 在调用 qwen3-embedding 系列模型时会挂起。
早期方案是包装 ollama 原生 `Client().embeddings()`（旧 /api/embeddings 端点）。
2026-08-05 shukongdashi 评测实测：旧端点在连续调用下会挂死且单次约 5s（新端点
`Client().embed()` 稳定、约 0.1s/次，backfill 4096 实体无异常），故改为新端点，
并加每调用超时（60s，Ollama 挂起时抛错而非无限等待）+ 重试（3 次退避）。
实现 llama-index BaseEmbedding 所需的最小接口，供全管道使用。
"""
from typing import List
import time

from pydantic import PrivateAttr
from llama_index.core.embeddings import BaseEmbedding


class OllamaLocalEmbedding(BaseEmbedding):
    _client: object = PrivateAttr(default=None)
    _model: str = PrivateAttr(default=None)
    _timeout: float = PrivateAttr(default=60.0)
    _max_retries: int = PrivateAttr(default=3)

    def __init__(self, model_name: str = "qwen3-embedding:0.6b",
                 base_url: str = "http://localhost:11434",
                 timeout: float = 60.0, max_retries: int = 3, **kwargs):
        super().__init__(**kwargs)
        import ollama
        self._model = model_name
        self._timeout = timeout
        self._max_retries = max_retries
        # timeout 传给底层 httpx 客户端：Ollama 挂起时 60s 抛错而非无限挂起
        self._client = ollama.Client(host=base_url, timeout=timeout)

    @property
    def model_id(self) -> str:
        """公开的模型身份（缓存签名/健康检查用；不改变 _embed_one 行为）。"""
        return str(self._model or "")

    def _embed_one(self, text: str) -> List[float]:
        """单文本 embed，带超时 + 重试（Ollama 偶发挂起/断连时自愈）。"""
        last_err = None
        for attempt in range(self._max_retries):
            try:
                resp = self._client.embed(model=self._model, input=[text])
                return list(resp["embeddings"][0])
            except Exception as e:
                last_err = e
                if attempt < self._max_retries - 1:
                    time.sleep(2.0 * (attempt + 1))
        raise last_err

    def _get_text_embedding(self, text: str) -> List[float]:
        return self._embed_one(text)

    def _get_query_embedding(self, query: str) -> List[float]:
        return self._embed_one(query)

    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        out = []
        for t in texts:
            out.append(self._embed_one(t))
        return out

    async def _aget_text_embedding(self, text: str) -> List[float]:
        return self._embed_one(text)

    async def _aget_query_embedding(self, query: str) -> List[float]:
        return self._embed_one(query)
