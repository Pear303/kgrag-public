"""ChatOpenAI 子类：强制 n=1 以兼容 deepseek-v4-flash。

RAGAS 评估框架内部默认使用 n=3（多采样），而 deepseek-v4-flash 仅支持 n=1。
本模块通过继承重写 _generate / _agenerate，在调用父类前剔除 n 参数。
"""
from typing import Any

from langchain_openai import ChatOpenAI
from langchain_core.callbacks import CallbackManagerForLLMRun, AsyncCallbackManagerForLLMRun


class DeepSeekChatOpenAI(ChatOpenAI):
    """ChatOpenAI 子类：强制 n=1，兼容 deepseek-v4-flash / vLLM 等仅支持 n=1 的后端。"""

    def _generate(
        self,
        messages: list[Any],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> Any:
        kwargs.pop("n", None)
        return super()._generate(messages, stop=stop, run_manager=run_manager, n=1, **kwargs)

    async def _agenerate(
        self,
        messages: list[Any],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> Any:
        kwargs.pop("n", None)
        return await super()._agenerate(messages, stop=stop, run_manager=run_manager, n=1, **kwargs)
