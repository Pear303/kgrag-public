"""qwen_judge.py 多平台 judge 单元测试。

重点守护 2026-08-08 修复的递归采样死锁：
- n 采样（_generate_multiple/_agenerate_multiple）内层调用必须显式 n=1，
  否则 self.n / 上层 kwargs 的 n >1 时会再次进入 multiple 分支，形成无限递归
  （py-spy 实测递归数十层、CPU 持续累积、无外部 API 请求）。
测试用 fake LLM，不发起真实 HTTP 请求；禁用磁盘缓存。
"""
import os
os.environ["RAGAS_JUDGE_CACHE"] = "0"  # 必须在 import qwen_judge 前设置
import sys
import asyncio
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qwen_judge import FailoverChatOpenAI
from langchain_core.outputs import ChatResult, ChatGeneration
from langchain_core.messages import AIMessage


class _FakeLLM:
    """返回单条 generation 的假 LLM，用于隔离 judge 路由/缓存逻辑。"""

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content="ok"))])

    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):
        return self._generate(messages, stop=stop, run_manager=run_manager, **kwargs)


@pytest.fixture
def judge(monkeypatch):
    monkeypatch.setattr(FailoverChatOpenAI, "_make", lambda self, p: _FakeLLM())
    return FailoverChatOpenAI(platforms=[{
        "base_url": "http://fake", "api_key": "k", "model": "m",
        "quota": 0, "name": "fake",
    }])


class TestNSamplingRecursionGuard:
    """回归测试：n 采样模拟不得无限递归（修复 2026-08-08）。"""

    def test_generate_multiple_returns_n_generations(self, judge):
        r = judge._generate_multiple([{"role": "user", "content": "hi"}], n=3)
        assert len(r.generations) == 3

    def test_agenerate_multiple_returns_n_generations(self, judge):
        r = asyncio.run(
            judge._agenerate_multiple([{"role": "user", "content": "hi"}], n=3))
        assert len(r.generations) == 3

    def test_generate_single_call_when_n_equals_1(self, judge):
        r = judge._generate([{"role": "user", "content": "hi"}])
        assert len(r.generations) == 1


def _real_make(platform):
    """绕过 fixture（它把 _make 换成了假 LLM），走真实的 _make。"""
    j = FailoverChatOpenAI(platforms=[platform])
    return j._make(platform)


def test_make_injects_session_header_for_opencode():
    """opencode 系平台缺 x-opencode-session 会直接 400 MissingSessionID ⇒ 判分一条
    都回不来（2026-09-25 b4 预检实测：答案生成走 api_client 正常、judge 全灭四指标
    NaN）。头必须复用 api_client 那一份，而不是在 judge 侧另写一套。"""
    p = {"base_url": "https://opencode.ai/zen/go/v1", "api_key": "sk-test",
         "model": "deepseek-v4-flash", "quota": 0, "name": "platform3"}
    h = _real_make(p).default_headers
    assert h.get("x-opencode-session"), "opencode 平台必须带会话头"
    assert h.get("Authorization", "").startswith("Bearer ")
    assert "sk-test" not in repr(sorted(h.keys()))


def test_make_does_not_add_session_header_elsewhere():
    """非 opencode 端点不得被塞会话头（否则旧端点行为被这次改动漂走）。"""
    p = {"base_url": "https://example.com/v1", "api_key": "sk-test",
         "model": "m", "quota": 0, "name": "other"}
    h = _real_make(p).default_headers
    assert "x-opencode-session" not in h
    assert h.get("Authorization") == "Bearer sk-test"


def test_make_keeps_reasoning_off_and_n_one():
    """这次改动不许动判分口径：思考仍关、n 仍强制 1。"""
    p = {"base_url": "https://example.com/v1", "api_key": "k", "model": "m",
         "quota": 0, "name": "other"}
    llm = _real_make(p)
    assert llm.n == 1
    assert llm.extra_body == {"enable_thinking": False, "reasoning_effort": "none"}
