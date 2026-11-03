"""backend/api_client.py 单元测试。

测试 ApiClient 单例行为与 get_client 函数。
使用 mock 替代 requests.Session，不发起真实 HTTP 请求。
多平台版（2026-08-08）：测试将实例重置为单平台 fake，验证 payload/重试/failover。
"""
from unittest.mock import patch, MagicMock
import os
import requests
import pytest
import importlib

from backend.api_client import ApiClient, get_client


def _as_single_platform(client, model=None):
    """把 ApiClient 实例重置为单平台 fake，使测试可控（不依赖真实 config）。"""
    client.platforms = [{"key": "fake-key", "base_url": "http://fake/api",
                         "model": model or os.getenv("API_MODEL", "deepseek-v4-flash"),
                         "quota": 0, "name": "fake"}]
    client._gates = {0: None}
    return client


class TestApiClientSingleton:
    """ApiClient 单例行为测试集。"""

    def test_same_instance_returned_from_constructor(self):
        """两次调用 ApiClient() 应返回同一个实例。"""
        a = ApiClient()
        b = ApiClient()
        assert a is b

    def test_get_client_returns_api_client(self):
        """get_client() 应返回 ApiClient 实例。"""
        client = get_client()
        assert isinstance(client, ApiClient)

    def test_get_client_returns_singleton(self):
        """get_client() 两次调用应返回同一个单例实例。"""
        c1 = get_client()
        c2 = get_client()
        assert c1 is c2

    def test_get_client_same_as_constructor(self):
        """get_client() 与 ApiClient() 应返回同一个实例。"""
        direct = ApiClient()
        via_func = get_client()
        assert direct is via_func

    def test_chat_method_structure(self):
        """chat() 方法应返回包含 content/reasoning/usage/platform 的字典。"""
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "choices": [{"message": {"content": "hello", "reasoning_content": "step 1"}}],
            "usage": {"total_tokens": 10},
        }

        client = _as_single_platform(ApiClient())
        client.session = MagicMock()
        client.session.post.return_value = mock_resp

        result = client.chat(
            messages=[{"role": "user", "content": "hi"}],
            model="deepseek-v4-flash",
            max_tokens=100,
            temperature=0.0,
            timeout=30,
        )

        assert result["content"] == "hello"
        assert result["reasoning"] == "step 1"
        assert result["usage"]["total_tokens"] == 10
        assert result["platform"] == "fake"

    def test_chat_sends_correct_payload(self):
        """chat() 应向正确的 URL 发送正确的 JSON payload 与平台 key 头。"""
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "choices": [{"message": {"content": "ok", "reasoning_content": ""}}],
            "usage": {},
        }

        client = _as_single_platform(ApiClient())
        client.session = MagicMock()
        client.session.post.return_value = mock_resp

        messages = [{"role": "user", "content": "hello world"}]
        client.chat(messages=messages, model="test-model", max_tokens=512, temperature=0.7, timeout=60)

        client.session.post.assert_called_once_with(
            "http://fake/api/chat/completions",
            json={
                "model": "test-model",
                "messages": messages,
                "max_tokens": 512,
                "temperature": 0.7,
                "reasoning_effort": "none",
            },
            headers={"Authorization": "Bearer fake-key"},
            timeout=60,
        )

    def test_chat_default_parameters(self):
        """未传参数时使用默认值：model=平台模型, max_tokens=8192, temperature=0.0。"""
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "choices": [{"message": {"content": "ok", "reasoning_content": ""}}],
            "usage": {},
        }

        client = _as_single_platform(ApiClient())
        client.session = MagicMock()
        client.session.post.return_value = mock_resp

        client.chat(messages=[{"role": "user", "content": "test"}])

        call_args = client.session.post.call_args
        assert call_args[1]["json"]["model"] == os.getenv("API_MODEL", "deepseek-v4-flash")
        assert call_args[1]["json"]["max_tokens"] == 8192
        assert call_args[1]["json"]["temperature"] == 0.0
        assert call_args[1]["timeout"] == int(os.getenv("API_TIMEOUT", "300"))

    def test_chat_retries_on_timeout_then_succeeds(self):
        """超时后应在同一平台重试，重试成功则返回正常结果。"""
        ok = MagicMock()
        ok.json.return_value = {
            "choices": [{"message": {"content": "ok"}}], "usage": {},
        }
        client = _as_single_platform(ApiClient())
        client.session = MagicMock()
        client.session.post.side_effect = [requests.Timeout("timeout"), ok]
        client.max_retries = 1
        client.retry_backoff = 0.0
        client.retry_statuses = set()

        result = client.chat(messages=[{"role": "user", "content": "hi"}], timeout=30)

        assert result["content"] == "ok"
        assert client.session.post.call_count == 2

    def test_chat_retries_on_5xx_then_succeeds(self):
        """代理 5xx 后应重试，重试成功返回正常结果。"""
        bad = MagicMock(status_code=502)
        ok = MagicMock(status_code=200)
        ok.json.return_value = {
            "choices": [{"message": {"content": "recovered"}}], "usage": {},
        }
        client = _as_single_platform(ApiClient())
        client.session = MagicMock()
        client.session.post.side_effect = [bad, ok]
        client.max_retries = 1
        client.retry_backoff = 0.0
        client.retry_statuses = {502}

        result = client.chat(messages=[{"role": "user", "content": "hi"}], timeout=30)

        assert result["content"] == "recovered"
        assert client.session.post.call_count == 2

    def test_chat_raises_after_retries_exhausted(self):
        """单平台重试耗尽后应抛出最后的异常，且请求次数 = 重试次数 + 1。"""
        client = _as_single_platform(ApiClient())
        client.session = MagicMock()
        client.session.post.side_effect = requests.Timeout("still down")
        client.max_retries = 2
        client.retry_backoff = 0.0
        client.retry_statuses = set()

        with pytest.raises(requests.Timeout):
            client.chat(messages=[{"role": "user", "content": "hi"}], timeout=30)
        assert client.session.post.call_count == 3

    def test_chat_failover_to_next_platform(self):
        """平台1 失败（5xx 重试耗尽）后应 failover 到平台2 并成功。"""
        ok2 = MagicMock(status_code=200)
        ok2.json.return_value = {
            "choices": [{"message": {"content": "from p2"}}], "usage": {},
        }
        client = ApiClient()
        client.platforms = [
            {"key": "k1", "base_url": "http://p1", "model": "m1", "quota": 0, "name": "p1"},
            {"key": "k2", "base_url": "http://p2", "model": "m2", "quota": 0, "name": "p2"},
        ]
        client._gates = {0: None, 1: None}
        client.session = MagicMock()
        client.session.post.side_effect = [
            MagicMock(status_code=502),  # p1 首次 5xx
            MagicMock(status_code=502),  # p1 重试仍 5xx → 换平台
            ok2,
        ]
        client.max_retries = 1
        client.retry_backoff = 0.0
        client.retry_statuses = {502}
        # 固定选平台顺序：先 p1(0)，排除后 p2(1)——避免加权随机破坏断言
        client._pick_index = lambda exclude=None: \
            0 if (exclude or set()) == set() else 1

        result = client.chat(messages=[{"role": "user", "content": "hi"}], timeout=30)

        assert result["content"] == "from p2"
        assert result["platform"] == "p2"
        assert client.session.post.call_count == 3

    def test_chat_no_reasoning_content(self):
        """API 未返回 reasoning_content 字段时，reasoning 应为空字符串。"""
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "choices": [{"message": {"content": "just content"}}],
            "usage": {"total_tokens": 5},
        }

        client = _as_single_platform(ApiClient())
        client.session = MagicMock()
        client.session.post.return_value = mock_resp

        result = client.chat(messages=[{"role": "user", "content": "test"}])

        assert result["content"] == "just content"
        assert result["reasoning"] == ""

    def test_session_has_content_type_header(self):
        """ApiClient 的 session 只设 Content-Type；Authorization 按平台请求时传入。"""
        original = ApiClient._instance
        ApiClient._instance = None
        try:
            client = ApiClient()
            headers = dict(client.session.headers)
            assert "Content-Type" in headers
            assert headers["Content-Type"] == "application/json"
            # 多平台下 Authorization 不放在 session（各平台 key 不同）
            assert "Authorization" not in headers
        finally:
            ApiClient._instance = original

    def test_reset_singleton_for_test_isolation(self):
        """通过重置 _instance 实现测试隔离（本测试不依赖真实 config）。"""
        original = ApiClient._instance
        try:
            ApiClient._instance = None
            fresh = ApiClient()
            assert fresh is not original
        finally:
            ApiClient._instance = original
