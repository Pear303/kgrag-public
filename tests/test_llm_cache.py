"""LLMCache 单元测试：命中/未命中/键区分/env 开关。"""
import os
import pytest

from backend.llm_cache import LLMCache, _cache_key


@pytest.fixture()
def cache(tmp_path):
    return LLMCache(str(tmp_path / "test_cache.sqlite"))


def test_miss_then_hit(cache):
    assert cache.get("m", "prompt", 0.0, 100) is None
    cache.put("m", "prompt", 0.0, 100, "resp")
    assert cache.get("m", "prompt", 0.0, 100) == "resp"


def test_key_differs_on_temperature(cache):
    assert _cache_key("m", "p", 0.0, 100) != _cache_key("m", "p", 0.5, 100)


def test_key_differs_on_max_tokens(cache):
    assert _cache_key("m", "p", 0.0, 100) != _cache_key("m", "p", 0.0, 200)


def test_key_differs_on_model(cache):
    assert _cache_key("m1", "p", 0.0, 100) != _cache_key("m2", "p", 0.0, 100)


def test_put_empty_response_ignored(cache):
    cache.put("m", "p", 0.0, 100, "")
    assert cache.get("m", "p", 0.0, 100) is None


def test_env_switch_off_returns_none(monkeypatch):
    monkeypatch.setenv("ENABLE_LLM_CACHE", "0")
    from backend.llm_cache import get_cache
    assert get_cache() is None
    monkeypatch.setenv("ENABLE_LLM_CACHE", "1")
    assert get_cache() is not None
