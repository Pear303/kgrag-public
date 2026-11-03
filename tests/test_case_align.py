# -*- coding: utf-8 -*-
"""OPT_CASE_ALIGN（A 案定案约束）反向测试。

门必须能为假：默认 0 ⇒ 模板与在册 _COT_PROMPT_TEMPLATE 逐字节相同；
=1 ⇒ 恰好在尾部追加定案约束；异常 env 值按 0 处理。
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.evaluator import RAGASEvaluator  # noqa: E402


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.delenv("OPT_CASE_ALIGN", raising=False)


def test_default_template_byte_identical():
    assert RAGASEvaluator._answer_template() == RAGASEvaluator._COT_PROMPT_TEMPLATE


def test_on_appends_suffix():
    os.environ["OPT_CASE_ALIGN"] = "1"
    t = RAGASEvaluator._answer_template()
    assert t != RAGASEvaluator._COT_PROMPT_TEMPLATE
    assert t == RAGASEvaluator._COT_PROMPT_TEMPLATE + RAGASEvaluator._CASE_ALIGN_PROMPT_SUFFIX


def test_suffix_contains_key_rules():
    s = RAGASEvaluator._CASE_ALIGN_PROMPT_SUFFIX
    assert "依据案例" in s
    assert "无法唯一确定对应案例" in s
    assert "其余案例的内容一律不写进答案" in s


def test_off_values_falsy():
    for v in ("0", "", "false", "no"):
        os.environ["OPT_CASE_ALIGN"] = v
        assert RAGASEvaluator._answer_template() == RAGASEvaluator._COT_PROMPT_TEMPLATE


def test_only_exact_one_activates():
    # 项目惯例：开关只认 "1"（与 OPT_C/OPT_DIRECT_CASE 一致），其余一律视为关
    for v in ("true", "yes", "01"):
        os.environ["OPT_CASE_ALIGN"] = v
        assert RAGASEvaluator._answer_template() == RAGASEvaluator._COT_PROMPT_TEMPLATE
