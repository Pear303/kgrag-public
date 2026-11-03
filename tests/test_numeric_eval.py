"""数值化评估闭环单元测试（借鉴 kg-rag extract_number 判分，适配中文单位）。"""
import pytest

from backend.evaluator import (
    parse_numeric_expectation, extract_number_from_answer, numeric_accuracy,
)


def test_parse_greater_than():
    e = parse_numeric_expectation(">80℃")
    assert e is not None
    assert e.operator == ">" and e.low == 80 and e.unit == "℃"


def test_parse_range():
    e = parse_numeric_expectation("0.02～0.06mm")
    assert e is not None
    assert e.operator == "range" and e.low == 0.02 and e.high == 0.06


def test_parse_fraction_range():
    e = parse_numeric_expectation("1/3至2/3")
    assert e is not None
    assert e.operator == "range"
    assert abs(e.low - 0.33) < 0.01 and abs(e.high - 0.67) < 0.01


def test_parse_plain_number_with_unit():
    e = parse_numeric_expectation("25000小时")
    assert e is not None
    assert e.operator == "=" and e.low == 25000


def test_parse_non_numeric_returns_none():
    assert parse_numeric_expectation("更换机械密封组件并重新调整压缩量") is None


def test_extract_number_from_answer():
    assert extract_number_from_answer("超过80℃") == 80.0
    assert extract_number_from_answer("答案为 75 mm") == 75.0
    assert extract_number_from_answer("无相关数据") is None


def test_numeric_accuracy_match():
    assert numeric_accuracy("超过80℃", ">80℃") == 0.0, "答案只给阈值80不满足'超过80'"
    assert numeric_accuracy("85℃", ">80℃") == 1.0
    assert numeric_accuracy("25000小时", "25000小时") == 1.0


def test_numeric_accuracy_mismatch():
    assert numeric_accuracy("75", ">80℃") == 0.0


def test_numeric_accuracy_unparseable_returns_none():
    assert numeric_accuracy("未知", ">80℃") is None
    assert numeric_accuracy("80", "更换机械密封组件") is None
