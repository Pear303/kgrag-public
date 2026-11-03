# -*- coding: utf-8 -*-
"""表格抽行通道（serve._table_row_contexts）反向测试。

⚠ 模块级 import serve 会把在役预设 45 键注入进程环境（10-27 实测，
同 test_direct_case.py）⇒ 导入前后环境快照还原。
"""
import os

_ENV_SNAPSHOT = dict(os.environ)

import serve  # noqa: E402  (导入即注入在役预设的45键)

os.environ.clear()
os.environ.update(_ENV_SNAPSHOT)

import pytest  # noqa: E402

BOOK = """第一章 概述
本章介绍维修方法。

表3-1 故障代码一览表
代码 含义 备注
1 过电压报警 电压超过上限
2 欠电压报警 电压低于下限
3 过电流报警 电流超过上限
采用新式结构的驱动器，代码含义见表3-2；电源模块详见第4章。

表3-2 新式驱动器故障代码一览表
代码 含义 备注
1 风机单元报警 风机停转
2 过热报警 散热器温度过高
第四章 电源模块维修
"""


@pytest.fixture()
def book():
    lines = BOOK.split("\n")
    return serve._norm_text(BOOK), serve._build_table_index(lines), lines


def test_gate_off_returns_empty(book):
    os.environ["OPT_TABLE_ROW"] = "0"
    blocks, markers = serve._table_row_contexts(
        "数码管显示数字2的含义（表3-1）", book[0], book[1], book[2])
    assert blocks == [] and markers == []


def test_unique_site_decisive_single_row(book):
    os.environ["OPT_TABLE_ROW"] = "1"
    blocks, markers = serve._table_row_contexts(
        "表3-1中，数码管显示数字2的含义是什么", book[0], book[1], book[2])
    assert len(blocks) == 1
    assert "表3-1 故障代码一览表" in blocks[0]
    assert "2 欠电压报警 电压低于下限" in blocks[0]
    assert "过电压报警" not in blocks[0].split("相关行：")[1]
    assert any("欠电压报警" in m for m in markers)


def test_duplicate_numbers_disambiguated(book):
    os.environ["OPT_TABLE_ROW"] = "1"
    blocks, _ = serve._table_row_contexts(
        "新式驱动器表3-2中数码管显示2的含义", book[0], book[1], book[2])
    assert len(blocks) == 1
    assert "表3-2 新式驱动器故障代码一览表" in blocks[0]
    assert "2 过热报警 散热器温度过高" in blocks[0]


def test_cross_reference_never_chosen(book):
    os.environ["OPT_TABLE_ROW"] = "1"
    blocks, _ = serve._table_row_contexts(
        "表3-1中数码管显示3的含义", book[0], book[1], book[2])
    assert len(blocks) == 1
    assert "故障代码一览表" in blocks[0]
    assert "3 过电流报警" in blocks[0]


def test_no_such_table_returns_empty(book):
    os.environ["OPT_TABLE_ROW"] = "1"
    blocks, markers = serve._table_row_contexts(
        "表9-9中数码管显示1的含义", book[0], book[1], book[2])
    assert blocks == [] and markers == []


def test_no_scoring_row_returns_empty(book):
    os.environ["OPT_TABLE_ROW"] = "1"
    blocks, markers = serve._table_row_contexts(
        "表3-1的表格结构是怎样的", book[0], book[1], book[2])
    assert blocks == [] and markers == []


def test_determinism_same_question_same_blocks(book):
    os.environ["OPT_TABLE_ROW"] = "1"
    q = "表3-1中数码管显示1的含义"
    a = serve._table_row_contexts(q, book[0], book[1], book[2])
    b = serve._table_row_contexts(q, book[0], book[1], book[2])
    assert a == b
