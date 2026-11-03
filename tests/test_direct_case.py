# -*- coding: utf-8 -*-
"""例号直取＋消歧通道的反向测试（serve._direct_case_contexts）。

门必须"能为假"：每条反向桩构造上就属于它要钓的类，
只有它响才算通道真的在防那类错误。
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# import serve 会在导入期注入 SERVING_PRESET_META 的 45 个键（OPT_C=1 等），
# 污染同进程后续测试的装配环境（10-11 runner 默认值事故同类）。导入前后
# 做环境快照还原，只留函数本身。
_ENV_SNAPSHOT = dict(os.environ)
import serve  # noqa: E402
os.environ.clear()
os.environ.update(_ENV_SNAPSHOT)
del _ENV_SNAPSHOT

from serve import (  # noqa: E402
    _build_case_site_map, _direct_case_contexts,
)

BOOK = """# 手册
例 10  主轴故障维修故障现象：一台配套某型号数控系统的车床在加工过程中主轴出现异常声响，经检查发现主轴轴承磨损严重。
更换轴承后机床恢复正常。轴承型号为 NSK6204。
例 11  进给故障维修故障现象：某加工中心在执行程序时 X 轴完全没有移动，检查发现 X 轴电机与丝杠之间的联轴器已经松动。
紧固后恢复。联轴器型号为 ML6。
例 10  丝杠故障维修故障现象：同型号车床在加工中出现 Z 轴爬行现象，检查发现该轴滚珠丝杠的预紧力调整得过大。
调整预紧力后恢复。丝杠型号为 SFU2005。
参见例11，此处略。"""



def _map():
    return _build_case_site_map(BOOK)


def test_gate_off_returns_empty():
    sites, all_sites, first_any = _map()
    os.environ.pop("OPT_DIRECT_CASE", None)
    try:
        assert _direct_case_contexts("在例10中轴承怎么了", BOOK, sites,
                                     all_sites, first_any) == []
    finally:
        os.environ.pop("OPT_DIRECT_CASE", None)


def test_gate_on_unique_site():
    sites, all_sites, first_any = _map()
    os.environ["OPT_DIRECT_CASE"] = "1"
    try:
        blocks = _direct_case_contexts("在例11中联轴器怎么了", BOOK, sites,
                                       all_sites, first_any)
        assert len(blocks) == 1 and "联轴器" in blocks[0]
    finally:
        os.environ.pop("OPT_DIRECT_CASE", None)


def test_disambiguation_picks_right_definition():
    """例10 有两个定义（轴承/丝杠）：题面说"丝杠"必须选中第二处。"""
    sites, all_sites, first_any = _map()
    os.environ["OPT_DIRECT_CASE"] = "1"
    try:
        blocks = _direct_case_contexts("在例10中丝杠爬行怎么处理", BOOK, sites,
                                       all_sites, first_any)
        assert len(blocks) == 1
        assert "丝杠" in blocks[0] and "轴承磨损" not in blocks[0]
    finally:
        os.environ.pop("OPT_DIRECT_CASE", None)


def test_disambiguation_tie_falls_back_to_first():
    """题面除编号外无实词（"例10中怎么了"）⇒ 平票回落首现位点，不臆选。"""
    sites, all_sites, first_any = _map()
    os.environ["OPT_DIRECT_CASE"] = "1"
    try:
        blocks = _direct_case_contexts("例10中怎么了", BOOK, sites,
                                       all_sites, first_any)
        assert len(blocks) == 1 and "轴承磨损" in blocks[0]
    finally:
        os.environ.pop("OPT_DIRECT_CASE", None)


def test_cross_reference_never_wins():
    """纯交叉引用块（"参见例11，此处略"）不是定义位点 ⇒ 永不入选。
    反向：若实现错把引用当定义，本题会返回引用句块。"""
    sites, all_sites, first_any = _map()
    assert "11" in sites and all(
        BOOK[o:o + 6].startswith("例 11") for o in sites["11"])
    os.environ["OPT_DIRECT_CASE"] = "1"
    try:
        blocks = _direct_case_contexts("例11的联轴器", BOOK, sites,
                                       all_sites, first_any)
        assert all("此处略" not in b for b in blocks)
    finally:
        os.environ.pop("OPT_DIRECT_CASE", None)


def test_no_such_case_returns_empty():
    sites, all_sites, first_any = _map()
    os.environ["OPT_DIRECT_CASE"] = "1"
    try:
        assert _direct_case_contexts("在例99中怎么样", BOOK, sites,
                                     all_sites, first_any) == []
    finally:
        os.environ.pop("OPT_DIRECT_CASE", None)


def test_letter_suffix_token():
    """例190A 这类带字母后缀的编号不能被当纯数字误吞。"""
    book = BOOK + "\n例 190A  驱动器故障维修故障现象：某机床在加工中步进驱动器输入电源跳闸，且机床设计时未对驱动器进行监控。\n"
    sites, all_sites, first_any = _build_case_site_map(book)
    assert "190A" in sites
    os.environ["OPT_DIRECT_CASE"] = "1"
    try:
        blocks = _direct_case_contexts("例190A中驱动器怎么了", book, sites,
                                       all_sites, first_any)
        assert len(blocks) == 1 and "驱动器" in blocks[0]
    finally:
        os.environ.pop("OPT_DIRECT_CASE", None)
