# -*- coding: utf-8 -*-
"""键锚定行＋案例全文上下文增强的反向测试（serve._keycase_contexts）。

预注册：files/_oneoff/anchor-fix-20261027/keycase_prereg_20261029.md（dc40cc84）
五项：①默认关逐字节不变；②K 门边界；③C 不动作条件；④门行为零变化
（源序钉：调用点在全部拒答门之后且被 refusal_reason is None 护住）；
⑤manual 缺失安全退化。每条反向桩构造上就属于它要钓的类。
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_ENV_SNAPSHOT = dict(os.environ)
import serve  # noqa: E402
os.environ.clear()
os.environ.update(_ENV_SNAPSHOT)
del _ENV_SNAPSHOT

from serve import (  # noqa: E402
    _build_case_site_map, _keycase_contexts, _keyrow_block, _case_full_block,
    _keyrow_keys,
)

BOOK_C = """# 手册
例 20  主轴故障维修故障现象：一台数控车床加工中主轴出现异常声响，停机检查发现主轴轴承磨损严重，游隙超标。
轴承型号 NSK6204，更换同型号轴承并重新调整预紧力后，机床恢复正常运行，异响消失。
维修小结：主轴异响多与轴承磨损、预紧力不当有关，定期检查润滑与游隙可预防复发。
例 21  进给故障维修故障现象：某加工中心执行程序时 X 轴完全没有移动，检查发现 X 轴电机与丝杠之间的联轴器已经松动打滑。
联轴器型号 ML6，重新紧固锁紧螺钉并复查定位精度后，机床恢复正常运行。
维修提示：联轴器松动是进给轴不动的常见原因，紧固之后务必再次核查定位精度与反向间隙。"""


def _cmap():
    return _build_case_site_map(BOOK_C)


# ---------- ① 默认关：两开关都不在 ⇒ 三个入口全部 None（装配逐字节不变） ----------

def test_default_off_all_none():
    os.environ.pop("OPT_KEYROW", None)
    os.environ.pop("OPT_CASE_FULL", None)
    sites, all_sites, _fa = _cmap()
    ctxs = ["例 20  主轴故障维修故障现象：主轴轴承磨损严重。"]
    assert _keyrow_block("CH18 报警怎么办", ctxs) is None
    assert _case_full_block("例20 轴承磨损怎么修", ctxs, BOOK_C, sites,
                            all_sites) is None
    assert _keycase_contexts("例20 轴承磨损怎么修", ctxs, BOOK_C,
                             sites, all_sites) is None


# ---------- ② K 门边界：无键／0 行／1–2 行／3 行；数码管数字通道 ----------

def test_keyrow_keys_extraction():
    codes, digit = _keyrow_keys("CH18 报警时 G117.0 信号异常，S1 灯不亮")
    assert codes == ["ch18", "g117.0", "s1"]
    # 纯数字（型号 6057、报警 2011）不当行键
    codes2, _ = _keyrow_keys("型号 6057 报警 2011 怎么处理")
    assert codes2 == []
    _c, d = _keyrow_keys("数码管显示 5 闪是什么故障")
    assert d == 5


def _kctx(*lines):
    return ["\n".join(lines)]


def test_keyrow_gate_boundary():
    os.environ["OPT_KEYROW"] = "1"
    try:
        # 无键 ⇒ None
        assert _keyrow_block("主轴异响怎么查",
                             _kctx("CH18 报警，主轴不能启动")) is None
        # 0 行 ⇒ None
        assert _keyrow_block("CH18 报警怎么办",
                             _kctx("X 轴爬行，检查丝杠预紧")) is None
        # 恰 1 行 ⇒ 前置块
        b = _keyrow_block("CH18 报警怎么办",
                          _kctx("例 26  故障现象：CH18 报警，主轴不能启动。",
                                "分析：位置控制板不良。"))
        assert b is not None and "【键锚定行】" in b \
            and "CH18 报警，主轴不能启动" in b
        # 恰 2 行（第二块里那行与第一块逐字重复，去重后 2）⇒ 仍出块
        b2 = _keyrow_block("CH18 报警怎么办",
                           ["例 26  故障现象：CH18 报警。\n检查 CH18 输入电压。",
                            "例 27  同类案例。\n例 26  故障现象：CH18 报警。"])
        assert b2 is not None
        # 3 行去重 ⇒ 超门，None（宁可不动作，不许猜）
        b3 = _keyrow_block("CH18 报警怎么办",
                           _kctx("CH18 报警。", "检查 CH18 输入。",
                                 "更换 CH18 板。"))
        assert b3 is None
        # 数码管数字：匹配只看行首 15 字
        d1 = _keyrow_matched = _keyrow_block(
            "数码管显示 5 闪是什么故障", _kctx("5  主轴电机故障，更换轴承。"))
        assert d1 is not None and "主轴电机故障" in d1
    finally:
        os.environ.pop("OPT_KEYROW", None)


# ---------- ③ C 不动作条件：段过短／已覆盖／平票 ----------

def test_case_full_no_action_conditions():
    os.environ["OPT_CASE_FULL"] = "1"
    try:
        sites, all_sites, _fa = _cmap()
        # 正例：单候选＋题面词面分独占 ⇒ 出案例全文块
        ctxs = ["例 20  主轴故障维修故障现象：主轴有异常声响。",
                "参见相关手册章节。"]
        b = _case_full_block("主轴轴承磨损游隙超标怎么修", ctxs, BOOK_C,
                             sites, all_sites)
        assert b is not None and "【案例全文】" in b \
            and "NSK6204" in b
        # 段过短（<120）⇒ 不动作（填充段放在位点之前，位点后即书尾）
        short_book = "前情提要。"*30 + "例 3  小修故障现象：轴承磨损。\n换新即可。"
        s2, a2, _ = _build_case_site_map(short_book)
        assert _case_full_block("轴承磨损怎么修", ["例 3"], short_book,
                                s2, a2) is None
        # 上下文已覆盖 ≥0.95 ⇒ 不动作（不重复塞同一段）
        off = sites["20"][0]
        nxt = next(o for o, _t in all_sites if o > off)
        full_span = BOOK_C[off:nxt].strip()
        ctx_covered = [full_span]
        assert _case_full_block("主轴轴承磨损游隙超标怎么修", ctx_covered,
                                BOOK_C, sites, all_sites) is None
        # 两候选平票（s1==s2）⇒ 不臆选，None（题面词面分对两案相等）
        ctx_both = ["例 20  主轴故障。", "例 21  进给故障。"]
        assert _case_full_block("机床恢复不了怎么办", ctx_both, BOOK_C,
                                sites, all_sites) is None
    finally:
        os.environ.pop("OPT_CASE_FULL", None)


# ---------- ④ 门行为零变化：源序钉（调用点在全部拒答门之后、有护栏、在 return 前） ----------

def test_call_site_after_all_gates():
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    src = open(os.path.join(here, "serve.py"), encoding="utf-8").read()
    i_joint = src.index("joint_gate 钩子异常")
    i_guard = src.index("refusal_reason is None and graph_contexts \\\n"
                        "                and (os.environ.get(\"OPT_KEYROW\")")
    i_ret = src.index("return graph_contexts, refusal_reason", i_guard)
    assert i_joint < i_guard < i_ret
    # 护栏里必须有 try/except 降级（异常保留原上下文，不许 500）
    seg = src[i_guard:i_ret]
    assert "except Exception" in seg and "logger.warning" in seg


# ---------- ⑤ manual 缺失安全退化：不抛异常，K 通道不受影响 ----------

def test_missing_manual_safe_degradation():
    os.environ["OPT_KEYROW"] = "1"
    os.environ["OPT_CASE_FULL"] = "1"
    try:
        ctxs = ["例 26  故障现象：CH18 报警，主轴不能启动。"]
        out = _keycase_contexts("CH18 报警怎么办", ctxs, "", {}, [])
        # C 通道静默不动作；K 通道照常前置
        assert out is not None and out[0].startswith("【键锚定行】") \
            and out[1:] == ctxs
        # manual 正常但 site_map 空（书里没有定义位点）⇒ C 不动作
        sites, all_sites, _fa = _cmap()
        out2 = _keycase_contexts("例20 轴承磨损怎么修", ["例 20  主轴故障。"],
                                 BOOK_C, {}, [])
        assert out2 is None
        assert sites and all_sites  # 前面构造确实有位点，排除假阴性
    finally:
        os.environ.pop("OPT_KEYROW", None)
        os.environ.pop("OPT_CASE_FULL", None)


# ---------- ⑥ K 前置块不许被 C 的最弱块替换吃掉 ----------

def test_c_never_eats_k_block():
    """反向桩：K 短块词面分必然最低，若 C 的最弱块搜索不含偏移，
    K 块会被案例全文替换掉（2026-10-30 重放实测 6 题因此 K 失效）。"""
    os.environ["OPT_KEYROW"] = "1"
    os.environ["OPT_CASE_FULL"] = "1"
    try:
        sites, all_sites, _fa = _cmap()
        # 例20 已在上下文（候选案），另有一行 CH18 键行；K 前置后 C 应替换原 filler 块
        ctxs = ["例 20  主轴故障维修故障现象：主轴有异常声响。",
                "CH18 报警，主轴不能启动，检查位置控制板。",
                " filler filler filler filler filler filler filler "]
        out = _keycase_contexts("CH18 报警 主轴轴承磨损游隙超标怎么修", ctxs,
                                BOOK_C, sites, all_sites)
        assert out is not None
        assert out[0].startswith("【键锚定行】"), "K 块必须活在第 0 位"
        assert sum(1 for c in out if c.startswith("【案例全文】")) == 1
        assert " filler filler" not in out  # 被替换的是原 filler 块
    finally:
        os.environ.pop("OPT_KEYROW", None)
        os.environ.pop("OPT_CASE_FULL", None)
