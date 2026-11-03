# -*- coding: utf-8 -*-
u"""图放行票（臂 a，2026-10-13）：把"已被拒"的题按命中实体子图边数放行。

定位与边界
----------
本模块**只做一件事**：在门链已经判出 refusal_reason 之后，若图谱证据够强，把它**清空**。
它**永远不会新设** refusal_reason —— 所以结构上不可能把"本来能答"的题变成"不能答"
（帕累托；这是本项目对拒答门改动的硬约束）。

判据来源（逐字对齐，不另立口径）
------------------------------
`references/安全门优化研究_从三发现到一个更大的发现.md` §1.4（四数据集实测）：
  - `edges>=2`  放行线（A组 84% / 口语 68% / 跨书OOC 27%）
  - `edges>=5`  免检强档（A组 32% / 口语 38% / **跨书OOC 0%**）
  - 占比类判据（lcc_ratio）已实测推翻：多路召回一题命中 20~30 实体，带分母必被稀释。
`edges` 的定义与 `references/experiments/probe_connectivity.py::analyse` **逐字相同**：
  命中实体集内、两端都在集合内的有向 `:REL` 边数。

开关
----
`GRAPH_TICKET=1` 才生效（缺省 0 = 不做任何事，生产路径零行为改变）。
`GRAPH_TICKET_MIN` 缺省 "2"（放行线）。
"""
import os

_TICKET_MIN_DEFAULT = u"2"


def ticket_enabled():
    return os.getenv(u"GRAPH_TICKET", u"0").strip().lower() in (u"1", u"true", u"yes", u"on")


def ticket_min_edges():
    try:
        return int(os.getenv(u"GRAPH_TICKET_MIN", _TICKET_MIN_DEFAULT))
    except Exception:
        return int(_TICKET_MIN_DEFAULT)


def count_inner_edges(driver, source_manual, matched):
    u"""命中实体**子图内部**的有向边数（与 probe_connectivity.py 同式）。

    matched 为空 ⇒ 0。只统计两端都在 matched 里的 :REL，跨手册不串。
    """
    matched = [m for m in (matched or []) if m]
    if not matched:
        return 0
    with driver.session(database=u"neo4j") as s:
        rows = s.run(
            u"MATCH (a:Entity {source_manual:$src})-[e:REL]->(b:Entity {source_manual:$src}) "
            u"WHERE a.name IN $n AND b.name IN $n "
            u"RETURN a.name AS u, b.name AS v",
            src=source_manual, n=matched).values()
    return len(rows)


def graph_ticket_hook(question, refusal_reason, manual_eng, driver, source_manual,
                      edge_counter=None):
    u"""门链钩子（与 vendor_gate_hook / sent_cov_gate_hook 同形）。

    返回 (refusal_reason_after, record|None)。
      - 未开开关或未配置引擎 ⇒ 原样返回、record=None（生产零改变）。
      - refusal_reason is None ⇒ 原样返回、record=None（**只处理"已被拒"的题**）。
      - 已被拒 + edges >= 阈值 ⇒ 返回 None（放行）。
      - 已被拒 + edges <  阈值 ⇒ 原样返回（维持拒答）。
    """
    if not ticket_enabled() or manual_eng is None or (driver is None and edge_counter is None):
        return refusal_reason, None
    if refusal_reason is None:
        return refusal_reason, None

    try:
        matched = [m for m in manual_eng._match_entities(question) if m]
        if edge_counter is not None:
            n_edges = edge_counter(source_manual, matched)
        else:
            n_edges = count_inner_edges(driver, source_manual, matched)
    except Exception as e:
        # 异常不折算成读数：记下类型，维持现门（降级放行/拒答都不许被我改判）
        return refusal_reason, {u"ticket": None, u"error": u"%s: %s" % (type(e).__name__, str(e)[:120])}

    thr = ticket_min_edges()
    rec = {u"ticket": bool(n_edges >= thr), u"edges": n_edges, u"min_edges": thr,
           u"n_matched": len(matched), u"was_refused": True}
    if n_edges >= thr:
        return None, rec
    return refusal_reason, rec


# ───────────────────────── 自测（三态反向桩） ─────────────────────────
def selftest():
    u"""三态反向桩：每条都必须为假一次，否则门是装饰。

    S1 没被拒 ⇒ 必须原样返回且 record=None（不许我插手"能答"的题）
    S2 被拒 + 边数足 ⇒ 必须清空为 None
    S3 被拒 + 边数不足 ⇒ 必须保留原 refusal_reason
    S4 关开关 ⇒ 必须原样返回、record=None
    S5 计数器抛异常 ⇒ 必须保留原 refusal_reason，且 record 记下 error（不许吞成放行）
    """
    class _Eng(object):
        def __init__(self, m):
            self._m = m

        def _match_entities(self, q):
            return self._m

    REASON = {u"relevance": 0.0}
    bad = []

    os.environ[u"GRAPH_TICKET"] = u"1"
    os.environ[u"GRAPH_TICKET_MIN"] = u"2"

    eng3 = _Eng([u"A", u"B", u"C"])
    eng1 = _Eng([u"A"])

    # S1
    r, rec = graph_ticket_hook(u"q", None, eng3, None, u"src", edge_counter=lambda s, m: 9)
    if r is not None or rec is not None:
        bad.append(u"S1 没被拒却被动了（r=%r rec=%r）" % (r, rec))
    # S2
    r, rec = graph_ticket_hook(u"q", REASON, eng3, None, u"src", edge_counter=lambda s, m: 5)
    if r is not None or not (rec and rec.get(u"ticket")):
        bad.append(u"S2 边数足却没放行（r=%r rec=%r）" % (r, rec))
    # S3
    r, rec = graph_ticket_hook(u"q", REASON, eng1, None, u"src", edge_counter=lambda s, m: 1)
    if r is not REASON or (rec and rec.get(u"ticket")):
        bad.append(u"S3 边数不足却放行了（r=%r rec=%r）" % (r, rec))
    # S4
    os.environ[u"GRAPH_TICKET"] = u"0"
    r, rec = graph_ticket_hook(u"q", REASON, eng3, None, u"src", edge_counter=lambda s, m: 9)
    if r is not REASON or rec is not None:
        bad.append(u"S4 开关关着却动了（r=%r rec=%r）" % (r, rec))
    # S5
    os.environ[u"GRAPH_TICKET"] = u"1"

    def _boom(s, m):
        raise RuntimeError(u"stub")

    r, rec = graph_ticket_hook(u"q", REASON, eng3, None, u"src", edge_counter=_boom)
    if r is not REASON or not (rec and rec.get(u"error")):
        bad.append(u"S5 异常被吞成放行（r=%r rec=%r）" % (r, rec))

    if bad:
        print(u"RED 图放行票自测 %d 条不过：" % len(bad))
        for x in bad:
            print(u"   - " + x)
        return 1
    print(u"OK 图放行票自测 S1~S5 全过（只清不设、关开关不动、异常不吞）")
    return 0


if __name__ == u"__main__":
    import sys
    sys.stdout.reconfigure(encoding=u"utf-8")
    sys.exit(selftest())
