# -*- coding: utf-8 -*-
u"""丙案（预注册 §14）：RAGAS 四指标的"配对外排除"——生产件与自检件共用同一函数。

三条判据各对应一段代码，都能为假：
  P2 只从"评分输入"里移除，输出记录表（调用方的 _all_records）一字节不动；
  P3 清单里的 id 在评分输入里找不到 ⇒ raise（静默少滤会把 226 变成 227，聚合值看不出来）；
  P4 env 空 ⇒ 原对象返回（恒等，不是"复制后恰好相等"），历史行为逐字节不变。
"""
import json
import os


def resolve_register(env_value):
    u"""env 值 → 盘上实路径。空值＝不排除；给了却两处都找不到 ⇒ 直接抛，不静默降级。"""
    v = (env_value or u'').strip()
    if not v:
        return None
    cands = [v]
    if not os.path.isabs(v):
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        cands.append(os.path.join(root, v))
    for c in cands:
        if os.path.exists(c):
            return c
    raise FileNotFoundError(
        u'RAGAS_EXCLUDE_JSON 指向的登记单不存在（试过 cwd 与工程根两处）：%s' % (cands,))


def excluded_ids(path):
    u"""登记单里点名的排除题号（排序后返回，便于逐字比对）。"""
    with open(path, 'r', encoding='utf-8') as f:
        reg = json.load(f)
    return sorted({str(x) for x in (reg.get('excluded_ids') or [])})


def apply_exclude(per_question, env_value):
    u"""返回 (评分输入, 被移除 id 列表, 登记单实路径或 None)；不就地改 per_question。"""
    real = resolve_register(env_value)
    if real is None:
        return per_question, [], None
    ids = set(excluded_ids(real))
    if not ids:
        return per_question, [], real
    present = set(str(r.get('id')) for r in per_question)
    miss = sorted(ids - present)
    if miss:
        raise ValueError(
            u'RAGAS_EXCLUDE_JSON=%s 里的 id 在本臂评分输入中找不到：%s'
            u'（题集与登记单不是同一世代，拒跑）' % (real, miss))
    kept = [r for r in per_question if str(r.get('id')) not in ids]
    return kept, sorted(ids), real
