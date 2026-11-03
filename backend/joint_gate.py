# -*- coding: utf-8 -*-
"""联合门（七特征两段式拒答）生产钩子。2026-10-20 装机（作者指令）。

依据链：_prereg_joint_gate_20261019.md（甄别预注册）→ _joint_gate_screen_20261019.json
（faaea94a，特征定义与 LOOCV AUC 0.9500）→ _prereg_jointgate_port_20261019.md（搬运协议）
→ _jointgate_port_20261019.json（2550e47f，部署系数 kg_fit：KG 残差一次拟合原样使用）。

设计（与句子尺票 sent_cov_gate_hook 同款纪律）：
- 加法钩子：只增拒不撤拒；前级门已拒 ⇒ 原样返回（零计算）；
- 开关 OOC_JOINT_GATE（默认 '1' 启用；'0' 关闭 ⇒ 零开销直通）；
- 任何异常 ⇒ 记 warning 原样放行（降级现门），绝不抛出；
- 全部图访问只读；邻居/实体清单查询均带 ORDER BY（确定性）；
- 特征与离线甄别/搬运件逐行同码：f6/f1/f2 纯词法 ⇒ 运行时零 embedding 依赖；
  f7 的实体名单复用 manual_eng._entity_names（与甄别件同源同序）。
"""
import json
import logging
import os
import re

logger = logging.getLogger(__name__)

FEATS = ['f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7']
_CHAIN_RELS = ('表现为', '可能原因', '解决方法')
_MODEL_PATH_DEFAULT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'files', '_j226_20261011', '_jointgate_deploy_20261020.json')

_model_cache = None


def _load_model():
    """加载并校验部署模型（进程内缓存；缺失/损坏 ⇒ False＝钩子直通）。"""
    global _model_cache
    if _model_cache is not None:
        return _model_cache
    path = os.getenv('OOC_JOINT_GATE_MODEL', _MODEL_PATH_DEFAULT)
    try:
        with open(path, 'rb') as f:
            d = json.loads(f.read().decode('utf-8'))
        assert d.get('enabled') is True, 'deploy 模型 enabled!=True'
        assert d.get('features') == FEATS, '特征序失配'
        for k in ('coefs', 'med', 'mu', 'sd'):
            assert len(d[k]) == len(FEATS), '%s 长度失配' % k
        assert isinstance(d.get('theta'), float) and 0.0 < d['theta'] < 1.0
        _model_cache = d
    except Exception as e:                                   # noqa: BLE001
        logger.warning('joint_gate 模型不可用，钩子直通（%s）: %s', path, e)
        _model_cache = False
    return _model_cache


def compute_features(question, graph_contexts, manual_eng):
    """七特征现算（与 _jointgate_port_20261019.py VC 侧特征逐行同码）。

    只读图（UNWIND 度数/邻居展开/路径展开全为 MATCH）；返回长度 7 的列表，
    取不到的 f4/f5/f6 为 None（部署侧按 med 插补，与离线一致）。
    """
    from backend.evaluator import _ooc_gate_relevance, _ooc_object_mismatch
    f1 = float(_ooc_gate_relevance(question, graph_contexts))
    f2 = float(bool(_ooc_object_mismatch(question, graph_contexts)))
    matched = manual_eng._match_entities(question) or []
    f3 = float(len(matched))
    f4 = f5 = f6 = None
    store = manual_eng.graph_store
    src = manual_eng.source_manual or 'shukongdashi_500cases.md'
    if matched:
        degs = {}
        for r in store._driver.session().run(
                'UNWIND $names AS nm OPTIONAL MATCH '
                '(n:Entity {source_manual:$src, name:nm})-[r:REL]-() '
                'RETURN nm AS name, count(r) AS deg',
                names=list(matched)[:12], src=src):
            degs[r['name']] = r['deg']
        dv = [degs.get(nm, 0) for nm in matched[:12]]
        f4 = sum(dv) / float(len(dv))
        from backend.engines import _expand_paths_per_depth
        have = examined = 0
        for e in matched[:3]:
            examined += 1
            try:
                prows = _expand_paths_per_depth(
                    store._driver, 'Entity', e, key_field='name',
                    source_manual=src, per_depth=2, max_hop=2)
            except Exception:                                # noqa: BLE001
                prows = []
            for row in prows:
                types = set(str(t) for t in (row[1] if len(row) >= 2 else []))
                if len(types & set(_CHAIN_RELS)) >= 2:
                    have += 1
                    break
        f5 = 1.0 - have / float(examined)
        nb = []
        for e in matched[:5]:
            if len(nb) >= 50:
                break
            for r in store._driver.session().run(
                    'MATCH (n:Entity {source_manual:$src, name:$e})-[r:REL]-(m:Entity) '
                    'WHERE m.name IS NOT NULL RETURN m.name AS name '
                    'ORDER BY name LIMIT 50',
                    src=src, e=e):
                nb.append(r['name'])
        nb = nb[:50]
        f6 = float(_ooc_gate_relevance(question, [''.join(nb)]))
    import jieba
    jieba.setLogLevel(60)
    toks = [t for t in jieba.lcut(question)
            if len(t) >= 2 and not re.fullmatch(r'[\d\.\-/#]+', t)]
    names = manual_eng._entity_names or []
    if toks and names:
        anchored = sum(1 for t in toks if any(t in nm for nm in names))
        f7 = anchored / float(len(toks))
    else:
        f7 = None
    return [f1, f2, f3, f4, f5, f6, f7]


def score_prob(feats, model):
    """按部署系数打分：med 插补 → z 归一 → logistic。返回 [0,1] 概率。"""
    import math
    coefs = model['coefs']
    z = float(model['intercept'])
    for f, v, med, mu, sd in zip(FEATS, feats, model['med'],
                                 model['mu'], model['sd']):
        vv = float(v) if v is not None else float(med)
        c = float(coefs[f] if isinstance(coefs, dict) else coefs[FEATS.index(f)])
        z += c * ((vv - float(mu)) / max(float(sd), 1e-12))
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    ez = math.exp(z)
    return ez / (1.0 + ez)


def joint_gate_hook(question, graph_contexts, refusal_reason, manual_eng=None):
    """加法钩子：前级门放行 ⇒ 七特征打分，≥theta ⇒ 拒答。

    返回 (refusal_reason, record)；record 非 None 表示本钩子作出了拒答。
    任何路径都不抛异常、不撤回已有拒答。
    """
    try:
        if os.getenv('OOC_JOINT_GATE', '1') != '1':
            return refusal_reason, None
        if refusal_reason is not None or not graph_contexts or manual_eng is None:
            return refusal_reason, None
        model = _load_model()
        if not model:
            return refusal_reason, None
        feats = compute_features(question, graph_contexts, manual_eng)
        p = score_prob(feats, model)
        if p >= model['theta']:
            rec = {'score': round(p, 4), 'theta': model['theta'],
                   'feats': dict((f, (None if v is None else round(float(v), 4)))
                                 for f, v in zip(FEATS, feats))}
            return {'joint_gate': rec}, rec
        return refusal_reason, None
    except Exception as e:                                   # noqa: BLE001
        logger.warning('joint_gate 钩子异常（降级现门）: %s', e)
        return refusal_reason, None
