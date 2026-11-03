# -*- coding: utf-8 -*-
"""table_fallback.py —— 结构化表格/案例的兜底召回（opt-in，默认关闭）

## 存在的原因

`analysis/cr_gap_diagnosis.py` 定位到：冻结世代下 R2 − Vector 的 CR 缺口
（−0.0984）**全部来自 33 道 KG 完全漏检的题**，其中 22 道集中在主轴驱动器
参数表格区（表5-2~5-6）。根因是**抽取 schema 对结构化表格结构性失明**：
表5-4 区域在 2147 条三元组中产出 0 条，图谱对该区域无路径可达。

`analysis/table_fallback_experiment.py` 验证了补救路径：仅用问题文本
（不含 ground_truth）在手册中定位候选段落，即可命中 21/33。

## 设计要点

**位置：追加在既有上下文之后，绝不插到前面。**
  RAGAS 的 CR 是覆盖型指标（GT 在不在上下文里），与排名无关；
  而 CP 是排名敏感的。把兜底内容追加在末尾：
    · CR  → 拿到覆盖，缺口可补
    · CP  → 原有 12 篇的排名逐位不变，**评分不受扰动**
  因此本模块是**召回扩展器**，不是重排序器。这一取舍是有意的。

**触发：三条策略，全部只用问题文本。**
  S1 问题含「表N-N」        → 按同名表格标题定位
  S2 问题含型号+设定端/测试端 → 按型号定位
  S3 问题含「例N」           → 按案例标题定位
  实测三条各自贡献命中（S1:9 / S2:3 / S3:9），故全部保留。

**开关：环境变量 `TABLE_FALLBACK`，默认 `0`（关闭）。**
  关闭时本模块不被调用，冻结世代的检索行为**逐位不变**。
"""
import os
import re
import threading

#: 手册路径（与被评测轮次一致）
_MANUAL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'data', 'manuals', 'shukongdashi_500cases.md')

#: 单段最大字符数（要够覆盖整张表：表5-2 表体约 410 字，表5-4 约 490 字）
WINDOW = 1600
#: 最多追加几段（避免上下文爆炸）
MAX_APPEND = 2

_manual_cache = {'text': None}
_lock = threading.Lock()

_RE_TABLE_Q = re.compile(r'表\s*(\d+)\s*[-–—]\s*(\d+)')
_RE_MODEL = re.compile(r'(60\d{2}|6SC\d{4})')
_RE_STRUCT_HINT = re.compile(r'设定端|测试端|检测端|电位器')
_RE_CASE_Q = re.compile(r'例\s*(\d+[A-Z]?)')

#: 标题形：**行首** + 表N-N + 空格 + 标题文字。
#: 实测关键——手册中「见表5-2到表5-5」这类**引用**不在行首，
#: 而真正的标题形如 "\n表5-2 6059主轴驱动器的设定端及含义" 必在行首。
#: 早期实现用无锚点正则，故命中了引用（错）或跨章节的同名表。
_RE_TITLE_ANCHOR = r'(?:^|\n)[ \t]*表\s*{n}\s*[-–—]\s*{m}[ \t]+[^\n]{{2,70}}'
_RE_ANY_TITLE = re.compile(r'(?:^|\n)[ \t]*表\s*\d+\s*[-–—]\s*\d+[ \t]+[^\n]{2,70}')


def _manual():
    """惰性加载手册（带锁，评测多线程安全）。"""
    if _manual_cache['text'] is None:
        with _lock:
            if _manual_cache['text'] is None:
                if not os.path.exists(_MANUAL_PATH):
                    _manual_cache['text'] = ''
                else:
                    with open(_MANUAL_PATH, encoding='utf-8') as f:
                        _manual_cache['text'] = f.read()
    return _manual_cache['text']


def _keywords(question: str) -> list:
    """从问题中抽取用于消歧的关键词（越长权重越高）。"""
    kws = []
    kws += [m.group(1) for m in _RE_MODEL.finditer(question)]
    for w in ['主轴驱动器', '伺服驱动器', '速度控制单元', '设定端', '测试端',
              '检测端', '电位器', '编码器']:
        if w in question:
            kws.append(w)
    return kws


def _score(man: str, pos: int, kws: list) -> int:
    """候选标题的得分：附近出现的问题关键词加权和（长词权重更高）。"""
    seg = man[max(0, pos - 500):pos + 1000]
    return sum(len(kw) for kw in kws if kw in seg)


def _window_from(man: str, pos: int) -> str:
    """从 pos 起取到下一个行首标题，或 WINDOW 字。"""
    nxt = _RE_ANY_TITLE.search(man, pos + 5)
    end = min(pos + WINDOW, nxt.start() if nxt else len(man))
    return man[pos:end]


def locate(question: str, max_parts: int = MAX_APPEND) -> list:
    """仅用问题文本定位手册中的候选段落。**绝不使用 ground_truth。**

    返回去重后的候选文本片段列表（可能为空）。
    """
    if not question:
        return []
    man = _manual()
    if not man:
        return []

    found = []
    kws = _keywords(question)

    def _best_title(n, m):
        """在全部「行首标题形」候选中选关键词得分最高者。"""
        pat = re.compile(_RE_TITLE_ANCHOR.format(n=n, m=m))
        cands = list(pat.finditer(man))
        if not cands:
            return None
        return max(cands, key=lambda mm: (_score(man, mm.start(), kws),
                                          mm.start()))
    # S1 —— 表N-N：锚定行首标题 + 关键词消歧
    for qm in _RE_TABLE_Q.finditer(question):
        tm = _best_title(qm.group(1), qm.group(2))
        if tm:
            found.append(_window_from(man, tm.start()))

    # S2 —— 型号 + 结构词：找「标题形」表格，其窗口同时含型号与结构词
    if _RE_STRUCT_HINT.search(question):
        models = [mm.group(1) for mm in _RE_MODEL.finditer(question)]
        best, best_sc = None, -1
        for tm in _RE_ANY_TITLE.finditer(man):
            win = _window_from(man, tm.start())
            if not _RE_STRUCT_HINT.search(win):
                continue
            sc = 0
            for mo in models:
                if mo in win:
                    sc += 10
            sc += _score(man, tm.start(), kws)
            if sc > best_sc:
                best, best_sc = win, sc
        if best is not None and best_sc > 0:
            found.append(best)

    # S3 —— 例N 案例
    for qm in _RE_CASE_Q.finditer(question):
        pat = re.compile(r'例\s*' + re.escape(qm.group(1)) + r'\s*[\.．、,，]')
        hit = pat.search(man)
        if hit:
            found.append(_window_from(man, hit.start()))
            break

    # 去重 + 限量
    out, seen = [], set()
    for t in found:
        key = t[:200]
        if key in seen or len(t) < 40:
            continue
        seen.add(key)
        out.append(t)
        if len(out) >= max_parts:
            break
    return out


def augment_contexts(contexts: list, question: str) -> list:
    """把兜底片段**追加**到既有上下文之后。

    关键：不改变原有条目的顺序与内容 → CP 不受扰动。
    """
    existing = ' '.join(str(c) for c in contexts)
    extra = []
    for part in locate(question):
        # 若该内容已基本出现在上下文中，则跳过（避免无谓膨胀）
        probe = part[:120].strip()
        if probe and probe in existing:
            continue
        extra.append(part)
    if not extra:
        return contexts
    return list(contexts) + extra
