# -*- coding: utf-8 -*-
"""表格原子分块（生产版，v6 新世代分块方案）

来源与一致性：本模块由 analysis/table_atomic_chunker.py（pilot P1a 抽出的独立模块）
移植而来，保留其全部逻辑与默认参数，使 pilot 阶段的确定性结论（表题 full 0.850→0.950）
在本模块上可复现。差异仅一处：返回 backend.parser.Chunk 对象（带 chunk_id/source），
以便直接接入既有的检索与评测链路。

规则：
  1. 表标题 → 下一表标题/例N/标题 之间的行视为该表正文；
  2. 表格区间整体成一个 chunk（**不进滑窗**，避免表被切断）；
  3. 超长表按**行边界**切分为多块，每块重复表头（【表号 标题】），
     默认单块上限 1200 字（不限长时 10,885 字的表块会拖累对照题）；
  4. 非表文本按 window/overlap 滑窗切分（与 parse_manual_chunks_sliding_window 同参数）。
"""
import re
from typing import List

from backend.parser import Chunk

W_DEFAULT = 500
OV_DEFAULT = 50
MAX_TABLE_CHARS_DEFAULT = 1200

RE_TITLE = re.compile(r'^\s*表\s*(\d+)\s*[-–—．.]\s*(\d+)\s*(.*)$')
RE_CASE = re.compile(r'^\s*例\s*\d+')
RE_HEAD = re.compile(r'^\s*#{1,6}\s')


def detect_tables(lines) -> list:
    """检出全部表标题并划定行范围（与 analysis/table_inventory.py 同逻辑）。

    返回 [{'idx': i, 'line': 标题行, 'end': 结束行, 'label': '表N-N', 'title': ...}, ...]
    """
    tables = []
    for i, ln in enumerate(lines):
        m = RE_TITLE.match(ln)
        if m:
            tables.append({'idx': len(tables), 'line': i,
                           'label': f'表{m.group(1)}-{m.group(2)}',
                           'title': m.group(3).strip()[:60]})
    for t in tables:
        end = len(lines)
        for j in range(t['line'] + 1, len(lines)):
            if RE_TITLE.match(lines[j]) or RE_CASE.match(lines[j]) or RE_HEAD.match(lines[j]):
                end = j
                break
        t['end'] = end
    return tables


def chunk_table_atomic(manual_path: str,
                       window_size: int = W_DEFAULT,
                       overlap_size: int = OV_DEFAULT,
                       max_table_chars: int = MAX_TABLE_CHARS_DEFAULT,
                       tables: list = None) -> List[Chunk]:
    """表格原子分块，返回 List[Chunk]（chunk_id 按输出顺序从 0 递增）。"""
    text = open(manual_path, encoding='utf-8').read()
    lines = text.split('\n')
    if tables is None:
        tables = detect_tables(lines)
    tsel = sorted(tables, key=lambda t: t['line'])
    segs = []
    cur = 0
    for t in tsel:
        a, b = t['line'], t['end']
        if a < cur:
            continue                          # 嵌套/重叠表区间跳过
        if a > cur:
            segs.append(('text', '\n'.join(lines[cur:a])))
        body = [l for l in lines[a:b] if l.strip()]
        head = f"【{t['label']} {t['title']}】"
        if max_table_chars and sum(len(x) + 1 for x in body) > max_table_chars:
            piece, size = [], 0
            for ln in body:
                if piece and size + len(ln) + 1 > max_table_chars:
                    segs.append(('table', head + '\n' + '\n'.join(piece)))
                    piece, size = [], 0
                piece.append(ln)
                size += len(ln) + 1
            if piece:
                segs.append(('table', head + '\n' + '\n'.join(piece)))
        else:
            segs.append(('table', head + '\n' + '\n'.join(body)))
        cur = b
    if cur < len(lines):
        segs.append(('text', '\n'.join(lines[cur:])))

    out_texts = []
    for kind, raw in segs:
        if kind == 'table':
            out_texts.append(raw.strip())
            continue
        paras = re.split(r'\n\s*\n', raw)
        full = '\n'.join(p.strip() for p in paras if p.strip())
        if not full:
            continue
        if len(full) <= window_size:
            out_texts.append(full)
            continue
        step = max(1, window_size - overlap_size)
        i = 0
        while i < len(full):
            piece = full[i:i + window_size]
            if len(piece) < 60 and out_texts:
                break
            out_texts.append(piece)
            i += step

    out_texts = [o for o in out_texts if len(o.strip()) >= 40]
    return [Chunk(chunk_id=i, text=t, source=manual_path)
            for i, t in enumerate(out_texts)]
