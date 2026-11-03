"""文档解析与三元组抽取入口（文章 3.1 / 3.2 节）。

职责：
1. parse_manual_chunks：把维保手册拆分为带 chunk_id 的文本块（原子单元）。
2. parse_manual：从 Markdown 表格中规则化提取确定性三元组（人工规则图谱）。
   例：(轴承座温度升高(>80°C), 可能原因, 润滑油位不足或径向载荷过大)
   这些三元组通过 seed_triples 写入 Neo4j，标签为 :Entity。
3. parse_manual_with_provenance：带行号溯源的三元组抽取（v2 新增）。
"""
from dataclasses import dataclass, field
from typing import List, Tuple

import re
import os

from config import MAX_CHUNK_SIZE, MANUALS_DIR, VECTOR_WINDOW_SIZE, VECTOR_OVERLAP_SIZE


@dataclass
class Chunk:
    chunk_id: int
    text: str
    source: str


def parse_manual_chunks(path: str, max_chunk_size: int = MAX_CHUNK_SIZE) -> List[Chunk]:
    """将手册按段落切分为带 chunk_id 的文本块。

    以空行 / 标题切分，超长段落再按字符数硬切，保证可追溯性。
    """
    with open(path, "r", encoding="utf-8") as f:
        raw = f.read()

    blocks = re.split(r"\n\s*\n", raw)
    chunks: List[Chunk] = []
    cid = 0
    buf = ""
    for blk in blocks:
        blk = blk.strip()
        if not blk:
            continue
        if len(buf) + len(blk) <= max_chunk_size:
            buf = (buf + "\n" + blk).strip()
        else:
            if buf:
                chunks.append(Chunk(chunk_id=cid, text=buf, source=path))
                cid += 1
            # 超长块硬切
            for i in range(0, len(blk), max_chunk_size):
                piece = blk[i:i + max_chunk_size]
                chunks.append(Chunk(chunk_id=cid, text=piece, source=path))
                cid += 1
            buf = ""
    if buf:
        chunks.append(Chunk(chunk_id=cid, text=buf, source=path))
    return chunks


def parse_manual_chunks_sliding_window(
    path: str,
    window_size: int = VECTOR_WINDOW_SIZE,
    overlap_size: int = VECTOR_OVERLAP_SIZE
) -> List[Chunk]:
    """将手册按滑动窗口切分为带 chunk_id 的文本块（Vector RAG 专用）。

    滑动窗口策略：
    1. 先按段落分割，合并到窗口大小
    2. 窗口内按固定步长滑动，产生重叠的 chunks
    3. 每个 chunk 保持语义完整性（尽量在句号处断开）

    Args:
        path: 手册文件路径
        window_size: 窗口大小（字符数），默认 500
        overlap_size: 重叠大小（字符数），默认 50
    """
    with open(path, "r", encoding="utf-8") as f:
        raw = f.read()

    # 先按段落分割
    paragraphs = re.split(r"\n\s*\n", raw)
    full_text = "\n".join(p.strip() for p in paragraphs if p.strip())

    if len(full_text) <= window_size:
        return [Chunk(chunk_id=0, text=full_text, source=path)]

    chunks: List[Chunk] = []
    cid = 0
    step = window_size - overlap_size  # 步长 = 窗口 - 重叠
    i = 0
    while i < len(full_text):
        end = min(i + window_size, len(full_text))
        chunk_text = full_text[i:end]
        next_start = i + step  # 默认下一次起点（修复 M2：真正推进游标）

        # 尽量在句号处断开，避免切断句子
        if end < len(full_text):
            last_period = chunk_text.rfind("。")
            last_comma = chunk_text.rfind("，")
            last_newline = chunk_text.rfind("\n")

            # 找到最后一个合适的断点，仅在断点足够靠后时应用
            cut_point = max(last_period, last_comma, last_newline)
            if cut_point > window_size * 0.5:  # 确保不会切太短
                chunk_text = full_text[i:i + cut_point + 1]
                next_start = i + cut_point + 1  # 下一次从断点后开始

        chunks.append(Chunk(chunk_id=cid, text=chunk_text.strip(), source=path))
        cid += 1

        if end >= len(full_text):
            break
        i = next_start  # 修复 M2：i_next 现在真正控制下一次窗口起点

    return chunks


Triplet = Tuple[str, str, str]


def normalize_entity(text: str) -> str:
    """实体归一化：统一全角/半角、℃/°C、去首尾空格与冗余标点。

    解决问题：`轴承座温度升高(>80℃)` 与 `轴承座温度升高(>80°C)` 实为同一实体。
    """
    text = text.strip()
    # 全角→半角括号
    text = text.replace("（", "(").replace("）", ")")
    # 统一温度符号
    text = text.replace("℃", "°C").replace("°C", "°C")
    # 全角逗号→半角
    text = text.replace("，", ",")
    # 去首尾标点和多余空格
    text = re.sub(r"\s+", " ", text).strip()
    text = text.strip("，。、；：,.:; ")
    return text


def parse_manual(path: str, do_normalize: bool = True) -> List[Triplet]:
    """从 Markdown 表格中规则化提取三元组（人工规则图谱）。

    支持表格行形如：| 实体A | 关系 | 实体B |
    仅保留 ALLOWED_RELATIONS 中的关系，过滤噪音。
    v2: 默认对实体做归一化（解决全角/半角重复节点问题）。
    """
    from config import ALLOWED_RELATIONS, FORBIDDEN_RELATIONS

    triples: List[Triplet] = []
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    for line in lines:
        if not line.strip().startswith("|"):
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) < 3:
            continue
        # 跳过表头分隔行 |---|---|
        if all(set(c) <= set("-: ") for c in cells):
            continue
        s, r, o = cells[0], cells[1], cells[2]
        if not s or not o:
            continue
        if r in FORBIDDEN_RELATIONS:
            continue
        if r in ALLOWED_RELATIONS:
            if do_normalize:
                s, o = normalize_entity(s), normalize_entity(o)
            triples.append((s, r, o))
    return triples


def parse_manual_with_provenance(path: str, do_normalize: bool = True) -> List[dict]:
    """带行号溯源的三元组抽取（v2 新增）。

    返回: [{"s":..., "r":..., "o":..., "source_manual":..., "source_line":...}, ...]
    每个三元组可反查到手册原文行号。
    """
    from config import ALLOWED_RELATIONS, FORBIDDEN_RELATIONS

    source_name = os.path.basename(path)
    results: List[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    for line_no, line in enumerate(lines, start=1):
        if not line.strip().startswith("|"):
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) < 3:
            continue
        if all(set(c) <= set("-: ") for c in cells):
            continue
        s, r, o = cells[0], cells[1], cells[2]
        if not s or not o:
            continue
        if r in FORBIDDEN_RELATIONS:
            continue
        if r in ALLOWED_RELATIONS:
            if do_normalize:
                s, o = normalize_entity(s), normalize_entity(o)
            results.append({
                "s": s, "r": r, "o": o,
                "source_manual": source_name,
                "source_line": line_no,
            })
    return results


def seed_triples(graph_store, triples: List[Triplet], source_manual: str = None, source_line: int = None) -> None:
    """将人工规则三元组写入 Neo4j，标签为 :Entity（文章 3.2 节）。

    graph_store 为 backend.graph_store.GraphStore 实例；为兼容旧签名，
    若传 None 则新建一个临时实例。
    v2: 支持传入 source_manual + source_line 写入溯源属性。
    """
    from backend.graph_store import GraphStore
    from config import NEO4J_BUILD_URI
    if graph_store is None:
        graph_store = GraphStore(uri=NEO4J_BUILD_URI)
    graph_store.seed_triples(triples, source_manual=source_manual, source_line=source_line)


def parse_manual_structured(path: str) -> List[Chunk]:
    """按 GSK983 手册结构解析：每个报警条目/参数表/诊断表/案例 = 一个 chunk
    
    解析策略：
    1. 报警条目: 按 "报警号 XXX" 或 "(N)报警号 XXX" 分割，每个报警条目 = 1个 chunk
    2. 参数表: 按 参数号（如 0000~0005）分割
    3. 诊断表: 按 "诊断 DGN#XXX" 或 "DGN#XXX" 分割
    4. 维修案例: 按案例条目分割
    5. 兜底: 其他内容按原有滑动窗口切分
    """
    with open(path, "r", encoding="utf-8") as f:
        raw = f.read()
    
    chunks: List[Chunk] = []
    
    # 1. 报警详情分块：3.2.3 根据报警号查找故障 中的诊断表
    # 格式: "(1)报警号 001 TH报警。..." 到下一个 "(N)报警号"
    detail_start = raw.find('3.2.3')
    if detail_start == -1:
        detail_start = raw.find('根据报警号查找故障')
    
    if detail_start >= 0:
        detail_sec = raw[detail_start:]
        # 每个报警详情条目: (N)报警号 XXX 开始，到下一个 (N+1)报警号 或 3.2.4
        alarm_entry_pattern = r'(\(\d+\)报警号\s*\d{3}.*?)(?=\(\d+\)报警号\s*\d{3}|3\.2\.\d+|$)'
        entries = list(re.finditer(alarm_entry_pattern, detail_sec, re.DOTALL))
        
        # 合并相邻条目，按 2-3 个一组分块
        grouped = []
        for i in range(0, len(entries), 3):
            group = entries[i:i+3]
            text = '\n'.join(m.group(1).strip() for m in group)
            if len(text.strip()) > 100:
                grouped.append(text.strip())
        
        # 如果分组太少，直接每个条目一个 chunk
        if len(grouped) < 5 and entries:
            for m in entries:
                text = m.group(1).strip()
                if len(text) > 50:
                    grouped.append(text)
        
        for text in grouped:
            chunks.append(Chunk(chunk_id=len(chunks), text=text, source=path))
    
    # 2. 报警含义表分块（2.1 程序错误或操作错误 等章节）
    # 按报警号分割
    meaning_sections = ['2.1 程序错误或操作错误', '2.2 行程极限开关报警',
                        '2.3 伺服系统报警', '2.4 主机各板之间连接报警',
                        '2.5 存储器错误']
    for sec_title in meaning_sections:
        sec_start = raw.find(sec_title)
        if sec_start < 0:
            continue
        # 找到下一章节
        sec_end = len(raw)
        for next_title in ['2.2', '2.3', '2.4', '2.5', '3．故障查找']:
            if next_title == sec_title[:3] and next_title + ' ' not in sec_title:
                continue
            idx = raw.find(next_title, sec_start + 10)
            if idx > sec_start and idx < sec_end:
                sec_end = idx
                break
        meaning_sec = raw[sec_start:sec_end]
        
        # 按每个报警号分组（如 "090 含义..." 到 "091 含义..."）
        # 模式: 行首3位数字 + 内容
        entry_pattern = r'(^\d{3}\s+.*?)(?=^\d{3}\s+|$|\n\n\s*\d+\n)'
        entries = list(re.finditer(entry_pattern, meaning_sec, re.MULTILINE | re.DOTALL))
        
        # 按每 15 个一组分块
        for i in range(0, len(entries), 15):
            group = entries[i:i+15]
            text = '\n'.join(m.group(1).strip() for m in group if m.group(1).strip())
            if len(text.strip()) > 100:
                chunks.append(Chunk(chunk_id=len(chunks), text=text.strip(), source=path))
    
    # 3. 参数表分块
    # 参数一览表部分
    for sec_title in ['3．参数的初始设定值', '1．参数号一览表', '2．参数功能一览表', '（1）与伺服有关的参数', '（2）主轴机能']:
        sec_start = raw.find(sec_title)
        if sec_start < 0:
            continue
        # 找到下一章节
        sec_end = len(raw)
        for marker in ['3.2.3', '3.2.4', '附录', '3.2.11']:
            idx = raw.find(marker, sec_start + 10)
            if idx > sec_start and idx < sec_end:
                sec_end = idx
        param_sec = raw[sec_start:min(sec_end, sec_start + 6000)]
        
        # 参数表条目，每 20 行一组
        lines = [l.strip() for l in param_sec.split('\n') if l.strip()]
        for i in range(0, len(lines), 20):
            group = lines[i:i+20]
            text = '\n'.join(group)
            if len(text) > 100:
                chunks.append(Chunk(chunk_id=len(chunks), text=text, source=path))
    
    # 4. 维修案例分块: 附录4
    case_start = raw.find('(附录 4)')
    if case_start >= 0:
        case_sec = raw[case_start:]
        # 每个案例条目按 数字. 分块
        entries = list(re.finditer(r'(故障类型\s+故障现象\s+故障检查及处理方法\s+备注.*?)(?=故障类型\s+故障现象|$)', case_sec, re.DOTALL))
        for m in entries:
            text = m.group(1).strip()
            if len(text) > 200:
                chunks.append(Chunk(chunk_id=len(chunks), text=text, source=path))
    
    # 5. 兜底：剩余内容用滑动窗口切分（排除已处理的部分）
    if len(chunks) < 10:
        return parse_manual_chunks_sliding_window(path)
    
    return chunks
