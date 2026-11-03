# -*- coding: utf-8 -*-
"""检索用分块世代分发器（v6 新世代分块方案的总开关）

背景：生产有三处分块入口，其中两处必须**同函数同参数**才能保证 KG-RAG 与 Vector-RAG
公平对比（历史上靠 runner 注释约定"两侧同一函数"）：

  1. KG-RAG 的 OPT_C dense 通道   → backend/evaluator.py::_get_sliding_retriever
  2. Vector-RAG 的文档池          → run_pump_fault.py::build_vector_rag
  3. KG-RAG 图谱路径的 chunk_texts → 走 parse_manual_chunks（结构化解析，**不在本开关范围**）

本模块只覆盖 1 与 2，使两者按同一开关切换到同一分块世代。

开关（默认全部保持历史行为）：
  OPT_TABLE_ATOMIC=0|1        0 = 滑窗分块（历史默认，bit-identical）；1 = 表格原子分块
  OPT_TABLE_ATOMIC_MAX=1200   表格原子分块的单块字数上限（0 = 不限长）

NONE of these change judge/metric/truncation semantics — 仅改变送入检索器的文本单元。
"""
import os

from backend.parser import parse_manual_chunks_sliding_window

TRUTHY = ("1", "true", "True", "yes", "on")


def table_atomic_enabled() -> bool:
    return os.getenv("OPT_TABLE_ATOMIC", "0") in TRUTHY


def table_atomic_max_chars() -> int:
    try:
        return int(os.getenv("OPT_TABLE_ATOMIC_MAX", "1200"))
    except ValueError:
        return 1200


def parse_manual_chunks_for_retrieval(path: str, window_size: int = None,
                                      overlap_size: int = None):
    """按 OPT_TABLE_ATOMIC 返回分块列表（元素为 backend.parser.Chunk）。

    window_size/overlap_size 为 None 时取 config 的 VECTOR_WINDOW_SIZE/OVERLAP_SIZE，
    保证与历史调用 parse_manual_chunks_sliding_window(path) 的行为一致。
    """
    if window_size is None or overlap_size is None:
        from config import VECTOR_WINDOW_SIZE, VECTOR_OVERLAP_SIZE
        window_size = VECTOR_WINDOW_SIZE if window_size is None else window_size
        overlap_size = VECTOR_OVERLAP_SIZE if overlap_size is None else overlap_size

    if not table_atomic_enabled():
        return parse_manual_chunks_sliding_window(path, window_size, overlap_size)

    from backend.table_atomic import chunk_table_atomic, detect_tables
    lines = open(path, encoding='utf-8').read().split('\n')
    tables = detect_tables(lines)
    return chunk_table_atomic(path, window_size, overlap_size,
                              table_atomic_max_chars(), tables=tables)


def chunk_generation_label() -> str:
    """落盘/日志用的世代标签。"""
    if table_atomic_enabled():
        return f"table_atomic(max={table_atomic_max_chars()})"
    return "sliding_window"
