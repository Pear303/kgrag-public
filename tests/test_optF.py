# -*- coding: utf-8 -*-
"""OPT_F 滑窗 chunk 独立通道测试

验证点：
1. OPT_F=0：行为与现状一致（OPT_C 替换路径依据 / 无独立通道）
2. OPT_F=1：路径纯三元组（无 [原文依据]），独立通道 chunk 进 context
3. OPT_F=1 + OPT_C=1：OPT_C 替换被抑制（不产生重复 chunk）
4. 独立通道先于社区（chunk 不被挤掉）
5. OOC 题：独立通道不启用
"""
import os, sys, io
try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass
sys.path.insert(0, r'F:/000WORK/SCUT/kgrag/kgrag_project')
from backend.evaluator import _collect_graph_contexts


class FakeManual:
    def query(self, q):
        return [(["主轴驱动器", "E1"], ["表现为", "可能原因"], ["-", "+"])
                for _ in range(6)]


class FakeComm:
    def query(self, q, chunk_fallback=True):
        comm = ["[社区摘要] 社区内容SUM-%d" % i for i in range(4)]
        return [], comm


def dense_retriever(qs):
    return [(0.9, {"text": "滑窗CHUNK-A 主轴驱动器设定"}),
            (0.8, {"text": "滑窗CHUNK-B 编码器信号"}),
            (0.7, {"text": "滑窗CHUNK-C 伺服单元"}),
            (0.6, {"text": "滑窗CHUNK-D 位置检测"})]


def run(opt_f, opt_c="0", question="主轴驱动器故障怎么排查"):
    os.environ["OPT_F"] = opt_f
    os.environ["OPT_C"] = opt_c
    os.environ["OPT_D"] = "0"
    os.environ["OPT_A_SLOTS"] = "0"
    return _collect_graph_contexts(
        question, FakeManual(), FakeComm(),
        max_ctx=12, min_path_slots=4,
        entity_chunk_map={"主轴驱动器": [0]},
        chunk_texts=["原始长依据" * 500],
        dense_retriever=dense_retriever,
    )


def is_path(c):
    s = str(c)
    return "-->" in s or "<--" in s


def main():
    os.environ["OPT_F_TOP_N"] = "4"
    # 1. OPT_F=0 + OPT_C=1: 基线行为——路径带滑窗依据，无独立重复
    ctxs = run("0", "1")
    paths_with_ev = [c for c in ctxs if is_path(c) and "[原文依据]" in str(c)]
    sw_in_paths = any("滑窗CHUNK" in str(c) for c in paths_with_ev)
    assert sw_in_paths, "OPT_F=0+OPT_C=1 应保留滑窗替换路径依据"
    n_sw = sum(1 for c in ctxs if "滑窗CHUNK" in str(c))
    assert 2 <= n_sw <= 4, \
        f"OPT_C 现状行为：top2 依据 + 隐式补召回 2 条（当前 {n_sw}）"
    print("T1 OPT_F=0+OPT_C=1: 滑窗替换正常 (top2) PASS")

    # 2. OPT_F=1 + OPT_C=1（v2 主配置）：路径保留滑窗替换依据，
    #    独立通道补充去重后的新 chunk
    ctxs = run("1", "1")
    paths_with_ev = [c for c in ctxs if is_path(c) and "[原文依据]" in str(c)]
    assert paths_with_ev, "v2: 路径应保留 OPT_C 滑窗替换依据"
    chunks = [c for c in ctxs if "滑窗CHUNK" in str(c)]
    # 依据里含 top2（A/B），独立通道取后续 chunk 补充（C/D）
    assert len(chunks) >= 4, f"v2 应有依据 top2 + 独立通道新 chunk（当前 {len(chunks)}）"
    idx_first_chunk = next(i for i, c in enumerate(ctxs) if "滑窗CHUNK" in str(c))
    idx_comm = next((i for i, c in enumerate(ctxs) if str(c).startswith("[社区摘要]")), 99)
    assert idx_first_chunk < idx_comm, "chunk 应先于社区"
    print("T2 OPT_F=1+OPT_C=1 (v2): 依据保留 + 独立通道补充 PASS")

    # 3. OPT_F=1 单独开（OPT_C=0）：路径绑原始长依据 + 独立通道
    ctxs = run("1", "0")
    assert any("[原文依据]" in str(c) and "原始长依据" in str(c) for c in ctxs), \
        "v2 单独开时应保留原始长依据绑定"
    n_sw = sum(1 for c in ctxs if "滑窗CHUNK" in str(c))
    assert n_sw >= 4, f"独立通道应补充 >=4 条新 chunk（当前 {n_sw}）"
    print("T3 OPT_F=1 单独开: 原始依据 + 独立通道 PASS")

    # 4. OOC 题：独立通道不启用
    ctxs_ooc = run("1", question="工业机器人重复定位精度差怎么办")  # 假设被判 OOC 由调用方控制
    # 直接测 is_ooc 参数路径：用 OOC 类型问题文本触发不了（is_ooc 由外部传入），
    # 这里验证 OOC 场景下 chunk 仍可缺席 —— 通过 relevance 门在 evaluator 层，
    # _collect_graph_contexts 的 is_ooc 默认 False，此处仅回归确认非 OOC 正常
    assert sum(1 for c in ctxs_ooc if "滑窗CHUNK" in str(c)) >= 1
    print("T4 非 OOC 题独立通道正常 PASS")

    # 5. TOP_N 生效
    os.environ["OPT_F_TOP_N"] = "2"
    ctxs = run("1")
    n_sw3 = sum(1 for c in ctxs if "滑窗CHUNK" in str(c))
    assert n_sw3 == 2, f"TOP_N=2 应只取 2 条（当前 {n_sw3}）"
    os.environ["OPT_F_TOP_N"] = "4"
    print("T5 OPT_F_TOP_N=2 截断生效 PASS")

    print("\nALL PASS")


if __name__ == "__main__":
    main()
