# -*- coding: utf-8 -*-
"""OPT_A_SLOTS 组装优先级测试（2026-08-24 修正断言语义）

原版断言"OPT_A 关闭时社区应为 0 条"与设计语义不符：
OPT_A 的功能点是【开启】时给 dense chunk 保 ≥3 槽（路径 cap 到
max_ctx-3）；【关闭】时回退原有动态组装——相关路径优先、
min_path_slots 兜底、剩余槽位按序给社区（2026-08-05 设计，
见 _collect_graph_contexts 注释）。关闭≠社区清零。

另外原场景 query 无 token 命中 → rel_paths 为空 → 走 else 分支，
根本没有进入 OPT_A 生效的 if rel_paths: 路径。本版改为
"主轴驱动器"短语命中全部路径，确保覆盖目标分支。
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
        # 10 条路径，节点名含"主轴驱动器"使 query 短语命中全部路径
        return [(["主轴驱动器%d" % i, "E%d" % i], ["表现为", "可能原因"], ["-", "+"])
                for i in range(10)]


class FakeComm:
    def query(self, q, chunk_fallback=True):
        comm = [
            "[硬事实] 社区硬事实内容DENSE-CHUNK-1",
            "[硬事实] 社区硬事实内容DENSE-CHUNK-2",
            "[硬事实] 社区硬事实内容DENSE-CHUNK-3",
            "[社区摘要] 社区摘要内容SUM-1",
            "[社区摘要] 社区摘要内容SUM-2",
        ]
        return [], comm


def _is_path(c):
    s = str(c)
    return "-->" in s or "<--" in s


def _is_comm(c):
    return str(c).startswith(("[硬事实]", "[社区摘要]", "[社区报告]"))


def run(opt_val):
    os.environ["OPT_A_SLOTS"] = opt_val
    return _collect_graph_contexts(
        "主轴驱动器故障怎么排查", FakeManual(), FakeComm(),
        max_ctx=12, min_path_slots=6,
        entity_chunk_map={"主轴驱动器0": [0]},
        chunk_texts=["A" * 2000],
    )


def test_opt_a_off():
    """关闭：路径不被 cap（10 条全进），社区仅自然补位剩余槽。"""
    ctxs = run("0")
    assert len(ctxs) == 12, "应填满 12 槽（当前 %d）" % len(ctxs)
    n_path = sum(1 for c in ctxs if _is_path(c))
    n_comm = sum(1 for c in ctxs if _is_comm(c))
    assert n_path == 10, "OPT_A 关闭时相关路径不应被 cap（当前 %d）" % n_path
    assert n_comm == 12 - n_path, \
        "社区应为自然补位 %d 条（当前 %d）" % (12 - n_path, n_comm)
    return len(ctxs), n_path, n_comm


def test_opt_a_on():
    """开启：路径 cap 到 max_ctx-3=9，dense chunk 社区保底 ≥3 槽。"""
    ctxs = run("1")
    assert len(ctxs) == 12, "应填满 12 槽（当前 %d）" % len(ctxs)
    n_path = sum(1 for c in ctxs if _is_path(c))
    n_comm = sum(1 for c in ctxs if _is_comm(c))
    assert n_path <= 9, "OPT_A 开启时路径应 cap 到 9（当前 %d）" % n_path
    assert n_comm >= 3, "OPT_A 开启时社区应保证 >=3 槽（当前 %d）" % n_comm
    return len(ctxs), n_path, n_comm


def main():
    l1, p1, c1 = test_opt_a_off()
    print("OPT_A=0: 总槽=%d 路径=%d 社区=%d (期望 路径10/社区2)" % (l1, p1, c1))
    l2, p2, c2 = test_opt_a_on()
    print("OPT_A=1: 总槽=%d 路径=%d 社区=%d (期望 路径<=9/社区>=3)" % (l2, p2, c2))
    print("PASS")


if __name__ == "__main__":
    main()
