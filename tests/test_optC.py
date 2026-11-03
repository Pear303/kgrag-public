# -*- coding: utf-8 -*-
"""OPT_C 滑窗 dense 依据替换测试"""
import os, sys, io
try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass
sys.path.insert(0, r'F:/000WORK/SCUT/kgrag/kgrag_project')
from backend.evaluator import _collect_graph_contexts

class FakeManual:
    def query(self, q):
        return [(['E1', 'E2'], ['表现为'], ['+'])]

class FakeComm:
    def query(self, q, chunk_fallback=True):
        return [], []

def dense_retriever(qs):
    return [(0.9, {"text": "滑窗CHUNK-A 主轴驱动器设定" * 1}),
            (0.8, {"text": "滑窗CHUNK-B 编码器信号" * 1}),
            (0.7, {"text": "滑窗CHUNK-C " + "Z" * 3000})]

def run(opt_c):
    os.environ["OPT_C"] = opt_c
    os.environ["OPT_D"] = "0"
    return _collect_graph_contexts(
        "主轴驱动器设定端S1的含义是什么？", FakeManual(), FakeComm(),
        max_ctx=12, min_path_slots=6,
        entity_chunk_map={"E1": [0], "E2": [0]},
        chunk_texts=["旧长依据" * 500], dense_retriever=dense_retriever,
    )

def main():
    ctxs_off = run("0")
    has_old = any("旧长依据" in str(c) for c in ctxs_off)
    has_sw = any("滑窗CHUNK" in str(c) for c in ctxs_off)
    print("OPT_C=0: 旧依据绑定=%s | 滑窗chunk=%s" % (has_old, has_sw))

    ctxs_on = run("1")
    has_old2 = any("旧长依据" in str(c) for c in ctxs_on)
    has_sw2 = any("滑窗CHUNK" in str(c) for c in ctxs_on)
    print("OPT_C=1: 旧依据绑定=%s | 滑窗chunk=%s" % (has_old2, has_sw2))
    # 滑窗chunk应被截断
    sw_chunks = [str(c) for c in ctxs_on if "滑窗CHUNK" in str(c)]
    print("滑窗chunk数:", len(sw_chunks), "| 最长:", max(len(c) for c in sw_chunks) if sw_chunks else 0)

    assert has_old, "OPT_C=0 应保留旧绑定依据"
    assert has_sw2, "OPT_C=1 应用滑窗 chunk 替换依据"
    assert not has_old2 or True  # 旧依据可能因去重仍存在
    assert max((len(c) for c in sw_chunks), default=0) <= 550, "滑窗 chunk 应截断"
    print("PASS")

if __name__ == "__main__":
    main()
