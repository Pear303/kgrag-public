# -*- coding: utf-8 -*-
"""OPT_D 实体二次 dense 检索补召回测试"""
import os, sys, io
try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass
sys.path.insert(0, r'F:/000WORK/SCUT/kgrag/kgrag_project')
from backend.evaluator import _collect_graph_contexts

class FakeManual:
    def query(self, q):
        # 返回 1 条路径
        return [(['E1', 'E2'], ['表现为'], ['+'])]

class FakeComm:
    def query(self, q, chunk_fallback=True):
        return [], ["[社区摘要] 社区内容"]

def dense_retriever(qs):
    # mock dense 检索：返回 4 个新 chunk（含一个超长）
    return [(0.9, {"text": "DENSE-CHUNK-A 高频内容"}),
            (0.8, {"text": "DENSE-CHUNK-B 主轴内容"}),
            (0.7, {"text": "DENSE-CHUNK-C 编码器内容"}),
            (0.6, {"text": "DENSE-CHUNK-D " + "X" * 3000})]

def run(opt_val):
    os.environ["OPT_D"] = opt_val
    return _collect_graph_contexts(
        "主轴驱动器设定端S1的含义是什么？", FakeManual(), FakeComm(),
        max_ctx=12, min_path_slots=6,
        entity_chunk_map={}, chunk_texts=[], dense_retriever=dense_retriever,
    )

def main():
    ctxs_off = run("0")
    n_off = sum(1 for c in ctxs_off if str(c).startswith("DENSE-CHUNK"))
    print("OPT_D=0: 总槽=%d dense补chunk=%d (期望 0)" % (len(ctxs_off), n_off))

    ctxs_on = run("1")
    n_on = sum(1 for c in ctxs_on if str(c).startswith("DENSE-CHUNK"))
    print("OPT_D=1: 总槽=%d dense补chunk=%d (期望 <=2)" % (len(ctxs_on), n_on))

    assert n_off == 0, "OPT_D 关闭时不应有 dense 补召回"
    assert n_on <= 2, "OPT_D 补召回应限制 <=2 条（当前 %d）" % n_on
    # 超长 chunk 应被截断（独立验证：只返回超长 chunk）
    os.environ["OPT_D"] = "1"
    def long_only(qs):
        return [(0.9, {"text": "DENSE-CHUNK-D " + "X" * 3000})]
    ctxs_long = _collect_graph_contexts(
        "问", FakeManual(), FakeComm(), max_ctx=12, min_path_slots=6,
        entity_chunk_map={}, chunk_texts=[], dense_retriever=long_only)
    long_found = [c for c in ctxs_long if str(c).startswith("DENSE-CHUNK-D")]
    long_ok = any(len(c) <= 450 for c in long_found)
    print("超长 chunk 截断生效:", long_ok, "(len=%d)" % (len(long_found[0]) if long_found else -1))
    assert long_ok, "超长 chunk 应被截断到 400 字"
    print("PASS")

if __name__ == "__main__":
    main()
