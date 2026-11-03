import numpy as np
from types import SimpleNamespace
from unittest.mock import patch

from config import ABLATE
from run_pump_fault import build_meta, select_graph_engines
from backend.engines import (
    _match_entities_smart,
    _norm_fragment_token,
    _fragment_window,
    ManualGraphQueryEngine,
    CommunityQueryEngine,
)
from backend.evaluator import _should_refuse_ooc


# ---------- A1: 4-gram 概念短语重叠召回 ----------

def _no_semantic_embed(x):
    """返回低相似向量，避免触发语义命中与保底召回。"""
    return [0.0]


def test_A1_off_overlap_recall_removed(monkeypatch):
    monkeypatch.setattr(ABLATE, "ablate_4gram", True)
    names = ["回参考点不能进行", "主轴"]
    embs = [None, None]
    matched = _match_entities_smart(
        "回参考点操作不能进行时", names, embs, _no_semantic_embed, top_k=10,
        threshold=0.6)
    assert "回参考点不能进行" not in matched


def test_A1_on_overlap_recall_works(monkeypatch):
    monkeypatch.setattr(ABLATE, "ablate_4gram", False)
    names = ["回参考点不能进行", "主轴"]
    embs = [None, None]
    matched = _match_entities_smart(
        "回参考点操作不能进行时", names, embs, _no_semantic_embed, top_k=10,
        threshold=0.6)
    assert "回参考点不能进行" in matched


# ---------- A3: 余弦嵌入匹配 ----------

def test_A3_off_semantic_recall_removed(monkeypatch):
    monkeypatch.setattr(ABLATE, "ablate_embed_match", True)
    names = ["entityx"]
    embs = [np.array([1.0, 0.0], dtype=np.float32)]
    def embed(x):
        return np.array([1.0, 0.0], dtype=np.float32)
    matched = _match_entities_smart(
        "完全无关的中文问题句", names, embs, embed, top_k=10, threshold=0.6)
    assert matched == []


def test_A3_on_semantic_recall_works(monkeypatch):
    monkeypatch.setattr(ABLATE, "ablate_embed_match", False)
    names = ["entityx"]
    embs = [np.array([1.0, 0.0], dtype=np.float32)]
    def embed(x):
        return np.array([1.0, 0.0], dtype=np.float32)
    matched = _match_entities_smart(
        "完全无关的中文问题句", names, embs, embed, top_k=10, threshold=0.6)
    assert "entityx" in matched


# ---------- A2: 实体二次打分 / 代码 token 优先 ----------

def _make_manual_engine(match_result):
    """绕过 __init__（不连 Neo4j）构造 ManualGraphQueryEngine。"""
    eng = ManualGraphQueryEngine.__new__(ManualGraphQueryEngine)
    eng.graph_store = SimpleNamespace(_driver=object())
    eng.source_manual = None
    eng.top_k = 12
    eng._match_entities = lambda q: list(match_result)
    return eng


def test_A2_off_code_token_priority_skipped(monkeypatch):
    monkeypatch.setattr(ABLATE, "ablate_rescore", True)
    eng = _make_manual_engine(["普通实体", "Q12的CE极在线短路"])
    calls = []
    monkeypatch.setattr(
        "backend.engines._code_token_priority_entities",
        lambda q, m: (calls.append(1), m)[1])
    monkeypatch.setattr("backend.engines._expand_paths_per_depth",
                        lambda *a, **k: [])
    monkeypatch.setattr("backend.engines._rank_paths", lambda p, q: p)
    monkeypatch.setattr("backend.engines._interleave_paths",
                        lambda p, n_priority=0: [x for sub in p for x in sub])
    eng.query("Q12元件在线短路问题")
    assert calls == []


def test_A2_on_code_token_priority_called(monkeypatch):
    monkeypatch.setattr(ABLATE, "ablate_rescore", False)
    eng = _make_manual_engine(["普通实体", "Q12的CE极在线短路"])
    calls = []
    monkeypatch.setattr(
        "backend.engines._code_token_priority_entities",
        lambda q, m: (calls.append(1), m)[1])
    monkeypatch.setattr("backend.engines._expand_paths_per_depth",
                        lambda *a, **k: [])
    monkeypatch.setattr("backend.engines._rank_paths", lambda p, q: p)
    monkeypatch.setattr("backend.engines._interleave_paths",
                        lambda p, n_priority=0: [x for sub in p for x in sub])
    eng.query("Q12元件在线短路问题")
    assert calls == [1]


# ---------- B1: 短语归一化 ----------

def test_B1_off_norm_becomes_identity(monkeypatch):
    monkeypatch.setattr(ABLATE, "ablate_frag_norm", True)
    assert _norm_fragment_token("端子 ３６") == "端子 ３６"
    assert _norm_fragment_token("abc") == "abc"


def test_B1_on_norm_normalizes(monkeypatch):
    monkeypatch.setattr(ABLATE, "ablate_frag_norm", False)
    assert _norm_fragment_token("端子 ３６") == "端子36"
    assert _norm_fragment_token("abc") == "ABC"


# ---------- B2: 扩窗 12000 → 1200 ----------

def test_B2_off_window_caps_at_1200(monkeypatch):
    monkeypatch.setattr(ABLATE, "ablate_window", True)
    out = _fragment_window("x" * 5000, [])
    assert len(out) == 1200


def test_B2_on_window_caps_at_12000(monkeypatch):
    monkeypatch.setattr(ABLATE, "ablate_window", False)
    out = _fragment_window("x" * 5000, [])
    assert len(out) == 5000


# ---------- B4: 社区摘要注入 ----------

def _make_community_engine(comm):
    eng = CommunityQueryEngine.__new__(CommunityQueryEngine)
    eng.communities = [comm]
    eng._match_communities = lambda q: eng.communities
    return eng


def test_B4_off_summary_injected(monkeypatch):
    monkeypatch.setattr(ABLATE, "ablate_chunk_summary", False)
    eng = _make_community_engine({
        "id": 0, "summary": "社区核心摘要", "report": "社区详细报告",
        "hard_facts": ["ALM100=报警"], "raw_context": ["原文片段"],
    })
    _, ctxs = eng.query("测试问题", chunk_fallback=False)
    assert any("[社区摘要]" in c for c in ctxs)
    assert any("[社区报告]" in c for c in ctxs)


def test_B4_on_summary_stripped(monkeypatch):
    monkeypatch.setattr(ABLATE, "ablate_chunk_summary", True)
    eng = _make_community_engine({
        "id": 0, "summary": "社区核心摘要", "report": "社区详细报告",
        "hard_facts": ["ALM100=报警"], "raw_context": ["原文片段"],
    })
    _, ctxs = eng.query("测试问题", chunk_fallback=False)
    assert not any("[社区摘要]" in c for c in ctxs)
    assert not any("[社区报告]" in c for c in ctxs)
    assert any("[硬事实]" in c for c in ctxs)
    assert any("[相关片段]" in c for c in ctxs)


# ---------- B3: OOC 对象一致性检查消融 ----------

_OOC_RELEVANCE = 0.05   # 低于 _OOC_RELEVANCE_THRESHOLD（触发相关度门）
_OOC_RELEVANCE_HIGH = 0.95  # 高于阈值（不触发相关度门）


def test_B3_on_object_mismatch_refuses(monkeypatch):
    monkeypatch.setattr(ABLATE, "ablate_ooc", False)
    with patch("backend.evaluator._ooc_object_mismatch", return_value=True):
        refuse, mismatch = _should_refuse_ooc(
            "机器人故障", ["图谱路径A"], _OOC_RELEVANCE_HIGH)
    assert refuse is True and mismatch is True


def test_B3_off_object_mismatch_ignored(monkeypatch):
    monkeypatch.setattr(ABLATE, "ablate_ooc", True)
    with patch("backend.evaluator._ooc_object_mismatch", return_value=True) as spy:
        refuse, mismatch = _should_refuse_ooc(
            "机器人故障", ["图谱路径A"], _OOC_RELEVANCE_HIGH)
    assert refuse is False and mismatch is False
    spy.assert_not_called()


def test_B3_both_gates_still_refuse_on_low_relevance(monkeypatch):
    monkeypatch.setattr(ABLATE, "ablate_ooc", True)
    refuse, mismatch = _should_refuse_ooc(
        "机器人故障", ["图谱路径A"], _OOC_RELEVANCE)
    assert refuse is True and mismatch is False


# ---------- C1: 社区宏观开关 + meta 消融快照 ----------

def _items(n=3):
    return [{"question": f"q{i}", "ground_truth": "g", "type": "single-hop",
             "source_manual": "s", "id": f"auto_00{i}"} for i in range(1, n + 1)]


def test_meta_includes_ablation_config(monkeypatch):
    monkeypatch.setattr(ABLATE, "ablate_4gram", True)
    monkeypatch.setattr(ABLATE, "ablate_window", True)
    gres = {"faithfulness": 0.8, "answer_relevancy": 0.7,
            "context_precision": 0.6, "context_recall": 0.5}
    meta = build_meta(_items(), gres, None, 300, 6, 6, "both")
    assert meta["ablation_config"] == {
        "ablate_4gram": True, "ablate_window": True}
    assert meta["retrieval_params"]["top_k"] == "12"


def test_meta_ablation_empty_when_full(monkeypatch):
    for f in ("ablate_4gram", "ablate_rescore", "ablate_embed_match",
              "ablate_frag_norm", "ablate_window", "ablate_ooc",
              "ablate_chunk_summary", "ablate_community"):
        monkeypatch.setattr(ABLATE, f, False)
    gres = {"faithfulness": 0.8, "answer_relevancy": 0.7,
            "context_precision": 0.6, "context_recall": 0.5}
    meta = build_meta(_items(), gres, None, 300, 6, 6, "both")
    assert meta["ablation_config"] == {}


def test_C1_community_engine_excluded(monkeypatch):
    """C1 开关：ABLATE_COMMUNITY=1 时 select_graph_engines 不含 community 键。"""
    from backend.engines import ManualGraphQueryEngine
    manual_eng = ManualGraphQueryEngine.__new__(ManualGraphQueryEngine)
    comm_eng = object()

    monkeypatch.setattr(ABLATE, "ablate_community", True)
    engines = select_graph_engines(manual_eng, comm_eng)
    assert set(engines.keys()) == {"manual"}
    assert engines["manual"] is manual_eng

    monkeypatch.setattr(ABLATE, "ablate_community", False)
    engines = select_graph_engines(manual_eng, comm_eng)
    assert set(engines.keys()) == {"manual", "community"}
    assert engines["community"] is comm_eng

    # 无社区引擎时仅 manual（不论开关）
    monkeypatch.setattr(ABLATE, "ablate_community", False)
    assert select_graph_engines(manual_eng, None) == {"manual": manual_eng}
