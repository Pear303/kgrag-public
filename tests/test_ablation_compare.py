import importlib.util
import json
import os
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SPEC_PATH = os.path.join(ROOT, ".sisyphus", "gen_ablation_compare.py")
SPEC = importlib.util.spec_from_file_location("gen_ablation_compare", SPEC_PATH)
cmp_mod = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(cmp_mod)


def _meta(round_name, fa=0.8, ar=0.7, cp=0.6, cr=0.5):
    return {
        "round": round_name,
        "scores": {"graph_rag": {
            "faithfulness": fa, "answer_relevancy": ar,
            "context_precision": cp, "context_recall": cr}},
        "ablation_config": {},
        "test_set_breakdown": {"single-hop": 4, "multi-hop": 10,
                               "out-of-context": 5},
    }


def _results(round_dir):
    with open(os.path.join(round_dir, "graph_rag_results.json"), "w",
              encoding="utf-8") as f:
        json.dump({"per_question": []}, f)


def _make_group(td, prefix, base_fa, tags_fa):
    """构造一组轮次目录。tags_fa: {tag: fa}"""
    names = [f"{prefix}_R00g"] + [f"{prefix}_{t}" for t in tags_fa]
    for i, name in enumerate(names):
        d = os.path.join(td, name)
        os.makedirs(d)
        fa = base_fa if i == 0 else tags_fa[names[i].rsplit("_", 1)[-1]]
        with open(os.path.join(d, "meta.json"), "w", encoding="utf-8") as f:
            json.dump(_meta(name, fa=fa), f)
        _results(d)


def test_classify_stable_significant():
    """两组 FA 均回落 >0.03 → 稳定显著。"""
    with tempfile.TemporaryDirectory() as td:
        _make_group(td, "round_ablation", 0.90, {"A1": 0.80})
        _make_group(td, "round_ablation2", 0.92, {"A1": 0.78})
        # build_groups 会同时读两组（glob 全目录），需分别传入
        s1 = cmp_mod.screen.build_rows(
            td, prefix="round_ablation", base_name="round_ablation_R00g")
        s2 = cmp_mod.screen.build_rows(
            td, prefix="round_ablation2", base_name="round_ablation2_R00g")
        base1 = next(r for r in s1 if r["round"] == "round_ablation_R00g")
        base2 = next(r for r in s2 if r["round"] == "round_ablation2_R00g")
        a1 = next(r for r in s1 if r["round"] == "round_ablation_A1")
        a2 = next(r for r in s2 if r["round"] == "round_ablation2_A1")
        a1["_base"], a2["_base"] = base1, base2
        cells = cmp_mod.classify("A1", {"s1": {"A1": a1}, "s2": {"A1": a2}})
        fa_cell = next(c for c in cells if c[0] == "faithfulness")
        assert fa_cell[3] == "稳定显著"


def test_classify_unstable_mixed_sign():
    """两组 FA 符号相反 → 子集不稳定。"""
    b1 = {"round": "round_ablation_R00g", "metrics": {"faithfulness": 0.90}}
    a1 = {"round": "round_ablation_A1", "metrics": {"faithfulness": 0.92}}
    a2 = {"round": "round_ablation2_A1", "metrics": {"faithfulness": 0.80}}
    b2 = {"round": "round_ablation2_R00g", "metrics": {"faithfulness": 0.90}}
    a1["_base"], a2["_base"] = b1, b2
    cells = cmp_mod.classify("A1", {"s1": {"A1": a1}, "s2": {"A1": a2}})
    fa_cell = next(c for c in cells if c[0] == "faithfulness")
    assert fa_cell[3] == "子集不稳定"


def test_classify_neither_significant():
    """两组均 ≤0.03 → 均不显著。"""
    b1 = {"round": "round_ablation_R00g", "metrics": {"faithfulness": 0.90}}
    b2 = {"round": "round_ablation2_R00g", "metrics": {"faithfulness": 0.90}}
    a1 = {"round": "round_ablation_A2", "metrics": {"faithfulness": 0.89}}
    a2 = {"round": "round_ablation2_A2", "metrics": {"faithfulness": 0.895}}
    a1["_base"], a2["_base"] = b1, b2
    cells = cmp_mod.classify("A2", {"s1": {"A2": a1}, "s2": {"A2": a2}})
    fa_cell = next(c for c in cells if c[0] == "faithfulness")
    assert fa_cell[3] == "均不显著"
