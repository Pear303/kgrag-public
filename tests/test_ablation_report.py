import importlib.util
import json
import os
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SPEC_PATH = os.path.join(ROOT, ".sisyphus", "gen_ablation_screening.py")
SPEC = importlib.util.spec_from_file_location("gen_ablation_screening", SPEC_PATH)
gen = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(gen)


def _meta(tag, fa=0.8, ar=0.7, cp=0.6, cr=0.5, ablation=None):
    return {
        "round": f"round_ablation_{tag}",
        "scores": {"graph_rag": {
            "faithfulness": fa, "answer_relevancy": ar,
            "context_precision": cp, "context_recall": cr}},
        "ablation_config": ablation or {},
        "test_set_breakdown": {"single-hop": 4, "multi-hop": 10,
                               "out-of-context": 5},
    }


def test_parse_ooc_refusal():
    md = ("## OOC\n- KG-RAG 正确拒答 5/5 题；Vector 正确拒答 1/5 题。\n")
    assert gen.parse_ooc_refusal(md) == (5, 5)


def test_parse_ooc_refusal_missing():
    assert gen.parse_ooc_refusal("no content") is None


def test_parse_ooc_from_results(tmp_path):
    res = {
        "per_question": [
            {"type": "out-of-context", "metrics": {"ooc_refusal": 1.0}},
            {"type": "out-of-context", "metrics": {"ooc_refusal": 0.0}},
            {"type": "multi-hop", "metrics": {"ooc_refusal": 1.0}},
        ]
    }
    p = os.path.join(tmp_path, "graph_rag_results.json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump(res, f)
    assert gen.parse_ooc_from_results(p) == (1, 2)


def test_parse_ooc_from_results_missing(tmp_path):
    assert gen.parse_ooc_from_results(
        os.path.join(tmp_path, "nope.json")) is None


def test_build_rows_and_render():
    with tempfile.TemporaryDirectory() as td:
        base = gen.ROUNDS_DIR
        gen.ROUNDS_DIR = td
        try:
            for tag, meta in (("R00g", _meta("R00g")),
                              ("A1", _meta("A1", fa=0.70,
                                           ablation={"ablate_4gram": True}))):
                d = os.path.join(td, f"round_ablation_{tag}")
                os.makedirs(d)
                with open(os.path.join(d, "meta.json"), "w",
                          encoding="utf-8") as f:
                    json.dump(meta, f)
            rows = gen.build_rows(td)
            assert len(rows) == 2
            r0, r1 = rows
            assert r1["d_faithfulness"] == -0.1
            assert r1["significant"] is True  # 回落 0.1 > 0.03
            md = gen.render_md(rows)
            assert "round_ablation_R00g" in md
            assert "round_ablation_A1" in md
        finally:
            gen.ROUNDS_DIR = base


def test_significant_boundary_rounding():
    """差值恰 0.0300（浮点近似 0.0300...027）不因未 round 被误标显著。

    0.80 - 0.77 = 0.030000000000000027 > 0.03，展示层 round 后为 +0.0300；
    显著判定必须与展示口径一致（round(4) 后比较），否则自相矛盾。
    """
    with tempfile.TemporaryDirectory() as td:
        base = gen.ROUNDS_DIR
        gen.ROUNDS_DIR = td
        try:
            for tag, meta in (("R00g", _meta("R00g", fa=0.80)),
                              ("A1", _meta("A1", fa=0.77,
                                           ablation={"ablate_4gram": True}))):
                d = os.path.join(td, f"round_ablation_{tag}")
                os.makedirs(d)
                with open(os.path.join(d, "meta.json"), "w",
                          encoding="utf-8") as f:
                    json.dump(meta, f)
            rows = gen.build_rows(td)
            a1 = rows[1]
            assert a1["round"] == "round_ablation_A1"
            assert a1["d_faithfulness"] == -0.03
            assert a1["significant"] is False
        finally:
            gen.ROUNDS_DIR = base
