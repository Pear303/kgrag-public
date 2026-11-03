import importlib.util
import json
import os
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SPEC_PATH = os.path.join(ROOT, ".sisyphus", "export_ablation_data.py")
SPEC = importlib.util.spec_from_file_location("export_ablation_data", SPEC_PATH)
exp = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(exp)


def _meta(round_name, fa=0.8, ablation=None):
    return {
        "round": round_name,
        "archived_at": "2026-08-12T00:00:00",
        "n_questions": 19,
        "test_set": "test_set_shukongdashi_19_ablation.json",
        "test_set_breakdown": {"single-hop": 4, "multi-hop": 10,
                               "out-of-context": 5},
        "graph_source": "v5 (2147)",
        "ablation_config": ablation or {},
        "retrieval_params": {"top_k": "12", "vector_weight": "0.7"},
        "scores": {"graph_rag": {
            "faithfulness": fa, "answer_relevancy": 0.7,
            "context_precision": 0.6, "context_recall": 0.5},
            "vector_rag": {}},
    }


def _results(round_dir, ooc_ok=5, ooc_total=5):
    per = [{"type": "out-of-context", "metrics": {"ooc_refusal": 1.0}}
           for _ in range(ooc_ok)]
    per += [{"type": "out-of-context", "metrics": {"ooc_refusal": 0.0}}
            for _ in range(ooc_total - ooc_ok)]
    with open(os.path.join(round_dir, "graph_rag_results.json"), "w",
              encoding="utf-8") as f:
        json.dump({"per_question": per}, f)


def test_export_all_rounds_including_full():
    with tempfile.TemporaryDirectory() as td:
        for name, fa in (("round_ablation_R00g", 0.8),
                         ("round_ablation_A1", 0.7),
                         ("round_ablation_A3_full", 0.6)):
            d = os.path.join(td, name)
            os.makedirs(d)
            with open(os.path.join(d, "meta.json"), "w", encoding="utf-8") as f:
                json.dump(_meta(name, fa=fa), f)
            _results(d)
        recs = exp.export(td)
        assert len(recs) == 3
        by_name = {r["round"]: r for r in recs}
        assert by_name["round_ablation_A1"]["scores"]["graph_rag"]["faithfulness"] == 0.7
        assert by_name["round_ablation_A1"]["ablation_config"] == {}
        assert by_name["round_ablation_A3_full"]["ooc_refusal"] == {"refused": 5, "total": 5}


def test_export_ooc_partial():
    with tempfile.TemporaryDirectory() as td:
        d = os.path.join(td, "round_ablation_B3")
        os.makedirs(d)
        with open(os.path.join(d, "meta.json"), "w", encoding="utf-8") as f:
            json.dump(_meta("round_ablation_B3"), f)
        _results(d, ooc_ok=2, ooc_total=5)
        recs = exp.export(td)
        assert recs[0]["ooc_refusal"] == {"refused": 2, "total": 5}


def test_export_missing_results():
    with tempfile.TemporaryDirectory() as td:
        d = os.path.join(td, "round_ablation_X")
        os.makedirs(d)
        with open(os.path.join(d, "meta.json"), "w", encoding="utf-8") as f:
            json.dump(_meta("round_ablation_X"), f)
        recs = exp.export(td)
        assert recs[0]["ooc_refusal"] is None
