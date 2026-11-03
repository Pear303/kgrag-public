import os
import importlib.util

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SPEC_PATH = os.path.join(ROOT, ".sisyphus", "build_ablation_subset.py")
SPEC = importlib.util.spec_from_file_location("build_ablation_subset", SPEC_PATH)
build_ablation_subset = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(build_ablation_subset)
build_subset = build_ablation_subset.build_subset


def _fake_items():
    items = []
    types = ["single-hop"] * 10 + ["multi-hop"] * 30 + ["out-of-context"] * 10
    for i, t in enumerate(types):
        items.append({"id": f"q{i:03d}", "type": t, "question": f"Q{i}",
                      "ground_truth": "g", "source_manual": "s"})
    return items


def test_subset_composition():
    out = build_subset(_fake_items(), seed=42)
    from collections import Counter
    c = Counter(x["type"] for x in out)
    assert c == {"single-hop": 4, "multi-hop": 10, "out-of-context": 5}
    assert len(out) == 19


def test_subset_reproducible():
    a = build_subset(_fake_items(), seed=42)
    b = build_subset(_fake_items(), seed=42)
    assert [x["id"] for x in a] == [x["id"] for x in b]


def test_subset_different_seed_different_order():
    a = build_subset(_fake_items(), seed=42)
    b = build_subset(_fake_items(), seed=7)
    assert [x["id"] for x in a] != [x["id"] for x in b]


def test_subset_exclude_disjoint():
    a = build_subset(_fake_items(), seed=42)
    a_ids = {x["id"] for x in a}
    b = build_subset(_fake_items(), seed=43, exclude_ids=a_ids)
    b_ids = {x["id"] for x in b}
    assert a_ids.isdisjoint(b_ids)
    from collections import Counter
    assert Counter(x["type"] for x in b) == {
        "single-hop": 4, "multi-hop": 10, "out-of-context": 5}


def test_subset_exclude_composition():
    """剩余样本恰好够抽（OOC 剩 5 题全取）。"""
    a = build_subset(_fake_items(), seed=42)
    a_ids = {x["id"] for x in a}
    b = build_subset(_fake_items(), seed=43, exclude_ids=a_ids)
    # 第一组 OOC 5/10，第二组 OOC 必为剩余 5 题
    from collections import Counter
    ca = Counter(x["type"] for x in a)
    cb = Counter(x["type"] for x in b)
    assert ca["out-of-context"] + cb["out-of-context"] == 10


def test_subset_exclude_insufficient_raises():
    """排除过多后某类型不足应抛 ValueError。"""
    items = _fake_items()
    all_ids = {x["id"] for x in items}
    import pytest
    with pytest.raises(ValueError):
        build_subset(items, seed=43, exclude_ids=all_ids)
