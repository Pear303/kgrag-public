"""提示词校准器自测（零 API，纯逻辑与桩注入）。

覆盖：compose 字节守恒门 / 槽位白名单 / 变更单校验器 / 预算采样 /
冻结件复用与损坏拒绝（D）/ run_calibration 桩注入 / runner 双旗互斥（C1）。
"""
import json
import os

import pytest

from backend.extractor import (
    DEFAULT_EXTRACTION_FLEX,
    EXTRACTION_PROMPT_V3,
    FLEX_SLOT_NAMES,
    compose_extraction_prompt,
)
from backend import prompt_calibrator as pc
from backend.prompt_calibrator import (
    validate_changes,
    estimate_tokens,
    select_calibration_text,
    run_calibration,
    load_calibration,
    get_adapted_prompt,
    sha8,
)


# ---------------- compose：字节守恒与槽位 ----------------

def test_compose_default_is_exported_v3():
    """默认槽合成结果必须就是导出的 EXTRACTION_PROMPT_V3。"""
    assert compose_extraction_prompt() == EXTRACTION_PROMPT_V3


def test_compose_default_no_slot_tokens_and_has_context():
    assert "⟦" not in EXTRACTION_PROMPT_V3 and "⟧" not in EXTRACTION_PROMPT_V3
    assert "{context}" in EXTRACTION_PROMPT_V3
    assert "只允许以下六种关系" in EXTRACTION_PROMPT_V3, "固定段：六类白名单"
    assert "【负向约束】" in EXTRACTION_PROMPT_V3, "固定段：负向约束"
    assert "禁止臆造" in EXTRACTION_PROMPT_V3, "固定段：禁止臆造"


def test_compose_default_slots_present_verbatim():
    """每个默认槽值都原样出现在合成结果里（守恒的结构面）。"""
    for name in FLEX_SLOT_NAMES:
        assert DEFAULT_EXTRACTION_FLEX[name] in EXTRACTION_PROMPT_V3, name


def test_compose_override_changes_only_that_region():
    """改 include_ratio_cap 只动那一处（1/3 在 V3 全文唯一）。"""
    composed = compose_extraction_prompt({"include_ratio_cap": "1/4"})
    assert composed == EXTRACTION_PROMPT_V3.replace("1/3", "1/4")
    assert "1/3" not in composed


def test_compose_unknown_slot_raises():
    with pytest.raises(ValueError, match="未知可变槽"):
        compose_extraction_prompt({"nonexistent_slot": "x"})


def test_compose_none_value_falls_back_to_default():
    composed = compose_extraction_prompt({"domain_intro": None})
    assert composed == EXTRACTION_PROMPT_V3


# ---------------- validate_changes ----------------

def _ok_change(slot="domain_intro", new="伺服驱动器应用指南"):
    return {"slot": slot, "new": new, "reason": "手册标题为伺服驱动器应用指南"}


def test_validate_changes_ok_single_and_empty():
    assert validate_changes([_ok_change()]) == {"domain_intro": "伺服驱动器应用指南"}
    assert validate_changes([]) == {}


def test_validate_changes_unknown_slot_rejected():
    with pytest.raises(ValueError, match="未知可变槽"):
        validate_changes([{"slot": "fixed_section", "new": "x", "reason": "r"}])


def test_validate_changes_duplicate_slot_rejected():
    with pytest.raises(ValueError, match="重复"):
        validate_changes([_ok_change(), _ok_change()])


def test_validate_changes_empty_new_rejected():
    with pytest.raises(ValueError, match="非空"):
        validate_changes([_ok_change(new="   ")])


def test_validate_changes_missing_reason_rejected():
    bad = {"slot": "domain_intro", "new": "伺服驱动器应用指南"}
    with pytest.raises(ValueError, match="reason"):
        validate_changes([bad])


def test_validate_changes_slot_marker_in_new_rejected():
    with pytest.raises(ValueError, match="⟦"):
        validate_changes([_ok_change(new="伺服⟦domain_intro⟧")])


def test_validate_changes_ratio_format():
    assert validate_changes([_ok_change(slot="include_ratio_cap", new="1/4")])
    with pytest.raises(ValueError, match="include_ratio_cap"):
        validate_changes([_ok_change(slot="include_ratio_cap", new="三分之一")])


def test_validate_changes_code_samples_structure():
    good = [{"slot": "code_samples",
             "new": "   - 报警号：Er.100 等；\n   - 参数号：Pn001 等；\n   - 端子：CN1 等；",
             "reason": "汇川手册用 Er./Pn. 前缀"}]
    assert validate_changes(good)
    bad_flat = [{"slot": "code_samples",
                 "new": "报警号 Er.100 与参数 Pn001 都要保留",
                 "reason": "r"}]
    with pytest.raises(ValueError, match="行式清单"):
        validate_changes(bad_flat)


def test_validate_changes_oversize_rejected():
    with pytest.raises(ValueError, match="超长"):
        validate_changes([_ok_change(new="伺" * 300)])


# ---------------- 预算与采样 ----------------

def test_estimate_tokens_is_conservative_upper_bound():
    assert estimate_tokens("abc中文字") == 6


def test_select_full_within_budget():
    text = "故障现象与解决方法" * 10
    out = select_calibration_text(text, budget=1000)
    assert out["mode"] == "full"
    assert out["text"] == text
    assert out["info"]["n_windows"] == 1


def test_select_sampled_respects_budget_and_keeps_head_tail():
    text = "".join(f"[段{i}内容]" for i in range(30000))  # ~ 300k 字
    budget = 100_000
    out = select_calibration_text(text, budget=budget)
    assert out["mode"] == "sampled"
    assert estimate_tokens(out["text"]) <= budget
    assert out["text"].startswith(text[:10]), "采样必须覆盖开头"
    assert out["text"].endswith(text[-10:]), "采样必须覆盖结尾"
    assert out["info"]["n_windows"] >= 2


# ---------------- 冻结件（D 纪律）----------------

@pytest.fixture()
def calib_env(tmp_path, monkeypatch):
    """把校准目录指到 tmp，并提供一本小手册。"""
    monkeypatch.setattr(pc, "CALIB_DIR", str(tmp_path / "calib"))
    manual = tmp_path / "test_manual.md"
    manual.write_text("# 测试手册\n" + "轴承温度过高时检查润滑油位。\n" * 50, encoding="utf-8")
    return str(manual)


def _stub_llm(changes):
    def llm_fn(prompt, model, max_tokens):
        return json.dumps({"changes": changes}, ensure_ascii=False)
    return llm_fn


def test_run_calibration_dry_run_no_api(calib_env):
    rec = run_calibration(calib_env, dry_run=True)
    assert rec["dry_run"] is True
    assert rec["mode"] == "full"
    assert rec["manual_tokens_est"] > 0
    assert load_calibration(calib_env, verify=False) is None, "dry-run 不落盘"


def test_run_calibration_then_reuse(calib_env):
    calls = []
    def llm_fn(prompt, model, max_tokens):
        calls.append(prompt)
        return json.dumps({"changes": [_ok_change()]}, ensure_ascii=False)
    rec = run_calibration(calib_env, llm_fn=llm_fn)
    assert rec["reused"] is False
    assert len(calls) == 1
    assert "伺服驱动器应用指南" in rec["adapted_prompt"]
    # 落盘完整性
    frozen = load_calibration(calib_env, verify=True)
    assert frozen is not None
    assert frozen["adapted_prompt_sha8"] == sha8(rec["adapted_prompt"])
    # D：第二次直接复用，不再调 API
    rec2 = run_calibration(calib_env, llm_fn=llm_fn)
    assert rec2["reused"] is True
    assert len(calls) == 1


def test_load_calibration_detects_tampered_prompt_file(calib_env):
    run_calibration(calib_env, llm_fn=_stub_llm([_ok_change()]))
    ptxt = pc.adapted_prompt_path(calib_env)
    with open(ptxt, "a", encoding="utf-8") as f:
        f.write("\n被人手改的一行")
    with pytest.raises(RuntimeError, match="sha8 不符"):
        load_calibration(calib_env, verify=True)


def test_get_adapted_prompt_requires_frozen(calib_env):
    with pytest.raises(RuntimeError, match="没有冻结校准件"):
        get_adapted_prompt(calib_env)


def test_get_adapted_prompt_after_calibration(calib_env):
    run_calibration(calib_env, llm_fn=_stub_llm([_ok_change()]))
    prompt = get_adapted_prompt(calib_env)
    assert "伺服驱动器应用指南" in prompt
    assert "{context}" in prompt
    assert prompt == compose_extraction_prompt({"domain_intro": "伺服驱动器应用指南"})


def test_run_calibration_rejects_bad_changes(calib_env):
    def bad_llm(prompt, model, max_tokens):
        return json.dumps({"changes": [{"slot": "fixed_section", "new": "x", "reason": "r"}]},
                          ensure_ascii=False)
    with pytest.raises(ValueError, match="未知可变槽"):
        run_calibration(calib_env, llm_fn=bad_llm)
    assert load_calibration(calib_env, verify=False) is None, "坏变更单不得落盘"


# ---------------- P0.3 抗性修复：重试 + 原始输出落盘 ----------------

def test_run_calibration_retries_then_succeeds(calib_env, monkeypatch):
    """第一次校验失败、第二次成功 ⇒ 成功落盘，且记录 n_attempts=2。"""
    monkeypatch.setattr(pc, "CALIBRATOR_MAX_ATTEMPTS", 3)
    calls = []

    def flaky(prompt, model, max_tokens):
        calls.append(1)
        if len(calls) == 1:
            return json.dumps({"changes": [{"slot": "code_samples",
                                            "new": "   - 报警号：E08 等；\n   - 参数号：H00-14 等；\n   - 端子：CN1 等；",
                                            "reason": ""}]}, ensure_ascii=False)
        return json.dumps({"changes": [_ok_change()]}, ensure_ascii=False)

    rec = run_calibration(calib_env, llm_fn=flaky)
    assert len(calls) == 2
    assert rec["n_attempts"] == 2
    assert rec["attempts"][0]["outcome"] == "rejected_by_validator"
    assert rec["attempts"][1]["outcome"] == "ok"
    assert load_calibration(calib_env, verify=True) is not None
    # 被拒收那一次的原始输出必须落盘（P0.3 修复点）
    raw_path = pc.raw_attempts_path(calib_env)
    assert os.path.exists(raw_path)
    with open(raw_path, encoding="utf-8") as f:
        lines = [json.loads(x) for x in f if x.strip()]
    assert len(lines) == 1
    assert "code_samples" in lines[0]["raw_output"]
    assert "reason" in lines[0]["error"]


def test_run_calibration_exhausts_attempts_and_logs_every_raw(calib_env, monkeypatch):
    """三次全失败 ⇒ 抛错、不落盘，但三次原始输出全部留档。"""
    monkeypatch.setattr(pc, "CALIBRATOR_MAX_ATTEMPTS", 3)
    calls = []

    def always_bad(prompt, model, max_tokens):
        calls.append(1)
        return json.dumps({"changes": [{"slot": "fixed_section", "new": "x", "reason": "r"}]},
                          ensure_ascii=False)

    with pytest.raises(ValueError, match="未知可变槽"):
        run_calibration(calib_env, llm_fn=always_bad)
    assert len(calls) == 3, "应按上限重试 3 次"
    assert load_calibration(calib_env, verify=False) is None
    raw_path = pc.raw_attempts_path(calib_env)
    with open(raw_path, encoding="utf-8") as f:
        lines = [json.loads(x) for x in f if x.strip()]
    assert len(lines) == 3, "每次被拒的原始输出都要留档"


def test_retry_does_not_trigger_when_first_attempt_ok(calib_env, monkeypatch):
    """首次即合法 ⇒ 只调一次，不产生 raw_attempts.jsonl。"""
    monkeypatch.setattr(pc, "CALIBRATOR_MAX_ATTEMPTS", 3)
    calls = []

    def ok(prompt, model, max_tokens):
        calls.append(1)
        return json.dumps({"changes": [_ok_change()]}, ensure_ascii=False)

    rec = run_calibration(calib_env, llm_fn=ok)
    assert len(calls) == 1
    assert rec["n_attempts"] == 1
    assert not os.path.exists(pc.raw_attempts_path(calib_env))


# ---------------- pipeline 接入（C1 互斥 / prompt 透传）----------------

def test_runner_rejects_calibrate_and_use_together():
    from argparse import Namespace
    from pipeline.runner import PipelineRunner
    args = Namespace(manual="x.md", calibrate_prompt=True, use_calibrated_prompt=True,
                     force_calibrate=False, dry_run=False)
    with pytest.raises(SystemExit, match="互斥"):
        PipelineRunner().run(args)


def test_stage3_passes_extraction_prompt(monkeypatch, tmp_path):
    """Stage 3 必须把适配版提示词透传给抽取函数。"""
    import pipeline.stages as stages
    from pipeline.context import PipelineContext
    captured = {}

    def fake_online(text, prompt=None, **kw):
        captured["prompt"] = prompt
        return [("主轴", "包含", "轴承")]

    monkeypatch.setattr("backend.extractor.extract_triples_online", fake_online)

    class FakeStore:
        def upsert_auto_triples(self, triples):
            pass

    ctx = PipelineContext()
    ctx.chunks = []
    ctx.store = FakeStore()

    class _C:
        text = "主轴包含轴承"
        chunk_id = 0
        source = "m.md"

    ctx.chunks = [_C()]
    stages.run_extract_stage(ctx, build_llm_graph=True, use_online=True,
                             extraction_prompt="适配版{context}")
    assert captured["prompt"] == "适配版{context}"
