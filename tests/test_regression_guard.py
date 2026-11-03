"""回归防护测试（9.10 全缺陷防回归，独立于功能测试）。

背景：评测链路在多次 review 中暴露多类"小改动导致结果失真"的缺陷——
CoT 解析赋值反转（answer 存成推理过程）、max_ctx 挤出图谱路径、保底召回
无下限、实体/社区阈值过低放行 OOC 噪声、OOC 指标覆盖丢失等。本文件把
**已修复缺陷的关键不变量固化为严格断言**，任何未来改动若破坏不变量将
立即失败。配套规则文档见 reports/regression_guard_rules.md。

设计原则：不依赖网络/Ollama/Neo4j；纯函数与 fake 对象；每个测试对应
一个已修复缺陷（编号见 docstring）。
"""
import pytest

from backend.evaluator import (
    RAGASEvaluator,
    _split_cot_output,
    _truncate_contexts,
    _patch_context_precision_parallel,
)
from backend.engines import (
    ENTITY_MATCH_THRESHOLD,
    COMMUNITY_FALLBACK_THRESHOLD,
    _match_entities_smart,
)


# ════════════════════════════════════════════════════════════════════
# 一、CoT 解析方向契约（9.10-③：G 侧 answer 曾存成【推理过程】）
# ════════════════════════════════════════════════════════════════════

def test_split_cot_output_contract_answer_first():
    """契约：`_split_cot_output` 恒返回 (answer, reasoning)，顺序不可交换。

    回归防护：曾因调用处写 `reasoning, answer = _split_cot_output(...)` 把
    两者赋值反了，G 侧 answer 实际存的是推理过程，RAGAS 给错误文本评分。
    """
    raw = "【推理过程】推理内容ABC\n【最终答案】\n答案内容XYZ"
    answer, reasoning = _split_cot_output(raw)
    assert answer == "答案内容XYZ", "返回值[0] 必须是【最终答案】部分"
    assert reasoning == "【推理过程】推理内容ABC", "返回值[1] 必须是推理过程"


def test_generate_answer_with_cot_preserves_contract():
    """回归防护：`_generate_answer_with_cot` 返回的 answer 必须是最终答案。

    这是 9.10-③ bug 的直接回归测试：若未来有人改回错误的解包顺序，
    answer 会变成推理过程（含"推理内容"字样）→ 断言失败。
    """
    ev = RAGASEvaluator.__new__(RAGASEvaluator)

    class FakeGen:
        def __call__(self, prompt):
            return ("【推理过程】推理内容ABC\n"
                    "【最终答案】\n【故障现象】轴承温度过高\n【可能原因】润滑不良")

    ev._generate_answer = FakeGen()
    answer, cot = ev._generate_answer_with_cot("ctx", "q")
    assert "故障现象" in answer, "answer 必须是【最终答案】部分（结构化诊断结论）"
    assert "推理内容" not in answer, "answer 不得混入推理过程"
    assert "推理内容" in cot, "cot_reasoning 必须是推理过程"
    assert "故障现象" not in cot, "cot_reasoning 不得混入最终答案"


def test_vector_lambda_uses_answer_slot():
    """回归防护：V 侧 answer_generator 必须取 `_split_cot_output(...)[0]`（answer）。

    run_pump_fault.py 注入 `lambda prompt: _split_cot_output(gen(prompt))[0]`。
    若有人误改为 `[1]`，V 侧 answer 将变成推理过程 → 两侧形态再次不对称。
    """
    raw = "【推理过程】推理内容ABC\n【最终答案】\n答案内容XYZ"
    answer = _split_cot_output(raw)[0]
    assert answer == "答案内容XYZ"


def test_split_cot_no_marker_returns_raw_answer():
    """契约：无【最终答案】标记时返回 (raw, "")，不得丢弃内容或复制答案。"""
    answer, reasoning = _split_cot_output("只有一段答案")
    assert answer == "只有一段答案"
    assert reasoning == ""


# ════════════════════════════════════════════════════════════════════
# 二、评分 context 截断（9.10-④：max_ctx=8 曾把图谱路径全部挤出）
# ════════════════════════════════════════════════════════════════════

def test_truncate_default_max_ctx_at_least_12():
    """回归防护：默认 max_ctx 不得低于 12。

    8 个位置会被社区正文占满，auto_008/013 的 32/31 条图谱路径全被挤出
    （路径 0 条），G 的图谱能力在评分中体现不足（9.10-④）。
    """
    ctxs = [f"社区正文片段{i}" * 20 for i in range(15)]  # 15 个 60+ 字符
    result = _truncate_contexts(ctxs)
    assert len(result) >= 12, "max_ctx 回退 < 12 会挤出图谱路径（9.10-④）"


def test_truncate_keeps_graph_paths_when_budget_allows():
    """回归防护：社区正文 + 路径混合时，路径必须进入评分 context。

    模拟 auto_008 场景：8 个社区正文（各 ~400 字符）+ 5 条路径碎片。
    预算 24000 内 12 个位置应保留 8 社区 + 4 路径，路径 `--[` 必须出现。
    """
    comm = ["社区正文" * 50] * 8       # 8 × 400 字符
    paths = ["实体A--[包含]-->实体B"] * 5
    result = _truncate_contexts(comm + paths)
    assert len(result) == 12
    n_paths = sum(1 for c in result if "--[" in c)
    assert n_paths >= 4, f"图谱路径必须进入评分 context，当前仅 {n_paths} 条"


def test_truncate_budget_guard_still_enforced():
    """契约：total_budget 仍是硬上限（防未来有人去掉预算保护）。

    2026-08-11 扩窗后：max_chars=12000、total_budget=48000。
    片段先按 max_chars=12000 截断，故用 5×20000 字符片段触发预算
    （截断后 5×12000=60000 > 48000 → 只能保留 4 个）。
    """
    ctxs = ["X" * 20000] * 5
    result = _truncate_contexts(ctxs)
    assert len(result) == 4, "预算 48000 必须生效（5×12000 只能保留 4 个）"
    assert sum(len(c) for c in result) <= 48000


# ════════════════════════════════════════════════════════════════════
# 三、检索阈值（9.10：0.35/0.25 太低放行 OOC 噪声，已数据驱动上调）
# ════════════════════════════════════════════════════════════════════

def test_threshold_constants_above_decision_boundary():
    """回归防护：阈值常量不得回退到旧值。

    v3.1（9.10）：0.35 → 0.55（in-context top-3 ≥0.56，完全 OOC ≤0.53）。
    v4（9.11-3 图谱扩至 317 条后重标定）：0.55 → 0.60（in-context top-1 ≥0.69、
    top-3 ≥0.602；完全 OOC auto_017 top-1=0.56 会被 0.55 放行导致编造）。
    """
    assert ENTITY_MATCH_THRESHOLD >= 0.60, (
        f"实体阈值回退 < 0.60 会让 v4 图谱下 OOC 题（top-1 0.56）误命中"
        f"（当前 {ENTITY_MATCH_THRESHOLD}）")
    assert COMMUNITY_FALLBACK_THRESHOLD >= 0.5, (
        f"社区保底阈值回退 < 0.5 会让 OOC 社区（0.43~0.53）被塞回"
        f"（当前 {COMMUNITY_FALLBACK_THRESHOLD}）")


def _fixed_sim_embed(q_vec):
    """构造返回固定 query 向量的 embed，便于精确控制余弦相似度。"""
    def embed(text):
        return q_vec
    return embed


def test_semantic_threshold_filters_weak_matches():
    """回归防护：相似度 < 阈值（0.55）的弱相关实体不得被语义分支命中。

    场景：query=[1,0]，实体 [0.5,0.866]（余弦 0.5）→ 0.35 阈值下误命中
    （旧行为，OOC 噪声），0.55 下必须过滤。
    """
    names = ["弱相似实体"]
    embeds = [[0.5, 0.8660254]]  # 与 [1,0] 余弦 = 0.5
    embed = _fixed_sim_embed([1.0, 0.0])
    matched_035 = _match_entities_smart(
        "q", names, embeds, embed=embed, top_k=5, threshold=0.35)
    assert matched_035 == ["弱相似实体"], "0.35 旧阈值会命中弱相似（OOC 噪声源）"
    matched_055 = _match_entities_smart(
        "q", names, embeds, embed=embed, top_k=5, threshold=0.55)
    assert matched_055 == [], "0.55 必须过滤余弦 0.5 的弱相似实体"


def test_fallback_never_bypasses_threshold():
    """回归防护：保底分支不得无下限硬塞 top-3（宁缺毋滥）。

    旧行为（H4）在语义全空时无条件取 top-3，让 OOC 题被塞入无关实体
    → 答非所问反而得分。修复后保底同样卡 threshold。
    """
    names = ["实体A", "实体B", "实体C"]
    # 余弦分别 0.4 / 0.3 / 0.2，全部 < 0.5
    embeds = [[0.4, 0.9165151], [0.3, 0.9539392], [0.2, 0.9797959]]
    embed = _fixed_sim_embed([1.0, 0.0])
    matched = _match_entities_smart(
        "q", names, embeds, embed=embed, top_k=5, threshold=0.5)
    assert matched == [], "保底不得把低于阈值的实体硬塞回来"


def test_literal_match_still_wins_over_threshold():
    """契约：字面命中不受阈值影响（实体名直接出现在问题中即命中）。"""
    names = ["暖泵操作", "热油泵停泵后"]
    embeds = [[0.1, 0.995], [0.2, 0.98]]  # 语义相似度都很低
    embed = _fixed_sim_embed([1.0, 0.0])
    matched = _match_entities_smart(
        "暖泵操作如何进行", names, embeds, embed=embed, top_k=5, threshold=0.99)
    assert matched == ["暖泵操作"], "字面命中必须优先且不受高阈值影响"


# ════════════════════════════════════════════════════════════════════
# 四、OOC 拒答指标（9.10：消除 OOC 口径不对称的实质妥协）
# ════════════════════════════════════════════════════════════════════

class _FakeJudge:
    def __init__(self, content="YES"):
        self._content = content

    def invoke(self, prompt, **kwargs):
        class R:
            content = self._content
        return R()


def test_ooc_refusal_yes_prefix_tolerance():
    """回归防护：judge 输出 YES 前缀（含标点/解释）仍判 1.0。"""
    ev = RAGASEvaluator.__new__(RAGASEvaluator)
    ev.judge_llm = _FakeJudge("YES。该回答正确拒答。")
    assert ev._judge_ooc_refusal("q", "知识库中未找到相关信息。") == 1.0


def test_ooc_refusal_fabricated_answer_scores_zero():
    """回归防护：编造具体答案 → 0.0（即使 judge 前缀混乱也要靠内容判定）。"""
    ev = RAGASEvaluator.__new__(RAGASEvaluator)
    ev.judge_llm = _FakeJudge("NO")
    assert ev._judge_ooc_refusal(
        "q", "通常采用渗透检测和磁粉检测。") == 0.0


def test_ooc_metrics_only_for_ooc_type():
    """回归防护：非 OOC 题不得写入 ooc_refusal（防污染四指标聚合）。"""
    ev = RAGASEvaluator.__new__(RAGASEvaluator)
    ev.judge_llm = _FakeJudge("YES")
    assert ev._ooc_metrics_for({"type": "single-hop"}, "答案") == {}
    m = ev._ooc_metrics_for(
        {"type": "out-of-context", "question": "q"}, "未找到相关信息。")
    assert m == {"ooc_refusal": 1.0}


# ════════════════════════════════════════════════════════════════════
# 五、逐题指标写入（9.10：合并而非覆盖，防 ooc_refusal 被 RAGAS 覆盖）
# ════════════════════════════════════════════════════════════════════

def test_attach_metrics_merges_preserves_ooc_refusal():
    """回归防护：_attach_per_question_metrics 必须合并（保留 ooc_refusal）。

    曾用 `rec["metrics"] = row_metrics` 直接覆盖，会把答案生成阶段写入的
    ooc_refusal 抹掉 → OOC 拒答节全部显示 N/A。
    """
    import pandas as pd

    class M:
        name = "faithfulness"

    ev = RAGASEvaluator.__new__(RAGASEvaluator)
    batch = [{"id": "auto_016", "metrics": {"ooc_refusal": 1.0}}]
    df = pd.DataFrame({"faithfulness": [0.0]})
    ev._attach_per_question_metrics(batch, df, [M()])
    assert batch[0]["metrics"] == {"ooc_refusal": 1.0, "faithfulness": 0.0}, (
        "合并写入：ooc_refusal 必须保留，RAGAS 指标追加共存")


# ════════════════════════════════════════════════════════════════════
# 六、空 context 归一化（9.10：G 与 V 空检索两侧形态必须一致）
# ════════════════════════════════════════════════════════════════════

def test_empty_contexts_normalize_to_empty_string():
    """回归防护：空/空串 context 都必须归一化为 [""]（两侧评分口径一致）。

    G 空检索返回 [""]、V 空检索返回 [] → 评分前都经 _truncate_contexts，
    若某侧归一化不一致，OOC 题的 faithfulness 会再次不对称。
    """
    assert _truncate_contexts([]) == [""]
    assert _truncate_contexts([""]) == [""]
    assert _truncate_contexts(["", "有内容"]) == ["有内容"]


# ════════════════════════════════════════════════════════════════════
# 七、RAGAS 超时配置（KG-14：ragas 默认 180s 会误杀 Graph context_precision）
# ════════════════════════════════════════════════════════════════════

def test_eval_timeout_default_above_ragas_180():
    """回归防护：RAGASEvaluator 默认 eval_timeout 必须 > ragas 默认 180s。

    Graph 的 context_precision 每题 8 次判定串行可达 200s+，若 eval_timeout
    回退 ≤ 180 会被误判超时 → NaN 缺失（KG-14）。
    """
    from backend.evaluator import RAGASEvaluator as EV

    class _FakeJudge2:
        def invoke(self, prompt, **kwargs):
            return None

    class _FakeEmbed2:
        def embed_documents(self, texts):
            return [[0.1, 0.2]] * len(texts)

        def embed_query(self, text):
            return [0.1, 0.2]

    ev = EV(_FakeJudge2(), _FakeEmbed2())
    assert ev.eval_timeout > 180, (
        f"eval_timeout 必须 > 180（ragas 默认会误杀超时），当前 {ev.eval_timeout}")
    assert ev.eval_timeout == 600, "默认 600s"


# ════════════════════════════════════════════════════════════════════
# 八、context_precision 并行 patch 幂等（方案 B：防重复 patch / 回退失败）
# ════════════════════════════════════════════════════════════════════

def test_context_precision_patch_idempotent():
    """回归防护：_patch_context_precision_parallel 必须幂等（二次调用直接成功）。

    该 patch 是防御式（版本/符号检查 + 失败回退），若未来有人改成非幂等
    （每次重包一层），并发评分会叠加 Semaphore/嵌套 gather → 行为异常。
    """
    try:
        import ragas  # noqa: F401
    except ImportError:
        pytest.skip("ragas 不可用，跳过 patch 幂等验证")
    first = _patch_context_precision_parallel()
    second = _patch_context_precision_parallel()
    assert first is True, "首次 patch 应成功"
    assert second is True, "二次调用应直接返回成功（幂等）"


# ════════════════════════════════════════════════════════════════════
# 九、短实体白名单（9.11-3：规则5 曾误杀 叶轮/口环/盘车 等 2~3 字部件）
# ════════════════════════════════════════════════════════════════════

def test_short_entity_whitelist_keeps_domain_parts():
    """回归防护：领域短部件实体（2~3 字）不得被 _is_noise_entity 过滤。

    规则5（<=3字且无允许字符→噪音）本意滤截断碎片，但曾把叶轮/口环/轴颈/
    油泵/盘车/暖泵/轴承/转子 全部滤掉（v3 图谱 0 个 <=3 字实体，而手册中
    '叶轮' 出现 28 次）。命中 SHORT_ENTITY_WHITELIST 必须放行。
    """
    from backend.extractor import _is_noise_entity, SHORT_ENTITY_WHITELIST

    assert SHORT_ENTITY_WHITELIST, "白名单不得为空"
    for ent in ("叶轮", "口环", "轴颈", "油泵", "盘车", "暖泵", "轴承",
                "转子", "密封", "主轴", "联轴器", "平衡盘", "导叶", "蜗壳"):
        assert _is_noise_entity(ent) is False, (
            f"领域短部件 {ent!r} 不应被过滤（规则5 误杀回归）")


def test_short_entity_noise_still_filtered():
    """回归防护：非白名单的短截断碎片/低信息短词仍须过滤（去噪能力不退化）。"""
    from backend.extractor import _is_noise_entity

    for ent in ("泵反(?", "反", "问题", "故障", "123", "正常", "设备"):
        assert _is_noise_entity(ent) is True, (
            f"{ent!r} 应被识别为噪音")


def test_industrial_parse_keeps_whitelisted_short_entities():
    """端到端：industrial_parse_fn 输出中白名单短实体不被剥掉。"""
    from backend.extractor import industrial_parse_fn

    out = industrial_parse_fn(
        '[{"s":"叶轮","r":"包含","o":"叶轮流道被杂质堵塞"}]'
    )
    assert ("叶轮", "包含", "叶轮流道被杂质堵塞") in out, (
        "白名单短实体在抽取解析输出中必须保留")


# ════════════════════════════════════════════════════════════════════
# 十、RAGAS 断点 metrics 恢复（9.11-3 附带：续跑后逐题分不得丢失）
# ════════════════════════════════════════════════════════════════════

def test_ragas_ckpt_restore_merges_metrics():
    """回归防护：断点恢复必须把已评分批次的逐题 metrics 补回 per_question。

    曾发生：进程中断后断点续跑，已完成批次不重跑 → 这些题的 metrics 缺失
    （G 侧双口径只剩 8/15 题有分、与 V 的 15 题不可比，9.2-A1 变体重现）。
    """
    from backend.evaluator import _restore_metrics_from_ckpt

    per_question = [
        {"id": "q1", "status": "ok", "metrics": {}},          # 已评分但 metrics 丢失
        {"id": "q2", "status": "ok", "metrics": {}},          # 待评分
        {"id": "q3", "status": "failed", "error": "x"},      # 失败题
    ]
    saved = {"per_question": [
        {"id": "q1", "status": "ok",
         "metrics": {"faithfulness": 0.9, "answer_relevancy": 0.8}},
        {"id": "q3", "status": "failed"},
    ]}
    _restore_metrics_from_ckpt(per_question, saved)
    assert per_question[0]["metrics"].get("faithfulness") == 0.9, (
        "已完成批次的逐题 metrics 必须从断点恢复")
    assert per_question[1]["metrics"] == {}, "未评分题不得被误填"
    assert "error" in per_question[2], "失败题记录不得被覆盖"


# ════════════════════════════════════════════════════════════════════
# 十一、9.13：测试题生成全量覆盖（不再截断到"前 N 字符"）
# ════════════════════════════════════════════════════════════════════

def test_generate_qa_split_batches_covers_all_chunks():
    """回归防护：分片必须覆盖全部 chunk（任何字符不得因截断被丢弃）。

    曾只取手册前 12000 字符生成测试题 → 后半本知识无题覆盖。分片后
    每片 ≤ batch_chars，且所有 chunk 恰好完整落入某一片。
    """
    from generate_qa import _split_batches

    class _C:
        def __init__(self, i, text):
            self.chunk_id = i
            self.text = text
            self.source = "m.md"

    chunks = [_C(i, "字" * (500 if i % 3 else 300)) for i in range(30)]
    batches = _split_batches(chunks, batch_chars=2400)
    assert batches, "分片结果不得为空"
    # 全量覆盖：每个 chunk 恰好出现一次
    flat = [c for b in batches for c in b]
    assert len(flat) == len(chunks), "不得丢 chunk"
    assert len({id(c) for c in flat}) == len(chunks), "不得重复 chunk"
    # 单片不超上限
    for b in batches:
        assert sum(len(c.text) for c in b) <= 2400, "单片超过 batch_chars"
    # 总字符守恒
    assert sum(len(c.text) for c in flat) == sum(len(c.text) for c in chunks)


def test_generate_qa_never_truncates_content():
    """回归防护：generate_qa 不得对手册内容做 [:N] 截断。

    直接断言源文件不再出现 `[:12000]`/`content[:` 之类的截断写法。
    """
    src = open("generate_qa.py", encoding="utf-8").read()
    assert "[:12000]" not in src, "不得把手册内容截断到前 12000 字符"
    assert "content[:num]" not in src and "content[:" not in src, "不得对 content 截断"


# ════════════════════════════════════════════════════════════════════
# 十二、9.13：图谱路径保底分槽（社区正文不得挤掉路径）
# ════════════════════════════════════════════════════════════════════

def test_interleave_paths_round_robin():
    """回归防护：跨实体路径必须轮询交错，前部实体不得垄断。

    曾按实体顺序拼接路径 → "SHS 01013-2019" 的 5 条限值路径占满前槽，
    答案核心实体（热油泵停泵后）的盘车路径被 12 槽挤掉（auto_008/013）。
    """
    from backend.engines import _interleave_paths

    out = _interleave_paths([["A1", "A2"], ["B1"], ["C1", "C2", "C3"]])
    assert out == ["A1", "B1", "C1", "A2", "C2", "C3"], "必须按实体轮询交错"


def test_collect_graph_contexts_reserves_path_slots():
    """回归防护：有图谱路径时，社区正文最多占 (max_ctx - min_path_slots) 槽。

    auto_008/013 实证：9 条社区正文把答案核心路径挤出 12 槽 → 得分靠原文
    托底、图谱检索失败被掩盖。社区必须为路径让出最少 min_path_slots 槽。
    """
    from backend.evaluator import _collect_graph_contexts

    class _Manual:
        def query(self, q, **kwargs):
            return [([f"实体{i}", "答案B"], ["包含"]) for i in range(8)]

    class _Comm:
        def query(self, q, **kwargs):
            return [], [f"社区正文{i}" for i in range(12)]

    ctxs = _collect_graph_contexts("q", _Manual(), _Comm(), max_ctx=12, min_path_slots=4)
    comm_n = sum(1 for c in ctxs if "--[" not in c)
    path_n = sum(1 for c in ctxs if "--[" in c)
    assert comm_n <= 12 - 4, "社区正文必须为路径让出至少 4 槽"
    assert path_n >= 4, "图谱路径必须保住最少 min_path_slots 条"

    # 无路径时社区不截断（保底行为不变）
    ctxs2 = _collect_graph_contexts("q", type("E", (), {"query": lambda s, q: []})(),
                                    _Comm(), max_ctx=12, min_path_slots=4)
    assert len(ctxs2) == 12, "无路径时社区正文不应被截断"


def test_context_attribution_counts_path_and_text():
    """回归防护：归因统计必须区分"图谱路径"（含 --[ 标记）与社区原文。"""
    ev = RAGASEvaluator.__new__(RAGASEvaluator)
    kg = {"per_question": [
        {"id": "q1", "status": "ok", "type": "single-hop",
         "contexts": ["A--[包含]-->B", "C--[可能原因]-->D", "社区正文", ""]},
        {"id": "q2", "status": "ok", "type": "out-of-context", "contexts": [""]},
        {"id": "q3", "status": "failed"},
    ]}
    attr = ev._context_attribution(kg)
    q1 = next(a for a in attr if a["id"] == "q1")
    assert q1["n_path"] == 2 and q1["n_text"] == 1 and q1["n_ctx"] == 3
    assert all(a["id"] != "q3" for a in attr), "failed 题不计入归因"


# ════════════════════════════════════════════════════════════════════
# 十三、9.13：生成失败必须抛异常（禁止"假答案真分数"）
# ════════════════════════════════════════════════════════════════════

def test_generate_answer_fallback_raises_not_placeholder():
    """回归防护：所有生成通道失败时必须抛异常，不得返回固定兜底串。

    曾返回"无法生成答案：LLM 调用失败。"并照常进 RAGAS 评分 + 断点缓存
    → 失败题以假答案参评（faithfulness 可能因'未编造'反而虚高）。
    """
    ev = RAGASEvaluator.__new__(RAGASEvaluator)
    ev.answer_generator = None
    ev.api_client = None

    class _FailJudge:
        def invoke(self, prompt, **kwargs):
            raise ConnectionError("proxy down")

    ev.judge_llm = _FailJudge()
    with pytest.raises(RuntimeError):
        ev._generate_answer("prompt")
    # 兜底串不得再作为返回值出现
    with pytest.raises(RuntimeError):
        ev._generate_answer("prompt")


def test_source_declaration_uses_meta_when_injected():
    """回归防护：数据来源声明优先取 _meta（数据驱动），不硬编码 v4 文本。"""
    ev = RAGASEvaluator.__new__(RAGASEvaluator)
    decl = ev._render_source_declaration({
        "_meta": {"graph_source": "llm_extract_v41_xxx.json (316 triples)",
                  "judge_llm": "deepseek-v4-flash (proxy)",
                  "n_questions": 18}})
    assert "v41" in decl, "声明必须反映 _meta 中的图谱版本"
    assert "316" in decl
    # 无 _meta 时回退默认声明，不输出空节
    fallback = ev._render_source_declaration({})
    assert fallback.strip(), "缺省声明不得为空"
    assert "透明声明" in fallback
