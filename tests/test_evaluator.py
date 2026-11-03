"""evaluator 模块单元测试。

覆盖 RAGASEvaluator 的初始化与 generate_report 方法。
不调用真实 Ollama/RAGAS，使用 fake 依赖保证快速执行。
"""
import pytest
import os

from backend.evaluator import RAGASEvaluator, _question_fingerprint

@pytest.fixture
def fake_judge_llm():
    """模拟 LangChain 兼容的 judge LLM，不发起网络请求。"""

    class FakeLLM:
        def invoke(self, prompt, **kwargs):
            class FakeResponse:
                content = "测试答案"
            return FakeResponse()

    return FakeLLM()


@pytest.fixture
def fake_embed():
    """模拟 LangChain 兼容的 embedding 模型。"""

    class FakeEmbed:
        def embed_documents(self, texts):
            return [[0.1, 0.2, 0.3] for _ in texts]

        def embed_query(self, text):
            return [0.1, 0.2, 0.3]

    return FakeEmbed()


@pytest.fixture
def evaluator(fake_judge_llm, fake_embed):
    """使用 fake 依赖构建 RAGASEvaluator 实例。"""
    return RAGASEvaluator(fake_judge_llm, fake_embed, judge_model_name="fake-test")


@pytest.fixture
def full_results():
    """包含 Vector RAG 与 KG-RAG 的完整评估结果字典。"""
    return {
        "vector_rag": {
            "faithfulness": 0.82,
            "answer_relevancy": 0.78,
            "context_precision": 0.70,
            "context_recall": 0.72,
            "n_questions": 25,
            "per_question": [
                {"id": "Q1", "type": "single-hop", "question": "温度过高原因？"},
                {"id": "Q2", "type": "multi-hop", "question": "连锁故障路径？"},
            ],
            "by_type": {
                "single-hop": {"faithfulness": 0.85},
                "multi-hop": {"faithfulness": 0.75},
            },
        },
        "graph_rag": {
            "faithfulness": 0.89,
            "answer_relevancy": 0.85,
            "context_precision": 0.75,
            "context_recall": 0.88,
            "n_questions": 25,
            "per_question": [
                {"id": "Q1", "type": "single-hop"},
                {"id": "Q2", "type": "multi-hop"},
            ],
            "by_type": {
                "single-hop": {"faithfulness": 0.88},
                "multi-hop": {"faithfulness": 0.92},
            },
        },
        "comparison": {
            "delta": {
                "faithfulness": 0.07,
                "answer_relevancy": 0.07,
                "context_precision": 0.05,
                "context_recall": 0.16,
            },
            "winner_per_metric": {
                "faithfulness": "graph_rag",
                "answer_relevancy": "graph_rag",
                "context_precision": "graph_rag",
                "context_recall": "graph_rag",
            },
        },
    }

def test_evaluator_initialization(evaluator, fake_judge_llm, fake_embed):
    """验证 RAGASEvaluator 初始化后关键属性正确赋值。"""
    assert evaluator is not None
    assert evaluator.judge_llm is fake_judge_llm
    # evaluator 会用 _SafeOllamaEmbed 包装 embed_model，检查底层是 fake_embed
    assert evaluator.ragas_embed._base is fake_embed
    assert evaluator.judge_model_name == "fake-test"


def test_generate_report_full(tmp_path, evaluator, full_results):
    """generate_report 应创建 Markdown 文件并包含所有关键指标数值。"""
    output_path = str(tmp_path / "eval_report.md")
    result_path = evaluator.generate_report(full_results, output_path)

    assert os.path.isabs(result_path)
    assert os.path.exists(result_path), f"报告文件未生成: {result_path}"

    content = (tmp_path / "eval_report.md").read_text(encoding="utf-8")

    assert "KGRAG 评估报告" in content
    assert "fake-test" in content, "应包含 judge_model_name"

    assert "0.82" in content
    assert "0.89" in content
    assert "+0.07" in content

    assert "结论" in content


def test_generate_report_missing_graph_rag(tmp_path, evaluator):
    """仅含 Vector RAG（graph_rag=None）时应生成无对比列的报告。"""
    partial = {
        "vector_rag": {
            "faithfulness": 0.80,
            "answer_relevancy": 0.72,
            "context_precision": 0.65,
            "context_recall": 0.55,
            "n_questions": 10,
            "per_question": [],
        },
        "graph_rag": None,
    }

    output_path = str(tmp_path / "partial.md")
    result_path = evaluator.generate_report(partial, output_path)
    assert os.path.exists(result_path)

    content = (tmp_path / "partial.md").read_text(encoding="utf-8")
    assert "仅评估了 Vector RAG" in content, "应提示未对比"
    assert "KG-RAG" not in content or "差值" not in content, (
        "无 graph_rag 时不应出现对比列"
    )


def test_generate_report_empty_results(tmp_path, evaluator):
    """空结果字典应优雅处理：要么生成最简报告，要么抛出明确错误。"""
    output_path = str(tmp_path / "empty.md")

    try:
        result_path = evaluator.generate_report({}, output_path)
        assert os.path.exists(result_path), (
            "空结果应仍能生成最简报告文件"
        )
    except (ValueError, KeyError) as exc:
        assert len(str(exc)) > 0


# ============================================================
# 修复项回归测试：C2 社区正文优先 + 全局去重；J2 CoT 分离
# ============================================================

def test_split_cot_output_separates_reasoning_and_answer():
    """J2：CoT 输出含【最终答案】标记时，应正确分离推理与答案。"""
    from backend.evaluator import _split_cot_output
    raw = (
        "【推理过程】根据检索路径，轴承温度过高可能由润滑不良导致。\n"
        "【最终答案】\n【故障现象】轴承温度过高\n【可能原因】润滑不良\n"
    )
    answer, reasoning = _split_cot_output(raw)
    assert "润滑不良" in reasoning
    assert "故障现象" in answer
    assert reasoning != answer, "J2 修复后推理不应等于答案"


def test_split_cot_output_without_marker():
    """J2 兼容：无标记时推理置空，不得复制答案。"""
    from backend.evaluator import _split_cot_output
    answer, reasoning = _split_cot_output("只有一段答案")
    assert answer == "只有一段答案"
    assert reasoning == ""


def test_collect_graph_contexts_community_first_and_dedup():
    """C2/H1：社区正文应排在路径碎片之前，且全局去重。"""
    from backend.evaluator import _collect_graph_contexts

    class FakeManualEngine:
        def query(self, q, **kwargs):
            # 返回 2 条短路径
            return [
                (["实体A", "实体B"], ["可能原因"]),
                (["实体A", "实体C"], ["解决方法"]),
            ]

    class FakeCommunityEngine:
        def query(self, q, **kwargs):
            # 社区返回长文本，其中第二条与第一条内容重复（H1 场景）
            return (
                ["comm1"],
                ["【社区】这是包含完整依据的长文本正文……" * 3,
                 "【社区】这是包含完整依据的长文本正文……" * 3],
            )

    contexts = _collect_graph_contexts("测试问题", FakeManualEngine(), FakeCommunityEngine())
    # 社区正文在前
    assert "长文本正文" in contexts[0]
    # 全局去重：重复的社区文本只保留一次（1 社区正文 + 2 条不同路径 = 3）
    assert len(contexts) == 3
    assert contexts.count(contexts[0]) == 1, "重复的社区正文只应出现一次"
    # 路径碎片在后
    assert "--[可能原因]-->" in contexts[1]


def test_truncate_contexts_keeps_more_than_four():
    """C2：评分 context 上限应 > 4，避免只看碎片。"""
    from backend.evaluator import _truncate_contexts
    ctxs = [f"片段{i}" * 20 for i in range(10)]  # 10 个 60 字符片段
    result = _truncate_contexts(ctxs)
    assert len(result) >= 8, "修复 C2 后应保留更多 context"
    assert len(result) == 10, "max_ctx=12（9.10-④）且预算内时 10 个片段应全部保留"


def test_generate_answer_with_cot_returns_final_answer():
    """9.10-③：_generate_answer_with_cot 应返回【最终答案】为 answer、推理过程为 cot_reasoning。

    回归：此前 `reasoning, answer = _split_cot_output(...)` 赋值反了（该函数
    返回 (answer, reasoning)），导致 G 侧 answer 字段实际存的是【推理过程】。
    """
    ev = RAGASEvaluator.__new__(RAGASEvaluator)

    class FakeGen:
        def __call__(self, prompt):
            return ("【推理过程】推理内容……\n"
                    "【最终答案】\n【故障现象】轴承温度过高\n【可能原因】润滑不良")

    ev._generate_answer = FakeGen()
    answer, cot = ev._generate_answer_with_cot("ctx", "q")
    assert "故障现象" in answer, "answer 应为【最终答案】部分"
    assert "推理内容" not in answer, "answer 不应混入推理过程"
    assert "推理内容" in cot, "cot_reasoning 应为推理过程"
    assert "故障现象" not in cot, "cot_reasoning 不应混入最终答案"


def test_build_type_breakdown_fills_scores():
    """KG-9：按题型分层应填充分数（不再只有题数），且均值正确。"""
    evaluator = RAGASEvaluator.__new__(RAGASEvaluator)
    vec = {"per_question": [
        {"type": "single-hop", "metrics": {"faithfulness": 0.8, "answer_relevancy": 0.6}},
        {"type": "single-hop", "metrics": {"faithfulness": 1.0, "answer_relevancy": 0.8}},
        {"type": "multi-hop", "metrics": {"faithfulness": 0.5}},
    ]}
    kg = {"per_question": [
        {"type": "single-hop", "metrics": {"faithfulness": 0.9}},
    ]}
    breakdown = evaluator._build_type_breakdown(vec, kg)
    sh = breakdown["single-hop"]
    assert sh["n"] == 2, "single-hop 题数应为 vec 侧 2"
    assert abs(sh["vec"]["faithfulness"] - 0.9) < 1e-9, "0.8/1.0 均值应为 0.9"
    assert abs(sh["vec"]["answer_relevancy"] - 0.7) < 1e-9
    assert abs(sh["kg"]["faithfulness"] - 0.9) < 1e-9
    assert abs(breakdown["multi-hop"]["vec"]["faithfulness"] - 0.5) < 1e-9
    # 无 metrics 的题不影响分层
    assert "unknown" not in breakdown


def test_build_type_breakdown_unknown_type():
    """KG-9：缺 type 字段时归入 unknown 且不崩溃。"""
    evaluator = RAGASEvaluator.__new__(RAGASEvaluator)
    results = {"per_question": [
        {"metrics": {"faithfulness": 0.4}},
        {"type": "multi-hop", "metrics": {"faithfulness": 0.7}},
    ]}
    breakdown = evaluator._build_type_breakdown(results, {"per_question": []})
    assert breakdown["unknown"]["n"] == 1
    assert abs(breakdown["unknown"]["vec"]["faithfulness"] - 0.4) < 1e-9
    assert abs(breakdown["multi-hop"]["vec"]["faithfulness"] - 0.7) < 1e-9


def test_attach_per_question_metrics_writes_scores():
    """KG-9：逐题指标应写回 per_question，NaN 被过滤且不抛异常。"""
    import pandas as pd
    evaluator = RAGASEvaluator.__new__(RAGASEvaluator)

    class M:
        name = "faithfulness"

    class M2:
        name = "answer_relevancy"

    batch = [{"id": "Q1"}, {"id": "Q2"}]
    df = pd.DataFrame({
        "faithfulness": [0.8, 0.6],
        "answer_relevancy": [float("nan"), 0.5],
    })
    evaluator._attach_per_question_metrics(batch, df, [M(), M2()])
    assert batch[0]["metrics"]["faithfulness"] == 0.8
    assert "answer_relevancy" not in batch[0]["metrics"], "NaN 应被过滤"
    assert batch[1]["metrics"]["answer_relevancy"] == 0.5


def test_agg_metrics_excluding_drops_ooc():
    """9.2-A2：剔除 out-of-context 后聚合，均值只基于非 OOC 题且标注样本数。"""
    evaluator = RAGASEvaluator.__new__(RAGASEvaluator)
    results = {"per_question": [
        {"id": "Q1", "type": "single-hop", "status": "ok",
         "metrics": {"faithfulness": 1.0, "answer_relevancy": 1.0}},
        {"id": "Q2", "type": "multi-hop", "status": "ok",
         "metrics": {"faithfulness": 0.0, "answer_relevancy": 0.0}},
        {"id": "Q3", "type": "out-of-context", "status": "ok",
         "metrics": {"faithfulness": 1.0, "answer_relevancy": 1.0}},
        {"id": "Q4", "type": "failed", "status": "failed"},
    ]}
    agg = evaluator._agg_metrics_excluding(results)
    assert abs(agg["faithfulness"] - 0.5) < 1e-9, "Q1/Q2 均值 0.5，Q3(OOC) 不参与"
    assert agg["faithfulness_n"] == 2
    assert agg["answer_relevancy_n"] == 2
    # 无 OOC 题时全量统计
    agg_all = evaluator._agg_metrics_excluding(results, exclude_types=())
    assert agg_all["faithfulness_n"] == 3


def test_generate_report_excludes_ooc_section(tmp_path, evaluator):
    """generate_report 应包含'剔除 out-of-context 后'双口径表，OOC 题不参与该表均值。"""
    results = {
        "vector_rag": {
            "faithfulness": 0.5, "answer_relevancy": 0.5,
            "context_precision": 0.5, "context_recall": 0.5,
            "n_questions": 3,
            "per_question": [
                {"id": "Q1", "type": "single-hop", "status": "ok",
                 "metrics": {"faithfulness": 1.0, "answer_relevancy": 1.0,
                             "context_precision": 1.0, "context_recall": 1.0}},
                {"id": "Q2", "type": "multi-hop", "status": "ok",
                 "metrics": {"faithfulness": 0.0, "answer_relevancy": 0.0,
                             "context_precision": 0.0, "context_recall": 0.0}},
                {"id": "Q3", "type": "out-of-context", "status": "ok",
                 "metrics": {"faithfulness": 1.0, "answer_relevancy": 0.0,
                             "context_precision": 0.0, "context_recall": 0.0}},
            ],
        },
        "graph_rag": {
            "faithfulness": 0.87, "answer_relevancy": 0.5,
            "context_precision": 0.5, "context_recall": 0.5,
            "n_questions": 3,
            "per_question": [
                {"id": "Q1", "type": "single-hop", "status": "ok",
                 "metrics": {"faithfulness": 0.8, "answer_relevancy": 0.5,
                             "context_precision": 0.5, "context_recall": 0.5}},
                {"id": "Q2", "type": "multi-hop", "status": "ok",
                 "metrics": {"faithfulness": 0.8, "answer_relevancy": 0.5,
                             "context_precision": 0.5, "context_recall": 0.5}},
                {"id": "Q3", "type": "out-of-context", "status": "ok",
                 "metrics": {"faithfulness": 1.0, "answer_relevancy": 0.0,
                             "context_precision": 0.0, "context_recall": 0.0}},
            ],
        },
    }
    output_path = str(tmp_path / "eval_report.md")
    evaluator.generate_report(results, output_path)
    content = open(output_path, encoding="utf-8").read()

    assert "剔除 out-of-context 题后（双口径）" in content
    # 主表（全部题）用顶层聚合字段：vec faithfulness=0.50、kg=0.87
    assert "| Faithfulness | 0.50 | 0.87 |" in content
    # 双口径表（剔除 Q3 后现算）：vec faithfulness=(1+0)/2=0.50(2)；kg=(0.8+0.8)/2=0.80(2)
    assert "0.50(2)" in content
    assert "0.80(2)" in content
    # 双口径剔除 OOC 后 KG 在 faithfulness 占优 → 结论含双口径胜场说明
    assert "剔除 out-of-context 题后：KG-RAG 在 1 项" in content


def test_question_fingerprint_deterministic_and_sensitive():
    """P2-⑦：题目指纹应确定性，且题目内容变化（id 不变）时指纹变化。"""
    item1 = {"id": "Q1", "question": "问题A", "ground_truth": "答案A"}
    item2 = {"id": "Q1", "question": "问题A", "ground_truth": "答案A"}
    item3 = {"id": "Q1", "question": "问题A", "ground_truth": "答案B改"}
    assert _question_fingerprint(item1) == _question_fingerprint(item2), "同题同内容指纹应一致"
    assert _question_fingerprint(item1) != _question_fingerprint(item3), "内容变化指纹应不同"
    assert len(_question_fingerprint(item1)) == 32, "md5 hexdigest 长度 32"


# ============================================================
# 9.10：OOC 拒答正确性指标（消除 OOC 口径不对称的实质妥协）
# ============================================================

class _FakeJudge:
    def __init__(self, content="YES"):
        self._content = content

    def invoke(self, prompt, **kwargs):
        class R:
            content = self._content
        return R()


def test_judge_ooc_refusal_yes_when_refuses():
    """9.10：回答明确拒答且未编造 → judge 判 YES → 1.0。"""
    ev = RAGASEvaluator.__new__(RAGASEvaluator)
    ev.judge_llm = _FakeJudge("YES")
    assert ev._judge_ooc_refusal(
        "燃气轮机叶片裂纹用哪种无损检测？",
        "知识库中未找到相关信息。") == 1.0


def test_judge_ooc_refusal_no_when_fabricates():
    """9.10：回答编造了具体答案 → judge 判 NO → 0.0。"""
    ev = RAGASEvaluator.__new__(RAGASEvaluator)
    ev.judge_llm = _FakeJudge("NO")
    assert ev._judge_ooc_refusal(
        "燃气轮机叶片裂纹用哪种无损检测？",
        "通常采用渗透检测和磁粉检测，叶片表面裂纹……") == 0.0


def test_judge_ooc_refusal_fallback_rule_when_judge_fails():
    """9.10：judge 调用异常时用规则信号兜底（含拒答信号 → 1.0）。"""
    ev = RAGASEvaluator.__new__(RAGASEvaluator)

    class Boom:
        def invoke(self, prompt, **kwargs):
            raise RuntimeError("judge 不可用")

    ev.judge_llm = Boom()
    assert ev._judge_ooc_refusal(
        "锅炉水位计堵塞如何判断？",
        "根据当前知识库未找到相关信息。") == 1.0
    assert ev._judge_ooc_refusal(
        "锅炉水位计堵塞如何判断？",
        "可通过观察水位波动是否迟缓来判断。") == 0.0


def test_ooc_metrics_for_only_ooc_questions():
    """9.10：仅 out-of-context 题计算拒答指标，其余题为空 dict。"""
    ev = RAGASEvaluator.__new__(RAGASEvaluator)
    ev.judge_llm = _FakeJudge("YES")
    assert ev._ooc_metrics_for({"type": "single-hop"}, "答案") == {}
    m = ev._ooc_metrics_for(
        {"type": "out-of-context", "question": "燃气轮机叶片裂纹？"}, "未找到相关信息。")
    assert m == {"ooc_refusal": 1.0}


def test_ooc_refusal_prompt_uses_injected_domain():
    """A1 修复：OOC 拒答 prompt 使用注入的领域描述，而非硬编码"工业离心泵"。

    原缺陷：_OOC_REFUSAL_PROMPT 写死 pump 领域，shukongdashi（数控机床）等
    非泵数据集的 10 道 OOC 题被灌入错误领域前提 → 拒答判定口径不严。
    """
    class _RecordingJudge:
        def __init__(self):
            self.prompts = []

        def invoke(self, prompt, **kwargs):
            self.prompts.append(prompt)

            class R:
                content = "YES"
            return R()

    ev = RAGASEvaluator.__new__(RAGASEvaluator)
    j = _RecordingJudge()
    ev.judge_llm = j
    ev.ooc_domain = "数控机床（FANUC/SIEMENS 等系统）故障维保手册"
    ev._judge_ooc_refusal("电火花加工电极丝断丝原因？", "知识库中未找到相关信息。")
    assert j.prompts, "judge 必须收到 prompt"
    p = j.prompts[0]
    assert "数控机床（FANUC/SIEMENS 等系统）故障维保手册" in p, \
        "注入的领域必须出现在 OOC 拒答 prompt 中"
    assert "工业离心泵" not in p, "不得再硬编码 pump 领域"


def test_generate_report_ooc_refusal_section(tmp_path, evaluator):
    """9.10：报告应包含 OOC 拒答正确性对比节与数据来源声明。"""
    results = {
        "vector_rag": {
            "faithfulness": 0.5, "answer_relevancy": 0.5,
            "context_precision": 0.5, "context_recall": 0.5,
            "n_questions": 3,
            "per_question": [
                {"id": "Q1", "type": "single-hop", "status": "ok",
                 "metrics": {"faithfulness": 1.0}},
                {"id": "auto_016", "type": "out-of-context", "status": "ok",
                 "metrics": {"ooc_refusal": 1.0}},
            ],
        },
        "graph_rag": {
            "faithfulness": 0.87, "answer_relevancy": 0.5,
            "context_precision": 0.5, "context_recall": 0.5,
            "n_questions": 3,
            "per_question": [
                {"id": "Q1", "type": "single-hop", "status": "ok",
                 "metrics": {"faithfulness": 0.8}},
                {"id": "auto_016", "type": "out-of-context", "status": "ok",
                 "metrics": {"ooc_refusal": 0.0}},
            ],
        },
    }
    output_path = str(tmp_path / "eval_report.md")
    evaluator.generate_report(results, output_path)
    content = open(output_path, encoding="utf-8").read()

    assert "Out-of-Context 题拒答正确性" in content
    assert "| auto_016 | 1.0 | 0.0 |" in content
    assert "Vector RAG 正确拒答 1/1 题；KG-RAG 正确拒答 0/1 题。" in content
    # 主表含 OOC 口径说明
    assert "主表含 out-of-context 题" in content
    # test-driven 数据来源声明
    assert "数据来源声明" in content
    assert "test-driven" in content


# ════════════════════════════════════════════════════════════════════
# 补充：纯聚合函数与本地分词
# ════════════════════════════════════════════════════════════════════

def test_local_tokenizers_no_network():
    """本地分句/分词零网络依赖，中英文混合文本可处理。"""
    from backend.evaluator import _sent_tokenize_local, _word_tokenize_local

    sents = _sent_tokenize_local("轴承过热。泵停机；\n继续排查")
    assert len(sents) >= 3, "应按中英文句号/分号/换行切分"
    words = _word_tokenize_local("叶轮磨损导致流量下降")
    assert any("叶轮" in w for w in words), "jieba 应切出领域词"


def test_compare_results_computes_delta_and_winner():
    """逐指标 delta 与胜者计算，含平局与缺分情况。"""
    ev = RAGASEvaluator.__new__(RAGASEvaluator)
    results = {
        "vector_rag": {"faithfulness": 0.5, "answer_relevancy": 0.6,
                       "context_precision": 0.7, "context_recall": 0.8},
        "graph_rag": {"faithfulness": 0.5, "answer_relevancy": 0.9,
                      "context_precision": None, "context_recall": 0.8},
    }
    out = ev._compare_results(results)
    assert out["delta"]["faithfulness"] == 0.0
    assert out["winner_per_metric"]["faithfulness"] == "tie"
    assert out["delta"]["answer_relevancy"] == pytest.approx(0.3)
    assert out["winner_per_metric"]["answer_relevancy"] == "kg_rag"
    assert out["delta"]["context_precision"] is None, "缺分 → delta None"
    assert out["winner_per_metric"]["context_precision"] == "unknown"
    assert out["winner_per_metric"]["context_recall"] == "tie"


def test_agg_metrics_excluding_no_valid_samples():
    """无有效样本时均值为 None、样本数为 0。"""
    ev = RAGASEvaluator.__new__(RAGASEvaluator)
    agg = ev._agg_metrics_excluding({"per_question": [
        {"id": "x", "status": "failed", "metrics": {}},
        {"id": "y", "type": "out-of-context", "status": "ok", "metrics": {}},
    ]})
    assert agg["faithfulness"] is None and agg["faithfulness_n"] == 0


def test_per_question_discrimination_counts_wins_and_ties():
    """逐题区分度：统计同分/胜场，缺分题与 OOC 题不参与。"""
    ev = RAGASEvaluator.__new__(RAGASEvaluator)

    def _mk(scores):
        return {"per_question": [
            {"id": "q1", "type": "single-hop", "status": "ok",
             "metrics": {"faithfulness": scores.get("q1_f")}},
            {"id": "q2", "type": "single-hop", "status": "ok",
             "metrics": {"faithfulness": scores.get("q2_f")}},
            {"id": "auto_ooc", "type": "out-of-context", "status": "ok",
             "metrics": {"faithfulness": 0.0}},
        ]}

    vec = _mk({"q1_f": 0.5, "q2_f": 0.5})
    kg = _mk({"q1_f": 0.8, "q2_f": 0.5})  # q1 G赢, q2 同分
    disc = ev._per_question_discrimination(vec, kg)
    f = disc["faithfulness"]
    assert f["total"] == 2, "OOC 题不参与逐题区分"
    assert f["same"] == 1 and f["g_wins"] == 1 and f["v_wins"] == 0
    assert f["ties"] == 1, "同分即平局"


def test_per_question_discrimination_skips_missing_metric():
    """一侧缺分时该题不计入 total（不制造虚假胜场）。"""
    ev = RAGASEvaluator.__new__(RAGASEvaluator)
    vec = {"per_question": [{"id": "q1", "type": "single-hop", "status": "ok",
                             "metrics": {"answer_relevancy": 0.5}}]}
    kg = {"per_question": [{"id": "q1", "type": "single-hop", "status": "ok",
                            "metrics": {}}]}  # kg 缺 answer_relevancy
    disc = ev._per_question_discrimination(vec, kg)
    assert disc["answer_relevancy"]["total"] == 0


# ════════════════════════════════════════════════════════════════════
# 补充：答案生成降级链 + 图评估 process 分支
# ════════════════════════════════════════════════════════════════════

def test_generate_answer_prefers_answer_generator():
    """answer_generator 存在时直接采用其输出。"""
    ev = RAGASEvaluator.__new__(RAGASEvaluator)
    ev.answer_generator = lambda prompt: "生成器答案"
    assert ev._generate_answer("prompt") == "生成器答案"


def test_generate_answer_falls_back_to_api_client_on_generator_error():
    """answer_generator 抛异常 → 回退到 api_client。"""
    ev = RAGASEvaluator.__new__(RAGASEvaluator)

    def _boom(prompt):
        raise RuntimeError("gen down")

    class _Client:
        def chat(self, messages, **kwargs):
            return {"content": "api_client 答案"}

    ev.answer_generator = _boom
    ev.api_client = _Client()
    assert ev._generate_answer("prompt") == "api_client 答案"


def test_generate_answer_falls_back_to_judge_on_empty_api():
    """api_client 返回空内容 → 回退 judge_llm。"""
    ev = RAGASEvaluator.__new__(RAGASEvaluator)
    ev.answer_generator = None

    class _EmptyClient:
        def chat(self, messages, **kwargs):
            return {"content": ""}

    class _Judge:
        def invoke(self, prompt, **kwargs):
            class R:
                content = "judge 答案"

                def __str__(self):
                    return self.content
            return R()

    ev.api_client = _EmptyClient()
    ev.judge_llm = _Judge()
    assert ev._generate_answer("prompt") == "judge 答案"


def test_generate_answer_judge_fallback_raises_on_failure():
    """全部通道失败 → 抛 RuntimeError（禁止假答案）。"""
    ev = RAGASEvaluator.__new__(RAGASEvaluator)
    ev.answer_generator = None
    ev.api_client = None

    class _FailJudge:
        def invoke(self, prompt, **kwargs):
            raise ConnectionError("down")

    ev.judge_llm = _FailJudge()
    with pytest.raises(RuntimeError):
        ev._generate_answer("prompt")


def test_evaluate_graph_rag_non_cot_with_context():
    """图评估 process：非 CoT 且有 context 时生成答案并返回路径上下文。"""
    ev = RAGASEvaluator.__new__(RAGASEvaluator)
    ev.use_cot = False
    ev.answer_generator = lambda prompt: "结构化诊断答案"

    captured = {}

    def _fake_ragas(test_items, process, label, output_dir):
        captured["results"] = [process(i) for i in test_items]
        return {"label": label, "per_question": []}

    ev._run_ragas_eval = _fake_ragas

    class _Manual:
        def query(self, q, **kwargs):
            return [(["泵", "叶轮"], ["包含"])]

    class _Comm:
        def query(self, q, **kwargs):
            return [], ["社区正文片段"]

    ev._evaluate_graph_rag(
        [{"id": "q1", "question": "叶轮有什么作用？"}],
        {"manual": _Manual(), "community": _Comm()}, "/tmp")
    ans, ctxs = captured["results"][0]
    assert ans == "结构化诊断答案"
    joined = "".join(ctxs)
    assert "--[包含]-->" in joined, "图谱路径必须进入评分 context"


def test_evaluate_graph_rag_empty_context_placeholder():
    """图评估 process：空检索返回拒答占位 + 空 context（两侧归一化）。"""
    ev = RAGASEvaluator.__new__(RAGASEvaluator)
    ev.use_cot = False

    class _EmptyManual:
        def query(self, q, **kwargs):
            return []

    class _EmptyComm:
        def query(self, q, **kwargs):
            return [], []

    captured = {}

    def _fake_ragas(test_items, process, label, output_dir):
        captured["results"] = [process(i) for i in test_items]
        return {}

    ev._run_ragas_eval = _fake_ragas
    ev._evaluate_graph_rag(
        [{"id": "q1", "question": "完全无关问题"}],
        {"manual": _EmptyManual(), "community": _EmptyComm()}, "/tmp")
    ans, ctxs = captured["results"][0]
    assert ans == "知识图谱中未找到相关信息。"
    assert ctxs == [""], "空 context 归一化为 [\"\"]"


def test_evaluate_graph_rag_cot_empty_context():
    """图评估 process：CoT 模式空检索时 cot_reasoning 与答案一致。"""
    ev = RAGASEvaluator.__new__(RAGASEvaluator)
    ev.use_cot = True

    class _EmptyManual:
        def query(self, q, **kwargs):
            return []

    class _EmptyComm:
        def query(self, q, **kwargs):
            return [], []

    captured = {}

    def _fake_ragas(test_items, process, label, output_dir):
        captured["results"] = [process(i) for i in test_items]
        return {}

    ev._run_ragas_eval = _fake_ragas
    ev._evaluate_graph_rag(
        [{"id": "q1", "question": "完全无关问题"}],
        {"manual": _EmptyManual(), "community": _EmptyComm()}, "/tmp")
    ans, ctxs, extra = captured["results"][0]
    assert ans == "知识图谱中未找到相关信息。"
    assert extra["cot_reasoning"] == ans


def test_evaluate_graph_rag_ooc_gate_not_applied_to_in_context():
    """2026-08-07：OOC 拒答门只对 out-of-context 生效，in-context 不误拒答。"""
    ev = RAGASEvaluator.__new__(RAGASEvaluator)
    ev.use_cot = True
    ev._generate_answer_with_cot = lambda ctx, q: ("合成答案", "推理")

    captured = {}

    def _fake_ragas(test_items, process, label, output_dir):
        captured["results"] = [process(i) for i in test_items]
        return {"label": label}

    ev._run_ragas_eval = _fake_ragas

    class _Manual:
        def query(self, q, **kwargs):
            return [(["实体A", "实体B"], ["可能原因"])]

    class _Comm:
        def query(self, q, **kwargs):
            return [], ["abcdefghij"]

    ev._evaluate_graph_rag(
        [{"id": "q1", "type": "single-hop", "question": "完全无关问题XYZ123"}],
        {"manual": _Manual(), "community": _Comm()}, "/tmp")
    ans, _, extra = captured["results"][0]
    assert ans == "合成答案", "in-context 题不得因相关度启发式被硬拒答"
    assert "ooc_refusal_reason" not in extra


def test_evaluate_graph_rag_ooc_gate_applied_to_ooc():
    """2026-08-07：out-of-context 题相关度不足时仍应正确拒答。"""
    ev = RAGASEvaluator.__new__(RAGASEvaluator)
    ev.use_cot = True
    ev._generate_answer_with_cot = lambda ctx, q: ("编造答案", "推理")

    captured = {}

    def _fake_ragas(test_items, process, label, output_dir):
        captured["results"] = [process(i) for i in test_items]
        return {"label": label}

    ev._run_ragas_eval = _fake_ragas

    class _Manual:
        def query(self, q, **kwargs):
            return [(["实体A", "实体B"], ["可能原因"])]

    class _Comm:
        def query(self, q, **kwargs):
            return [], ["abcdefghij"]

    ev._evaluate_graph_rag(
        [{"id": "q1", "type": "out-of-context", "question": "完全无关问题XYZ123"}],
        {"manual": _Manual(), "community": _Comm()}, "/tmp")
    ans, _, extra = captured["results"][0]
    assert "未检索到与问题相关的内容" in ans, "OOC 题相关度不足应拒答"
    assert "ooc_refusal_reason" in extra


def test_evaluate_system_dispatches_dict_to_graph():
    """_evaluate_system：engine 为 dict 时转发到图评估，否则向量路径。"""
    ev = RAGASEvaluator.__new__(RAGASEvaluator)
    ev.use_cot = False
    ev.answer_generator = lambda p: "答案"

    class _VecEngine:
        def query(self, q, **kwargs):
            return {"answer": "向量答案", "contexts": ["向量上下文"]}

    captured = {}

    def _fake_ragas(test_items, process, label, output_dir):
        captured["label"] = label
        captured["results"] = [process(i) for i in test_items]
        return {"label": label}

    ev._run_ragas_eval = _fake_ragas
    ev._evaluate_system([{"id": "q1", "question": "问题"}], _VecEngine(), "vector_rag", "/tmp")
    assert captured["label"] == "vector_rag"
    ans, ctxs = captured["results"][0]
    assert ans == "向量答案"
    assert ctxs == ["向量上下文"]

    # dict engine → 图评估
    ev._evaluate_system([{"id": "q1", "question": "问题"}],
                        {"manual": None, "community": None}, "graph_rag", "/tmp")
    assert captured["label"] == "graph_rag"


# ════════════════════════════════════════════════════════════════════
# 补充：纯函数分支补齐（_split_cot_output / _truncate_contexts /
#      _collect_graph_contexts / _restore_metrics_from_ckpt）
# ════════════════════════════════════════════════════════════════════

def test_split_cot_output_empty_and_variants():
    """CoT 解析：空输入与多种【最终答案】标记变体。"""
    from backend.evaluator import _split_cot_output
    assert _split_cot_output("") == ("", "")
    assert _split_cot_output("推理最终答案：答案") == ("答案", "推理")   # 全角冒号
    assert _split_cot_output("推理最终答案:答案") == ("答案", "推理")    # 半角冒号
    assert _split_cot_output("推理[最终答案]答案") == ("答案", "推理")   # 方括号
    assert _split_cot_output("无标记内容") == ("无标记内容", "")


def test_truncate_contexts_edge_branches():
    """截断：空输入、空串跳过、超长截断、预算超支、max_ctx 上限。"""
    from backend.evaluator import _truncate_contexts
    assert _truncate_contexts([]) == [""], "空输入应返回 [\"\"]"

    out = _truncate_contexts(["", "有效片段"])
    assert "有效片段" in out and "" not in out, "空字符串应被跳过"

    out = _truncate_contexts(["X" * 100], max_chars=10)
    assert len(out[0]) == 10, "超过 max_chars 的片段应被截断"

    out = _truncate_contexts(["aaaa", "bbbb"], max_chars=100, total_budget=6)
    assert out == ["aaaa"], "预算耗尽时应停止追加"

    out = _truncate_contexts(["a", "b", "c"], max_chars=100, max_ctx=2, total_budget=1000)
    assert len(out) == 2, "max_ctx 上限应中断"


def test_collect_graph_contexts_min_path_slots_and_no_rel():
    """收集：names 无 rels 时逗号拼接，且社区正文让位 min_path_slots。"""
    from backend.evaluator import _collect_graph_contexts

    class Manual:
        def query(self, q, **kwargs):
            return [(["实体A", "实体B"], []),            # 无 rels → 逗号拼接
                    (["实体X", "实体Y"], ["可能原因"])]  # 有 rels → 箭头

    class Comm:
        def query(self, q, **kwargs):
            return [], ["社区1"] * 20  # 大量社区，验证 min_path_slots 让位

    ctxs = _collect_graph_contexts("问题", Manual(), Comm(), max_ctx=6, min_path_slots=2)
    assert ctxs.count("社区1") == 1, "全局去重后社区正文只保留一次"
    assert "实体A, 实体B" in ctxs, "names 无 rels 时应逗号拼接"
    assert ctxs.index("实体A, 实体B") > ctxs.index("社区1"), "社区正文仍在前"


def test_collect_graph_contexts_relevant_paths_first():
    """动态组装：问题含代码 token 时，含该代码的图谱路径应前置。"""
    from backend.evaluator import _collect_graph_contexts

    class Manual:
        def query(self, q, **kwargs):
            return [
                (["无关实体A", "无关实体B"], ["包含"]),
                (["X轴参考点减速信号(*DECX)", "输入地址X018-5"], ["包含"]),
            ]

    class Comm:
        def query(self, q, **kwargs):
            return [], ["社区正文……" * 5]

    ctxs = _collect_graph_contexts(
        "X轴参考点减速信号（*DECX）的固定输入地址X018-5是什么？",
        Manual(), Comm())
    assert "X018-5" in ctxs[0], "含代码 token 的路径应排到首位"
    assert "社区正文" in "".join(ctxs[1:]), "相关路径之后仍应保留社区正文"


def test_collect_graph_contexts_community_first_without_code():
    """动态组装：问题无代码 token 时保持社区正文前置。"""
    from backend.evaluator import _collect_graph_contexts

    class Manual:
        def query(self, q, **kwargs):
            return [(["实体A", "实体B"], ["可能原因"])]

    class Comm:
        def query(self, q, **kwargs):
            return [], ["社区正文" * 3]

    ctxs = _collect_graph_contexts("主轴停止的故障原因是什么？", Manual(), Comm())
    assert ctxs[0].startswith("社区正文"), "无代码 token 时应保持社区正文在前"


def test_collect_graph_contexts_dynamic_keeps_path_slots():
    """动态组装：相关路径不足 min_path_slots 时用其余路径补足槽位。"""
    from backend.evaluator import _collect_graph_contexts

    class Manual:
        def query(self, q, **kwargs):
            return [
                (["实体A", "X018-5"], ["包含"]),
                (["实体B", "实体C"], ["包含"]),
                (["实体D", "实体E"], ["包含"]),
                (["实体F", "实体G"], ["包含"]),
                (["实体H", "实体I"], ["包含"]),
            ]

    class Comm:
        def query(self, q, **kwargs):
            return [], [f"社区{i}" for i in range(12)]

    ctxs = _collect_graph_contexts("X018-5地址", Manual(), Comm(),
                                   max_ctx=12, min_path_slots=4)
    assert ctxs[0] == "实体A--[包含]-->X018-5"
    path_n = sum(1 for c in ctxs if "--[包含]-->" in c)
    assert path_n >= 4, "路径槽位不得低于 min_path_slots"


def test_query_code_tokens_extracts_codes():
    """动态组装的关键 token 提取：只提取代码类 token。"""
    from backend.evaluator import _query_code_tokens

    assert "X018-5" in _query_code_tokens("X018-5地址")
    assert "DGN700" in _query_code_tokens("检查DGN700和DGN701")
    assert "ALM414" in _query_code_tokens("系统显示ALM414报警")
    assert _query_code_tokens("主轴停止") == []


def test_restore_metrics_from_ckpt_merges_ok_metrics():
    """断点恢复：只合并 status=ok 且有非空 metrics 的题目。"""
    from backend.evaluator import _restore_metrics_from_ckpt
    per_q = [{"id": "q1", "status": "ok"}, {"id": "q2", "status": "ok"}]
    saved = {"per_question": [
        {"id": "q1", "status": "ok", "metrics": {"faithfulness": 0.8}},
        {"id": "q2", "status": "failed"},                     # 失败题不补
        {"id": "q3", "status": "ok", "metrics": {}},          # 空 metrics 不补
    ]}
    _restore_metrics_from_ckpt(per_q, saved)
    assert per_q[0]["metrics"] == {"faithfulness": 0.8}
    assert "metrics" not in per_q[1]


def test_query_code_tokens_matches_component_codes():
    """2026-08-07：代码 token 正则覆盖元件代号/参数号/地址分隔写法。

    auto_013 的 Q11/Q12/VS11/DB11/JR11/F11/F12 此前全部漏识别，
    导致“相关路径前置”失效；P145/PRM 等参数号也应识别。
    """
    from backend.evaluator import _query_code_tokens

    q = "检查熔断器F11、F12熔断，检查VS11、DB11均正常，断开JR11后发现Q11、Q12短路"
    tokens = _query_code_tokens(q)
    for t in ("F11", "F12", "VS11", "DB11", "JR11", "Q11", "Q12"):
        assert t in tokens, f"{t} 应被识别为元件代号"
    assert "P145" in _query_code_tokens("参数P145应设定为什么值")
    assert "PRM2023" in _query_code_tokens("检查PRM2023参数")
    assert "G008.4" in _query_code_tokens("G008.4信号为0")
    assert "G008.0" in _query_code_tokens("G008#0信号")
    assert "端子36" in _query_code_tokens("端子36用于检测哪一路内部电压")
    assert "*ESP" in _query_code_tokens("急停信号（*ESP）的输入地址")
    assert "OVL位" in _query_code_tokens("DGN700的OVL位表示什么含义")
    assert "7段数码管" in _query_code_tokens("7段数码管显示1代表什么故障")
    assert _query_code_tokens("主轴停止") == []


def test_query_code_tokens_extracts_alarm_numbers():
    """2026-09-06（修复②）：问句"报警+数字 / 数字+报警"提取纯数字报警号。

    GSK983 报警号实体为 "GSK983报警405" 三位命名，用户口语写"报警405/405报警"；
    _QUERY_CODE_RE 只匹配字母开头代号，纯数字 405 需要追加才能参与路径评分。
    """
    from backend.evaluator import _query_code_tokens
    assert "405" in _query_code_tokens("报警405怎么处理")
    assert "405" in _query_code_tokens("GSK983报警405怎么处理")
    assert "405" in _query_code_tokens("405报警什么含义")
    assert "212" in _query_code_tokens("GSK983报警212是什么原因")
    # 型号内的数字不得误提取（GSK983 的 983 不是报警号）
    toks = _query_code_tokens("GSK983机床超程报警如何复位")
    assert "983" not in toks, "型号内数字不应被当成报警号"
    # 1 位数字不追加（避免泛化加分稀释），无报警语境不触发
    assert _query_code_tokens("报警2是什么") == []
    assert _query_code_tokens("主轴停止") == []
    # 原有字母代号提取不受影响
    assert "SK983" in _query_code_tokens("GSK983报警405怎么处理")


def test_cot_prompt_forbids_cross_alarm_reference():
    """2026-09-06（修复④）：答案生成约束禁止跨报警号/故障代码套用。"""
    tpl = RAGASEvaluator._COT_PROMPT_TEMPLATE
    assert "其他报警号" in tpl or "其它报警号" in tpl, \
        "prompt 必须禁止把其它报警号原因/方法套到所问报警号上"
    assert "不得互相套用" in tpl or "不得互相引用" in tpl or "张冠李戴" in tpl


def test_is_strong_code_token_distinguishes_weak_model_tokens():
    """2026-09-06（修复③）：型号弱代码（SK983）无鉴别力，报警号/地址为强代码。"""
    from backend.evaluator import _is_strong_code_token
    assert _is_strong_code_token("405"), "报警号纯数字是强代码"
    assert _is_strong_code_token("ALM414"), "ALM 报警代号是强代码"
    assert _is_strong_code_token("DGN700"), "DGN 诊断号是强代码"
    assert _is_strong_code_token("端子36"), "端子号是强代码"
    assert not _is_strong_code_token("SK983"), "型号子串对图内路径无鉴别力"
    assert not _is_strong_code_token("0MC"), "纯型号不含报警/地址语义"
    assert not _is_strong_code_token(""), "空 token 非强代码"


def test_cot_prompt_constrains_answer_generation():
    """2026-08-07：答案生成模板必须约束外推、标注依据并限制篇幅。

    低 faithfulness 根因是长 COT 答案补充了大量知识库外声明；模板需强制
    “只输出知识中明确出现的结论 + 200 字以内”，且代码/地址类问题直接给值。
    """
    from backend.evaluator import RAGASEvaluator

    tpl = RAGASEvaluator._COT_PROMPT_TEMPLATE
    assert "只输出【故障诊断知识】中明确出现的结论" in tpl
    assert "禁止补充外部经验、额外排查步骤或推测性诊断" in tpl
    assert "200 字以内" in tpl
    assert "不要强行套故障三段式" in tpl
    prompt = tpl.format(context="X018-5", question="X轴参考点减速信号输入地址是什么？")
    assert "X轴参考点减速信号输入地址是什么？" in prompt


def test_collect_graph_contexts_renders_true_direction():
    """P0-2：路径按真实方向渲染，反向边不得伪装成正向。"""
    from backend.evaluator import _collect_graph_contexts

    class _Manual:
        def query(self, q, **kwargs):
            return [(["A", "B", "C"], ["r1", "r2"], ["+", "-"])]

    class _Comm:
        def query(self, q, **kwargs):
            return [], []

    ctxs = _collect_graph_contexts("q", _Manual(), _Comm(),
                                   max_ctx=12, min_path_slots=4)
    assert ctxs[0] == "A--[r1]-->B<--[r2]--C", ctxs[0]


def test_collect_graph_contexts_appends_path_evidence():
    """P0-3：路径节点绑定的来源 chunk 必须回填到评分 context。"""
    from backend.evaluator import _collect_graph_contexts

    class _Manual:
        def query(self, q, **kwargs):
            return [(["A", "B"], ["可能原因"], ["+"])]

    class _Comm:
        def query(self, q, **kwargs):
            return [], []

    entity_chunk_map = {"A": {0}, "B": {0}}
    chunk_texts = ["X018-5 原始证据原文，用于覆盖 ground truth。"]
    ctxs = _collect_graph_contexts(
        "X018-5地址", _Manual(), _Comm(),
        max_ctx=12, min_path_slots=4,
        entity_chunk_map=entity_chunk_map, chunk_texts=chunk_texts,
    )
    assert "[原文依据]" in ctxs[0]
    assert "X018-5 原始证据原文" in ctxs[0]

def test_backfill_type_field_fills_missing():
    """回填函数：按 id 从测试集补 type，已有 type 不覆盖。"""
    from backend.evaluator import _backfill_type_field

    records = [
        {"id": "q1", "answer": "a"},                      # 缺 type → 回填
        {"id": "q2", "type": "", "answer": "b"},          # 空 type → 回填
        {"id": "q3", "type": "multi-hop", "answer": "c"}, # 已有 → 不动
        {"id": "q9", "answer": "d"},                      # 测试集无此 id → 不动
    ]
    test_items = [
        {"id": "q1", "type": "single-hop"},
        {"id": "q2", "type": "out-of-context"},
        {"id": "q3", "type": "multi-hop"},
    ]
    filled = _backfill_type_field(records, test_items)
    assert filled == 2
    assert records[0]["type"] == "single-hop"
    assert records[1]["type"] == "out-of-context"
    assert records[2]["type"] == "multi-hop"
    assert records[3].get("type") is None


def test_backfill_type_field_unknown_fallback():
    """测试集 id 存在但无 type 时回填 unknown。"""
    from backend.evaluator import _backfill_type_field

    records = [{"id": "q1", "answer": "a"}]
    test_items = [{"id": "q1"}]
    filled = _backfill_type_field(records, test_items)
    assert filled == 1
    assert records[0]["type"] == "unknown"
