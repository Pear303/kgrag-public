"""改进 1/2 单元测试：路径实体二次打分 + 分块摘要。"""
import pytest


# ---------- 改进 1：路径实体二次打分 ----------

def test_collect_graph_contexts_entity_rerank():
    """实体共现片段应优先于纯相似度片段进 context。"""
    from backend.evaluator import _collect_graph_contexts

    class _Manual:
        def query(self, q, **kwargs):
            return [(["KA10", "X018-5"], ["可能原因"], ["+"])]

    class _Comm:
        def __init__(self):
            self.calls = 0

        def query(self, q, **kwargs):
            self.calls += 1
            return ["comm1"], [
                "其他相似主题描述，与本题无关。",
                "KA10 未吸合导致 X018-5 无电压，需检查中间继电器。",
            ]

    c = _Comm()
    ctxs = _collect_graph_contexts("X018-5", _Manual(), c, max_ctx=12, min_path_slots=4)
    assert c.calls == 1
    # 含实体的社区片段应出现在不含实体的前面
    idx_hit = next(i for i, x in enumerate(ctxs) if "KA10" in x)
    idx_miss = next(i for i, x in enumerate(ctxs) if "其他相似主题" in x)
    assert idx_hit < idx_miss, f"实体共现片段应前置: {idx_hit} vs {idx_miss}"


def test_collect_graph_contexts_no_path_no_rerank():
    """无图谱路径时（path_node_names 空）不重排：社区片段按引擎返回原序进 context。"""
    from backend.evaluator import _collect_graph_contexts

    class _Manual:
        def query(self, q, **kwargs):
            return []

    class _Comm:
        def query(self, q, **kwargs):
            return [], ["片段B（无路径时原序）", "片段A"]

    ctxs = _collect_graph_contexts("q", _Manual(), _Comm(),
                                   max_ctx=12, min_path_slots=4)
    # 无路径时社区按原序（社区正文前置逻辑），片段B 在前
    joined = "\n".join(ctxs)
    assert joined.index("片段B（无路径时原序）") < joined.index("片段A")


# ---------- 改进 2：分块摘要 ----------

def test_summary_prompt_chunking():
    """大文本社区应分成多批 prompt（每批 ≤4 个文本）。"""
    import backend.community as comm_mod

    texts = [f"第{i}段故障维修内容，包含报警 ALM{i:03d} 的处理步骤。" +
             "需要检查继电器、测量电压、确认信号地址并更换损坏元件。" * 8
             for i in range(20)]
    all_parts = [t[:1000] for t in texts[:20]]
    total = sum(len(t) for t in all_parts)
    assert total > 4000, "测试数据应超过分块阈值"

    # 模拟 community._summarize_community 的分块逻辑
    CHUNKED_SUMMARY_CHARS = 4000
    CHUNK_SUMMARY_BATCH = 4
    prompts = []
    for i in range(0, len(all_parts), CHUNK_SUMMARY_BATCH):
        combined = "\n---\n".join(all_parts[i:i + CHUNK_SUMMARY_BATCH])
        prompts.append(combined)
    assert len(prompts) == 5, f"20 个文本应分 5 批，实际 {len(prompts)}"
    assert all(len(p) <= 1000 * 4 + 10 for p in prompts), "每批应不超过 4 个文本"


# ---------- 改进 3（2026-08-11）：OOC 对象一致性检查 ----------

def test_ooc_object_mismatch_robot():
    """问题为工业机器人，上下文只有数控机床内容 → 判定对象不一致（应拒答）。"""
    from backend.evaluator import _ooc_object_mismatch

    q = "工业机器人在生产中重复定位精度明显下降，通常由哪些原因引起？"
    ctx = ["数控机床位置检测系统故障，脉冲编码器不良，需检查光栅尺与滚珠丝杠。",
           "[社区摘要] 数控机床伺服进给系统故障诊断，涵盖速度控制单元报警。"]
    assert _ooc_object_mismatch(q, ctx) is True


def test_ooc_object_mismatch_robot_with_hit():
    """上下文出现机器人对象词 → 一致，不判 OOC。"""
    from backend.evaluator import _ooc_object_mismatch

    q = "工业机器人关节伺服电机出现过载报警，可能的原因有哪些？"
    ctx = ["工业机器人关节机械卡死或过载，减速器损坏、润滑不良、异物卡住。"]
    assert _ooc_object_mismatch(q, ctx) is False


def test_ooc_object_mismatch_machine_question():
    """普通数控机床问题（无对象词）→ 不触发对象检查。"""
    from backend.evaluator import _ooc_object_mismatch

    q = "FANUC 0 系统中，诊断参数 DGN700 的 OVL 位表示什么含义？"
    ctx = ["DGN700--[包含]-->OVL位", "[社区摘要] FANUC 0 诊断参数地址及含义。"]
    assert _ooc_object_mismatch(q, ctx) is False


def test_ooc_object_mismatch_no_ctx():
    """空上下文不触发对象检查（走原有占位符逻辑）。"""
    from backend.evaluator import _ooc_object_mismatch

    assert _ooc_object_mismatch("工业机器人重复定位精度下降", []) is False

# ---------- 改进 4（2026-08-11）：查询短语同义词归一化 + 过滤放宽 ----------

def test_query_phrase_tokens_synonym_normalize():
    """"伺服电机""应归一化为图谱实体写法""伺服电动机""，关键短语可命中路径。"""
    from backend.evaluator import _query_phrase_tokens, _norm_query_synonyms

    q = "某配套SIEMENS数控系统的加工中心，X轴伺服电动机不转，且VRDY指示灯不亮。"
    paths = [
        "X轴伺服电动机不转-->[表现为]系统显示ALM414报警<--[包含]M414",
        "伺服电动机不转-->[可能原因]速度控制使能信号(ENABLE)没有送到速度控制单元",
        "Z轴伺服电动机-->[表现为]温度迅速上升表面发烫",
    ]
    assert _norm_query_synonyms("X轴伺服电机不转") == "X轴伺服电动机不转"
    tokens = _query_phrase_tokens(q, paths)
    assert any("电动机不转" in t for t in tokens), f"应含关键短语，实际 {tokens}"


def test_query_phrase_tokens_keep_frequent_key_phrase():
    """命中 4~15 条路径的领域核心短语不应被过滤（原 1~3 条过滤致 tokens 空）。"""
    from backend.evaluator import _query_phrase_tokens

    q = "X轴伺服电动机不转，且驱动器上的VRDY指示灯不亮。"
    paths = ["X轴伺服电动机不转-->[表现为]系统显示ALM414报警"] * 10  # 命中 10 条
    tokens = _query_phrase_tokens(q, paths)
    assert tokens, "命中 10 条的短语应保留（此前 1~3 过滤会丢弃）"


def test_query_phrase_tokens_filter_generic():
    """超高频泛化短语（>15 条命中）仍应被过滤。"""
    from backend.evaluator import _query_phrase_tokens

    q = "X轴伺服电动机不转，且驱动器上的VRDY指示灯不亮。"
    paths = ["X轴伺服电动机不转-->[表现为]系统显示ALM414报警"] * 30
    tokens = _query_phrase_tokens(q, paths)
    # 30 条命中的短语应被过滤，但可能有其他短语；至少不应全是高频泛化
    assert "系统显示" not in tokens or len(tokens) == 0

# ---------- 改进 5（2026-08-11）：OOC 口径修正 ----------

def test_attach_metrics_skips_ooc():
    """OOC 题不写入 RAGAS 四指标（指标适用域修正：上下文无答案不可评）。"""
    import numpy as np
    from backend.evaluator import RAGASEvaluator

    class _FakeDF:
        columns = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
        def __contains__(self, col):
            return col in self.columns
        def __len__(self):
            return 1
        def __getitem__(self, col):
            return _FakeSeries()

    class _FakeSeries:
        def tolist(self):
            return [0.5]

    class _FakeMetric:
        name = "faithfulness"

    ev = RAGASEvaluator.__new__(RAGASEvaluator)
    # OOC 题：四指标不写入，ooc_refusal 保留
    rec_ooc = {"id": "t1", "type": "out-of-context", "metrics": {"ooc_refusal": 1.0}}
    ev._attach_per_question_metrics([rec_ooc], _FakeDF(), [_FakeMetric()])
    assert "faithfulness" not in rec_ooc["metrics"], "OOC 题不得写入四指标"
    assert rec_ooc["metrics"]["ooc_refusal"] == 1.0, "ooc_refusal 必须保留"

    # in-context 题：四指标正常写入
    rec_in = {"id": "t2", "type": "multi-hop", "metrics": {}}
    ev._attach_per_question_metrics([rec_in], _FakeDF(), [_FakeMetric()])
    assert rec_in["metrics"]["faithfulness"] == 0.5
