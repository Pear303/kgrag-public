"""pipeline 模块单元测试。

覆盖 PipelineContext 初始化、各 stage 函数、PipelineRunner 阶段顺序。
使用 mock 替代所有外部依赖（Neo4j、文件 I/O、LLM、API 调用等）。
"""
import sys
from unittest.mock import MagicMock, patch, call
import pytest
import os

# Neo4j / pyvis 未安装时使用 mock 模块（纯单元测试不需要真实依赖）
for _mod_name in ("neo4j", "pyvis"):
    if _mod_name not in sys.modules:
        sys.modules[_mod_name] = MagicMock()
        sys.modules[f"{_mod_name}.network"] = MagicMock()

from pipeline.context import PipelineContext
from pipeline.runner import PipelineRunner
from pipeline.stages import (
    run_parse_stage,
    run_write_stage,
    run_extract_stage,
    run_community_stage,
    run_engine_setup_stage,
    run_verify_stage,
    run_visualize_stage,
    _build_compare_summary,
)
from backend.parser import Chunk


# ============================================================
# PipelineContext 初始化与默认值测试
# ============================================================

def test_context_default_initialization():
    """PipelineContext 默认值：所有字段应有合理默认值。"""
    ctx = PipelineContext()
    assert ctx.manual_path == ""
    assert ctx.source_name == ""
    assert ctx.chunks == []
    assert ctx.triples == []
    assert ctx.triples_with_prov == []
    assert ctx.write_count == 0
    assert ctx.vector_mode == ""
    assert ctx.llm is None
    assert ctx.embed is None
    assert ctx.store is None
    assert ctx.driver is None
    assert ctx.all_llm_triples == []
    assert ctx.llm_compare_summary is None
    assert ctx.communities == []
    assert ctx.community_count == 0
    assert ctx.manual_eng is None
    assert ctx.comm_eng is None
    assert ctx.auto_eng is None
    assert ctx.router is None
    assert ctx.micro_rows == []
    assert ctx.viz_output_path == ""


def test_context_with_fields_set():
    """PipelineContext 可在构造时通过关键字参数赋值。"""
    chunks = [Chunk(chunk_id=0, text="测试文本", source="test.md")]
    ctx = PipelineContext(
        manual_path="/path/to/manual.md",
        source_name="manual.md",
        chunks=chunks,
        write_count=3,
        vector_mode="native",
    )
    assert ctx.manual_path == "/path/to/manual.md"
    assert ctx.source_name == "manual.md"
    assert ctx.chunks is chunks
    assert ctx.write_count == 3
    assert ctx.vector_mode == "native"


def test_context_mutation_across_stages():
    """PipelineContext 应支持在各 stage 之间通过变异传递数据。"""
    ctx = PipelineContext()
    ctx.manual_path = "/tmp/test.md"
    ctx.chunks = [Chunk(chunk_id=0, text="chunk0", source="/tmp/test.md")]
    ctx.triples = [("A", "可能原因", "B")]
    ctx.write_count = 1
    assert ctx.manual_path == "/tmp/test.md"
    assert len(ctx.chunks) == 1
    assert len(ctx.triples) == 1
    assert ctx.write_count == 1


# ============================================================
# Stage 1: run_parse_stage — 解析手册
# ============================================================

def test_run_parse_stage_outputs(mock_manual_path):
    """Stage 1 应填充 source_name、chunks、triples_with_prov、triples。"""
    ctx = PipelineContext(manual_path=mock_manual_path)

    fake_chunks = [Chunk(chunk_id=0, text="chunk A", source=mock_manual_path)]
    fake_prov = [
        {"s": "温度过高", "r": "可能原因", "o": "润滑不足", "source_manual": "test.md", "source_line": 5},
        {"s": "异响",   "r": "表现为",   "o": "密封失效", "source_manual": "test.md", "source_line": 10},
    ]

    with (
        patch("pipeline.stages.parser.parse_manual_chunks", return_value=fake_chunks),
        patch("pipeline.stages.parser.parse_manual_with_provenance", return_value=fake_prov),
    ):
        ctx = run_parse_stage(ctx)

    assert ctx.source_name == os.path.basename(mock_manual_path)
    assert ctx.chunks is fake_chunks
    assert ctx.triples_with_prov is fake_prov
    assert ctx.triples == [("温度过高", "可能原因", "润滑不足"), ("异响", "表现为", "密封失效")]


def test_run_parse_stage_empty_manual(mock_manual_path):
    """空手册：Stage 1 应产出空 chunks 和空 triples。"""
    ctx = PipelineContext(manual_path=mock_manual_path)

    with (
        patch("pipeline.stages.parser.parse_manual_chunks", return_value=[]),
        patch("pipeline.stages.parser.parse_manual_with_provenance", return_value=[]),
    ):
        ctx = run_parse_stage(ctx)

    assert ctx.source_name == os.path.basename(mock_manual_path)
    assert ctx.chunks == []
    assert ctx.triples == []
    assert ctx.triples_with_prov == []


# ============================================================
# Stage 2: run_write_stage — 写入 Neo4j
# ============================================================

def test_run_write_stage_calls_seed(mock_manual_path):
    """Stage 2 应遍历每条三元组调用 parser.seed_triples。"""
    prov = [
        {"s": "A", "r": "可能原因", "o": "B", "source_manual": "test.md", "source_line": 1},
        {"s": "C", "r": "表现为",   "o": "D", "source_manual": "test.md", "source_line": 2},
        {"s": "E", "r": "解决方法", "o": "F", "source_manual": "test.md", "source_line": 3},
    ]
    ctx = PipelineContext(manual_path=mock_manual_path, source_name="test.md", triples_with_prov=prov)

    with patch("pipeline.stages.parser.seed_triples") as mock_seed:
        ctx = run_write_stage(ctx)

    assert mock_seed.call_count == 3
    assert ctx.write_count == 3


def test_run_write_stage_no_triples(mock_manual_path):
    """无三元组时 Seed 不应被调用，write_count 为 0。"""
    ctx = PipelineContext(manual_path=mock_manual_path, source_name="test.md", triples_with_prov=[])

    with patch("pipeline.stages.parser.seed_triples") as mock_seed:
        ctx = run_write_stage(ctx)

    mock_seed.assert_not_called()
    assert ctx.write_count == 0


# ============================================================
# Stage 3: run_extract_stage — 可选 LLM 抽取
# ============================================================

def test_run_extract_stage_skipped_when_false():
    """build_llm_graph=False 时应直接返回，不执行任何抽取。"""
    ctx = PipelineContext()
    result = run_extract_stage(ctx, build_llm_graph=False, use_online=False)
    assert result is ctx


def test_run_extract_stage_with_online_model(mock_manual_path):
    """use_online=True 时应调用 extract_triples_online。"""
    chunks = [
        Chunk(chunk_id=0, text="content 0", source=mock_manual_path),
        Chunk(chunk_id=1, text="content 1", source=mock_manual_path),
    ]
    mock_store = MagicMock()
    ctx = PipelineContext(
        manual_path=mock_manual_path,
        chunks=chunks,
        store=mock_store,
        triples=[("X", "可能原因", "Y")],
    )

    fake_triples = [("A", "包含", "B"), ("C", "表现为", "D")]

    with patch("backend.extractor.extract_triples_online", return_value=fake_triples):
        ctx = run_extract_stage(ctx, build_llm_graph=True, use_online=True)

    # store.upsert_auto_triples 每个 chunk 各调一次
    assert mock_store.upsert_auto_triples.call_count == 2
    assert ctx.all_llm_triples == fake_triples * 2
    assert ctx.llm_compare_summary is not None
    assert "manual_count" in ctx.llm_compare_summary
    assert "llm_count" in ctx.llm_compare_summary


# ============================================================
# _build_compare_summary 测试
# ============================================================

def test_build_compare_summary_counts():
    """验证 manual vs LLM 三元组对比摘要的计数与结构。"""
    manual = [("A", "可能原因", "B"), ("C", "表现为", "D"), ("E", "解决方法", "F")]
    llm = [("A", "可能原因", "B"), ("G", "包含", "H")]

    summary = _build_compare_summary(manual, llm)

    assert summary["manual_count"] == 3
    assert summary["llm_count"] == 2
    assert summary["manual_relations"] == {"可能原因": 1, "表现为": 1, "解决方法": 1}
    assert summary["llm_relations"] == {"可能原因": 1, "包含": 1}
    assert summary["overlap_entities"] == sorted(["A", "B"])
    assert summary["only_manual_entities"] == sorted(["C", "D", "E", "F"])
    assert summary["only_llm_entities"] == sorted(["G", "H"])


def test_build_compare_summary_no_overlap():
    """二者无交集时 overlap 应为空。"""
    manual = [("A", "可能原因", "B")]
    llm = [("X", "包含", "Y")]

    summary = _build_compare_summary(manual, llm)

    assert summary["overlap_entities"] == []
    assert "A" in summary["only_manual_entities"]
    assert "B" in summary["only_manual_entities"]
    assert "X" in summary["only_llm_entities"]
    assert "Y" in summary["only_llm_entities"]


def test_build_compare_summary_empty_llm():
    """LLM 三元组为空时不应报错。"""
    manual = [("A", "可能原因", "B")]
    llm = []

    summary = _build_compare_summary(manual, llm)

    assert summary["manual_count"] == 1
    assert summary["llm_count"] == 0
    assert summary["llm_relations"] == {}


# ============================================================
# Stage 4: run_community_stage
# ============================================================

def test_run_community_stage_builds_communities():
    """Stage 4 应调用 build_communities 并设置 communities / community_count。"""
    chunks = [Chunk(chunk_id=0, text="chunk text", source="test.md")]
    mock_driver = MagicMock()
    mock_session = MagicMock()
    mock_driver.session.return_value.__enter__.return_value = mock_session
    # 模拟 Neo4j 查询返回
    mock_session.run.side_effect = [
        [{"name": "Entity_A"}, {"name": "Entity_B"}],
        [],
        [],
        [{"s": "Entity_A", "r": "REL_1", "o": "Entity_B"}],
    ]

    ctx = PipelineContext(manual_path="/tmp/test.md", chunks=chunks, driver=mock_driver)

    fake_communities = [
        {"id": 0, "nodes": {"A", "B"}, "chunk_ids": [0], "raw_context": "ctx", "description": "desc"}
    ]
    with patch("pipeline.stages.build_communities", return_value=fake_communities):
        ctx = run_community_stage(ctx)

    assert ctx.communities is fake_communities
    assert ctx.community_count == 1
    assert ctx.driver is mock_driver


def test_run_community_stage_no_driver_falls_back():
    """ctx.driver=None 时应在内部获取 driver。"""
    chunks = [Chunk(chunk_id=0, text="text", source="test.md")]
    mock_driver = MagicMock()
    mock_session = MagicMock()
    mock_driver.session.return_value.__enter__.return_value = mock_session
    mock_session.run.side_effect = [[], [], [], []]

    ctx = PipelineContext(manual_path="/tmp/test.md", chunks=chunks, driver=None)

    with (
        patch("pipeline.stages.get_raw_driver", return_value=mock_driver),
        patch("pipeline.stages.build_communities", return_value=[]),
    ):
        ctx = run_community_stage(ctx)

    assert ctx.driver is mock_driver


# ============================================================
# Stage 5: run_engine_setup_stage
# ============================================================

def test_run_engine_setup_stage_creates_engines():
    """Stage 5 应初始化 ManualGraph / AutoGraph / Community 引擎。"""
    ctx = PipelineContext(communities=[{"id": 0, "nodes": set()}])

    with (
        patch("pipeline.stages.get_property_graph_store") as mock_get_store,
        patch("pipeline.stages.ManualGraphQueryEngine") as MockManual,
        patch("pipeline.stages.AutoGraphQueryEngine") as MockAuto,
        patch("pipeline.stages.CommunityQueryEngine") as MockComm,
        patch("backend.router.build_router", return_value=MagicMock()),
    ):
        mock_store = mock_get_store.return_value
        mock_manual = MockManual.return_value
        mock_auto = MockAuto.return_value
        mock_comm = MockComm.return_value

        ctx = run_engine_setup_stage(ctx, source_filter=None)

    assert ctx.manual_eng is mock_manual
    assert ctx.auto_eng is mock_auto
    assert ctx.comm_eng is mock_comm
    assert ctx.store is mock_store
    assert ctx.router is not None


def test_run_engine_setup_stage_community_failure_graceful():
    """Community 引擎初始化失败时设置 comm_eng=None，不阻断其他引擎。"""
    ctx = PipelineContext(communities=[{"id": 0, "nodes": set()}])

    with (
        patch("pipeline.stages.get_property_graph_store"),
        patch("pipeline.stages.ManualGraphQueryEngine"),
        patch("pipeline.stages.AutoGraphQueryEngine"),
        patch("pipeline.stages.CommunityQueryEngine", side_effect=RuntimeError("社区初始化失败")),
        patch("backend.router.build_router", return_value=MagicMock()),
    ):
        ctx = run_engine_setup_stage(ctx)

    assert ctx.manual_eng is not None
    assert ctx.auto_eng is not None
    assert ctx.comm_eng is None  # 降级
    assert ctx.router is not None


# ============================================================
# Stage 6: run_verify_stage
# ============================================================

def test_run_verify_stage_runs_manual_query():
    """Stage 6 应调用 manual_eng.query 并存储 micro_rows。"""
    mock_manual = MagicMock()
    fake_rows = [
        (["温度过高", "润滑不足"], ["可能原因"]),
    ]
    mock_manual.query.return_value = fake_rows
    ctx = PipelineContext(manual_eng=mock_manual, comm_eng=None, router=None)

    ctx = run_verify_stage(ctx)

    mock_manual.query.assert_called_once()
    assert ctx.micro_rows is fake_rows


def test_run_verify_stage_with_community(mock_manual_path):
    """有社区引擎时应执行宏观查询。"""
    mock_manual = MagicMock()
    mock_manual.query.return_value = []
    mock_comm = MagicMock()
    mock_comm.query.return_value = ([{"id": 1}], ["ctx1"])
    ctx = PipelineContext(manual_eng=mock_manual, comm_eng=mock_comm, router=None)

    ctx = run_verify_stage(ctx)

    mock_comm.query.assert_called_once()


def test_run_verify_stage_with_router(mock_manual_path):
    """有路由器时执行智能路由查询。"""
    mock_manual = MagicMock()
    mock_manual.query.return_value = []
    mock_comm = MagicMock()
    mock_comm.query.return_value = ([], [])
    mock_router = MagicMock()
    mock_router.query.return_value = "community"
    ctx = PipelineContext(manual_eng=mock_manual, comm_eng=mock_comm, router=mock_router)

    ctx = run_verify_stage(ctx)

    mock_router.query.assert_called_once()


# ============================================================
# Stage 7: run_visualize_stage
# ============================================================

def test_run_visualize_stage_calls_render_micro():
    """Stage 7 应调用 render_micro 并设置 viz_output_path。"""
    fake_rows = [(("A", "B"), ("R1",))]
    ctx = PipelineContext(micro_rows=fake_rows)

    with (
        patch("pipeline.stages.render_micro") as mock_render,
        patch("pipeline.stages.PROJECT_ROOT", "/fake/project"),
    ):
        ctx = run_visualize_stage(ctx)

    mock_render.assert_called_once_with(fake_rows, os.path.join("/fake/project", "knowledge_graph.html"))
    assert ctx.viz_output_path == os.path.join("/fake/project", "knowledge_graph.html")


# ============================================================
# PipelineRunner — 阶段顺序验证
# ============================================================

def test_runner_stage_ordering():
    """验证 PipelineRunner.run() 按正确顺序调用所有阶段函数。"""
    runner = PipelineRunner()

    mock_args = MagicMock()
    mock_args.manual = "/tmp/manual.md"
    mock_args.build_llm_graph = True
    mock_args.use_online = False
    mock_args.source_filter = None
    mock_args.build_vector_rag = False  # 不触发可选阶段
    mock_args.community_summaries = False
    mock_args.calibrate_prompt = False   # 提示词校准旗（C1），MagicMock 需显式置假
    mock_args.use_calibrated_prompt = False
    mock_args.force_calibrate = False
    mock_args.dry_run = False

    with (
        patch("pipeline.runner.resolve_vector_mode", return_value="memory"),
        patch("pipeline.runner.get_llm", return_value=MagicMock()),
        patch("pipeline.runner.get_embed_model", return_value=MagicMock()),
        patch("pipeline.runner.get_property_graph_store", return_value=MagicMock()),
        patch("pipeline.runner.get_raw_driver", return_value=MagicMock()),
        patch("pipeline.runner.run_parse_stage") as mock_parse,
        patch("pipeline.runner.run_write_stage") as mock_write,
        patch("pipeline.runner.run_extract_stage") as mock_extract,
        patch("pipeline.runner.run_community_stage") as mock_community,
        patch("pipeline.runner.run_engine_setup_stage") as mock_engine,
        patch("pipeline.runner.run_verify_stage") as mock_verify,
        patch("pipeline.runner.run_visualize_stage") as mock_viz,
    ):
        runner.run(mock_args)

    # 确保 setup 完成后各阶段按正确顺序调用
    mock_parse.assert_called_once()
    mock_write.assert_called_once()
    mock_extract.assert_called_once()
    mock_community.assert_called_once()
    mock_engine.assert_called_once()
    mock_verify.assert_called_once()
    mock_viz.assert_called_once()


def test_runner_skips_llm_extract_when_disabled():
    """build_llm_graph=False 时不应调用 Stage 3。"""
    runner = PipelineRunner()

    mock_args = MagicMock()
    mock_args.manual = "/tmp/manual.md"
    mock_args.build_llm_graph = False
    mock_args.use_online = False
    mock_args.source_filter = None
    mock_args.build_vector_rag = False
    mock_args.community_summaries = False
    mock_args.calibrate_prompt = False
    mock_args.use_calibrated_prompt = False
    mock_args.force_calibrate = False
    mock_args.dry_run = False

    with (
        patch("pipeline.runner.resolve_vector_mode", return_value="memory"),
        patch("pipeline.runner.get_llm", return_value=MagicMock()),
        patch("pipeline.runner.get_embed_model", return_value=MagicMock()),
        patch("pipeline.runner.get_property_graph_store", return_value=MagicMock()),
        patch("pipeline.runner.get_raw_driver", return_value=MagicMock()),
        patch("pipeline.runner.run_parse_stage"),
        patch("pipeline.runner.run_write_stage"),
        patch("pipeline.runner.run_extract_stage") as mock_extract,
        patch("pipeline.runner.run_community_stage"),
        patch("pipeline.runner.run_engine_setup_stage"),
        patch("pipeline.runner.run_verify_stage"),
        patch("pipeline.runner.run_visualize_stage"),
    ):
        runner.run(mock_args)

    mock_extract.assert_not_called()


def test_runner_sets_context_inputs():
    """Runner.run() 应从 args 提取 manual_path，并设置 setup 属性。"""
    runner = PipelineRunner()

    mock_args = MagicMock()
    mock_args.manual = "/tmp/manual.md"
    mock_args.build_llm_graph = False
    mock_args.use_online = False
    mock_args.source_filter = None
    mock_args.build_vector_rag = False
    mock_args.community_summaries = False
    mock_args.calibrate_prompt = False
    mock_args.use_calibrated_prompt = False
    mock_args.force_calibrate = False
    mock_args.dry_run = False

    with (
        patch("pipeline.runner.resolve_vector_mode", return_value="native"),
        patch("pipeline.runner.get_llm", return_value="fake_llm"),
        patch("pipeline.runner.get_embed_model", return_value="fake_embed"),
        patch("pipeline.runner.get_property_graph_store", return_value="fake_store"),
        patch("pipeline.runner.get_raw_driver", return_value="fake_driver"),
        patch("pipeline.runner.run_parse_stage", side_effect=lambda ctx: ctx),
        patch("pipeline.runner.run_write_stage", side_effect=lambda ctx: ctx),
        patch("pipeline.runner.run_community_stage", side_effect=lambda ctx, **kw: ctx),
        patch("pipeline.runner.run_engine_setup_stage", side_effect=lambda ctx, **kw: ctx),
        patch("pipeline.runner.run_verify_stage", side_effect=lambda ctx: ctx),
        patch("pipeline.runner.run_visualize_stage", side_effect=lambda ctx: ctx),
    ):
        runner.run(mock_args)

    assert runner.ctx.manual_path == "/tmp/manual.md"
    assert runner.ctx.vector_mode == "native"
    assert runner.ctx.llm == "fake_llm"
    assert runner.ctx.embed == "fake_embed"
    assert runner.ctx.store == "fake_store"
    assert runner.ctx.driver == "fake_driver"


# ============================================================
# conftest fixtures（仅本模块使用）
# ============================================================

@pytest.fixture
def mock_manual_path():
    """返回一个用于测试的虚拟手册路径。"""
    return "/tmp/test_manual.md"
