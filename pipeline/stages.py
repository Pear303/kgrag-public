"""管道阶段函数：每个阶段独立可调用、可测试。

阶段编号与文章章节对应：
  Stage 1: 解析手册 → Chunks + 人工规则三元组（文章 3.1/3.2）
  Stage 2: 写入 Neo4j 图存储（文章 3.2）
  Stage 3: 可选 LLM 三元组抽取（文章 3.3/3.4）
  Stage 4: 构建社区 Louvain + Chunk 绑定（文章 5.2~5.4）
  Stage 5: 设置查询引擎 Auto/Manual/Community（文章 4.1/4.2/5.5）
  Stage 6: 微观/宏观查询验证（文章 4.2/5.5）
  Stage 7: pyvis 可视化导出（文章 6.2）
"""
import os
from collections import Counter
from typing import Any, Optional

from config import (
    PROJECT_ROOT, LOCAL_LLM_BASE_URL, LOCAL_LLM_MODEL,
    get_llm, get_embed_model,
)
from backend import parser
from backend.graph_store import get_property_graph_store, get_raw_driver, resolve_vector_mode
from backend.community import build_communities
from backend.engines import (
    AutoGraphQueryEngine, ManualGraphQueryEngine, CommunityQueryEngine,
)
from frontend.graph_viz import render_micro

from pipeline.context import PipelineContext


# ============================================================
# Stage 1: 解析手册 → Chunks + 人工规则三元组
# ============================================================

def run_parse_stage(ctx: PipelineContext) -> PipelineContext:
    """Stage 1: 解析手册，产出 chunks 和带行号溯源的三元组。

    输入:
        ctx.manual_path — 手册文件路径

    输出 (写入 ctx):
        ctx.chunks — 段落切片列表
        ctx.triples_with_prov — 带溯源的三元组列表 [{"s","r","o","source_manual","source_line"}, ...]
        ctx.triples — 纯三元组 [(s, r, o), ...]
        ctx.source_name — 手册文件名（去路径）
    """
    print("=== 1. 解析手册 ===")
    ctx.source_name = os.path.basename(ctx.manual_path)
    ctx.chunks = parser.parse_manual_chunks(ctx.manual_path)
    ctx.triples_with_prov = parser.parse_manual_with_provenance(ctx.manual_path)
    ctx.triples = [(r["s"], r["r"], r["o"]) for r in ctx.triples_with_prov]
    print(f"  chunks={len(ctx.chunks)}, 人工三元组={len(ctx.triples)}")
    return ctx


# ============================================================
# Stage 2: 写入 Neo4j（人工图谱）
# ============================================================

def run_write_stage(ctx: PipelineContext) -> PipelineContext:
    """Stage 2: 将人工规则三元组写入 Neo4j :Entity 空间。

    输入:
        ctx.triples_with_prov — 带溯源的三元组
        ctx.source_name — 手册文件名

    输出 (写入 ctx):
        ctx.write_count — 写入的三元组条数
    """
    print("=== 2. 写入 Neo4j（人工图谱）===")
    for r in ctx.triples_with_prov:
        parser.seed_triples(
            None,
            [(r["s"], r["r"], r["o"])],
            source_manual=ctx.source_name,
            source_line=r["source_line"],
        )
    ctx.write_count = len(ctx.triples_with_prov)
    print(f"  已写入 {ctx.write_count} 条三元组（含行号溯源）")
    return ctx


# ============================================================
# Stage 3: 可选 LLM 三元组抽取
# ============================================================

def run_extract_stage(ctx: PipelineContext, build_llm_graph: bool, use_online: bool,
                      extraction_prompt: Optional[str] = None) -> PipelineContext:
    """Stage 3: 用 LLM 从非结构化文本自动抽取三元组（可选）。

    输入:
        ctx.chunks — 已解析的文本块
        ctx.store — Neo4j 图存储实例
        build_llm_graph — 是否执行 LLM 抽取
        use_online — 是否使用在线强模型
        extraction_prompt — 覆盖内置提示词（校准器合成的适配版，须含 {context} 槽）

    输出 (写入 ctx):
        ctx.all_llm_triples — 抽取到的所有三元组
        ctx.llm_compare_summary — 人工 vs LLM 对比摘要
    """
    if not build_llm_graph:
        return ctx

    print("\n=== 5b. LLM Auto Extract Triples ===")
    try:
        store = ctx.store or get_property_graph_store()
        ctx.store = store  # 确保后续阶段可用
        all_llm_triples: list = []
        auto_count = 0

        if use_online:
            from backend.extractor import extract_triples_online
            print("  Using online model: deepseek-v4-flash"
                  + (" + 校准适配版提示词" if extraction_prompt else ""))
            for idx, c in enumerate(ctx.chunks):
                chunk_triples = extract_triples_online(c.text, prompt=extraction_prompt)
                if chunk_triples:
                    store.upsert_auto_triples(chunk_triples)
                    all_llm_triples.extend(chunk_triples)
                    auto_count += len(chunk_triples)
                    print(f"  chunk {idx+1}/{len(ctx.chunks)}: {len(chunk_triples)} triples")
        else:
            from backend.extractor import extract_triples_with_llm
            print(f"  Using local model: {LOCAL_LLM_MODEL}"
                  + (" + 校准适配版提示词" if extraction_prompt else ""))
            for idx, c in enumerate(ctx.chunks):
                chunk_triples = extract_triples_with_llm(
                    c.text, model=LOCAL_LLM_MODEL, base_url=LOCAL_LLM_BASE_URL,
                    prompt=extraction_prompt,
                )
                if chunk_triples:
                    store.upsert_auto_triples(chunk_triples)
                    all_llm_triples.extend(chunk_triples)
                    auto_count += len(chunk_triples)
                    print(f"  chunk {idx+1}/{len(ctx.chunks)}: {len(chunk_triples)} triples")

        print(f"  LLM extracted triples total: {auto_count}")
        for t in all_llm_triples[:5]:
            print(f"    {t}")
        if len(all_llm_triples) > 5:
            print(f"    ... 共 {len(all_llm_triples)} 条")
        print(f"  已写入 Neo4j :__Entity__ 空间")
        ctx.all_llm_triples = all_llm_triples

        # 对比摘要
        ctx.llm_compare_summary = _build_compare_summary(ctx.triples, all_llm_triples)

    except Exception as e:
        import traceback
        print(f"  LLM 抽取失败: {e}")
        traceback.print_exc()
        ctx.llm_compare_summary = None

    return ctx


def _build_compare_summary(manual_triples: list, llm_triples: list) -> dict:
    """生成人工 vs LLM 三元组对比摘要。

    返回:
        {"manual_count": int, "llm_count": int,
         "manual_relations": dict, "llm_relations": dict,
         "only_manual_entities": list, "only_llm_entities": list,
         "overlap_entities": list}
    """
    print("\n  === 人工 vs LLM 对比 ===")
    print(f"  人工三元组: {len(manual_triples)} 条")
    print(f"  LLM 三元组: {len(llm_triples)} 条")

    manual_rels = Counter(t[1] for t in manual_triples)
    llm_rels = Counter(t[1] for t in llm_triples)
    all_rels = set(list(manual_rels.keys()) + list(llm_rels.keys()))
    print(f"\n  关系类型分布:")
    print(f"    {'关系':<20} {'人工':>8} {'LLM':>8}")
    for rel in sorted(all_rels):
        print(f"    {rel:<20} {manual_rels.get(rel, 0):>8} {llm_rels.get(rel, 0):>8}")

    manual_entities = set(t[0] for t in manual_triples) | set(t[2] for t in manual_triples)
    llm_entities = set(t[0] for t in llm_triples) | set(t[2] for t in llm_triples)
    overlap = manual_entities & llm_entities
    only_manual = manual_entities - llm_entities
    only_llm = llm_entities - manual_entities
    print(f"\n  实体重叠分析:")
    print(f"    仅人工: {len(only_manual)} 个")
    print(f"    仅 LLM: {len(only_llm)} 个")
    print(f"    重叠:   {len(overlap)} 个")

    return {
        "manual_count": len(manual_triples),
        "llm_count": len(llm_triples),
        "manual_relations": dict(manual_rels),
        "llm_relations": dict(llm_rels),
        "only_manual_entities": sorted(only_manual),
        "only_llm_entities": sorted(only_llm),
        "overlap_entities": sorted(overlap),
    }


# ============================================================
# Stage 4: 构建社区（Louvain + Chunk 绑定 + 去重）
# ============================================================

def run_community_stage(ctx: PipelineContext, generate_summaries: bool = False) -> PipelineContext:
    """Stage 4: 构建社区 — Louvain 聚类 + Chunk 绑定 + 去重。

    输入:
        ctx.chunks — 文本块
        ctx.embed — embedding 模型
        ctx.driver — Neo4j 驱动（若未初始化则自动获取）
        ctx.store — Neo4j 图存储实例
        generate_summaries — 是否生成 LLM 社区摘要

    输出 (写入 ctx):
        ctx.communities — 社区列表 [{"id","nodes","chunk_ids","raw_context","description","description_embedding"}, ...]
        ctx.community_count — 社区数量
        ctx.driver — Neo4j 驱动（确保后续可用）
    """
    print("=== 4. 构建社区（Louvain + Chunk 绑定）===")
    driver = ctx.driver or get_raw_driver()
    ctx.driver = driver  # 确保后续阶段可用

    with driver.session() as s:
        entity_nodes = [(r["name"], ["Entity"]) for r in s.run("MATCH (n:Entity) WHERE n.name IS NOT NULL RETURN n.name AS name")]
        llm_nodes = [(r["id"], ["__Entity__"]) for r in s.run("MATCH (n:__Entity__) WHERE n.id IS NOT NULL RETURN n.id AS id")]
        v2_nodes = [(r["id"], ["Entity"]) for r in s.run("MATCH (n:Entity) WHERE n.id IS NOT NULL RETURN n.id AS id")]
        nodes = entity_nodes + llm_nodes + v2_nodes
        edges = [(r["s"], r["r"], r["o"]) for r in s.run("""
            MATCH (n:Entity)-[r:REL]->(m:Entity)
            WHERE n.name IS NOT NULL AND m.name IS NOT NULL
            RETURN n.name AS s, r.type AS r, m.name AS o
            UNION ALL
            MATCH (n:__Entity__)-[r:REL]->(m:__Entity__)
            WHERE n.id IS NOT NULL AND m.id IS NOT NULL
            RETURN n.id AS s, r.type AS r, m.id AS o
            UNION ALL
            MATCH (n:Entity)-[r:REL]->(m:Entity)
            WHERE n.id IS NOT NULL AND m.id IS NOT NULL
            RETURN n.id AS s, r.type AS r, m.id AS o
        """)]

    chunk_texts = [c.text for c in ctx.chunks]
    ctx.communities = build_communities(
        nodes, edges, chunk_texts, ctx.embed,
        generate_summaries=generate_summaries,
        graph_store=ctx.store if generate_summaries else None,
    )
    ctx.community_count = len(ctx.communities)
    print(f"  社区数={ctx.community_count}")
    return ctx


# ============================================================
# Stage 5: 设置查询引擎
# ============================================================

def run_engine_setup_stage(ctx: PipelineContext, source_filter: Optional[str] = None) -> PipelineContext:
    """Stage 5: 初始化 AutoGraph / ManualGraph / Community 查询引擎及路由器。

    输入:
        ctx.store — Neo4j 图存储
        ctx.embed — embedding 模型
        ctx.llm — LLM 模型（路由器需要）
        ctx.communities — 社区数据
        source_filter — 可选：限制 KG 查询到指定手册

    输出 (写入 ctx):
        ctx.manual_eng — ManualGraphQueryEngine 实例
        ctx.comm_eng — CommunityQueryEngine 实例（可能为 None）
        ctx.auto_eng — AutoGraphQueryEngine 实例
        ctx.router — 路由器实例（可能为 None）
    """
    store = ctx.store or get_property_graph_store()
    ctx.store = store  # 确保后续可用

    # ManualGraph 引擎
    ctx.manual_eng = ManualGraphQueryEngine(store, ctx.embed, source_manual=source_filter)  # type: ignore[arg-type]

    # AutoGraph 引擎
    ctx.auto_eng = AutoGraphQueryEngine(store, ctx.embed)

    # Community 引擎
    try:
        ctx.comm_eng = CommunityQueryEngine(store, ctx.embed, ctx.communities)
    except Exception as e:
        print(f"  社区引擎初始化失败（降级为仅 manual 引擎）: {e}")
        ctx.comm_eng = None

    # 智能路由器
    try:
        from backend.router import build_router
        ctx.router = build_router(ctx.llm, ctx.auto_eng, ctx.comm_eng)
    except Exception as e:
        print(f"  路由器初始化失败（不影响主管道）: {e}")
        ctx.router = None

    return ctx


# ============================================================
# Stage 6: 微观 / 宏观查询验证
# ============================================================

def run_verify_stage(ctx: PipelineContext) -> PipelineContext:
    """Stage 6: 运行微观和宏观查询验证，展示智能路由选择。

    输入:
        ctx.manual_eng — ManualGraph 引擎
        ctx.comm_eng — Community 引擎
        ctx.router — 路由器

    输出 (写入 ctx):
        ctx.micro_rows — 微观查询路径结果
    """
    print("=== 5. 微观查询验证 ===")
    rows = ctx.manual_eng.query("轴承座温度升高怎么处理")
    for names, rels in rows:
        chain = names[0]
        for i, rel in enumerate(rels):
            chain += f" --[{rel}]--> {names[i + 1]}"
        print("  ", chain)
    ctx.micro_rows = rows

    print("=== 6. 宏观查询验证 ===")
    if ctx.comm_eng is not None:
        matched, ctx_ = ctx.comm_eng.query("总结一下密封系统的维护建议")
        print(f"  命中社区 {len(matched)} 个，召回 chunk {len(ctx_)} 段")
    else:
        print("  社区引擎不可用，跳过宏观查询")

    if ctx.router is not None:
        print("\n=== 7. 智能路由选择演示 ===")
        choice = ctx.router.query("轴承座温度升高怎么处理")
        print(f"  路由选择: {choice}")

    return ctx


# ============================================================
# Stage 7: pyvis 可视化导出
# ============================================================

def run_visualize_stage(ctx: PipelineContext) -> PipelineContext:
    """Stage 7: 渲染微观多跳路径为 pyvis HTML。

    输入:
        ctx.micro_rows — 微观查询结果

    输出 (写入 ctx):
        ctx.viz_output_path — 可视化 HTML 文件路径
    """
    out = os.path.join(PROJECT_ROOT, "knowledge_graph.html")
    render_micro(ctx.micro_rows, out)
    ctx.viz_output_path = out
    print(f"  微观图谱已生成: {out}")
    return ctx
