"""构建 pipeline：把一本手册从纯文本 md 构建为可查询的 KG-RAG 资产。

8 步（可断点续跑）：
  0 文档解析（由 document_loader 完成，本模块从 md 开始）
  1 chunk 化
  2 LLM 三元组抽取（在线 API，断点 .done）
  3 Neo4j 导入（:Entity + name + source_manual）
  4 embedding 回填（本地 Ollama）
  5 社区检测（Louvain，本地）
  6 社区摘要（在线 API）
  7 完成（由 task_manager 更新 registry）

顺序约束：Neo4j 导入必须先于社区检测（社区依赖图中拓扑）。
"""
import os
import json

from config import PROJECT_ROOT, LOCAL_LLM_BASE_URL, MAX_CHUNK_SIZE, NEO4J_BUILD_URI

CACHE_DIR = os.path.join(PROJECT_ROOT, ".sisyphus", "evidence", "cache")
EXTRACT_DIR = os.path.join(PROJECT_ROOT, "reports", "kg_visual")
PARSED_DIR = os.path.join(PROJECT_ROOT, "data", "manuals", "_parsed")
EMBED_MODEL = "qwen3-embedding:0.6b"


def extract_path_for(manual_id: str) -> str:
    return os.path.join(EXTRACT_DIR, f"llm_extract_{manual_id}.json")


def communities_cache_for(manual_id: str) -> str:
    return os.path.join(CACHE_DIR, f"communities_{manual_id}.json")


def chunk_index_cache_for(manual_id: str) -> str:
    return os.path.join(CACHE_DIR, f"chunk_index_{manual_id}.json")


def parsed_md_path_for(manual_id: str) -> str:
    return os.path.join(PARSED_DIR, f"{manual_id}.md")


def _backfill_embeddings(store, source_manual: str) -> int:
    """Ollama batch 补 :Entity.embedding（source_manual 过滤），返回回填数。

    使用独立 driver（get_raw_driver），finally 只关闭该独立连接，绝不关闭
    store._driver——否则后续 _load_graph_topology / generate_community_summaries
    会在已关闭的 driver 上 session() 抛 DriverError("Driver is closed")。
    Ollama embed 调用带 3 次重试 + 指数退避，避免瞬时抖动即失败。
    """
    import time
    import ollama
    from backend.graph_store import get_raw_driver, assert_writable, assert_build_uri
    assert_writable()
    assert_build_uri(NEO4J_BUILD_URI)
    client = ollama.Client(host=LOCAL_LLM_BASE_URL, timeout=120.0)
    driver = get_raw_driver(NEO4J_BUILD_URI)
    done = 0
    try:
        with driver.session() as s:
            rows = s.run(
                "MATCH (n:Entity) WHERE n.name IS NOT NULL AND n.embedding IS NULL "
                "AND n.source_manual = $src RETURN n.name AS name",
                src=source_manual,
            ).data()
        names = [r["name"] for r in rows]
        if not names:
            return 0
        for i in range(0, len(names), 100):
            batch = names[i:i + 100]
            for attempt in range(3):
                try:
                    resp = client.embed(model=EMBED_MODEL, input=batch)
                    break
                except Exception:
                    if attempt == 2:
                        raise
                    time.sleep(3.0 * (attempt + 1))
            embeds = resp["embeddings"]
            with driver.session() as s:
                s.run(
                    "UNWIND $rows AS row "
                    "MATCH (n:Entity {name: row.name, source_manual: $src}) "
                    "SET n.embedding = row.emb",
                    rows=[{"name": n, "emb": v} for n, v in zip(batch, embeds)],
                    src=source_manual,
                )
            done += len(batch)
    finally:
        driver.close()
    return done


def _load_graph_topology(store, source_manual: str):
    """从 Neo4j 读 source_manual 过滤的 nodes/edges（供社区检测）。"""
    driver = store._driver
    with driver.session() as s:
        nodes = [(r["name"], ["Entity"]) for r in s.run(
            "MATCH (n:Entity) WHERE n.name IS NOT NULL AND n.source_manual = $src "
            "RETURN n.name AS name", src=source_manual)]
        edges = [(r["s"], r["r"], r["o"]) for r in s.run(
            "MATCH (n:Entity)-[r:REL]->(m:Entity) "
            "WHERE n.name IS NOT NULL AND m.name IS NOT NULL "
            "AND n.source_manual = $src AND m.source_manual = $src "
            "RETURN n.name AS s, r.type AS r, m.name AS o", src=source_manual)]
    return nodes, edges


def _delete_manual_graph(store, source_manual: str) -> None:
    """精确删除 source_manual 匹配的节点/边（不清全库）。"""
    store.assert_write_target()
    driver = store._driver
    with driver.session() as s:
        s.run(
            "MATCH (n:Entity {source_manual: $src}) DETACH DELETE n",
            src=source_manual,
        )
        s.run(
            "MATCH (c:Community {source_manual: $src}) DETACH DELETE c",
            src=source_manual,
        )


def _save_communities_cache(communities: list, path: str) -> None:
    serializable = []
    for c in communities:
        cc = dict(c)
        if isinstance(cc.get("nodes"), set):
            cc["nodes"] = list(cc["nodes"])
        serializable.append(cc)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(serializable, f, ensure_ascii=False)


def _write_extract_provenance(manual_id: str, source_manual: str, md_path: str,
                              extraction_prompt, triples: list, n_chunks: int,
                              use_online: bool) -> str:
    """抽取产物的记账单（AGENTS.md 规矩一/二）：产出那一刻落盘，不事后追认。"""
    import datetime
    import hashlib

    def _sha(path):
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for block in iter(lambda: f.read(1 << 20), b""):
                h.update(block)
        return h.hexdigest()

    if use_online:
        from config import DEEPSEEK_PLATFORMS
        provider = "deepseek"
        models = [p.model for p in DEEPSEEK_PLATFORMS]
        hosts = [p.base_url.split("//")[-1].split("/")[0] for p in DEEPSEEK_PLATFORMS]
    else:
        from config import LOCAL_LLM_MODEL
        provider = "local_ollama"
        models = [LOCAL_LLM_MODEL]
        hosts = [LOCAL_LLM_BASE_URL.split("//")[-1]]

    if extraction_prompt:
        prompt_sha = hashlib.sha256(extraction_prompt.encode("utf-8")).hexdigest()
        prompt_source = "calibrated"
    else:
        from backend.extractor import EXTRACTION_PROMPT_V2
        prompt_sha = hashlib.sha256(EXTRACTION_PROMPT_V2.encode("utf-8")).hexdigest()
        prompt_source = "EXTRACTION_PROMPT_V2"

    rec = {
        "produced_by": "backend/build_pipeline.py::build_pipeline",
        "written_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "manual_id": manual_id,
        "source_manual": source_manual,
        "md_path": md_path,
        "md_sha256": _sha(md_path) if os.path.exists(md_path) else None,
        "prompt_source": prompt_source,
        "prompt_sha256": prompt_sha,
        "chunking": {"fn": "parse_manual_chunks", "max_chunk_size": MAX_CHUNK_SIZE,
                     "n_blocks": n_chunks},
        "provider": provider,
        "models": models,
        "hosts": hosts,
        "n_triples": len(triples),
        "calls": None,
        "calls_note": "extract_from_chunks 未按调用计数；断点跳过的块不产生调用（永久追不回）",
    }
    out = os.path.join(EXTRACT_DIR, f"llm_extract_{manual_id}_provenance.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(rec, f, ensure_ascii=False, indent=2)
    return out


def build_pipeline(manual_id: str, source_manual: str, md_path: str,
                   progress_cb=None, extraction_prompt: str = None) -> dict:
    """执行构建，返回统计 dict。progress_cb(step, pct, message) 供进度上报。

    extraction_prompt：校准器合成的适配版提示词（须含 {context} 槽）；
    None = 用内置默认（存量行为不变）。
    """
    from backend.graph_store import assert_writable, assert_build_uri
    assert_writable()
    assert_build_uri(NEO4J_BUILD_URI)
    def report(step, pct, message):
        if progress_cb:
            progress_cb(step, pct, message)

    from backend.parser import parse_manual_chunks_sliding_window
    from backend.extractor import extract_from_chunks
    from backend.graph_store import GraphStore
    from backend.community import build_communities, generate_community_summaries
    from config import get_embed_model

    # Step 1 chunk 化
    report(1, 5, "chunk 化")
    chunks = parse_manual_chunks_sliding_window(md_path)
    chunk_texts = [c.text for c in chunks]

    # Step 2 LLM 三元组抽取
    # 2026-10-25 作者定：抽取默认走远程 deepseek-v4.1-flash（与社区摘要/出题/判分同渠道）；
    # 本机 qwen3 保留为显式回退：EXTRACT_LOCAL=1。两条通道产出的图不可混表，provenance 必须记。
    use_online = os.getenv("EXTRACT_LOCAL", "0").strip() != "1"
    report(2, 15, f"LLM 抽取 {len(chunks)} chunks"
                  + ("（校准适配版提示词）" if extraction_prompt else ""))
    extract_path = extract_path_for(manual_id)
    if os.path.exists(extract_path):
        with open(extract_path, "r", encoding="utf-8") as f:
            triples = json.load(f)
    else:
        triples = extract_from_chunks(chunks, llm=None, use_online=use_online,
                                      extraction_prompt=extraction_prompt)
        os.makedirs(os.path.dirname(extract_path), exist_ok=True)
        with open(extract_path, "w", encoding="utf-8") as f:
            json.dump(triples, f, ensure_ascii=False)
        _write_extract_provenance(manual_id, source_manual, md_path,
                                  extraction_prompt, triples, len(chunks), use_online)

    # Step 3 Neo4j 导入
    report(3, 45, "Neo4j 导入")
    store = GraphStore(uri=NEO4J_BUILD_URI)
    try:
        store.seed_triples([(t["s"], t["r"], t["o"], t.get("meta", "none")) for t in triples],
                           source_manual=source_manual)

        # Step 4 embedding 回填
        report(4, 65, "embedding 回填")
        n_emb = _backfill_embeddings(store, source_manual)

        # Step 5 社区检测
        report(5, 80, "社区检测")
        embed = get_embed_model()
        nodes, edges = _load_graph_topology(store, source_manual)
        communities = build_communities(nodes, edges, chunk_texts, embed,
                                        llm=None, generate_summaries=False,
                                        graph_store=store)
        comm_cache = communities_cache_for(manual_id)
        _save_communities_cache(communities, comm_cache)

        # Step 6 社区摘要
        report(6, 92, "社区摘要")
        if any(not c.get("summary") for c in communities):
            communities = generate_community_summaries(communities, chunk_texts,
                                                       graph_store=store,
                                                       source_manual=source_manual)
            _save_communities_cache(communities, comm_cache)
    finally:
        store.close()

    report(7, 100, "完成")
    return {
        "chunks": len(chunks),
        "triples": len(triples),
        "communities": len(communities),
        "embeddings": n_emb,
        "extract": extract_path,
        "communities_cache": comm_cache,
    }
