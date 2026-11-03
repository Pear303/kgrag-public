"""GraphRAG 宏观层：社区发现与 Chunk 绑定（文章 5.2 / 5.3 / 5.4 节）。

- _run_louvain：用 NetworkX Louvain 对图谱做社区聚类（转无向图）。
- _bind_chunk_ids：实体名子串匹配 + Embedding 回退，把节点绑定回源 Chunk。
- 社区去重：chunk_ids 完全相同的社区合并；孤立 Chunk 各自成独立社区。
- generate_community_summaries：LLM 为每个社区生成 report + summary。
- 存储到 Neo4j :Community 节点，供 CommunityQueryEngine 召回。
"""
import time
import json
import threading
import numpy as np
import networkx as nx

from config import SIMILARITY_THRESHOLD

# 生成摘要的硬性字数上限（字符数，中文字符）
COMMUNITY_REPORT_MAX_CHARS = 500
COMMUNITY_SUMMARY_MAX_CHARS = 120

COMMUNITY_SUMMARY_PROMPT = """你是一个知识图谱社区分析专家。以下内容来自一个文档社区的多个文本片段，请仔细阅读并生成：

1. **report**: 一段详细描述，概括该社区涵盖的主要主题、关键实体及其关系。**report 总长度必须严格控制在 {report_max} 个中文字符以内，超过即无效。**
2. **summary**: 一句简短摘要，概括该社区的核心内容。**summary 总长度必须严格控制在 {summary_max} 个中文字符以内，超过即无效。**

【硬性要求】
- report 和 summary 必须原样保留片段中出现的报警号（如 ALM414、ALM100、NC2020）、
  输入/输出地址（如 X018-5、X1000.4、G008.0、Y026-0）、诊断参数号（如 DGN700、DGN701）、
  参数号（如 P/S90、PRM2023）、端子/熔断器编号（如 CP1、F11/F12、JR11）、
  元件型号（如 A169、Q11、D12、6SN1123-1AA00-0AA0）以及关键数值；禁止省略、改写或丢弃。
- 具体故障案例的排查结论（如"更换D12(A169)后短路消失""刀库气动冲击导致后位信号瞬间断开"）
  必须写进 report，禁止只保留泛化主题。
- **严格基于给定文本片段归纳，禁止引入片段中不存在的系统品牌/型号/概念**。片段未出现
  任何品牌时，不得擅自提及 FANUC、SIEMENS、发那科、西门子、HEIDENHAIN、海德汉、三菱、
  华中、广州数控等名称；片段未涉及机械结构（丝杠/齿轮/轴承/齿面硬度等）时不得外推该类内容。

文本片段：
{chunk_texts}

请严格按以下JSON格式回复（不要包含任何其他内容）：
{{"report": "详细描述（不超过{report_max}字）...", "summary": "简短摘要（不超过{summary_max}字）..."}}"""


def _run_louvain(g: nx.DiGraph, seed: int = 42) -> dict:
    """Louvain 社区发现（文章 5.2 节）。要求无向图。"""
    ug = g.to_undirected()
    communities = nx.community.louvain_communities(ug, seed=seed)
    node_to_comm = {}
    for comm_id, nodes in enumerate(communities):
        for node in nodes:
            node_to_comm[node] = comm_id
    return node_to_comm


def _bind_chunk_ids(self_names, chunk_texts, embed_model, threshold: float = 0.70):
    """实体 -> 源 Chunk 绑定（文章 5.3 节）。

    先用实体名子串匹配；匹配不到用 Embedding 相似度 top-1 回退。
    """
    node_to_chunks = {}
    chunk_embeddings = [embed_model.get_text_embedding(t) for t in chunk_texts]
    for name in self_names:
        matched_ids = [idx for idx, txt in enumerate(chunk_texts) if name in txt]
        if not matched_ids and chunk_texts:
            name_emb = np.array(embed_model.get_text_embedding(name), dtype=np.float32)
            sims = [
                float(np.dot(name_emb, np.array(ce, dtype=np.float32)) /
                      (np.linalg.norm(name_emb) * np.linalg.norm(np.array(ce, dtype=np.float32)) + 1e-9))
                for ce in chunk_embeddings
            ]
            best = int(np.argmax(sims))
            if sims[best] >= threshold:
                matched_ids.append(best)
        node_to_chunks[name] = matched_ids
    return node_to_chunks


def _parse_json_lenient(content):
    """宽松 JSON 解析：容忍 markdown 围栏/前后缀噪音。"""
    try:
        return json.loads(content)
    except Exception:
        pass
    start = content.find("{")
    end = content.rfind("}")
    if start >= 0 and end > start:
        try:
            return json.loads(content[start:end + 1])
        except Exception:
            pass
    return None


def _summarize_community(idx, total, comm, chunk_texts, graph_store, client,
                         source_manual=None):
    """单个社区摘要任务（并行 worker），返回 (idx, status)。"""
    cid = str(comm.get("id", idx))

    # —— 断点跳过：只跳过**同一本手册**里已有摘要的同号社区 ——
    # 只按 id 查会跨手册命中：第二本手册的社区 0 会拿到第一本手册社区 0 的摘要，
    # 且命中的那条一次调用都不花（status=skip），所以"有 N 条摘要"不等于"付了 N 次钱"。
    # source_manual 缺失时宁可不跳过（多花一次钱可查，抄串无痕迹）——失败安全方向唯一。
    existing = None
    if graph_store is not None and source_manual:
        try:
            rows = graph_store.structured_query(
                "MATCH (c:Community {id: $cid, source_manual: $sm}) "
                "RETURN c.summary AS summary, c.report AS report",
                {"cid": cid, "sm": source_manual})
            if rows and rows[0].get("summary"):
                existing = rows[0]
        except Exception:
            pass
    if existing:
        comm["summary"] = existing["summary"] or ""
        comm["report"] = existing["report"] or ""
        return idx, "skip"

    texts = comm.get("raw_context", [])
    if not texts:
        comm["summary"] = ""
        comm["report"] = ""
        return idx, "empty"

    # 2026-08-10：分块摘要（顽固社区根因修复）。
    # 实测：代理对 >4k 字符 prompt 在高负载下稳定返回 500/空（v13 中 34 个顽固社区
    # 失败组长度中位数 4015 vs 成功组 1086）。当社区文本超过 CHUNKED_SUMMARY_CHARS
    # 时，按 CHUNK_SUMMARY_BATCH 个文本一组分批摘要，再把各批 report/summary 合并，
    # 规避代理长输入不稳定区（不改变摘要口径，只分治输入长度）。
    CHUNKED_SUMMARY_CHARS = 4000
    CHUNK_SUMMARY_BATCH = 4

    def _chunked_prompt(texts_part):
        combined = "\n---\n".join(t[:1000] for t in texts_part)
        return COMMUNITY_SUMMARY_PROMPT.format(
            chunk_texts=combined,
            report_max=COMMUNITY_REPORT_MAX_CHARS,
            summary_max=COMMUNITY_SUMMARY_MAX_CHARS,
        )

    all_parts = [t[:1000] for t in texts[:20]]
    if sum(len(t) for t in all_parts) <= CHUNKED_SUMMARY_CHARS or len(all_parts) <= CHUNK_SUMMARY_BATCH:
        prompts = [_chunked_prompt(all_parts)]
    else:
        prompts = [_chunked_prompt(all_parts[i:i + CHUNK_SUMMARY_BATCH])
                   for i in range(0, len(all_parts), CHUNK_SUMMARY_BATCH)]
        print(f"  [WARN] 社区 {idx+1}/{total} 大文本分 {len(prompts)} 批摘要", flush=True)
    prompt = prompts[0]  # 缓存键用首批（供 get 预查）

    try:
        # LLM 磁盘缓存（开发/构建场景，env ENABLE_LLM_CACHE=1 启用，默认关）：
        # 同一社区同 prompt 命中不重复付费；禁止接入评测答案生成链。
        from backend.llm_cache import get_cache as _get_llm_cache
        _cache = _get_llm_cache()
        _model = "deepseek-v4-flash"

        def _call_once(p):
            resp = client.chat(
                messages=[{"role": "user", "content": p}],
                max_tokens=8192,
                temperature=0.1,
            )
            return resp.get("content", "")

        def _summarize_prompt(p):
            """单批摘要（带缓存 + 重试），返回 (report, summary) 或抛异常。"""
            cached = _cache.get(_model, p, 0.1, 8192) if _cache else None
            if cached is not None:
                res = _parse_json_lenient(cached)
                if res is not None:
                    return res.get("report", ""), res.get("summary", "")
                raise ValueError("缓存 JSON 解析失败")
            content = ""
            last_err = None
            for attempt in range(3):
                try:
                    content = _call_once(p)
                    if _parse_json_lenient(content) is not None:
                        break
                    last_err = ValueError("JSON 解析失败（第 %d 次）" % (attempt + 1))
                except Exception as e:
                    last_err = e
                if attempt < 2:
                    print(f"  [WARN] 社区 {idx+1}/{total} 第 {attempt + 2} 次重试: {last_err}",
                          flush=True)
                    time.sleep(2 * (attempt + 1))
            else:
                raise last_err or ValueError("JSON 解析失败")
            if _cache:
                _cache.put(_model, p, 0.1, 8192, content)
            res = _parse_json_lenient(content)
            if res is None:
                raise ValueError("JSON 解析失败")
            return res.get("report", ""), res.get("summary", "")

        reports, summaries = [], []
        for p in prompts:
            r, s = _summarize_prompt(p)
            reports.append(r)
            summaries.append(s)
        report = "\n".join(r for r in reports if r)
        summary = "；".join(s for s in summaries if s)
        # 兜底截断：即使模型超长也强制限制在提示词规定的上限内
        if len(report) > COMMUNITY_REPORT_MAX_CHARS:
            report = report[:COMMUNITY_REPORT_MAX_CHARS] + "……"
        if len(summary) > COMMUNITY_SUMMARY_MAX_CHARS:
            summary = summary[:COMMUNITY_SUMMARY_MAX_CHARS] + "……"
        comm["report"] = report
        comm["summary"] = summary
    except Exception as e:
        print(f"  [WARN] 社区 {idx+1}/{total} 摘要生成失败: {e}", flush=True)
        comm["report"] = ""
        comm["summary"] = ""
        return idx, "fail"

    if graph_store is not None:
        try:
            graph_store.upsert_community(
                community_id=cid,
                label=cid,
                summary=comm["summary"],
                report=comm["report"],
                size=len(comm.get("nodes", set())),
                chunk_count=len(comm.get("chunk_ids", [])),
                chunk_ids=comm.get("chunk_ids", []),
                source_manual=source_manual,
            )
        except Exception as e:
            print(f"  [WARN] 社区 {cid} 写入 Neo4j 失败: {e}", flush=True)
    return idx, "ok"


def generate_community_summaries(communities: list, chunk_texts: list,
                                 graph_store=None, source_manual: str = None) -> list:
    """为每个社区调用 LLM 生成 report + summary，并存入 Neo4j（支持断点续跑）。

    参数:
        communities: build_communities() 返回的社区列表
        chunk_texts: 原始 chunk 文本列表
        graph_store: GraphStore 实例（可选，若提供则写入 Neo4j 并支持断点跳过）
        source_manual: 这批社区来自哪本手册。**跨手册复用摘要的唯一屏障**：
            传了才可能跳过，不传则一律重新生成（宁可多花，不可抄串）。

    返回:
        communities: 增强后的社区列表，每项新增 summary / report 字段
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from backend.api_client import get_client

    if graph_store is not None and not source_manual:
        print("  [WARN] 未给 source_manual：社区摘要既不跳过也不写库"
              "（社区层按 (id, source_manual) 定身份，缺它无法安全落库）", flush=True)

    client = get_client()
    total = len(communities)
    skipped = 0
    lock = threading.Lock()

    def _run_one(idx_comm):
        idx, comm = idx_comm
        return _summarize_community(idx, total, comm, chunk_texts, graph_store, client,
                                    source_manual=source_manual)

    with ThreadPoolExecutor(max_workers=4) as ex:
        futures = {ex.submit(_run_one, (idx, comm)): idx
                   for idx, comm in enumerate(communities)}
        for fut in as_completed(futures):
            idx, status = fut.result()
            cid = communities[idx].get("id", idx)
            with lock:
                if status == "skip":
                    skipped += 1
                print(f"  社区 {idx+1}/{total}: id={cid} 完成({status})", flush=True)

    print(f"  共生成 {total - skipped} 个新摘要，跳过 {skipped} 个已有", flush=True)
    return communities



def build_communities(graph_nodes: list, graph_edges: list, chunk_texts: list,
                      embed_model, llm=None, seed: int = 42, generate_summaries: bool = False,
                      graph_store=None, source_manual: str = None):
    """端到端构建社区（文章 5.2~5.4）。

    参数:
      graph_nodes: [(name, labels), ...]
      graph_edges: [(s, rel, o), ...]
      chunk_texts: 原始 chunk 文本列表
      generate_summaries: 是否调用 LLM 为每个社区生成 report + summary
      graph_store: GraphStore 实例（generate_summaries=True 时用于写入 Neo4j）
    返回: list[dict]，每项含 nodes / chunk_ids / raw_context / description / description_embedding
          （generate_summaries=True 时额外含 summary / report）
    """
    g = nx.DiGraph()
    for name, _ in graph_nodes:
        if name is not None:
            g.add_node(name)
    for s, _, o in graph_edges:
        if s is not None and o is not None:
            g.add_edge(s, o)

    node_to_comm = _run_louvain(g, seed=seed)
    # 收集每个社区的节点
    comm_nodes: dict = {}
    for node, cid in node_to_comm.items():
        comm_nodes.setdefault(cid, []).append(node)

    binding = _bind_chunk_ids(list(node_to_comm.keys()), chunk_texts, embed_model)

    communities_out = []
    seen_chunk_sets = {}
    for cid, nodes in comm_nodes.items():
        chunk_ids = sorted({cid_ for n in nodes for cid_ in binding.get(n, [])})
        raw_context = [chunk_texts[i] for i in chunk_ids if i < len(chunk_texts)]
        key = tuple(chunk_ids)
        if key in seen_chunk_sets:
            # 社区去重：合并节点（文章 5.4）
            seen_chunk_sets[key]["nodes"] |= set(nodes)
            continue
        comm = {
            "id": cid,
            "nodes": set(nodes),
            "chunk_ids": chunk_ids,
            "raw_context": raw_context,
        }
        # TODO(生产化): 用 LLM 基于 raw_context 生成 community description
        # 修复 KG-6：description 优先用社区节点名（主题词）拼接——短词集合与
        # 问句的 embedding 语义可比性优于"整段 raw 前 2000 字"，社区匹配更聚焦；
        # raw 拼接仅作孤立社区（无节点）的兜底。
        desc_text = "; ".join(sorted(n for n in nodes if n)[:30])
        if not desc_text.strip():
            desc_text = " ".join(raw_context)[:2000]
        if not desc_text.strip():
            desc_text = "; ".join(sorted(n for n in nodes if n)[:20])[:2000]
        comm["description"] = desc_text
        comm["description_embedding"] = embed_model.get_text_embedding(comm["description"])
        communities_out.append(comm)
        seen_chunk_sets[key] = comm

    # 孤立 Chunk 处理（文章 5.4）
    all_bound = {c for comm in communities_out for c in comm["chunk_ids"]}
    for i, txt in enumerate(chunk_texts):
        if i not in all_bound:
            comm = {
                "id": f"orphan_{i}",
                "nodes": set(),
                "chunk_ids": [i],
                "raw_context": [txt],
                "description": txt[:2000],
                "description_embedding": embed_model.get_text_embedding(txt[:2000]),
            }
            communities_out.append(comm)

    if generate_summaries:
        print("  开始 LLM 社区摘要生成...")
        communities_out = generate_community_summaries(
            communities_out, chunk_texts, graph_store=graph_store,
            source_manual=source_manual
        )

    return communities_out
