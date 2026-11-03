# -*- coding: utf-8 -*-
"""一键建库脚本 —— 从维修手册构建可服务的 KG-RAG 知识库

流程（面向论文手册体系，与 main.py 的 L1 骨架独立）：
  手册 .md → (可选 LLM 抽取) → Neo4j 图谱（目标由 --uri 指定）→ 社区摘要

用法：
  python scripts/build_kb.py --manual xxx.md --extract reports/kg_visual/llm_extract_v5_xxx.json \
      --uri bolt://localhost:7691
  python scripts/build_kb.py --manual xxx.md --uri bolt://localhost:7691 --dry-run

安全：
  - **目标库没有默认值**（2026-10-08 改：旧默认 bolt://localhost:7690 是冻结评测库）。
    --uri 与 KGRAG_BUILD_URI 都没给 ⇒ 拒。
  - 主库 7688／冻结评测库 7690／第二语料库 7692 是受保护实例 ⇒ 即使命中也拒，
    除非再加 --allow-protected（表示你知道要覆盖的是什么）。
  - 目标库里已有内容而没给 --wipe ⇒ 拒（不许往别人的图上叠）。
  - SERVING_READONLY=1 ⇒ 连 --wipe 也拒（旧版 --wipe 走裸驱动，绕过过只读闸）。
"""
import argparse
import io
import json
import os
import sys

out = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stdout = out

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

OK = "[OK]   "
WA = "[WARN] "
FA = "[FAIL] "


def log(tag, msg):
    print(tag + msg, flush=True)

#: 受保护实例：写它们要么不可回灌、要么线上在读。命中即拒，除非 --allow-protected。
PROTECTED = {
    "localhost:7688": "主库（线上服务在读）",
    "localhost:7690": "评测库（覆盖后重灌不回来）",
    "localhost:7692": "第二语料库",
}
HINT_WRITABLE = "bolt://localhost:7691（一次性隔离实例）"


def uri_key(u):
    """URI 归一化：小写、剥 scheme、剥尾斜杠。

    bolt://localhost:7690/ 与 bolt://localhost:7690 是同一个库，
    不许因为写法不同绕过目标门。
    """
    u = (u or "").strip().lower().rstrip("/")
    for p in ("bolt://", "neo4j://", "neo4j+s://", "bolt+s://"):
        if u.startswith(p):
            u = u[len(p):]
    return u


def expected_counts(triples):
    """按 seed_triples 的 MERGE 语义现算这批行**应该**产生多少节点/关系。

    节点 = 不同 s/o 名字（同一 source_manual）；关系 = 不同 (s, r, meta, o)。
    """
    names, rels = set(), set()
    for row in triples:
        if isinstance(row, dict):
            s, r, o, m = row.get("s"), row.get("r"), row.get("o"), row.get("meta") or "none"
        elif len(row) >= 4:
            s, r, o, m = row[0], row[1], row[2], row[3] or "none"
        else:
            s, r, o, m = row[0], row[1], row[2], "none"
        if s is None or r is None or o is None:
            continue
        names.add(s)
        names.add(o)
        rels.add((s, r, m, o))
    return {"nodes": len(names), "rels": len(rels)}


def graph_counts(driver):
    with driver.session() as s:
        n = s.run("MATCH (n) RETURN count(n) AS c").single()["c"]
        e = s.run("MATCH ()-[x]->() RETURN count(x) AS c").single()["c"]
    return {"nodes": n, "rels": e}


def main():
    ap = argparse.ArgumentParser(description="构建 KG-RAG 知识库")
    ap.add_argument("--manual", required=True,
                    help="手册 markdown 路径（如 data/manuals/xxx.md）")
    ap.add_argument("--extract", default=None,
                    help="已有三元组抽取 JSON（无则需 LLM 抽取）")
    ap.add_argument("--wipe", action="store_true",
                    help="清空目标库后重建（默认幂等 upsert）")
    ap.add_argument("--communities", action="store_true",
                    help="构建后生成社区摘要（需 LLM API）")
    ap.add_argument("--dry-run", action="store_true",
                    help="仅检查输入、连接与目标门，不写库")
    ap.add_argument("--uri", default=None,
                    help="目标 Neo4j URI（必填，无默认值；如 bolt://localhost:7691）")
    ap.add_argument("--allow-protected", action="store_true",
                    help="明知目标是受保护库（7688/7690/7692）仍要写")
    args = ap.parse_args()

    print("=" * 60)
    print("KG-RAG 知识库构建")
    print("=" * 60)

    # 1. 手册存在
    manual_path = os.path.join(ROOT, args.manual)
    if not os.path.exists(manual_path):
        log(FA, f"手册不存在: {manual_path}")
        return 1
    log(OK, f"手册: {args.manual}")

    # 2. 抽取文件
    triples = None
    if args.extract:
        ep = os.path.join(ROOT, args.extract)
        if not os.path.exists(ep):
            log(FA, f"抽取文件不存在: {ep}")
            return 1
        with open(ep, encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            triples = data.get("triples", [])
        else:
            triples = data
        log(OK, f"抽取文件: {args.extract}（{len(triples or [])} 条）")
    else:
        log(WA, "未提供 --extract，将跳过三元组导入（只建分块）")
        log(WA, "提示: 若手册已建过库，可不传 --extract 仅补社区")

    # 3. 目标库：单一来源，不给默认值
    target = args.uri or os.getenv("KGRAG_BUILD_URI")
    if not target:
        log(FA, "没给目标库 ⇒ 拒。本脚本**不给默认目标**（旧默认 bolt://localhost:7690 是冻结评测库）")
        for k, v in sorted(PROTECTED.items()):
            log(WA, f"  受保护 {k} = {v}")
        log(WA, f"  本次唯一允许写的一次性实例：{HINT_WRITABLE}")
        log(WA, "  例：python scripts/build_kb.py --manual <手册.md> --extract <抽取件.json> "
                "--uri bolt://localhost:7691")
        return 1
    tk = uri_key(target)
    if tk in PROTECTED and not args.allow_protected:
        log(FA, f"目标是受保护库 {tk}（{PROTECTED[tk]}）⇒ 拒写。"
                f"明知要覆盖它才加 --allow-protected")
        return 1
    if tk in PROTECTED:
        log(WA, f"--allow-protected：你已确认要写 {tk}（{PROTECTED[tk]}）")
    #: config 在 import 那一刻取值（config/__init__.py:88 默认就是主库 7688），
    #: 所以必须在 import backend.graph_store **之前**把它写进环境。
    os.environ["NEO4J_URI"] = target

    try:
        from neo4j import GraphDatabase
        pwd = os.getenv("NEO4J_PASSWORD", "")
        drv = GraphDatabase.driver(target, auth=("neo4j", pwd))
        drv.verify_connectivity()
        log(OK, f"Neo4j 连通: {target}")
    except Exception as e:
        log(FA, f"Neo4j 不可达: {e}")
        return 1

    #: 绑定回读门：命令行目标 / 裸驱动 URI / GraphStore 模块绑定的 URI 必须是同一个库。
    #: 这一格焊的是"用 7690 的驱动清空、用 7688 的 store 写入"那种两库分裂。
    import backend.graph_store as GS
    if uri_key(GS.NEO4J_URI) != tk:
        log(FA, f"目标门不过：GraphStore 实际绑定 {GS.NEO4J_URI}，命令行目标 {target} ⇒ 两库分裂，拒")
        drv.close()
        return 1
    log(OK, f"目标门：驱动与 GraphStore 绑定同为 {tk}")

    if args.dry_run:
        log(OK, "dry-run 通过：输入就绪、Neo4j 连通、目标门三处同库。未写库。")
        drv.close()
        return 0

    # 4. 解析分块
    log(WA, "解析分块...")
    from backend.parser import parse_manual_chunks
    chunks = parse_manual_chunks(manual_path)
    chunk_texts = [c.text for c in chunks]
    log(OK, f"分块: {len(chunk_texts)} 个")

    # 5. 导入三元组
    gs = None
    if triples:
        from backend.graph_store import GraphStore, assert_writable
        manual_name = os.path.basename(manual_path)
        try:
            assert_writable()
        except PermissionError as e:
            log(FA, f"只读闸拒绝写入: {e}")
            drv.close()
            return 1
        c0 = graph_counts(drv)
        if not args.wipe and c0["nodes"]:
            want = expected_counts(triples)
            with drv.session() as s:
                src0 = [r["m"] for r in
                        s.run("MATCH (n:Entity) RETURN DISTINCT n.source_manual AS m")]
            if not (c0 == want and src0 == [manual_name]):
                log(FA, f"目标库里已有内容 {json.dumps(c0, ensure_ascii=False)} "
                        f"source_manual={src0}，而这批行的预期是 "
                        f"{json.dumps(want, ensure_ascii=False)} ⇒ 不许往别人的图上叠；"
                        f"确认要重建再加 --wipe")
                drv.close()
                return 1
            log(OK, f"目标库已正好是这批行（{want['nodes']} 点／{want['rels']} 边）⇒ 幂等重跑")
        if args.wipe:
            log(WA, "--wipe: 清空图谱...")
            try:
                with drv.session() as s:
                    s.run("MATCH (n) DETACH DELETE n")
            except Exception as e:
                log(WA, f"清空失败（可能无权限）: {e}")
        #: GraphStore.__init__ 不收参数，它自己按 config 绑定的 URI 建驱动
        gs = GraphStore()
        gs.seed_triples(triples, source_manual=manual_name)
        log(OK, f"导入 {len(triples)} 条三元组")

    # 6. 社区（可选）
    if args.communities:
        log(WA, "构建社区摘要（需 LLM API）...")
        try:
            from backend.community import build_communities
            from backend.ollama_embed import OllamaLocalEmbedding
            embed = OllamaLocalEmbedding()
            if gs is None:
                from backend.graph_store import GraphStore
                gs = GraphStore()
            comms = build_communities([], [], chunk_texts, embed,
                                      generate_summaries=True, graph_store=gs)
            log(OK, f"社区: {len(comms)} 个（含摘要）")
        except Exception as e:
            log(WA, f"社区构建失败: {e}")
            log(WA, "可稍后单独构建（参考 scripts/generate_community_summaries_*.py）")

    if gs is not None:
        gs.close()
    drv.close()
    log(OK, "建库完成。更新 .env 的 SERVING_ROUND 后启动系统。")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())