"""纯 LLM 抽取模式管道：从自然语言手册抽取三元组并落库。

用法：
    python main.py --mode extract --manual data/manuals/pump_fault_diagnosis_real.md
"""
import json
import os
from collections import Counter
from typing import Any

from config import PROJECT_ROOT, LOCAL_LLM_MODEL, LOCAL_LLM_BASE_URL
from backend import parser
from backend.graph_store import GraphStore
from backend.extractor import extract_from_chunks


def run_extract_pipeline(args: Any) -> None:
    """v2 纯 LLM 抽取模式：从自然语言手册抽取三元组并落库。

    流程：
    1. 分块（parse_manual_chunks）
    2. LLM 抽取（在线/本地）
    3. 写入 Neo4j :Entity（带 provenance）
    4. 输出统计报告
    """
    manual_path = args.manual
    manual_name = os.path.basename(manual_path)
    print(f"=== LLM 抽取模式 ===")
    print(f"  手册: {manual_path}")
    print(f"  后端: {'online (强模型)' if args.use_online else 'local (Ollama)'}")

    # 1. 分块
    print("\n--- 1. 分块 ---")
    chunks = parser.parse_manual_chunks(manual_path)
    print(f"  共 {len(chunks)} 个 chunks")
    for i, c in enumerate(chunks):
        print(f"    chunk {i}: {len(c.text)} 字")

    # 2. LLM 抽取
    print("\n--- 2. LLM 抽取 ---")
    if args.use_online:
        print(f"  使用在线模型: deepseek-v4-flash (OpenAI 兼容接口)")

    results = extract_from_chunks(
        chunks,
        llm=None,
        use_online=args.use_online,
        model=LOCAL_LLM_MODEL,
        base_url=LOCAL_LLM_BASE_URL,
        verbose=True,
    )

    if not results:
        print("  [WARN] 未抽取到任何三元组！请检查模型是否可用。")
        return

    # 3. 写入 Neo4j
    print("\n--- 3. 写入 Neo4j (:Entity) ---")
    store = GraphStore()
    store.upsert_triples_with_provenance(results, label="Entity")
    print(f"  已写入 {len(results)} 条三元组")

    # 4. 统计报告
    print("\n--- 4. 抽取统计 ---")
    rel_dist = Counter(r["r"] for r in results)
    print(f"  关系分布:")
    for rel, cnt in rel_dist.most_common():
        print(f"    {rel}: {cnt}")

    entities = set(r["s"] for r in results) | set(r["o"] for r in results)
    print(f"  唯一实体数: {len(entities)}")
    print(f"  三元组总数: {len(results)}")

    # 保存抽取结果到 JSON
    out_dir = os.path.join(PROJECT_ROOT, "reports", "kg_visual")
    os.makedirs(out_dir, exist_ok=True)
    out_json = os.path.join(out_dir, f"llm_extract_v2_{manual_name.replace('.md', '')}.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"  结果已保存: {out_json}")

    # 抽样展示
    print("\n  抽样三元组 (前10条):")
    for r in results[:10]:
        print(f"    ({r['s']}, {r['r']}, {r['o']})")
    if len(results) > 10:
        print(f"    ... 共 {len(results)} 条")

    store.close()
    print("\n抽取完成。")
