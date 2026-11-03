"""快速重评脚本：读取已保存的 per_question 答案，用 qwen 系列模型跑 RAGAS 评分。

judge 降级链：qwen-max → qwen-plus-2025-07-28 → qwen3.7-plus → qwen3.6-plus → qwen3.7-flash
（DashScope 官方兼容端点，支持 n=3）。

用法:
  python rescore_ragas.py [label]                # 默认 graph_rag，读 .sisyphus/evidence/
  python rescore_ragas.py graph_rag --round round_002   # 读指定 round 目录
  python rescore_ragas.py graph_rag --model qwen3.7-plus  # 指定 judge 模型（不指定则全降级链）
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from qwen_judge import make_evaluator

ROOT = os.path.dirname(os.path.abspath(__file__))
EVIDENCE = os.path.join(ROOT, ".sisyphus", "evidence")


def main():
    args = sys.argv[1:]
    label = "graph_rag"
    round_dir = None
    model = None
    i = 0
    while i < len(args):
        if args[i] == "--round" and i + 1 < len(args):
            round_dir = args[i + 1]
            i += 2
        elif args[i] == "--model" and i + 1 < len(args):
            model = args[i + 1]
            i += 2
        else:
            label = args[i]
            i += 1

    if round_dir:
        base = os.path.join(EVIDENCE, "rounds", round_dir)
    else:
        base = EVIDENCE
    src = os.path.join(base, f"{label}_results.json")
    with open(src, "r", encoding="utf-8") as f:
        data = json.load(f)

    per_q = data.get("per_question", [])
    n_ok = sum(1 for r in per_q if r.get("status") == "ok")
    n_fail = sum(1 for r in per_q if r.get("status") == "failed")
    print(f"\n=== {label} ({os.path.basename(base)}): {n_ok} ok, {n_fail} failed, total {len(per_q)} ===")

    evaluator, judge = make_evaluator(model)
    result = evaluator._run_ragas(per_q, n_ok, n_fail, label, base)
    for m in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]:
        print(f"  {m}: {result.get(m, 'N/A')}")

    print("\nDone!")


if __name__ == "__main__":
    main()
