"""主管道入口：端到端串联（文章 3~6 章路径，L1 骨架）。

本文件为薄编排层，实际管道逻辑已拆分为：
  pipeline/stages.py   — 7 个独立阶段函数
  pipeline/runner.py   — PipelineRunner 编排器
  pipeline/extract.py  — 纯 LLM 抽取模式

用法：
  python main.py --manual data/manuals/pump_manual.md
  python main.py --mode extract --manual data/manuals/pump_fault_diagnosis_real.md
"""
import argparse
import os

from config import MANUALS_DIR
from pipeline import PipelineRunner, run_extract_pipeline


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manual", default=os.path.join(MANUALS_DIR, "pump_manual.md"))
    ap.add_argument("--mode", choices=["pipeline", "extract"], default="pipeline",
                    help="pipeline=原有完整管道; extract=纯 LLM 抽取模式（适配自然语言手册）")
    ap.add_argument("--build-llm-graph", action="store_true",
                    help="额外执行 LLM 自动抽取（场景A）")
    ap.add_argument("--build-vector-rag", action="store_true",
                    help="构建并测试纯向量 RAG 基线")
    ap.add_argument("--vector-scope", choices=["single", "all"], default="single",
                    help="向量库范围: single=仅当前手册, all=已seed的全部手册")
    ap.add_argument("--eval", action="store_true",
                    help="运行 RAGAS 评估对比（需配合 --build-vector-rag）")
    ap.add_argument("--test-set", default="data/eval/test_set.json",
                    help="RAGAS 测试集路径，默认 data/eval/test_set.json")
    ap.add_argument("--source-filter", default=None,
                    help="限制 KG 查询到指定手册（如 pump_manual.md），默认查全部")
    ap.add_argument("--use-online", action="store_true",
                    help="抽取时使用在线强模型（需 .env 配置）")
    ap.add_argument("--community-summaries", action="store_true",
                    help="社区构建后调用 LLM 为每个社区生成 report + summary（需 .env 配置 DEEPSEEK_API_KEY）")
    ap.add_argument("--calibrate-prompt", action="store_true",
                    help="只校准抽取提示词并落盘冻结件，打印变更单（C1：须人工过目，不进抽取）")
    ap.add_argument("--use-calibrated-prompt", action="store_true",
                    help="用已过目的冻结校准件跑 LLM 抽取（与 --calibrate-prompt 互斥，无冻结件则拒绝）")
    ap.add_argument("--force-calibrate", action="store_true",
                    help="忽略已有冻结校准件强制重校准（重校准后须重新人工过目）")
    ap.add_argument("--dry-run", action="store_true",
                    help="配合 --calibrate-prompt：只报预算/采样计划，不调 API 不花钱")
    args = ap.parse_args()

    if args.calibrate_prompt and args.use_calibrated_prompt:
        ap.error("--calibrate-prompt 与 --use-calibrated-prompt 互斥（C1：校准→人工过目→再抽取）")

    if args.mode == "extract":
        run_extract_pipeline(args)
    else:
        runner = PipelineRunner()
        runner.run(args)


if __name__ == "__main__":
    main()
