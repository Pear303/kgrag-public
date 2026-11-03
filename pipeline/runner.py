"""管道编排器：按顺序执行所有管道阶段。

PipelineRunner 是一个薄的编排类，负责：
1. 初始化共享上下文
2. 按顺序调用各阶段函数
3. 管理可选阶段（LLM 抽取、Vector RAG、RAGAS 评估）
"""
import os
import glob
from typing import Any, Optional

from config import (
    PROJECT_ROOT, MANUALS_DIR,
    get_llm, get_embed_model,
    DEEPSEEK_PLATFORMS, LOCAL_LLM_BASE_URL, LOCAL_LLM_MODEL,
)
from backend.graph_store import get_property_graph_store, get_raw_driver, resolve_vector_mode

from pipeline.context import PipelineContext
from pipeline.stages import (
    run_parse_stage,
    run_write_stage,
    run_extract_stage,
    run_community_stage,
    run_engine_setup_stage,
    run_verify_stage,
    run_visualize_stage,
)


class PipelineRunner:
    """主管道编排器：链式执行所有阶段。"""

    def __init__(self) -> None:
        self.ctx = PipelineContext()

    def run(self, args: Any) -> None:
        """执行完整的管道流程。

        Args:
            args: argparse.Namespace — 包含所有 CLI 参数

        提示词校准（C1 纪律，2026-10-03 签认）：
            --calibrate-prompt        只校准落盘并打印变更单，**不进抽取**（等人工过目）
            --use-calibrated-prompt   用已过目的冻结校准件跑抽取（无冻结件则拒绝）
            两旗互斥；重校准加 --force-calibrate（覆盖冻结件，需重新过目）
        """
        ctx = self.ctx
        ctx.manual_path = args.manual

        calibrate = bool(getattr(args, "calibrate_prompt", False))
        use_calibrated = bool(getattr(args, "use_calibrated_prompt", False))
        if calibrate and use_calibrated:
            raise SystemExit(
                "[C1] --calibrate-prompt 与 --use-calibrated-prompt 互斥："
                "先只跑校准并人工过目变更单，再单独用 --use-calibrated-prompt 跑抽取")
        if calibrate:
            self._run_calibration_stage(args, force=bool(getattr(args, "force_calibrate", False)))
            return

        # ---- Setup: 解析向量策略、初始化 LLM / Embedding ----
        print("=== 3. 向量策略 ===")
        ctx.vector_mode = resolve_vector_mode()
        print(f"  vector_mode={ctx.vector_mode}（Neo4j 版本决定；>=5.23 原生向量索引，否则降级内存余弦）")
        ctx.llm = get_llm()
        ctx.embed = get_embed_model()
        ctx.store = get_property_graph_store()
        ctx.driver = get_raw_driver()

        # ---- Stage 1+2: 解析 + 写入 ----
        run_parse_stage(ctx)
        run_write_stage(ctx)

        # ---- Stage 3: 可选 LLM 抽取 ----
        extraction_prompt = None
        if use_calibrated:
            from backend.prompt_calibrator import get_adapted_prompt
            extraction_prompt = get_adapted_prompt(ctx.manual_path)
            print("  [C1] 已加载冻结校准件的适配版提示词")
        if args.build_llm_graph:
            run_extract_stage(ctx, build_llm_graph=True, use_online=args.use_online,
                              extraction_prompt=extraction_prompt)

        # ---- Stage 4: 社区构建 ----
        run_community_stage(ctx, generate_summaries=args.community_summaries)

        # ---- Stage 5: 引擎设置 ----
        run_engine_setup_stage(ctx, source_filter=args.source_filter)

        # ---- Stage 6: 查询验证 ----
        run_verify_stage(ctx)

        # ---- Stage 7: 可视化 ----
        run_visualize_stage(ctx)

        # ---- 可选: Vector RAG + RAGAS 评估 ----
        if args.build_vector_rag:
            self._run_vector_rag_stage(args)

        print("\n全部步骤执行完毕。运行 frontend/app.py 启动 UI。")

    # ================================================================
    # 可选阶段：提示词校准（只校准落盘，不进抽取——C1）
    # ================================================================

    def _run_calibration_stage(self, args: Any, force: bool = False) -> None:
        """校准抽取提示词并落盘冻结件，打印变更单供人工过目（C1）。"""
        from backend.prompt_calibrator import (
            TOKEN_BUDGET, adapted_prompt_path, calibration_json_path, run_calibration,
        )
        print("=== 提示词校准（只校准不抽取，C1：变更单须人工过目后才可用于建图） ===")
        print(f"  手册: {args.manual}")
        rec = run_calibration(args.manual, budget=TOKEN_BUDGET,
                              force=force, dry_run=getattr(args, "dry_run", False))
        if rec.get("dry_run"):
            print("  [预检/dry-run，未调 API]")
            for k in ("mode", "manual_tokens_est", "selected_chars", "n_windows",
                      "token_budget", "model"):
                print(f"  {k} = {rec.get(k)}")
            return
        if rec.get("reused"):
            print("  已存在冻结校准件，直接复用（D 纪律）；如需重校准加 --force-calibrate")
        print(f"  采样模式: {rec['mode']}（手册 {rec['manual_chars']} 字 / 送入 {rec['selected_chars']} 字 /"
              f" {rec['n_windows']} 窗，预算 {rec['token_budget']}）")
        print(f"  变更 {len(rec['updates'])} 处:")
        for ch in rec["changes"]:
            print(f"    - [{ch['slot']}] {ch['reason']}")
        print(f"  校准件: {calibration_json_path(args.manual)}")
        print(f"  适配版提示词: {adapted_prompt_path(args.manual)}")
        if rec.get("reused"):
            print("  下一步：人工过目变更单后，用 --use-calibrated-prompt --build-llm-graph 跑抽取")
        else:
            print("  下一步：人工过目上面的变更单，认可后用 --use-calibrated-prompt --build-llm-graph 跑抽取")

    # ================================================================
    # 可选阶段：Vector RAG + RAGAS 评估
    # ================================================================

    def _run_vector_rag_stage(self, args: Any) -> None:
        """可选阶段：构建纯向量 RAG 基线 + 可选 RAGAS 评估。

        注意：此阶段保持在 runner 内，因为它高度耦合于评估流程且
        与核心管道解耦不紧密。
        """
        print("\n=== 8. 纯向量 RAG 基线 ===")
        from backend.vector_rag import VectorRAGQueryEngine
        from llama_index.core import Document
        from backend.parser import parse_manual_chunks_sliding_window

        all_chunks = parse_manual_chunks_sliding_window(args.manual)
        print(f"  主手册 chunks（滑动窗口）: {len(all_chunks)} chunks")

        if args.vector_scope == "all":
            manual_files = glob.glob(os.path.join(MANUALS_DIR, "*.md"))
            for mf in manual_files:
                if mf == args.manual or "fault_diagnosis_real" in mf:
                    continue
                try:
                    extra = parse_manual_chunks_sliding_window(mf)
                    all_chunks.extend(extra)
                    print(f"  +{os.path.basename(mf)}: {len(extra)} chunks")
                except Exception as e:
                    print(f"  [SKIP] {os.path.basename(mf)}: {e}")

        documents = [Document(text=c.text, metadata={"chunk_id": c.chunk_id, "source": c.source}) for c in all_chunks]
        print(f"  文档数: {len(documents)} chunks")

        vec_eng = VectorRAGQueryEngine(documents, self.ctx.embed, self.ctx.llm)

        # Smoke test
        sample_q = "轴承座温度升高怎么处理"
        result = vec_eng.query(sample_q)
        print(f"\n  样例查询: {sample_q}")
        print(f"  答案前200字: {result['answer'][:200]}")
        print(f"  检索节点数: {len(result['contexts'])}")
        print(f"  Top-3 相似度: {[round(s, 4) for s in result['scores'][:3]]}")

        if args.eval:
            self._run_ragas_eval(args, vec_eng)

    def _run_ragas_eval(self, args: Any, vec_eng: Any) -> None:
        """可选阶段：RAGAS 评估对比。"""
        print("\n=== 9. RAGAS 评估对比 ===")
        import json

        test_set_path = args.test_set
        if not os.path.exists(test_set_path):
            print(f"  [SKIP] 测试集不存在: {test_set_path}")
            return

        with open(test_set_path, "r", encoding="utf-8") as f:
            test_items = json.load(f)
        print(f"  测试集: {len(test_items)} 题")

        try:
            from langchain_ollama import ChatOllama, OllamaEmbeddings
            from backend.evaluator import RAGASEvaluator
            from backend.deepseek_llm import DeepSeekChatOpenAI

            primary_platform = DEEPSEEK_PLATFORMS[0] if DEEPSEEK_PLATFORMS else None
            if not primary_platform:
                raise RuntimeError("No DeepSeek platform configured (DEEPSEEK_PLATFORMS empty)")
            api_model = primary_platform.model
            print(f"  初始化法官模型 ({api_model} @ {primary_platform.base_url})...")
            judge_llm_eval = DeepSeekChatOpenAI(
                model=api_model,
                openai_api_key=primary_platform.key,
                openai_api_base=primary_platform.base_url,
                temperature=0,
            )

            ragas_embed = OllamaEmbeddings(
                model="qwen3-embedding:0.6b",
                base_url=LOCAL_LLM_BASE_URL,
            )

            # 修复 2026-09-12：Vector 侧答案生成必须与 KG-RAG 用同一生成器
            # （deepseek judge），否则本地 qwen2.5:3b 生成"根据手册未找到相关答案"
            # 导致 Vector 分数系统性崩塌（评测口径 bug）。口径与 run_pump_fault.py
            # 对齐：top_k=12（与 G max_ctx=12 对齐）+ 同一 CoT prompt + 同一截断 +
            # 同一答案解析。仅评测路径注入，日常查询仍用本地 llm。
            from backend.evaluator import _truncate_contexts, _split_cot_output
            vec_eng.top_k = 12
            vec_eng.context_truncator = _truncate_contexts
            vec_eng.prompt_template = RAGASEvaluator._COT_PROMPT_TEMPLATE

            def _vec_answer_generator(prompt):
                resp = judge_llm_eval.invoke(prompt)
                text = str(getattr(resp, "content", resp))
                return _split_cot_output(text)[0]

            vec_eng.answer_generator = _vec_answer_generator

            graph_engines: dict[str, Any] = {"manual": self.ctx.manual_eng}
            if self.ctx.communities and self.ctx.comm_eng:
                graph_engines["community"] = self.ctx.comm_eng
                print(f"  KG-RAG 引擎: manual + community")
            else:
                print(f"  KG-RAG 引擎: manual only (no communities)")

            evaluator = RAGASEvaluator(judge_llm_eval, ragas_embed, judge_model_name=api_model)
            print(f"\n  开始评估（{len(test_items)} 题，预计 {len(test_items) * 2} 分钟）...")
            print(f"  注意: 每轮调用 RAGAS 需要多次 LLM 法官打分，请耐心等待。")

            results = evaluator.compare(test_items, vec_eng, graph_engines)

            os.makedirs("reports", exist_ok=True)
            report_path = evaluator.generate_report(results, "reports/eval_report.md")
            print(f"\n  报告已生成: {report_path}")

            if results.get("vector_rag") and results.get("graph_rag"):
                vr = results["vector_rag"]
                gr = results["graph_rag"]
                print("\n  +-----------------------+---------+---------+")
                print("  | 指标                  | Vec RAG | KG-RAG  |")
                print("  +-----------------------+---------+---------+")
                for m in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]:
                    v = vr.get(m, 0)
                    g = gr.get(m, 0)
                    print(f"  | {m:<21} | {v:>7.3f} | {g:>7.3f} |")
                print("  +-----------------------+---------+---------+")

        except ImportError as e:
            print(f"  错误: 缺少依赖，无法运行评估: {e}")
            print(f"  请安装: pip install langchain-ollama ragas")
        except Exception as e:
            import traceback
            print(f"  评估失败: {e}")
            traceback.print_exc()
