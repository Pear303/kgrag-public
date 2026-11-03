"""全链路评估脚本：丢文档→seed→出题→评估，一条命令跑通。

用法:
  # 结构化手册（自动 pipeline 模式）
  python run_eval.py --doc valve_manual.md

  # 自然语言手册（自动 extract 模式）+ 指定评估入口
  python run_eval.py --doc pump_fault_diagnosis_real.md --eval-doc pump_manual.md --qa-count 15

  # 精细控制
  python run_eval.py --doc valve_manual.md --test-doc valve_manual.md --vector-scope all --qa-count 20
  python run_eval.py --doc valve_manual.md --skip-seed --skip-qa --eval-only
"""
import argparse
import os
import subprocess
import sys

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PYTHON = sys.executable
NATURAL_LANG_KEYWORD = "fault_diagnosis_real"


def run(cmd: list[str]) -> bool:
    print(f"\n  -> {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=THIS_DIR)
    ok = result.returncode == 0
    if not ok:
        print(f"  [FAIL] exit={result.returncode}")
    return ok


def is_natural_language(manual: str) -> bool:
    return NATURAL_LANG_KEYWORD in manual


def seed_manual(manual: str):
    path = os.path.join("data", "manuals", manual)
    if not os.path.exists(path):
        print(f"  [SKIP] 手册不存在: {path}")
        return False
    if is_natural_language(manual):
        print(f"\n  === Seed (extract 模式): {manual} ===")
        return run([PYTHON, 'main.py', '--mode', 'extract', '--manual', path, '--use-online'])
    else:
        print(f"\n  === Seed (pipeline 模式): {manual} ===")
        return run([PYTHON, 'main.py', '--manual', path])


def generate_qa(test_doc: str, count: int) -> str | None:
    path = os.path.join("data", "manuals", test_doc)
    if not os.path.exists(path):
        print(f"  [SKIP] 测试文档不存在: {path}")
        return None
    name = test_doc.replace(".md", "")
    output = f"data/eval/test_set_{name}_auto.json"
    print(f"\n  === 生成 QA ({count} 题): {test_doc} ===")
    ok = run([PYTHON, 'generate_qa.py', '--manual', path, '--num', str(count), '--output', output])
    return output if ok else None


def run_evaluation(eval_manual: str, test_set: str, vector_scope: str, source_filter: str | None):
    path = os.path.join("data", "manuals", eval_manual)
    cmd = [PYTHON, 'main.py', '--manual', path, '--build-vector-rag',
           '--vector-scope', vector_scope, '--test-set', test_set, '--eval']
    if source_filter:
        cmd += ['--source-filter', source_filter]
    print(f"\n  === RAGAS 评估 ===")
    print(f"  入口手册: {eval_manual}")
    print(f"  测试集:   {test_set}")
    print(f"  向量范围: {vector_scope}")
    print(f"  图谱过滤: {source_filter or '全部'}")
    return run(cmd)


def main():
    ap = argparse.ArgumentParser(description="全链路评估脚本")
    ap.add_argument("--doc", required=True,
                    help="要处理的手册名（如 valve_manual.md 或 pump_fault_diagnosis_real.md）")
    ap.add_argument("--test-doc", default=None,
                    help="生成测试题的手册（默认=--doc）")
    ap.add_argument("--eval-doc", default=None,
                    help="评估时作为入口的手册（默认=--doc，自然语言手册需指定结构化手册）")
    ap.add_argument("--vector-scope", choices=["single", "all"], default="single")
    ap.add_argument("--source-filter", default=None,
                    help="限制 KG 到指定手册（默认=自动从 --doc 推断）")
    ap.add_argument("--qa-count", type=int, default=12)
    ap.add_argument("--test-set", default=None,
                    help="手动指定测试集（设置后跳过 QA 生成）")
    ap.add_argument("--skip-seed", action="store_true")
    ap.add_argument("--skip-qa", action="store_true")
    ap.add_argument("--skip-eval", action="store_true")
    ap.add_argument("--eval-only", action="store_true")
    args = ap.parse_args()

    test_doc = args.test_doc or args.doc
    eval_doc = args.eval_doc or args.doc

    if is_natural_language(eval_doc) and not args.eval_doc:
        print("[WARN] 自然语言手册无法直接用于评估，请用 --eval-doc 指定结构化手册")
        print("  例如: --doc pump_fault_diagnosis_real.md --eval-doc pump_manual.md")
        return

    if args.eval_only:
        test_set = args.test_set or "data/eval/test_set.json"
        source_filter = args.source_filter or eval_doc
        run_evaluation(eval_doc, test_set, args.vector_scope, source_filter)
        return

    if not args.skip_seed:
        print("=" * 60)
        print("  STEP 1: 文档入库")
        print("=" * 60)
        seed_manual(args.doc)

    test_set = args.test_set
    if not args.skip_qa and not test_set:
        print("\n" + "=" * 60)
        print("  STEP 2: 生成测试题")
        print("=" * 60)
        qa_result = generate_qa(test_doc, args.qa_count)
        if qa_result:
            test_set = qa_result
    if not test_set:
        test_set = "data/eval/test_set.json"

    if not args.skip_eval:
        source_filter = args.source_filter or args.doc
        print("\n" + "=" * 60)
        print("  STEP 3: RAGAS 评估")
        print("=" * 60)
        run_evaluation(eval_doc, test_set, args.vector_scope, source_filter)

    print("\n全流程完成。")


if __name__ == "__main__":
    main()
