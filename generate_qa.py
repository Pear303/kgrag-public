"""用 deepseek-v4-flash 从手册内容自动生成 RAGAS 测试题（全量覆盖版）。

修复 9.13-①：不再只基于手册"前 N 字符"生成测试题——把手册全量按 chunk
分片，每片内容 ≤ --batch-chars（默认 24000 字符，单次请求 token 规模可控），
各片分别生成题目后合并去重，保证测试题覆盖全手册而非开头片段。

用法：
  python generate_qa.py --manual data/manuals/pump_manual.md
  python generate_qa.py --manual data/manuals/pump_manual.md --num 18 \
      --batch-chars 24000 --output data/eval/test_set_pump.json
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from backend import parser
from backend.api_client import get_client

QA_PROMPT = """你是一名工业设备故障诊断专家。请根据以下【手册内容】，生成 {num} 道测试题。

要求：
1. 题型分四类，均匀分布：
   - single-hop：单一事实检索（如"XX的维护周期是多长？"）
   - multi-hop：跨句推理（如"YY故障的可能原因和解决方法分别是什么？"）
   - out-of-context：手册中不包含的问题（如"锅炉过热器的维护周期？"），占 {ooc} 题
   - numeric：答案必须为"单一数值+单位"（可含 >/</≥/≤/范围"a～b"），如"轴承座温度升高超过多少℃视为异常？"→">80℃"，占 {numeric} 题
2. 每题需包含 question 和 ground_truth（正确答案）
3. ground_truth 必须能在手册中找到依据，不能编造
4. **严格格式要求**：ground_truth 中不得包含换行符，多要点用中文分号分隔
5. 输出格式为 **JSONL**（每行一个 JSON 对象），格式如下：
   {{"id": "auto_001", "question": "...", "ground_truth": "...", "type": "single-hop|multi-hop|out-of-context|numeric", "source_manual": "{manual_name}"}}

【手册内容】
{content}

直接输出 JSONL，不要解释、不要 markdown 代码块包裹。"""

# 单次请求喂给模型的内容上限（字符）。中文字符 ≈ 1 token/字，
# 24000 字符 ≈ 24k token，远低于上下文窗口，避免"几百 k token 单请求"。
DEFAULT_BATCH_CHARS = 24000


def _split_batches(chunks, batch_chars: int) -> list:
    """把 chunk 列表按累计字符数分成多片，每片内容 ≤ batch_chars。

    返回的每片是 chunk 列表；全手册必然被完整覆盖（不截断任何字符）。
    """
    batches = []
    cur, cur_len = [], 0
    for c in chunks:
        if cur and cur_len + len(c.text) > batch_chars:
            batches.append(cur)
            cur, cur_len = [], 0
        cur.append(c)
        cur_len += len(c.text)
    if cur:
        batches.append(cur)
    return batches


def _call_llm(prompt: str, max_tokens: int) -> str:
    client = get_client()
    response = client.chat(
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=max_tokens,
    )
    return (response.get("content") or "").strip()


def _parse_items(raw: str) -> list:
    """解析 LLM 输出：优先按行 JSONL，失败回退 JSON 数组（含 markdown 包裹）。"""
    items = []
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1].rsplit("```", 1)[0]
    lines = [l.strip() for l in raw.strip().split("\n") if l.strip()]
    if lines and lines[0].startswith("{"):
        for line in lines:
            try:
                items.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    if not items:
        import re
        raw_compact = re.sub(r'"\s*\n\s*"', '"; "', raw)
        raw_compact = raw_compact.replace('\n', '；').replace('\r', '')
        try:
            items = json.loads(raw_compact)
        except json.JSONDecodeError:
            m = re.search(r'\[.*\]', raw_compact, re.DOTALL)
            if m:
                try:
                    items = json.loads(m.group())
                except json.JSONDecodeError:
                    pass
    return items


def generate_qa(manual_path: str, num: int = 12, batch_chars: int = DEFAULT_BATCH_CHARS) -> list:
    """从手册全量内容分片生成测试题，合并去重后返回。

    Args:
        manual_path: 手册文件路径
        num: 目标题目数
        batch_chars: 单片内容字符上限（默认 24000）

    Returns:
        list[dict]，每项含 id/question/ground_truth/type/source_manual
    """
    manual_name = os.path.basename(manual_path)
    chunks = parser.parse_manual_chunks(manual_path)
    batches = _split_batches(chunks, batch_chars)
    if not batches:
        return []
    total_chars = sum(len(c.text) for c in chunks)

    per_batch = max(1, round(num / len(batches)))
    ooc_per_batch = max(1, per_batch // 5)
    numeric_per_batch = max(1, per_batch // 5)

    items = []
    for bi, batch in enumerate(batches):
        content = "\n\n".join(c.text for c in batch)
        prompt = QA_PROMPT.format(
            num=per_batch, ooc=ooc_per_batch, numeric=numeric_per_batch,
            manual_name=manual_name, content=content,
        )
        try:
            raw = _call_llm(prompt, max_tokens=4096)
        except Exception as e:
            print(f"  [WARN] 第 {bi + 1}/{len(batches)} 片生成失败: {e}")
            continue
        batch_items = _parse_items(raw)
        for it in batch_items:
            if not it.get("id"):
                it["id"] = f"auto_{len(items):03d}"
            items.append(it)
        print(f"  [片 {bi + 1}/{len(batches)}] 覆盖 {len(content)} 字符 → {len(batch_items)} 题")

    # 按 question 去重（保序），重新编号
    seen = set()
    deduped = []
    for it in items:
        q = (it.get("question") or "").strip()
        if q and q not in seen:
            seen.add(q)
            it["source_manual"] = manual_name
            deduped.append(it)
    for i, it in enumerate(deduped):
        it["id"] = f"auto_{i:03d}"

    if len(deduped) > num:
        print(f"  [WARN] 合并去重后 {len(deduped)} 题 > 目标 {num}，截取前 {num} 题")
        deduped = deduped[:num]

    print(f"  全手册 {total_chars} 字符，分 {len(batches)} 片全部覆盖 → {len(deduped)} 题")
    return deduped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manual", required=True)
    ap.add_argument("--num", type=int, default=12)
    ap.add_argument("--batch-chars", type=int, default=DEFAULT_BATCH_CHARS,
                    help="单片内容字符上限（默认 24000），保证单次请求 token 规模可控")
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    items = generate_qa(args.manual, args.num, args.batch_chars)
    if not items:
        print("生成失败，无有效输出。")
        return

    output_path = args.output or f"data/eval/test_set_{os.path.basename(args.manual).replace('.md', '')}.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)
    print(f"已生成 {len(items)} 题 → {output_path}")


if __name__ == "__main__":
    main()
