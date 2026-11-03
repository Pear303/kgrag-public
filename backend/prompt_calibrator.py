"""提示词校准器：按手册内容适配抽取提示词的可变槽（场景A增强，2026-10-03）。

设计签认（作者拍板：A≤100k 通读 / B 槽位划分 / C1 人工过目 / D 冻结复用 / E3 双入口）：
  输入 = 默认抽取提示词（固定段 + 5 个可变槽，见 backend/extractor.py 模板）+ 手册文本
  输出 = JSON 变更单（只许改可变槽，每处带理由）→ 合成适配版提示词落盘冻结
  固定段防篡改是**结构性**的：适配版永远由 compose_extraction_prompt 从模板合成，
  校准器只产出槽值，模板本身不出现在它的可达范围内。
  C1 = 校准件落盘后须经人工过目/前端确认才进抽取（管线两旗互斥即为此设计）；
  D  = 同手册重跑直接复用冻结件，不重调校准器（重校准须显式 force）。

token 预算：中文按 1 字 ≈ 1 token 的保守上界估算；手册超预算时等距分桶采样
（保首尾、保覆盖），采样事实记入校准件（mode=sampled）。
"""
import hashlib
import json
import math
import os
import re
from datetime import datetime
from typing import Optional

from config import PROJECT_ROOT
from backend.extractor import (
    FLEX_SLOT_LIMITS,
    FLEX_SLOT_NAMES,
    compose_extraction_prompt,
)

TOKEN_BUDGET = 100_000  # 校准器单次调用的输入 token 上界（作者签认）
CALIBRATION_SCHEMA = "kg-prompt-calibration-v1"
CALIB_DIR = os.path.join(PROJECT_ROOT, "data", "manuals", "_processed",
                         "prompt_calibration")
# 2026-10-03 P0.3 发现：校准器输出格式合规率仅 1/3（code_samples 缺 reason）。
# 抗性修复＝校验失败即重试（上限见下），且**每次尝试的原始输出都落盘**（原先拒收即丢）。
CALIBRATOR_MAX_ATTEMPTS = int(os.getenv("CALIBRATOR_MAX_ATTEMPTS", "3"))

_SEP = "\n\n[[采样省略]]\n\n"

CALIBRATION_INSTRUCTION = """你是工业设备故障知识图谱的提示词校准专家。下面给你两份材料：
【材料一】默认三元组抽取提示词。它由固定段与 5 个可变槽组成——固定段（输出格式、关系元类型、
六种关系白名单、禁止臆造、负向约束）绝对不许改；可变槽可以也应该按手册特点适配：
- domain_intro：领域导语里"××手册"的设备/领域名（当前值见【可变槽当前值】）
- code_samples：代码实体样例清单（报警号/地址/参数号/端子/元件型号的示例行，格式为"   - 类别：样例 等；"共 5 行）
- fewshot：Few-shot 示例（一段"输入：\\"…\\"\\n输出：[…]"，含 meta 标记，JSON 花括号写成 {{ 与 }}）
- generic_blacklist：泛化实体黑名单（顿号分隔的泛化机型/系统词）
- include_ratio_cap："包含"占每段三元组总数的比例上限（形如 1/3）

【材料二】一本维修手册的文本（可能经等距采样）。

任务：通读手册，判断默认提示词的 5 个可变槽是否需要按这本手册微调，输出严格 JSON（不要输出其他内容）：
{"changes": [{"slot": "槽名", "new": "该槽完整新文本（换行写成 \\n）", "reason": "一句话理由"}]}
规则：
1. slot 只能是上述 5 个名字之一；不需要修改的槽不要出现在 changes 里；全部不需要改则输出 {"changes": []}
2. new 必须是该槽的完整替换文本，不是 diff；new 中不得出现 ⟦ 或 ⟧ 字符
3. code_samples 的类别五行的行式结构要保持（"   - 类别：… 等；"×5）；fewshot 的 JSON 花括号必须写成 {{ 与 }}
4. reason 引用手册中的具体证据（如该手册报警号的真实格式）
5. 固定段一个字都不许出现在 new 里
6. **每个 change 条目必须同时含 slot、new、reason 三个字段，缺一不可**——
   缺 reason 的条目会被下游校验器整条拒收（实测最常见的失败模式，请务必逐条自查）
7. code_samples 的 new 里，报警号/参数号/地址/端子/型号这五类示例行**每一行都必须以
   "   - " 开头并以"；"结尾**，共 5 行；缺行或改掉行首标记同样会被拒收

【可变槽当前值】
"""

CALIBRATION_TAIL = """

【手册文本】
"""


# ---------------- 基础工具 ----------------

def sha8(text: str) -> str:
    """文本 sha256 前 8 位十六进制（项目惯用指纹码口径）。"""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:8]


def estimate_tokens(text: str) -> int:
    """token 保守上界：中文 1 字 ≈ 1 token（只用于预算判断，不用于计费口径）。"""
    return len(text)


def calibration_dir_for(manual_path: str) -> str:
    stem = re.sub(r"[\\/:*?\"<>|]", "_", os.path.splitext(os.path.basename(manual_path))[0])
    return os.path.join(CALIB_DIR, stem)


def calibration_json_path(manual_path: str) -> str:
    return os.path.join(calibration_dir_for(manual_path), "calibration.json")


def adapted_prompt_path(manual_path: str) -> str:
    return os.path.join(calibration_dir_for(manual_path), "adapted_prompt.txt")


# ---------------- 手册文本与预算采样 ----------------

def read_manual_text(manual_path: str) -> str:
    with open(manual_path, "r", encoding="utf-8") as f:
        return f.read()


def select_calibration_text(text: str, budget: int = TOKEN_BUDGET) -> dict:
    """预算内全量；超预算等距分桶采样（保首尾）。返回 {text, mode, info}。"""
    total_chars = len(text)
    if estimate_tokens(text) <= budget:
        return {"text": text, "mode": "full",
                "info": {"manual_chars": total_chars, "selected_chars": total_chars,
                         "n_windows": 1}}
    # 超预算：预算字符里预留分隔符开销后均分给 n_windows 个等距窗口
    n_windows = math.ceil(total_chars / budget)
    sep_cost = len(_SEP) * (n_windows - 1)
    window = max(1000, (budget - sep_cost) // n_windows)
    starts = []
    if n_windows == 1:
        starts = [0]
    else:
        span = total_chars - window
        for i in range(n_windows):
            starts.append(round(i * span / (n_windows - 1)))
    # 去重相邻起点并保首尾（首窗口起点 0、末窗口起点 span ⇒ 天然覆盖头尾）
    parts, last = [], -1
    for s in starts:
        if s <= last:
            continue
        parts.append(text[s:s + window])
        last = s
    selected = _SEP.join(parts)
    # 极端情况（分隔符开销挤占）兜底截断，保证 ≤ budget
    if estimate_tokens(selected) > budget:
        selected = selected[:budget]
    return {"text": selected, "mode": "sampled",
            "info": {"manual_chars": total_chars, "selected_chars": len(selected),
                     "n_windows": len(parts)}}


# ---------------- 变更单校验器 ----------------

_RATIO_RE = re.compile(r"^\d+\s*/\s*\d+$")


def validate_changes(changes) -> dict:
    """校验校准器产出的变更单，返回 {slot: new} 增量 dict；违规直接 ValueError。"""
    if not isinstance(changes, list):
        raise ValueError(f"changes 必须是数组，得到 {type(changes).__name__}")
    updates = {}
    for item in changes:
        if not isinstance(item, dict):
            raise ValueError(f"变更项必须是对象: {item!r}")
        slot = item.get("slot")
        new = item.get("new")
        reason = item.get("reason")
        if slot not in FLEX_SLOT_NAMES:
            raise ValueError(f"未知可变槽: {slot!r}；允许: {list(FLEX_SLOT_NAMES)}")
        if slot in updates:
            raise ValueError(f"可变槽重复出现: {slot}")
        if not isinstance(new, str) or not new.strip():
            raise ValueError(f"槽 {slot} 的 new 必须是非空字符串")
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError(f"槽 {slot} 缺少 reason（C1 要求人工可读的修改理由）")
        if "⟦" in new or "⟧" in new:
            raise ValueError(f"槽 {slot} 的 new 含槽位标记字符 ⟦/⟧")
        limit = FLEX_SLOT_LIMITS[slot]
        if len(new) > limit:
            raise ValueError(f"槽 {slot} 的 new 超长: {len(new)} > 上限 {limit}")
        if slot == "include_ratio_cap" and not _RATIO_RE.match(new.strip()):
            raise ValueError(f"include_ratio_cap 必须形如 1/3，得到 {new!r}")
        if slot == "code_samples":
            lines = [ln for ln in new.splitlines() if ln.strip()]
            if len(lines) < 3 or not all(ln.lstrip().startswith("- ") for ln in lines):
                raise ValueError("code_samples 应保持 \"   - 类别：… 等；\" 的行式清单结构")
        updates[slot] = new
    return updates


# ---------------- 校准器提示词组装 ----------------

def build_calibration_prompt(selected_text: str) -> str:
    """组装校准调用的用户消息：指令 + 可变槽当前值 + 手册文本。"""
    from backend.extractor import DEFAULT_EXTRACTION_FLEX
    slots = []
    for name in FLEX_SLOT_NAMES:
        value = DEFAULT_EXTRACTION_FLEX[name]
        display = value if len(value) <= 1200 else value[:1200] + "…[截断显示]"
        slots.append(f"- {name}（上限 {FLEX_SLOT_LIMITS[name]} 字）:\n{display}")
    return (CALIBRATION_INSTRUCTION + "\n".join(slots)
            + CALIBRATION_TAIL + selected_text)


def _parse_changes_json(output: str) -> list:
    """从校准器输出提取 changes 数组（容忍 ```json 包裹）。"""
    text = output.strip()
    m = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    if m:
        text = m.group(1).strip()
    # 取首个 { 到末个 } 之间的内容，容忍前后杂讯
    i, j = text.find("{"), text.rfind("}")
    if i < 0 or j <= i:
        raise ValueError(f"校准器输出不含 JSON 对象：前 120 字={output[:120]!r}")
    data = json.loads(text[i:j + 1])
    if not isinstance(data, dict) or "changes" not in data:
        raise ValueError("校准器 JSON 缺少 changes 字段")
    return data["changes"]


def _call_llm(prompt: str, model: str = "deepseek-v4-flash",
              max_tokens: int = 8192) -> str:
    """校准调用（deepseek 单次；temperature=0 保证同输入同输出）。"""
    from backend.api_client import ApiClient
    client = ApiClient()
    response = client.chat(messages=[{"role": "user", "content": prompt}],
                           max_tokens=max_tokens)
    return response["content"]


# ---------------- 落盘与复用 ----------------

def _atomic_write(path: str, content: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        f.write(content)
    os.replace(tmp, path)


def raw_attempts_path(manual_path: str) -> str:
    """校验失败/成功的每次原始 LLM 输出落盘路径（P0.3 修复：原先拒收即丢）。"""
    return os.path.join(calibration_dir_for(manual_path), "raw_attempts.jsonl")


def _append_raw_attempt(manual_path: str, attempt: int, output: str, err) -> None:
    """把被校验器拒收的原始输出原样追加落盘（供事后分析格式缺陷形态）。"""
    rec = {
        "at": datetime.now().isoformat(timespec="seconds"),
        "attempt": attempt,
        "error": f"{type(err).__name__}: {err}"[:300],
        "output_sha8": sha8(output),
        "output_chars": len(output),
        "raw_output": output,
    }
    path = raw_attempts_path(manual_path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def load_calibration(manual_path: str, verify: bool = True) -> Optional[dict]:
    """读冻结校准件（D 纪律：存在即复用，不重调）。损坏/换件 ⇒ 报错而非静默重算。"""
    path = calibration_json_path(manual_path)
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        record = json.load(f)
    if record.get("schema") != CALIBRATION_SCHEMA:
        raise RuntimeError(f"冻结校准件 schema 不符: {path}")
    ptxt_path = adapted_prompt_path(manual_path)
    if verify:
        if not os.path.exists(ptxt_path):
            raise RuntimeError(f"冻结校准件缺 adapted_prompt.txt: {ptxt_path}")
        with open(ptxt_path, "r", encoding="utf-8") as f:
            adapted = f.read()
        if sha8(adapted) != record.get("adapted_prompt_sha8"):
            raise RuntimeError(
                f"adapted_prompt.txt 与 calibration.json 的 sha8 不符（冻结件被人改动，"
                f"请人工核查后删除重校准）: {path}")
    return record


def recalibrate_allowed(manual_path: str, force: bool) -> bool:
    if force:
        return True
    return load_calibration(manual_path, verify=False) is None


# ---------------- 主入口 ----------------

def run_calibration(manual_path: str, budget: int = TOKEN_BUDGET,
                    model: str = "deepseek-v4-flash", force: bool = False,
                    dry_run: bool = False, llm_fn=None) -> dict:
    """校准一本手册的抽取提示词。

    dry_run=True：不调 API，只返回预算/采样计划（花钱前 precheck 用）。
    force=False 且冻结件已存在：直接复用（D 纪律），llm_fn 一律不调。
    llm_fn：测试注入点，签名 (prompt:str, model:str, max_tokens:int) -> str。
    """
    frozen = load_calibration(manual_path, verify=False)
    if frozen is not None and not force:
        out = dict(frozen)
        out["reused"] = True
        return out

    manual_text = read_manual_text(manual_path)
    plan = select_calibration_text(manual_text, budget)
    plan_info = {"manual_sha8": sha8(manual_text), "manual_chars": plan["info"]["manual_chars"],
                 "selected_chars": plan["info"]["selected_chars"],
                 "n_windows": plan["info"]["n_windows"], "mode": plan["mode"],
                 "token_budget": budget, "manual_tokens_est": estimate_tokens(manual_text),
                 "model": model}
    if dry_run:
        return {"schema": CALIBRATION_SCHEMA, "dry_run": True, **plan_info}

    prompt = build_calibration_prompt(plan["text"])
    call = llm_fn if llm_fn is not None else _call_llm
    # 抗性重试：校验失败不丢输出（落 raw_attempts.jsonl），重试至上限后原样抛出最后一个错误
    attempts = []
    last_err = None
    output = None
    changes = None
    updates = None
    for attempt in range(1, CALIBRATOR_MAX_ATTEMPTS + 1):
        output = call(prompt, model, 8192)
        try:
            changes = _parse_changes_json(output)
            updates = validate_changes(changes)
            attempts.append({"attempt": attempt, "outcome": "ok",
                             "output_sha8": sha8(output), "output_chars": len(output)})
            last_err = None
            break
        except Exception as e:  # noqa: BLE001 —— 记录后重试
            last_err = e
            attempts.append({"attempt": attempt, "outcome": "rejected_by_validator",
                             "error": f"{type(e).__name__}: {e}"[:300],
                             "output_sha8": sha8(output), "output_chars": len(output)})
            _append_raw_attempt(manual_path, attempt, output, e)
            continue
    if last_err is not None:
        raise last_err

    adapted = compose_extraction_prompt(updates)
    record = {
        "schema": CALIBRATION_SCHEMA,
        "written_at": datetime.now().isoformat(timespec="seconds"),
        "manual_name": os.path.basename(manual_path),
        "manual_sha8": plan_info["manual_sha8"],
        "default_prompt_sha8": sha8(compose_extraction_prompt()),
        "calibrator_model": model,
        "mode": plan["mode"],
        "manual_chars": plan_info["manual_chars"],
        "selected_chars": plan_info["selected_chars"],
        "n_windows": plan_info["n_windows"],
        "token_budget": budget,
        "n_attempts": len(attempts),
        "attempts": attempts,
        "changes": [{"slot": c["slot"], "reason": c["reason"]}
                    for c in changes if isinstance(c, dict) and c.get("slot") in updates],
        "updates": updates,
        "adapted_prompt_sha8": sha8(adapted),
    }
    _atomic_write(calibration_json_path(manual_path),
                  json.dumps(record, ensure_ascii=False, indent=2))
    _atomic_write(adapted_prompt_path(manual_path), adapted)
    out = dict(record)
    out["adapted_prompt"] = adapted
    out["reused"] = False
    return out


def get_adapted_prompt(manual_path: str) -> str:
    """读冻结校准件的适配版提示词（供抽取链使用；先过 D 纪律一致性校验）。"""
    record = load_calibration(manual_path, verify=True)
    if record is None:
        raise RuntimeError(
            f"该手册没有冻结校准件，请先跑校准并人工过目（C1）: "
            f"{calibration_json_path(manual_path)}")
    with open(adapted_prompt_path(manual_path), "r", encoding="utf-8") as f:
        return f.read()
