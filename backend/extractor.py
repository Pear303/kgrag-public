"""LLM 自动三元组抽取与后处理（文章 3.3 / 3.4 节）。

场景A：对非结构化正文用 LLM 抽取故障知识三元组。
支持两种后端：
  - 本地 Ollama（qwen3:latest，慢但离线）
  - 在线 API（MiniMax/Claude/DeepSeek，快且质量高）

v2 改进（适配真实自然语言手册）：
  - 重写 prompt：明确禁止泛化实体、要求原文依据、强制 JSON 输出
  - 增强 industrial_parse_fn：支持 JSON / 括号 / 箭头多种格式
  - 实体归一化：全角/半角统一、去冗余标点
  - provenance：记录每个三元组来源 chunk_id
"""
import hashlib
import json
import os
import re
from typing import List, Tuple, Optional

from config import ALLOWED_RELATIONS, RELATION_ALIASES, FORBIDDEN_RELATIONS


# ================ 实体归一化 ================

def normalize_entity(text: str) -> str:
    """实体归一化：统一全角/半角、℃/°C、去首尾空格与冗余标点。"""
    text = text.strip()
    # 全角→半角括号
    text = text.replace("（", "(").replace("）", ")")
    # 统一温度符号：℃ / °C / ℃ → °C
    text = text.replace("℃", "°C").replace("°C", "°C").replace("℃", "°C")
    # 全角逗号→半角（实体内部）
    text = text.replace("，", ",")
    # 去首尾标点和多余空格
    text = re.sub(r"\s+", " ", text).strip()
    text = text.strip("，。、；：,.:; ")
    return text


# ================ 基础清洗 ================

def _clean_token(token: str) -> str:
    """清洗实体/关系的边界括号与空白。"""
    token = token.strip()
    token = token.strip("（）()【】[]<>《》\"'\"' ")
    return token.strip()


def _canonicalize_relation(rel: str) -> str:
    """将同义词关系映射为标准关系（文章 RELATION_ALIASES）。"""
    rel = _clean_token(rel)
    return RELATION_ALIASES.get(rel, rel)


def _is_forbidden_relation(rel: str) -> bool:
    return rel in FORBIDDEN_RELATIONS


# ================ 关系类型标记（P0-1：多跳语义增强）===============
# 用于标记多跳推理的语义结构：
# - parallel: 并列关系（如并列原因 ①程序错误、②设定不正确）
# - sequence: 顺序关系（如处理步骤 ①反向移动 → ②急停/G23 → ③修正返参考点）
# - conditional: 条件分支（如 ②若不能移动 → 急停/G23）
RELATION_META_TYPES = {"parallel", "sequence", "conditional", "none"}

def _infer_relation_meta(r: str, context: str = "") -> str:
    """根据关系词与上下文推断元类型（用于后处理补全）。"""
    if r in {"可能原因", "解决方法", "处理方法", "故障原因", "可能原因"}:
        return "parallel"  # 故障诊断中原因/方法默认并列
    if r in {"处理步骤", "确认方法", "检查处理方法", "检查步骤"}:
        return "sequence"  # 步骤类默认顺序
    if r in {"确认方法", "检查处理方法"} and "若" in context:
        return "conditional"
    return "none"


# 修复 9.11-3：短实体白名单——2~3 字的中文领域部件/操作名是合法实体。
# 规则5（<=3字且无允许字符→噪音）本意是滤"短截断碎片"（如"泵反(?"），
# 但会误杀领域核心部件（实测 pump 手册"叶轮"28 次、"口环"19 次、"轴颈"
# 7 次、"油泵"8 次，v3 图谱却 0 个 <=3 字实体——全被规则5 滤掉）。
# 命中白名单的短实体直接放行（宁多勿漏，具体噪声由 NOISE_ENTITIES 黑名单兜底）。
SHORT_ENTITY_WHITELIST = {
    # 泵部件
    "叶轮", "口环", "轴颈", "泵盖", "泵轴", "导叶", "蜗壳", "压盖",
    "轴承", "转子", "密封", "机封", "滤网", "油池", "油泵", "联轴器", "主轴",
    "平衡盘", "平衡鼓", "密封圈", "止推盘", "推力盘", "推力瓦",
    # 泵操作/状态
    "盘车", "暖泵", "灌泵", "排气", "试车",
    # 通用设备部件（数控/电气领域）
    "伺服", "编码器", "变频器", "继电器", "接触器", "断路器", "变压器",
    "刀架", "导轨", "丝杠", "卡盘", "导轨油", "冷却泵",
}


def _is_noise_entity(entity: str) -> bool:
    """判断是否为泛化噪音实体（v2 增强版）。

    噪音规则：
    1. 单字实体 → 噪音
    2. 在黑名单集合中 → 噪音（泛化概念/纯修饰词/低信息量术语）
    3. 括号不配对 → 噪音（截断实体如"泵反(?"）
    4. 纯裸数字 → 噪音；但带单位的数值（如"8000小时"）保留
    5. 短截断中文（<=3字且无任何允许字符） → 噪音；但命中
       SHORT_ENTITY_WHITELIST 的领域部件名放行（修复 9.11-3）
    """
    NOISE_ENTITIES = {
        # === 泛化概念（无具体信息量，保留） ===
        "设备", "系统",
        "正常", "异常", "故障", "问题", "损坏", "失效",
        "良好", "合格", "不合格",
        # === 已移除（修复 M1）：离心泵领域具体部件/介质/物理量词 ===
        # 轴承/泵体/叶轮/密封/密封圈/润滑油/介质/管道/进口/出口/
        # 过滤器/滤网/温度/振动/压力/流量/扬程/效率/电流/功率/转速 等
        # 这些是泵故障诊断的核心实体，不应被过滤。
        # 单字词（泵/反/转）已由下方 len<=1 规则统一捕获。
    }
    e = entity.strip()
    if len(e) <= 1:
        return True
    if e in NOISE_ENTITIES:
        return True

    # 规则3：括号不配对 → 截断噪音（如 "泵反(?" ）
    has_left = "(" in e
    has_right = ")" in e
    if has_left != has_right:
        return True

    # 规则4：裸数字（仅有数字和小数点）→ 噪音；带单位的保留
    if re.match(r'^\d+(\.\d+)?$', e):
        return True

    # 规则5：短截断中文（<=3字，不含任何允许符号/数字）→ 噪音
    # 修复 9.11-3：命中领域部件白名单的短实体放行（叶轮/口环/盘车…）
    if len(e) <= 3:
        if e in SHORT_ENTITY_WHITELIST:
            return False
        ALLOWED_SHORT_CHARS = set("()°C℃0123456789><")
        if not any(c in ALLOWED_SHORT_CHARS for c in e):
            return True

    return False


def _split_chain(text: str) -> List[str]:
    """按箭头/逗号切分链条，支持 A→B→C 与 A,B,C。"""
    text = text.replace("→", "->").replace("→", "->")
    parts = re.split(r"[,\n]|->", text)
    return [p for p in (i.strip() for i in parts) if p]


# ================ 后处理解析器 ================

def industrial_parse_fn(output: str, max_length: int = 128,
                        include_meta: bool = False) -> list:
    """自定义后处理解析器：把 LLM 输出转换为标准化三元组。

    支持三种输出格式：
    1. JSON 数组: [{"s":"...","r":"...","o":"...","meta":"..."}, ...]
    2. 括号格式: (实体A, 关系, 实体B)
    3. 箭头格式: 实体A -关系-> 实体B

    返回 [(subject, relation, object), ...]（include_meta=False，兼容旧调用）；
    include_meta=True 时返回 [(s, r, o, meta), ...]，meta ∈ RELATION_META_TYPES。
    仅保留 ALLOWED_RELATIONS。
    """
    triples = []

    # 格式1：尝试 JSON 解析（含 meta，见 _try_parse_json）
    json_triples = _try_parse_json(output)
    if json_triples:
        triples.extend(json_triples)

    # 格式2：括号格式 (A, 关系, B)
    bracket_pattern = re.compile(r"\(([^()]*?)[,，]\s*([^()]*?)[,，]\s*([^()]*?)\)")
    for m in bracket_pattern.finditer(output):
        s, r, o = (_clean_token(x) for x in m.groups())
        r = _canonicalize_relation(r)
        if r in ALLOWED_RELATIONS and s and o:
            triples.append((s[:max_length], r, o[:max_length], "none"))

    # 格式3：箭头格式 A -关系-> B 或 A --关系--> B
    _arrow_rels = "|".join(re.escape(r) for r in ALLOWED_RELATIONS)
    arrow_pattern = re.compile(r"(.+?)\s*[-—]+\s*(" + _arrow_rels + r")\s*[-—]*>+\s*(.+)")
    for m in arrow_pattern.finditer(output):
        s, r, o = (_clean_token(x) for x in m.groups())
        if s and o:
            triples.append((s[:max_length], r, o[:max_length], "none"))

    # 去重 + 归一化 + 噪音过滤
    seen = set()
    result = []
    for item in triples:
        s, r, o, meta = item[0], item[1], item[2], (item[3] if len(item) > 3 else "none")
        s, o = normalize_entity(s), normalize_entity(o)
        if not s or not o:
            continue
        if _is_noise_entity(s) or _is_noise_entity(o):
            continue
        if r not in ALLOWED_RELATIONS:
            continue
        if meta not in RELATION_META_TYPES:
            meta = "none"
        key = (s, r, o, meta)
        if key not in seen:
            seen.add(key)
            if include_meta:
                result.append((s, r, o, meta))
            else:
                result.append((s, r, o))
    return result


def _try_parse_json(text: str) -> list:
    """尝试从 LLM 输出中解析 JSON 格式的三元组（内部 4 元组含 meta，供 parse 归一）。"""
    triples = []
    # 尝试找到 JSON 数组（可能被 markdown 代码块包裹）
    json_match = re.search(r"```(?:json)?\s*(\[.*?\])\s*```", text, re.DOTALL)
    if not json_match:
        json_match = re.search(r"(\[\s*\{.*?\}\s*\])", text, re.DOTALL)
    if not json_match:
        return triples
    try:
        data = json.loads(json_match.group(1))
        for item in data:
            if isinstance(item, dict):
                s = item.get("s") or item.get("subject") or item.get("实体A") or ""
                r = item.get("r") or item.get("relation") or item.get("关系") or ""
                o = item.get("o") or item.get("object") or item.get("实体B") or ""
                meta = item.get("meta") or item.get("关系元类型") or "none"
                r = _canonicalize_relation(str(r))
                meta = str(meta).strip().lower()
                if meta not in RELATION_META_TYPES:
                    meta = "none"
                if s and r and o:
                    triples.append((str(s), r, str(o), meta))
            elif isinstance(item, (list, tuple)) and len(item) >= 3:
                s, r, o = str(item[0]), _canonicalize_relation(str(item[1])), str(item[2])
                meta = str(item[3]).strip().lower() if len(item) > 3 else "none"
                if meta not in RELATION_META_TYPES:
                    meta = "none"
                triples.append((s, r, o, meta))
    except (json.JSONDecodeError, TypeError):
        pass
    return triples


# ================ 抽取 Prompt（v2：适配自然语言段落）================

EXTRACTION_PROMPT_V2 = """你是工业设备故障知识图谱构建专家。请从以下维修手册文本中抽取故障诊断三元组。

【输出格式】严格输出 JSON 数组，每个元素为 {{"s":"主体","r":"关系","o":"客体"}}，不要输出其他内容。

【关系约束】只允许以下六种关系：
- 包含：部件/系统的组成与层级关系（如"润滑系统 包含 油泵"、"机床 包含 主轴"）。这是最重要的关系类型，请优先抽取所有部件的父子结构。
- 表现为：故障/状态的外在表现（如"轴承损坏 表现为 振动值逐步爬升"）
- 可能原因：导致某故障/现象的原因（如"轴承过热 可能原因 润滑油位不足"）
- 解决方法：针对某故障/原因的处理措施（如"润滑油位不足 解决方法 补充L-AN46润滑油"）
- 配套使用：两个部件/系统需要配合、关联使用的关系（如"数控系统 配套使用 伺服驱动器"、"PLC 配套使用 变频器"）
- 影响：一个部件/状态对另一个部件/状态产生的作用或后果（如"主轴振动 影响 加工精度"、"电压波动 影响 电机寿命"）

【抽取规则】
1. 实体必须是文本中出现的具体部件、具体故障现象、具体原因、具体措施，禁止抽取"泵""轴承""设备"等顶层泛化概念作为独立实体。
2. 每个三元组必须能在原文中找到依据，禁止臆造。
3. 优先抽取完整的故障链：故障现象→可能原因→解决方法。
4. 特别重视"包含"关系：任何涉及部件构成、子系统归属、零件从属的描述都必须抽取为包含三元组。
5. 对于含数值/标准的描述，将数值作为客体（如"机械密封检修周期 包含 8000小时"）。
6. 一段文本中可抽取多个三元组，尽量完整覆盖。

【待抽取文本】
{context}
"""

# v3：面向 shukongdashi 500 例手册的强化抽取。
# 核心变化：报警号/地址/端子/参数号/元件型号必须原样保留为实体；
# 抑制"数控车床/数控系统/CNC"类泛化实体；"包含"只用于真实部件组成，
# 故障/原因/措施描述必须使用 表现为/可能原因/解决方法。
# P0-1 增强：要求 LLM 标记关系元类型 (parallel/sequence/conditional/none)
# v3 模板 + 可变槽（2026-10-03 提示词自适应改造，作者签认方案 A≤100k/B/C1/D/E3）：
# 固定段 = 输出格式/meta/六类关系白名单/禁止臆造/负向约束，任何手册不许动（结构性防篡改：
#          适配版提示词永远由 compose_extraction_prompt 从本模板合成，校准器只产出槽值）；
# 可变槽 = 5 个 ⟦槽⟧ 标记，按手册特点由 prompt_calibrator 适配（默认槽值 = 拆分前 V3 原文）。
EXTRACTION_PROMPT_V3_TEMPLATE = """你是工业设备故障知识图谱构建专家。请从以下⟦domain_intro⟧文本中抽取故障诊断三元组。

【输出格式】严格输出 JSON 数组，每个元素为 {{"s":"主体","r":"关系","o":"客体","meta":"关系元类型"}}，不要输出其他内容、注释或 Markdown。

关系元类型 meta（必填，四选一）：
- parallel: 并列关系（如并列原因 ①程序错误、②存储行程极限设定不正确；并列处理 ①反向移动、②急停/G23、③修正返参考点）
- sequence: 顺序关系（如处理步骤 ①反向移动 → ②急停/G23 → ③修正返参考点）
- conditional: 条件分支（如 ②若不能移动 → 急停/G23；条件分支通常含"若/如果/则"）
- none: 无明显多跳结构（如单纯的包含关系、单一表现/原因/方法）

【关系约束】只允许以下六种关系：
- 包含：真实的部件-子部件/系统-子系统组成关系（如"主轴驱动系统 包含 主轴驱动器"、"FANUC 0系统 包含 输入地址X018-5"）。禁止用"包含"表达故障现象、处理步骤、参数含义、类别归属等描述性内容；禁止把位定义表逐位展开（如"DGN700 包含 BIT3"）；禁止把元器件清单逐项平铺（除非该元件是故障案例中的关键元件）。
- 表现为：故障/状态的外在表现（如"主轴停止 表现为 换刀时主轴停止"）
- 可能原因：导致故障/现象的原因（如"DNC通信错误 可能原因 通信参数设置不当"）
- 解决方法：处理/排查/维修措施（如"通信参数设置不当 解决方法 重新设置波特率和数据位"）
- 配套使用：需要配合使用的部件/端子/板卡（如"电源模块 配套使用 控制板"）
- 影响：一个部件/状态对另一个部件/状态的作用（如"伺服电源无法接通 影响 机床无法起动"）

【代码实体硬性规则】
1. 报警号、输入/输出地址、诊断参数号、参数号、端子号、元件型号必须作为实体原样保留，禁止省略、丢弃、改写或只放进长句里：
⟦code_samples⟧
2. 输出前把所有代码写成规范半角形式：全角转半角；去掉代码内部空格（如"X018 - 5"写成"X018-5"、"DGN700、701"写成"DGN700/DGN701"）；地址位标记统一用小数点（如"G008#0"写成"G008.0"）。同一个代码在全篇只保留一种写法。
3. 当原文描述"某信号的地址为X018-5"时，抽取 ("信号名", 包含, "输入地址X018-5") 或 ("信号名", 包含, "X018-5")，保证 X018-5 作为连续实体出现。
4. 当原文出现"更换D12（A169）"时，实体写"续流二极管D12(A169)"或"A169"，禁止只写"二极管"而丢掉型号。

【Few-shot 示例】（含 meta 标记）
⟦fewshot⟧

【负向约束】（借鉴 KG_RAG）
11. 只报告原文中出现的具体名称，不要报告类别名本身："端子""地址""报警号""信号"这类泛化类别词禁止作为实体；必须带具体编号（如"端子36""地址X018-5""报警ALM414"）。
12. 禁止把位定义表逐位展开（如"DGN700 包含 BIT3"）；禁止把元器件清单逐项平铺（除非该元件是故障案例中的关键元件）。

【泛化实体抑制】
5. 实体必须是手册中出现的具体对象：具体部件、具体故障现象、具体原因、具体措施、具体代码。禁止把"⟦generic_blacklist⟧"等泛化机型/系统词作为三元组主体或客体，除非原文是明确的部件组成且客体是具体部件/型号，且每段不超过 1 条。
6. 禁止"数控机床 包含 加工中心"这类类别关系；禁止"系统 包含 输入/输出装置"这类空泛组成。

【抽取规则】
7. 优先抽取完整故障链：故障现象（表现为）→ 可能原因（可能原因）→ 解决方法（解决方法），再抽取部件组成（包含）与配套关系。
8. 每个三元组必须能在原文中找到依据，禁止臆造；一段文本可抽取多个三元组，尽量完整覆盖。
9. 含数值/标准的描述将数值作为客体保留（如"通信参数 包含 波特率9600"）。
10. "包含"关系必须克制：每段文本中"包含"数量不超过该段三元组总数的 ⟦include_ratio_cap⟧；诊断参数/地址表只抽 1~2 条汇总（如"F0系统 包含 诊断参数DGN700"），禁止 BIT0~BIT7 逐位抽取。

【待抽取文本】
{context}
"""

# 可变槽名（顺序即合成顺序）与长度上限（校准器校验用）
FLEX_SLOT_NAMES = ("domain_intro", "code_samples", "fewshot",
                   "generic_blacklist", "include_ratio_cap")
FLEX_SLOT_LIMITS = {
    "domain_intro": 200,
    "code_samples": 4000,
    "fewshot": 8000,
    "generic_blacklist": 600,
    "include_ratio_cap": 20,
}

# 默认槽值（= 拆分前的 V3 原文，勿改——改了会破坏字节守恒门）
DEFAULT_EXTRACTION_FLEX = {
    "domain_intro": "数控机床维修手册",
    "code_samples": (
        "   - 报警号：ALM414、ALM100、NC2020、SV001、AL-12、T08000001 等；\n"
        "   - 输入/输出地址：X018-5、X1000.4、X008.4、G008.0、Y026-0、DGN700、DGN701 等；\n"
        "   - 参数号：P/S90、P/S00、PRM2023、PRM2022 等；\n"
        "   - 端子/接插件：CP1、JR11、CH20、F11、F12、1U1/1V1/1W1 等；\n"
        "   - 元件型号：A169、Q11、D12、6SN1123-1AA00-0AA0、LM311、DV47 等。"
    ),
    "fewshot": (
        "输入：\"存储行程限位报警(软件限位报警)。原因:①程序错误。②存储行程极限设定不正确。"
        "处理方法:①在相反方向上移动机床。②若不能移动机床，按下急停按钮或执行G23指令。"
        "③若设定值有错误，则修正设定值，并返回到参考点。\"\n"
        "输出：[\n"
        "  {{\"s\":\"存储行程限位报警(软件限位报警)\",\"r\":\"可能原因\",\"o\":\"程序错误\",\"meta\":\"parallel\"}},\n"
        "  {{\"s\":\"存储行程限位报警(软件限位报警)\",\"r\":\"可能原因\",\"o\":\"存储行程极限设定不正确\",\"meta\":\"parallel\"}},\n"
        "  {{\"s\":\"存储行程限位报警(软件限位报警)\",\"r\":\"解决方法\",\"o\":\"在相反方向上移动机床\",\"meta\":\"sequence\"}},\n"
        "  {{\"s\":\"存储行程限位报警(软件限位报警)\",\"r\":\"解决方法\",\"o\":\"若不能移动机床，按下急停按钮或执行G23指令\",\"meta\":\"conditional\"}},\n"
        "  {{\"s\":\"存储行程限位报警(软件限位报警)\",\"r\":\"解决方法\",\"o\":\"若设定值有错误，则修正设定值，并返回到参考点\",\"meta\":\"sequence\"}}\n"
        "]"
    ),
    "generic_blacklist": "数控车床、数控系统、数控机床、加工中心、CNC、伺服电动机",
    "include_ratio_cap": "1/3",
}


def compose_extraction_prompt(flex: Optional[dict] = None) -> str:
    """把可变槽填入 V3 模板，返回仍带 {context} 槽位的提示词模板。

    flex 为 None/缺槽时用默认槽值 ⇒ 合成结果与拆分前的 EXTRACTION_PROMPT_V3
    逐字节一致（守恒门见 tests/test_prompt_calibrator.py）。
    固定段不在 flex 可达范围内 ⇒ 校准器结构性无法篡改固定段。
    """
    values = dict(DEFAULT_EXTRACTION_FLEX)
    if flex:
        unknown = sorted(set(flex) - set(FLEX_SLOT_NAMES))
        if unknown:
            raise ValueError(f"未知可变槽: {unknown}；允许: {list(FLEX_SLOT_NAMES)}")
        for k, v in flex.items():
            if v is not None:
                values[k] = v
    prompt = EXTRACTION_PROMPT_V3_TEMPLATE
    for name in FLEX_SLOT_NAMES:
        prompt = prompt.replace(f"⟦{name}⟧", values[name])
    if "⟦" in prompt:
        raise ValueError("模板存在未消费的槽位标记，槽位定义与模板不一致")
    return prompt


# v3 现行默认（模板 + 默认槽合成；与拆分前逐字节一致）
EXTRACTION_PROMPT_V3 = compose_extraction_prompt()

# 兼容旧版 prompt（表格类手册仍可用）
EXTRACTION_PROMPT_V1 = (
    "请从以下文本抽取工业设备故障知识三元组，每行一个，格式严格为 (实体A, 关系, 实体B)。\n"
    "强制要求：\n"
    "1. 提取完整链路：部件 -> 表现为 -> 故障现象 -> 可能原因 -> 解决方法。\n"
    "2. 只允许四种关系：包含、表现为、可能原因、解决方法。\n"
    "3. 禁止提取 MTBF、环境温度、湿度等统计/噪音关系。\n"
    "文本：\n{context}\n"
)


# ================ 断点续跑 ================

CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "manuals", "_processed")


# ================ 抽取入口 ================

def extract_triples_with_llm(text: str, model: str = "qwen3:latest",
                              base_url: str = "http://localhost:11434",
                              use_v2_prompt: bool = True,
                              include_meta: bool = False,
                              prompt: Optional[str] = None) -> list:
    """用本地 Ollama 模型抽取三元组（直连 /api/generate）。

    v2 改进：使用更强的 prompt + JSON 输出 + 噪音过滤。
    include_meta=True 时返回 [(s, r, o, meta), ...]。
    prompt 传入时覆盖内置模板（如校准器合成的适配版，须仍含 {context} 槽）。
    """
    import requests
    if prompt is not None:
        prompt_template = prompt
    else:
        prompt_template = EXTRACTION_PROMPT_V2 if use_v2_prompt else EXTRACTION_PROMPT_V1
    prompt = prompt_template.format(context=text)

    url = base_url.rstrip("/") + "/api/generate"
    resp = requests.post(url, json={"model": model, "prompt": prompt,
                                    "stream": False, "options": {"temperature": 0}},
                         timeout=300)
    resp.raise_for_status()
    return industrial_parse_fn(resp.json().get("response", ""), include_meta=include_meta)


def extract_triples_online(text: str, llm=None,
                           prompt: Optional[str] = None,
                           include_meta: bool = False) -> list:
    """用远程强模型（deepseek-v4-flash / MiniMax）抽取三元组。

    环境变量 EXTRACT_LLM_PROVIDER 控制后端：
    - deepseek（默认）: deepseek-v4-flash 通过 ApiClient 单例
    - minimax: MiniMax-Text-01 @ api.minimax.chat

    deepseek 有 reasoning 模式，max_tokens=8192 预留思考空间。
    include_meta=True 时返回 [(s, r, o, meta), ...]。
    """
    provider = os.getenv("EXTRACT_LLM_PROVIDER", "deepseek")

    if provider == "minimax":
        from openai import OpenAI
        client = OpenAI(
            api_key=os.getenv("MINIMAX_API_KEY", ""),
            base_url=os.getenv("MINIMAX_BASE_URL", "https://api.minimax.chat/v1"),
        )
        model = os.getenv("MINIMAX_LLM_MODEL", "MiniMax-Text-01")
        max_tokens = 4096

        prompt = (prompt or EXTRACTION_PROMPT_V2).format(context=text)
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=max_tokens,
        )
        output = resp.choices[0].message.content or ""
    else:
        from backend.api_client import ApiClient
        client = ApiClient()
        _prompt = (prompt or EXTRACTION_PROMPT_V2).format(context=text)
        # 修复 9.11-3 附带发现：max_tokens 8192 会截断长 chunk（2000+ 字）的
        # JSON 数组输出 → 解析 0 条（v4 重抽时 chunk 2/4/6 首轮全部 0 条）。
        # 提到 16384 后同一 chunk 一次成功（78/57/35 条）。
        # LLM 磁盘缓存（开发/构建场景，env ENABLE_LLM_CACHE=1 启用，默认关）：
        # 同一 prompt 命中不重复付费；禁止接入评测答案生成链（R10 新鲜性）。
        from backend.llm_cache import get_cache as _get_llm_cache
        _cache = _get_llm_cache()
        _cached = _cache.get("deepseek-v4-flash", _prompt, 0.0, 16384) if _cache else None
        if _cached is not None:
            return industrial_parse_fn(_cached, include_meta=include_meta)
        response = client.chat(
            messages=[{"role": "user", "content": _prompt}], max_tokens=16384
        )
        output = response['content']
        if _cache:
            _cache.put("deepseek-v4-flash", _prompt, 0.0, 16384, output)

    return industrial_parse_fn(output, include_meta=include_meta)


def extract_from_chunks(chunks, llm=None, use_online: bool = True,
                        model: str = "qwen3:latest",
                        base_url: str = "http://localhost:11434",
                        verbose: bool = True,
                        extraction_prompt: Optional[str] = None) -> List[dict]:
    """批量从 chunks 抽取三元组，带 provenance（来源追踪）和断点续跑。

    参数:
        chunks: parser.parse_manual_chunks() 返回的 Chunk 列表
        llm: LlamaIndex LLM 实例（在线模式需传入）
        use_online: True=用在线模型, False=用本地 Ollama
        verbose: 是否打印进度
        extraction_prompt: 覆盖内置提示词（校准器合成的适配版，须含 {context} 槽）

    返回:
        [{"s":..., "r":..., "o":..., "meta":..., "source":..., "chunk_id":...}, ...]
    """
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    all_results = []
    skipped = 0
    for idx, chunk in enumerate(chunks):
        source_basename = re.sub(r'[\\/:*?"<>|]', '_', os.path.basename(chunk.source))
        marker_file = os.path.join(CHECKPOINT_DIR, f"{source_basename}_{chunk.chunk_id}.done")
        _ = hashlib.md5(f"{chunk.source}{chunk.chunk_id}".encode('utf-8')).hexdigest()

        if os.path.exists(marker_file):
            if verbose:
                print(f"  [SKIP] chunk {idx+1}/{len(chunks)} 已处理过")
            skipped += 1
            continue

        if verbose:
            print(f"  抽取 chunk {idx+1}/{len(chunks)} ({len(chunk.text)} 字)...")
        try:
            if use_online:
                triples = extract_triples_online(chunk.text, llm=llm, include_meta=True,
                                                 prompt=extraction_prompt)
            else:
                triples = extract_triples_with_llm(
                    chunk.text, model=model, base_url=base_url, use_v2_prompt=True,
                    include_meta=True, prompt=extraction_prompt,
                )
        except Exception as e:
            if verbose:
                print(f"    [WARN] chunk {idx+1} 抽取失败: {e}")
            continue

        for item in triples:
            s, r, o = item[0], item[1], item[2]
            meta = item[3] if len(item) > 3 else "none"
            all_results.append({
                "s": s, "r": r, "o": o, "meta": meta,
                "source": chunk.source,
                "chunk_id": chunk.chunk_id,
            })
        if verbose:
            print(f"    → {len(triples)} 条三元组")

        # 修复 9.13-⑤：仅"成功产出三元组"才写断点标记——此前抽取返回 0 条
        # （非异常，如 max_tokens 截断致 JSON 解析失败）也写 .done，重跑会
        # 静默跳过该 chunk，缺陷永远得不到补抽。0 条不写标记，下次重试。
        if triples:
            with open(marker_file, 'w', encoding='utf-8') as f:
                _ = f.write('')
        else:
            print(f"    [WARN] chunk {idx+1} 抽取 0 条，不写断点标记（重跑会重试）")

    if verbose:
        print(f"  总计: {len(all_results)} 条三元组 (来自 {len(chunks)} chunks, 跳过 {skipped})")

    # 语义去重：按归一化实体名合并近似重复的三元组
    all_results = _dedup_semantic(all_results, verbose=verbose)
    return all_results


def _dedup_semantic(triples: List[dict], verbose: bool = False) -> List[dict]:
    """按 normalize_entity 对实体做语义去重，合并近似重复的三元组（保留 meta）。

    策略：
    1. 对每条三元组实体做 normalize_entity
    2. 按 (归一化s, r, 归一化o, meta) 去重
    3. 对归一化后相同但原文不同的实体，保留最长/最完整的原文
    """
    if not triples:
        return triples

    # Step1: 按归一化key分组，收集所有变体
    groups: dict = {}
    for t in triples:
        ns = normalize_entity(t["s"])
        no = normalize_entity(t["o"])
        meta = t.get("meta", "none")
        key = (ns, t["r"], no, meta)
        if key not in groups:
            groups[key] = {"s_variants": {}, "o_variants": {}, "sources": set(), "chunk_ids": set()}
        groups[key]["s_variants"][t["s"]] = len(t["s"])
        groups[key]["o_variants"][t["o"]] = len(t["o"])
        groups[key]["sources"].add(t["source"])
        groups[key]["chunk_ids"].add(t["chunk_id"])

    # Step2: 选最优实体名（最长者），合并 source/chunk_id
    merged = []
    dupes = 0
    for (ns, r, no, meta), grp in groups.items():
        best_s = max(grp["s_variants"], key=grp["s_variants"].get)
        best_o = max(grp["o_variants"], key=grp["o_variants"].get)
        merged.append({
            "s": best_s, "r": r, "o": best_o, "meta": meta,
            "source": sorted(grp["sources"])[0] if grp["sources"] else "",
            "chunk_id": sorted(grp["chunk_ids"])[0] if grp["chunk_ids"] else "",
        })
        variants = (len(grp["s_variants"]), len(grp["o_variants"]))
        if variants[0] > 1 or variants[1] > 1:
            dupes += 1

    if verbose:
        print(f"  语义去重: {len(triples)} → {len(merged)} 条 ({dupes} 组合并)")
    return merged


# ================ 兼容旧接口 ================

def build_kg_extractor(llm):
    """构造 LlamaIndex SimpleLLMPathExtractor（文章 3.3 节，兼容旧调用）。"""
    from llama_index.core.indices.property_graph import SimpleLLMPathExtractor
    return SimpleLLMPathExtractor(
        llm=llm,
        extract_prompt=EXTRACTION_PROMPT_V1,
        parse_fn=industrial_parse_fn,
        max_paths_per_chunk=30,
    )
