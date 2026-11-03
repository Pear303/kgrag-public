# -*- coding: utf-8 -*-
"""厂商/系统型号检测门（2026-10-12，v2 重写）。

目的：拦截"跨厂商/跨型号"的超纲问题（如问西门子840D，但知识库只有FANUC）。

原理（v2 新设计）：
1. 从问题中提取厂商/系统关键词（通用模式匹配，不依赖穷举白名单）
2. 在**图谱实体名**里检索这些关键词（`Entity {source_manual}.name CONTAINS term`）
3. 如果问题提到了厂商/系统，但**图谱里没有这个词条** → 拒答

⚠️ 这条门的判据是"图谱抽没抽出这个型号"，**不是"手册正文里有没有"**。实测（2026-10-17，
files/_oneoff/vendorgate/_vendorgate_forensics_20261017.py）：被喂的 500 案例手册里
「880M」出现 4 次、「TNC355/360/370」各 2 次，而图谱实体名里一个都没有 ⇒ 门会拒答，
而答案其实逐字在书里。所以对外话术只可说"图谱未收录该词条"，不许说"手册/知识库没有"。
要把判据换成"正文里有没有"，那是改线上拒答行为，须另立预注册（作者点头才做）。

用法：
    from backend.vendor_gate import vendor_gate_hook
    refusal_reason, record = vendor_gate_hook(question, refusal_reason)
"""
import os
import re
import json
import time
import logging
import hashlib
from datetime import datetime
from typing import Optional, Tuple, Dict, Any, Set

logger = logging.getLogger("kgrag.vendor_gate")

# ── 中英文厂商名映射 ─────────────────────────────────────────────────────────
_VENDOR_ALIAS = {
    "西门子": "SIEMENS",
    "三菱": "MITSUBISHI",
    "海德汉": "HEIDENHAIN",
    "发那科": "FANUC",
    "法纳克": "FANUC",
    "马扎克": "MAZAK",
    "大隈": "OKUMA",
    "日立": "HITACHI",
    "广数": "GSK",
    "华中": "HNC",
    "凯恩帝": "KND",
    "安川": "YASKAWA",
}

# ── 通用厂商/系统提取模式 ────────────────────────────────────────────────────
# 英文厂商（不要求 \b，因为可能紧跟中文）
_VENDOR_EN_PATTERN = (
    r"(FANUC|SIEMENS|MITSUBISHI|HEIDENHAIN|FAGOR|HASS|YASKAWA|MAZAK|OKUMA|HITACHI)"
)
_VENDOR_CN_PATTERN = (
    r"(GSK|HNC|KND)"
)
_VENDOR_CN_NAME_PATTERN = (
    r"(西门子|三菱|海德汉|发那科|法纳克|马扎克|大隈|日立|广数|新代|华中|凯恩帝|安川)"
)

# 系统型号通用模式
# ⚠️ 2026-10-27 修：每条都加数字边界。旧尺 `\d{3}` 无边界，会把报警号"2011"
# 咬走前三位当系统型号（→ token "201"），再靠 `e.name CONTAINS '201'` 撞上
# DGN2015/N2015 等无关实体而"证明"该型号存在。边界保证只整词匹配。
_SYSTEM_PATTERNS = [
    r"(?<![0-9A-Za-z])M\d{2,4}(?![0-9A-Za-z])",        # M64, M700
    r"(?<![0-9A-Za-z])TNC\s*\d{2,3}(?![0-9A-Za-z])",   # TNC 155, TNC355
    r"(?<![0-9A-Za-z])FS\s*\d{1,2}(?![0-9A-Za-z])",    # FS 6, FS 16
    r"(?<![0-9A-Za-z])\d{3}[MmDdTtCc](?![0-9A-Za-z])", # 810M, 840D, 850M
    r"(?<![0-9A-Za-z])\d{1,2}[iI](?![0-9A-Za-z])",     # 0i, 16i, 18i
    r"(?<!\d)\d{3}(?!\d)",                             # 983, 810, 840（纯三位数，整串）
]


def _body_fallback_enabled() -> bool:
    """手册正文回退尺开关（默认开）。=0 时逐字节回到旧行为，便于成对复算。"""
    return os.getenv("VENDOR_GATE_BODY_FALLBACK", "1").strip() != "0"


_manual_text_cache = {}


def _manual_text(source_manual: str) -> str:
    """当前服务手册的正文（判"手册到底写没写"用这把尺）。进程内缓存。"""
    name = (source_manual or "").strip()
    if not name:
        return ""
    if name in _manual_text_cache:
        return _manual_text_cache[name]
    from config import MANUALS_DIR
    path = os.path.join(MANUALS_DIR, os.path.basename(name))
    text = ""
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
    except OSError as e:
        logger.warning("厂商门：手册正文读不到（%s）: %s —— 本次只用实体名尺", path, e)
    _manual_text_cache[name] = text
    return text


def _body_has_term(text: str, term: str) -> bool:
    """整词字面匹配：左右不挨字母数字，避免 840 冒充 840D、201 冒充 2011。"""
    if not text or not term:
        return False
    pat = r"(?<![0-9A-Za-z])" + re.escape(term) + r"(?![0-9A-Za-z])"
    return re.search(pat, text, re.IGNORECASE) is not None


def _get_neo4j_driver():
    """获取 Neo4j 驱动（延迟导入，避免循环依赖）。"""
    from neo4j import GraphDatabase

    uri = os.getenv("SERVING_NEO4J_URI") or os.getenv("NEO4J_URI", "bolt://localhost:7688")
    user = os.getenv("SERVING_NEO4J_USER") or os.getenv("NEO4J_USER", "neo4j")
    password = (os.getenv("SERVING_NEO4J_PASSWORD") or os.getenv("NEO4J_PASSWORD")
            or "")   # 不再内置默认口令：缺了就由连库失败响亮报出来

    return GraphDatabase.driver(uri, auth=(user, password))


def _extract_vendor_system(text: str) -> Tuple[Set[str], Set[str]]:
    """从文本中提取厂商/系统关键词。

    返回：(vendors, systems)
    """
    vendors = set()
    systems = set()

    # 提取厂商
    for pattern in [_VENDOR_EN_PATTERN, _VENDOR_CN_PATTERN, _VENDOR_CN_NAME_PATTERN]:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            vendors.add(match.group())

    # 提取系统型号
    for pattern in _SYSTEM_PATTERNS:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            systems.add(match.group())

    return vendors, systems


def _get_vendor_search_terms(vendor: str) -> Set[str]:
    """获取厂商的搜索词（中文和英文）。"""
    terms = {vendor}
    # 如果是中文，加上对应的英文
    if vendor in _VENDOR_ALIAS:
        terms.add(_VENDOR_ALIAS[vendor])
    # 如果是英文，加上对应的中文
    for cn, en in _VENDOR_ALIAS.items():
        if en == vendor.upper():
            terms.add(cn)
    return terms


def _get_system_search_variants(system: str) -> Set[str]:
    """获取系统型号的搜索变体（处理空格）。"""
    variants = {system}
    # 去掉空格
    variants.add(system.replace(" ", ""))
    # 加空格（如 TNC155 → TNC 155）
    match = re.match(r"^([A-Za-z]+)(\d+)$", system)
    if match:
        variants.add("{} {}".format(match.group(1), match.group(2)))
    return variants


def _query_in_graph(session, source_manual: str, term: str) -> bool:
    """查询知识库中是否存在包含指定词的实体。"""
    query = """
        MATCH (e:Entity {source_manual: $src})
        WHERE e.name CONTAINS $term
        RETURN count(e) as cnt
    """
    result = session.run(query, src=source_manual, term=term)
    return result.single()["cnt"] > 0


def check_vendor_support(question: str) -> Optional[Dict[str, Any]]:
    """检查问题中的厂商/系统是否在知识库支持范围内。

    返回：
        None - 通过（不拒答）
        dict - 拒答信息 {"reason": str, "detected_vendor": str, ...}
    """
    # 提取厂商/系统
    vendors, systems = _extract_vendor_system(question)

    # 没有提取到厂商/系统 → 放行
    if not vendors and not systems:
        return None

    # 获取当前服务手册
    source_manual = os.getenv("SERVING_MANUAL", "")

    try:
        driver = _get_neo4j_driver()
        body = _manual_text(source_manual) if _body_fallback_enabled() else ""
        with driver.session(database="neo4j") as session:
            # 检查厂商是否存在（用中英文双语搜索）
            for vendor in vendors:
                search_terms = _get_vendor_search_terms(vendor)
                found = False
                for term in search_terms:
                    # 正文整词命中 = 手册确实写了该厂商（旧尺只查实体名，
                    # 手册写了但没成为实体名的型号会被误报"未收录"）
                    if _body_has_term(body, term) or _query_in_graph(session, source_manual, term):
                        found = True
                        break
                if not found:
                    return {
                        "reason": "图谱中未收录「{}」的词条（不代表手册没写）".format(vendor),
                        "detected_vendor": vendor,
                    }

            # 检查系统型号是否存在（只检查具体型号，不检查纯数字）
            for system in systems:
                # 跳过纯三位数（可能是误匹配）
                if re.match(r"^\d{3}$", system):
                    continue

                variants = _get_system_search_variants(system)
                found = False
                for variant in variants:
                    if _body_has_term(body, variant) or _query_in_graph(session, source_manual, variant):
                        found = True
                        break
                if not found:
                    return {
                        "reason": "图谱中未收录「{}」的词条（不代表手册没写）".format(system),
                        "detected_vendor": list(vendors)[0] if vendors else None,
                        "detected_system": system,
                    }

            return None  # 通过

    except Exception as e:
        logger.warning("厂商门查询失败（降级放行）: %s", e)
        return None


def vendor_gate_hook(question: str, refusal_reason: Optional[dict]) -> Tuple[Optional[dict], Optional[dict]]:
    """厂商门钩子（与 sent_cov_gate_hook 同风格）。

    返回 (refusal_reason_after, record|None)。
    """
    # 如果已经被其他门拒答，不再检查
    if refusal_reason is not None:
        return refusal_reason, None

    # 检查开关
    enabled = os.getenv("OOC_VENDOR_GATE", "1") == "1"
    if not enabled:
        return refusal_reason, None

    record = {
        "ts": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "q_hash": hashlib.sha256(str(question).encode("utf-8")).hexdigest()[:16],
        "gate": "vendor",
    }

    try:
        result = check_vendor_support(question)
        if result:
            record.update({
                "refused": True,
                "reason": result["reason"],
                "vendor": result.get("detected_vendor"),
                "system": result.get("detected_system"),
            })
            refusal_reason = {
                "vendor_gate": True,
                "reason": result["reason"],
                "detected_vendor": result.get("detected_vendor"),
            }
        else:
            record["refused"] = False
    except Exception as e:
        logger.warning("厂商门异常（降级放行）: %s", e)
        record["error"] = str(e)[:100]

    # 写日志（异步，不阻塞）
    try:
        _write_log(record)
    except Exception:
        pass

    return refusal_reason, record


def _write_log(record: dict):
    """写厂商门日志。"""
    log_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "logs"
    )
    if not os.path.isdir(log_dir):
        os.makedirs(log_dir)
    logfile = os.path.join(log_dir, "vendor_gate_%s.jsonl" % datetime.now().strftime("%Y%m%d"))
    with open(logfile, "a", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


# ── 自检函数（用于测试）───────────────────────────────────────────────────────
def selftest():
    """自检：验证厂商门基本功能。"""
    import sys
    sys.stdout.reconfigure(encoding="utf-8")

    print("=== 厂商门自检 v2 ===")

    test_cases = [
        ("GSK983系统如何回零", "应拒答"),
        ("西门子840D主轴定向怎么设置", "应拒答"),
        ("FANUC 0i-MF 怎么回零点", "应放行"),
        ("三菱M64 系统报警AL.32", "应放行"),
        ("海德汉TNC155 如何设置参数", "应放行"),
        ("怎么车一个轴", "应放行"),
        ("液压油多久更换一次", "应放行"),
        ("FANUC系统如何进行螺距误差补偿", "应放行"),
        ("华中数控系统报警", "应拒答"),
        ("SIEMENS 810M参数如何设置", "应放行"),
    ]

    for question, expected in test_cases:
        result = check_vendor_support(question)
        status = "❌ 拒答" if result else "✅ 通过"
        print("   [{}] {} | 预期: {}".format(status, question[:30], expected))
        if result:
            print("          原因: {}".format(result["reason"]))

    print("\n=== 自检完成 ===")


if __name__ == "__main__":
    selftest()
