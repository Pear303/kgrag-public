# -*- coding: utf-8 -*-
"""问答服务层：图谱检索 + 社区召回 + 答案生成（生产链路，剥离评测依赖）。

设计：
- 数据源：评测隔离库 7690（v5 数据 + M64/DGN700 补抽，最新最全）；
  可用环境变量 SERVING_NEO4J_URI 覆盖。
- 检索链路与评测完全一致（_collect_graph_contexts 实体二次打分 +
  OOC 拒答门 + _truncate_contexts），确保"评测什么表现，服务就什么表现"。
- 答案生成：与评测同款 CoT prompt，但**只走本机模型**（config.get_llm()，`LLM_PROVIDER=local`）。
  这是作者定死的约束（2026-10-09）：真实生产环境里给用户写答案不许依赖外部大模型；
  远程强模型只允许用在建库抽取、出题、答案质量评审。配置不满足时 `_build_engines` 直接抛错、
  服务不起（require_local_answer_llm）。
- 提供三种查询：
    query()        微观问答（图谱路径 + 答案）
    query_macro()  宏观总结（社区召回 + 原始 Chunk + LLM 总结）
    query_compare() KG vs Vector 对比

部署要点（2026-09-09 修复 Windows 下 uvicorn 进程卡死）：
- 服务实例全局单例：KGRAGService 构建含 embedding 模型加载 + 手册解析 +
  Vector 索引构建（约数十秒），此前 /health 与业务请求各自重新实例化，
  重活同步阻塞事件循环 → 整个服务"监听中但无响应"。改为懒加载单例。
- 重活下沉线程池：业务查询经 run_in_executor 执行，事件循环永远不被
  阻塞，/health、/test 在模型加载/推理期间依然秒回。
- 请求超时保护：单请求超时 QUERY_TIMEOUT_S（默认 300s）返回 504，
  防止异常请求长时间占用查询线程。
- Windows 事件循环：入口处切换 SelectorEventLoopPolicy，规避 proactor
  已知兼容性问题。

用法：
    from serve import KGRAGService
    svc = KGRAGService()
    svc.query("三菱M64系统中，若使用变频器作为主轴驱动系统，参数P145应设定为什么值？")

    # 生产启动：python serve.py 或 uvicorn serve:app --host 0.0.0.0 --port 8000
"""
import os
import sys
import re
import copy
import asyncio
import unicodedata

# Windows 事件循环策略必须在任何事件循环创建之前设置（模块导入期）。
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

import json
import hashlib
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

logger = logging.getLogger("kgrag.serve")

# 允许独立进程直接运行本模块（python serve.py 自检）
if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))

# 数据源库：默认评测库（最新数据），可用 SERVING_NEO4J_URI 覆盖。
# 注意：必须强制覆盖（.env 已把 NEO4J_URI 加载为 7688 主库；GraphStore
# 读 os.environ，setdefault 不会覆盖已存在的值 → 会连错库）。
_serving_uri = os.getenv("SERVING_NEO4J_URI") or os.getenv("NEO4J_EVAL_URI") or "bolt://localhost:7690"
# 建图库与查询库分离：先把 .env 的 NEO4J_URI（7688）另存为 NEO4J_BUILD_URI，
# 再覆盖 NEO4J_URI 供查询侧使用。否则 api_server 先 import serve 会让
# config.NEO4J_URI 冻成 7690，上传手册的建图/删除会直接写冻结评测库。
os.environ.setdefault("NEO4J_BUILD_URI", os.environ.get("NEO4J_URI", ""))
os.environ["NEO4J_URI"] = _serving_uri

# 评测对齐预设（2026-10-16 丙）：**默认关闭 ⇒ 线上行为一字节不变**。
# 打开方式：SERVING_EVAL_PARITY=1 且 SERVING_PRESET_META=<某轮 meta.json 路径>。
# 必须在这里（import config 之前）注入：分块世代 VECTOR_WINDOW_SIZE / VECTOR_OVERLAP_SIZE
# 在 config 的 import 期就被冻成常量，晚一步注了就无效——评测侧 R1/R2 那两轮的静默退化
# 正是"声明了开关却没真正生效"，此处的硬断言与 evaluator 的 EvalIntegrityError 同型。
_SERVING_EVAL_PARITY = (os.getenv("SERVING_EVAL_PARITY") or "").strip() == "1"
if _SERVING_EVAL_PARITY:
    _preset_meta = (os.getenv("SERVING_PRESET_META") or "").strip()
    if not _preset_meta or not os.path.exists(_preset_meta):
        raise RuntimeError(
            "SERVING_EVAL_PARITY=1 但 SERVING_PRESET_META 缺失或不存在：%r" % (_preset_meta,))
    with open(_preset_meta, "r", encoding="utf-8") as _f:
        _preset_ec = (json.load(_f) or {}).get("effective_config") or {}
    if not _preset_ec:
        raise RuntimeError("预设 meta 里没有 effective_config：%s" % _preset_meta)
    # 窄覆盖（2026-10-17 丙，作者选的乙案）：SERVING_PRESET_OVERRIDE 是逗号分隔的键名清单，
    # 只有列在里面的键，若调用方（启动环境或 .env）已经显式给了值，就不被预设盖掉。
    # 为什么不做成"全局调用方优先"：预设一注入就写 45 个键，全局优先会让 .env 里任何
    # 过期残留静默压住在册预设——那正是本块要防的"声明了开关却没真正生效"的反面。
    # 清单里出现预设没有的键 ⇒ 硬停：拼错的覆盖键静默无效，和没生效是同一种事故。
    _override_keys = {k.strip() for k in
                      (os.getenv("SERVING_PRESET_OVERRIDE") or "").split(",") if k.strip()}
    _unknown = sorted(_override_keys - set(_preset_ec.keys()))
    if _unknown:
        raise RuntimeError(
            "SERVING_PRESET_OVERRIDE 列了预设里没有的键：%s（预设键来自 %s）"
            % (_unknown, os.path.basename(_preset_meta)))
    _kept = {}
    for _pk, _pv in _preset_ec.items():
        if _pk in _override_keys and (os.environ.get(str(_pk)) or "").strip():
            _kept[str(_pk)] = (os.environ[str(_pk)], str(_pv))
            continue
        os.environ[str(_pk)] = str(_pv)
    os.environ["NEO4J_URI"] = _serving_uri      # 图库归属不许被预设改走
    logger.info("评测对齐预设已注入：%d 键（来自 %s）；KG_MAX_CTX=%s window=%s overlap=%s",
                len(_preset_ec), os.path.basename(_preset_meta),
                os.getenv("KG_MAX_CTX"), os.getenv("VECTOR_WINDOW_SIZE"),
                os.getenv("VECTOR_OVERLAP_SIZE"))
    for _k, (_v, _pv) in sorted(_kept.items()):
        logger.warning("预设覆盖生效：%s 保留调用方值 %s（预设本会给 %s）", _k, _v, _pv)

from config import (get_embed_model, get_llm, PROJECT_ROOT,
                   LLM_PROVIDER, LOCAL_LLM_BASE_URL, LOCAL_LLM_MODEL,
                   LOCAL_EMBEDDING_MODEL)
from backend.graph_store import GraphStore
from backend.engines import ManualGraphQueryEngine, CommunityQueryEngine
from backend.evaluator import (
    RAGASEvaluator, _truncate_contexts, _rerank_contexts, _split_cot_output,
    _collect_graph_contexts, _context_relevance_score, _ooc_object_mismatch,
    _ooc_gate_relevance, _OOC_RELEVANCE_THRESHOLD, sent_cov_gate_hook,
    relsem_release_hook,
)
from backend.parser import parse_manual_chunks, parse_manual_chunks_sliding_window, parse_manual_structured
from backend.vector_rag import VectorRAGQueryEngine
from llama_index.core import Document
from llama_index.core.llms import ChatMessage

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="KGRAG API", version="1.0.0")

STATIC_DIR = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- 服务单例与查询线程池 ----------
_START_TIME = time.time()
_service = None
_service_lock = threading.Lock()
_service_error = None  # 预热/构建失败原因（供 /health 暴露）

# 业务查询线程池：重活（图谱检索 + LLM 生成）全部在此执行，
# 事件循环只做 await，保证控制面端点（/health、/test）始终响应。
_query_pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix="kgrag-query")

# 单请求超时（秒）：覆盖 embedding/LLM 正常延迟即可，可用环境变量调整
QUERY_TIMEOUT_S = float(os.getenv("QUERY_TIMEOUT_S", "300"))


def require_local_answer_llm(llm=None, expected_model: str = None):
    """生产环境的答案生成必须走本地模型（作者定死的约束，2026-10-09 落定）。

    只查两件事：配置的 provider 是 local、且它指向的主机是本机。任一条不满足直接抛错，
    绝不静默退回远程——`.env` 写着 local 却实际调远程，是这条约束被打破的方式。
    expected_model 用于运行时切换的本地模型；不传时仍以 .env 的 LOCAL_LLM_MODEL 为准。
    """
    if LLM_PROVIDER != "local":
        raise RuntimeError(
            u"答案生成被配成远程模型（LLM_PROVIDER=%s）：生产环境必须走本地模型" % LLM_PROVIDER)
    host = str(LOCAL_LLM_BASE_URL).split(u"://", 1)[-1].split(u":", 1)[0].split(u"/", 1)[0]
    if host not in (u"localhost", u"127.0.0.1"):
        raise RuntimeError(u"本地通道指向 %s（不是本机地址）：拒绝起服务" % LOCAL_LLM_BASE_URL)
    expected = expected_model or LOCAL_LLM_MODEL
    if llm is not None and getattr(llm, "model", None) != expected:
        raise RuntimeError(u"答案生成用的模型 %r 与登记的本机模型 %r 不一致"
                             % (getattr(llm, "model", None), expected))
    return True


# ---------- 例号直取＋消歧通道（2026-10-27）----------
# 背景：shukongdashi_500cases.md 是多本书拼接语料，例号（例16/例18 等 12 个）
# 与节号在书里有多个定义位点 ⇒ 按例号检索天然歧义；且 24 道"上下文零覆盖"题里
# 13 道题面带例号、金子就躺在所引案例的正文里，dense 召回却拿不到。
# 本通道：题面含 例N 时直接截取该例定义的正文进上下文；同一编号多定义时按
# 题面实词（剔编号本身）打分挑定义。
# OPT_DIRECT_CASE 默认 0 ⇒ 完全不参与装配，行为逐位不变。
_DIRECT_CASE_RE = re.compile(r"例\s*(\d{1,4}[A-Za-z]?)")
_DIRECT_CASE_SEP = re.compile(r"\s{2,}|[.．、:：]\s")
_DIRECT_CASE_END_PUNCT = "。！？；"


def _norm_text(t):
    return re.sub(r"\s+", "", unicodedata.normalize("NFKC", t or "")).lower()


def _case_content_tokens(question, drop):
    """题面实词：拉丁/数字串（≥2 字，剔编号）＋中文 2 字组，去重。"""
    t = _norm_text(question)
    toks = set()
    for x in re.findall(r"[a-z0-9][a-z0-9\-]*", t):
        if len(x) >= 2 and x != drop:
            toks.add(x)
    for run in re.findall(r"[\u4e00-\u9fff]{2,}", t):
        if len(run) <= 4:
            toks.add(run)
        toks.update(run[i:i + 2] for i in range(len(run) - 1))
    return toks


def _is_case_site(rest):
    """定义位点判据：编号后有标题式分隔（≥2 空白或 .：、+空白），
    标题体 ≥3 字且 40 字内无句末标点。行文交叉引用（"见例19，电池…"）不满足。"""
    m = _DIRECT_CASE_SEP.match(rest)
    if not m:
        return False
    body = rest[m.end():].strip()
    if len(body) < 3:
        return False
    return not any(p in body[:40] for p in _DIRECT_CASE_END_PUNCT)


def _build_case_site_map(manual_text):
    """手册原文 → ({例N: [该编号各定义位点的行起始 offset]},
    全位点有序表, {例N: 首次出现 offset}（含非定义样式，作 fallback））。"""
    sites = {}
    first_any = {}
    all_offsets = []   # (offset, token)，全局有序，供跨度端点
    off = 0
    for line in manual_text.split("\n"):
        m = _DIRECT_CASE_RE.search(line)
        if m:
            first_any.setdefault(m.group(1), off)
            if _is_case_site(line[m.end():]):
                sites.setdefault(m.group(1), []).append(off)
                all_offsets.append((off, m.group(1)))
        off += len(line) + 1
    all_offsets.sort()
    return sites, all_offsets, first_any


def _direct_case_contexts(question, manual_text, site_map, all_offsets,
                          first_any=None, max_chars=3000):
    """题面含 例N ⇒ 返回 [所选定义正文块]；开关未开/无例号/无位点 ⇒ []。
    多定义：题面实词打分（可挑 ⇔ 最高−次高 ≥2 且最高 ≥2），平票回落首现位点。"""
    if os.getenv("OPT_DIRECT_CASE", "0") != "1":
        return []
    tokens = _DIRECT_CASE_RE.findall(question)
    if not tokens or not manual_text or site_map is None:
        return []
    blocks, seen = [], set()
    for tk in tokens:
        site_list = site_map.get(tk) or []
        how = "unique"
        if not site_list:
            # 无定义位点（标题格式特殊）⇒ 回落该编号首次出现处（原型实测
            # g1_new_114/例190A 属此类，回落可拿全金子）。
            fb = first_any.get(tk) if first_any else None
            if fb is None:
                continue
            chosen, how = fb, "fallback"
        elif len(site_list) == 1:
            chosen = site_list[0]
        else:
            toks = _case_content_tokens(question, _norm_text(tk))
            scored = []
            for off in site_list:
                nxt = None
                for o2, _t2 in all_offsets:
                    if o2 > off:
                        nxt = o2
                        break
                seg_end = nxt if nxt is not None else len(manual_text)
                span_txt = _norm_text(
                    manual_text[off:min(off + max_chars, seg_end)])
                scored.append((sum(1 for w in toks if w in span_txt), off))
            scored.sort(key=lambda x: (-x[0], x[1]))
            s1 = scored[0][0]
            s2 = scored[1][0] if len(scored) > 1 else 0
            if s1 - s2 >= 2 and s1 >= 2:
                chosen, how = scored[0][1], "disambiguated"
            else:
                chosen, how = site_list[0], "tie"
        nxt = None
        for o2, _t2 in all_offsets:
            if o2 > chosen:
                nxt = o2
                break
        seg_end = nxt if nxt is not None else len(manual_text)
        block = manual_text[chosen:min(chosen + max_chars, seg_end)].strip()
        if block and block not in seen:
            seen.add(block)
            blocks.append(block)
            logger.info("例号直取: 题=%r 例%s → offset=%d(%s)", question[:40], tk,
                        chosen, how)
    return blocks


# ---------- 报警号病例锚定 ----------
# 触发实题：2011 换刀题的实际上下文里没有 2011 病例原文，却混进 SV003 的
# A 轴电缆例；路径后的 [原文依据] 也挂到别的病例。本通道只在题面含报警号时：
# ①把手册里该报警号所在病例段前置；②剔除引用了别的报警号的上下文；
# ③把路径后的 [原文依据] 改为从本病例现场截取，挂不上就明写"无原文依据"。
# OPT_ALARM_CASE 默认 0 ⇒ 不进入任何新分支，装配逐字节与改动前相同。
_ALARM_GAP = r"[^\n。！？；,，\d]{0,6}?"
_ALARM_RE_AFTER = re.compile(
    r"(?<![A-Za-z0-9])"
    r"([A-Za-z]{1,8}[\s\-－_–—]*\d{1,10}|\d{1,10})"
    r"(?![A-Za-z0-9])" + _ALARM_GAP + r"(?:号)?\s*(?:报警|ALARM)", re.I)
_ALARM_RE_LETTER = re.compile(
    r"(?<![A-Za-z0-9])([A-Za-z]{2,8})(?=\s*(?:号)?\s*(?:报警|ALARM))", re.I)
_ALARM_RE_BEFORE = re.compile(
    r"(?:报警号?|ALARM)\s*(?:为|是|：|:)?\s*"
    r"([A-Za-z]{1,8}[\s\-－_–—]*\d{1,10}|\d{1,10})(?![A-Za-z0-9])", re.I)
_ALARM_STOP_LETTER = {
    "SERVO", "ALARM", "NC", "CNC", "PLC", "CRT", "MDI", "PWR", "CPU",
    "ROM", "RAM", "USB", "LED", "LCD", "AC", "DC", "FANUC", "SIEMENS",
    "OK", "ERROR", "EXCESS", "AXIS", "SPEED", "DRIVE", "DRIVER", "SYSTEM",
    "READY", "STOP", "START", "ON", "OFF", "THE", "AND", "FOR", "WITH",
}
# 厂商/系统名不是报警号：SIEMENS 611、FANUC 0 这类前缀一律丢掉
_ALARM_VENDOR = {
    "FANUC", "SIEMENS", "MITSUBISHI", "HEIDENHAIN", "OKUMA", "MAZAK",
    "GSK", "HNC", "KND", "NUM", "BOSCH", "DELTA", "YASKAWA", "PANASONIC",
    "OMRON", "SCHNEIDER", "ABB", "KUKA", "SANYO", "TOSHIBA", "HITACHI",
    "FUJI", "RENISHAW", "FAGOR", "SODICK",
}
_ALARM_SECTION_RE = re.compile(
    r"^\s*(?:第[一二三四五六七八九十百]+章|\d+(?:\.\d+)+\s"
    r"|\d+\.\s*[\u4e00-\u9fff]{2,}"
    r"|\d+\.\s+[A-Za-z][^\n]{0,40}(?:故障维修|维修))")
# 行首就是病例标题（用于"下一例"截断）
_ALARM_CASE_HDR_LINE_RE = re.compile(
    r"^\s*(?:#{1,6}\s*)?例\s*[^\n]{0,80}?(?:维修|故障现象)")
# 段内任意位置出现病例标题（用于把标题从页面噪声里切出来）
_ALARM_CASE_HDR_ANY_RE = re.compile(
    r"例\s*[^\n]{0,60}?(?:故障维修|故障的维修|故障的故障维修|报警的维修|"
    r"故障现象)")
_ALARM_EVIDENCE_MARK = "\n[原文依据]\n"
_ALARM_GENERIC = {
    "报警", "故障", "系统", "出现", "原因", "什么", "如何", "处理", "维修",
    "可能", "最终", "确认", "元件", "含义", "现象", "加工", "中心", "数控",
    "机床", "驱动", "问题", "哪些", "几种", "通过", "操作", "进一步", "判定",
    "应该", "怎么", "是否", "以及", "并且", "正常", "工作", "时间", "显示",
    "进行", "检查", "分析", "排除", "更换", "方法", "步骤", "位置", "参数",
    "信号", "动作", "发生", "情况", "时候", "说明", "表示", "一个", "这种",
    "然后", "同时", "由于", "因此", "所以", "但是", "如果", "这个", "表现为",
    "可能原因", "解决方法", "推理链", "起点", "包含", "配套使用",
    "the", "and", "for", "with", "what", "how", "why", "which", "system",
    "alarm", "machine", "error",
}


def _alarm_norm_text(text):
    """报警号通道归一化：NFKC＋去空白＋去各种横线＋小写。"""
    return re.sub(r"[\-－_–—]", "", _norm_text(text))


def _alarm_norm_code(code):
    return _alarm_norm_text(code).upper()


def _alarm_is_vendor_code(tok):
    m = re.match(r"^([A-Za-z]+)[\s\-－_–—]*\d", tok or "")
    return bool(m and m.group(1).upper() in _ALARM_VENDOR)


def _alarm_is_short_bare(tok):
    """裸 1~2 位数字多为列表序号（"2）1160报警"），不当报警号。"""
    return bool(re.fullmatch(r"\d{1,2}", (tok or "").strip()))


def _alarm_tokens(text):
    """题面里的报警号（按出现顺序去重）。只认与"报警/ALARM"同句的编号。"""
    if not text:
        return []
    out = []
    for m in _ALARM_RE_AFTER.finditer(text):
        tok = m.group(1)
        gap = m.group(0)[m.end(1) - m.start(0):]
        if any(ch in gap for ch in ("年", "月", "出版")):
            continue
        if re.search(r"\d", gap) or _alarm_is_short_bare(tok):
            continue
        if _alarm_is_vendor_code(tok):
            continue
        out.append(tok)
    for m in _ALARM_RE_BEFORE.finditer(text):
        tok = m.group(1)
        if _alarm_is_short_bare(tok) or _alarm_is_vendor_code(tok):
            continue
        out.append(tok)
    for m in _ALARM_RE_LETTER.finditer(text):
        t = m.group(1).upper()
        if t not in _ALARM_STOP_LETTER and t not in _ALARM_VENDOR:
            out.append(m.group(1))
    seen, ans = set(), []
    for tok in out:
        k = _alarm_norm_code(tok)
        if k and k not in seen:
            seen.add(k)
            ans.append(tok.strip())
    return ans


def _alarm_code_regex(token):
    """把规范化编号还原成能容忍空格/连字符/全角的检索正则。"""
    t = unicodedata.normalize("NFKC", token).upper()
    m = re.match(r"^([A-Z]{1,8})[\s\-－_–—]*(\d{1,10})$", t)
    if m:
        p, d = m.groups()
        return re.compile(
            r"(?<![A-Za-z0-9])" + p + r"[\s\-－_–—]*" + d +
            r"(?![A-Za-z0-9])", re.I)
    if re.match(r"^\d{1,10}$", t):
        return re.compile(r"(?<!\d)" + t + r"(?!\d)")
    return re.compile(
        r"(?<![A-Za-z0-9])" + re.escape(t) + r"(?![A-Za-z0-9])", re.I)


def _alarm_paragraphs(manual_text):
    paras, off = [], 0
    for line in (manual_text or "").split("\n"):
        if line.strip():
            paras.append({"start": off, "end": off + len(line), "text": line})
        off += len(line) + 1
    return paras


def _alarm_occurrences(manual, paras, token, limit=80):
    rx = _alarm_code_regex(token)
    hits = []
    for m in rx.finditer(manual):
        o = m.start()
        win = manual[max(0, o - 160): o + 160]
        if not re.search(r"报警|ALARM", win, re.I):
            continue
        pi = None
        for i, p in enumerate(paras):
            if p["start"] <= o <= p["end"]:
                pi = i
                break
        if pi is not None:
            hits.append((o, pi, m.group(0)))
        if len(hits) >= limit:
            break
    return hits


def _alarm_has_case_body(t):
    return ("故障现象" in t or "分析及处理过程" in t
            or "分析与处理过程" in t)


def _alarm_block_for_para(paras, pi, token, max_chars=2400):
    start = pi
    cur = paras[pi]["text"]
    # 标题在上一段：当前段以"故障现象"开头，上一段是病例标题
    if cur.lstrip().startswith("故障现象") and pi > 0:
        prev = paras[pi - 1]["text"]
        if _ALARM_CASE_HDR_ANY_RE.search(prev) and not (
                "分析及处理过程" in prev or "分析与处理过程" in prev):
            start = pi - 1
    # 当前段是"分析及处理过程"，标题在更上面：最多回退两段
    elif ("分析及处理过程" in cur or "分析与处理过程" in cur) \
            and not _ALARM_CASE_HDR_ANY_RE.search(cur):
        for back in (1, 2):
            k = pi - back
            if k < 0:
                break
            prev = paras[k]["text"]
            if _ALARM_CASE_HDR_LINE_RE.search(prev) or "故障现象" in prev:
                start = k
                break
            if _ALARM_SECTION_RE.match(prev):
                break
    end = pi
    total = len(paras[pi]["text"])
    while end + 1 < len(paras):
        nxt = paras[end + 1]["text"]
        if _ALARM_SECTION_RE.match(nxt):
            break
        if _ALARM_CASE_HDR_LINE_RE.search(nxt):
            break
        if total + len(nxt) > max_chars:
            break
        end += 1
        total += len(nxt)
    parts = [p["text"] for p in paras[start:end + 1]]
    # 首段若混着上一例尾巴，切到段内病例标题处（只切前 300 字内，防误切正文）
    if parts:
        last = None
        for m in _ALARM_CASE_HDR_ANY_RE.finditer(parts[0]):
            last = m
        if last and 0 < last.start() <= 300:
            parts[0] = parts[0][last.start():]
    text = re.sub(r"^例{2,}", "例", "\n".join(parts).strip())
    return text, start, end


def _alarm_content_tokens(text, drop):
    t = _alarm_norm_text(text)
    toks = set()
    for x in re.findall(r"[a-z0-9][a-z0-9\-]*", t):
        x = x.replace("-", "")
        if len(x) >= 2 and x != _alarm_norm_text(drop) and x not in _ALARM_GENERIC:
            toks.add(x)
    for run in re.findall(r"[\u4e00-\u9fff]{2,}", t):
        if len(run) <= 4 and run not in _ALARM_GENERIC:
            toks.add(run)
        for i in range(len(run) - 1):
            bg = run[i:i + 2]
            if bg not in _ALARM_GENERIC:
                toks.add(bg)
    return toks


def _alarm_score_block(text, question, token):
    nt = _alarm_norm_text(text)
    score = 0
    if _ALARM_CASE_HDR_ANY_RE.search(text):
        score += 100
    if _ALARM_CASE_HDR_LINE_RE.search(text.lstrip()):
        score += 50
    if "故障现象" in text:
        score += 30
    if "分析及处理过程" in text or "分析与处理过程" in text:
        score += 30
    if _alarm_norm_code(token) in _alarm_norm_code(text):
        score += 20
    if _alarm_norm_code(token) in _alarm_norm_code(text[:160]):
        score += 20
    qts = _alarm_content_tokens(question, token)
    score += min(40, 4 * sum(1 for w in qts if w in nt))
    if len(text) > 1200:
        score -= (len(text) - 1200) / 1000.0
    return score


def _alarm_case_blocks(manual, paras, token, question=""):
    out = []
    for o, pi, _raw in _alarm_occurrences(manual, paras, token):
        text, s, e = _alarm_block_for_para(paras, pi, token)
        if not text or text in [x["text"] for x in out]:
            continue
        out.append({"offset": o, "para": pi, "span": [s, e], "text": text,
                    "score": _alarm_score_block(text, question, token)})
    return out


def _alarm_best_case_block(manual, paras, token, question=""):
    blocks = _alarm_case_blocks(manual, paras, token, question)
    if not blocks:
        return None
    blocks.sort(key=lambda x: (-x["score"], x["offset"]))
    return blocks[0]


def _alarm_case_contexts(question, manual_text, max_blocks=2):
    """题面含报警号 ⇒ ([带标签病例块], meta)；开关未开/无编号/无病例 ⇒ ([], {})。"""
    if os.getenv("OPT_ALARM_CASE", "0") != "1":
        return [], {}
    if not question or not manual_text:
        return [], {}
    tokens = _alarm_tokens(question)
    if not tokens:
        return [], {}
    paras = _alarm_paragraphs(manual_text)
    blocks, bodies, raws, norms = [], [], [], []
    for raw in tokens:
        key = _alarm_norm_code(raw)
        if not key or key in norms:
            continue
        best = _alarm_best_case_block(manual_text, paras, raw, question)
        if best is None:
            continue
        body = best["text"]
        if key not in _alarm_norm_code(body):
            continue
        qts = _alarm_content_tokens(question, raw)
        if qts:
            nt = _alarm_norm_text(body)
            if not any(w in nt for w in qts):
                logger.info("报警号病例锚定: 跳过 %s（病例段与题面实词零重叠）", raw)
                continue
        if body in bodies:
            continue
        label = "【手册原文·%s 报警所在%s】\n" % (
            raw.strip(), "病例" if _alarm_has_case_body(body) else "条目")
        blocks.append(label + body)
        bodies.append(body)
        raws.append(raw.strip())
        norms.append(key)
        logger.info("报警号病例锚定: 题=%r 编号=%s → offset=%d span=%r 段长=%d",
                    question[:40], raw, best["offset"], best["span"], len(body))
        if len(blocks) >= max_blocks:
            break
    return blocks, {"tokens": raws, "norms": norms,
                    "bodies": bodies, "blocks": blocks}


def _alarm_case_evidence(head, body, token, max_chars=600, max_sent=3):
    """从病例段里挑与路径节点最相关的句子；挂不上 ⇒ None（不许挂错例）。"""
    if not body:
        return None
    toks = _alarm_content_tokens(head, token)
    if not toks:
        return None
    parts = [p.strip() for p in re.split(r"(?<=[。！？；\n])", body) if p.strip()]
    scored = []
    for p in parts:
        np = _alarm_norm_text(p)
        sc = sum(1 for w in toks if w in np)
        if sc:
            scored.append((sc, len(p), p))
    if not scored:
        return None
    scored.sort(key=lambda x: (-x[0], x[1]))
    picked, total = [], 0
    for _sc, ln, p in scored:
        if picked and total + ln > max_chars:
            continue
        picked.append(p)
        total += ln
        if len(picked) >= max_sent:
            break
    if not picked:
        return None
    return "\n".join(p.strip() for p in picked if p.strip())


def _alarm_replace_evidence(ctx, body, token):
    if not body or _ALARM_EVIDENCE_MARK not in ctx:
        return ctx
    head = ctx.split(_ALARM_EVIDENCE_MARK, 1)[0]
    ev = _alarm_case_evidence(head, body, token)
    if not ev:
        ev = "（无原文依据：该路径在本报警病例原文中找不到对应句）"
    return head + _ALARM_EVIDENCE_MARK + ev


def _alarm_is_path_ctx(ctx):
    return ("--[" in ctx or "<--[" in ctx
            or str(ctx).lstrip().startswith("[推理链]"))


_ALARM_PATH_ARROW_RE = re.compile(r"(--\[[^\]]+\]-->|<--\[[^\]]+\]--)")
_ALARM_NODE_SUFFIX = ("故障", "报警", "异常", "问题", "现象", "原因",
                      "解决方法", "方法", "处理", "维修")


def _alarm_node_supported(node, body):
    """路径节点是否由本病例正文支持：原名或去掉通用后缀后的核心名出现在正文里。
    （"伺服电动机故障"由"伺服电动机"支持；"…内部切削液导致线圈短路"不支持。）"""
    n = _alarm_norm_text(node)
    b = _alarm_norm_text(body)
    if not n or len(n) <= 2:
        return True
    if n in b:
        return True
    core = n
    for _ in range(2):
        for suf in _ALARM_NODE_SUFFIX:
            if core.endswith(suf) and len(core) > len(suf):
                core = core[:-len(suf)]
                break
    if core and core in b:
        return True
    # 非连续写法（"刀塔"+"转动不到位"）按实词覆盖率判；混入别例的尾巴覆盖率会掉
    toks = _alarm_content_tokens(node, "")
    if not toks:
        return True
    hit = sum(1 for w in toks if w in b)
    return (hit / len(toks)) >= 0.6


def _alarm_trim_path(ctx, body, token):
    """把路径裁到"每个节点都由本病例正文支持"的最长前缀；裁不下两条节点则弃。

    只对病例正文块生效（定义/表格类锚点不裁，避免把路径删光）。
    """
    if not body or not _alarm_has_case_body(body):
        return ctx
    if _ALARM_EVIDENCE_MARK in ctx:
        head, ev = ctx.split(_ALARM_EVIDENCE_MARK, 1)
    else:
        head, ev = ctx, None
    if not head or head.lstrip().startswith("[推理链]"):
        return ctx
    parts = _ALARM_PATH_ARROW_RE.split(head)
    if len(parts) < 3:
        return ctx
    keep = -1
    for i in range(0, len(parts), 2):
        if _alarm_node_supported(parts[i], body):
            keep = i
        else:
            break
    if keep < 2:
        return None
    trimmed = "".join(parts[:keep + 1]).strip()
    if ev is None:
        return trimmed
    return trimmed + _ALARM_EVIDENCE_MARK + ev


def _alarm_overlap_count(text, body, token):
    toks = _alarm_content_tokens(text, token)
    if not toks:
        return 1
    nt = _alarm_norm_text(body)
    return sum(1 for w in toks if w in nt)


def _alarm_apply_anchor(ctxs, blocks, meta, question):
    """异例上下文剔除 + 路径依据改为本病例现场截取。病例块自身永不删。"""
    asked = set(meta.get("norms") or [])
    raws = meta.get("tokens") or []
    body = (meta.get("bodies") or [""])[0]
    token = raws[0] if raws else ""
    out, seen = [], set()
    for c in ctxs or []:
        if c in blocks:
            if c not in seen:
                seen.add(c)
                out.append(c)
            continue
        ct = {_alarm_norm_code(t) for t in _alarm_tokens(c)}
        has_asked = bool(ct & asked) or any(
            t and _alarm_norm_text(t) in _alarm_norm_text(c) for t in raws)
        if ct and not has_asked:
            continue
        # 例号直取块与本病例段是同一段时只留带标签的那一份，避免重复占槽
        if body and not _alarm_is_path_ctx(c) \
                and _alarm_norm_text(c) in _alarm_norm_text(body):
            continue
        # 非路径原文/社区块：不含所问报警号就一律隔离（2011_2 的 A 轴电缆块
        # 正文里有 2340 却没写 2011，旧判据漏掉；预注册 R0 要求零跨例）。
        if not _alarm_is_path_ctx(c) and not has_asked:
            continue
        if _alarm_is_path_ctx(c):
            if not has_asked and body \
                    and _alarm_overlap_count(c, body, token) == 0:
                continue
            c = _alarm_trim_path(c, body, token)
            if c is None:
                continue
            c = _alarm_replace_evidence(c, body, token)
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out


# ---------- 表格抽行通道（2026-10-27，实验 tablefix_exp4：36 题配对救回 7 带坏 0）----------
# 病灶：4B 模型细粒度查表时会把 A 行编号与 B 行内容缝成一行还标"原文"（提示词、
# 删干扰表、还原渲染三疗法均无效）。治法＝程序定位题面所引 表X-Y 并抽行；
# 且须把底座里整块携带该表体的检索块剔掉——块与底座并存时模型会无视块仍抄底座
# （new_040 实测）。开关 OPT_TABLE_ROW 默认 0 ⇒ 完全不参与装配。
_TBL_REF_RE = re.compile(r"表\s*(\d{1,2})\s*[-–—]\s*(\d{1,3})")
_TBL_CHAPTER_RE = re.compile(r"^\s*第\s*[0-9一二三四五六七八九十]{1,3}\s*章")


def _is_table_head_line(line):
    """表头行判据：行内出现 表X-Y 且其后是 ≥3 字标题体；
    句读开头（"…见表 2-4。"）＝行文交叉引用，不算表头。"""
    m = _TBL_REF_RE.search(line)
    if not m:
        return None
    rest = line[m.end():].strip()
    if len(rest) < 3 or rest[0] in "。！？；，、）":
        return None
    return (m.group(1) + "-" + m.group(2), m.start())


def _build_table_index(manual_lines):
    idx = {}
    for i, l in enumerate(manual_lines):
        r = _is_table_head_line(l)
        if r:
            idx.setdefault(r[0], []).append(i)
    return idx


def _table_q_tokens(question):
    """题面查询值：剔表号后取 单字数字/字母（前邻字母的并入型号串不算，
    _norm_text 去空白保证 'FANUC 0'→'fanuc0'）＋拉丁串（≥2）＋中文 2 字组。"""
    q = _TBL_REF_RE.sub(" ", question)
    t = _norm_text(q)
    singles = set(re.findall(r"(?<![a-z0-9])([a-z0-9])(?![a-z0-9])", t))
    multis = set(w for w in re.findall(r"[a-z0-9][a-z0-9.\-]*", t) if len(w) >= 2)
    grams = set()
    for run in re.findall(r"[\u4e00-\u9fff]{2,}", t):
        grams.update(run[i:i + 2] for i in range(len(run) - 1))
    return singles, multis, grams


def _table_row_score(line, singles, multis, grams):
    t = _norm_text(line)
    rs = set(re.findall(r"(?<![a-z0-9])([a-z0-9])(?![a-z0-9])", t))
    s = 3 * len(singles & rs)
    s += 2 * sum(1 for w in multis if w in t)
    s += sum(1 for g in grams if g in t)
    return s


def _table_span_end(manual_lines, h):
    """表体终点：下一表头行 / 章节行 / 见表行 / >70 字行文行，或 60 个非空行。"""
    cnt, j = 0, h + 1
    while j < len(manual_lines) and cnt < 60:
        st = manual_lines[j].strip()
        if st:
            if _is_table_head_line(manual_lines[j]) or _TBL_CHAPTER_RE.match(manual_lines[j]) \
                    or "见表" in st or len(st) > 70:
                break
            cnt += 1
        j += 1
    return j


def _table_row_contexts(question, manual_text, tbl_index, manual_lines,
                        max_chars=1600):
    """题面含 表X-Y ⇒ ([抽行块], [该表体行])；开关未开/无引用/选不出行 ⇒ ([], [])。
    选行打分：单字查询值 3 分、拉丁串 2 分、中文 2 字组 1 分；
    决定性（最高≥3 且比次高≥2）通常只喂 1 行，但最高行未直接命中
    查询值（纯词组堆分）时喂前 3 行；非决定性喂前 2 行；全 0 分不出块。"""
    if os.getenv("OPT_TABLE_ROW", "0") != "1":
        return [], []
    refs = _TBL_REF_RE.findall(question)
    if not refs or not manual_text or tbl_index is None or manual_lines is None:
        return [], []
    refs = list(dict.fromkeys(n + "-" + s for n, s in refs))
    singles, multis, grams = _table_q_tokens(question)
    blocks, markers = [], []
    for key in refs:
        sites = tbl_index.get(key) or []
        if not sites:
            continue
        if len(sites) == 1:
            h = sites[0]
        else:
            toks = _case_content_tokens(_TBL_REF_RE.sub(" ", question), "")
            scored = []
            for h2 in sites:
                seg = _norm_text(
                    "\n".join(manual_lines[h2:_table_span_end(manual_lines, h2)])[:3000])
                scored.append((sum(1 for w in toks if w in seg), h2))
            scored.sort(key=lambda x: (-x[0], x[1]))
            s1 = scored[0][0]
            s2 = scored[1][0] if len(scored) > 1 else 0
            h = scored[0][1] if (s1 - s2 >= 2 and s1 >= 2) else sites[0]
        e = _table_span_end(manual_lines, h)
        body = [l for l in manual_lines[h + 1:e] if l.strip()]
        if not body:
            continue
        scored = []
        for idx, l in enumerate(body[1:]):
            if len(l.strip()) <= 60:
                scored.append((_table_row_score(l, singles, multis, grams), idx, l))
        scored.sort(key=lambda x: (-x[0], x[1]))
        if not scored or scored[0][0] <= 0:
            continue
        s2v = scored[1][0] if len(scored) > 1 else 0
        # 决定性必须建立在"最高行直接命中题面查询值（单字/拉丁串）"上：
        # 纯中文词组堆分的最高行不可信（线上实测 097/098：OV/OL 题被
        # 词组分送错行，而正确行随底座剔除 ⇒ 答反）。其余与实验件一致。
        _tok_hit = bool(
            singles & set(re.findall(r"(?<![a-z0-9])([a-z0-9])(?![a-z0-9])",
                                     _norm_text(scored[0][2])))
            or [w for w in multis if w in _norm_text(scored[0][2])])
        if scored[0][0] >= 3 and scored[0][0] - s2v >= 2:
            sel = scored[:1] if _tok_hit else scored[:3]
        else:
            sel = scored[:2]
        blk = "\n".join([manual_lines[h].strip(), body[0].strip()]
                        + [x[2].strip() for x in sel])
        blocks.append("【表格抽行】问题所引表格的表头与相关行：\n" + blk[:max_chars])
        markers.extend(l.strip() for l in body if len(l.strip()) >= 8)
    return blocks, markers


# ---------- 键锚定前置＋案例整取（2026-10-29，keycase_prereg_20261029.md sha8 dc40cc84）----------
# 病灶（49 道人读归因）：错答案＝同案读错行 55%＋结论段不在上下文 37%。
# K＝题面行键（数码管数字/端子/地址）在上下文中的匹配行去重 ≤2 时整块前置；
# C＝上下文已出现的例号里按词面分取 top1，从书里取该例全文替换最弱块。
# 两开关默认 0 ⇒ 逐字节不变；改动只发生在 _retrieve_and_gate 全部门判定之后
# ⇒ 拒答集合零变化（反向桩 tests/test_keycase.py）。
_KEYROW_CODE_RE = re.compile(
    r"(?<![A-Za-z0-9])([A-Za-z]{1,6}\d{1,4}(?:\.\d+)?[A-Za-z]{0,3})(?![A-Za-z0-9])")
_KEYROW_DIGIT_RE = re.compile(r"数字\s*([0-9])|数码管[^0-9]{0,8}([0-9])|显示\s*[\"“]?([0-9])")
_CASE_FULL_MIN_SPAN = 120      # 书里切出的案例段短于这个 ⇒ 视为取不到，不动作
_CASE_FULL_COVER_SKIP = 0.95   # 案例段已被现有上下文覆盖到这个程度 ⇒ 不动作


def _keyrow_keys(question):
    codes, seen = [], set()
    for m in _KEYROW_CODE_RE.findall(question or ""):
        k = m.lower()
        if k not in seen and not k.isdigit():
            seen.add(k)
            codes.append(k)
    dm = _KEYROW_DIGIT_RE.search(question or "")
    digit = next((g for g in (dm.groups() if dm else []) if g), None)
    return codes, (int(digit) if digit else None)


def _keyrow_matched_rows(ctxs, codes, digit):
    rows = []
    for blk in ctxs:
        for ln in blk.split("\n"):
            ln = ln.strip()
            if not ln:
                continue
            if codes:
                low = ln.lower()
                if any(re.search(r"(?<![a-z0-9])" + re.escape(k) + r"(?![a-z0-9])", low)
                       for k in codes):
                    rows.append(ln)
            elif digit is not None:
                if re.search(r"(?<![0-9.])%d(?![0-9])" % digit, ln[:15]):
                    rows.append(ln)
    seen, out = set(), []
    for r in rows:
        k = _norm_text(r).lower()
        if k not in seen:
            seen.add(k)
            out.append(r)
    return out


def _keyrow_block(question, ctxs):
    """OPT_KEYROW=1 且匹配行去重后恰好 1–2 行 ⇒ 返回前置块文本；否则 None。"""
    if os.getenv("OPT_KEYROW", "0") != "1":
        return None
    codes, digit = _keyrow_keys(question)
    if not codes and digit is None:
        return None
    rows = _keyrow_matched_rows(ctxs, codes, digit)
    if not rows or len(rows) > 2:
        return None
    return "【键锚定行】问题所询编号/数码值对应的原文行：\n" + "\n".join(rows)


def _case_full_block(question, ctxs, manual_text, site_map, all_offsets,
                     max_chars=3000):
    """OPT_CASE_FULL=1 ⇒ 从上下文已出现的例号中按词面分取 top1，返回书里该例全文；否则 None。"""
    if os.getenv("OPT_CASE_FULL", "0") != "1":
        return None
    if not manual_text or site_map is None or all_offsets is None or not ctxs:
        return None
    joined = _norm_text("\n".join(ctxs))
    cands = [tk for tk in dict.fromkeys(_DIRECT_CASE_RE.findall(joined))
             if tk in site_map]
    if not cands:
        return None
    toks = _case_content_tokens(question, "")
    scored = []
    for tk in cands:
        off = (site_map.get(tk) or [None])[0]
        if off is None:
            continue
        nxt = next((o2 for o2, _t in all_offsets if o2 > off), None)
        seg_end = nxt if nxt is not None else len(manual_text)
        span = manual_text[off:min(off + max_chars, seg_end)].strip()
        if len(span) < _CASE_FULL_MIN_SPAN:
            continue
        seg_n = _norm_text(span)
        cover = sum(1 for g in [seg_n[i:i + 4] for i in range(0, max(0, len(seg_n) - 3))]
                    if g in joined[:60000]) / max(1, len(seg_n) - 3)
        if cover >= _CASE_FULL_COVER_SKIP:
            continue
        scored.append((sum(1 for w in toks if w in seg_n), off, span))
    if not scored:
        return None
    scored.sort(key=lambda x: (-x[0], x[1]))
    s1 = scored[0][0]
    s2 = scored[1][0] if len(scored) > 1 else 0
    if s1 <= 0 or (len(scored) > 1 and s1 - s2 < 1):
        return None
    off, span = scored[0][1], scored[0][2]
    return "【案例全文】问题相关案例的完整分析与处理过程：\n" + span


def _keycase_contexts(question, ctxs, manual_text, site_map, all_offsets):
    """依次套 K/C；返回 None 表示逐字节不变。
    K 前置的新块不参与 C 的"最弱块"竞争（预注册：K 不挤槽，
    C 只替换原 12 块中的最弱块——否则 C 会把刚前置的 K 短块吃掉）。"""
    out = ctxs
    kb = _keyrow_block(question, out)
    base_start = 0
    if kb is not None:
        out = [kb] + list(out)
        base_start = 1
    cb = _case_full_block(question, out, manual_text, site_map, all_offsets)
    if cb is not None and cb not in out:
        toks = _case_content_tokens(question, "")
        weakest = min(range(base_start, len(out)),
                      key=lambda i: (sum(1 for w in toks if w in _norm_text(out[i])), i))
        out = list(out)
        out[weakest] = cb
    return None if out is ctxs else out


def get_service() -> "KGRAGService":
    """懒加载单例：进程内只构建一次 KGRAGService（线程安全，双重检查）。"""
    global _service, _service_error
    if _service is None:
        with _service_lock:
            if _service is None:
                logger.info("首次构建 KGRAGService（含模型加载，预计数十秒）...")
                t0 = time.time()
                try:
                    _service = KGRAGService()
                    logger.info("KGRAGService 构建完成，耗时 %.1fs", time.time() - t0)
                except Exception as e:
                    _service_error = str(e)
                    logger.exception("KGRAGService 构建失败")
                    raise
                _service_error = None
    return _service


def _run_query(question: str, mode: str) -> dict:
    """线程池内执行的同步查询入口（供 run_in_executor 调用）。"""
    svc = get_service()
    if mode == "compare":
        return svc.query_compare(question)
    return svc.query(question)


@app.on_event("startup")
async def _prewarm_service():
    """启动后台线程预热服务实例：不阻塞事件循环，/health 立即可用。"""

    def _build():
        try:
            get_service()
        except Exception as e:
            logger.error("预热失败（首次请求将重试构建）: %s", e)

    threading.Thread(target=_build, name="kgrag-prewarm", daemon=True).start()


# ---------- 控制面端点（轻量，永不触发模型加载）----------
@app.get("/test")
async def test_endpoint():
    return {"message": "test ok v5", "version": "20260909-4"}


@app.get("/health")
async def health_check():
    """健康检查：只读进程级状态，绝不实例化 KGRAGService。"""
    ready = _service is not None
    resp = {
        "status": "ok",
        "ready": ready,
        "uptime_s": round(time.time() - _START_TIME, 1),
        "neo4j_uri": os.environ.get("NEO4J_URI"),
        "query_timeout_s": QUERY_TIMEOUT_S,
    }
    if _service_error:
        resp["status"] = "degraded"
        resp["build_error"] = _service_error
    if ready:
        try:
            resp["service"] = _service.health()
        except Exception as e:
            resp["service_health_error"] = str(e)
    return resp


# ---------- 业务端点（重活经线程池 + 超时保护）----------
def _normalize_result(result: dict) -> dict:
    """归一化查询结果：mode=query 时为平铺结构，统一包成 compare 的嵌套形态。"""
    return result if "kg" in result else {"kg": result}


def extract_chains(result: dict) -> list:
    """从结果中提取推理链"""
    chains = []
    if result.get("kg", {}).get("path_detail"):
        for path in result["kg"]["path_detail"]:
            if path.get("seq") and len(path["seq"]) > 0 and path["seq"][0].get("rel") == "推理链":
                chains.append({"text": path["seq"][0]["entity"], "meta": path["seq"][0]})
    return chains


def extract_paths(result: dict) -> list:
    """从结果中提取原始路径"""
    if result.get("kg", {}).get("paths"):
        return result["kg"]["paths"]
    return []


@app.post("/api/chain/query")
async def api_chain_query(request: Request):
    body = await request.json()
    question = body.get("question", "")
    mode = body.get("mode", "query")
    if not question:
        return JSONResponse(status_code=400, content={"error": "question required"})
    loop = asyncio.get_running_loop()
    try:
        result = await asyncio.wait_for(
            loop.run_in_executor(_query_pool, _run_query, question, mode),
            timeout=QUERY_TIMEOUT_S)
    except asyncio.TimeoutError:
        logger.warning("查询超时(%.0fs): %s", QUERY_TIMEOUT_S, question[:50])
        return JSONResponse(status_code=504, content={
            "error": f"查询超时（>{QUERY_TIMEOUT_S:.0f}s），请稍后重试或缩短问题"})
    return result


@app.get("/api/chain/paths/{question}")
async def api_chain_paths(question: str, mode: str = "query"):
    loop = asyncio.get_running_loop()
    try:
        result = await asyncio.wait_for(
            loop.run_in_executor(_query_pool, _run_query, question, mode),
            timeout=QUERY_TIMEOUT_S)
    except asyncio.TimeoutError:
        logger.warning("路径查询超时(%.0fs): %s", QUERY_TIMEOUT_S, question[:50])
        return JSONResponse(status_code=504, content={
            "error": f"查询超时（>{QUERY_TIMEOUT_S:.0f}s），请稍后重试或缩短问题"})
    # Extract chains and paths for visualization（query 模式先归一化结构）
    result = _normalize_result(result)
    return {
        "chains": extract_chains(result),
        "paths": extract_paths(result),
        "raw": result
    }


# ---------- 生成过程留痕（诊断"证据到手了但答案里丢了"）----------
_PATH_MARKS = ("[推理链]", "--[", ">--", "-[]")


def _ctx_kind(text: str) -> str:
    """把喂给模型的每一块分成"路径碎片"或"手册原文"。

    字面尺：出现推理链前缀或路径箭头就算碎片。口径写死在这里，
    复核时一眼能看出这把尺量的是什么，不靠语义判断。
    """
    t = text or ""
    return "路径碎片" if any(m in t for m in _PATH_MARKS) else "手册原文"


def _prompt_trace(prompt: str, contexts: list, raw: str,
                  max_raw: int = 6000) -> dict:
    """记录"这次到底把什么送进了模型、模型原样吐了什么"。

    不存整份 prompt（一次 ~30KB，会话存档会胀），只存它的 sha 与逐块画像；
    画像里每块的字数/种类/开头 80 字足以判定"槽位被谁占了"。
    raw_output 是**切分前**的模型原文，含【推理过程】段——切分后的 answer 丢了
    什么，只能从这里看出来。
    """
    raw = raw or ""
    prof = [{"i": i + 1, "chars": len(c or ""), "kind": _ctx_kind(c),
             "head": (c or "")[:80].replace("\n", " ")}
            for i, c in enumerate(contexts)]
    return {
        "prompt_sha": hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:12],
        "prompt_chars": len(prompt),
        "ctx_sha": hashlib.sha256("\n".join(contexts).encode("utf-8")).hexdigest()[:12],
        "ctx_chars_total": sum(p["chars"] for p in prof),
        "n_path": sum(1 for p in prof if p["kind"] == "路径碎片"),
        "n_text": sum(1 for p in prof if p["kind"] == "手册原文"),
        "ctx_profile": prof,
        # 全文上下文随 trace 下发（单用户 ~8KB/次）：前端链图"点节点看原文"靠它在
        # 送进模型的块里现场子串检索，不依赖命中率过低的 entity_chunk_map。
        "contexts": [(c or "")[:2000] for c in (contexts or [])][:16],
        "raw_chars": len(raw),
        "raw_output": raw[:max_raw],
        "raw_truncated": len(raw) > max_raw,
    }


# Existing KGRAGService class and endpoints
def default_embedding_model() -> str:
    return os.getenv("LOCAL_EMBEDDING_MODEL") or LOCAL_EMBEDDING_MODEL


def default_llm_model() -> str:
    return os.getenv("LOCAL_LLM_MODEL") or LOCAL_LLM_MODEL


def model_slug(name: str) -> str:
    """模型名 → 文件名安全串（qwen3-embedding:4b → qwen3-embedding-4b）。"""
    s = re.sub(r"[^A-Za-z0-9._-]+", "-", str(name).replace(":", "-")).strip("-").lower()
    return s or "model"


def model_cache_variant(base: str, model: str, is_dir: bool = False) -> str:
    """默认模型沿用原路径（继续用生产既有缓存），其余模型加 __<slug> 后缀。"""
    slug = model_slug(model)
    if is_dir:
        return base + "__" + slug
    root, ext = os.path.splitext(base)
    return root + "__" + slug + ext


def model_cache_paths(round_name: str, embedding_model: str = None,
                      communities_cache: str = None,
                      chunk_index_cache: str = None,
                      vector_index_cache: str = None) -> dict:
    """按 embedding 模型给出一组缓存路径。

    默认模型（.env 的 LOCAL_EMBEDDING_MODEL）沿用历史路径，保证生产既有缓存
    不被改名、不触发重建；其他模型各自一份，切换来回不互相覆盖。
    """
    cache_dir = os.path.join(PROJECT_ROOT, ".sisyphus", "evidence", "cache")
    emb = embedding_model or default_embedding_model()
    base_comm = communities_cache or os.path.join(
        cache_dir, f"communities_{round_name}.json")
    base_chunk = chunk_index_cache or os.path.join(
        cache_dir, f"chunk_index_{round_name}.json")
    base_vec = vector_index_cache or os.path.join(
        cache_dir, f"vector_index_{round_name}")
    is_default = emb == default_embedding_model()
    slug = model_slug(emb)
    return {
        "embedding_model": emb,
        "is_default": is_default,
        "communities": base_comm,
        "chunk": base_chunk if is_default else model_cache_variant(base_chunk, emb),
        "vector": base_vec if is_default else model_cache_variant(base_vec, emb, is_dir=True),
        "community_embeddings": None if is_default else os.path.join(
            cache_dir, f"community_embeddings_{round_name}__{slug}.json"),
        "entity": None if is_default else os.path.join(
            cache_dir, f"entity_embeddings_{round_name}__{slug}.json"),
        # OPT_C 滑窗 dense 通道的向量缓存：默认模型沿用历史文件
        "sliding_embed": (os.path.join(cache_dir, "sliding_embed_cache.json")
                          if is_default else
                          os.path.join(cache_dir,
                                       f"sliding_embed_cache__{slug}.json")),
    }


def _apply_community_embedding_cache(communities: list, embed, model: str,
                                     path: str, progress=None):
    """非默认 embedding：按模型重算/复用社区 description 向量（深拷贝，不污染原缓存）。"""
    comms = copy.deepcopy(communities)
    descs = [str(c.get("description") or c.get("summary") or "") for c in comms]
    desc_sha = hashlib.sha256(
        "\n\x1f".join(descs).encode("utf-8")).hexdigest()[:16]
    saved = {}
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                cached = json.load(f)
            if cached.get("model") == model and cached.get("desc_sha") == desc_sha:
                saved = cached.get("embeddings") or {}
        except Exception as e:
            logger.warning("社区向量缓存读取失败（将重算）: %s", e)
    total = len(comms)
    rows = {}
    dirty = len(saved) < total
    for i, comm in enumerate(comms):
        vec = saved.get(str(i))
        if not (isinstance(vec, list) and len(vec) > 1):
            try:
                vec = embed.get_text_embedding(descs[i])
            except Exception as e:
                logger.warning("社区描述向量计算失败（第 %d 个）: %s", i, e)
                vec = None
            dirty = True
        comm["description_embedding"] = vec
        rows[str(i)] = vec
        if progress and (i + 1) % 20 == 0:
            progress("communities", i + 1, total)
    if dirty:
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                json.dump({"model": model, "desc_sha": desc_sha,
                           "n": total, "embeddings": rows},
                          f, ensure_ascii=False)
        except Exception as e:
            logger.warning("社区向量缓存回写失败: %s", e)
    if progress:
        progress("communities", total, total)
    return comms, ("cache" if not dirty else "built")


class KGRAGService:
    """KG-RAG 问答服务（微观/宏观/对比）。"""

    def __init__(self, manual_name: str = None, round_name: str = None,
                 communities_cache: str = None, chunk_index_cache: str = None,
                 neo4j_uri: str = None, embedding_model: str = None,
                 llm_model: str = None, extract_path: str = None,
                 progress=None):
        # 允许环境变量覆盖，默认 GSK983（避免 Windows 环境变量中文传递编码问题）
        self.manual_name = manual_name or os.getenv("SERVING_MANUAL", "13_GSK983数控系统维修手册_中文.md")
        self.round_name = round_name or os.getenv("SERVING_ROUND", "13_GSK983数控系统维修手册_中文")
        # 上传手册建在 NEO4J_BUILD_URI(7688)，预构建手册在评测库(7690)；
        # 由调用方按手册指定，None 保持原来的查询库默认。
        self._neo4j_uri = neo4j_uri
        prefs = {}
        if os.getenv("SERVING_IGNORE_MODEL_PREFS") != "1":
            try:
                from backend.model_prefs import load_prefs
                prefs = load_prefs()
            except Exception:
                prefs = {}
        self.embedding_model = (embedding_model or prefs.get("embedding_model")
                                or default_embedding_model())
        self.llm_model = (llm_model or prefs.get("llm_model")
                          or default_llm_model())
        self._progress = progress
        self.manual_path = os.path.join(PROJECT_ROOT, "data", "manuals", self.manual_name)
        self.extract_path = extract_path or os.path.join(
            PROJECT_ROOT, "reports", "kg_visual",
            os.getenv("SERVING_EXTRACT", "llm_extract_v5_shukongdashi_500cases.json"),
        )
        paths = model_cache_paths(
            self.round_name, self.embedding_model,
            communities_cache=communities_cache,
            chunk_index_cache=chunk_index_cache)
        self.cache_paths = paths
        self.communities_cache = paths["communities"]
        self.chunk_index_cache = paths["chunk"]
        # Vector 索引落盘目录（新前缀 vector_index_，不与冻结评测产物同名）：
        # 同一轮次、同一 embedding 模型、同一分块结果才认，见 backend/vector_rag.py
        self.vector_index_cache = paths["vector"]
        self.community_embeddings_cache = paths["community_embeddings"]
        self.entity_embeddings_cache = paths["entity"]
        self.sliding_embed_cache = paths["sliding_embed"]

        logger.info("初始化 KGRAGService: manual=%s round=%s embed=%s llm=%s",
                    self.manual_name, self.round_name,
                    self.embedding_model, self.llm_model)
        self._build_engines()

    def _report(self, stage, done, total):
        if self._progress:
            try:
                self._progress(stage, done, total)
            except Exception:
                pass

    # ---------- 初始化 ----------
    def _build_engines(self):
        self._report("connect", 0, 1)
        self.store = GraphStore(uri=self._neo4j_uri)
        self._report("embedding_init", 0, 1)
        self.embed = get_embed_model(model_name=self.embedding_model)
        self.llm = get_llm(model_name=self.llm_model)
        # 答案侧只许本地模型，配错就不起服务（显式传入当前选择）
        require_local_answer_llm(self.llm, expected_model=self.llm_model)
        self._report("embedding_init", 1, 1)

        # 社区（摘要缓存）
        communities = []
        if os.path.exists(self.communities_cache):
            with open(self.communities_cache, "r", encoding="utf-8") as f:
                communities = json.load(f)
            logger.info("社区缓存: %d 个", len(communities))
        else:
            logger.warning("社区缓存不存在: %s（宏观查询不可用）", self.communities_cache)

        self.community_embedding_source = "cache-file"
        if communities and self.community_embeddings_cache:
            communities, self.community_embedding_source = \
                _apply_community_embedding_cache(
                    communities, self.embed, self.embedding_model,
                    self.community_embeddings_cache, progress=self._report)
            logger.info("社区向量: %s（%d 个，模型 %s）",
                        self.community_embedding_source, len(communities),
                        self.embedding_model)

        self._report("graph", 0, 1)
        self.manual_eng = ManualGraphQueryEngine(
            self.store, self.embed, source_manual=self.manual_name,
            embedding_cache_path=self.entity_embeddings_cache,
            embedding_cache_model=(self.embedding_model
                                   if self.entity_embeddings_cache else None),
            progress=self._report if self.entity_embeddings_cache else None)
        self._report("graph", 1, 1)
        self.comm_eng = CommunityQueryEngine(
            self.store, self.embed, communities,
            chunk_index_cache=self.chunk_index_cache,
            chunk_index_cache_model=self.embedding_model,
            reuse_chunk_embeddings=True,
            progress=self._report) if communities else None

        # entity_chunk_map（路径→原文证据回填，与评测一致）
        self.entity_chunk_map = {}
        self.chunk_texts = []
        # 例号直取＋消歧通道（2026-10-27）：位点图只依赖手册原文，初始化建一次。
        # 只在 OPT_DIRECT_CASE=1 时参与装配（_retrieve_and_gate 里判）。
        self.manual_text = ""
        self._case_site_map = {}
        self._case_site_all = []
        try:
            with open(self.manual_path, "r", encoding="utf-8") as f:
                self.manual_text = f.read().replace("\ufeff", "")
            self._case_site_map, self._case_site_all, self._case_first_any = \
                _build_case_site_map(self.manual_text)
            logger.info("例号位点图: %d 个编号 / %d 个定义位点",
                        len(self._case_site_map), len(self._case_site_all))
        except Exception as e:
            logger.warning("例号位点图构建失败（直取通道不可用）: %s", e)
        # 表格抽行通道：表号→表头行号索引，只依赖手册原文，初始化建一次。
        self._tbl_index = {}
        self._manual_lines = []
        try:
            if self.manual_text:
                self._manual_lines = self.manual_text.split("\n")
                self._tbl_index = _build_table_index(self._manual_lines)
                logger.info("表格索引: %d 个表号", len(self._tbl_index))
        except Exception as e:
            logger.warning("表格索引构建失败（抽行通道不可用）: %s", e)
        if os.path.exists(self.extract_path):
            with open(self.extract_path, "r", encoding="utf-8") as f:
                triples = json.load(f)
            chunks = parse_manual_chunks(self.manual_path)
            self.chunk_texts = [c.text for c in chunks]
            for t in triples:
                cid = t.get("chunk_id")
                if cid is None:
                    continue
                for ent in (t.get("s"), t.get("o")):
                    if ent:
                        self.entity_chunk_map.setdefault(str(ent), set()).add(cid)
        logger.info("图谱引擎就绪: 实体 %d, 社区 %s",
                    len(self.manual_eng._entity_names),
                    len(communities) if communities else "N/A")

        # Vector 基线（结构化分块，适配 GSK983 结构化手册）
        try:
            chunks = parse_manual_structured(self.manual_path)
            docs = [Document(text=c.text,
                             metadata={"chunk_id": c.chunk_id, "source": c.source})
                    for c in chunks]
            # Vector RAG 参数优化：针对 GSK983 结构化手册
            from functools import partial
            self.vec_eng = VectorRAGQueryEngine(
                docs, self.embed, self.llm, top_k=20,
                similarity_threshold=0.35,
                context_truncator=partial(_truncate_contexts, max_ctx=16),
                prompt_template=RAGASEvaluator._answer_template(),
                answer_generator=self._vector_generate,
                index_cache_dir=self.vector_index_cache,
                progress=self._report)
            logger.info("Vector 引擎就绪: %d chunks（索引来源 %s）",
                        len(chunks), getattr(self.vec_eng, "index_source", "?"))
        except Exception as e:
            logger.warning("Vector 引擎初始化失败（对比功能不可用）: %s", e)
            self.vec_eng = None
        # OPT_C 滑窗 dense 通道：构建期就按当前 embedding 建好，
        # 避免切换后的第一道题替整条通道付 363 块 embedding 的时间。
        self.sliding_retriever_ready = None
        if os.getenv("OPT_C", "0") == "1":
            self._report("sliding", 0, 1)
            try:
                from backend.evaluator import _get_sliding_retriever
                self.sliding_retriever_ready = _get_sliding_retriever(
                    model_name=self.embedding_model,
                    cache_path=self.sliding_embed_cache,
                    manual_path=self.manual_path) is not None
                logger.info("OPT_C 滑窗检索器: %s（embedding=%s）",
                            "就绪" if self.sliding_retriever_ready else "不可用",
                            self.embedding_model)
            except Exception as e:
                self.sliding_retriever_ready = False
                logger.warning("OPT_C 滑窗检索器构建失败（查询时将重试）: %s", e)
            self._report("sliding", 1, 1)
        self._report("ready", 1, 1)

    def switch_llm(self, model_name: str) -> str:
        """进程内热切换答案 LLM（embedding/索引不动，秒级完成）。

        先构造并校验新模型，成功后才替换引用；失败时旧模型继续服务。
        """
        new_llm = get_llm(model_name=model_name)
        require_local_answer_llm(new_llm, expected_model=model_name)
        self.llm = new_llm
        self.llm_model = model_name
        if getattr(self, "vec_eng", None) is not None:
            self.vec_eng.llm = new_llm
        logger.info("答案 LLM 已切换: %s", model_name)
        return model_name

    def _sliding_stats(self) -> dict:
        try:
            from backend.evaluator import get_retriever_stats
            return get_retriever_stats(self.embedding_model)
        except Exception:
            return {}

    def cache_status(self) -> dict:
        """当前模型的一组缓存状态（只读，不触发计算）。"""
        def file_info(path):
            if not path:
                return {"path": None, "exists": False}
            return {"path": path, "exists": os.path.exists(path),
                    "size_mb": (round(os.path.getsize(path) / 1048576, 1)
                                if os.path.isfile(path) else None)}

        vector_dir = self.vector_index_cache
        return {
            "embedding_model": self.embedding_model,
            "is_default_embedding": self.embedding_model == default_embedding_model(),
            "vector": {
                "path": vector_dir,
                "exists": os.path.exists(os.path.join(vector_dir, "_meta.json")),
                "source": getattr(self.vec_eng, "index_source", None),
                "reason": getattr(self.vec_eng, "index_reason", None),
            },
            "chunk": dict(file_info(self.chunk_index_cache),
                          source=getattr(self.comm_eng, "_chunk_index_source", None)),
            "entity": dict(file_info(self.entity_embeddings_cache),
                           source=getattr(self.manual_eng, "embedding_source", None)),
            "communities": dict(
                file_info(self.community_embeddings_cache or self.communities_cache),
                source=self.community_embedding_source),
            "sliding": dict(
                file_info(self.sliding_embed_cache),
                enabled=os.getenv("OPT_C", "0") == "1",
                source=("ready" if self.sliding_retriever_ready else None)
                if os.getenv("OPT_C", "0") == "1" else "off"),
        }

    def _answer_via_local(self, prompt: str) -> str:
        """面向用户的答案文本只经本地模型生成（约束见 require_local_answer_llm）。

        Ollama 偶发 500（"model runner has unexpectedly stopped"，2026-10-25/26
        各实测一次）：一次短退避重试，第二次仍失败才向上抛——瞬时抖动不再
        让整题变"生成失败"。
        """
        import time as _time
        for attempt in range(2):
            try:
                resp = self.llm.chat([ChatMessage(role="user", content=prompt)])
                return (resp.message.content or u"").strip()
            except Exception:
                if attempt == 1:
                    raise
                logger.warning("本地模型调用失败，3 秒后重试一次")
                _time.sleep(3.0)

    # ---------- 检索 + 拒答门（query / stream_query 共用，保证口径一致） ----------
    def _retrieve_and_gate(self, question: str, max_ctx: int = 16):
        """图谱检索 + 拒答门，返回 (graph_contexts, refusal_reason)。

        query() 与 stream_query() 必须走同一份逻辑：否则"流式版"和"非流式版"
        的拒答判定会漂移，线上表现与评测就不再一致。
        """
        is_ooc = False
        # 方案 C 的 dense 检索器：只在 OPT_C=1 时接（与评测侧同一条件）。
        # OPT_C 不为 1 时传 None ⇒ 装配逐位与改动前相同（常驻门
        # analysis/prod_assembly_parity_gate.py 的 M3 就是钉这条"接线无副作用"）。
        # 开了却拿不到检索器 ⇒ 硬停，不许静默退化成 base 形状对外宣称开了优化。
        _dense = None
        if os.getenv("OPT_C", "0") == "1":
            from backend.evaluator import _get_sliding_retriever
            _dense = _get_sliding_retriever(
                model_name=self.embedding_model,
                cache_path=self.sliding_embed_cache,
                manual_path=self.manual_path)
            if _dense is None:
                raise RuntimeError(
                    "[线上完整性] OPT_C=1 但滑窗 dense 检索器不可用："
                    "请检查本机 Ollama 与 sliding_embed_cache.json，或显式设 OPT_C=0。")
        _mc = max_ctx
        if _SERVING_EVAL_PARITY:
            _mc = int(os.getenv("KG_MAX_CTX", str(max_ctx)))   # 对齐评测的槽预算
        raw_ctxs = _collect_graph_contexts(
            question, self.manual_eng, self.comm_eng, is_ooc=is_ooc,
            max_ctx=_mc, entity_chunk_map=self.entity_chunk_map,
            chunk_texts=self.chunk_texts, dense_retriever=_dense)
        # 例号直取＋消歧通道（2026-10-27）：题面含例N 时把所引案例的定义正文
        # 放到最前（挤掉最弱槽）。开关判在 _direct_case_contexts 内部（默认关
        # ⇒ 返回 []，raw_ctxs 逐位不变）。
        _dc = _direct_case_contexts(
            question, self.manual_text, self._case_site_map,
            self._case_site_all, first_any=self._case_first_any)
        if _dc:
            _dc_set = set(_dc)
            raw_ctxs = _dc + [c for c in raw_ctxs if c not in _dc_set]
        # 表格抽行通道（2026-10-27）：题面引用 表X-Y ⇒ 块放最前，并把底座里
        # 整块携带该表体（命中 ≥2 条表体行）的检索块剔掉——块与底座并存时
        # 模型会无视块仍抄底座（tablefix_exp4 new_040 实测）。
        try:
            _tr_blocks, _tr_markers = _table_row_contexts(
                question, self.manual_text, getattr(self, "_tbl_index", None),
                getattr(self, "_manual_lines", None))
        except Exception:
            _tr_blocks, _tr_markers = [], []
            logger.exception("表格抽行通道异常（按无块处理）")
        if _tr_markers:
            _mk = [m for m in _tr_markers if m]
            raw_ctxs = [c for c in raw_ctxs
                        if sum(1 for m in _mk if m in c) < 2]
        if _tr_blocks:
            raw_ctxs = _tr_blocks + raw_ctxs
        # 报警号病例锚定（2026-11-01）：开关在 _alarm_case_contexts 内部判，
        # 默认关 ⇒ 返回 []，raw_ctxs 逐位不变。病例块放在表格块之后、截断之前，
        # 保证它不被 12/16 槽预算挤掉，并且仍在最终上下文最前。
        _ac_blocks, _ac_meta = _alarm_case_contexts(question, self.manual_text)
        if _ac_blocks:
            raw_ctxs = _alarm_apply_anchor(
                raw_ctxs, _ac_blocks, _ac_meta, question)
            _ac_set = set(_ac_blocks)
            raw_ctxs = _ac_blocks + [c for c in raw_ctxs if c not in _ac_set]
        graph_contexts = _truncate_contexts(raw_ctxs, max_ctx=_mc)
        # 装配末道相关度重排（默认关；与评测侧同一函数同一开关，见 evaluator._rerank_contexts）
        graph_contexts = _rerank_contexts(graph_contexts, question,
                                          corpus=getattr(self, "chunk_texts", None))
        if _ac_blocks:
            _ac_set = set(_ac_blocks)
            graph_contexts = _ac_blocks + [
                c for c in graph_contexts if c not in _ac_set]

        # OOC 拒答门（与评测一致：relevance 低或对象不一致 → 拒答）
        # 2026-10-04 D5 分层补丁：打分走 _ooc_gate_relevance（门层专用 token/
        # 词组抽取＋分通道结算），检索排序用的 _query_code_tokens 零接触。
        refusal_reason = None

        # 厂商/系统型号门（2026-10-12）：拦截跨厂商/跨型号超纲问题
        # 原理：从图谱查询支持的厂商/系统，问题中出现不支持的厂商 → 拒答
        # 开关：OOC_VENDOR_GATE=1 启用（默认启用），=0 禁用
        try:
            from backend.vendor_gate import vendor_gate_hook
            refusal_reason, vendor_rec = vendor_gate_hook(question, refusal_reason)
            if vendor_rec is not None:
                logger.info("厂商门: %s", json.dumps(vendor_rec, ensure_ascii=False))
        except Exception as e:
            logger.warning("厂商门异常（降级放行）: %s", e)

        # 只有厂商门未拒答时，才检查 OOC 门
        if refusal_reason is None and raw_ctxs and graph_contexts:
            relevance = _ooc_gate_relevance(question, graph_contexts)
            ooc_mismatch = _ooc_object_mismatch(question, graph_contexts)
            if relevance < _OOC_RELEVANCE_THRESHOLD or ooc_mismatch:
                refusal_reason = {
                    "relevance": round(relevance, 3),
                    "object_mismatch": ooc_mismatch,
                }

        # 领域语义邻近度门（影子先行；预注册：半月reports/领域语义邻近度拒答门接入
        # 生产_预注册与判读表_20261010.md §3/§4）。两个开关默认关 ⇒ 生产路径零开销、
        # 零行为改变；任何异常降级现门，不许 500。
        try:
            refusal_reason, sent_rec = sent_cov_gate_hook(
                question, graph_contexts, refusal_reason,
                embed_fn=self.embed.get_text_embedding)
            if sent_rec is not None:
                logger.info("sent_cov 影子: %s", json.dumps(sent_rec, ensure_ascii=False))
        except Exception as e:
            logger.warning("sent_cov 钩子异常（降级现门）: %s", e)

        # 口语措辞放行票（2026-10-28）：撤相关度门单独开火的误拦（口语卷实测
        # 救 4/6、超纲漏 0/37，见 colloq_experiment_20261028.md 预注册）。
        # 开关 OOC_RELSEM=1 启用（默认 0＝零行为改变）；异常降级现门。
        try:
            refusal_reason, _relsem_rec = relsem_release_hook(
                question, graph_contexts, refusal_reason)
        except Exception as e:
            logger.warning("relsem 钩子异常（降级现门）: %s", e)

        # 联合门（2026-10-20 装机，作者指令）：七特征两段式拒答的加法钩子——只增拒不
        # 撤拒，仅在前级门（厂商/词法/对象/句子票）放行时评估。OOC_JOINT_GATE=0 关闭；
        # 任何异常降级现门。部署系数与冻结阈值：_jointgate_deploy_20261020.json
        # （sha8 5ac8fdd5）。代码落盘即生效于下次服务启动（重启动作归作者）。
        try:
            from backend.joint_gate import joint_gate_hook
            refusal_reason, jg_rec = joint_gate_hook(
                question, graph_contexts, refusal_reason, manual_eng=self.manual_eng)
            if jg_rec is not None:
                logger.info("joint_gate 拒答: %s", json.dumps(jg_rec, ensure_ascii=False))
        except Exception as e:
            logger.warning("joint_gate 钩子异常（降级现门）: %s", e)

        # 键锚定行 + 案例全文上下文增强：
        # 只在门后放行、有上下文的题上做"加块不换门"——拒答集合零变化。
        # 两开关 OPT_KEYROW / OPT_CASE_FULL 默认 0 ⇒ _keycase_contexts 不被调，
        # 装配逐字节与改动前相同。
        if refusal_reason is None and graph_contexts \
                and (os.environ.get("OPT_KEYROW") == "1"
                     or os.environ.get("OPT_CASE_FULL") == "1"):
            try:
                _kc = _keycase_contexts(
                    question, graph_contexts, self.manual_text,
                    self._case_site_map, self._case_site_all)
                if _kc is not None:
                    graph_contexts = _kc
            except Exception as e:
                logger.warning("keycase 增强异常（保留原上下文）: %s", e)

        # 报警病例块永远置顶：keycase 等后置增强不得把它挤走。
        if _ac_blocks and refusal_reason is None and graph_contexts:
            _ac_set = set(_ac_blocks)
            graph_contexts = _ac_blocks + [
                c for c in graph_contexts if c not in _ac_set]

        return graph_contexts, refusal_reason

    # ---------- 微观问答（图谱路径 + 答案生成）----------
    @staticmethod
    def _refusal_answer(refusal_reason, fmt_paths):
        """拒答时的人话解释（2026-10-24 产品化 2.1）。

        判定逻辑零改动（refusal/refusal_detail 字段与各门结构不变），
        只把一句话冷硬答复升级为结构化解释：为什么拒、判定依据、
        检索概况（召回线索展示，帮用户理解"查到了什么但不可靠"）、建议问法。
        refusal_reason 按各门实际返回容错取用：OOC 门带 relevance/object_mismatch，
        厂商门/词法门/联合门结构可能不同，缺字段一律走泛化表述。
        """
        detail = refusal_reason or {}
        reasons = []
        if detail.get("vendor_gate"):
            reasons.append(detail.get("reason") or "图谱中未收录问题提到的厂商或型号词条")
        if detail.get("object_mismatch"):
            reasons.append("检索到的内容与提问对象不匹配（知识库中没有该对象的相关记载）")
        jg = detail.get("joint_gate") or {}
        if jg:
            # 只补措辞：判定与阈值一律沿用联合门自己的产出，这里一个字节都不改
            reasons.append("综合校验判定该问题超出本手册覆盖范围"
                           "（综合得分 %.2f，判线 %.2f）"
                           % (float(jg.get("score") or 0), float(jg.get("theta") or 0)))
        sc = detail.get("sent_cov")
        if sc is not None and not reasons:
            reasons.append("问题与这本手册的领域语义相似度偏低（相似度 %s）" % sc)
        rel = detail.get("relevance")
        if rel is not None and not reasons:
            reasons.append("检索内容与问题的相关度过低（相关度 %.2f）" % rel)
        if not reasons:
            reasons.append("未通过知识范围校验")
        glimpse = "；".join(str(p) for p in fmt_paths[:3]) if fmt_paths else "无有效候选"
        return (
            "【未能回答】根据当前知识库未检索到与问题可靠相关的内容。"
            "为避免误导，本系统选择不作答，而不是勉强编造。\n"
            "【判定依据】%s。\n"
            "【检索概况】图谱召回的候选线索：%s。\n"
            "【建议】请确认问题是否属于本手册覆盖的设备与故障范围；"
            "可尝试补充报警号、系统型号（如 FANUC、SIEMENS）或具体故障现象后重新提问。"
            % ("；".join(reasons), glimpse)
        )

    def query(self, question: str, max_ctx: int = 16) -> dict:
        graph_contexts, refusal_reason = self._retrieve_and_gate(question, max_ctx)

        trace = None
        if refusal_reason:
            answer = "根据当前知识库未检索到与问题相关的内容，无法回答。"
        elif graph_contexts and graph_contexts != [""]:
            prompt = RAGASEvaluator._answer_template().format(
                context="\n".join(graph_contexts), question=question)
            try:
                raw = self._answer_via_local(prompt)
                answer = _split_cot_output(raw)[0]
                trace = _prompt_trace(prompt, graph_contexts, raw)
            except Exception as e:
                logger.error("答案生成失败: %s", e)
                answer = "答案生成失败（LLM 调用异常）。"
        else:
            answer = "知识图谱中未找到相关信息。"

        # 图谱路径：一次检索，同时产出单行文本与逐跳结构化
        fmt_paths, detail_paths = self._collect_paths(question)

        # 2026-10-24 产品化 2.1：拒答 answer 升级为人话解释（判定零改动，
        # refusal/refusal_detail 结构不变，只是 answer 文本更友好）。
        if refusal_reason:
            answer = self._refusal_answer(refusal_reason, fmt_paths)

        return {
            "answer": answer,
            "refusal": refusal_reason is not None,
            "refusal_detail": refusal_reason,
            "n_contexts": len(graph_contexts),
            "paths": fmt_paths,
            "path_detail": detail_paths,
            "trace": trace,
        }

    # ---------- 微观问答·流式（图谱路径先出，再逐块吐最终答案）----------
    def stream_query(self, question: str, max_ctx: int = 16):
        """流式微观问答生成器。事件（dict，由 api_server 包成 SSE）：

          - {"type":"meta", refusal, refusal_detail, n_contexts, paths, path_detail}
          - {"type":"reasoning", "chars": int}      # 标记出现前的推理阶段（只报进度，不吐内容）
          - {"type":"answer_delta", "text": str}    # 【最终答案】之后的增量
          - {"type":"done", answer, ...同 meta}     # 终态，字段与 query() 的返回一一对应
          - {"type":"error", "message": str}        # 未产出任何答案时的失败（不假装成功）

        口径保证：检索/拒答门共用 _retrieve_and_gate；答案切分共用同一个【最终答案】标记，
        与 query() 的 _split_cot_output 同义。只多一层"边生成边吐"，不改判定。
        """
        graph_contexts, refusal_reason = self._retrieve_and_gate(question, max_ctx)

        # 图谱路径与答案生成无关，先算出来随 meta 下发 ⇒ 前端可立刻画图，不必等 LLM。
        fmt_paths, detail_paths = self._collect_paths(question)
        meta = {
            "refusal": refusal_reason is not None,
            "refusal_detail": refusal_reason,
            "n_contexts": len(graph_contexts),
            "paths": fmt_paths,
            "path_detail": detail_paths,
        }
        yield dict(meta, type="meta")

        # 拒答 / 无上下文：没有可流式生成的内容，直接给终态。
        if refusal_reason or not (graph_contexts and graph_contexts != [""]):
            answer = (self._refusal_answer(refusal_reason, fmt_paths)
                      if refusal_reason else "知识图谱中未找到相关信息。")
            yield {"type": "answer_delta", "text": answer}
            yield dict(meta, type="done", answer=answer)
            return

        prompt = RAGASEvaluator._answer_template().format(
            context="\n".join(graph_contexts), question=question)

        marker = "【最终答案】"
        buf = ""           # 标记出现前的缓冲（推理阶段，不外发）
        answer_text = ""   # 标记之后的最终答案累积
        emitted = 0        # 已下发的答案字符数
        after = False      # 是否已越过标记
        last_progress = 0
        raw_all = []       # 模型原样吐出的全量（含【推理过程】），只为留痕
        try:
            for chunk in self.llm.stream_chat([ChatMessage(role="user", content=prompt)]):
                piece = getattr(chunk, "delta", None) or ""
                if not piece:
                    continue
                raw_all.append(piece)
                if not after:
                    buf += piece
                    idx = buf.find(marker)
                    if idx < 0:
                        if len(buf) - last_progress >= 60:
                            last_progress = len(buf)
                            yield {"type": "reasoning", "chars": len(buf)}
                        continue
                    after = True
                    answer_text = buf[idx + len(marker):].lstrip()
                else:
                    answer_text += piece
                if len(answer_text) > emitted:
                    delta = answer_text[emitted:]
                    emitted = len(answer_text)
                    yield {"type": "answer_delta", "text": delta}
        except Exception as e:
            logger.error("流式答案生成失败: %s", e)
            if not answer_text:
                # 一个字都没吐出去：如实报错，别把异常折算成"空答案成功"
                yield {"type": "error", "message": "答案生成失败（LLM 调用异常）。"}
                return

        # 兜底：模型始终没输出标记 ⇒ 与 _split_cot_output 同口径，整段当答案。
        if not after:
            answer_text = buf.strip() or "（模型未返回内容）"
            yield {"type": "answer_delta", "text": answer_text}

        answer = answer_text.strip()
        yield dict(meta, type="done", answer=answer,
                   trace=_prompt_trace(prompt, graph_contexts, "".join(raw_all)))

    # ---------- 宏观问答（社区召回 + 原始 Chunk 总结）----------
    def query_macro(self, question: str, max_comm: int = 3) -> dict:
        if self.comm_eng is None:
            return {"answer": "社区数据未构建（宏观查询不可用）。", "communities": []}
        matched, ctx = self.comm_eng.query(question)
        if not ctx:
            return {"answer": "未检索到相关社区。", "communities": []}
        context = "\n\n".join(ctx[:max_comm])
        prompt = (
            "你是一名工业设备故障诊断专家。请根据以下检索到的原始手册片段，"
            "总结回答用户的问题。\n"
            "【硬性约束】\n"
            "- 每一句话都必须在下方【原始手册片段】中找到出处，逐句标注依据。\n"
            "- 严禁编造、严禁引入片段之外的外部知识。\n"
            "- 若片段不足以回答，明确说明「根据当前知识库无法总结」。\n\n"
            "【原始手册片段】\n" + context +
            f"\n\n【问题】{question}\n\n【总结】"
        )
        try:
            answer = self._answer_via_local(prompt)
        except Exception as e:
            logger.error("宏观总结失败: %s", e)
            answer = "总结生成失败（LLM 调用异常）。"
        return {
            "answer": answer,
            "communities": [{"id": c.get("id"), "nodes": len(c.get("nodes", [])),
                             "summary": (c.get("summary") or c.get("report") or "")[:400],
                             "top_nodes": [str(n) for n in (c.get("nodes") or [])][:8]}
                            for c in matched[:max_comm]],
            "n_context_chars": len(context),
        }

    # ---------- 对比（KG vs Vector）----------
    def _vector_generate(self, prompt: str) -> str:
        """Vector RAG 答案生成器：与 KG RAG 共用同一 CoT prompt + LLM。"""
        try:
            return _split_cot_output(self._answer_via_local(prompt))[0]
        except Exception as e:
            logger.error("Vector 答案生成失败: %s", e)
            return "根据手册未找到相关答案。"

    def query_compare(self, question: str) -> dict:
        kg = self.query(question)
        vec = {"answer": "Vector 引擎不可用。", "n_contexts": 0}
        if self.vec_eng is not None:
            try:
                r = self.vec_eng.query(question)
                contexts = r.get("contexts", []) or []
                scores = r.get("scores", []) or []
                vec = {
                    "answer": str(r.get("answer", "")),
                    "n_contexts": len(contexts),
                    "scores": [round(s, 3) for s in scores[:5]],
                    # 候选片段（text+score 配对），供前端展示"基于哪些片段回答"
                    "candidates": [
                        {"text": c, "score": round(float(s), 3)}
                        for c, s in zip(contexts, scores)
                    ][:8],
                }
            except Exception as e:
                vec = {"answer": f"Vector 查询失败: {e}", "n_contexts": 0,
                       "candidates": []}
        return {"kg": kg, "vector": vec}

    # ---------- 辅助 ----------
    def _collect_paths(self, question: str, max_paths: int = 5):
        """调图谱引擎查询一次，返回 (格式化串列表, 逐跳结构化列表)。

        逐跳结构化：每路径为 {"seq": [{"entity":..., "rel":..., "dir":...}, ...],
        "text": 单行文本}。seq 含每跳的实体与关系，供前端展示"从哪步到哪步"。
        """
        paths = self.manual_eng.query(question)
        fmt_out = []
        detail_out = []
        for p in paths[:max_paths]:
            if len(p) >= 3:
                names, rels, dirs = p[0], p[1], p[2]
                # 推理链聚合行：names 长度 1、关系为"推理链"，整行文本即链
                if len(names) == 1 and rels and rels[0] == "推理链":
                    seq = [{"entity": str(names[0]), "rel": "推理链", "dir": "?"}]
                    fmt_out.append(str(names[0]))
                    detail_out.append({"seq": seq, "text": str(names[0])})
                    continue
                parts = [str(names[0])]
                seq = [{"entity": str(names[0]), "rel": None, "dir": None}]
                for i, rel in enumerate(rels):
                    if i + 1 < len(names):
                        d = dirs[i] if i < len(dirs) else "?"
                        arrow = "<--" if d == "-" else "-->"
                        parts.append(f"{arrow}[{rel}]")
                        parts.append(str(names[i + 1]))
                        seq.append({"entity": str(names[i + 1]),
                                    "rel": str(rel), "dir": d})
                fmt_out.append("".join(parts))
                detail_out.append({"seq": seq, "text": "".join(parts)})
            elif p:
                fmt_out.append(str(p[0]))
                detail_out.append({"seq": [{"entity": str(p[0]), "rel": None, "dir": None}],
                                   "text": str(p[0])})
        return fmt_out, detail_out

    def _format_paths(self, question: str, max_paths: int = 5) -> list:
        return self._collect_paths(question, max_paths)[0]

    def _path_detail(self, question: str, max_paths: int = 5) -> list:
        return self._collect_paths(question, max_paths)[1]

    def health(self) -> dict:
        return {
            "status": "ok",
            "manual": self.manual_name,
            "embedding_model": self.embedding_model,
            "llm_model": self.llm_model,
            "embedding_source": getattr(self.manual_eng, "embedding_source", None),
            "community_embedding_source": self.community_embedding_source,
            "chunk_index_source": (getattr(self.comm_eng, "_chunk_index_source", None)
                                   if self.comm_eng else None),
            "sliding_retriever_ready": self.sliding_retriever_ready,
            "sliding_calls": self._sliding_stats().get("calls"),
            "sliding_hits": self._sliding_stats().get("hits"),
            "opt_c": os.getenv("OPT_C", "0") == "1",
            # 轮次由服务自己报：/health/lite 未就绪时只能读 env，而 env 不随
            # "设为当前"改变，之前只能靠社区数反推线上是哪一轮。
            "round": self.round_name,
            "graph_entities": len(self.manual_eng._entity_names),
            "communities": len(self.comm_eng.communities) if self.comm_eng else 0,
            "vector_ready": self.vec_eng is not None,
            # 甲的开关可观测：cache＝复用了落盘索引（跳过 120 块 embedding），
            # built＝这次现算的。
            "vector_index": (getattr(self.vec_eng, "index_source", None)
                             if self.vec_eng else None),
            "neo4j_uri": self.store._uri,
        }


if __name__ == "__main__":
    import uvicorn
    import asyncio
    import sys
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    logging.basicConfig(level=logging.DEBUG)
    uvicorn.run("serve:app", host="0.0.0.0", port=8000, reload=False, loop="asyncio", log_level="debug")
