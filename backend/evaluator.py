"""RAGAS 评估器 —— 对比 Vector RAG 与 KG-RAG 的检索质量。

使用本地 Ollama 作为 LLM 法官 (judge)，RAGAS 0.4.3 指标:
- faithfulness: 答案是否仅基于提供的上下文
- answer_relevancy: 答案与问题的相关程度
- context_precision: 检索到的上下文中相关片段的比例
- context_recall: ground_truth 中信息被上下文覆盖的比例
"""
import os
import json
import logging
import warnings
import re
import sys
import math
import hashlib
import threading
from dataclasses import dataclass
from typing import Callable, Optional

# Windows fix (v9, selector policy): make ragas' asyncio.run() use the
# selector event loop policy instead of the Proactor one. ProactorEventLoop
# close() hangs forever when httpx keepalive connections to the proxy stay
# registered on the IOCP (observed twice in v9: 80min and 5min stalls in
# IocpProactor.close() -> `while self._cache` after aborting transports and
# force-closing sockets; socket.close() is refcounted so the fd never
# closed and pending overlapped ops never completed). The selector loop has
# no IOCP, so close() cannot hang by construction.
if sys.platform == "win32":
    import asyncio as _asyncio_policy
    _asyncio_policy.set_event_loop_policy(
        _asyncio_policy.WindowsSelectorEventLoopPolicy())
from datetime import datetime

from config import ABLATE

# 环境适配：代理阻断 nltk_data 网络下载 → monkeypatch 为本地 jieba 分词 + 内置停用词
import os
_NLTK_DATA = os.path.expanduser(os.getenv("NLTK_DATA", "~/nltk_data"))
os.environ.setdefault("NLTK_DATA", _NLTK_DATA)
# 2026-08-08：禁用 ragas 用量埋点——ragas 0.4.3 在 metric 协程内同步 requests.post
# 上报埋点，DNS 解析卡死会阻塞事件循环（wait_for 超时永不触发，评分整体挂起）。
os.environ.setdefault("RAGAS_DO_NOT_TRACK", "true")
import nltk  # noqa: E402
nltk.data.path.insert(0, _NLTK_DATA)
try:
    import nltk.pathsec
    nltk.pathsec.validate_path = lambda *args, **kwargs: None
except Exception:
    pass
import jieba

def _sent_tokenize_local(text):
    """中英文混合分句 — 按标点 + 换行切分。"""
    parts = re.split(r"(?<=[。！？!?；;.\n])", str(text))
    return [p.strip() for p in parts if p.strip()]

def _word_tokenize_local(text):
    """jieba 中文分词（零网络依赖），空格分词作为英文回退。"""
    try:
        return jieba.lcut(str(text))
    except Exception:
        return str(text).split()

nltk.sent_tokenize = _sent_tokenize_local
nltk.word_tokenize = _word_tokenize_local

# 内置最小英文停用词集 — 绕开 nltk_data 网络下载
_MIN_STOP = [
    "a","an","the","and","or","but","is","are","was","were","be","been","being",
    "to","of","in","on","at","for","with","as","by","that","this","it","from","not"
]

# Monkeypatch nltk.data.load — 拦截 stopwords/punkt 请求返回伪造对象，其余透传
try:
    _orig_load = nltk.data.load
    def _safe_load(resource_url, *args, **kwargs):
        from types import SimpleNamespace
        url_str = str(resource_url)
        if "stopwords" in url_str:
            fake = SimpleNamespace()
            fake.words = lambda lang="english": _MIN_STOP
            return fake
        if "punkt" in url_str:
            fake = SimpleNamespace()
            fake.tokenize = _sent_tokenize_local
            return fake
        return _orig_load(resource_url, *args, **kwargs)
    nltk.data.load = _safe_load
    # Patch 更早两层的 nltk 网络请求入口
    nltk.download = lambda *args, **kwargs: True
    _orig_find = nltk.data.find
    def _safe_find(resource_name, paths=None):
        res = str(resource_name)
        if any(k in res for k in ("punkt", "stopwords", "tokenizers", "corpora")):
            return resource_name  # 返回假路径，_safe_load 会处理
        return _orig_find(resource_name, paths)
    nltk.data.find = _safe_find
except Exception:
    pass

logger = logging.getLogger(__name__)

# 过渡期抑制 ragas 旧导入路径弃用警告
warnings.filterwarnings("ignore", category=DeprecationWarning, module="ragas")


# ── context_precision 判定级并行（安全并发，方案 B）──────────────────────
# ragas 0.4.3 的 LLMContextPrecisionWithReference._ascore 对每题最多 8 个
# context 判定是串行 await（同一时刻在途请求数 = batch_size），deepseek 代理
# 吞吐未打满 → 补跑 18 题耗时 44min。这里 monkeypatch 为 asyncio.gather 并行
# + Semaphore 限流（默认 2，总在途 = batch_size × 2，冒烟验证代理 12 并发无失败）。
# 防御：版本/符号检查 + 失败回退 ragas 串行原逻辑 + 幂等（已 patch 则跳过）。
# 默认 4（原 2）：代理冒烟验证 12 并发无失败，context_precision 判定级并行
# 提到 12，12 个 context 一轮全并行，消除串行轮次；env RAGAS_CP_CONCURRENCY 仍可覆盖。
_CP_SEMAPHORE = max(1, int(os.getenv("RAGAS_CP_CONCURRENCY", "12")))

# Windows ProactorEventLoop.close() hang fix (v9):
# ThreadPoolExecutor two-way parallel runs ragas_evaluate in worker threads,
# each via asyncio.run(). On loop.close(), if httpx keepalive connections to
# the proxy remain registered on the IOCP, ProactorEventLoop.close() blocks
# forever in windows_events._poll -> deadlock (observed: 80min stall).
# Fix: abort leftover transports before closing the loop.
if sys.platform == "win32":
    import asyncio as _asyncio
    _orig_proactor_close = _asyncio.ProactorEventLoop.close

    def _safe_proactor_close(self):
        # layer 1: abort transports and pump callbacks so sock.close() runs
        try:
            for tr in list(getattr(self, "_transports", {})):
                if hasattr(tr, "abort"):
                    tr.abort()
            try:
                self.run_until_complete(_asyncio.sleep(0))
            except Exception:
                pass
        except Exception:
            pass
        # layer 2: force-close sockets/pipes still registered on the IOCP,
        # so their pending overlapped ops get cancelled and the
        # `while self._cache` loop in IocpProactor.close() can exit
        try:
            proactor = getattr(self, "_proactor", None)
            cache = getattr(proactor, "_cache", None)
            for entry in list(cache.items()):
                obj = entry[1][2] if (isinstance(entry[1], (tuple, list))
                                      and len(entry[1]) > 2) else None
                if obj is not None and hasattr(obj, "close"):
                    try:
                        obj.close()
                    except Exception:
                        pass
        except Exception:
            pass
        _orig_proactor_close(self)

    _asyncio.ProactorEventLoop.close = _safe_proactor_close



# ── CP 逐条判词截获（2026-09-25，零新增调用）──────────────────────────────
# ragas 的 context_precision 每题只回一个 rank-weighted 标量，而**逐 context 的
# verdict 本来就在 _ascore_parallel 里算出来了、算完即丢**。W1 的异族判官复核卡在
# S2 层 n=12 个题级点（MDE≈0.40 远大于待检效应 0.10）；把检验粒度抬到"题×context"
# 只需要留下已在算的中间量 ⇒ 不改判分口径、不多花一次调用。
# 交付一律走 sidecar：`per_question` 的形状不许动（同出题审计件的先例 ——
# 改产物格式等于要求所有已注册闸门重新签认）。
_CP_CAPTURE = os.getenv("RAGAS_CAPTURE_CP_ITEMS", "1") != "0"
_CP_ITEM_SINK = {}
_CP_LOCK = threading.Lock()


def _ctx_fp(chunk) -> str:
    return hashlib.sha256(str(chunk).encode("utf-8")).hexdigest()[:12]


def _cp_key(question, contexts, reference) -> str:
    """由**判官真正看到的那三元组**导出的连接键。

    必须用截断后的 context 算（送进 ragas 的就是截断版），否则事后按题集原文
    重算会连不上 —— 这正是最容易悄悄失配的地方。
    """
    payload = json.dumps({"q": str(question),
                          "c": [_ctx_fp(x) for x in contexts],
                          "r": str(reference)},
                         ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _cp_record(question, contexts, reference, answers, score):
    """纯副作用：把逐 context 判词存进 sink。同键重复调用记 n_calls；
    同键给出不同判词 ⇒ 显式累加 conflict，**绝不静默后写覆盖**
    （判官非确定性正是这份数据要量的东西之一）。"""
    verdicts = [1 if getattr(a, "verdict", False) else 0 for a in answers]
    reasons = [str(getattr(a, "reason", "") or "")[:200] for a in answers]
    key = _cp_key(question, contexts, reference)
    with _CP_LOCK:
        old = _CP_ITEM_SINK.get(key)
        if old is None:
            _CP_ITEM_SINK[key] = {
                "key": key, "question": str(question), "n_ctx": len(verdicts),
                "ctx_fps": [_ctx_fp(x) for x in contexts],
                "verdicts": verdicts, "reasons": reasons, "cp": float(score),
                "n_calls": 1, "conflict": 0}
        else:
            old["n_calls"] += 1
            if old["verdicts"] != verdicts:
                old["conflict"] += 1


def cp_capture_report() -> dict:
    """三态报告：捕获开/关、patch 是否生效、条数与冲突数。

    "没捕获" 与 "捕获到零条判词" 必须可区分 —— 前者不许被读成后者。
    """
    with _CP_LOCK:
        items = list(_CP_ITEM_SINK.values())
    return {"capture_on": _CP_CAPTURE, "n_items": len(items),
            "n_repeat_calls": sum(i["n_calls"] - 1 for i in items),
            "n_conflict": sum(i["conflict"] for i in items)}


def write_cp_sidecar(path, records, extra=None):
    """把 sink 里的逐条判词按题号连好，落 sidecar。返回状态字符串。

    'skipped'   开关关闭 ⇒ 不写文件（不得冒充"零判词"）
    'no-capture' 捕获开着但一条没截到（patch 未生效 / 本轮没跑 CP）⇒ 不写文件并告警
    'written'   正常落盘
    """
    if not _CP_CAPTURE:
        return "skipped"
    with _CP_LOCK:
        items = [dict(v) for v in _CP_ITEM_SINK.values()]
    if not items:
        return "no-capture"
    by_key = {i["key"]: i for i in items}
    rows, missing = [], 0
    q_to_ids = {}
    for r in records:
        ctxs = _truncate_contexts(r.get("contexts", [""]))
        key = _cp_key(r.get("question", ""), ctxs, r.get("ground_truth", ""))
        hit = by_key.get(key)
        if hit is None:
            missing += 1
            continue
        q_to_ids.setdefault(str(r.get("question", "")), set()).add(str(r.get("id")))
        row = dict(hit)
        row.pop("question", None)
        row["id"] = r.get("id")
        rows.append(row)
    # 同一题面被多个 id 使用 ⇒ **只作诊断**（W1 实测 229 题剥完标点只剩 172 种题面）。
    # 连接键含 contexts 与 reference，题面重复不会造成误连，故这里不判错、只报数。
    dup_q = sum(1 for ids in q_to_ids.values() if len(ids) > 1)
    payload = {"_note": "CP 逐 context 判词（ragas 已在算的中间量，零新增调用）；"
                        "sidecar 不动 per_question 形状",
               "capture": cp_capture_report(), "dup_question_texts": dup_q,
               "n_records": len(records), "n_joined": len(rows), "n_unjoined": missing,
               "items": rows}
    if extra:
        payload.update(extra)
    _atomic_json_dump(path, payload)
    return "written"


def _patch_context_precision_parallel() -> bool:
    """将 context_precision 的逐 context 判定改为并行。失败时静默回退原逻辑。"""
    try:
        import asyncio
        from ragas.metrics._context_precision import (
            LLMContextPrecisionWithReference,
            QAC,
            Verification,
            ensembler,
        )
    except Exception as e:
        logger.warning("context_precision 并行 patch 导入失败，回退串行原逻辑: %s", e)
        return False

    orig = LLMContextPrecisionWithReference._ascore
    if getattr(orig, "_cp_parallel", False):
        return True  # 已 patch 过（幂等）

    async def _ascore_parallel(self, row, callbacks):
        user_input, retrieved_contexts, reference = self._get_row_attributes(row)
        sem = asyncio.Semaphore(_CP_SEMAPHORE)

        async def _judge_one(ctx):
            async with sem:
                verdicts = await self.context_precision_prompt.generate_multiple(
                    data=QAC(question=user_input, context=ctx, answer=reference),
                    llm=self.llm,
                    callbacks=callbacks,
                )
                return [v.model_dump() for v in verdicts]

        # 逐 context 判定并行（限流 sem），结果顺序与 retrieved_contexts 一致
        responses = await asyncio.gather(*[_judge_one(c) for c in retrieved_contexts])
        answers = []
        for resp in responses:
            agg = ensembler.from_discrete([resp], "verdict")
            answers.append(Verification(**agg[0]))
        score = self._calculate_average_precision(answers)
        if _CP_CAPTURE:
            # 纯副作用；开关关掉时返回值必须与不接这段代码逐位相同（自测 A 组钉）
            _cp_record(user_input, retrieved_contexts, reference, answers, score)
        return score

    _ascore_parallel._cp_parallel = True
    LLMContextPrecisionWithReference._ascore = _ascore_parallel
    logger.info("context_precision 判定级并行已启用 (Semaphore=%d)", _CP_SEMAPHORE)
    return True


# ── pydantic Ollama 适配：ragas 传 QueryBundle 时转 string ──
class EvalIntegrityError(RuntimeError):
    """评测完整性致命错误（2026-09-13 学术审查新增）：必须中止整轮。

    触发场景：声明 OPT_C=1 但滑窗检索器不可用。此前该情形会静默退化为 base
    检索（R1/R2 事故）；即使抛出，逐题重试包装也会把它降级成"单题失败"，
    整轮仍跑完并写出空结果 meta。故用专用异常类型穿透重试层。
    """


class _SafeOllamaEmbed:
    """包装 langchain_ollama.OllamaEmbeddings，将 QueryBundle/LlamaIndex 对象转为 str。"""
    def __init__(self, base_embed):
        self._base = base_embed
    def embed_documents(self, texts):
        return self._base.embed_documents([str(t) for t in texts])
    def embed_query(self, text):
        return self._base.embed_query(str(text))


class RAGASEvaluator:
    """对比 Vector RAG 与 KG-RAG 的 RAGAS 评估器。

    使用本地 Ollama 作为 judge LLM，支持单系统/双系统评估、
    按查询类型分层分析、以及 Markdown 报告生成。
    """

    _COT_PROMPT_TEMPLATE = (
        "你是一个工业故障诊断专家。请根据以下检索到的知识，逐步推理回答问题。\n\n"
        "【故障诊断知识】\n{context}\n\n"
        "【问题分类】\n"
        "先判断问题类型（仅内心判断，不输出）：\n"
        "  A. 故障诊断类：问'原因有哪些'/'怎么处理/排查/复位/解决'/'故障现象是什么'等 → 输出三段式\n"
        "  B. 参数/含义查询类：问'参数X功能是什么'/'报警Y含义/意思是什么'/'地址Z是什么'等 → 直接给出数值/定义，不输出三段式\n\n"
        "【推理步骤】\n"
        "1. 识别问题类型（A 或 B）\n"
        "2. 若 A：识别故障现象 → 追溯可能原因 → 确定解决方法\n"
        "   若 B：直接定位目标参数/报警号的定义或数值\n"
        "3. 综合回答\n\n"
        "【问题】{question}\n\n"
        "请严格按以下两部分输出（必须分开）：\n"
        "【推理过程】先写出你的完整推理思路，说明你依据了知识中的哪些条目、如何一步步得出结论。\n"
        "【最终答案】再输出最终答案，格式要求：\n"
        "  - 若判断为 A（故障诊断类）：【故障现象】...\n【可能原因】...\n【解决方法】...\n"
        "  - 若判断为 B（参数/含义查询类）：直接给出数值/定义/含义，格式为：【答案】...\n"
        "【最终答案硬性约束】\n"
        "- 只输出【故障诊断知识】中明确出现的结论；如需标注依据，统一用括号内联格式紧跟结论，\n"
        "  如「结论（依据：条目关键词）」；不要输出「依据：…」独立尾缀，也不要混合两种格式。（2026-10-24 格式统一）\n"
        "- 知识中没有的内容一律不写，禁止补充外部经验、额外排查步骤或推测性诊断。\n"
        "- 报警号/地址/参数号/端子号类问题（地址是什么、含义是什么、设定为什么值等）直接给出数值或代号和一句依据，不要强行套故障三段式。\n"
        "- 若问题指名了具体报警号/故障代码（如报警405、ALM414），只能引用与【该号本身】相关的知识条目（含其含义/原因/处理方法，或明确写'参见'的指引）。\n"
        "- 严禁把其它报警号/故障代码/其它故障的原因或处理方法当作所问号的内容引用——不同报警号对应不同故障，即使主题相近（如同一轴系的400系、不同报错码）也不得互相套用。\n"
        "**即使条件/现象描述相似（如 090/091 条件 A/B），也必须只引用所问报警号自身的知识条目。**\n"
        "- 报警号/故障代码 **含义** 类问题（问'是什么含义/是什么意思/是什么故障'）：\n"
        "  直接给出该报警号的 **含义原文**（含现象/触发条件），**不要**输出可能原因/解决方法，\n"
        "  除非问题显式问'原因'或'处理'。\n"
        "- 【P0-2 结构化多跳约束】当问题问原因/处理方法/复位/排查（A类），且知识中同一对象存在多条并列原因或有序处理步骤时：\n"
        "  (a) 并列性：同一层级的并列项（多条原因 / 多条并列措施）**必须全部逐条列出**，编号 ①②③ 或 1.2.3.，\n"
        "      禁止只列第一/前几条、禁止合并成一句概括而丢失条目。\n"
        "  (b) 顺序性：处理步骤若存在先后（先…再…最后…/①②③顺序），**必须按知识中出现的先后顺序输出**，\n"
        "      不得打乱、不得把收尾步骤（如 '修正设定值并返回到参考点'、'重新设定存储行程极限'）前置或省略。\n"
        "  (c) 条件分支：含'若/如果/当…则/如不能'的条件性步骤，**必须完整保留其'若…则…'结构**，\n"
        "      输出为'若【条件】，则【动作】'，禁止把条件剥掉只留动作（如禁止把'若不能移动，按急停或G23'写成'按急停'）。\n"
        "  (d) 因果配对：若知识分别列出'原因'与'处理方法'且一一对应（如 原因1→方法1），输出时尽量配对呈现\n"
        "      （'原因X（处理：方法X）'），配对信息不足时可分列但不得张冠李戴。\n"
        "- 若问题涉及多个并列原因/步骤/措施，**必须全部列出**，不得只给第一条。\n"
        "- 每条并列项后均需标注依据，用分号或换行分隔。\n"
        "- 对 '原因有哪些'/'处理方法有哪些' 类问题，必须在推理中先统计知识中有几条并列项，再逐条写入答案。\n"
        "- 涉及分步骤/分阶段处理的问题（如 '如何复位/处理/排查'），必须按知识中的先后顺序列出 **所有步骤**，\n"
        "  不得省略后续步骤（如 '设定值有误则修正并返参考点' 这类收尾步骤常被漏写）。\n"
        "- 最终答案控制在 200 字以内，宁可少写，不可编造。\n"
    )

    # A 案（2026-10-28）：生成侧"先定案、后作答"。开关 OPT_CASE_ALIGN 默认 0
    # ⇒ 模板逐字节不变；=1 时在模板尾部追加定案约束（口语化提问没有"例N"锚点，
    # 上下文常同时装载多个相近案例，模型会选错案或答成通用可能原因清单）。
    _CASE_ALIGN_PROMPT_SUFFIX = (
        "\n【定案约束】\n"
        "- 推理开始时，先从【故障诊断知识】里挑出与问题最相符的**那一个**案例：\n"
        "  逐项核对其故障现象、系统/驱动器型号、元件代号、报警号是否与问题一致，\n"
        "  并在【推理过程】的最开头写明「依据案例：例N」和一句核对理由。\n"
        "- 知识里若有多个现象相近的案例，必须先逐个对比差异再选定，\n"
        "  禁止把相近案例的结论互相套用；选不出来时，答案必须写明"
        "「知识中无法唯一确定对应案例」，不得取近似案例作答。\n"
        "- 选定案例后，最终答案只取该案例自己的分析/处理/维修体会内容，\n"
        "  其余案例的内容一律不写进答案。\n"
    )

    @classmethod
    def _answer_template(cls) -> str:
        t = cls._COT_PROMPT_TEMPLATE
        if os.getenv("OPT_CASE_ALIGN", "0") == "1":
            t = t + cls._CASE_ALIGN_PROMPT_SUFFIX
        return t

    def __init__(
        self,
        judge_llm,
        embed_model,
        judge_model_name: str = "qwen2.5:3b",
        answer_generator: Optional[Callable[[str], str]] = None,
        api_client=None,
        use_cot: bool = True,
        answer_parallel: int = 1,
        eval_parallel: int = 2,
        eval_timeout: int = 600,
        ooc_domain: str = "工业离心泵故障维保手册",
    ):
        """
        Args:
            judge_llm: LangChain 兼容的 judge LLM（推荐 ChatOllama，
                       不要传 ragas llm_factory 产物，兼容性差）
            embed_model: LangChain 兼容的 embedding 模型（推荐
                         OllamaEmbeddings，不要传 llama_index BaseEmbedding）
            judge_model_name: 日志/报告中显示的模型名
            answer_generator: 可选，用于 KG-RAG 答案生成的可调用对象
                              (prompt: str) -> str。若未提供，回退到 judge_llm
            api_client: 可选，backend.api_client.ApiClient 实例，作为
                        answer_generator 的备选方案（通过服务层调用 LLM）
            use_cot: 是否启用 Chain-of-Thought 推理（默认 True）。
                     启用后，KG-RAG 答案生成采用"检索→推理→回答"模式，
                     并保存 CoT 推理过程到评估结果中。
            answer_parallel: 答案生成阶段的题目级并行度（线程池 max_workers，
                             默认 1 = 串行）。
            eval_parallel: RAGAS 评分并发度（evaluate 的 batch_size，默认 2）。
            eval_timeout: RAGAS 单题指标判定的超时上限（秒，默认 600）。
                         ragas 默认 RunConfig.timeout=180s——Graph 的 context_precision
                         每题 8 次判定串行可达 200s+，会被误判超时导致 NaN（修复）。
        """
        self.judge_llm = judge_llm
        self.ragas_embed = _SafeOllamaEmbed(embed_model)
        self.judge_model_name = judge_model_name
        self.answer_generator = answer_generator
        self.api_client = api_client
        self.use_cot = use_cot
        self.answer_parallel = max(1, int(answer_parallel or 1))
        self.eval_parallel = max(1, int(eval_parallel or 2))
        self.eval_timeout = max(1, int(eval_timeout or 600))
        # A1 修复：OOC 拒答 prompt 的领域描述参数化（原硬编码"工业离心泵"，
        # shukongdashi 等非泵数据集会收到错误领域前提）。构造时注入领域描述，
        # 缺省保持 pump 旧口径（兼容既有调用/测试）。
        self.ooc_domain = ooc_domain or "工业离心泵故障维保手册"

    def evaluate_pair(
        self,
        test_items: list,
        vector_engine,
        graph_engines: Optional[dict] = None,
        output_dir: str = ".sisyphus/evidence",
    ) -> dict:
        """运行双系统评估，逐题收集答案后计算 RAGAS 指标。

        Args:
            test_items: 测试项列表，每项必含 "question" / "ground_truth"，
                        可选 "id" / "type"
            vector_engine: 实现 query(q) -> dict(answer, contexts, scores)
            graph_engines: 图查询引擎字典 {"manual": ..., "community": ...}，
                           为 None 则仅评估向量 RAG
            output_dir: 中间结果保存目录

        Returns:
            {
                "vector_rag": dict,    # 指标 + per_question + n_questions
                "graph_rag": dict|None,
                "comparison": {"delta": dict, "winner_per_metric": dict}
            }
        """
        results = {}

        logger.info("Evaluating Vector RAG on %d items...", len(test_items))
        results["vector_rag"] = self._evaluate_system(
            test_items, vector_engine, system_label="vector_rag", output_dir=output_dir
        )

        if graph_engines:
            logger.info("Evaluating KG-RAG on %d items...", len(test_items))
            results["graph_rag"] = self._evaluate_system(
                test_items, graph_engines, system_label="graph_rag", output_dir=output_dir
            )
        else:
            results["graph_rag"] = None

        results["comparison"] = self._compare_results(results)
        return results

    def compare(self, test_items, vector_engine, graph_engines=None) -> dict:
        """evaluate_pair 的别名，语义等价。"""
        return self.evaluate_pair(test_items, vector_engine, graph_engines=graph_engines)

    def generate_report(self, results: dict, output_path: str = "reports/eval_report.md") -> str:
        """渲染 Markdown 评估报告并写入文件。

        包含总体指标表、按类型分层表、综合结论与系统信息。
        """
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        vec = results.get("vector_rag") or {}
        kg = results.get("graph_rag") or {}
        n_total = max(vec.get("n_questions", 0), kg.get("n_questions", 0))

        _LABELS = {
            "faithfulness": "Faithfulness",
            "answer_relevancy": "Answer Relevancy",
            "context_precision": "Context Precision",
            "context_recall": "Context Recall",
        }
        _KEYS = list(_LABELS)

        type_breakdown = self._build_type_breakdown(vec, kg)

        lines = [
            f"# KGRAG 评估报告\n",
            f"生成时间: {now}",
            f"法官模型: {self.judge_model_name}",
            f"测试集规模: {n_total} 题\n",
            "## 总体指标\n",
        ]

        if kg:
            lines.append("| 指标 | Vector RAG | KG-RAG | 差值 (KG-Vec) | 胜者 |")
            lines.append("|------|-----------|--------|---------------|------|")
        else:
            lines.append("| 指标 | Vector RAG |")
            lines.append("|------|-----------|")

        kg_wins = 0
        vec_wins = 0
        for key in _KEYS:
            v_val = vec.get(key)
            k_val = kg.get(key) if kg else None
            label = _LABELS[key]
            if v_val is not None and k_val is not None:
                delta = k_val - v_val
                delta_s = f"{delta:+.2f}"
                winner = "KG-RAG" if delta > 0 else ("Vector RAG" if delta < 0 else "平局")
                if delta > 0:
                    kg_wins += 1
                elif delta < 0:
                    vec_wins += 1
                lines.append(f"| {label} | {v_val:.2f} | {k_val:.2f} | {delta_s} | {winner} |")
            else:
                v_s = f"{v_val:.2f}" if isinstance(v_val, (int, float)) else "N/A"
                lines.append(f"| {label} | {v_s} |")
        lines.append("")

        # —— 评测口径拆分：剔除 out-of-context 题后的对比（双口径）——
        # 修复 9.2-A2：RAGAS 四指标对 OOC 题无"拒答"口径——Vector 空 context 答
        # "未检索到"→低分，Graph 被保底召回塞入无关 context 答"无法回答"→虚高，
        # 3 题会系统性偏向一侧。主表（全部题）之外补充剔除 OOC 后的视图，
        # 并逐指标标注有效样本数（faithfulness 存在 NaN 缺分，样本可能不足）。
        vec_no = self._agg_metrics_excluding(vec)
        kg_no = self._agg_metrics_excluding(kg) if kg else None
        kg_wins_no = 0
        vec_wins_no = 0
        if any(v is not None for v in vec_no.values()):
            lines.append("## 剔除 out-of-context 题后（双口径）\n")
            lines.append(f"> 为避免 3 道 out-of-context 题的指标口径不一致（拒答 vs 塞无关 "
                         f"context）扭曲总体对比，下表剔除 type=out-of-context 的题目重算；"
                         f"括号内为该指标有效样本数（部分题因 judge 输出格式问题缺分）。\n")
            if kg:
                lines.append("| 指标 | Vector RAG | KG-RAG | 差值 (KG-Vec) | 胜者 |")
                lines.append("|------|-----------|--------|---------------|------|")
            else:
                lines.append("| 指标 | Vector RAG |")
                lines.append("|------|-----------|")
            for key in _KEYS:
                label = _LABELS[key]
                v_val, v_n = vec_no.get(key), vec_no.get(key + "_n")
                k_val, k_n = (kg_no.get(key), kg_no.get(key + "_n")) if kg_no else (None, None)
                if v_val is not None and k_val is not None:
                    delta = k_val - v_val
                    delta_s = f"{delta:+.2f}"
                    winner = "KG-RAG" if delta > 0 else ("Vector RAG" if delta < 0 else "平局")
                    if delta > 0:
                        kg_wins_no += 1
                    elif delta < 0:
                        vec_wins_no += 1
                    lines.append(f"| {label} | {v_val:.2f}({v_n}) | {k_val:.2f}({k_n}) | "
                                 f"{delta_s} | {winner} |")
                elif v_val is not None:
                    lines.append(f"| {label} | {v_val:.2f}({v_n}) | N/A | | |")
                else:
                    lines.append(f"| {label} | N/A | N/A | | |")
            lines.append("")

        # —— 9.10：OOC 题拒答正确性（消除"OOC 四指标口径不对称"的实质妥协）——
        # RAGAS 四指标无法衡量拒答：空 context 的正确拒答被判 faithfulness=0.0，
        # 塞入无关 context 后"答非所问"反而可能得分。OOC 题改用对称的拒答
        # 正确性指标（1=明确拒答且未编造，0=编造/答非所问），与四指标口径正交。
        ooc_vec = [r for r in vec.get("per_question", [])
                   if r.get("type") == "out-of-context"
                   and (r.get("metrics") or {}).get("ooc_refusal") is not None]
        ooc_kg = [r for r in (kg or {}).get("per_question", [])
                  if r.get("type") == "out-of-context"
                  and (r.get("metrics") or {}).get("ooc_refusal") is not None]
        if ooc_vec or ooc_kg:
            lines.append("## Out-of-Context 题拒答正确性\n")
            lines.append("> 正确答案不在知识库中的题，RAGAS 四指标口径不对称（空 context 的"
                         "正确拒答得 0 分、塞无关 context 后答非所问反而可能得分）。此处改用"
                         "对称指标：**1**=正确拒答（明确表示无法回答且未编造），**0**=编造/"
                         "答非所问。与四指标口径正交，不参与上方总体对比。\n")
            ooc_ids = sorted({r["id"] for r in ooc_vec + ooc_kg})
            if kg:
                lines.append("| 题目 | Vector RAG | KG-RAG |")
                lines.append("|------|-----------|--------|")

                def _refusal(recs, qid):
                    for r in recs:
                        if r.get("id") == qid:
                            v = (r.get("metrics") or {}).get("ooc_refusal")
                            return f"{v:.1f}" if v is not None else "N/A"
                    return "N/A"

                for qid in ooc_ids:
                    lines.append(f"| {qid} | {_refusal(ooc_vec, qid)} | "
                                 f"{_refusal(ooc_kg, qid)} |")
                v_ok = sum(1 for r in ooc_vec
                           if (r.get("metrics") or {}).get("ooc_refusal") == 1.0)
                k_ok = sum(1 for r in ooc_kg
                           if (r.get("metrics") or {}).get("ooc_refusal") == 1.0)
                lines.append("")
                lines.append(f"- Vector RAG 正确拒答 {v_ok}/{len(ooc_vec)} 题；"
                             f"KG-RAG 正确拒答 {k_ok}/{len(ooc_kg)} 题。")
            else:
                for r in ooc_vec:
                    v = (r.get("metrics") or {}).get("ooc_refusal")
                    lines.append(f"- {r.get('id')}: "
                                 f"{f'{v:.1f}' if v is not None else 'N/A'}")
            lines.append("")

        # —— 数值型题目精确命中率（与 RAGAS 语义判分正交）——
        # 对 ground_truth 为数值/范围/比较符的题，正则提取答案数值做机器判分。
        # 样本数 = 该侧数值型题中可判分的题数；无法解析的 gt/answer 不计入。
        vec_num = _agg_numeric_accuracy(vec)
        kg_num = _agg_numeric_accuracy(kg) if kg else None
        if vec_num is not None or kg_num is not None:
            lines.append("## 数值型题目精确命中率\n")
            lines.append("> 对 ground_truth 为数值/范围/比较符的题，正则提取答案数值做"
                         "机器判分，与 RAGAS 语义判分正交。样本数 = 该侧数值型题中"
                         "可判分的题数。\n")
            if kg:
                lines.append("| 系统 | 命中率 | 判分样本 |")
                lines.append("|------|--------|----------|")
                lines.append(f"| Vector RAG | {_fmt_num(vec_num)} | "
                             f"{_num_samples(vec)} |")
                lines.append(f"| KG-RAG | {_fmt_num(kg_num)} | "
                             f"{_num_samples(kg)} |")
            else:
                lines.append(f"- Vector RAG 数值命中率: {_fmt_num(vec_num)} "
                             f"（样本 {_num_samples(vec)}）")
            lines.append("")

        if type_breakdown:
            lines.append("## 按查询类型分层\n")
            for qtype, data in type_breakdown.items():
                lines.append(f"### {qtype} ({data.get('n', 0)} 题)\n")
                if kg:
                    lines.append("| 指标 | Vector RAG | KG-RAG | 差值 | 胜者 |")
                    lines.append("|------|-----------|--------|------|------|")
                else:
                    lines.append("| 指标 | Vector RAG |")
                    lines.append("|------|-----------|")
                for key in _KEYS:
                    v_val = data.get("vec", {}).get(key)
                    k_val = data.get("kg", {}).get(key) if kg else None
                    label = _LABELS[key]
                    if v_val is not None and k_val is not None:
                        delta = k_val - v_val
                        delta_s = f"{delta:+.2f}"
                        winner = "KG-RAG" if delta > 0 else ("Vector RAG" if delta < 0 else "平局")
                        lines.append(f"| {label} | {v_val:.2f} | {k_val:.2f} | {delta_s} | {winner} |")
                    else:
                        v_s = f"{v_val:.2f}" if isinstance(v_val, (int, float)) else "N/A"
                        lines.append(f"| {label} | {v_s} |")
                lines.append("")

        # —— 9.13-②：KG 侧 context 构成归因（图谱路径 vs 社区原文）——
        # 透明呈现"得分机制"：此前 auto_008 的 context_precision≈1.0 实为社区
        # 长文托底、图谱路径与答案零相关——"KG 领先"可能来自图谱引导的社区
        # 检索而非路径推理。逐题列出路径/原文占比，供结论归因。
        if kg:
            attr = self._context_attribution(kg)
            if attr:
                lines.append("## KG-RAG context 构成归因（图谱路径 vs 社区原文）\n")
                lines.append("> 路径 = 图谱三元组展开（`实体--[关系]-->实体`）；原文 = 社区/手册片段。"
                             "路径占比高的题，其得分机制更依赖图谱推理；原文占比高的题，"
                             "得分可能由社区检索（类似向量召回）托底。\n")
                lines.append("| 题目 | 类型 | 图谱路径数 | 社区原文数 | 总context |")
                lines.append("|------|------|-----------|-----------|----------|")
                for a in attr:
                    lines.append(f"| {a['id']} | {a['type']} | {a['n_path']} | {a['n_text']} | {a['n_ctx']} |")
                zero_path = [a["id"] for a in attr if a["n_path"] == 0 and a["n_ctx"] > 0]
                if zero_path:
                    lines.append("")
                    lines.append(f"> 提示：{len(zero_path)} 题（{', '.join(zero_path)}）的评分 context "
                                 "不含任何图谱路径，其分数完全由社区/原文托底，不作为图谱能力证据。")
                lines.append("")

        lines.append("## 结论\n")
        if kg:
            total = len(_KEYS)
            # 9.10：主表为全量口径（含 OOC 题），但 OOC 题的四指标对"拒答"无
            # 区分度——正确拒答得 0 分、答非所问可能得分。正式对比以"剔除 OOC
            # 后（双口径）"与"OOC 拒答正确性"两节为准。
            if ooc_vec or ooc_kg:
                lines.append("> 说明：主表含 out-of-context 题，其四指标口径不对称，"
                             "不作为对比依据。正式对比见「剔除 out-of-context 题后」"
                             "与「Out-of-Context 题拒答正确性」两节。\n")
            if kg_wins_no or vec_wins_no:
                # 双口径有有效样本 → 以对称口径下结论
                if kg_wins_no > vec_wins_no:
                    lines.append(f"KG-RAG 在剔除 out-of-context 后的 {kg_wins_no} 项指标上"
                                 f"优于 Vector RAG（共 {total} 项）。")
                elif vec_wins_no > kg_wins_no:
                    lines.append(f"Vector RAG 在剔除 out-of-context 后的 {vec_wins_no} 项"
                                 f"指标上优于 KG-RAG（共 {total} 项）。")
                else:
                    lines.append("剔除 out-of-context 后两种方案在各项指标上各有胜负，"
                                 "表现接近。")
            elif kg_wins or vec_wins:
                # 双口径无有效样本 → 退回全量口径（已在说明行标注口径限制）
                if kg_wins > vec_wins:
                    lines.append(f"KG-RAG 在 {kg_wins}/{total} 项指标上优于 Vector RAG。")
                elif vec_wins > kg_wins:
                    lines.append(f"Vector RAG 在 {vec_wins}/{total} 项指标上优于 KG-RAG。")
                else:
                    lines.append("两种方案在各项指标上各有胜负，表现接近。")
            lines.append("")
            # 9.11-1（R11）：逐题区分度/胜场——同分率 > 2/3 的指标对两系统对比
            # 无区分力（均值差不得作领先证据）；均值差 ±0.1 内一律按"接近"表述。
            disc = self._per_question_discrimination(vec, kg)
            for key in _KEYS:
                v_val = vec_no.get(key) if vec_no.get(key) is not None else vec.get(key)
                k_val = kg_no.get(key) if kg_no.get(key) is not None else kg.get(key)
                if v_val is not None and k_val is not None:
                    delta = k_val - v_val
                    direction = "优于" if delta > 0 else ("劣于" if delta < 0 else "持平于")
                    note = ""
                    d = disc.get(key, {})
                    if d.get("total"):
                        note = f"（胜场 G{d.get('g_wins')}/V{d.get('v_wins')}/平{d.get('ties')}"
                        if d["same"] / d["total"] > 2 / 3:
                            note += "，区分度低：逐题同分率高，均值差不作领先证据"
                        elif abs(delta) <= 0.1:
                            note += "，差幅≤0.1，按接近表述"
                        note += "）"
                    lines.append(
                        f"- {_LABELS[key]}: KG-RAG {direction} Vector RAG ({delta:+.2f}){note}")
        else:
            lines.append("仅评估了 Vector RAG 系统，未进行对比。")
            for key in _KEYS:
                v_val = vec.get(key)
                if v_val is not None:
                    s = f"{v_val:.2f}" if isinstance(v_val, (int, float)) else str(v_val)
                    lines.append(f"- {_LABELS[key]}: {s}")
        # 双口径补充：剔除 out-of-context 后的胜场（诚实呈现，避免 3 题 OOC 扭曲结论）
        if kg and (kg_wins_no or vec_wins_no):
            lines.append("")
            lines.append(f"- 剔除 out-of-context 题后：KG-RAG 在 {kg_wins_no} 项、"
                         f"Vector RAG 在 {vec_wins_no} 项指标上占优（详见上方双口径表）")
        lines.append("")

        lines.append("## 数据来源声明\n")
        lines.append(self._render_source_declaration(results))

        lines.append("## 系统信息\n")
        # 修复 KG-9：不再硬编码 0.45，取 config 实际口径（A1 修复后阈值 0.5）
        from config import SIMILARITY_THRESHOLD, TOP_K
        lines.append(f"- Vector RAG: llama-index, top_k={TOP_K}, similarity_threshold={SIMILARITY_THRESHOLD}")
        if kg:
            lines.append("- KG-RAG: 手工构建知识图谱 + Neo4j + 路径展开")
        lines.append("- 评估环境: Python 3.10, RAGAS 0.4.3\n")

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        content = "\n".join(lines)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(content)
        abs_path = os.path.abspath(output_path)
        logger.info("Report generated: %s", abs_path)
        return abs_path

    def _render_source_declaration(self, results: dict) -> str:
        """渲染"数据来源声明"（修复 9.13-⑤：不再硬编码，优先 _meta 数据驱动）。

        调用方可在 results 注入 {"_meta": {...}}（如 gen_report_dual 从 meta.json
        读取 graph_source/judge/test_set 等）；缺省回退到内置 v4 说明文本，
        避免旧调用输出空节。
        """
        meta = results.get("_meta") or {}
        if meta:
            lines = [
                f"- 图谱版本：{meta.get('graph_source') or 'N/A'}。",
            ]
            if meta.get("judge_llm"):
                lines.append(f"- judge/答案生成：{meta['judge_llm']}（同模型，自评偏差见下）。")
            if meta.get("embedding"):
                lines.append(f"- embedding：{meta['embedding']}。")
            if meta.get("test_set"):
                lines.append(f"- 测试集：{meta['test_set']}（{meta.get('n_questions', '?')} 题）。")
            lines.append(
                "- 9.13 透明声明：① judge==生成器（自评偏差，绝对分不可跨模型比）；"
                "② 评分 context 数量两侧上限已对齐（V top_k=12 / G max_ctx=12）；"
                "③ 实体噪音规则已加短部件白名单；④ 测试集/图谱的 test-driven 补抽取与 "
                "context 归因，见上方归因节与 code_review_findings.md 9.13。")
            return "\n".join(lines) + "\n"
        return ("- 图谱为 v4 版本（llm_extract_v4_graph_pump_fault_diagnosis_real.json，"
                "317 条 = v3 153 条确定性清洗 + **164 条短部件补抽**）。\n"
                "- 9.13 透明声明：judge==生成器（自评偏差）；context 数量两侧对齐"
                "（V top_k=12 / G max_ctx=12）；短部件白名单已加；test-driven 补抽取与"
                "归因见 code_review_findings.md 9.13。\n")

    def _evaluate_system(self, test_items, engine, system_label, output_dir) -> dict:
        """对单个系统逐题查询并计算 RAGAS 指标。

        自动分流：若 engine 为 dict 则走 KG-RAG 路径，否则走向量 RAG 路径。
        """
        if isinstance(engine, dict):
            return self._evaluate_graph_rag(test_items, engine, output_dir)
        # 统一协议：单引擎（RagEngine）→ 走统一入口（新引擎零成本接入）
        if hasattr(engine, "query_structured"):
            return self._evaluate_structured_engine(
                test_items, engine, system_label, output_dir)

        def process(item):
            result = engine.query(item["question"])
            return result.get("answer", ""), result.get("contexts", [])

        return self._run_ragas_eval(test_items, process, system_label, output_dir)

    def _evaluate_structured_engine(self, test_items, engine, system_label, output_dir) -> dict:
        """评估实现了 RagEngine 协议的引擎（统一走 query_structured）。

        兼容 dict 与 RetrievalResult 两种返回形态：dict 时取 answer/contexts，
        RetrievalResult 时取字段；兜底到旧 query()（兼容无协议的历史引擎）。
        """
        def process(item):
            r = engine.query_structured(item["question"])
            if isinstance(r, dict):
                return r.get("answer", ""), r.get("contexts", [])
            return r.answer, r.contexts

        return self._run_ragas_eval(test_items, process, system_label, output_dir)

    def _evaluate_graph_rag(self, test_items, graph_engines, output_dir,
                            entity_chunk_map=None, chunk_texts=None) -> dict:
        """评估知识图谱 RAG：路径展开 → 文本化 → answer_generator 合成 → RAGAS 评分。

        ManualGraphQueryEngine.query() 返回 [(names, rels)]，
        格式化为 "实体A--[关系]-->实体B" 作为上下文。

        当 self.use_cot=True 时，使用 Chain-of-Thought 推理生成答案，
        并将推理过程存入 per_question 的 cot_reasoning 字段。
        """
        manual_engine = graph_engines.get("manual")
        community_engine = graph_engines.get("community")

        # —— OPT_C 预检（2026-09-13）：在花费任何 API 调用与时间之前先确认检索器可用 ——
        # 逐题守卫虽已存在，但其异常会被逐题重试降级为"单题失败"，整轮仍会跑完并
        # 写出空结果；此处提前失败，保证"声明 OPT_C=1 却无法生效"绝不产出半成品归档。
        if os.getenv("OPT_C", "0") == "1":
            if community_engine is None:
                raise EvalIntegrityError(
                    "[OPT_C 完整性] OPT_C=1 但社区引擎不可用（社区消融与 OPT_C 不兼容）。"
                    "请显式设置 OPT_C=0，或关闭社区消融开关。")
            if _get_sliding_retriever() is None:
                raise EvalIntegrityError(
                    "[OPT_C 完整性] OPT_C=1 但滑窗 dense 检索器构建失败。请检查 "
                    "Ollama（qwen3-embedding:0.6b）与 "
                    ".sisyphus/evidence/cache/sliding_embed_cache.json 后重试。")

        use_cot = self.use_cot

        def process(item):
            q = item.get("question", "")
            # OOC 拒答门只对 out-of-context 题生效：此前对全部题生效，
            # 导致 in-context 题（auto_007 等）被相关度启发式误拒答。
            is_ooc = str(item.get("type", "")) == "out-of-context"
            _dense_retriever = None
            _opt_c = os.getenv("OPT_C", "0") == "1"
            _opt_d = os.getenv("OPT_D", "0") == "1"
            if (_opt_c or _opt_d) and community_engine is not None:
                if _opt_c:
                    # 方案 C：滑窗 dense 检索器（363 chunk，对齐 Vector）
                    _dense_retriever = _get_sliding_retriever()
                else:
                    # 方案 D：社区索引 dense 召回
                    _mc = getattr(community_engine, "_match_chunks", None)
                    if callable(_mc):
                        _dense_retriever = lambda qs, _mc=_mc: _mc(qs)
            # —— 完整性硬断言（2026-09-13 学术审查新增）——
            # 事故背景：OPT_C=1 但检索器不可用时，下游 `if _opt_c and dense_retriever
            # is not None` 不成立 → **静默退化**为 base 的整 chunk 绑定，不报错、不告警，
            # 产出"看起来完全正常"的分数。round_optC_pure_v2_288c（R1，对外报告为
            # OPT_C pure）与 round_optC_closeA2_288c（R2）即因此失效——其 context 与
            # base 逐字节相同（288/288），依据块 2040 字而非 OPT_C 生效时的 988 字。
            # 此处改为立即失败，杜绝同类静默降级再次进入归档结果。
            if _opt_c and _dense_retriever is None:
                raise EvalIntegrityError(
                    "[OPT_C 完整性] OPT_C=1 但滑窗 dense 检索器不可用 "
                    f"(community_engine={'None' if community_engine is None else 'OK'})。"
                    "继续运行会静默退化为 base 检索，配置与声明不符。"
                    "请检查 Ollama 服务与 sliding_embed_cache.json，或显式设置 OPT_C=0。"
                )
            # 槽位预算参数化（2026-08-26）：KG_MAX_CTX 默认 12 保持历史
            # 口径；OPT_G 扩槽实验置 16。同步传入组装与截断两层。
            _kg_max_ctx = int(os.getenv("KG_MAX_CTX", "12"))
            raw_ctxs = _collect_graph_contexts(
                q, manual_engine, community_engine, is_ooc=is_ooc,
                entity_chunk_map=entity_chunk_map, chunk_texts=chunk_texts,
                dense_retriever=_dense_retriever, max_ctx=_kg_max_ctx)
            # 修复 KG-1：答案生成与 RAGAS 评分必须看到同一份 context。
            # 此前生成用 graph_contexts[:5]、评分用 _truncate_contexts（最多 8 个），
            # 评分多见的 context 未参与生成 → 两侧不对称（Vector 侧同样处理）。
            graph_contexts = _truncate_contexts(raw_ctxs, max_ctx=_kg_max_ctx)
            graph_contexts = _rerank_contexts(graph_contexts, q, corpus=chunk_texts)

            # 方案 F-addon（2026-10-14）：dense 通道只在既有槽位预算之外追加。
            # ⚠️ 必须在 `_truncate_contexts` 之后追加（同 TABLE_FALLBACK 的先例：
            # 在截断前追加会被硬截到 max_ctx，实测 0/288 生效）。前 12 槽的顺序
            # 与内容逐位不变（CP 不受扰动），追加槽以超出 max_ctx 的方式附加。
            if os.getenv("OPT_F_ADDON", "0") == "1" and _dense_retriever is not None and not is_ooc:
                try:
                    _addon_n = int(os.getenv("OPT_F_ADDON_N", "6"))
                    _have = {str(c) for c in graph_contexts}
                    _added = 0
                    for _sc, _row in _dense_retriever(q):
                        _t = (_row.get("text") or "").strip()
                        if len(_t) > 500:
                            _t = _t[:500]
                        if not _t or _t in _have:
                            continue
                        graph_contexts.append(_t)
                        _have.add(_t)
                        _added += 1
                        if _added >= _addon_n:
                            break
                except Exception:
                    logger.warning("[OPT_F_ADDON] 追加槽失败，按未追加处理", exc_info=True)

            # —— 表格/案例兜底召回（实验轮，opt-in，默认关闭）——
            # ⚠️ 必须在 `_truncate_contexts` **之后**追加。该函数会重排并硬截到
            # max_ctx（默认 12）块；若在此之前追加，兜底内容恰好位于末尾 →
            # 必被截掉。**上一轮实验实测 0/288 生效**，即为此坑。
            # 追加在截断之后的效果：原有 12 条的**顺序与内容逐位不变**
            # （CP 不受扰动），兜底条目以超出 max_ctx 的方式附加（CR 可得覆盖）。
            # 单条兜底窗口 ≤1600 字，远低于 max_chars=12000，不会被二次截断。
            if os.getenv("TABLE_FALLBACK", "0") == "1":
                try:
                    from backend.table_fallback import augment_contexts
                except Exception:
                    try:
                        from table_fallback import augment_contexts
                    except Exception:
                        augment_contexts = None
                if augment_contexts is not None:
                    graph_contexts = augment_contexts(graph_contexts, q)

            # 修复 2026-08-06（shukongdashi 优化）：OOC 拒答机制
            # 当有 context 但与问题的相关度低于阈值时，直接拒答而非强行生成答案。
            # 此前 10 道 OOC 题全部不拒答，强行编答案拉低 faithfulness。
            # 注意：只在有 context 时检查相关度，空 context 走原有占位符逻辑。
            # 2026-08-11：对象一致性检查——问题对象（机器人/磨床/电火花等）
            # 与上下文对象不符时视为 OOC（语义邻近但对象不符，防止用机床知识
            # 编造机器人答案，见 auto_010/040/050 修复）。
            if raw_ctxs and graph_contexts:
                if is_ooc:
                    # 2026-10-04 D5 分层补丁：门层打分走 _ooc_gate_relevance
                    #（与 serve.py 同一入口；_query_code_tokens 检索排序零接触）
                    relevance = _ooc_gate_relevance(q, graph_contexts)
                    refuse, ooc_mismatch = _should_refuse_ooc(q, graph_contexts, relevance)
                    # 2026-10-11 全侧激活（作者指令"所有侧都激活"）：批测 OOC 分支与
                    # serve 共用同一张句子尺票（同一钩子、同一组开关
                    # OOC_SENT_COVER_T／OOC_SENT_MODE）；票是加法（只增拒不撤拒），
                    # 库缺失/指纹不符/超时自动退回现门。未激活时钩子直接返回零开销。
                    _rr = None
                    try:
                        _rr, _ = sent_cov_gate_hook(
                            q, graph_contexts,
                            {"relevance": round(relevance, 3),
                             "object_mismatch": bool(ooc_mismatch)} if refuse else None,
                            embed_fn=_sentcov_embed_fn())
                    except Exception as _e:                      # noqa: BLE001
                        logger.warning("批测句子尺票异常，退回现门: %s", _e)
                    # 口语措辞放行票（2026-10-28）：撤相关度门单独开火的误拦；
                    # 默认关（OOC_RELSEM=0）⇒ 行为逐位不变。
                    try:
                        _rr, _relsem_rec = relsem_release_hook(q, graph_contexts, _rr)
                    except Exception as _e:                      # noqa: BLE001
                        logger.warning("批测 relsem 票异常，退回现门: %s", _e)
                    refuse = _rr is not None
                    # 2026-10-20 联合门装机（作者指令）：加法钩子，仅在前级门放行时评估；
                    # 部署系数与冻结阈值：_jointgate_deploy_20261020.json。
                    if not refuse:
                        try:
                            from backend.joint_gate import joint_gate_hook
                            _jr, _jrec = joint_gate_hook(
                                q, graph_contexts, None, manual_eng=manual_engine)
                            if _jr is not None:
                                refuse = True
                                _rr = _jrec
                        except Exception as _e:                  # noqa: BLE001
                            logger.warning("批测联合门异常，退回现门: %s", _e)
                    if refuse:
                        ans = "根据当前知识库未检索到与问题相关的内容，无法回答。"
                        if use_cot:
                            _reason_txt = (json.dumps(_rr, ensure_ascii=False) if _rr
                                           else "relevance=%.2f object_mismatch=%s"
                                           % (relevance, ooc_mismatch))
                            return ans, graph_contexts, {"cot_reasoning": ans,
                                                         "ooc_refusal_reason": _reason_txt}
                        return ans, graph_contexts

                ctx_text = "\n".join(graph_contexts)

                if use_cot:
                    ans, cot_reasoning = self._generate_answer_with_cot(ctx_text, q)
                    return ans, graph_contexts, {"cot_reasoning": cot_reasoning}
                else:
                    prompt = (
                        "你是一名工业设备故障诊断专家。严格遵守以下约束：\n"
                        "1. 仅使用下方【知识图谱路径】中的信息回答，禁止编造、扩展或引入外部知识。\n"
                        "2. 每条结论必须能在图谱路径中找到直接依据，无法确证的条目不得输出。\n"
                        "3. 如果路径信息不足以完整回答，明确说明「根据当前图谱未找到相关答案」，不要猜测。\n"
                        "4. 答案需结构化：先列出故障现象→可能原因→解决方法，逐条给出路径依据。\n\n"
                        "【知识图谱路径】\n"
                        + ctx_text +
                        f"\n\n【问题】{q}\n\n【答案】"
                    )
                    ans = self._generate_answer(prompt)
                    return ans, graph_contexts
            else:
                ans = "知识图谱中未找到相关信息。"
                graph_contexts = [""]
                if use_cot:
                    return ans, graph_contexts, {"cot_reasoning": ans}
                return ans, graph_contexts

        return self._run_ragas_eval(test_items, process, "graph_rag", output_dir)

    def _generate_answer(self, prompt: str) -> str:
        """通过依赖注入的 answer_generator 或 api_client 生成 KG-RAG 答案。

        优先级：answer_generator -> api_client -> judge_llm（回退，不推荐）。
        KGRAG_ANSWER_LOCAL=1 时反转这条阶梯：只准用注入的本机 answer_generator，
        失败就抛异常——静默回退远程会让"线上答案由本机模型生成"这条作者硬约束
        在最容易被忽略的地方失守（`.env` 写着 local、实际调远程，读数照样产出）。
        """
        if os.getenv(u"KGRAG_ANSWER_LOCAL", u"0") == u"1":
            if self.answer_generator is None:
                raise RuntimeError(
                    u"KGRAG_ANSWER_LOCAL=1 但未注入 answer_generator ⇒ 拒绝用远程通道生成答案")
            return str(self.answer_generator(prompt))

        if self.answer_generator is not None:
            try:
                return str(self.answer_generator(prompt))
            except Exception as e:
                logger.warning(
                    "answer_generator 调用失败: %s，尝试回退到 api_client/judge_llm", e
                )

        if self.api_client is not None:
            try:
                response = self.api_client.chat(
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                )
                content = response.get("content", "")
                if content:
                    return content
                logger.warning("api_client 返回空内容，回退到 judge_llm")
            except Exception as e:
                logger.warning(
                    "api_client 调用失败: %s，回退到 judge_llm", e
                )

        logger.warning(
            "未配置 answer_generator 或 api_client，"
            "回退到 judge_llm 生成 KG-RAG 答案（不推荐：judge_llm 应仅用于评估）"
        )
        try:
            response = self.judge_llm.invoke(prompt)
            return str(response)
        except Exception as e:
            # 修复 9.13-⑤：此前兜底返回固定串"无法生成答案：LLM 调用失败。"
            # 并照常进入 RAGAS 评分 + 断点缓存——失败题以"假答案"参评
            # （faithfulness 可能因"未编造"反而虚高）。改为抛异常：题目
            # 标记为 failed、排除出评分，由下次运行重试，分数不被污染。
            logger.error("judge_llm 回退也失败: %s", e)
            raise RuntimeError(f"答案生成失败（LLM 调用失败）: {e}") from e

    def _generate_answer_with_cot(self, context: str, question: str) -> tuple:
        """使用 Chain-of-Thought 推理生成 KG-RAG 答案。

        流程：检索到的知识 → CoT 逐步推理 → 结构化回答。
        修复 J2：prompt 强制模型分开输出【推理过程】与【最终答案】，
        解析后分别返回，cot_reasoning 不再是 answer 的复制。
        修复 9.10-③：`_split_cot_output` 返回 (answer, reasoning)——此前
        `reasoning, answer = _split_cot_output(...)` 把两者**赋值反了**，导致
        G 侧 `answer` 字段实际存的是【推理过程】、`cot_reasoning` 存的是
        【最终答案】→ RAGAS 评分基于推理过程而非最终答案，且与 V 侧
        （取 `[0]`=answer，正确）答案形态不对称。已修正赋值顺序。

        Args:
            context: 检索到的知识图谱路径上下文
            question: 用户问题

        Returns:
            (answer: str, cot_reasoning: str)
        """
        prompt = self._answer_template().format(context=context, question=question)
        raw_output = self._generate_answer(prompt)
        answer, reasoning = _split_cot_output(raw_output)
        return answer, reasoning

    def _judge_ooc_refusal(self, question: str, answer: str) -> float:
        """判定 out-of-context 题的回答是否为正确拒答（9.10 新增指标）。

        正确拒答 = 明确表示无法回答/未找到相关信息，且未编造具体答案 → 1.0；
        编造了看似合理的答案或答非所问 → 0.0。judge 调用失败时用规则信号兜底。
        """
        if not answer or not answer.strip():
            return 1.0  # 空回答视为正确拒答（未编造）
        prompt = _OOC_REFUSAL_PROMPT.format(
            domain=getattr(self, "ooc_domain", "工业离心泵故障维保手册"),
            question=str(question)[:800], answer=str(answer)[:2500],
        )
        try:
            resp = self.judge_llm.invoke(prompt)
            text = str(getattr(resp, "content", resp) or "").strip().upper()
            return 1.0 if text.startswith("YES") else 0.0
        except Exception as e:
            logger.warning("OOC 拒答判定 judge 失败，使用规则兜底: %s", e)
            return 1.0 if any(sig in answer for sig in _OOC_REFUSAL_SIGNALS) else 0.0

    def _ooc_metrics_for(self, item: dict, answer: str) -> dict:
        """仅对 out-of-context 题计算拒答正确性指标，其余题返回空 dict。"""
        if item.get("type") != "out-of-context":
            return {}
        return {"ooc_refusal": self._judge_ooc_refusal(
            item.get("question", ""), answer)}

    def _run_ragas_eval(self, test_items, processor, system_label, output_dir) -> dict:
        """共享评估循环：题目级并发生成答案，逐题落盘断点续跑。

        并行度由 self.answer_parallel 控制（线程池 max_workers）；断点按题目 id
        去重恢复，并发完成顺序不影响最终 per_question 顺序（完成后按 test_items 重排）。

        Args:
            processor: callable(item) -> (answer: str, contexts: list[, metadata: dict])
                       支持返回 2-tuple 或 3-tuple，第 3 个元素为可选的元数据字典
                       （如 {"cot_reasoning": "..."}），会被合并到 per_question 记录中。
        """
        import threading
        import concurrent.futures
        import time

        os.makedirs(output_dir, exist_ok=True)
        ckpt_path = os.path.join(output_dir, f"{system_label}_answers_ckpt.json")
        lock = threading.RLock()

        # —— 断点恢复（按题目 id 去重）——
        per_question = []
        n_success = 0
        n_failed = 0

        if os.path.exists(ckpt_path):
            try:
                with open(ckpt_path, "r", encoding="utf-8") as f:
                    saved = json.load(f)
                if isinstance(saved, dict) and "per_question" in saved:
                    # 修复 P2-⑦：只恢复 status=="ok" 且题目指纹匹配的记录——
                    # failed 题下次重试；测试集题目内容改动（id 未变）时旧答案视为过期丢弃
                    items_by_id = {
                        item.get("id", f"Q{i + 1}"): item
                        for i, item in enumerate(test_items)
                    }
                    kept = []
                    for r in saved["per_question"]:
                        if r.get("status") != "ok":
                            continue
                        it = items_by_id.get(r.get("id"))
                        if it is None:
                            continue
                        if r.get("_qhash") != _question_fingerprint(it):
                            continue  # 题目内容已改，旧答案过期
                        kept.append(r)
                    per_question = kept
                    n_success = len(kept)
                    n_failed = 0
                    print(f"  [{system_label}] 断点恢复: {len(per_question)}/{len(test_items)} 题有效答案")
            except Exception as e:
                logger.warning("[%s] 断点文件损坏，从头开始: %s", system_label, e)
                per_question = []
                n_success = 0
                n_failed = 0

        done_ids = {r.get("id") for r in per_question if r.get("status") == "ok"}
        pending = [(i, item) for i, item in enumerate(test_items)
                   if item.get("id", f"Q{i + 1}") not in done_ids]
        total = len(test_items)

        def _save_ckpt():
            with lock:
                ok = sum(1 for r in per_question if r.get("status") == "ok")
                fail = sum(1 for r in per_question if r.get("status") == "failed")
                _atomic_json_dump(ckpt_path, {"per_question": per_question, "n_success": ok, "n_failed": fail})

        def _work(item_and_idx):
            idx, item = item_and_idx
            q = item.get("question", "")
            gt = item.get("ground_truth", "")
            item_id = item.get("id", f"Q{idx + 1}")
            last_exc = None
            # 修复：单题重试 1 次（代理偶发挂起/超时，重试一次常可恢复）。
            # 此前一次异常直接标 failed，一次 900s 挂起 = 该题白等 15 分钟。
            for attempt in range(2):
                try:
                    result = processor(item)
                    if len(result) == 3:
                        ans, ctxs, metadata = result
                    else:
                        ans, ctxs = result
                        metadata = {}

                    # 9.10：OOC 题在答案生成阶段即计算拒答正确性指标（独立于
                    # RAGAS 四指标，见 _judge_ooc_refusal）；非 OOC 题为空 dict
                    # 数值型题：与 RAGAS 语义判分正交的机器判分（见 regression R8）
                    # 2026-10-11：SKIP_OOC_JUDGE=1（只生成答案那一趟）时不调判官——
                    # 否则"只生成不判分"这一趟仍会对 21 道 OOC 题各打一次远程判官，
                    # 与"本机生成、判分另走子代理"的口径冲突。OOC 拒答判定改到判分阶段统一做。
                    if os.getenv("SKIP_OOC_JUDGE", "0") == "1":
                        _metrics = {}
                    else:
                        _metrics = self._ooc_metrics_for(item, ans)
                    _num = numeric_accuracy(ans, gt)
                    if _num is not None:
                        _metrics["numeric_accuracy"] = _num
                    record = {
                        "id": item_id, "question": q, "answer": ans,
                        "contexts": ctxs, "ground_truth": gt, "status": "ok",
                        # 修复 KG-9：保留题型字段，供按查询类型分层分析
                        "type": item.get("type", "unknown"),
                        # 修复 P2-⑦：题目内容指纹，供断点恢复时检测题目是否已改动
                        "_qhash": _question_fingerprint(item),
                        "metrics": _metrics,
                    }
                    record.update(metadata)
                    with lock:
                        global_n = sum(1 for r in per_question if r.get("status") == "ok") + 1
                        per_question.append(record)
                        _save_ckpt()
                    print(f"  [{system_label}] {global_n}/{total} OK: {q[:50]}...")
                    return
                except EvalIntegrityError:
                    # 完整性致命错误：立即中止整轮，不重试、不降级为"单题失败"
                    raise
                except Exception as e:
                    last_exc = e
                    if attempt == 0:
                        logger.warning("Q %s 第 1 次尝试失败，5s 后重试: %s", item_id, e)
                        time.sleep(5)
            with lock:
                per_question.append(
                    {"id": item_id, "question": q, "error": str(last_exc), "status": "failed",
                     "type": item.get("type", "unknown")}
                )
                _save_ckpt()

        # —— 并发生成答案（题目级并行，max_workers=answer_parallel）——
        if pending:
            max_workers = max(1, int(self.answer_parallel or 1))
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
                list(ex.map(_work, pending))

        # 完成后按 test_items 顺序重排（并发完成顺序不稳定）
        if per_question:
            id_to_rec = {}
            for r in per_question:
                id_to_rec.setdefault(r.get("id"), r)
            per_question = []
            for i, item in enumerate(test_items):
                rid = item.get("id", f"Q{i + 1}")
                if rid in id_to_rec:
                    per_question.append(id_to_rec[rid])
        n_success = sum(1 for r in per_question if r.get("status") == "ok")
        n_failed = sum(1 for r in per_question if r.get("status") == "failed")

        if n_success == 0:
            logger.error("All %d questions failed for %s", total, system_label)
            return {"n_questions": 0, "n_successful": 0,
                    "n_failed": n_failed, "per_question": per_question}

        # —— 只生成、不评分（2026-10-11）：答案侧换成本机模型后，判分要另走子代理通道，
        # 故先把"生成本机答案"这一半单独跑完并保住断点，评分阶段再接入。
        # 语义是"停在这里"，不是"跳过并当作已完成"：断点文件保留、不写任何分。
        if os.getenv("SKIP_RAGAS_EVAL", "0") == "1":
            print(f"  [{system_label}] SKIP_RAGAS_EVAL=1：答案生成完成 "
                  f"({n_success}/{total})，保留断点、不评分")
            return {"n_questions": len(per_question), "n_successful": n_success,
                    "n_failed": n_failed, "n_skipped_batches": 0,
                    "per_question": per_question, "eval_skipped": True}

        print(f"  [{system_label}] 答案生成完成 ({n_success}/{total})，开始 RAGAS 评分...")
        result = self._run_ragas(
            per_question, n_success, n_failed, system_label, output_dir,
        )
        # RAGAS 完成，删除断点文件（最终结果已在 _run_ragas 中保存）
        if os.path.exists(ckpt_path):
            os.remove(ckpt_path)
        return result

    def _run_ragas(self, per_question, n_success, n_failed, system_label, output_dir) -> dict:
        """小批 RAGAS 评分（5题一批），每批完成即落盘断点，支持续跑。"""
        from datasets import Dataset
        import warnings
        warnings.filterwarnings('ignore', category=DeprecationWarning)
        from ragas.metrics import (
            faithfulness, answer_relevancy, context_precision, context_recall,
        )
        from ragas import evaluate as ragas_evaluate
        from ragas.run_config import RunConfig
        import numpy as np

        # 方案 B：context_precision 判定级并行（防御式 patch，失败自动回退串行）
        _patch_context_precision_parallel()

        def _truncate_contexts_wrapper(contexts):
            return _truncate_contexts(contexts)

        # —— OOC 跳过（2026-09-13）：SKIP_OOC_RAGAS=1 时不为 OOC 题跑四指标 ——
        # OOC 题的四指标本就被下面的 ooc_mask 丢弃、且 _attach_per_question_metrics
        # 明确不写回（其正确性由 ooc_refusal 单独计），继续送评分等于白烧
        # 60/288≈21% 的 judge 调用。此处只从"评分输入"移除，OOC 记录仍保留在
        # 输出 per_question 中，因此对外指标与聚合口径不变（恒等，非近似）。
        _all_records = per_question
        if os.getenv("SKIP_OOC_RAGAS", "0") == "1":
            _kept = [r for r in per_question
                     if str(r.get("type", "")) != "out-of-context"]
            if len(_kept) != len(per_question):
                print(f"  [{system_label}] 跳过 OOC 题 RAGAS 评分："
                      f"{len(per_question)} → {len(_kept)} 题"
                      "（OOC 正确性由 ooc_refusal 单独计，四指标对其不适用）")
            per_question = _kept

        # —— 配对外排除（2026-10-11 丙案，预注册 §14.2 P2/P3）：登记单点名的题不进四指标 ——
        # 手法与上面 OOC 跳过段完全一致：只从"评分输入"里移除，记录仍留在 _all_records，
        # 所以其它口径的分母一字节不变。判据实现放在 backend/ragas_exclude.py，
        # 自检桩（driver S15–S17）打的就是这一个函数对象，不是另写一份平行实现。
        from backend.ragas_exclude import apply_exclude
        per_question, _excl_removed, _excl_real = apply_exclude(
            per_question, os.getenv("RAGAS_EXCLUDE_JSON"))
        if _excl_removed:
            print(f"  [{system_label}] 按配对登记单排除 {len(_excl_removed)} 题不进 RAGAS 四指标："
                  f"送判 {len(per_question)} 题（登记单 {os.path.basename(_excl_real)}，"
                  f"预注册 §14 丙案；输出记录表仍完整）")

        ckpt_path = os.path.join(output_dir, f"{system_label}_ragas_ckpt.json")
        # 2026-08-07 修复：评分粒度改为 1 题 1 次 ragas_evaluate。
        # 原 5 题整批只要有一个 judge 调用超时，整批重试且已算分全部丢失；
        # 长 context 下 TimeoutError 高频时会把耗时放大数倍。逐题调用后，
        # 单题失败只重试/跳过该题，且每题完成即落盘断点。
        BATCH = int(os.getenv("RAGAS_BATCH_SIZE", "1"))
        os.makedirs(output_dir, exist_ok=True)

        # —— 断点恢复 ——
        accum_scores = {}  # {metric_name: [val, ...]}
        start_q = 0
        if os.path.exists(ckpt_path):
            try:
                with open(ckpt_path, "r", encoding="utf-8") as f:
                    saved = json.load(f)
                if isinstance(saved, dict):
                    accum_scores = saved.get("accum", {})
                    start_q = saved.get("done", 0)
                    # 修复 P2：断点只存"已完成题数"，题目集增删/顺序变化会错位——
                    # 用 done_ids 校验与当前 per_question 前缀是否一致，不一致则评分从头开始
                    expected_ids = [r.get("id") for r in per_question[:start_q]]
                    if saved.get("done_ids") != expected_ids:
                        print(f"  [{system_label}] RAGAS 断点与当前题目集不一致，评分从头开始")
                        accum_scores, start_q = {}, 0
                    elif start_q >= len(per_question):
                        print(f"  [{system_label}] RAGAS 已完成（从断点恢复），跳过评分...")
                        _restore_metrics_from_ckpt(per_question, saved)
                        os.remove(ckpt_path)
                        result_dict = {"n_questions": n_success, "n_successful": n_success,
                                       "n_failed": n_failed, "n_skipped_batches": 0,
                                       "per_question": _all_records}
                        result_dict.update({m: float(np.mean(vals)) for m, vals in accum_scores.items() if vals})
                        return result_dict
                    else:
                        print(f"  [{system_label}] RAGAS 断点恢复: {start_q}/{len(per_question)} 题已完成")
                        _restore_metrics_from_ckpt(per_question, saved)
            except Exception as e:
                logger.warning("[%s] RAGAS 断点损坏: %s", system_label, e)

        def _save_ckpt():
            # 原子写：强杀不截断断点文件（2026-08-30 根因修复）
            _atomic_json_dump(ckpt_path, {"accum": accum_scores, "done": start_q,
                                          "done_ids": [r.get("id") for r in per_question[:start_q]],
                                          "per_question": per_question})
            # CP 逐条判词 sidecar（零新增调用；与 ckpt 同频落盘，中断也留得下半成品）
            try:
                st = write_cp_sidecar(
                    ckpt_path.replace("_ragas_ckpt.json", "_cp_items.json"),
                    per_question, extra={"round_ckpt": os.path.basename(ckpt_path)})
                if st == "no-capture":
                    print("  [CP-sidecar] 捕获开关开着但一条判词都没截到"
                          "（并行 patch 未生效？）⇒ 不落空文件，勿读成"
                          "\"本轮零条逐 context 判词\"")
            except Exception as e:
                print(f"  [CP-sidecar] 写失败（不影响判分，仅少一份 sidecar）：{e}")

        metrics = [faithfulness, answer_relevancy, context_precision, context_recall]
        # v8: 4 metric two-way parallel (ThreadPoolExecutor), ~40% per-question speedup.
        # v9b: 4-way parallel (one metric per thread). ragas evaluates metrics of
        # one sample serially, so CP+CR in the same thread were serial (CP 12
        # verdicts then CR big-input call ~120s each). Splitting every metric
        # into its own thread runs CP and CR concurrently -> ~30% faster.
        # Proxy probe showed 24 concurrent requests with zero degradation.
        from concurrent.futures import ThreadPoolExecutor
        import threading
        import time
        import pandas as pd

        _metric_groups = [[m] for m in metrics]  # v9b: 4-way parallel, one metric per thread

        def _eval_one_group(m_group, lock):
            """Evaluate one metric group in a thread, return (df, scores_dict).

            batch_size=BATCH（原硬编码 1）：让同一 metric 的多道题（BATCH 道）由
            ragas 内部 executor 并发跑——这是安全的跨题并行：每线程一个事件循环，
            并发单元是"同 metric 多题"，总在途受 qwen_judge 全局闸（≤12）限制。
            若保持 batch_size=1，BATCH>1 时同 metric 多题会被 ragas 串行批处理，
            跨题并发不生效。
            """
            g_ds = Dataset.from_dict({
                "question": [r["question"] for r in batch_records],
                "answer": [r["answer"] for r in batch_records],
                "contexts": [_truncate_contexts_wrapper(r.get("contexts", [""])) for r in batch_records],
                "ground_truth": [r.get("ground_truth", "") for r in batch_records],
            })
            try:
                g_result = ragas_evaluate(
                    g_ds, metrics=m_group, llm=self.judge_llm,
                    embeddings=self.ragas_embed, batch_size=max(1, BATCH),
                    run_config=RunConfig(
                        timeout=max(
                            self.eval_timeout,
                            int(os.getenv("RAGAS_TIMEOUT_MIN", "300")),
                        ),
                        max_retries=int(os.getenv("RAGAS_MAX_RETRIES", "1")),
                        max_wait=int(os.getenv("RAGAS_MAX_WAIT", "30")),
                    ),
                )
                g_df = g_result.to_pandas()
            except Exception as e:
                # 稳定性加固（2026-08-09）：judge 900s 超时后 RAGAS 内部 asyncio
                # 事件循环可能在已关闭循环上清理（RuntimeError: Event loop is closed），
                # 泄漏到进程层会直接崩溃整个评测。此处兜底为"该指标组本批无分"，
                # 由外层批次重试/跳过逻辑处理，进程不退出（不牺牲性能）。
                logger.warning(
                    "[%s] RAGAS 指标组 %s 批次 %d-%d 异常: %s",
                    system_label, [m.name for m in m_group], start_q, end_q, e,
                )
                g_df = None
            g_scores = {}
            # 2026-08-11：OOC 口径修正——OOC 题的 RAGAS 四指标不参与聚合
            # （指标适用域：上下文有答案才可评，OOC 拒答=0 分/编造有分是
            # 口径盲区，出错方反而占优）。OOC 正确性由 ooc_refusal 单独计。
            ooc_mask = [str(r.get("type", "")) == "out-of-context"
                        for r in batch_records]
            for m in m_group:
                col = m.name
                if col in g_df.columns:
                    for vi, v in enumerate(g_df[col].tolist()):
                        try:
                            fv = float(v)
                            if not np.isnan(fv) and not ooc_mask[vi]:
                                with lock:
                                    accum_scores.setdefault(col, []).append(fv)
                                g_scores.setdefault(col, []).append(fv)
                        except (TypeError, ValueError):
                            pass
            return g_df, g_scores

        TOTAL = len(per_question)
        n_skipped_batches = 0

        while start_q < TOTAL:
            _t_q = time.time()
            end_q = min(start_q + BATCH, TOTAL)
            batch_records = [r for r in per_question[start_q:end_q] if r.get("status") == "ok"]
            if not batch_records:
                start_q = end_q
                _save_ckpt()
                continue

            # 修复 KG-8：批次失败不再静默跳过——重试 1 次，仍失败才跳过并计数，
            # 最终结果记录 n_skipped_batches，避免"分数基于不完整样本且无感知"。
            batch_ok = False
            for attempt in range(2):
                try:
                    _lock = threading.Lock()
                    with ThreadPoolExecutor(max_workers=4) as executor:
                        futures = [executor.submit(_eval_one_group, mg, _lock)
                                   for mg in _metric_groups]
                        results = [f.result() for f in futures]
                    # 稳定性加固：个别指标组异常（g_df=None）不影响其余组——
                    # 过滤后再合并；全部失败才抛错走重试/跳过。
                    ok_dfs = [r[0] for r in results if r[0] is not None]
                    if not ok_dfs:
                        raise RuntimeError("本批所有 RAGAS 指标组均失败")
                    dfs = ok_dfs
                    merged_df = dfs[0] if len(dfs) == 1 else pd.concat(dfs, axis=1)
                    # KG-9: per-question metrics
                    self._attach_per_question_metrics(batch_records, merged_df, metrics)
                    batch_ok = True
                    break
                except Exception as e:
                    logger.warning(
                        "[%s] RAGAS 批次 %d-%d 第 %d 次尝试失败: %s",
                        system_label, start_q, end_q, attempt + 1, e,
                    )
            if not batch_ok:
                n_skipped_batches += 1
            start_q = end_q
            _save_ckpt()
            print(f"  [{system_label}] RAGAS {start_q}/{TOTAL} OK ({time.time() - _t_q:.0f}s/题)")

        # —— 聚合最终分数 ——
        result_dict = {
            "n_questions": n_success, "n_successful": n_success,
            "n_failed": n_failed, "n_skipped_batches": n_skipped_batches,
            "per_question": _all_records,
        }
        for m in metrics:
            vals = accum_scores.get(m.name, [])
            result_dict[m.name] = float(np.mean(vals)) if vals else float("nan")

        os.makedirs(output_dir, exist_ok=True)
        save_path = os.path.join(output_dir, f"{system_label}_results.json")
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(result_dict, f, ensure_ascii=False, indent=2, default=str)
        logger.info("Saved %s results → %s", system_label, save_path)

        if os.path.exists(ckpt_path):
            os.remove(ckpt_path)
        return result_dict

    def _attach_per_question_metrics(self, batch_records, df, metrics) -> None:
        """修复 KG-9：把 RAGAS 逐题指标写回 per_question 记录。

        df 行顺序与 batch_records 一致（Dataset 构造顺序），按行取分，
        使 _build_type_breakdown 能按题型的 type 字段做分层统计。

        2026-08-11：OOC 口径修正——out-of-context 题的 RAGAS 四指标
        判定为"不适用"（不写入）。RAGAS 四指标的设计前提是"上下文里有
        答案"，OOC 题上下文无答案，四指标衡量的是"编造能力"而非"回答
        能力"：正确拒答被判 0 分、编造答案反而有分（出错方占优）。
        OOC 题的唯一正确性指标 = ooc_refusal（答案生成阶段已写入）。
        """
        import numpy as np
        for j, rec in enumerate(batch_records):
            is_ooc = str(rec.get("type", "")) == "out-of-context"
            row_metrics = {}
            for m in metrics:
                if is_ooc:
                    # OOC 题四指标不适用：不写入（保留 ooc_refusal）
                    continue
                col = m.name
                if col in df.columns and j < len(df):
                    try:
                        fv = float(df[col].tolist()[j])
                        if not (isinstance(fv, float) and np.isnan(fv)):
                            row_metrics[col] = fv
                    except (TypeError, ValueError, IndexError):
                        pass
            # 9.10：合并而非覆盖——OOC 拒答指标（ooc_refusal）在答案生成阶段已
            # 写入 metrics，此处追加 RAGAS 四指标，避免互相覆盖
            rec["metrics"] = {**rec.get("metrics", {}), **row_metrics}

    def _extract_ragas_metrics(self, eval_result) -> dict:
        """从 RAGAS EvaluationResult 提取各指标的均值（忽略 NaN）。"""
        import numpy as np
        metrics = {}
        scores = eval_result.scores if hasattr(eval_result, "scores") else []
        if not scores:
            return metrics
        for key in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]:
            vals = []
            for s in scores:
                v = s.get(key) if isinstance(s, dict) else getattr(s, key, None)
                if v is not None and (not isinstance(v, float) or not np.isnan(v)):
                    vals.append(float(v))
            if vals:
                metrics[key] = float(np.mean(vals))
            else:
                metrics[key] = float("nan")
        return metrics

    def _compare_results(self, results: dict) -> dict:
        """逐指标计算 delta 与胜者。"""
        vec = results.get("vector_rag") or {}
        kg = results.get("graph_rag") or {}
        delta = {}
        winner = {}
        for key in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]:
            v = vec.get(key)
            k = kg.get(key)
            if v is not None and k is not None:
                d = float(k) - float(v)
                delta[key] = d
                winner[key] = "kg_rag" if d > 0 else ("vector_rag" if d < 0 else "tie")
            else:
                delta[key] = None
                winner[key] = "unknown"
        return {"delta": delta, "winner_per_metric": winner}

    def _agg_metrics_excluding(self, results: dict,
                               exclude_types=("out-of-context",)) -> dict:
        """按指标聚合均值，可剔除指定 type 的题（'含/不含 OOC'双口径）。

        per_question 记录的 metrics 由 _attach_per_question_metrics 写入（RAGAS
        逐题分数）。返回 {metric: mean, metric_n: 有效样本数}；无有效样本时均值为 None。
        """
        import numpy as np
        _KEYS = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
        agg = {}
        for m in _KEYS:
            vals = [
                r["metrics"][m] for r in results.get("per_question", [])
                if r.get("status") == "ok"
                and r.get("type") not in exclude_types
                and m in (r.get("metrics") or {})
            ]
            agg[m] = float(np.mean(vals)) if vals else None
            agg[m + "_n"] = len(vals)
        return agg

    def _per_question_discrimination(self, vec_results: dict, kg_results: dict,
                                     exclude_types=("out-of-context",)) -> dict:
        """9.11-1：逐题区分度与胜场统计（剔除 OOC 后）。

        R11 规则：逐题同分率 > 2/3 的指标对两系统对比无区分力，均值差不得作为
        领先证据；均值差 ±0.1 内一律按"两系统接近"表述。

        Returns:
            {metric: {"same": int, "total": int, "g_wins": int, "v_wins": int, "ties": int}}
        """
        _KEYS = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
        vmap = {r.get("id"): r for r in vec_results.get("per_question", [])
                if r.get("status") == "ok" and r.get("type") not in exclude_types}
        kmap = {r.get("id"): r for r in kg_results.get("per_question", [])
                if r.get("status") == "ok" and r.get("type") not in exclude_types}
        out = {}
        for m in _KEYS:
            same = total = g_wins = v_wins = ties = 0
            for qid in kmap:
                if qid not in vmap:
                    continue
                vv = (vmap[qid].get("metrics") or {}).get(m)
                kv = (kmap[qid].get("metrics") or {}).get(m)
                if vv is None or kv is None:
                    continue
                total += 1
                if abs(vv - kv) < 1e-9:
                    same += 1
                    ties += 1
                elif kv > vv:
                    g_wins += 1
                else:
                    v_wins += 1
            out[m] = {"same": same, "total": total,
                      "g_wins": g_wins, "v_wins": v_wins, "ties": ties}
        return out

    def _context_attribution(self, kg_results: dict) -> dict:
        """9.13-②：统计 KG 侧每题 context 的"图谱路径 vs 社区原文"构成。

        路径 context 含 "--[关系]-->" 标记（_collect_graph_contexts 生成）；
        其余为社区正文/原文片段。用于报告透明呈现"得分机制"——此前
        auto_008 的 context_precision≈1.0 实际来自社区长文而非图谱路径。
        """
        out = []
        for r in kg_results.get("per_question", []):
            if r.get("status") != "ok":
                continue
            ctxs = r.get("contexts") or []
            n_path = sum(1 for c in ctxs if "--[" in str(c))
            n_text = sum(1 for c in ctxs if str(c) and "--[" not in str(c))
            out.append({"id": r.get("id"), "type": r.get("type"),
                        "n_path": n_path, "n_text": n_text,
                        "n_ctx": len([c for c in ctxs if str(c)])})
        return out

    def _build_type_breakdown(self, vec_results: dict, kg_results: dict) -> dict:
        """修复 KG-9：按 type 字段分层统计逐题指标均值。

        per_question 记录的 metrics 由 _attach_per_question_metrics 写入
        （RAGAS 逐题分数），此处按题型分组后取均值，渲染真实分层表。
        返回 {"type": {"n": int, "vec": {metric: mean}, "kg": {metric: mean}}}。
        """
        _KEYS = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
        breakdown = {}
        for label, results in (("vec", vec_results), ("kg", kg_results)):
            for item in results.get("per_question", []):
                qtype = item.get("type", "unknown")
                data = breakdown.setdefault(qtype, {"n": 0, "vec": {}, "kg": {}})
                if label == "vec":
                    data["n"] += 1
                metrics = item.get("metrics") or {}
                for k in _KEYS:
                    if k in metrics:
                        data[label].setdefault(k, []).append(metrics[k])
        for qtype, data in breakdown.items():
            for label in ("vec", "kg"):
                agg = {k: (sum(v) / len(v)) for k, v in data[label].items() if v}
                data[label] = agg
        return breakdown


# ── 装配末道相关度重排（OPT_ASM_RERANK，默认关；2026-10-17 甲案）──────────────
# 依据 files/_j226_20261011/_cp_rerank_screen3_20261017.json（sha8 5dc91f3c）的离线重放：
# KG 侧按"与题面的 IDF 加权 2-gram 重叠 ÷ sqrt(该条 gram 数)"降序重排 12 条，逐题 CP 复算
# 0.5612→0.7127；同一把尺套到向量臂上，向量也从 0.6729→0.7874。所以这是**方法无关的装配步骤**，
# 不是图谱专属补丁——两臂必须同时开才可比（否则只是把向量臂的天然优势搬过来）。
# 只重排、不增删不改内容 ⇒ CP 的输入集合与 CR 的输入集合逐字不变。
_RERANK_IDF_CACHE = {}


def _ctx_bigrams(s):
    t = re.sub(r"\s+", "", s or "")
    return set(t[i:i + 2] for i in range(len(t) - 1))


def _corpus_idf(corpus):
    """检索语料的行级 2-gram IDF，进程内按语料指纹缓存（建表本身 O(语料)）。"""
    if not corpus:
        return {}
    key = "%d:%s" % (len(corpus),
                     hashlib.sha256((str(corpus[0])[:200] + str(corpus[-1])[:200])
                                    .encode("utf-8")).hexdigest()[:12])
    hit = _RERANK_IDF_CACHE.get(key)
    if hit is not None:
        return hit
    df, n = {}, 0
    for doc in corpus:
        # 文档粒度取"行"：与离线重放的建表口径同源（整块当一篇会让长块的 gram 全部
        # 变成"高频"，IDF 排序与测量不一致——10-17 验证件 R6 实测仅 13/30 排列相同）。
        for line in str(doc).splitlines():
            g = _ctx_bigrams(line)
            if not g:
                continue
            n += 1
            for x in g:
                df[x] = df.get(x, 0) + 1
    idf = {x: math.log((n - v + 0.5) / (v + 0.5) + 1.0) for x, v in df.items()} if n else {}
    _RERANK_IDF_CACHE[key] = idf
    return idf


def _rerank_contexts(contexts, query, corpus=None):
    """OPT_ASM_RERANK=1 时按相关度重排最终 context 顺序；默认原样返回。

    语料缺失/IDF 表为空 ⇒ 保持原序并告警，绝不静默乱排（那种"看起来生效其实没生效"
    正是本项目 R1/R2 两轮静默退化的成因）。
    """
    if (os.getenv("OPT_ASM_RERANK") or "").strip() != "1":
        return contexts
    if not contexts or len(contexts) < 2:
        return contexts
    idf = _corpus_idf(corpus or [])
    if not idf:
        logger.warning("[OPT_ASM_RERANK] IDF 表为空（语料缺失），按原序返回")
        return contexts
    qg = _ctx_bigrams(query)

    def _score(s):
        g = _ctx_bigrams(s)
        if not g:
            return -1.0
        return sum(idf.get(x, 0.0) for x in (g & qg)) / math.sqrt(len(g))

    order = sorted(range(len(contexts)), key=lambda i: (-_score(contexts[i]), i))
    return [contexts[i] for i in order]


def _truncate_contexts(contexts, max_chars=12000, max_ctx=12, total_budget=48000):
    """截断过长的 contexts，避免超过 judge 模型的上下文窗口。

    修复 C2：不再只保留"前 4 个"（GraphRAG 前几个往往是路径碎片，
    导致 faithfulness 判错）。改为保留更多 context（默认 12 个），
    且总字符预算控制。
    修复 9.10-④：max_ctx 8 → 12——实测 auto_008/013 原始 40 个 context
    中 32/31 条图谱路径被 8 个社区正文挤掉（路径 0 条），含答案核心三元组
    （如"热油泵停泵后--[包含]-->每0.5h盘车一次"）。
    2026-08-11：max_chars 6000→12000、total_budget 24000→48000——
    配合 _fragment_window 扩窗（1200→12000），GT 原因/方法句的原文证据
    不再被单条截断丢弃；总预算 48000 字符 ≈ 12 条 × 4000 平均，仍在
    judge 上下文窗口内（此前 24000 仅容 6 条 × 4000）。
    2026-09-09（P1-2）：[推理链] 前缀行保护——链行是同主体多因多果的
    聚合高密度行：① 在保留列表中优先（社区正文可能排在路径之前，链行
    原顺序下会被挤到 max_ctx 边缘甚至丢弃）；② 单条截断上限放宽一倍，
    防止聚合行被常规 max_chars 拦腰截断而丢失尾部并列项
    （090 类"处理方法(5)"的末项）。无链行时行为与旧版完全一致。
    """
    # P1-2：链行前置（保持原相对顺序），其余 context 在后
    # 2026-10-14（方案 F-addon）：追加的 dense 槽必须活到判分侧——本函数在生成与
    # 判分两层都会把 context 硬截到 max_ctx，若追加槽不在此处放行，就会重演
    # TABLE_FALLBACK 当年"实测 0/288 生效"的坑。默认关 ⇒ 行为逐位不变。
    if os.getenv("OPT_F_ADDON", "0") == "1":
        max_ctx += int(os.getenv("OPT_F_ADDON_N", "6"))
    chain_ctxs = [c for c in contexts if c and c.startswith("[推理链]")]
    other_ctxs = [c for c in contexts if c and not c.startswith("[推理链]")]
    out = []
    used = 0
    for c in chain_ctxs + other_ctxs:
        # 链行单条上限放宽一倍：聚合行信息密度最高，截断代价最大
        limit = max_chars * 2 if c.startswith("[推理链]") else max_chars
        c = c if len(c) <= limit else c[:limit]
        if used + len(c) > total_budget:
            break
        out.append(c)
        used += len(c)
        if len(out) >= max_ctx:
            break
    return out or [""]


def _split_cot_output(raw: str) -> tuple:
    """解析 CoT 输出，分离【推理过程】与【最终答案】。

    兼容多种格式：
      - 含【最终答案】标记 → 以该标记为界切分
      - 含"最终答案:"等变体 → 同样处理
      - 无法识别时 → 返回 (raw, "")，推理过程置空（不复制答案）
    """
    if not raw:
        return "", ""
    markers = ["【最终答案】", "最终答案：", "最终答案:", "[最终答案]"]
    for m in markers:
        idx = raw.find(m)
        if idx >= 0:
            reasoning = raw[:idx].strip()
            answer = raw[idx + len(m):].strip()
            return answer, reasoning
    return raw, ""


def _question_fingerprint(item: dict) -> str:
    """题目内容指纹（question + ground_truth），用于断点恢复时检测题目是否已改动。

    修复 P2-⑦：此前断点只按题目 id 去重恢复——测试集某题内容改了但 id 没变时，
    会复用旧答案导致结果不新鲜。指纹不匹配视为过期，重新生成。
    """
    import hashlib
    q = str(item.get("question", ""))
    gt = str(item.get("ground_truth", ""))
    return hashlib.md5((q + "|" + gt).encode("utf-8")).hexdigest()


def _backfill_type_field(records: list, test_items: list) -> int:
    """按 id 从测试集回填记录的 type 字段（round_001 答案缺题型标签）。

    2026-08-09：复用旧答案评分的路径（round_001 vector 答案）不带 type，
    导致分层分析（OOC/题型）不可用。返回实际回填的条数。
    """
    type_by_id = {it.get("id"): it.get("type", "unknown") for it in test_items}
    filled = 0
    for r in records:
        if not r.get("type"):
            t = type_by_id.get(r.get("id"))
            if t:
                r["type"] = t
                filled += 1
    return filled


def _restore_metrics_from_ckpt(per_question: list, saved: dict) -> None:
    """把 RAGAS 断点中已评分的逐题 metrics 合并回 per_question（9.11-3 附带修复）。

    曾发生：进程中断后断点续跑，已完成的批次不重跑 → 这些题的 per_question.metrics
    缺失（G 侧双口径只剩 8/15 题有分、与 V 的 15 题不可比）。断点 ckpt 现在保存
    per_question 快照，恢复时把已评分题的 metrics 补回。
    """
    saved_pq = saved.get("per_question") or []
    saved_by_id = {}
    for r in saved_pq:
        if r.get("status") == "ok" and (r.get("metrics") or {}):
            saved_by_id[r.get("id")] = r
    for r in per_question:
        sr = saved_by_id.get(r.get("id"))
        if sr and (sr.get("metrics") or {}):
            r["metrics"] = sr["metrics"]


# ── OOC 拒答正确性判定（9.10，消除 OOC 指标口径不对称的实质妥协）──────────
# RAGAS 四指标对 out-of-context 题无"拒答"口径：空 context 的正确拒答被判
# faithfulness=0.0，而塞入无关 context 后"答非所问"反而可能得分。该指标对称地
# 衡量"当正确答案不在知识库时，系统是否正确拒答"——1.0=明确拒答且未编造；
# 0.0=编造了具体答案或答非所问。与 RAGAS 用同一 judge（self.judge_llm）。
_OOC_REFUSAL_PROMPT = (
    "你是评估员。以下问题对应的正确答案【不包含在】系统知识库中"
    "（{domain}）。\n"
    "【问题】{question}\n"
    "【系统回答】{answer}\n\n"
    "请判断该回答是否属于\"正确拒答\"：\n"
    "- 正确拒答（YES）：回答明确表示无法回答/知识库中未找到相关信息/未检索到"
    "相关内容，且没有编造任何具体的诊断结论、数值或方法。\n"
    "- 拒答失败（NO）：回答编造了看似合理的具体答案（即使开头声称无法回答，"
    "只要后续给出编造内容即失败），或答非所问。\n"
    "只输出一个词：YES 或 NO。"
)

# 规则兜底信号（judge 调用失败时使用）：出现任一即视为正确拒答
_OOC_REFUSAL_SIGNALS = (
    "未找到相关信息", "未检索到相关内容", "无法回答", "无法从",
    "不能回答", "知识库中未找到", "未找到相关答案",
)


# —— 装配候选池观测钩子（默认关；仅供离线分析件使用）——
# 为什么要它：`_collect_graph_contexts` 只交回"选完并截断后"的那 max_ctx 条，
# 落选候选一出门就没了 ⇒ 想离线问一句"换一种挑法会不会更好"（重排模拟）
# 就没有输入。这里在出口处把**截断前各通道的有序候选**交给一个回调。
# 两条硬约束：
#   1) 本函数体里**不读任何新环境变量**——`analysis/is620f_b5_assembly_ab.py`
#      的门 A9 按"函数体里 getenv 的键"清点开关，凭空加键会让那份清点说谎；
#      开与关由调用方（分析件）决定。
#   2) 回调抛异常必须吞掉：观测件不许有任何能力改变生产读数。
# sink 为 None 时本函数与接钩子之前逐位相同（单线程离线件用，非线程安全）。
_ASM_POOL_SINK = {"fn": None}


# —— 臂 D 预注册 #1 旋钮一（sha8 479f3677）归一化常量 ——
# 口径与 analysis/final_v3.py:224 _norm 同款：标点剥离（含全半角）+ 小写。
# AGENTS.md 纪律：自写弱版 norm 会把全半角差异算成"不存在"。
_UC_PUNCT = re.compile(
    r'[\s，。、；：？！“”‘’（）【】《》,.;:?!"\'()\[\]<>—\-_/\\|~`]+')


def set_assembly_pool_sink(sink):
    """注册（或传 None 清空）装配候选池回调。生产链路从不调用它。"""
    _ASM_POOL_SINK["fn"] = sink


def _collect_graph_contexts(query_str: str, manual_engine, community_engine,
                            max_ctx: int = 12, min_path_slots: int = 6,
                            is_ooc: bool = False,
                            entity_chunk_map: Optional[dict] = None,
                            chunk_texts: Optional[list] = None,
                            dense_retriever=None) -> list:
    """收集图谱路径上下文（独立函数便于测试）。

    顺序策略（修复 C2/H1）：
      - 社区长文本（正文）排在前，图谱 PATH 碎片排在后 → RAGAS 评分时
        前几个 context 是有信息量的正文而非 20~50 字符的路径碎片。
      - 全局去重（跨社区/跨路径的重复 chunk 只保留一次）。
      - 修复 9.13-②：社区正文最多占用 (max_ctx - min_path_slots) 个槽位，
        为图谱路径保留最少 min_path_slots 个槽——曾实证 auto_008/013 的
        9 条社区正文把答案核心图谱路径全部挤出 12 槽（路径 0 相关），
        得分靠原文托底、图谱检索失败被掩盖。
      - 修复 2026-08-05（shukongdashi 检索调优）：min_path_slots 4 → 6。
        集线实体（FANUC/SIEMENS 变体）被实体去重压到 ~6 个后，轮询交错
        第 1 轮恰好覆盖每个实体 1 条最短路径；4 槽会把概念实体（回参考点
        报警）的路径切在第 5-6 位。6 槽在保住 6 个社区槽的同时让概念路径
        进入评分 context。
      - 修复 2026-08-06（shukongdashi 优化）：弱语义路径（配套使用/包含）
        最多占 2 槽。低分组中这类路径占 42.6%，挤占了有效路径槽位。
    """
    # —— 上下文槽位消融旋钮（2026-10-13，登记见 半月reports/上下文槽位消融_预注册与判读表_20261013.md）——
    # KG_PATH_CAP = 本次装配最多允许占用的"路径槽"数。未设/空 ⇒ None ⇒ 全部走历史分支，
    # 行为逐位不变。设了就同时把"最少路径槽"压到同一上限——否则被压掉的路径会被
    # min_path_slots 的补足逻辑拉回来。空出来的槽位交给社区/原文池，用以检验
    # "差在图谱产物占掉了原文槽位"这一猜想（本旋钮只改组装，不改检索/生成/判分）。
    _gcap_env = os.getenv("KG_PATH_CAP")
    _gcap = None if _gcap_env in (None, "") else max(0, min(max_ctx, int(_gcap_env)))
    # —— 显式路径预算开关（2026-10-14，K≈4 臂甲案；登记见
    # 半月reports/路径预算K4臂_预注册与判读表_20261014.md）——
    # OPT_PATH_SLOTS=N：本次装配**恰**给路径 N 个槽（有候选时不少于 N、也不多于 N），
    # 其余槽交给 dense chunk / 社区。与 KG_PATH_CAP 的区别：KG_PATH_CAP 只压上限，
    # 而两条装配分支里路径与 chunk/社区的**先后相反**（else 分支 chunk/社区先入 ⇒
    # 即便设了 KG_PATH_CAP，路径也拿 0 槽——彩排 2/3 题实证）。本键在两条分支里都
    # 保证路径**先于** chunk/社区入槽，故 K 均匀落在 N。
    # 未设/空 ⇒ None ⇒ 不进入任何新分支，行为与历史逐位相同。
    # 实现上复用 _gcap：置 _pslots 时同时把 _gcap 钉到同一值，让既有的"路径上限/
    # 补足上限/rest_other 抑制"逻辑一并生效（_gcap 非 None 时 rest_other 不被补入）。
    _pslots_env = os.getenv("OPT_PATH_SLOTS")
    _pslots = None if _pslots_env in (None, "") else max(0, min(max_ctx, int(_pslots_env)))
    if _pslots is not None:
        min_path_slots = _pslots
        _gcap = _pslots
    if _gcap is not None and _gcap < min_path_slots:
        min_path_slots = _gcap
    path_contexts = []
    path_rel_types = []  # 记录每条路径的关系类型
    path_node_names = []
    if manual_engine:
        paths = manual_engine.query(query_str)
        for path in paths:
            if len(path) >= 3:
                names, rels, dirs = path[0], path[1], path[2]
            else:
                names, rels = path
                dirs = []
            if names and rels:
                parts = [str(names[0])]
                for i, rel in enumerate(rels):
                    if i + 1 < len(names):
                        d = dirs[i] if i < len(dirs) else "?"
                        if d == "-":
                            parts.append(f"<--[{rel}]--")
                        else:
                            parts.append(f"--[{rel}]-->")
                        parts.append(str(names[i + 1]))
                path_contexts.append("".join(parts))
                path_rel_types.append(set(rels))
                path_node_names.append([str(n) for n in names])
            elif names:
                path_contexts.append(", ".join(str(n) for n in names))
                path_rel_types.append(set())
                path_node_names.append([str(n) for n in names])

    # —— 严格文档臂（2026-10-13）：微观引擎＝只要路径本身 ——
    # 文档（知乎那篇）的微观模式回的是 (节点名, 关系类型) 拼成的路径，
    # **不含**我们额外挂的 [原文依据] 长文、不掺社区正文、也没有推理链聚合行。
    # 本开关默认关 ⇒ 历史分支逐位不变；只用于"严格模仿文档"那一臂。
    if os.getenv("STRICT_DOC_ASSEMBLY", "0") == "1":
        _pure = [c for c in path_contexts if not str(c).startswith("[推理链]")]
        return _pure[:max_ctx]

    # 方案 F 开关定义于条件块外（2026-08-24）：独立通道段在函数主流程上，
    # 无 chunk 绑定参数的场景（test_optD 等）也必须可用。
    _opt_f = os.getenv("OPT_F", "0") == "1"
    _opt_f_n = int(os.getenv("OPT_F_TOP_N", "4"))
    # 方案 F-addon（2026-10-14）：OPT_F 的"加法"版。诊断实读（287 题逐题数）：
    # 原 OPT_F 从没给任何一题加过路径槽（路径槽 optf 少 166 题、多 0 题，中位
    # 7→3；推理链少 124、多 0），它是严格用"路径/推理链"换"正文" ⇒ 四指标
    # 一格没赢过 optC。本开关把 dense 通道从"在 12 槽预算内插队"改成"12 槽
    # 之外追加"：路径与社区组装与不加通道时逐位相同，只在其后补 k 条去重新
    # chunk。默认 0 ⇒ 行为与历史逐位相同。
    _opt_f_addon = os.getenv("OPT_F_ADDON", "0") == "1"
    _opt_f_addon_n = int(os.getenv("OPT_F_ADDON_N", "6"))

    if entity_chunk_map and chunk_texts and path_node_names:
        # P1 依据压缩开关（2026-08-22）：EVIDENCE_WINDOW=1 时原文依据
        # 从整 chunk 改为实体邻句窗口（默认 0 = 历史行为，保证旧轮次可复现）。
        _ev_win = os.getenv("EVIDENCE_WINDOW", "0") == "1"
        # 方向 B 优化（2026-08-23）：OPT_B_HEAD=1 时依据固定截断到 400 字
        # （保留路径头部 38 字高密度信息 + 依据开头）。依据中位 2035 字占
        # 片段 98% 长度，是 CP 判定的噪声源。固定截断（非定位）避免
        # P1 的"答非字面切错证据"问题。
        _opt_b = os.getenv("OPT_B_HEAD", "0") == "1"
        _OPT_B_MAX = int(os.getenv("OPT_B_MAX", "400"))
        # 方案 C 修订（2026-08-23）：OPT_C=1 时，路径依据改用滑窗 dense
        # 检索的短 chunk（对齐 Vector 分块），替换长绑定依据。离线验证：
        # 滑窗检索对召回失败题找回率 70%（vs 社区索引低），且 chunk 短
        # （Vector 487字/条 vs KG 2400字/条），同时解召回与效率。
        _opt_c = os.getenv("OPT_C", "0") == "1"
        _opt_c_n = int(os.getenv("OPT_C_TOP_N", "2"))
        # 方案 F（2026-08-24）：滑窗 dense top-K 作为独立通道进 context。
        # 归因实证（第四十五节）：27 题 CR<0.1 中 52% 是"组装丢失"——
        # OPT_C 让所有路径绑定相同 top2 chunk，context 实际只含 ~2 条
        # 滑窗 chunk，检索器能力被组装层浪费。模拟：独立通道 top4 使
        # 全中题 2→9、GT 平均覆盖 18%→40%。
        # 行为：①chunk 插在相关路径后、社区前（社区/弱路径让位）；
        # ②跳过 OPT_C 路径依据替换（防同 chunk 重复占位，路径变纯
        # 三元组更短，字符预算让给 chunk）；③OOC 题不启用。
        _seen_snippets = set()  # 方案 A：跨路径依据去重
        # 默认关闭（与历史 optC_r2 一致）：显式置 1 才启用 A/B
        _opt_a_dedup = os.getenv("OPT_A_DEDUP", "0") == "1"  # A 开关
        _opt_b_nodes = os.getenv("OPT_B_NODES", "0") == "1"  # B 开关
        for i, p in enumerate(path_contexts):
            nodes = path_node_names[i]
            if _opt_f:
                # 方案 F v2（2026-08-25 修正）：保留 OPT_C 滑窗替换/原始
                # 绑定不变——v1 曾在此 continue 剥离一切依据，实测 CR 崩
                # （0.819→0.562）：模拟测的是"原 context+追加"，实现却
                # 做成"剥离+替换"，长依据承载的 GT 全丢。v2 独立通道只
                # 补充与依据去重后的新 chunk。
                pass
            if _opt_c and dense_retriever is not None:
                # 方案 A+B（2026-08-26）：路径级依据去重 + 节点名聚焦检索
                # A: 多路径绑到同一 chunk 时去重，释放槽位给被挤出的 dense chunk
                # B: 检索 query 拼接路径节点名，让每条路径绑定各自相关的 chunk
                _path_query = query_str
                if _opt_b_nodes and nodes:
                    _path_query = query_str + " " + " ".join(str(n) for n in nodes[:3])
                snippets = []
                snippets_raw = []  # 去重用截断版，绑定用原文
                try:
                    for _sc, _row in dense_retriever(_path_query):
                        _t = _row.get("text") or ""
                        if not _t:
                            continue
                        _t_key = _t[:500] if len(_t) > 500 else _t
                        if _opt_a_dedup and _t_key in _seen_snippets:
                            continue
                        snippets.append(_t_key)
                        snippets_raw.append(_t)
                        if len(snippets) >= _opt_c_n:
                            break
                except Exception:
                    snippets = []
                    snippets_raw = []
                if snippets:
                    if _opt_a_dedup:
                        for _sk in snippets:
                            _seen_snippets.add(_sk)
                    path_contexts[i] = p + "\n[原文依据]\n" + "\n---\n".join(snippets_raw)
                    continue
                # dense 检索失败则回退到原绑定逻辑
            cids = []
            for n in nodes:
                # 2026-10-05（P0 检索确定性 F2）：sorted 化集合迭代序——chunk id
                # 为 str 的域（IS620F）set 迭代序跨进程随机，这里确定化。
                for cid in sorted(entity_chunk_map.get(n, [])):
                    if cid not in cids and 0 <= cid < len(chunk_texts):
                        cids.append(cid)
            if not cids:
                continue
            snippets = []
            for cid in cids[:2]:
                text = chunk_texts[cid]
                if _ev_win:
                    text = _evidence_window(text, nodes, query_str)
                elif _opt_b and len(text) > _OPT_B_MAX:
                    text = text[:_OPT_B_MAX]
                elif len(text) > 12000:
                    hit = min(
                        (text.find(t) for t in _query_code_tokens(query_str)
                         if text.find(t) >= 0),
                        default=0,
                    )
                    start = max(0, hit - 600)
                    text = text[start:start + 12000]
                snippets.append(text)
            path_contexts[i] = p + "\n[原文依据]\n" + "\n---\n".join(snippets)

    community_contexts = []
    if community_engine:
        # OOC 题不做文本单元级兜底召回：避免语义相近但不属于知识库的原文
        # 把“正确拒答”重新拉回编造路径。
        matched_comms, comm_contexts = community_engine.query(
            query_str, chunk_fallback=not is_ooc)
        for ctx in comm_contexts:
            if isinstance(ctx, str):
                community_contexts.append(ctx)

    # 2026-08-10：图谱路径实体二次打分（借鉴 kg-rag entity-weighted re-ranking）。
    # 社区片段中同时出现"图谱路径实体"（如 KA10、X018-5、中间继电器）的，
    # 与图谱证据一致、信息密度更高——按实体命中数加权排序，优先进 context。
    # 纯相似度召回可能把"提到类似主题但非同一实体"的片段排前，实体共现修正。
    if path_node_names and community_contexts:
        path_entities = {str(n) for names in path_node_names for n in names}

        def _entity_hit(ctx):
            cnt = 0
            for ent in path_entities:
                if len(ent) >= 2 and ent in ctx:
                    cnt += 1
            return cnt

        scored_comm = [(i, _entity_hit(c), c) for i, c in enumerate(community_contexts)]
        scored_comm.sort(key=lambda x: (-x[1], x[0]))
        # 实体命中 >0 的片段前置（保持原相对顺序），未命中的保持原序在后
        hit_comm = [c for _, s, c in scored_comm if s > 0]
        miss_comm = [c for _, s, c in scored_comm if s == 0]
        community_contexts = hit_comm + miss_comm

    # 动态组装（2026-08-05 调优）：
    # - 若图谱路径中存在与问题关键代码 token（报警号/地址/参数号等）
    #   高度相关的路径，则这些路径前置，其余槽位再给社区正文；
    # - 否则保持社区正文前置（保住 pump 已验证的 faithfulness 优势）；
    # - 只要存在图谱路径，路径槽位不少于 min_path_slots。
    code_tokens = _query_code_tokens(query_str)
    # 有代码 token 时只按代码判定相关路径，避免中文短语把代码路径挤出；
    # 无代码时才用中文短语兜底（如"定向准停"这类概念题）。
    # 2026-09-06（修复③）：当 code_tokens 全是"型号弱代码"（如 GSK983 里的
    # SK983——手册内每条路径都含该型号子串，无任何鉴别力）时，仍应叠加中文
    # 短语打分——否则现象实体的语义路径（OT报警(超程报警)--[处理方法]-->…）
    # 因不含型号词而 score=0，被含型号的泛化"故障现象"路径挤出 12 槽
    # （GSK983 实测：超程/存储限位复位方法进不了 context）。强代码（报警号
    # 纯数字/ALM/DGN/X 地址等）存在时保持 code-only 原行为，杜绝回归。
    _opt_e = os.getenv("OPT_E", "0") == "1"
    if _opt_e:
        # OPT_E：无论有无代码 token，都提取概念短语用于多跳路径增强评分
        phrase_tokens = _query_phrase_tokens(query_str, path_contexts)
    else:
        _need_phrase = (not code_tokens) or not any(
            _is_strong_code_token(t) for t in code_tokens)
        phrase_tokens = _query_phrase_tokens(query_str, path_contexts) if _need_phrase else []
    tokens = code_tokens + phrase_tokens

    def _path_score(p):
        # 2026-08-11：打分只针对图谱路径部分（[原文依据] 之前）。
        # 原文片段是给 judge/答案生成补充的证据文本，若参与打分，含问题
        # 高频词（如"指示灯不亮""系统显示"）的无关路径会因原文加分而
        # 挤出真正相关的路径（auto_023 实证：702 报警路径原文含多个
        # 问题短语，score 虚高占满 12 槽，X 轴路径被挤出）。
        path_part = str(p).split("[原文依据]")[0]
        score = sum(1 for t in tokens if t in path_part)
        if _opt_e:
            # OPT_E 增强：多跳路径（>=2 跳）额外加分——确保完整链进入 context。
            # 评分逻辑：每条路径的文本中 "--[关系]-->" 出现次数即为跳数。
            # 多跳路径含更多实体→答案覆盖概率更高，给予概念 token 的匹配
            # 额外 0.3 权重（不替代代码 token 的满权重，防注入旧问题 auto_023）。
            hop_count = path_part.count("-->") + path_part.count("<--")
            if hop_count >= 2:
                phrase_hit = sum(0.3 for t in phrase_tokens if t in path_part)
                score += phrase_hit
        return score

    # 修复 2026-08-06（shukongdashi 优化）：分离强/弱语义路径
    # 弱语义路径（配套使用/包含/影响）最多占 2 槽，为强语义路径让出空间
    # 但含代码 token 的路径即使是"包含"关系也应视为强语义（如 X018-5 地址）
    _WEAK_RELS = {"配套使用", "包含", "影响"}
    strong_paths = []  # 含可能原因/解决方法/表现为 或 含代码 token 的路径
    weak_paths = []    # 只含配套使用/包含/影响 且 不含代码 token 的路径
    for i, p in enumerate(path_contexts):
        rels = path_rel_types[i] if i < len(path_rel_types) else set()
        # 含代码 token 的路径视为强语义（只查图谱路径部分，排除原文证据干扰）
        path_part = str(p).split("[原文依据]")[0]
        has_code = any(t in path_part for t in code_tokens) if code_tokens else False
        if rels and rels.issubset(_WEAK_RELS) and not has_code:
            weak_paths.append(p)
        else:
            strong_paths.append(p)

    if tokens:
        # 相关路径（含代码/短语 token）优先
        scored_strong = [(idx, _path_score(p), p)
                         for idx, p in enumerate(strong_paths) if _path_score(p) > 0]
        scored_strong.sort(key=lambda x: (-x[1], x[0]))
        # 2026-10-04（装配层线 E1，预注册件 §2 臂 A2）：OPT_ASSEM_GP=1 时做
        # **稳定分区**——含查询代码 token 的路径整体排在纯短语命中路径之前，
        # 分区内保持原 (−score, idx) 序（不引入新的次级排序）。默认关 ⇒
        # 行为与改造前逐位相同；开关由分析件（E1 臂）控制，生产默认不启。
        if os.getenv("OPT_ASSEM_GP", "0") == "1":
            def _has_code(pth):
                part = str(pth).split("[原文依据]")[0]
                return any(t in part for t in code_tokens)
            scored_strong = ([x for x in scored_strong if _has_code(x[2])]
                             + [x for x in scored_strong if not _has_code(x[2])])
        rel_paths = [p for _, _, p in scored_strong]
        rel_set = set(rel_paths)
        other_strong = [p for p in strong_paths if p not in rel_set]
        other_paths = other_strong + weak_paths
    else:
        rel_paths = []
        other_strong = []               # 本分支不被读，仅为候选池登记有个确定值
        other_paths = strong_paths + weak_paths

    seen = set()
    contexts = []

    def _add(items):
        for c in items:
            if c and c not in seen:
                seen.add(c)
                contexts.append(c)
                if len(contexts) >= max_ctx:
                    return

    def _has_community_summary(ctxs):
        return any(
            str(c).startswith(("[硬事实]", "[社区摘要]", "[社区报告]"))
            for c in ctxs
        )

    # 方案 F：滑窗 chunk 独立通道（在两分支的社区段之前插入）
    # 注：_opt_f 定义于下方 entity_chunk_map 条件块之外（见上），
    # 保证无 chunk 绑定参数的场景（如 test_optD）也可用。
    _opt_f_chunks = []
    if _opt_f and not _opt_f_addon and dense_retriever is not None and not is_ooc:
        try:
            for _sc, _row in dense_retriever(query_str):
                _t = (_row.get("text") or "").strip()
                if not _t:
                    continue
                if len(_t) > 500:
                    _t = _t[:500]
                if _t not in _opt_f_chunks:
                    _opt_f_chunks.append(_t)
                # v2: 多取 2 条候选，插入时与已入 context 的依据去重
                if len(_opt_f_chunks) >= (_opt_f_addon_n if _opt_f_addon
                                          else _opt_f_n) + 2:
                    break
        except Exception:
            _opt_f_chunks = []

    # —— 臂 D 预注册 #1 旋钮一（OPT_ASSEM_UC，默认关=行为逐位不变）——
    # 统一竞争：全部路径（强/弱语义）按 统一分 = _path_score + 0.5×锚点归一化
    # 命中数 排序后竞争槽位；废除 rel/other 通道预分配与弱路径 cap=2。
    # 槽位总数不变（不扩槽——GSK983 E1 H4 教训：扩槽稀释纯度）。
    # 锚点匹配 = 全量 shingle + 尾部片段（AGENTS.md 纪律 2：实体名常只含
    # GT token 尾部片段，只取前缀会漏配 ⇒ 假 0→0）。
    # 开关注册：沿 OPT_ASSEM_GP 先例（E1 登记链），默认关，由臂 D 驱动件显式置 1。
    _uc = os.getenv("OPT_ASSEM_UC", "0") == "1" and bool(tokens) and bool(path_contexts)
    _uc_ranked = None
    if _uc:
        def _uc_shingles(s):
            return {s[i:i + 2] for i in range(len(s) - 1)} if len(s) >= 2 else {s}

        def _uc_anchor_hits(i):
            nodes = path_node_names[i] if i < len(path_node_names) else []
            ents = [_UC_PUNCT.sub('', str(n)).lower() for n in nodes if n]
            hits = 0
            for t in tokens:
                tt = _UC_PUNCT.sub('', str(t)).lower()
                if not tt:
                    continue
                ts = _uc_shingles(tt)
                hit = False
                for e in ents:
                    if not e:
                        continue
                    if tt in e or (len(tt) >= 4 and tt[-4:] in e):
                        hit = True
                        break
                    es = _uc_shingles(e)
                    if ts and es and len(ts & es) >= max(2, int(0.6 * len(ts))):
                        hit = True
                        break
                if hit:
                    hits += 1
            return hits

        _uc_ranked = sorted(
            ((-(_path_score(p) + 0.5 * _uc_anchor_hits(i)), i, p)
             for i, p in enumerate(path_contexts)),
            key=lambda x: (x[0], x[1]))
        _uc_paths = [p for _, _, p in _uc_ranked]
        _add(_uc_paths[:max_ctx])
        comm_pool = community_contexts
        if not _has_community_summary(comm_pool):
            comm_pool = comm_pool[:4]
        if _opt_f_chunks and not _opt_f_addon:
            _ctx_texts = {str(c) for c in contexts}
            _fresh = [t for t in _opt_f_chunks if t not in _ctx_texts]
            _add(_fresh[:_opt_f_n])
        _add(comm_pool)
    elif rel_paths:
        # 方向 A 优化（2026-08-23）：OPT_A_SLOTS=1 时给 dense chunk 社区召回
        # 保证槽位（≥3 槽），路径让出。修复"图谱路径+绑定原文占满 12 槽，
        # dense chunk 被挤出，单跳题 37% 召回失败"的问题。
        _opt_a = os.getenv("OPT_A_SLOTS", "0") == "1"
        if _pslots is not None:
            # _pslots：路径预算按"已入槽的**去重后**行数"计。不可对
            # rel_paths[:_gcap] 先切片再去重——路径池头部常成对重复（同一路径
            # 渲两次），先在切片内去重会让 N 落不满（全量 288 实测 45 题低于 4）。
            _rel_added = 0
            for p in rel_paths:
                if _rel_added >= _gcap:
                    break
                if p and p not in seen:
                    seen.add(p)
                    contexts.append(p)
                    _rel_added += 1
                    if len(contexts) >= max_ctx:
                        break
            _n_added = len(contexts)
        else:
            _path_cap = max_ctx - 3 if _opt_a else max_ctx
            if _gcap is not None:
                _path_cap = min(_path_cap, _gcap)
            _add(rel_paths[:_path_cap])
            _n_added = len(rel_paths[:_path_cap])
        need = max(0, min_path_slots - _n_added)
        # 限制弱语义路径最多占 2 槽，但不低于 min_path_slots
        weak_cap = 2
        strong_need = min(need, len(other_strong))
        weak_need = min(need - strong_need, len(weak_paths), weak_cap)
        _add(other_strong[:strong_need])
        _add(weak_paths[:weak_need])
        # 补足：若强+弱不足 min_path_slots，放宽弱语义 cap 补足
        # 非 chunk 行数不能用 `'-->' in c` 计数（反向 `<--[关系]--` 行无尾 `>`、
        # `[推理链]` 行无箭头 ⇒ 会少算 ⇒ 弱路径补足超额）。_pslots 下改用行数。
        path_total = (len(contexts) if _pslots is not None
                      else sum(1 for c in contexts if '-->' in c))
        if path_total < min_path_slots:
            extra = min(min_path_slots - path_total,
                        len(weak_paths) - weak_need)
            _add(weak_paths[weak_need:weak_need + extra])
        rest_other = [p for p in other_paths if p not in seen]
        comm_pool = community_contexts
        if not _has_community_summary(comm_pool):
            comm_pool = comm_pool[:4]
        # 方案 F v2：独立通道先于社区；与路径依据（已含 OPT_C top2）
        # 去重后补充新 chunk，新增量受 _opt_f_n 约束
        if _opt_f_chunks and not _opt_f_addon:
            _ctx_texts = {str(c) for c in contexts}
            _fresh = [t for t in _opt_f_chunks if t not in _ctx_texts]
            _add(_fresh[:_opt_f_n])
        if _opt_a:
            # 保证 dense chunk 至少 3 槽（社区摘要/硬事实优先）
            _add(comm_pool[:3])
            # 槽位未满时补剩余路径与社区
            remaining = max_ctx - len(contexts)
            if remaining > 0:
                _add(comm_pool[3:3 + remaining])
            if _gcap is None:
                _add(rest_other)
        else:
            _add(comm_pool)
            if _gcap is None:
                _add(rest_other)
    else:
        if _pslots is not None:
            # —— OPT_PATH_SLOTS（甲案，2026-10-14）：路径先于 chunk/社区入槽 ——
            # 历史 else 分支把 dense chunk（_opt_f_chunks）与社区排在路径之前，
            # 设了 KG_PATH_CAP 也拿不到保留槽（预注册 §5 彩排 2/3 题实证 K=0）。
            # 本块先入恰 ≤_gcap(=N) 条路径，再由 dense chunk、社区填剩余槽。
            # 默认 _pslots=None ⇒ 本块不执行，历史逐位不变。
            # —— 去重后计满 _gcap 条：不可对 strong_paths[:_gcap] 先切片再去重 ——
            # 路径池头部常成对重复（同一路径按两端各渲一次），切片内去重只剩一半
            # ⇒ K 落不满（全量 288 实测 45 题低于 4，其中 9 题有充足去重候选）。
            # 逐条走去重后计满 N 条，空出的槽交 dense chunk / 社区。
            _strong_added = 0
            for p in strong_paths:
                if _strong_added >= _gcap:
                    break
                if p and p not in seen:
                    seen.add(p)
                    contexts.append(p)
                    _strong_added += 1
                    if len(contexts) >= max_ctx:
                        break
            # 非 chunk 行数 = 已入槽的行数（此处 contexts 只有路径/推理链，尚未加 chunk）。
            # 不能用 `'-->' in c` 计数：反向 `<--[关系]--` 行没有尾 `>`，`[推理链]` 行无箭头
            # ⇒ 会少算 ⇒ 弱路径补足超额（彩排实测 5~6 条非 chunk，目标 4）。
            path_total = len(contexts)
            weak_added = 0
            weak_limit = max(0, _gcap - path_total)
            for p in weak_paths:
                if weak_added >= weak_limit:
                    break
                if p and p not in seen:
                    seen.add(p)
                    contexts.append(p)
                    weak_added += 1
                    if len(contexts) >= max_ctx:
                        break
            if _opt_f_chunks and not _opt_f_addon:
                _ctx_texts = {str(c) for c in contexts}
                _fresh = [t for t in _opt_f_chunks if t not in _ctx_texts]
                _add(_fresh[:_opt_f_n])
            comm_cap = max_ctx
            if not _has_community_summary(community_contexts):
                comm_cap = min(comm_cap, 4)
            _add(community_contexts[:comm_cap])
        else:
            # 无相关路径时保持社区正文前置；社区只占少量槽位，让路径拿到更多槽。
            comm_cap = max_ctx - min_path_slots if path_contexts else max_ctx
            if path_contexts and not _has_community_summary(community_contexts):
                comm_cap = min(comm_cap, 4)
            # 方案 F v2：独立通道先于社区（与 rel_paths 分支同序，同样去重截断）
            if _opt_f_chunks and not _opt_f_addon:
                _ctx_texts = {str(c) for c in contexts}
                _fresh = [t for t in _opt_f_chunks if t not in _ctx_texts]
                _add(_fresh[:_opt_f_n])
            _add(community_contexts[:comm_cap])
            # 强语义路径全部进入，弱语义路径 cap=2 但不低于 min_path_slots
            _add(strong_paths if _gcap is None else strong_paths[:_gcap])
            path_total = sum(1 for c in contexts if '-->' in c)
            weak_added = 0
            weak_limit = max(2, min_path_slots - path_total)
            if _gcap is not None:
                weak_limit = min(weak_limit, max(0, _gcap - path_total))
            for p in weak_paths:
                if weak_added >= weak_limit:
                    break
                if p and p not in seen:
                    seen.add(p)
                    contexts.append(p)
                    weak_added += 1
                    if len(contexts) >= max_ctx:
                        break

    # 方案 F-addon 的追加发生在调用点、`_truncate_contexts` **之后**（见 process()
    # 内 "方案 F-addon" 段）：在本函数内追加会被截断硬切到 max_ctx（实测无效）。

    # 方案 D 优化（2026-08-23）：OPT_D=1 时，对 query 与路径实体做 dense
    # 二次检索，把高相关 chunk（短而准）补进 context。修复"图谱路径+绑定
    # 依据覆盖不足，单跳题 26% 召回失败（GT 所在 chunk 不在检索结果）"。
    # dense_retriever 提供 (query_str) -> [(score, row)] 接口（如 comm_eng
    # 的 _match_chunks），返回的 row["text"] 作为补召回候选。
    # 效率约束：补召回 chunk 截断到 OPT_D_MAX(默认400) 字，每题最多补
    # OPT_D_TOP_N(默认2) 条——防止 context 过长拖慢 judge（实测 5min/题）。
    # 归因纯化开关（2026-08-24）：OPT_C=1 会隐式启用本段补召回行为，
    # 导致 optC 轮次的提升混入 OPT_D 效应。设 OPT_D_DISABLE=1 可在
    # OPT_C=1 时强制关闭补召回，单独度量滑窗替换的净贡献。
    _opt_d_pool = []                    # 候选池登记：本段真正递给装配器的候选
    _opt_d_enabled = (
        (os.getenv("OPT_D", "0") == "1" or os.getenv("OPT_C", "0") == "1")
        and os.getenv("OPT_D_DISABLE", "0") != "1"
    )
    if _opt_d_enabled \
            and dense_retriever is not None and not is_ooc:
        _opt_d_n = int(os.getenv("OPT_D_TOP_N", "2"))
        _opt_d_max = int(os.getenv("OPT_D_MAX", "400"))
        _dense_cands = []
        for _dq in [query_str] + [str(n) for n in (path_node_names or [])[:3]]:
            if not _dq:
                continue
            try:
                _dense_cands.extend(dense_retriever(_dq))
            except Exception:
                continue
        # 按相似度降序去重
        _seen_texts = {str(c) for c in contexts}
        _added = 0
        for _score, _row in sorted(_dense_cands, key=lambda x: -x[0]):
            _t = (_row.get("text") or "")
            if not _t:
                continue
            if len(_t) > _opt_d_max:
                _t = _t[:_opt_d_max]
            _opt_d_pool.append(_t)
            if _t in _seen_texts:
                continue
            _seen_texts.add(_t)
            contexts.append(_t)
            _added += 1
            if len(contexts) >= max_ctx or _added >= _opt_d_n:
                break

    _sink = _ASM_POOL_SINK["fn"]
    if _sink is not None:
        _sel = contexts[:max_ctx]
        try:                                # 观测件吞掉一切异常：不许影响生产读数
            _sink({"query": query_str, "is_ooc": bool(is_ooc),
                   "max_ctx": max_ctx, "min_path_slots": min_path_slots,
                   "channels": {"rel_paths": rel_paths,
                                "other_strong": other_strong,
                                "strong_paths": strong_paths,
                                "weak_paths": weak_paths,
                                "community": community_contexts,
                                "f_chunks": _opt_f_chunks,
                                "d_pool": _opt_d_pool,
                                # 2026-10-05 修复：UC 关时 _uc_ranked 为 None，
                                # 池观测件（PoolSink）迭代通道遇 None 即崩并被
                                # 本处 try/except 吞掉 ⇒ rows 永远为空（x2 池
                                # 再生实测 3405 blobs/0 行）。契约：通道值必须可迭代。
                                "uc_paths": (_uc_ranked or [])},
                   "selected": _sel})
        except Exception:
            pass
    return contexts[:max_ctx]


_QUERY_CODE_RE = re.compile(
    r"P/S\d+|SV\d+|ALM-?\d+[A-Za-z0-9]*|NC\d+[A-Za-z0-9]*|DGN\d+|"
    r"X\d{3,4}[-\d.]*|Y\d{3,4}[-\d.]*|F\d+\.\d+|F\d{2,}|G\d+[.#-]\d+|"
    r"M\d{2,}|T\d{2,}|O\d{4,}|A\d{3,}[-\d]*|"
    r"Q\d{2,}|D\d{2,}|VS\d+|DB\d+|JR\d+|CP\d+|KA\d+|QF\d+|SK\d+|PRM\d+|P\d{3,5}|"
    r"端子\d+|7段数码管|\*ESP|\*DECX|\*DEC|\*SP|OVL位|PIL|CH\d{2,}"
)
# 2026-09-06（修复②）：问句中"报警+数字 / 数字+报警"的纯数字报警号。
# 例：报警405、405报警、405号报警、ALARM 90。只提取与"报警"相邻的 2~4 位数字，
# 且数字前不得紧跟字母/数字（避免把 "GSK983报警" 的型号 983 误当报警号）。
_ALARM_PLAIN_NUM_RE = re.compile(
    r"(?:报警|ALARM)\s*号?\s*#?\s*(\d{2,4})|(?<![A-Za-z0-9])(\d{2,4})\s*(?:号\s*)?报警"
)


def _query_code_tokens(query_str: str) -> list:
    """提取问题中的代码类关键 token（报警号/地址/参数号等）。

    位地址分隔符统一为小数点：G008#0/G008-0 → G008.0，便于与清洗后的
    图谱实体/路径文本匹配（清洗阶段已把 # 归一化为 .）。

    2026-09-06（修复②，GSK983 报警号适配）：追加"报警+数字"的纯数字 token。
    GSK983 报警号实体为 "GSK983报警405" 三位命名，用户口语写"报警405/405报警"；
    _QUERY_CODE_RE 只匹配字母开头代号（SK983 等），纯数字 405 不参与路径评分 →
    相关路径无法靠 code token 提到 context 前部（Q2/Q3 实测 212 原因边被挤出）。
    追加规则：仅当问句出现"报警/ALARM+数字"或"数字+报警"且数字 ≥2 位时，
    追加该 int 归一数字。1 位数字（如"报警2"）不追加，避免 405/21x 等含
    子串数字的无关路径全部 +1 分造成排序稀释（别名召回层已能精确处理 1 位号）。
    """
    toks = list(dict.fromkeys(
        m.group(0).replace("#", ".") for m in _QUERY_CODE_RE.finditer(query_str)
    ))
    # 报警号数字（两种语序：报警405 / 405报警 / 405号报警）
    for m in _ALARM_PLAIN_NUM_RE.finditer(query_str):
        raw = m.group(1) if m.group(1) is not None else m.group(2)
        num = str(int(raw))
        if len(num) >= 2 and num not in toks:
            toks.append(num)
    return toks


def _is_strong_code_token(tok: str) -> bool:
    """code token 是否具有"图内鉴别力"（能区分不同路径）。

    GSK983 手册内每条图谱路径都含型号子串 "SK983"——它作为 code token 给
    全部路径 +1 分，等价于不排序。真正能指到具体报警/参数/地址的才是强代码：
      纯数字（报警号，如 405）、字母代号前缀（DGN/ALM/SV/P/S/X/Y/端子 等）。
    弱代码（型号/未指明对象）存在时，调用方应叠加中文短语打分（见组装处）。
    """
    if not tok:
        return False
    if re.fullmatch(r"\d{2,4}", tok):  # 报警号纯数字
        return True
    if re.match(
        r"(?:P/S|SV|ALM|NC|DGN|PRM|X|Y|F|G|M|T|O|A|Q|D|VS|DB|JR|CP|KA|QF|CH)\d",
        tok,
    ):
        return True
    if any(k in tok for k in ("端子", "*ESP", "*DEC", "*SP", "OVL位", "7段数码管")):
        return True
    return False


def _split_sents(text: str) -> list:
    """中文优先的轻量分句：句号/问号/叹号/分号/换行为边界，保留分隔符。"""
    return [s for s in re.split(r"(?<=[。！？；!?;\n])", text) if s.strip()]

def _atomic_json_dump(path, data):
    """原子写 JSON：先写 tmp 再 os.replace，强杀不产生半成品文件。
    若 tmp 残留（上次被杀），replace 会覆盖它。"""
    import tempfile
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=str)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _cached_embed_fn(embed_fn, cache_path=None):
    """滑窗索引 embedding 落盘缓存（2026-08-26，R2 §4.7.2 后续修复）。
    R2 实证：Ollama GPU 浮点抖动使索引两次构建间 22% 题 context 变动，
    CR 效应无法重现。本缓存按文本 sha256 固定向量，索引跨运行
    比特级一致。缓存 miss 时调 Ollama 并写回。

    2026-08-30 修复（断点/缓存频繁损坏根因）：
      - 原子写（tmp+replace）：强杀不再截断缓存文件
      - 读自愈：主文件损坏时尝试 .tmp 残留，最大化复用
      - 延迟批量落盘：每 _FLUSH_EVERY 条才写一次，写放大从 O(n²) 降到 O(n/k)

    cache_path 由调用方按 embedding 模型给出；不传＝沿用历史默认文件
    （离线评测与 8001 默认模型的行为逐字节不变）。
    """
    import hashlib
    cache_path = cache_path or os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        ".sisyphus", "evidence", "cache", "sliding_embed_cache.json")
    cache = {}
    _loaded_from = None
    for candidate in (cache_path, cache_path + ".tmp"):
        if os.path.exists(candidate):
            try:
                with open(candidate, "r", encoding="utf-8") as f:
                    cache = json.load(f)
                _loaded_from = candidate
                print(f"[OPT_C] embedding 落盘缓存加载: {len(cache)} 条 ({os.path.basename(candidate)})")
                break
            except Exception as e:
                print(f"[OPT_C] 缓存读取失败({os.path.basename(candidate)}): {e}")
                cache = {}
    if _loaded_from is None:
        print("[OPT_C] embedding 缓存不可用，全新构建")

    _FLUSH_EVERY = int(os.getenv("EMBED_CACHE_FLUSH_EVERY", "20"))
    _dirty = 0

    def fn(text):
        nonlocal _dirty
        key = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if key in cache:
            return cache[key]
        vec = embed_fn(text)
        cache[key] = [float(x) for x in vec]
        _dirty += 1
        if _dirty >= _FLUSH_EVERY:
            _dirty = 0
            try:
                os.makedirs(os.path.dirname(cache_path), exist_ok=True)
                _atomic_json_dump(cache_path, cache)
            except Exception:
                pass
        return vec

    return fn


_SLIDING_RETRIEVERS = {}
_SLIDING_BUILD_LOCK = threading.Lock()
_RETRIEVER_STATS_BY_MODEL = {}
# 运行时状态（2026-09-13）：记录滑窗检索器是否真的建成、是否真的改变了检索。
# 这是世代无关的静默降级判据——比"依据块字数"推断更可靠（R1/R2 事故即
# OPT_C 声明开启但检索器静默失效、退化为 base 绑定所致）。
RETRIEVER_STATS = {
    "built": False,       # 检索器是否构建成功
    "n_chunks": None,     # 滑窗 chunk 数（世代指纹：500 窗宽 → 363）
    "window_size": None,
    "overlap_size": None,
    "calls": 0,           # 被调用次数（OPT_C 生效时 > 0）
    "hits": 0,            # 返回非空次数（==0 即退化为 base 绑定）
    "reason": "not_attempted",
}


def _stats_store(model_name=None):
    if not model_name:
        return RETRIEVER_STATS
    return _RETRIEVER_STATS_BY_MODEL.setdefault(str(model_name), {
        "built": False, "n_chunks": None, "window_size": None,
        "overlap_size": None, "calls": 0, "hits": 0,
        "reason": "not_attempted", "model": str(model_name),
    })


def get_retriever_stats(model_name=None) -> dict:
    """只读快照：供评测脚本把"OPT_C 是否真的生效"写进 meta.json。

    不传 model_name＝默认模型的历史口径（评测脚本零改动）。
    """
    return dict(_stats_store(model_name))


def _get_sliding_retriever(model_name=None, cache_path=None, manual_path=None):
    """惰性构建滑窗 dense 检索器（方案 C，对齐 Vector 的滑窗分块）。

    同一进程内按 (模型, 缓存文件, 手册) 各建一份：默认参数＝历史 0.6B 单例行为；
    模型设置页切换 embedding 时传入所选模型与按模型缓存路径，使 OPT_C dense
    通道与图谱/向量检索使用同一个 embedding。
    """
    key = (str(model_name or ""), str(cache_path or ""),
           str(manual_path or ""))
    if key in _SLIDING_RETRIEVERS:
        return _SLIDING_RETRIEVERS[key]
    with _SLIDING_BUILD_LOCK:
        if key in _SLIDING_RETRIEVERS:
            return _SLIDING_RETRIEVERS[key]
        return _build_sliding_retriever(model_name, cache_path, manual_path, key)


def _build_sliding_retriever(model_name, cache_path, manual_path, key):
    stats = _stats_store(model_name)
    try:
        from backend.chunking import (parse_manual_chunks_for_retrieval,
                                      chunk_generation_label)
        from backend.ollama_embed import OllamaLocalEmbedding
        from backend.sliding_retriever import SlidingWindowDenseRetriever
        from config import VECTOR_WINDOW_SIZE, VECTOR_OVERLAP_SIZE
        # 2026-10-22：手册名可由 SLIDING_MANUAL env 覆盖（多手册服务需要——SV660N
        # 在役化开考实测：此路径写死 shukongdashi 时，OPT_C dense 通道会给其它
        # 手册的灌错书原文）。默认不设 env ⇒ 路径逐字节不变，8001 产线零影响。
        manual_path = manual_path or os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "data", "manuals",
            os.getenv("SLIDING_MANUAL", "shukongdashi_500cases.md"))
        # v6：分块世代由 OPT_TABLE_ATOMIC 决定（默认 0 = 滑窗，行为不变）。
        # 必须与 Vector 侧（run_pump_fault.build_vector_rag）走同一函数，
        # 否则两系统的文档池不一致，KG-RAG vs Vector-RAG 对比失效。
        chunks = parse_manual_chunks_for_retrieval(manual_path)
        texts = [c.text for c in chunks]
        stats["window_size"] = VECTOR_WINDOW_SIZE
        stats["overlap_size"] = VECTOR_OVERLAP_SIZE
        stats["n_chunks"] = len(texts)
        stats["chunk_generation"] = chunk_generation_label()
        stats["model"] = str(model_name or "")
        stats["cache_path"] = str(cache_path or "")
        embed = (OllamaLocalEmbedding(model_name=model_name)
                 if model_name else OllamaLocalEmbedding())
        # 预热检测：Ollama 挂起时快速失败（60s 超时×3 重试内见分晓），
        # 避免进入 chunk 构建循环后每 chunk 180s 的隐性灾难
        try:
            embed.get_text_embedding("预热测试")
        except Exception as e:
            print(f"[OPT_C] Ollama 预热失败，跳过滑窗检索器构建: {e}")
            stats["reason"] = "ollama_warmup_failed"
            return None
        _retriever = SlidingWindowDenseRetriever(
            texts, _cached_embed_fn(embed.get_text_embedding, cache_path),
            threshold=0.3, top_n=12)
        if not texts or _retriever._matrix.size == 0:
            print("[OPT_C] 滑窗检索器构建失败（embedding 异常），置空")
            stats["reason"] = "build_failed"
            return None
        else:
            # 计数包装：直接测量"OPT_C 是否真的改变了检索"（hits==0 即静默退化）
            def _counting_retriever(qs, _inner=_retriever):
                res = _inner(qs)
                stats["calls"] += 1
                if res:
                    stats["hits"] += 1
                return res

            _SLIDING_RETRIEVERS[key] = _counting_retriever
            stats["built"] = True
            stats["reason"] = "built"
            print(f"[OPT_C] 滑窗 dense 检索器构建: {len(texts)} chunks "
                  f"(window={VECTOR_WINDOW_SIZE}, overlap={VECTOR_OVERLAP_SIZE}, "
                  f"embed={model_name or 'default'})")
    except Exception as e:
        print(f"[OPT_C] 滑窗检索器构建失败: {e}")
        stats["reason"] = f"exception: {e}"
    return _SLIDING_RETRIEVERS.get(key)

def _evidence_window(text: str, entities, query_str: str,
                     max_len: int = 600, hard_cap: int = 800) -> str:
    """P1 依据压缩（2026-08-22，第三十一节诊断）：原文依据从整 chunk
    改为实体邻句窗口。

    诊断实证：三元组路径仅占三元组 context 的 2.6% 长度，其余 97.4%
    为绑定的整 chunk 原文依据，其中大量与三元组语义弱相关——12 条
    预算被噪声占满，CP 被稀释且 CR 无增益。

    定位采用 jieba 分词打分制：图谱实体多为规范化短语（如
    "面板指示灯同时亮"），substring 全匹配常失配（原文为
    "…指示灯同时亮"）；按词命中数加权选种子句最稳。
    种子句各取前后 1 句组成窗口；完全零命中时回退取前 max_len 字。
    """
    if len(text) <= max_len:
        return text
    sents = _split_sents(text)

    words = set()
    for e in (entities or []):
        if not e:
            continue
        for w in jieba.cut(str(e)):
            if len(w) >= 2:
                words.add(w)
    code_toks = _query_code_tokens(query_str or "") if query_str else []
    for t in code_toks:
        words.add(t)

    best_idx, best_score = -1, 0
    for i, s in enumerate(sents):
        sc = sum(len(w) for w in words if w in s)
        if sc > best_score:
            best_idx, best_score = i, sc

    if best_idx < 0:
        return text[:max_len]

    picked = {best_idx}
    total = len(sents[best_idx])
    left, right = best_idx - 1, best_idx + 1
    while total < hard_cap and (left >= 0 or right < len(sents)):
        take_l = left >= 0 and (
            right >= len(sents) or len(sents[left]) <= len(sents[right]))
        if take_l:
            picked.add(left); total += len(sents[left]); left -= 1
        else:
            picked.add(right); total += len(sents[right]); right += 1
    return "".join(sents[i] for i in sorted(picked))


_GENERIC_PHRASES = {
    "故障现象", "可能原因", "解决方法", "应该怎么", "应该如何处理", "如何排查",
    "数控机床", "加工中心", "维修手册", "故障诊断", "维修方法", "是什么",
}

# 查询-图谱同义词归一化（2026-08-11）：
# 问题表述与图谱实体存在常用同义差异时，短语匹配会失败（如问题"伺服电机"
# 图谱"伺服电动机"，4 字窗口切出残缺短语无法命中路径），导致相关路径
# 在 _collect_graph_contexts 中排序失效、被挤出 12 槽（auto_023 实证 CR=0.33）。
# 归一化方向以图谱实体为准：图谱 55 个"电动机" vs 5 个"测速发电机"（无歧义），
# 故问题侧的"电机"统一归一为"电动机"。归一化在短语提取前应用，只影响匹配。
_QUERY_SYNONYMS = (
    ("伺服电机", "伺服电动机"),
)


def _norm_query_synonyms(text: str) -> str:
    """问题文本同义词归一化（长词优先，避免"电机"先替换"电动机"产生残缺）。"""
    for long, short in sorted(_QUERY_SYNONYMS, key=lambda x: -len(x[0])):
        text = text.replace(long, short)
    return text


def _query_phrase_tokens(query_str: str, path_contexts: list) -> list:
    """提取问题中的中文关键短语（4-6 字），且至少出现在一条图谱路径中。

    2026-08-11：①查询同义词归一化（问题"伺服电机"→图谱"伺服电动机"）；
    ②放宽"1~3 条"过滤为"1~15 条"——领域核心短语（如"伺服电动机不转"
    命中 10+ 条路径）此前被过滤导致 tokens 为空、路径排序失效、答案路径
    被挤出 12 槽（auto_023 实证 CR=0.33）。仅超高频泛化短语（>15 条命中，
    如"系统显示"）仍被过滤。
    2026-09-06（GSK983 适配）：固定上限 15 与图谱规模耦合——Manual 现象实体
    按 per_depth 逐跳展开（每实体最多 16 条路径），核心短语"超程报警"命中
    28/152 条仍属"精确"（都指向 OT报警 现象族），却因 >15 被滤掉 → 现象式
    问法（"超程报警如何复位"）tokens 空、语义路径永远进不了 context。
    上限改为相对值：max(15, 0.35×路径总数)，泛化短语（500cases 数百条命中）
    仍被滤，中规模图谱的核心短语得以保留。
    """
    if not path_contexts:
        return []
    query_norm = _norm_query_synonyms(query_str)
    runs = re.findall(r"[\u4e00-\u9fff]+", query_norm)
    cands = []
    for run in runs:
        for n in (4, 5, 6):
            cands.extend(run[i:i + n] for i in range(0, max(1, len(run) - n + 1)))
    counts = {}
    for p in dict.fromkeys(cands):
        if len(p) < 4 or p in _GENERIC_PHRASES:
            continue
        counts[p] = sum(1 for path in path_contexts if p in path)
    out = []
    hit_cap = max(15, int(len(path_contexts) * 0.35))
    # 保留命中 1~hit_cap 条的具体短语（按命中数升序：更具体的短语优先）
    for p, n in sorted(counts.items(), key=lambda kv: (kv[1], -len(kv[0]))):
        if 1 <= n <= hit_cap:
            out.append(p)
    return out[:30]


def _context_relevance_score(question: str, contexts: list) -> float:
    """计算 context 与问题的相关度分数（0.0~1.0）。

    用于 OOC 拒答：当所有 context 都与问题无关时，直接拒答而非强行生成答案。

    策略：
      1. 代码 token（报警号/参数号/地址等）：完全匹配 → 每条 +0.3
      2. 中文关键词（2-6字短语）：出现在 context 中 → 长短语 +0.2，短短语 +0.1
      3. 归一化到 [0, 1]，超过阈值视为有相关信息

    Returns:
        float: 0.0=完全无关，1.0=高度相关
    """
    if not contexts:
        return 0.0

    ctx_text = " ".join(contexts)

    # 代码 token 匹配（权重高，代码是强信号）
    code_tokens = _query_code_tokens(question)
    code_hits = sum(1 for t in code_tokens if t in ctx_text)
    code_score = min(1.0, code_hits * 0.3) if code_tokens else 0.0

    # 中文关键词匹配（权重较低，容易误匹配）
    # 提取问题中的中文短语（2-6字），排除通用词
    _OOC_GENERIC = {
        "故障现象", "可能原因", "解决方法", "应该怎么", "应该如何",
        "如何排查", "数控机床", "加工中心", "维修手册", "故障诊断",
        "是什么", "什么原因", "怎么处理", "如何维修", "有什么",
        "有哪些", "怎么办", "是什么",
    }
    runs = re.findall(r"[\u4e00-\u9fff]+", question)
    key_phrases = []
    for run in runs:
        if len(run) < 2:
            continue
        for n in (4, 5, 6, 3, 2):
            for i in range(0, max(1, len(run) - n + 1)):
                phrase = run[i:i + n]
                if phrase not in _OOC_GENERIC and len(phrase) >= 2:
                    key_phrases.append(phrase)
    key_phrases = list(dict.fromkeys(key_phrases))[:15]  # 去重，最多 15 条

    # 短短语（2-3字）权重低于长短语（4-6字）
    phrase_hit_score = 0.0
    for p in key_phrases:
        if p in ctx_text:
            phrase_hit_score += 0.2 if len(p) >= 4 else 0.1
    phrase_score = min(1.0, phrase_hit_score) if key_phrases else 0.0

    # 综合分数：代码权重 0.6 + 短语权重 0.4
    if code_tokens and key_phrases:
        return 0.6 * code_score + 0.4 * phrase_score
    elif code_tokens:
        return code_score
    elif key_phrases:
        return phrase_score
    return 0.0


# OOC 拒答阈值：低于此分数视为 context 与问题无关，直接拒答
# 0.1 = 至少有一个关键词（代码或中文短语）在 context 中出现
_OOC_RELEVANCE_THRESHOLD = 0.1

# OOC 对象一致性检查（2026-08-11）：
# 问题包含"领域对象词"（如工业机器人/磨床/电火花），而检索上下文完全
# 不含该对象的词族 → 语义邻近但对象不符，判定 OOC 拒答。
# 背景：auto_010/040/050（工业机器人重复定位/关节电机过载）被图谱检索到
# 高度字面相关的数控机床上下文（"位置检测系统""电机过载报警"等），
# relevance 分数越过 0.1 阈值，模型基于机床知识编造机器人答案。
# 对象词族与图谱域（数控机床）不同族时，即便字面 token 高度重叠也不应放行。
_OOC_OBJECT_FAMILIES = (
    ("机器人", ("机器人", "机械臂", "关节", "末端执行器", "示教")),
    ("磨床", ("磨床", "磨削", "砂轮", "外圆磨", "磨削液")),
    ("电火花", ("电火花", "线切割", "电极丝", "EDM", "放电加工")),
    ("激光加工", ("激光切割", "激光焊接", "激光加工")),
)


def _ooc_object_mismatch(question: str, contexts: list) -> bool:
    """问题对象词族与上下文对象词族不一致 → True（判定为 OOC）。

    规则：
      - 提取问题中命中的对象词族（最多取一个最强族）；
      - 若上下文完全不含该词族的任何词 → 对象不一致。
    仅在 OOC 题判定路径使用（不影响 in-context 题）。
    """
    if not contexts:
        return False
    ctx_text = "".join(str(c) for c in contexts)
    # 问题命中的对象族（按词表顺序取第一个）
    hit_family = None
    for fam_name, words in _OOC_OBJECT_FAMILIES:
        if any(w in question for w in words):
            hit_family = (fam_name, words)
            break
    if hit_family is None:
        return False
    fam_name, words = hit_family
    # 上下文出现任一对象词 → 一致
    if any(w in ctx_text for w in words):
        return False
    # 上下文完全是机床域（数控机床/车床等）而问题是机器人/磨床等 → 不一致
    return True


def _should_refuse_ooc(question, graph_contexts, relevance):
    """OOC 拒答判定：相关度门 + 对象一致性门。

    消融 B3（ABLATE_OOC=1）只关对象一致性门，相关度门保持；
    返回 (refuse, ooc_mismatch)，mismatch 带出供调用方写拒答原因，
    避免 process 内重复计算 _ooc_object_mismatch。
    """
    ooc_mismatch = False
    if not ABLATE.ablate_ooc:
        ooc_mismatch = _ooc_object_mismatch(question, graph_contexts)
    return (relevance < _OOC_RELEVANCE_THRESHOLD) or ooc_mismatch, ooc_mismatch


# ════════════════════════════════════════════════════════════════════
# OOC 门层打分补丁（2026-10-04，OOC 词法线 D5；只动打分层，检索排序零接触）
#
# E1/E2 实测依据（reports/ooc_lex_baseline.{md,json}、ooc_lex_e2_attr.{md,json}，
# 冻结快照 sha8 5f118a0b）：serve.py 把 OOC 门套在所有用户问句上，228 道 in-domain
# 规范题被拒 93（FPR 0.408）、口语题被拒 145（0.636）。三层机制（A 数字-称谓分离/
# B 前导零 int 归一化丢弃/C 短语 [:15] 截断＋按出场顺序被插入段挤占）＋权重混通
# （0.6c+0.4p：纯短语题加空代码通道被稀释成 0.4p）。
#
# 本补丁只新增打分层入口 `_ooc_gate_relevance`（serve 与批测 OOC 分支共用）；
# `_query_code_tokens`（喂检索排序）与 `_context_relevance_score`（原函数保留）
# 一个字不动 ⇒ 判分链 ctx_sha 咬合与历史评测可比性不受影响。回滚＝删除本区块
# ＋两个调用点改回原函数。
# ════════════════════════════════════════════════════════════════════

#: 门层通用词表（与 _context_relevance_score 内的 _OOC_GENERIC 同一份词；那里是
#: 函数局部变量取不到，故此处复制一份——两处若需增删须同步，见各自注释）
_OOC_GATE_GENERIC = {
    "故障现象", "可能原因", "解决方法", "应该怎么", "应该如何",
    "如何排查", "数控机床", "加工中心", "维修手册", "故障诊断",
    "是什么", "什么原因", "怎么处理", "如何维修", "有什么",
    "有哪些", "怎么办",
}
#: 修复①（机制 C 面）：短语槽 15→40（E2 预演 fixC=106/145）
_OOC_GATE_PHRASE_CAP = 40
#: 弱证据门槛（v2，10-04 护栏迭代）：v1（fixABC 直译版）E3 实测护栏破——10 道通用
#: OOC 题靠 1~2 个两字小词的巧合命中溜过门（OOC 拒答 45→35）。v2 把证据分级：
#: 强证据（代码 token／≥4 字短语命中）单独即可放行；**只有弱证据（2~3 字小词）时
#: 须 ≥ _OOC_GATE_WEAK_MIN 个不同弱命中**才放行——1~2 个两字巧合按噪声拒。
_OOC_GATE_WEAK_MIN = 3


def _ooc_gate_tokens(question: str):
    """OOC 门层专用 token/词组抽取（只喂拒答门，绝不喂检索排序）。

    修复②（机制 A 面）：独立数字串（2–4 位、前后不贴字母数字/小数点）进 token 通道
    ——口语"405 这个报警/9102 报警"数字与'报警'分离后，_ALARM_PLAIN_NUM_RE 两支
    均不命中，原打分器代码通道归零。
    修复③（机制 B 面）：报警号保留前导零原串——"报警001"被 int 归一化成 '1' 后
    被 ≥2 位规则丢弃；此处把原始串 '001' 直接入通道（不做数值化）。
    修复①（机制 C 面）：词组仍全量抽出，但**长词组优先**排序（内容优先，不再按
    出场顺序被插入段挤占），槽 15→40。
    返回 (code_tokens, phrases)。
    """
    code_tokens = list(dict.fromkeys(_query_code_tokens(question)))
    for m in re.finditer(r"(?<![A-Za-z0-9.#])(\d{2,4})(?!\d)", question):
        if m.group(1) not in code_tokens:
            code_tokens.append(m.group(1))
    for m in _ALARM_PLAIN_NUM_RE.finditer(question):
        raw = m.group(1) if m.group(1) is not None else m.group(2)
        if raw not in code_tokens:
            code_tokens.append(raw)
    phrases = []
    for run in re.findall(r"[\u4e00-\u9fff]+", question):
        if len(run) < 2:
            continue
        for n in (4, 5, 6, 3, 2):
            for i in range(0, max(1, len(run) - n + 1)):
                p = run[i:i + n]
                if p not in _OOC_GATE_GENERIC and len(p) >= 2 and p not in phrases:
                    phrases.append(p)
    phrases.sort(key=lambda p: -len(p))
    return code_tokens, phrases[:_OOC_GATE_PHRASE_CAP]


def _ooc_gate_evidence(question: str, contexts: list):
    """门层证据分级（v2）：返回 (code_score, strong_score, weak_hits)。

    强证据＝代码 token 命中／≥4 字短语命中；弱证据＝2~3 字短语命中（不同词计数）。
    """
    if not contexts:
        return 0.0, 0.0, 0
    ctx_text = " ".join(contexts)
    code_tokens, phrases = _ooc_gate_tokens(question)
    code_hits = sum(1 for t in code_tokens if t in ctx_text)
    code_score = min(1.0, code_hits * 0.3) if code_tokens else 0.0
    strong_hits = sum(1 for p in phrases if len(p) >= 4 and p in ctx_text)
    strong_score = min(1.0, strong_hits * 0.2) if strong_hits else 0.0
    weak_hits = sum(1 for p in phrases if len(p) <= 3 and p in ctx_text)
    return code_score, strong_score, weak_hits


def _ooc_gate_relevance(question: str, contexts: list) -> float:
    """OOC 门层相关度（分层补丁的打分层入口 v2；serve 与批测 OOC 分支共用）。

    权重**分通道结算＋证据分级**：
      rel = max(code_score, strong_score,
                0.1×weak_hits  若 weak_hits ≥ _OOC_GATE_WEAK_MIN  否则 0)
    ——修掉 0.6c+0.4p 的权重混通（E2：纯短语题加空代码通道被稀释成 0.4p）；
    ——修掉 v1 护栏破（E3 首跑：1~2 个两字巧合命中放跑了 10 道通用 OOC）。
    强证据（代码 token／≥4 字短语）单独过线即视为相关；阈值不变（0.1）。
    """
    code_score, strong_score, weak_hits = _ooc_gate_evidence(question, contexts)
    weak_score = 0.1 * weak_hits if weak_hits >= _OOC_GATE_WEAK_MIN else 0.0
    return max(code_score, strong_score, weak_score)


# ── 口语措辞放行票（2026-10-28，作者令①）─────────────────────────────
# 相关度词法尺（代码 token／≥4 字短语／≥3 个 2~3 字弱命中）对口语题系统性
# 0.00 误拦（10-27/28 口语卷实验：口语题拒答 16、考卷同批 1；词法尺漏计
# CRT/FANUC 等拉丁词是直接原因）。本票只撤"相关度门单独开火"的拒答：
# 题面非通用拉丁词（品牌/专用缩写 ≥2 字母）在检索路径文本命中
# ≥ OOC_RELSEM_LATIN_MIN ⇒ 放行。对象不一致票不撤；句子票/联合门照常在后。
# 分离度实测（_relsem_probe2.py + 开火门分账）：救 4/6 相关度门误拦，
# 漏 0/37 相关度门开火的超纲题。嵌入语义尺（整题/分句↔路径余弦）
# AUC 0.647/0.630 判负，未采用（探针 relsem_probe_rows/2_rows.json）。
_RELSEM_GENERIC = {'PLC', 'CNC', 'NC', 'CRT', 'CPU', 'RAM', 'ROM', 'LED', 'LCD'}
_RELSEM_LATIN_RE = re.compile(
    r"(?<![A-Za-z0-9])([A-Za-z][A-Za-z0-9]{1,7})(?![A-Za-z0-9])")


def relsem_latin_hits(question: str, contexts: list) -> int:
    """题面非通用拉丁词在检索上下文中的命中数（去重、大小写不敏感）。"""
    if not contexts:
        return 0
    ctx = " ".join(contexts).upper()
    toks = {t.upper() for t in _RELSEM_LATIN_RE.findall(question)} - _RELSEM_GENERIC
    return sum(1 for t in toks if t in ctx)


def relsem_release_hook(question, contexts, refusal_reason):
    """放行票：撤销"相关度门单独开火"的拒答（serve 与批测同入口同行为）。

    开关 OOC_RELSEM=1（默认 0＝零行为改变）。条件：
      · refusal_reason 恰为相关度门形状（含 relevance，无 sent_cov/joint_gate）；
      · object_mismatch 为假（对象不一致票不归它撤）；
      · relsem_latin_hits ≥ OOC_RELSEM_LATIN_MIN（默认 1）。
    返回 (refusal_reason_after, record|None)。任何异常原样返回（降级现门）。
    """
    try:
        if os.getenv("OOC_RELSEM", "0") != "1" or not refusal_reason:
            return refusal_reason, None
        if ('relevance' not in refusal_reason
                or refusal_reason.get('object_mismatch')
                or any(k in refusal_reason
                       for k in ('sent_cov', 'joint_gate', 'vendor_gate'))):
            return refusal_reason, None
        try:
            tmin = int(os.getenv("OOC_RELSEM_LATIN_MIN", "1"))
        except ValueError:
            tmin = 1
        n = relsem_latin_hits(question, contexts or [])
        if n >= tmin:
            rec = {'relsem_release': {'latin_hits': n,
                                      'relevance': refusal_reason.get('relevance')}}
            logger.info("relsem 放行票: %s", json.dumps(rec, ensure_ascii=False))
            return None, rec
        return refusal_reason, None
    except Exception as e:                                      # noqa: BLE001
        logger.warning("relsem 钩子异常（降级现门）: %s", e)
        return refusal_reason, None


# ════════════════════════════════════════════════════════════════════
# 引用防线（2026-10-06，OOC 词法线 D6；预注册件
# 半月reports/底座切换与引用防线预注册与预估单_20261006.md §2.4 G1–G6）
#
# 为什么要新增这一层而不是调相关度阈值：追加八十把五种免费仪器全试了一遍，
# "这道题的答案在不在被喂的这本书里"最好的一把尺（句子级嵌入覆盖）也只有
# B1|B2＝0.580，且含答案那句排进 top5 只有 29~32% ⇒ 天花板在句子级定位本身，
# 不在参数量。相关度门量不到"书里没有"，但**引用**量得到：答案挂不上原文，
# 就等于这本书没写过（挂得上则不必知道它在不在）。
#
# G1 硬约束：引用串**现场**从被喂的手册正文子串检索得到，禁止复用
# `entity_chunk_map`——实测那条通道在生产底座上命中率 5.8%（随机基线 2.1%，
# 见 files/_chunk_binding_audit_20261006.py），等于噪声。
#
# 回滚＝删除本区块＋调用点改回原样。本层只读文本、不碰检索排序。
#
# 状态（2026-10-06 晚）：**本线已按预注册 §5.3 判读线终止，无生产调用点**——
# v1 的 G2/G3 红、v2 的 V2-a/V2-b 红。此区块仅为 `files/_citation_defense_eval_20261006.py`
# 与 `files/_citation_defense_v2_20261006.py` 两份判读件的判据单源而保留，
# **不得当作线上能力引用**，也不得在没写新预注册之前挂 serve.py 调用点。
# ════════════════════════════════════════════════════════════════════

#: 引用成立所需的最短连续原文串字数（与诊断件 _anchor_diag_armC.NARROW 同尺，
#: 保证"能否挂上原文"与"答案在不在书里"用的是同一把量尺）
_CITE_MIN_RUN = 10

#: 挂不上原文时的对外话术（工业侧要求：明说没写过，不要硬答，也不要假称检索失败）
_CITE_REFUSAL = "这本手册没有记载相关内容，无法凭本手册回答。"

_CITE_SENT_SPLIT = re.compile(r"[。！？!?；;\n]")


def _cite_norm(s: str) -> str:
    """去空白归一（手册正文与答案句同一处理，否则缩进/换行会打断连续串）。"""
    return re.sub(r"\s+", "", str(s or ""))


def _cite_longest_run(sentence_norm: str, book_norm: str,
                      min_run: int = _CITE_MIN_RUN) -> int:
    """答案句在手册正文里能找到的最长连续原文串长度（找不到返回 0）。

    从长到短滑窗：首个命中即最长，代价 O(len(sentence)) 次子串查。
    """
    n = len(sentence_norm)
    for w in range(n, min_run - 1, -1):
        for i in range(0, n - w + 1):
            if sentence_norm[i:i + w] and sentence_norm[i:i + w] in book_norm:
                return w
    return 0


def split_fact_sentences(answer: str) -> list:
    """把答案切成待核单句，丢掉空段与纯符号段（只有标点不构成可核对的断言）。"""
    out = []
    for piece in _CITE_SENT_SPLIT.split(str(answer or "")):
        s = piece.strip(" \t\r,，:：-—*#.、")
        if len(_cite_norm(s)) >= 4 and re.search(r"[\u4e00-\u9fffA-Za-z0-9]", s):
            out.append(s)
    return out


def answer_citation(answer: str, book_text: str,
                    min_run: int = _CITE_MIN_RUN) -> dict:
    """逐句核引用：每句能否在被喂的手册正文里找到 ≥min_run 字连续原文串。

    返回 dict（纯函数，不落字段外的副作用）：
      n_sentences / n_grounded / grounded_ratio / longest_run /
      ungrounded（挂不上原文的句子，供上层决定"删句"还是"整题拒答"）
      first_grounded（首句是否挂上；v2 的"答案主体"判据用，2026-10-06 追加）
    """
    book_norm = _cite_norm(book_text)
    sents = split_fact_sentences(answer)
    grounded, ungrounded, longest = [], [], 0
    first_grounded = False
    for k, s in enumerate(sents):
        r = _cite_longest_run(_cite_norm(s), book_norm, min_run)
        if r >= min_run:
            grounded.append(s)
            longest = max(longest, r)
            if k == 0:
                first_grounded = True
        else:
            ungrounded.append(s)
    return {'n_sentences': len(sents), 'n_grounded': len(grounded),
            'grounded_ratio': round(len(grounded) / float(len(sents)), 4) if sents else 0.0,
            'longest_run': longest, 'ungrounded': ungrounded,
            'min_run': min_run, 'first_grounded': first_grounded}


def answer_subject_grounded(citation: dict) -> bool:
    """v2 主判据（预注册 §5.2 R2）：答案主体＝它开头那一句，主体无据＝整答无据。

    与 should_refuse_uncited 的差别只在"看哪一句"：旧的是"任一句挂上即算有证"，
    这条是"首句必须挂上"。两者阈值同值同报，不许事后择优（§5.2 选定纪律）。
    """
    return bool(citation.get('first_grounded'))


def should_refuse_by_subject(citation: dict) -> bool:
    """整题级拒答判定（v2 口径）：首句挂不上原文 ⇒ 明说这本手册没写过。"""
    return not answer_subject_grounded(citation)


def should_refuse_uncited(citation: dict, min_grounded: int = 1) -> bool:
    """整题级引用拒答：挂上原文的句子数 < min_grounded ⇒ 拒答。

    默认只要有 1 句挂得上就不整题拒（工业侧宁可少拒）；判据 G2/G3/G4 用默认值，
    逐句删除（部分引用）留给上层，两种口径都读数，不在此混。
    """
    return int(citation.get('n_grounded', 0)) < min_grounded


# ════════════════════════════════════════════════════════════════════
# 数值化评估闭环（借鉴 kg-rag 的 extract_number 判分，适配中文单位）
# 与 RAGAS 四指标正交：RAGAS 做语义判分，这里做"数值/范围/比较符"的
# 机器精确判分，绕开语义判分的主观性。写入 per_question.metrics 的
# numeric_accuracy 字段（1.0/0.0），无法解析时返回 None 不计入。
# ════════════════════════════════════════════════════════════════════

@dataclass
class NumericExpectation:
    operator: str            # ">", "<", ">=", "<=", "=", "range"
    low: Optional[float] = None
    high: Optional[float] = None
    unit: str = ""
    raw: str = ""


def parse_numeric_expectation(gt: str) -> Optional[NumericExpectation]:
    """解析数值型 ground_truth。返回 None = 非数值型（留给 RAGAS）。

    支持：>80℃、0.02～0.06mm、1/3至2/3、25000小时、≥10滴 等中文单位格式。
    """
    if not gt:
        return None
    gt = gt.strip()
    # 1/3至2/3 分数范围
    m = re.match(r"^\s*(\d+)\s*/\s*(\d+)\s*至\s*(\d+)\s*/\s*(\d+)\s*", gt)
    if m:
        return NumericExpectation(operator="range",
                                  low=int(m.group(1)) / int(m.group(2)),
                                  high=int(m.group(3)) / int(m.group(4)),
                                  unit="", raw=gt)
    # 范围 a～b / a~b / a至b
    m = re.match(r"^\s*([\d.]+)\s*[～~至到]\s*([\d.]+)\s*([^\d\s]*)\s*$", gt)
    if m:
        return NumericExpectation(operator="range",
                                  low=float(m.group(1)), high=float(m.group(2)),
                                  unit=m.group(3) or "", raw=gt)
    # 比较符 + 数值 + 单位
    m = re.match(r"^\s*(>=|<=|>|<|≥|≤|=)?\s*([\d.]+)\s*([^\d\s]*)\s*$", gt)
    if m:
        op = m.group(1) or "="
        op = {"≥": ">=", "≤": "<="}.get(op, op)
        return NumericExpectation(operator=op, low=float(m.group(2)),
                                  unit=m.group(3) or "", raw=gt)
    return None


def extract_number_from_answer(answer: str) -> Optional[float]:
    """从答案中提取首个数值（含小数/负数）。"""
    if not answer:
        return None
    m = re.search(r"-?\d+(?:\.\d+)?", answer)
    return float(m.group()) if m else None


def numeric_accuracy(answer: str, gt: str) -> Optional[float]:
    """数值判分：1.0/0.0；gt 非数值型或 answer 无数值 → None（不计入）。"""
    exp = parse_numeric_expectation(gt)
    if exp is None:
        return None
    val = extract_number_from_answer(answer)
    if val is None:
        return None
    if exp.operator == ">":
        return 1.0 if val > exp.low else 0.0
    if exp.operator == ">=":
        return 1.0 if val >= exp.low else 0.0
    if exp.operator == "<":
        return 1.0 if val < exp.low else 0.0
    if exp.operator == "<=":
        return 1.0 if val <= exp.low else 0.0
    if exp.operator == "range":
        return 1.0 if exp.low <= val <= exp.high else 0.0
    # "="：单位相同才判等（℃ vs mm 不可比）；省略单位比较时仅数值判等
    return 1.0 if abs(val - exp.low) < 1e-9 else 0.0


def _agg_numeric_accuracy(results: dict) -> Optional[float]:
    """聚合数值命中率：per_question 中可判分题的平均命中率；无样本 → None。"""
    if not results or not results.get("per_question"):
        return None
    vals = [r.get("metrics", {}).get("numeric_accuracy")
            for r in results["per_question"]
            if (r.get("metrics") or {}).get("numeric_accuracy") is not None]
    return sum(vals) / len(vals) if vals else None


def _num_samples(results: dict) -> int:
    """数值型题中可判分的样本数（numeric_accuracy 非 None）。"""
    if not results or not results.get("per_question"):
        return 0
    return sum(1 for r in results["per_question"]
               if (r.get("metrics") or {}).get("numeric_accuracy") is not None)


def _fmt_num(v) -> str:
    """报告数字格式化：None → 'N/A'，否则保留 2 位小数。"""
    if v is None:
        return "N/A"
    return f"{v:.2f}"


# ════════════════════════════════════════════════════════════════════
# 领域语义邻近度拒答门（2026-10-10；预注册：
# 半月reports/领域语义邻近度拒答门接入生产_预注册与判读表_20261010.md §1/§3/§4）
#
# 名字纪律（P7 教训，台账追加九十五）：这把尺量的是"问题离这本书的领域多近"，
# 不是"书里有没有讲过这事"（砸碎句库 AUC 0.967→0.951 没塌 ⇒ 句子级语义覆盖的说法
# 已被实测否掉）。在册读数：留出正样本 130 误拦 4、跨书 211 拦住 180＝0.853、
# 旧 60 拦住 32；盲区＝同书不可答 23 道（命中 0）。
# 本区块只新增函数，现有函数一字不动（与 10-04 D5 补丁同款纪律）；回滚＝删本区块
# ＋ serve.py 的钩子调用点。影子开关默认关 ⇒ 接线后生产行为零改变。
# ════════════════════════════════════════════════════════════════════

import io

#: T 冻结（gen2 登记处 files/_sentcov_gen2_20261010.json 的 T_gen2；gen1 0.694 挂旧仪器留档）。
#: 比较口径与在册仪器逐字一致＝score <= T（电池件 P1 即按 <= 计数）。
_SENTCOV_T = 0.6956
_SENTCOV_EMBED_TIMEOUT_S = 3.0

_SENTCOV_GATE = None          # 懒加载单例（SentCoverGate 或 _SentCovDisabled）
_SENTCOV_FAIL_COUNT = 0       # 降级/失败计数：影子期报警可见性的最低要求


class _SentCovDisabled:
    """库缺失/指纹不符/加载失败时的常驻降级对象：永远 ok=False，绝不抛异常。"""

    def __init__(self, reason):
        self.reason = reason

    def score(self, question, embed_fn, timeout_s=_SENTCOV_EMBED_TIMEOUT_S):
        return None, self.reason


class SentCoverGate:
    """常驻句向量矩阵＋指纹校验。score 返回 (score|None, reason|None)。"""

    def __init__(self, manual_name):
        import numpy as np
        from config import LOCAL_EMBEDDING_MODEL
        self.np = np
        self.manual_name = manual_name
        self.model_name = LOCAL_EMBEDDING_MODEL
        stem = os.path.splitext(manual_name)[0]
        lib_dir = os.path.join(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))), "data", "vectors")
        self.bin_path = os.path.join(lib_dir, "%s_sentcov_f32.bin" % stem)
        self.fp_path = os.path.join(lib_dir, "%s_sentcov_f32_fingerprint.json" % stem)
        self.ok = False
        self.reason = None
        self.dim = None
        self._mat = None            # 单位化后的句向量矩阵（n×dim）
        self._load()

    def _load(self):
        np = self.np
        try:
            if not (os.path.isfile(self.bin_path) and os.path.isfile(self.fp_path)):
                self.reason = "library_missing"
                return
            with io.open(self.fp_path, encoding="utf-8") as f:
                fp = json.load(f)
            # W1 指纹对拍：模型＋书名必须逐字一致（换书 ⇒ 降级，宁可退回现门）
            if fp.get("model") != self.model_name or fp.get("book") != self.manual_name:
                self.reason = "fingerprint_mismatch(model=%s book=%s)" % (
                    fp.get("model"), fp.get("book"))
                return
            raw = np.fromfile(self.bin_path, dtype=np.float32)
            n, dim = int(fp.get("n_sents") or 0), int(fp.get("dim") or 0)
            if not n or not dim or raw.size != n * dim:
                self.reason = "bin_shape_mismatch(size=%d expect=%d)" % (raw.size, n * dim)
                return
            mat = raw.reshape(n, dim)
            norms = np.linalg.norm(mat, axis=1)
            norms[norms == 0] = 1.0
            self._mat = mat / norms[:, None]
            self.dim = dim
            self.ok = True
        except Exception as e:                              # noqa: BLE001
            self.reason = "load_error:%s:%s" % (type(e).__name__, str(e)[:120])

    def score(self, question, embed_fn, timeout_s=_SENTCOV_EMBED_TIMEOUT_S):
        """返回 (sent_cov|None, reason|None)。计算式与在册电池逐字同源：
        unit(norm(q)[:600]) @ unit(句库).T 取 max。"""
        if not self.ok:
            return None, self.reason
        q = re.sub(r"\s+", "", str(question or ""))[:600]
        if not q:
            return None, "empty_question"
        box = {}

        def _run():
            try:
                box["v"] = embed_fn(q)
            except Exception as e:                          # noqa: BLE001
                box["err"] = e

        th = threading.Thread(target=_run, daemon=True)
        th.start()
        th.join(timeout_s)
        if th.is_alive():
            return None, "embed_timeout"
        if "err" in box:
            return None, "embed_error:%s" % type(box["err"]).__name__
        v = box.get("v")
        v = v["embedding"] if isinstance(v, dict) else v
        vec = self.np.asarray(v, dtype="float32")
        if vec.ndim != 1 or vec.shape[0] != self.dim:
            return None, "query_dim_mismatch(%s expect %d)" % (vec.shape, self.dim)
        nq = float(self.np.linalg.norm(vec))
        if nq == 0.0:
            return None, "zero_query_vector"
        sims = self._mat @ (vec / nq)
        return float(self.np.max(sims)), None


def _get_sent_gate():
    """懒加载单例。失败也缓存为 _SentCovDisabled（进程内不再反复试坏文件）。"""
    global _SENTCOV_GATE, _SENTCOV_FAIL_COUNT
    if _SENTCOV_GATE is None:
        manual = os.getenv("SERVING_MANUAL", "13_GSK983数控系统维修手册_中文.md")
        _SENTCOV_GATE = SentCoverGate(manual)
        if not _SENTCOV_GATE.ok:
            _SENTCOV_FAIL_COUNT += 1
            logger.warning("领域语义邻近度门降级：%s（现门接管，不影响服务）",
                           _SENTCOV_GATE.reason)
    return _SENTCOV_GATE


_SENTCOV_EMBED_FN = None


def _sentcov_embed_fn():
    """批测侧共用的懒加载嵌入句柄（与 config 同源单例；serve 用自己的 self.embed）。"""
    global _SENTCOV_EMBED_FN
    if _SENTCOV_EMBED_FN is None:
        from config import get_embed_model
        _SENTCOV_EMBED_FN = get_embed_model().get_text_embedding
    return _SENTCOV_EMBED_FN


def sent_cov_refuse(question, contexts, embed_fn, threshold=None):
    """新票打分：返回 dict(ok, score, b_refuse, c_refuse, reason)。

    b_refuse＝纯替换接法（score <= T）；c_refuse＝b 或 对象核对分支（读检索上下文）。
    库不可用时 ok=False 且两票都 False（调用方必须退回现门，不许猜）。

    threshold=None ⇒ 用代码常量 _SENTCOV_T。2026-10-27 前，钩子把
    `.env` 的 OOC_SENT_COVER_T 解析成浮点后只判"能不能转"就丢弃，
    判定始终走常量 ⇒ 那是个死旋钮（改 .env 什么都不发生）。现在 env 值真消费，
    并在 rec 里回显生效阈值。今天两者数值相同（0.6956）⇒ 零行为改变。
    """
    global _SENTCOV_FAIL_COUNT
    gate = _get_sent_gate()
    score, reason = gate.score(question, embed_fn)
    if score is None:
        _SENTCOV_FAIL_COUNT += 1
        return {"ok": False, "score": None, "b_refuse": False,
                "c_refuse": False, "reason": reason}
    t = _SENTCOV_T if threshold is None else float(threshold)
    b = score <= t
    c = b or (not ABLATE.ablate_ooc and _ooc_object_mismatch(question, contexts))
    return {"ok": True, "score": score, "b_refuse": b, "c_refuse": c,
            "reason": None, "T_effective": t}


def sent_cov_gate_hook(question, graph_contexts, refusal_reason, embed_fn):
    """serve 拒答门钩子（预注册 §3：影子先行，默认零行为改变）。

    返回 (refusal_reason_after, record|None)。
      · OOC_SENT_SHADOW=1            ⇒ 只打分落 logs/ooc_shadow_*.jsonl，不改决策；
      · OOC_SENT_COVER_T＝浮点 且 OOC_SENT_MODE ∈ {B,C}
                                       ⇒ 升格真票（激活即改可见行为，须作者显式设）；
      · 两者都没设                   ⇒ 直接返回，连分都不算（生产默认路径零开销）。
    任何异常都不许往上抛（调用方还有 try/except 兜底，但本函数自己先兜）。"""
    global _SENTCOV_FAIL_COUNT
    shadow = os.getenv("OOC_SENT_SHADOW", "") == "1"
    cover_raw = os.getenv("OOC_SENT_COVER_T", "")
    if not shadow and not cover_raw:
        return refusal_reason, None
    # env 阈值在此解析并真的送进判定（旧写法解析完就丢 ⇒ 死旋钮）
    t_env = None
    if cover_raw:
        try:
            t_env = float(cover_raw)
        except ValueError:
            logger.error("OOC_SENT_COVER_T 不是浮点（%r）⇒ 本次退回代码常量 %s，"
                         "但请尽快修正 .env", cover_raw, _SENTCOV_T)
    rec = {"ts": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
           "q_hash": hashlib.sha256(str(question).encode("utf-8")).hexdigest()[:16],
           "legacy_refuse": refusal_reason is not None, "T": _SENTCOV_T,
           "T_env": t_env}
    try:
        r = sent_cov_refuse(question, graph_contexts, embed_fn, threshold=t_env)
        rec["T_effective"] = r.get("T_effective")
        rec.update({"ok": r["ok"], "sent_cov": (round(r["score"], 6)
                                                if r["score"] is not None else None),
                    "b_refuse": r["b_refuse"], "c_refuse": r["c_refuse"],
                    "reason": r["reason"]})
        if r["ok"] and cover_raw:
            t = t_env
            mode = os.getenv("OOC_SENT_MODE", "")
            if t is not None and mode in ("B", "C"):
                new_vote = r["b_refuse"] if mode == "B" else r["c_refuse"]
                rec["mode"] = mode
                rec["activated_T"] = t
                if new_vote and refusal_reason is None:
                    refusal_reason = {"relevance": None,
                                      "sent_cov": round(r["score"], 6),
                                      "sent_mode": mode}
    except Exception as e:                                  # noqa: BLE001
        _SENTCOV_FAIL_COUNT += 1
        rec.update({"ok": False, "reason": "hook_error:%s:%s" % (
            type(e).__name__, str(e)[:120])})
    try:
        log_dir = os.path.join(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))), "logs")
        if not os.path.isdir(log_dir):
            os.makedirs(log_dir)
        with io.open(os.path.join(
                log_dir, "ooc_shadow_%s.jsonl" % datetime.now().strftime("%Y%m%d")),
                "a", encoding="utf-8", newline="\n") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except Exception as e:                                  # noqa: BLE001
        _SENTCOV_FAIL_COUNT += 1
        rec["log_error"] = "%s:%s" % (type(e).__name__, str(e)[:120])
    return refusal_reason, rec
