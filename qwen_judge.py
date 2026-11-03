"""judge 配置模块：主模型 qwen-max（DashScope），降级 qwen-plus-2025-07-28、qwen3.7-plus。

RAGAS 的 answer_relevancy 通过 generate_multiple 内部循环采样（n=strictness），
不依赖 API 层 n 参数，因此任意 OpenAI 兼容后端（含仅支持 n=1 的 deepseek 代理）均可作 judge。

可用后端（环境变量 JUDGE_PROVIDER 选择，默认 dashscope）：
  - dashscope: 阿里云 qwen 降级链（qwen-max → qwen-plus-2025-07-28 → qwen3.7-plus → ...）
  - deepseek:  多平台代理分摊并发（config.DEEPSEEK_PLATFORMS：每平台 key/base/model/配额），
               零额外额度，且与答案生成器同模型（统一 J3 口径）
  - ollama:    本地 OpenAI 兼容端点（LOCAL_LLM_BASE_URL/v1，默认 qwen3:4b），免费离线但模型弱

多平台路由（2026-08-08）：请求按平台并发配额加权随机路由（如 12/6/6 → 总在途 24），
每平台独立限流闸；平台失败（403/429/超时）标记黑名单并 failover 到下一平台。
judge 缓存按 (base_url, model) 隔离——不同平台同模型名不得共享响应（口径隔离）。
"""
import sys, os
import hashlib
import json
import random
import threading
import asyncio
import atexit
sys.path.insert(0, os.path.dirname(__file__))
from typing import ClassVar

from langchain_openai import ChatOpenAI
from langchain_ollama import OllamaEmbeddings
from backend.evaluator import RAGASEvaluator
from config import (
    LOCAL_LLM_BASE_URL, API_BASE_URL, DEEPSEEK_API_KEY,
    DEEPSEEK_PLATFORMS,
)

# 修复 9.13-⑤：移除明文硬编码的 DashScope API key（此前源码含真实密钥，
# 属泄露面）。改为环境变量 DASHSCOPE_API_KEY 注入；评测走 deepseek 代理，
# 无需配置即可工作。
DASHSCOPE_KEY = os.getenv("DASHSCOPE_API_KEY", "")
DASHSCOPE_BASE = "https://dashscope.aliyuncs.com/compatible-mode/v1"

MODELS = [
    "qwen-max",
    "qwen-plus-2025-07-28",
    "qwen3.7-plus",
    "qwen3.6-plus",
    "qwen3.7-flash-2026-07-15",
]


# ── judge 并发闸（2026-08-08 多平台版）──────────────────────────────────
# 单平台场景（dashscope/ollama/legacy，quota=0）：全局闸 ≤RAGAS_GLOBAL_CONCURRENCY(默认12)
# 防止代理 prefill 过载退化（conc=20 实测等效 99.5s）。
# 多平台场景（quota>0）：每平台独立 BoundedSemaphore(quota)，总在途 = Σquota（如 24）。
_GATE = threading.BoundedSemaphore(int(os.getenv("RAGAS_GLOBAL_CONCURRENCY", "12")))


def _platform_gates():
    """惰性创建 per-platform 配额闸，key=(base_url, model)。quota=0 返回全局闸。"""
    return _PLATFORM_GATES


_PLATFORM_GATES = {}


def _gate_for(platform):
    key = (platform["base_url"], platform["model"])
    g = _PLATFORM_GATES.get(key)
    if g is None:
        quota = int(platform.get("quota") or 0)
        g = threading.BoundedSemaphore(quota) if quota > 0 else _GATE
        _PLATFORM_GATES[key] = g
    return g


# ── judge 响应持久化磁盘缓存（2026-08-08）────────────────────────────────
# 只缓存成功响应（temperature=0 确定性输出），key=base_url+模型+消息内容哈希。
# 用途：同 (question, answer, contexts) 跨轮次重评时 FA/CR/CP 大输入调用直接命中，
# 跳过代理调用。命中不占并发闸。落盘：每新增 50 条 + 进程退出（atexit）。
class _JudgeDiskCache:
    def __init__(self, path):
        self._path = path
        self._lock = threading.Lock()
        self._data = {}
        self._dirty = False
        try:
            if os.path.exists(path):
                with open(path, "r", encoding="utf-8") as f:
                    self._data = json.load(f)
        except Exception:
            self._data = {}

    def get(self, key):
        with self._lock:
            return self._data.get(key)

    def set(self, key, value):
        with self._lock:
            if key in self._data:
                return
            self._data[key] = value
            self._dirty = True
            if len(self._data) % 50 == 0:
                self._flush_locked()

    def _flush_locked(self):
        try:
            tmp = self._path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(self._data, f, ensure_ascii=False)
            os.replace(tmp, self._path)
            self._dirty = False
        except Exception:
            pass

    def flush(self):
        with self._lock:
            if self._dirty:
                self._flush_locked()


_JUDGE_CACHE = _JudgeDiskCache(os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    ".sisyphus", "evidence", "cache", "judge_cache.json"))
atexit.register(_JUDGE_CACHE.flush)

# 缓存开关：RAGAS_JUDGE_CACHE=0 禁用（校准/比对时需要真实请求）。
_JUDGE_CACHE_ENABLED = os.getenv("RAGAS_JUDGE_CACHE", "1") != "0"


def _msg_to_plain(m):
    if hasattr(m, "type") and hasattr(m, "content"):  # langchain BaseMessage
        return {"role": m.type, "content": str(m.content)}
    if isinstance(m, dict):  # {"role":..., "content":...} 直传形态
        return {"role": m.get("role"), "content": str(m.get("content"))}
    return str(m)


def _messages_key(messages):
    # 兼容：单条消息 / 单层 list / 两层 list（ragas agenerate_prompt 批量路径）
    if isinstance(messages, list) and messages and isinstance(messages[0], list):
        return [[_msg_to_plain(m) for m in inner] for inner in messages]
    if isinstance(messages, list):
        return [_msg_to_plain(m) for m in messages]
    return _msg_to_plain(messages)


# ── 推理开关的缓存一致性（2026-09-18 发现的隐患）─────────────────────────
# 隐患：缓存键为 (base_url, model, messages, max_tokens, temperature, stop)，
# **不含 extra_body**。而 reasoning ON / OFF 的唯一差别就在 extra_body
# （ON = 不注入；OFF = 注入 enable_thinking=False / reasoning_effort=none）。
# 故同一模型在两种推理配置下**共用缓存键** → 后跑的那次会**静默复用**
# 前一次的结果，实验会误以为"两种配置结果相同"。
# （本轮实测：xfam 三臂均为 ON、v41 全量为 OFF，但模型名不同
#   —— mimo-v2.5 vs deepseek-v4.1-flash —— 故未实际串味。）
#
# 处置：**不换键**。换键会作废整个历史缓存（约 38 MB），而判官并非完全确定，
# 重判会改变已冻结的数值。改为：
#   ① 写入时在缓存值内记录推理签名 `think`；
#   ② 命中时校验：签名不一致 → 视为未命中（重判）；
#      无签名（历史条目）→ 放行但计数并告警一次；
#   ③ 需要硬保证时设 JUDGE_CACHE_STRICT_REASONING=1，签名**进入键**
#      （显式代价：历史缓存全部失效）。
_STRICT_THINK_KEY = os.getenv("JUDGE_CACHE_STRICT_REASONING", "0") == "1"
_cache_unverified = 0      # 历史条目（无 think 标记）被放行的次数
_cache_think_mismatch = 0  # 签名不一致而被判为未命中的次数


def _reasoning_signature(kwargs):
    """从请求参数导出推理配置的稳定签名（用于缓存校验）。"""
    eb = kwargs.get("extra_body")
    if isinstance(eb, dict):
        return json.dumps(eb, ensure_ascii=False, sort_keys=True)
    if eb is None:
        return ""
    return str(eb)


def _cache_entry_ok(value, sig):
    """True=命中可用；False=签名不一致（按未命中处理）；None=历史条目无签名。"""
    if not isinstance(value, dict):
        return None
    stored = value.get("think")
    if stored is None:
        return None
    return stored == sig


def _judge_cache_key(base_url, model, messages, kwargs):
    payload = {
        "base_url": base_url,
        "model": model,
        "messages": _messages_key(messages),
        "max_tokens": kwargs.get("max_tokens"),
        "temperature": kwargs.get("temperature"),
        "stop": kwargs.get("stop"),
    }
    # 严格模式：把推理签名并入键（历史缓存将失效，需显式开启）
    if _STRICT_THINK_KEY:
        payload["think"] = _reasoning_signature(kwargs)
    s = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def _cache_take(hit, kwargs):
    """统一的命中判定：返回可用的缓存值，或 None（含签名不符/历史条目放行）。

    副作用：维护 unverified / mismatch 计数，供运行结束时报出。
    """
    global _cache_unverified, _cache_think_mismatch
    if hit is None:
        return None
    ok = _cache_entry_ok(hit, _reasoning_signature(kwargs))
    if ok is True:
        return hit
    if ok is None:
        _cache_unverified += 1
        return hit
    _cache_think_mismatch += 1
    return None


def cache_think_report():
    """运行结束时的缓存一致性报告（供流水线打印）。"""
    return {"unverified_legacy": _cache_unverified,
            "think_mismatch": _cache_think_mismatch,
            "strict_mode": _STRICT_THINK_KEY}


def _result_to_cache(r, sig=None):
    gens = []
    for g in getattr(r, "generations", []) or []:
        m = g.message if hasattr(g, "message") else g
        gens.append({
            "content": getattr(m, "content", "") or "",
            "reasoning": (getattr(m, "additional_kwargs", None) or {}).get("reasoning_content", ""),
        })
    out = {"generations": gens}
    # 记录产生该响应的推理配置（供命中时校验；历史条目无此字段）
    if sig is not None:
        out["think"] = sig
    return out


def _result_from_cache(value):
    from langchain_core.outputs import ChatResult, ChatGeneration
    from langchain_core.messages import AIMessage
    gens = []
    for g in value.get("generations", []):
        msg = AIMessage(content=g.get("content", ""),
                        additional_kwargs={"reasoning_content": g.get("reasoning", "")})
        gens.append(ChatGeneration(message=msg))
    return ChatResult(generations=gens)


def _platform_key(platform):
    return (platform["base_url"], platform["model"])


class FailoverChatOpenAI(ChatOpenAI):
    """多平台降级包装：请求按平台并发配额加权路由，失败依次 failover 到候选平台。

    platforms: list[dict(base_url, api_key, model, quota, name)]——
      - deepseek:   3 平台分摊并发（DEEPSEEK_PLATFORMS）
      - dashscope:  同一 key/base 的模型降级链（quota=0，受全局闸）
      - ollama:     本地单端点（quota=0，受全局闸）

    403/429（额度耗尽）被缓存到类级黑名单，后续请求直接跳过该平台，
    避免每个请求都先失败 1~2 次（修复 J1）。
    """

    model_config = {"extra": "allow"}

    # 类级黑名单：已确认 403 额度耗尽/429 限流的 (base_url, model)，进程内永久跳过。
    # 修复 J1：必须标注 ClassVar，否则 pydantic v2 会把带下划线且无注解的
    # 类属性当作私有属性（ModelPrivateAttr），导致 ragas 调用 judge 时报
    # "argument of type 'ModelPrivateAttr' is not iterable"。
    _dead_platforms: ClassVar[set] = set()

    def __init__(self, platforms, **kwargs):
        super().__init__(
            model=platforms[0]["model"],
            openai_api_key=platforms[0]["api_key"],
            openai_api_base=platforms[0]["base_url"],
            **kwargs,
        )
        self._platforms = list(platforms)
        self._base_kwargs = dict(kwargs)

    def _alive_platforms(self):
        alive = [p for p in self._platforms
                 if _platform_key(p) not in FailoverChatOpenAI._dead_platforms]
        if not alive:
            alive = self._platforms  # 全黑名单时兜底全试
        return alive

    def _weighted_order(self, platforms):
        """固定优先级排列：按 platforms 列表顺序（靠前优先），
        跳过黑名单平台。列表顺序由 config 按 KEY_N 序号排列。
        2026-08-30 修改：原配额加权随机 → 固定优先级，保证主链路
        （如官方 Key2）用到底，余额不足才 failover 到备用（Key1）。"""
        return [p for p in platforms
                if _platform_key(p) not in FailoverChatOpenAI._dead_platforms]

    def _mark_dead(self, platform, exc):
        msg = str(exc)
        is_quota = any(k in msg.lower() for k in (
            "403", "quota", "free quota", "insufficient", "exceeded",
            "429", "rate limit", "throttl", "并发", "限流", "balance",
        ))
        if is_quota:
            FailoverChatOpenAI._dead_platforms.add(_platform_key(platform))
            print(f"  [judge] 平台 {platform.get('name')} ({platform['model']}) 额度受限(403/429)，加入黑名单跳过")
        else:
            print(f"  [judge] 平台 {platform.get('name')} ({platform['model']}) 失败: "
                  f"{type(exc).__name__}: {str(exc)[:150]}")

    def _make(self, platform):
        kw = dict(self._base_kwargs)
        model = platform["model"]
        # 2026-08-17: 所有模型一律关闭思考/reasoning。
        # - deepseek 系用 reasoning_effort=none（与 api_client 口径一致，避免
        #   reasoning_content 抢占 max_tokens 预算导致长输出截断）
        # - qwen 系用 enable_thinking=False（qwen 后端参数）
        # extra_body 两参数都传，后端各自识别，兼容所有平台。
        kw["extra_body"] = {"enable_thinking": False, "reasoning_effort": "none"}
        # 修复 judge 平替：一律强制 n=1。deepseek 等后端仅支持 n=1，
        # 多采样由下方 _generate/_agenerate 串行模拟（兼容任意 OpenAI 端点）。
        kw["n"] = 1
        # 复用 api_client 那一份头：opencode 系平台缺 x-opencode-session 时直接
        # 400 MissingSessionID，判分一条都回不来（2026-09-25 b4 预检实测：答案生成
        # 走 api_client 正常，judge 走 ChatOpenAI 全灭 ⇒ 四指标 NaN）。
        # 只加请求头，不动模型/温度/指标口径；judge 缓存键亦不含头 ⇒ 读数不漂。
        from backend.api_client import _platform_headers
        kw["default_headers"] = _platform_headers({
            "key": platform["api_key"],
            "base_url": platform["base_url"],
            "model": platform["model"],
        })
        return ChatOpenAI(
            model=model,
            openai_api_key=platform["api_key"],
            openai_api_base=platform["base_url"],
            **kw,
        )

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        n = int(kwargs.pop("n", getattr(self, "n", None) or 1))
        if n > 1:
            # 模拟 n 采样：串行 n 次 n=1 调用，合并为多 generation 结果
            return self._generate_multiple(messages, stop=stop,
                                           run_manager=run_manager, n=n, **kwargs)
        last_exc = None
        for platform in self._weighted_order(self._alive_platforms()):
            key = _judge_cache_key(platform["base_url"], platform["model"], messages, kwargs)
            hit = _JUDGE_CACHE.get(key) if _JUDGE_CACHE_ENABLED else None
            hit = _cache_take(hit, kwargs)   # 推理签名校验（不符则按未命中）
            if hit is not None:
                return _result_from_cache(hit)
            gate = _gate_for(platform)
            gate.acquire()
            try:
                llm = self._make(platform)
                r = llm._generate(messages, stop=stop, run_manager=run_manager, **kwargs)
            except Exception as e:
                last_exc = e
                self._mark_dead(platform, e)
                continue
            finally:
                gate.release()
            if _JUDGE_CACHE_ENABLED:
                _JUDGE_CACHE.set(key, _result_to_cache(
                    r, _reasoning_signature(kwargs)))
            return r
        if last_exc is not None:
            raise last_exc
        raise RuntimeError("所有 judge 平台均不可用（额度耗尽或被永久跳过）")

    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):
        n = int(kwargs.pop("n", getattr(self, "n", None) or 1))
        if n > 1:
            return await self._agenerate_multiple(messages, stop=stop,
                                                  run_manager=run_manager, n=n, **kwargs)
        last_exc = None
        for platform in self._weighted_order(self._alive_platforms()):
            key = _judge_cache_key(platform["base_url"], platform["model"], messages, kwargs)
            hit = _JUDGE_CACHE.get(key) if _JUDGE_CACHE_ENABLED else None
            hit = _cache_take(hit, kwargs)   # 推理签名校验（不符则按未命中）
            if hit is not None:
                return _result_from_cache(hit)
            gate = _gate_for(platform)
            # 轮询非阻塞 acquire：取消安全（睡眠中取消不持锁，不泄漏 slot）
            acquired = False
            try:
                while not gate.acquire(blocking=False):
                    await asyncio.sleep(0.05)
                acquired = True
                llm = self._make(platform)
                r = await llm._agenerate(messages, stop=stop, run_manager=run_manager, **kwargs)
            except Exception as e:
                last_exc = e
                self._mark_dead(platform, e)
                continue
            finally:
                if acquired:
                    gate.release()
            if _JUDGE_CACHE_ENABLED:
                _JUDGE_CACHE.set(key, _result_to_cache(
                    r, _reasoning_signature(kwargs)))
            return r
        if last_exc is not None:
            raise last_exc
        raise RuntimeError("所有 judge 平台均不可用（额度耗尽或被永久跳过）")

    def _generate_multiple(self, messages, stop=None, run_manager=None, n=3, **kwargs):
        """同步版 n 采样模拟：n 次 n=1 调用，合并结果（兼容仅支持 n=1 的后端）。

        修复 2026-08-08：递归采样死锁——n 采样调用必须显式 n=1。此前 `self._generate(...)`
        不传 n 时，若 self.n / 上层 kwargs 的 n 仍 >1，会再次进入 _generate_multiple，
        形成无限递归（py-spy 实测递归深度数十层，CPU 持续累积、无外部 API 调用）。
        """
        from langchain_core.outputs import ChatResult
        generations = []
        llm_output = None
        for _ in range(n):
            try:
                r = self._generate(messages, stop=stop, run_manager=run_manager,
                                   n=1, **kwargs)
                generations.append(r.generations[0])
                llm_output = r.llm_output
            except Exception:
                pass
        if len(generations) == n:
            return ChatResult(generations=generations, llm_output=llm_output)
        raise RuntimeError(f"n 采样仅成功 {len(generations)}/{n} 次")

    async def _agenerate_multiple(self, messages, stop=None, run_manager=None, n=3, **kwargs):
        """异步版 n 采样模拟：n 次 n=1 调用，合并结果（兼容仅支持 n=1 的后端）。

        修复 2026-08-08：与 _generate_multiple 同根因的递归采样死锁——内层调用必须
        显式 n=1，否则 self.n / 上层 kwargs 的 n >1 时无限递归（py-spy 堆栈见
        qwen_judge.py:296 ↔ 348 互相嵌套数十层，CPU 100% 且无外网请求）。
        """
        from langchain_core.outputs import ChatResult
        generations = []
        llm_output = None
        for _ in range(n):
            try:
                r = await self._agenerate(messages, stop=stop, run_manager=run_manager,
                                          n=1, **kwargs)
                generations.append(r.generations[0])
                llm_output = r.llm_output
            except Exception:
                pass
        if len(generations) == n:
            return ChatResult(generations=generations, llm_output=llm_output)
        raise RuntimeError(f"n 采样仅成功 {len(generations)}/{n} 次")


def _resolve_judge_cfg(provider=None) -> dict:
    """解析 judge 后端配置。

    优先级：显式 provider 参数 > 环境变量 JUDGE_PROVIDER > 默认 dashscope。
    deepseek 复用 config.DEEPSEEK_PLATFORMS（多平台分摊并发，统一 J3 口径）；
    ollama 用本地 OpenAI 兼容端点。
    """
    provider = (provider or os.getenv("JUDGE_PROVIDER", "dashscope")).lower().strip()
    if provider == "deepseek":
        if DEEPSEEK_PLATFORMS:
            return {"platforms": [
                {"base_url": p.base_url, "api_key": p.key, "model": p.model,
                 "quota": p.quota, "name": p.name}
                for p in DEEPSEEK_PLATFORMS
            ]}
        # 无多平台配置时退回旧单平台变量
        models = [m.strip() for m in os.getenv("JUDGE_MODELS", "deepseek-v4-flash").split(",") if m.strip()]
        return {"platforms": [
            {"base_url": API_BASE_URL, "api_key": DEEPSEEK_API_KEY,
             "model": m, "quota": 0, "name": "legacy"} for m in (models or ["deepseek-v4-flash"])
        ]}
    if provider == "ollama":
        models = [m.strip() for m in os.getenv("JUDGE_MODELS", "qwen3:4b").split(",") if m.strip()]
        return {"platforms": [
            {"base_url": f"{LOCAL_LLM_BASE_URL}/v1", "api_key": "ollama",  # 本地端点占位
             "model": m, "quota": 0, "name": "ollama"} for m in (models or ["qwen3:4b"])
        ]}
    if not DASHSCOPE_KEY and provider == "dashscope":
        raise ValueError(
            "dashscope judge 需要环境变量 DASHSCOPE_API_KEY；"
            "评测走 deepseek 代理（JUDGE_PROVIDER=deepseek），无需配置。"
        )
    return {"platforms": [
        {"base_url": DASHSCOPE_BASE, "api_key": DASHSCOPE_KEY,
         "model": m, "quota": 0, "name": "dashscope"} for m in MODELS
    ]}


def make_judge(model=None, provider=None, platform_index=None):
    """构建 judge。

    model 为 None 时用 provider 的模型/平台链；否则只用指定模型。
    platform_index 指定后只保留对应平台（用于 3 平台同题校准）。
    provider: dashscope（默认）| deepseek | ollama。
    """
    cfg = _resolve_judge_cfg(provider)
    platforms = cfg["platforms"]
    if model is not None:
        platforms = [dict(p, model=model) for p in platforms]
    if platform_index is not None:
        platforms = [dict(platforms[platform_index])]
    return FailoverChatOpenAI(
        platforms=platforms,
        temperature=0,
        request_timeout=int(os.getenv("JUDGE_TIMEOUT", "300")),
        max_retries=int(os.getenv("JUDGE_MAX_RETRIES", "3")),
    )


def make_ragas_embed():
    return OllamaEmbeddings(
        model="qwen3-embedding:0.6b",
        base_url=LOCAL_LLM_BASE_URL,
    )


def make_evaluator(model=None):
    judge = make_judge(model)
    emb = make_ragas_embed()
    name = model or "qwen-max(failover)"
    return RAGASEvaluator(judge, emb, judge_model_name=name), judge
