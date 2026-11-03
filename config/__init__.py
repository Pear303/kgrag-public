"""全局配置与双适配工厂。

按文章设计，系统支持两种运行模式：
- 在线模式 (online)：MiniMax 等 API，稳定、质量高、需联网与 API Key
- 本地模式 (local)：Ollama + sentence-transformers，离线、免费、需本地模型

通过 .env 的 *_PROVIDER 开关切换；LLM 与 Embedding 可独立选择。
"""
import os
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# 显式加载 canonical .env（kgrag_project/.env），与进程 CWD 无关。
# 默认 load_dotenv() 的 find_dotenv 在 `python -c`/REPL 下按 CWD 向上找，
# 从根目录运行时会导致密钥静默漏载；绝对路径彻底消除该依赖。
_ENV_PATH = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(_ENV_PATH, override=False)

# DeepSeek API 配置（供 Task 3 等模块使用）
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
API_BASE_URL = os.getenv("API_BASE_URL", "")

# ---------------- 端点黑名单 ----------------
# 命中的主机在平台清单解析时直接剔除，并在发送前再次硬校验。
# 用 FROZEN_API_HOSTS 环境变量配置，逗号分隔；默认为空。
FROZEN_API_HOSTS = tuple(
    h.strip() for h in os.getenv("FROZEN_API_HOSTS", "").split(",") if h.strip()
)


def is_frozen_base_url(base_url) -> bool:
    """任何形态的 base_url（含带默认值兜底的写法）命中黑名单主机即为真。"""
    s = str(base_url or "")
    return any(h in s for h in FROZEN_API_HOSTS)


def assert_not_frozen(base_url, who: str = "") -> None:
    """判分/生成发送前的硬闸：命中禁用端点直接抛错，绝不静默改走别的路。"""
    if is_frozen_base_url(base_url):
        raise RuntimeError(
            "[端点禁用] %s 试图使用已禁用的判分/生成端点 %s。"
            "该主机在 config.FROZEN_API_HOSTS 名单内，任何脚本/通道都不得调用。"
            % (who or "调用方", str(base_url)[:80]))


# ---------------- 多平台 DeepSeek 代理（2026-08-08） ----------------
# 支持多套 (key, base_url) 分摊并发：评测/答案生成的 judge 请求在平台间加权路由，
# 每平台按其并发配额限流（如 15/6 → 总在途 21）。上下文窗口差异仅影响超长输入
# （平台2 200k vs 其余 100万，当前 ~12k token 无碍）。
@dataclass
class DeepSeekPlatform:
    key: str
    base_url: str
    model: str
    quota: int = 0          # 混合配额场景下 0 = 禁用该平台；全 0 = 受全局闸兜底
    name: str = ""


def _build_deepseek_platforms() -> list:
    """解析 DEEPSEEK_API_KEY_1/2/3 + API_BASE_URL_1/2/3 [+ API_MODEL_N] [+ API_QUOTA_N]。

    无 _N 后缀时退回旧单平台变量（DEEPSEEK_API_KEY / API_BASE_URL / API_MODEL）。
    端点黑名单：主机在 FROZEN_API_HOSTS 里的平台
    **在任何配置状态下都不进清单**——不看 quota、不看 key 是否存在、不看顺序。
    把禁用逻辑写在解析器里而不是只靠 API_QUOTA=0 或删 .env，是为了避免
    "脚本自带 IP 兜底默认值"把禁用状态悄悄复原。
    配额语义（2026-08-08）：
      - 任一平台 quota>0：quota=0 的平台视为"禁用"被剔除（如 API_QUOTA_2=0 即关闭
        平台2），便于在 3 平台里快速关掉不稳定/口径有偏差的端点；
      - 全部 quota=0：视为未设 per-platform 配额，全部保留，首平台兜底
        RAGAS_GLOBAL_CONCURRENCY（默认 12）。
    """
    platforms = []
    for i in range(1, 10):
        key = os.getenv(f"DEEPSEEK_API_KEY_{i}", "")
        base = os.getenv(f"API_BASE_URL_{i}", "")
        if not key or not base:
            continue
        if is_frozen_base_url(base):
            logger.warning("平台%d 主机在冻结名单内（作者令：永久禁用），已剔除：%s",
                           i, base.split("//")[-1][:60])
            continue
        model = os.getenv(f"API_MODEL_{i}",
                          os.getenv("API_MODEL", "deepseek-v4-flash"))
        try:
            quota = int(os.getenv(f"API_QUOTA_{i}", "0") or 0)
        except ValueError:
            quota = 0
        platforms.append(DeepSeekPlatform(
            key=key, base_url=base, model=model, quota=quota, name=f"platform{i}"))
    if not platforms and DEEPSEEK_API_KEY and API_BASE_URL and not is_frozen_base_url(API_BASE_URL):
        platforms.append(DeepSeekPlatform(
            key=DEEPSEEK_API_KEY, base_url=API_BASE_URL,
            model=os.getenv("API_MODEL", "deepseek-v4-flash"),
            quota=0, name="legacy"))
    if platforms:
        if any(p.quota for p in platforms):
            platforms = [p for p in platforms if p.quota > 0]
            # 2026-08-30：固定优先级——按 quota 降序排列，quota 高的主链路
            # 在前（优先使用），quota 低/1 的备用在后（主链路失败才兜底）。
            platforms.sort(key=lambda p: p.quota, reverse=True)
        else:
            platforms[0].quota = int(os.getenv("RAGAS_GLOBAL_CONCURRENCY", "12"))
    return platforms


DEEPSEEK_PLATFORMS = _build_deepseek_platforms()

# ---------------- 基础设置 ----------------

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
MANUALS_DIR = os.path.join(DATA_DIR, "manuals")

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7688")
# 建图专用库：serve.py 会在 import config 前把 NEO4J_URI 改成查询库（7690），
# 因此建图侧必须另存一份 .env 的原始值，否则 api_server 的上传建图会直写冻结库。
NEO4J_BUILD_URI = os.getenv("NEO4J_BUILD_URI") or NEO4J_URI
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")

VECTOR_MODE = os.getenv("VECTOR_MODE", "auto")  # auto | native | memory

# LLM 后端总开关：online(MiniMax) | local(Ollama qwen3)
# 优先读 LLM_PROVIDER，回退 ONLINE_LLM_PROVIDER（兼容旧配置）
LLM_PROVIDER = os.getenv("LLM_PROVIDER") or os.getenv("ONLINE_LLM_PROVIDER", "minimax")
# Embedding 后端总开关：online(MiniMax) | local(Ollama/HuggingFace)
# 默认 local（本机网络受限，用 Ollama qwen3-embedding:0.6b 离线）
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "local")

LOCAL_LLM_BASE_URL = os.getenv("LOCAL_LLM_BASE_URL", "http://localhost:11434")
LOCAL_LLM_MODEL = os.getenv("LOCAL_LLM_MODEL", "qwen2.5:3b")
LOCAL_EMBEDDING_MODEL = os.getenv("LOCAL_EMBEDDING_MODEL", "BAAI/bge-small-zh-v1.5")

# ---------------- Schema / 抽取约束（文章 3.3 节 STRICT_KG_PROMPT） ----------------

ALLOWED_RELATIONS = {"包含", "表现为", "可能原因", "解决方法", "配套使用", "影响"}

RELATION_ALIASES = {
    "核心部件": "包含", "组成部分": "包含", "属于": "包含",
    "引起": "可能原因", "导致": "可能原因", "成因": "可能原因",
    "处理": "解决方法", "维修": "解决方法", "建议": "解决方法", "对策": "解决方法",
}

# 噪音关系过滤（文章 3.3：禁止 MTBF、环境温度、湿度等统计/噪音关系）
FORBIDDEN_RELATIONS = {"MTBF", "环境温度", "湿度", "重量", "尺寸", "价格"}

MAX_CHUNK_SIZE = 2048
MAX_PATHS_PER_CHUNK = 30
SIMILARITY_THRESHOLD = 0.35
# 9.11-2：V 检索 top_k 10 → 12，与 G 的评分 context 上限（max_ctx=12）对齐，
# 消除"上限对称 ≠ 数量对称"（context_precision 按位置取平均，context 多的一方
# 被额外扣分）。评测链路已固定 12，全局默认必须一致，避免新运行静默退回 10。
# 优化：Vector RAG 召回增大到 20，降低相似度阈值到 0.35，适配结构化手册
TOP_K = 20
SIMILARITY_THRESHOLD = 0.35

# 混合检索权重
VECTOR_WEIGHT = float(os.getenv("VECTOR_WEIGHT", "0.7"))
BM25_WEIGHT = float(os.getenv("BM25_WEIGHT", "0.3"))
RETRIEVAL_MODE = os.getenv("RETRIEVAL_MODE", "hybrid")  # vector | bm25 | hybrid

# Vector RAG 滑动窗口切片参数
VECTOR_WINDOW_SIZE = int(os.getenv("VECTOR_WINDOW_SIZE", "1500"))  # 窗口大小（字符），优化：增大到 1500 适配结构化报警条目
VECTOR_OVERLAP_SIZE = int(os.getenv("VECTOR_OVERLAP_SIZE", "100"))  # 重叠大小（字符），优化：增大到 100

# ---------------- 双适配：LLM 工厂 ----------------

def get_llm(provider: Optional[str] = None, model_name: Optional[str] = None):
    """返回 LlamaIndex LLM 实例。

    双适配：online -> MiniMax；local -> Ollama(qwen3)。
    model_name 显式给出时用它，否则用 .env 的 LOCAL_LLM_MODEL。
    """
    provider = provider or LLM_PROVIDER
    model_name = model_name or LOCAL_LLM_MODEL
    if provider == "local":
        from llama_index.llms.ollama import Ollama
        # 2026-10-11：本机生成是"一次算一道"的慢通道，客户端耐心必须可调。
        # 实测本机 Ollama OLLAMA_NUM_PARALLEL=1（服务端一次只服务一个请求），
        # 客户端开并发只会排成队列、把"排队时间"算进每个请求的超时 ⇒ 长题必超时。
        # 默认仍是 300 保持既有行为；local 生成任务用 LOCAL_LLM_TIMEOUT 放宽。
        # 2026-10-11 新增 LOCAL_NUM_PREDICT：实测本机 qwen2.5:3b 在个别题上不吐 EOS
        # （900 秒产出 75,738 字仍在「3128）3129）…」无限编号），加耐心只会产出更长的清单。
        # 只有显式设了这个 env 才把 num_predict 传给 Ollama；不设＝一个字节都不变，
        # 历史轮次与主表全部不受影响（登记表见 半月reports/重跑预注册_金子修正_20261010.md §11）。
        # 接法必须走 additional_kwargs：实测 Ollama 是 pydantic 模型，
        # 裸 kwarg num_predict=… 会被静默丢弃（构造不报错但 _model_kwargs 里没有它），
        # 而请求体只发 options=self._model_kwargs（base.py:414 等四处）。
        _np = (os.getenv("LOCAL_NUM_PREDICT") or u"").strip()
        _extra = {u"num_predict": int(_np)} if _np else {}
        # qwen3.5:4b 生成器：与 LOCAL_NUM_PREDICT 同款策略——
        # 显式设 env 才生效，不设时保持既有行为。
        # LOCAL_NUM_CTX：不设时 llama_index 会自动向 Ollama 探测模型原生 context_length 当 num_ctx
        # （llama_index/llms/ollama/base.py:209-221），4b 原生 262144 ⇒ 6GB 显存 KV cache 装不下，
        # 必须显式给 24576（换装验收验证值，prompt 实际 ~1 万 token，非质量变量）。
        # LOCAL_THINKING_OFF=1：qwen3.5 系默认开思考，思考段吃 num_predict 预算且生产
        # _split_cot_output 不处理 <think>（backend/evaluator.py 只认中文标记），必须构造层关掉。
        _ctx = (os.getenv("LOCAL_NUM_CTX") or u"").strip()
        _th = (os.getenv("LOCAL_THINKING_OFF") or u"").strip()
        return Ollama(model=model_name, base_url=LOCAL_LLM_BASE_URL,
                      temperature=0.0,
                      request_timeout=float(os.getenv("LOCAL_LLM_TIMEOUT", "300")),
                      context_window=int(_ctx) if _ctx else -1,
                      thinking=False if _th == u"1" else None,
                      additional_kwargs=_extra)
    # 默认 online: MiniMax
    from llama_index.llms.openai import OpenAI as LlamaOpenAI
    from llama_index.core.llms import LLMMetadata

    # MiniMax 模型名不在 llama_index 的 OpenAI 模型枚举中，
    # 需创建子类绕过 metadata 属性中的模型名校验
    class _MiniMaxLLM(LlamaOpenAI):
        @property
        def metadata(self) -> LLMMetadata:
            return LLMMetadata(
                context_window=32768,        # MiniMax-Text-01 上下文窗口
                num_output=self.max_tokens or 4096,
                is_chat_model=True,
                is_function_calling_model=True,
                model_name=self.model,
                system_role="system",
            )

    return _MiniMaxLLM(
        model=os.getenv("MINIMAX_LLM_MODEL", "MiniMax-Text-01"),
        api_key=os.getenv("MINIMAX_API_KEY", ""),
        api_base=os.getenv("MINIMAX_BASE_URL", "https://api.minimax.chat/v1"),
        temperature=0.0,
    )


# ---------------- 双适配：Embedding 工厂 ----------------

def get_embed_model(provider: Optional[str] = None, model_name: Optional[str] = None):
    """返回 LlamaIndex Embedding 实例。

    双适配：online -> MiniMax text-embedding；local -> HuggingFace(bge-small-zh)。
    model_name 显式给出时用它，否则用 .env 的 LOCAL_EMBEDDING_MODEL。
    """
    provider = provider or EMBEDDING_PROVIDER
    model_name = model_name or LOCAL_EMBEDDING_MODEL
    if provider == "local":
        # 本地 embedding 双后端：
        #   - Ollama 模型名（含 ':'，如 qwen3-embedding:0.6b）走 OllamaLocalEmbedding
        #     （包装 ollama 原生 embeddings 接口，离线、稳定；llama-index 自带
        #      OllamaEmbedding 在 qwen3-embedding 上会挂起，故自封装）
        #   - 其余（如 BAAI/bge-small-zh-v1.5）走 HuggingFaceEmbedding
        if ":" in model_name or model_name.startswith("qwen"):
            from backend.ollama_embed import OllamaLocalEmbedding
            return OllamaLocalEmbedding(model_name=model_name,
                                        base_url=LOCAL_LLM_BASE_URL)
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding
        return HuggingFaceEmbedding(model_name=model_name)
    from llama_index.embeddings.openai import OpenAIEmbedding
    # 注意：MiniMax 的 embedding 模型名(embo-01)不在 OpenAI 枚举内，
    # 通过 model_name= 传入可绕过 OpenAIEmbedding 的模型枚举校验。
    return OpenAIEmbedding(
        model_name=os.getenv("MINIMAX_EMBED_MODEL", "embo-01"),
        api_key=os.getenv("MINIMAX_API_KEY", ""),
        base_url=os.getenv("MINIMAX_BASE_URL", "https://api.minimax.chat/v1"),
    )


@dataclass
class QueryEngineConfig:
    top_k: int = TOP_K
    similarity_threshold: float = SIMILARITY_THRESHOLD
    max_paths_per_chunk: int = MAX_PATHS_PER_CHUNK
    vector_mode: str = VECTOR_MODE


# 运行时配置（供各模块共享）
RUNTIME = QueryEngineConfig()

# ---------------- 消融实验开关（2026-08-11，默认全关，正式链路行为不变） ----------------
# 摸底消融矩阵见 docs/superpowers/specs/2026-08-11-ablation-design.md。
# 每个开关只关闭 v15 全配置中的一个组件；env=1/true/yes/on 视为开启。
@dataclass
class AblationConfig:
    ablate_4gram: bool = False          # A1 关 4-gram 概念短语重叠召回
    ablate_rescore: bool = False        # A2 关实体二次打分/代码 token 优先
    ablate_embed_match: bool = False    # A3 关余弦嵌入匹配（仅 4-gram + 字面）
    ablate_frag_norm: bool = False      # B1 关短语归一化（_norm_fragment_token 退化为恒等）
    ablate_window: bool = False         # B2 扩窗 12000 回退 1200（_fragment_window）
    ablate_ooc: bool = False            # B3 关 OOC 对象一致性检查（相关度门保留）
    ablate_chunk_summary: bool = False  # B4 关社区摘要/报告注入（保留硬事实+原文）
    ablate_community: bool = False      # C1 关社区宏观查询（仅实体路径引擎）


def _env_bool(name: str) -> bool:
    return os.getenv(name, "0").strip().lower() in ("1", "true", "yes", "on")


def _build_ablation_config() -> AblationConfig:
    return AblationConfig(
        ablate_4gram=_env_bool("ABLATE_4GRAM"),
        ablate_rescore=_env_bool("ABLATE_RESCORE"),
        ablate_embed_match=_env_bool("ABLATE_EMBED_MATCH"),
        ablate_frag_norm=_env_bool("ABLATE_FRAG_NORM"),
        ablate_window=_env_bool("ABLATE_WINDOW"),
        ablate_ooc=_env_bool("ABLATE_OOC"),
        ablate_chunk_summary=_env_bool("ABLATE_CHUNK_SUMMARY"),
        ablate_community=_env_bool("ABLATE_COMMUNITY"),
    )


ABLATE = _build_ablation_config()


def ablation_snapshot() -> dict:
    """生效中的消融开关快照（meta.json 的 ablation_config 字段）。"""
    from dataclasses import fields
    return {f.name: getattr(ABLATE, f.name)
            for f in fields(ABLATE) if getattr(ABLATE, f.name)}
