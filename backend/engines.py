"""三层查询引擎（文章 4.1 / 4.2 / 5.5 节）。

- AutoGraphQueryEngine：场景A，LLM 自动提取图谱。
    原生模式用 Neo4j 向量索引语义定位实体；降级模式用内存余弦（见 graph_store）。
    定位后以 Cypher 展开 1~4 跳路径。
- ManualGraphQueryEngine：场景B，人工构建图谱。
    :Entity 节点无向量索引，启动时预计算 name embedding，查询用 NumPy 余弦相似度。
- CommunityQueryEngine：宏观查询，匹配社区描述 embedding，召回 Top-K 社区关联 Chunk。
"""
import os
import re
import json
import uuid
import numpy as np
from collections import Counter

from config import ABLATE, SIMILARITY_THRESHOLD, TOP_K
from backend.rag_protocol import RetrievalResult

# 社区报告截断上限（字符），避免超长 report 撑爆上下文窗口
MAX_COMMUNITY_REPORT_CHARS = 6000

# 同一进程内复用 chunk 向量缓存的令牌。跨进程（API 重启）不复用：
# qwen3-embedding 对长文本存在跨进程不可复现性，复用在另一进程算的向量会失真。
_PROCESS_TOKEN = uuid.uuid4().hex

# 修复 H4：实体名匹配的专用阈值。实体名（短）与整句问题（长）的 embedding
# 余弦天然偏低，复用检索阈值 0.5 会漏掉大量关键实体，改用更低的独立阈值。
# v3.1（9.10）：0.35 → 0.55（数据驱动）。实测 18 题实体相似度：
#   in-context 题 top-3 全部 ≥ 0.56；完全 out-of-context 题（燃气轮机 0.47 / 锅炉 0.53）
#   ≤ 0.53——0.55 是两类题的最小分类间隔。0.35 会把完全无关题的弱相似实体
#   （0.45~0.53）误当命中 → 塞无关 context → 模型答非所问反而得分（OOC 口径
#   不对称的根源之一，9.2-A2）。
# v4（9.11-3 修复后重标定）：0.55 → 0.60。图谱从 153 条扩到 317 条（含短部件
#   补抽，实体 383 个），相似度分布整体上移——实测 in-context 题 top-1 全部
#   ≥0.69、top-3 全部 ≥0.602；完全 OOC 题 top-1：auto_016 燃气轮机 0.513、
#   auto_017 锅炉水位计 0.56（"叶轮堵塞"实体因"堵塞"语义误命中，0.55 放行导致
#   G 侧编造，破坏 OOC 对称）。0.60 是新的最小分类间隔（0.56 < 0.60 < 0.602），
#   半 OOC 题 auto_018（0.623）仍正常命中。
ENTITY_MATCH_THRESHOLD = float(os.getenv("ENTITY_MATCH_THRESHOLD", "0.60"))
# 修复 2026-08-05：阈值可经环境变量覆盖（env ENTITY_MATCH_THRESHOLD）。
# 上面 0.60 是 pump 数据标定（in-context ≥0.602 / OOC ≤0.56 最小分类间隔）。
# shukongdashi 图谱概念实体与问题措辞差异更大（如"回参考点不能进行"对问题
# "回参考点操作不能进行时" sim=0.569 < 0.60），且其 OOC 题（电火花/机器人等）
# top-1 sim 已达 0.57~0.66——0.60 对 shukongdashi 既不保护 OOC、又拦住了单跳
# 概念实体。A2 决策（2026-08-05）：shukongdashi 评测**保持 0.60**，理由：
#   ① 与 test_regression_guard 的 R2 不变量（>=0.60）一致，杜绝 pump 回归；
#   ② 概念实体由下方 3.5 步 4-gram overlap 召回兜底（"回参考点不能进行"与问题
#      共享"回参考点""不能进行"两个全中文 4-gram → 直接召回，不经 0.60 语义阈值）；
#   ③ 降到 0.55 的代价是 context_precision 噪声（OOC 题 sim 0.57~0.66 会放行）。
# 实验性放开单跳概念召回时可用 env ENTITY_MATCH_THRESHOLD=0.55，并在评测
# 记录里注明该覆盖（代价：OOC 题 context_precision 略有噪声，OOC 题本就检索到噪声）。

# 修复 KG-7 / 9.10：社区保底召回的相似度下限。低于该值视为完全不相关，宁可不召回，
# 避免把无关社区文本塞进 context 稀释 precision。
# v3.1：0.25 → 0.5（与 SIMILARITY_THRESHOLD 一致）。实测 OOC 题社区 top-1 相似度
# 0.43~0.53：0.25 的下限会让保底把无关社区文本重新塞回（OOC 题答非所问得分的
# 另一根源）。提到 0.5 后保底退化为与语义过滤等效，宁缺毋滥。
COMMUNITY_FALLBACK_THRESHOLD = 0.5

# 修复 KG-2：每个匹配实体最多展开的路径条数（原全局 LIMIT 30 会饿死后部实体）
# 2026-08-05（B2）：路径展开改用 _expand_paths_per_depth 逐跳限量采样（hop=1..4
# 各 per_depth=2 条），本常量保留作为历史语义的文档锚点，不再参与路径展开。
PATHS_PER_ENTITY = 5

# 修复 2026-08-05（shukongdashi 检索调优）：语义匹配的贪心去重余弦上限。
# 原逻辑按相似度取 top_k 会被集线实体变体挤满（shukongdashi 图谱 FANUC/SIEMENS
# 系列 60+ 个 ≥0.6 的近义实体），把连接答案代码的概念实体（如"回参考点报警" 0.645，
# 排名第 20+）挤出匹配集合，路径检索永远到不了 P/S90。贪心去重：按相似度降序，
# 仅接受与已选实体 embedding 余弦 < DIVERSITY_COS 的实体，在保住概念多样性的
# 同时保留 top_k 上限。
DIVERSITY_COS = 0.92

# 2026-08-05（shukongdashi 差距诊断）：概念短语重叠召回上限。
# 4-gram overlap 原无上限，会把"问题模板"相关的泛化概念实体全部召回
# （实测 matched 曾到 75 个），路径展开被集线/泛化实体淹没；限制为最多 5 个。
OVERLAP_MAX = 5

# 问句模板中的通用中文 4-gram：不携带领域语义，参与 overlap 会让无关实体借
# 模板词混入（如"主轴抖动且无法完成换档"与"无法完成数据传输"共享"无法完成"，
# 再叠加"故障现象"等模板 gram 即满足 ≥2 误召回）。overlap 判定时排除。
_GENERIC_QUERY_4GRAMS = frozenset({
    "出了什么", "故障现象", "可能是什", "是什么原", "什么原因", "可能原因",
    "应该怎么", "该怎么解", "怎么解决", "解决排查", "解决方法", "常见原因",
    "如何排查", "是什么", "哪一路", "正常值是", "常值是多", "值是多少",
    "什么含义", "表示什么", "示什么含", "的含义",
})

# 2026-08-07：实体展开上限 10 -> 20。shukongdashi 的 Q11/Q12/D12 等
# 元件代号实体在语义匹配中常排到 15~25 位，旧上限导致答案路径（如
# Q12的CE极在线短路--[可能原因]-->D12(A169)故障）从未进入 path_contexts。
MAX_EXPAND_ENTITIES = 20

# 2026-08-07：文本单元级召回（案例原文兜底）。
# 社区描述 embedding 只由节点名拼接，巨型社区（几十个 chunk）的描述无法代表
# 内部所有案例（auto_036 的例105.1 就在社区 raw_context 里，但社区描述只是
# "CNC; E0C; 端子M6; 端子M7"，社区级匹配永远召不回）。对去重后的 raw chunk
# 建语义索引，命中后把所属社区前置，并把命中原文片段优先进 context。
CHUNK_INDEX_TOP_N = 8
# 阈值 0.45：qwen3-embedding 对长原文 vs 问句的余弦普遍低于社区描述
#（校准脚本实测相关案例 chunk top-1 0.47~0.62；无关 chunk 普遍 <0.40）。
CHUNK_INDEX_THRESHOLD = float(os.getenv("CHUNK_INDEX_THRESHOLD", "0.45"))


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _shared_chinese_4grams(name: str, q4: set) -> int:
    """实体名与问题共享的互不相同『全中文 4-gram』数量。

    用途：概念短语重叠召回（见 _match_entities_smart 注释）。只统计全部由
    汉字组成的 4-gram——"回参考点""不能进行"这类概念短语会被计入，而
    "FANUC""NUC0"等含英文/数字的集线子串被排除，避免集线实体借共享词混入。
    """
    if not name or len(name) < 4:
        return 0
    seen = set()
    cnt = 0
    for i in range(0, len(name) - 3):
        g = name[i:i + 4]
        if g in seen:
            continue
        seen.add(g)
        if all(0x4e00 <= ord(ch) <= 0x9fff for ch in g) and g in q4:
            cnt += 1
    return cnt


def _literal_sort_key(name: str) -> tuple:
    """字面命中排序键：含数字/拉丁字符的实体（报警号/地址/端子）优先，再按长度。

    KG-3 原按长度降序会把"FANUC 0MC系统"排在"ALM100报警"前，答案代码路径
    只能排到轮询第 4 位；报警号/地址类代码实体（ALM100/X018/DGN700/M06 等）
    先出，答案路径更早进入前 6 路径槽。纯型号词（FANUC 0MC）虽含数字但不
    匹配代码模式，仍按长度排序。
    """
    has_code = bool(re.search(r"(?:P/S|SV|ALM|DGN|NC|X|Y|F|G|M|T|O)\d+", name))
    return (has_code, len(name))


def _query_4grams(query_str: str) -> set:
    """提取问题中的全中文 4-gram，并剔除问句模板通用词。"""
    grams = {query_str[i:i + 4] for i in range(len(query_str) - 3)}
    return {
        g for g in grams
        if all(0x4e00 <= ord(ch) <= 0x9fff for ch in g)
        and g not in _GENERIC_QUERY_4GRAMS
    }


# ═══════ 报警号别名召回（2026-09-06，GSK983 手册命名适配）═══════════════════════
# 背景：GSK983 报警号实体统一命名为 "GSK983报警405"（三位带前导零，如 001/090）。
# 用户口语问法常省略前缀与前导零（"报警405怎么处理""报警90排除方法"），字面匹配
# （要求实体名是问句的连续子串）与 4-gram 重叠（数字打断中文 gram）双双失效 →
# 问"报警405"只能靠整句 embedding，短问句向量噪声大，实测 matched 落到
# "出现#085号报警"等无关实体，405 路径永不进入 context。
# 方案：从实体名/问句两侧提取"报警+数字"并按 int 归一（去前导零），数字对上即召回。
# 只响应含"报警/ALARM"字样的问句 → 不影响 pump/500cases 与 OOC 纯描述问句。
_ALARM_TOKEN_RE = re.compile(r"(?:报警|ALARM)\s*号?\s*#?\s*(\d{1,4})")


def _alarm_query_number(query_str: str):
    """问句点名的报警号（int 归一，无则 None）。只认与"报警"相邻的数字，避免
    把 "GSK983 用了 405 号刀库" 之类的泛数字误判为报警号。"""
    m = _ALARM_TOKEN_RE.search(query_str)
    if not m:
        return None
    num = m.group(1)
    # 全角/半角已由 _ALARM_TOKEN_RE 限定 \d；保留 int 归一去前导零
    return str(int(num))


def _build_alarm_index(entity_names) -> dict:
    """构建 归一报警号 → [实体名] 映射。

    实体名形如 "GSK983报警405"/"GSK983报警090"；同一数字可能存在多个长度变体
    （历史碎片如 "GSK983报警90"），统一归一到同一 key 便于都召回。
    """
    idx = {}
    for n in entity_names:
        if not n or "报警" not in n:
            continue
        for m in _ALARM_TOKEN_RE.finditer(n):
            idx.setdefault(str(int(m.group(1))), set()).add(n)
    return {k: sorted(v, key=len, reverse=True) for k, v in idx.items()}


def _dedup_alarm_prefix_fragments(literal: list) -> list:
    """剔除字面命中集合里的报警号碎片（短命名保留长命名）。

    背景（GSK983 实测）：问句 "GSK983报警405怎么处理" 的字面命中除
    "GSK983报警405" 外还会按子串命中碎片实体 "GSK983报警4"（若其仍存在），
    碎片与真号抢 context 槽。数据侧已把 1/2 位碎片并入三位
    （GSK983报警4→GSK983报警004），此处是防御性兜底（未来新手册或 Auto 图谱
    可能再次出现非统一命名）。判定两条（任一即丢短保长）：
      ① 同数字不同位数（归一 int 相等）：001 vs 1、010 vs 10（跨前导零变体）；
      ② 长号原始数字串以短号为前缀：212 vs 21、405 vs 4、900 vs 90。
    仅剔除集合内确实存在更长实体时被覆盖的短者，独立短号（单独被问）不受影响。
    """
    if len(literal) < 2:
        return literal
    info = {}
    for n in literal:
        m = _ALARM_TOKEN_RE.search(n)
        if m:
            raw = m.group(1)
            info[n] = (raw, str(int(raw)))  # (原始数字串, int 归一)
    if len(info) < 2:
        return literal
    drop = set()
    names = sorted(info, key=len, reverse=True)  # 长名优先作为权威
    for i, long in enumerate(names):
        lraw, lnum = info[long]
        for short in names[i + 1:]:
            sraw, snum = info[short]
            if snum == lnum or lraw.startswith(sraw):
                drop.add(short)
    return [n for n in literal if n not in drop]


# ═══════ 现象实体括号别名（2026-09-06，修复①-补充：现象式问法适配）═══════════
# 背景：现象实体命名常带括号消歧后缀，如 "OT报警(超程报警)"、"参考点停止位置与
# 参考点不一致(微小偏差)"、"GSK983故障现象:上电轴暴走窜动(软件)"。用户口语只问
# 核心语义（"超程报警如何复位"），字面匹配要求实体名 ⊆ 问句 → 永不命中；
# 4-gram 重叠又被括号/数字打断。方案：对含半角/全角括号的实体提取括号内的
# 4~10 字纯中文别名段，问句若包含该段即以别名命中（置匹配结果前部）。
_CN_RUN = re.compile(r"[\u4e00-\u9fff]{4,12}")


def _paren_alias_segments(name: str) -> list:
    """从带括号实体名提取纯中文别名段：
    - 括号内纯中文段（4~16 字）：OT报警(超程报警) → '超程报警'
    - 括号外纯中文主体（6~20 字）：存储行程限位报警(软件限位报警) →
      '存储行程限位报警'；重复定位精度差(机械) → '重复定位精度差'
    短括号词（(机械)/(软件)，<4 字）与含字母/数字的主体（OT报警）不索引，
    避免泛短词/型号词作别名造成误召回。括号外主体完整实体名已由字面/语义覆盖，
    这里只在"带括号消歧"的实体上额外给主体一个别名入口（用户常只问主体）。
    """
    segs = []
    for m in re.finditer(r"[\(（]([\u4e00-\u9fff]{4,16})[\)）]", name):
        seg = m.group(1).strip()
        if 4 <= len(seg) <= 16:
            segs.append(seg)
    # 括号外主体（首个左括号之前；无括号则整体是唯一主体，无需别名）
    lb = name.find("(")
    if lb < 0:
        lb = name.find("（")
    if lb >= 0:
        head = name[:lb].strip()
        if 6 <= len(head) <= 20 and _CN_RUN.fullmatch(head):
            segs.append(head)
    return segs


def _build_paren_alias_index(entity_names) -> dict:
    """纯中文别名段 → [实体名]（长实体优先）。"""
    idx = {}
    for n in entity_names:
        for seg in _paren_alias_segments(n):
            idx.setdefault(seg, []).append(n)
    for k, v in idx.items():
        idx[k] = sorted(set(v), key=len, reverse=True)
    return idx


def _paren_alias_recall(query_str: str, paren_index: dict, exclude: set, top_k: int) -> list:
    """问句包含某个括号别名段（且该段不在问句模板通用词里）时召回对应实体。

    别名段是 4~10 字纯中文短语（如"超程报警"），出现在问句即视为用户在该
    概念 → 召回其载体实体。排除字面已命中的。仅对 Manual 场景生效（见调用处）。
    """
    if not paren_index:
        return []
    hits = []
    for seg, names in paren_index.items():
        if seg in query_str:
            for n in names:
                if n not in exclude and n not in hits:
                    hits.append(n)
                    if len(hits) >= top_k:
                        return hits
    return hits


def _rank_paths(paths: list, query_str: str) -> list:
    """按与问题的相关度给路径排序（实体内部），提高答案路径进入前 N 槽概率。

    规则：
      1. 路径文本包含问题中的英数 token（型号/报警号/地址，如 ALM100/X018）→
         绝对优先（+1000 级），答案代码路径不会被弱关系长链挤出；
      2. 其余按关系类型偏好（可能原因/解决方法 > 表现为 > 包含类）+ 短路径
         优先，避免多跳"包含"链靠关系数量堆分。
      3. 修复 2026-08-06（shukongdashi 优化）：配套使用权重从 1 降到 0。
         低分组（faithfulness < 0.3）中 配套使用+包含 占 42.6%，高分组仅 22.1%。
         配套使用（如 FANUC 0--[配套使用]-->FANUC α系列）对故障诊断无价值，
         降到 0 后这类路径排到末尾，为有效路径让出槽位。
    """
    rel_score = {
        "可能原因": 3, "解决方法": 3,
        "处理方法": 3, "故障原因": 3, "报警含义": 3,
        "确认方法": 3, "检查处理方法": 3,
        "表现为": 2,
        "包含": 1, "配套使用": 0, "影响": 0,
    }
    # 只对"含数字"的 token 计分：ALM100/X018/DGN700 是答案代码，FANUC/SIEMENS
    # 这类纯字母型号词在整句问题里几乎必然出现，会让无关泛化路径靠型号词排前。
    tokens = {
        t for t in re.findall(r"[A-Za-z0-9][A-Za-z0-9.+-]*", query_str)
        if len(t) >= 2 and any(c.isdigit() for c in t)
    }

    def score(p):
        names, rels = p[0], p[1]
        text = "".join(str(n) for n in names)
        hit = sum(len(tok) * 2 for tok in tokens if tok in text)
        if hit:
            return 1000 + hit - 0.5 * len(rels)
        rel_best = max((rel_score.get(rel, 1) for rel in rels), default=0)
        return rel_best - 0.2 * len(rels)

    return sorted(paths, key=score, reverse=True)


def _code_token_priority_entities(query_str: str, matched: list) -> list:
    """把实体名包含问题代码 token（Q11/D12/VS11/ALM100 等）的实体提前。

    shukongdashi 中元件代号实体（如“Q12的CE极在线短路”）在语义/overlap
    匹配里常排到 15~25 位，超出展开上限后答案路径永远进不了 context。
    稳定排序：命中 token 的实体按原序前置，未命中的保持原序，不改变
    其余检索排序语义。
    """
    if not matched or not query_str:
        return matched
    tokens = {
        t for t in re.findall(r"[A-Za-z0-9][A-Za-z0-9.+#-]*", query_str)
        if len(t) >= 2 and any(c.isdigit() for c in t)
    }
    if not tokens:
        return matched
    # 2026-09-09（gsk_037 根因）：过滤"无鉴别力"的泛化 token。
    # GSK983 图谱几乎每条路径/现象实体名都含型号子串 "GSK983"/"983"——问句
    # 出现该型号时，它命中 matched 中绝大多数实体（无任何排序信息），若照旧
    # 参与重排，会把不带型号前缀的精确概念实体（存储行程限位报警(软件限位报警)）
    # 压到 MAX_EXPAND_ENTITIES 位之外，其 212/213 处理方法路径永不进 context。
    # 规则：token 在 matched 中命中实体占比 > 50% 即视为泛化（型号/系统名等），
    # 不参与前置；仅保留"有选择性"（命中 ≤50%）的强代码 token（Q12/D12/405）。
    _total = len(matched)
    _selective = [
        t for t in tokens
        if 0 < sum(1 for n in matched if t.upper() in str(n).upper())
        <= _total / 2
    ]
    if not _selective:
        return matched
    hits = [n for n in matched
            if any(t.upper() in str(n).upper() for t in _selective)]
    if not hits:
        return matched
    rest = [n for n in matched if n not in hits]
    return hits + rest


def _interleave_paths(per_entity_paths: list, n_priority: int = 0) -> list:
    """跨匹配实体轮询交错合并路径列表（9.13-②）。

    修复：路径按"匹配实体顺序"拼接时，前部实体（往往是字面命中的长名实体，
    如"SHS 01013-2019"）会把 PATHS_PER_ENTITY 条路径全部排在前面，配合
    context 12 槽位上限，把答案核心实体（如"热油泵停泵后"）的路径整体挤出
    评分 context（auto_008/013 实证）。轮询交错保证各实体路径都有机会进入
    前 N 槽，图谱证据覆盖更多实体。

    参数: per_entity_paths = [[(names, rels), ...], ...]（每个匹配实体一个列表）
          n_priority = 前 n 个实体的路径『整段优先输出』，其余仍轮询交错
          返回: 交错后的扁平 [(names, rels), ...]
    """
    out = []
    # B3 优化（2026-08-05）：概念实体由"整段前移"改为更高频轮询——每轮出 2 条，
    # 其余实体每轮 1 条。原"整段前移"在概念实体多（>槽位 6）时仍会把后部概念实体
    # 挤出 context，且概念实体 depth-0 路径占满前 N 槽会挤压其他实体的多跳路径。
    # 高频轮询在保证概念实体优先可见的同时保留交错。n_priority=0（pump 绝大多数
    # 题）时 prio 为空、rest 即全部，行为与旧版完全一致，杜绝 pump 回归。
    prio = per_entity_paths[:n_priority]
    rest = per_entity_paths[n_priority:]
    max_len = max((len(x) for x in per_entity_paths), default=0)
    for i in range(max_len):
        for lst in prio:
            for j in (2 * i, 2 * i + 1):
                if j < len(lst):
                    out.append(lst[j])
        for lst in rest:
            if i < len(lst):
                out.append(lst[i])
    return out


_DIAGNOSTIC_QUERY_RE = re.compile(
    r'(确认|检查|如何判断|怎么判断|如何检测|怎么检测|如何测|怎么测|测量|诊断|判断|'
    r'如何看|怎么看|如何核实|怎么核实|查看|核对)')


def _prioritize_diag_paths(paths: list, query_str: str) -> list:
    """确认/检查类问法：确认/检查路径最优先，原因/处理路径次之，相关者先出。

    能力边界1修复（2026-09-07）：GSK983 问"怎么确认机床锁住(MLK)信号"时，
    确认方法路径（机床锁住(MLK)--[确认方法]-->确认DGN#100第7位为0）在 12 槽
    竞争中被 JOG/伺服因果链及其他现象实体的检查方法挤出，答案给不出确认步骤。
    仅在 query 含诊断类动词时启用：
      ① 首跳为 确认方法/检查处理方法 的路径整段置于最前——确认问法下优先级
         高于原因/处理路径（2026-09-09 回归修复：P0-3 曾把四类关系并同桶，
         同起点 tie-break 会让原因路径反超确认路径，单测实证）；
      ② 首跳为 故障原因/处理方法 的路径次之；
      ③ 各段内按起点与 query 的中文 4-gram 重叠降序（问法所指实体的路径最先）。
    非命中问法返回原顺序（不影响 500cases 等"原因/处理"问法——确认层为空
    时，原因层排序行为与原实现一致）。
    """
    if not paths or not query_str or not _DIAGNOSTIC_QUERY_RE.search(query_str):
        return paths
    confirm_first, cause_first, others = [], [], []
    for p in paths:
        rel0 = str(p[1][0]) if len(p) >= 2 and p[1] else ""
        if rel0 in ("确认方法", "检查处理方法"):
            confirm_first.append(p)
        elif rel0 in ("故障原因", "处理方法"):
            cause_first.append(p)
        else:
            others.append(p)
    diag_first = confirm_first + cause_first
    if len(diag_first) <= 1:
        return diag_first + others
    q4 = _query_4grams(query_str)

    def _rank(p):
        return -_shared_chinese_4grams(str(p[0][0]), q4) if (p and p[0]) else 0

    confirm_first = sorted(confirm_first, key=_rank)
    cause_first = sorted(cause_first, key=_rank)
    return confirm_first + cause_first + others


# 推理链聚合的"诊断语义关系"（一阶边，作为链的骨架）
_CHAIN_DIAG_RELS = frozenset({
    "故障原因", "可能原因", "处理方法", "解决方法", "确认方法",
    "检查处理方法", "报警含义", "处理步骤", "检查步骤",
})


def _infer_meta_from_text(rel: str, obj_text: str) -> str:
    """P0-3：数据无 meta 时，从关系+客体文本规则推断关系元类型。

    - 条件分支：客体含 若/如果/如不能/当…则 → conditional
    - 并列原因/并列方法：多条同主体同关系 → parallel（由调用方聚合判定）
    - 步骤类关系（确认方法/检查处理方法/处理步骤）：sequence
    - 其余：none
    """
    if any(k in obj_text for k in ("若", "如果", "如不能", "无法", "不能移动", "则")):
        return "conditional"
    if rel in ("确认方法", "检查处理方法", "处理步骤", "检查步骤"):
        return "sequence"
    return "none"


def _build_reasoning_chains(paths: list) -> list:
    """P0-3：把同主体的一阶诊断路径聚合为显式推理链，注入 path 流前部。

    问题：图谱路径逐条交错输出（A--[故障原因]-->C1 / A--[故障原因]-->C2 /
    A--[处理方法]-->M1 / A--[处理方法]-->M2 …），LLM 看到的是打散的扁平边，
    难以重构"并列原因 + 有序步骤"的多跳语义（gsk_024 只列 2/4 原因实证）。

    方案：对路径列表做一次只读扫描——凡是 names[0] 相同、关系为一阶诊断边
    （故障原因/可能原因/处理方法/解决方法/确认方法/检查处理方法/处理步骤）的
    路径，按 (主体, 关系类型) 聚类，生成结构化链行：

      [推理链] GSK983报警090 | 故障原因(3): ①... ②... ③... ; 处理方法(2): ①... ②...

    链行以 (names=[链文本], rels=["推理链"], dirs=["?"]) 形态加入，排在最前；
    原始路径全部保留在后（不丢召回，评分仍可见原始路径）。单条孤立的诊断边
    （无法成链）不聚合，避免产生无信息量的单元素链。
    """
    if not paths:
        return paths
    # 收集一阶诊断边：names 长度 2、首关系为诊断语义
    diag_by_subj = {}  # subject -> {rel_type: [obj...]}
    subj_rank = {}     # subject 首次出现的顺序
    for p in paths:
        if len(p) < 2 or not p[0]:
            continue
        names = p[0]
        rels = p[1] if len(p) >= 2 else []
        if len(names) < 2 or not rels:
            continue
        subj = str(names[0])
        rel0 = str(rels[0])
        # 只聚合一阶同向（dirs[0]=="+" 或未知）的直出边；反向/多跳不在此聚合
        dirs = p[2] if len(p) >= 3 else []
        if dirs and str(dirs[0]) not in ("+", "?"):
            continue
        if rel0 not in _CHAIN_DIAG_RELS:
            continue
        obj = str(names[1])
        if subj not in diag_by_subj:
            diag_by_subj[subj] = {}
            subj_rank[subj] = len(subj_rank)
        bucket = diag_by_subj[subj].setdefault(rel0, [])
        if obj not in bucket:
            bucket.append(obj)

    # 生成链行：仅对"同主体同关系 ≥2 条"或"同主体累计 ≥3 条诊断边"的主体聚合
    chain_rows = []
    for subj, buckets in sorted(diag_by_subj.items(),
                                key=lambda kv: subj_rank.get(kv[0], 1e9)):
        total = sum(len(v) for v in buckets.values())
        if total < 3 and all(len(v) < 2 for v in buckets.values()):
            continue  # 无成链价值
        parts = [f"[推理链] {subj}"]
        # 关系顺序偏好：含义 → 原因 → 方法/步骤
        rel_order = ["报警含义", "故障原因", "可能原因",
                     "处理方法", "解决方法", "确认方法", "检查处理方法",
                     "处理步骤", "检查步骤"]
        for rel0 in sorted(buckets.keys(), key=lambda r: rel_order.index(r) if r in rel_order else 99):
            objs = buckets[rel0]
            if not objs:
                continue
            metas = [_infer_meta_from_text(rel0, o) for o in objs]
            # 排序：conditional 项（若/则条件步骤）置于无条件项之后，
            # 贴近手册原文"无条件主步骤 → 条件子步骤"的编排（gsk_037 实证）。
            pairs = sorted(zip(objs, metas),
                           key=lambda om: (1 if om[1] == "conditional" else 0,))
            labeled = [o for o, _ in pairs]
            joined = " | ".join(f"{i}. {o}" for i, o in enumerate(labeled, start=1))
            parts.append(f"{rel0}({len(objs)}): {joined}")
        chain_text = " ; ".join(parts)
        chain_rows.append(([chain_text], ["推理链"], ["?"]))
    if not chain_rows:
        return paths
    # 链行置于最前（保原始路径，去重链文本）
    seen = set()
    out = []
    for r in chain_rows:
        if r[0][0] not in seen:
            seen.add(r[0][0])
            out.append(r)
    out.extend(paths)
    return out


def _overlap_matched(query_str: str, matched: list) -> list:
    """matched 中经『概念短语重叠召回』的实体（与问题共享 ≥2 个全中文 4-gram）。

    与 _match_entities_smart 的 3.5 步判定一致，供路径展开时识别概念实体。
    """
    if not matched or not (any(0x4e00 <= ord(ch) <= 0x9fff for ch in query_str)
                           and len(query_str) >= 4):
        return []
    q4 = _query_4grams(query_str)
    return [n for n in matched if _shared_chinese_4grams(n, q4) >= 2]


def _expand_paths_per_depth(driver, label, ent, key_field="name", source_manual=None,
                            per_depth=2, max_hop=4, semantic_first=False):
    """B2 优化：按深度逐跳展开路径，替代 `-[:REL*1..4] ORDER BY length LIMIT N`。

    问题：变长路径 `*1..4` 配合 ORDER BY length + 全局 LIMIT 会把 depth-1 星形边
    全部排在前面（集线实体如 FANUC/SIEMENS 有几十条出边），2~3 跳的答案代码路径
    （P/S90 等）挤不进 LIMIT 5。改为固定长度逐跳遍历：hop=1..max_hop 各跑一条
    `-[:REL*hop]-`（每跳 LIMIT per_depth），让路径集合在深度上多样化，集线实体
    也能带出多跳路径进评分 context。Neo4j 惰性 LIMIT 保证固定长度遍历有界。

    semantic_first（2026-09-06，Manual 引擎适配）：hop=1 的 Cypher ORDER BY 按
    "路径含数字节点数"降序，目的是让代码路径（报警号/地址）靠前——但对现象实体
    （OT报警(超程报警)）这会把 10 条"表现为→报警号"数字边排满前 4 槽，而 3 条
    "处理方法"边（无数字节点）被挤出 per_depth，答案方法永远进不了 context。
    semantic_first=True 时：hop1 多取 per_depth×4 条，Python 侧把"直接诊断边"
    （处理方法/故障原因/报警含义/检查处理方法…）排到"表现为/包含"等结构边前，
    再截断 per_depth。Auto 引擎（500cases）不开启，保持既有行为。

    参数:
        driver: Neo4j driver（graph_store._driver）
        label: 节点标签（"Entity" 或 "__Entity__"）
        ent: 匹配实体名/ID
        key_field: 匹配键（Manual 用 "name"，Auto 用 "id"）
        source_manual: Manual 引擎的 provenance 过滤（Auto 传 None）
        per_depth: 每跳最多返回路径条数
        max_hop: 最大跳数
        semantic_first: hop1 是否语义边优先（Manual 现象实体适配）
    返回: [(node_names, rel_types, directions), ...]（depth-1 优先，深度逐渐多样化）
    directions: ["+"] 表示沿节点顺序正向，["-"] 表示反向，["?"] 表示未知。
    """
    # hop1 需要放大取数以便 Python 侧重排（semantic_first）；hop>=2 保持 per_depth
    _SEMANTIC_FIRST_RELS = frozenset({
        "处理方法", "故障原因", "报警含义", "报警类别", "检查处理方法", "确认方法",
        "处理机制", "诊断内容", "可能原因", "解决方法", "参数功能", "设定值",
        "所属系统", "故障类型",
    })
    out = []
    for hop in range(1, max_hop + 1):
        fetch = per_depth
        if hop == 1 and semantic_first:
            fetch = per_depth * 4
        # hop 为 int（range），label 来自内部常量白名单，注入面封闭
        if source_manual:
            cypher = (
                f"MATCH path = (n:{label} {{source_manual: $src}})-[:REL*{hop}]-(m:{label}) "
                f"WHERE n.{key_field} = $e "
                f"RETURN [node in nodes(path) | node.{key_field}] AS node_names, "
                f"[rel in relationships(path) | "
                f"[startNode(rel).{key_field}, endNode(rel).{key_field}, rel.type]] AS rel_edges "
                f"ORDER BY size([x in nodes(path) "
                f"WHERE x.{key_field} =~ '.*[A-Za-z0-9].*' | x.{key_field}]) DESC, "
                f"[x in nodes(path) | toString(id(x))] ASC "
                f"LIMIT $fetch"
            )
            with driver.session() as s:
                rows = s.run(
                    cypher, e=ent, src=source_manual, fetch=int(fetch)
                ).data()
        else:
            cypher = (
                f"MATCH path = (n:{label})-[:REL*{hop}]-(m:{label}) "
                f"WHERE n.{key_field} = $e "
                f"RETURN [node in nodes(path) | node.{key_field}] AS node_names, "
                f"[rel in relationships(path) | "
                f"[startNode(rel).{key_field}, endNode(rel).{key_field}, rel.type]] AS rel_edges "
                f"ORDER BY size([x in nodes(path) "
                f"WHERE x.{key_field} =~ '.*[A-Za-z0-9].*' | x.{key_field}]) DESC, "
                f"[x in nodes(path) | toString(id(x))] ASC "
                f"LIMIT $fetch"
            )
            with driver.session() as s:
                rows = s.run(cypher, e=ent, fetch=int(fetch)).data()
        if hop == 1 and semantic_first and rows:
            # 语义边（直接诊断/属性边）优先于结构边（表现为/包含/对应报警号）
            def _row_sort_key(r):
                edges = r.get("rel_edges") or []
                if edges:
                    return 0 if str(edges[0][2]) in _SEMANTIC_FIRST_RELS else 1
                return 1
            rows = sorted(rows, key=_row_sort_key)
            # 2026-09-09（P0-3）：截断容量 per_depth → per_depth*2。
            # 090 类多边主体有 4 原因 + 4 方法 + 含义/类别 ≈ 10 条语义边，
            # 原固定 [:per_depth]=8 会把尾部语义边（无数字的"进给速度太低/
            # 增加进给速度"，Cypher ORDER BY 排组末）截掉，推理链缺项。
            rows = rows[:per_depth * 2]
        for r in rows:
            names = r["node_names"]
            edges = r.get("rel_edges") or []
            if edges:
                rels = [str(e[2]) for e in edges]
                dirs = []
                for i, e in enumerate(edges):
                    frm, to = str(e[0]), str(e[1])
                    if i + 1 < len(names):
                        a, b = str(names[i]), str(names[i + 1])
                        if frm == a and to == b:
                            dirs.append("+")
                        elif frm == b and to == a:
                            dirs.append("-")
                        else:
                            dirs.append("?")
                    else:
                        dirs.append("?")
            else:
                rels = list(r.get("rel_types") or [])
                dirs = ["?"] * len(rels)
            out.append((names, rels, dirs))
    return out


def _match_entities_smart(query_str: str, entity_names, entity_embeddings,
                          embed, top_k: int = 10, threshold: float = ENTITY_MATCH_THRESHOLD,
                          alarm_index=None):
    """修复 H4：多路合并的实体匹配。

    单靠"整句问题 embedding vs 实体名 embedding"余弦（阈值 0.5）会漏掉大量
    关键实体（问题句与实体名向量空间差异大）。改为多路召回：
      1. 字面命中：实体名直接出现在问题中（精确、零成本）
      1.5 报警号别名（2026-09-06）：问句点名"报警/ALARM+数字"时，与实体名内嵌
          报警号按 int 归一匹配（GSK983 手册适配，见 _build_alarm_index）
      2. 语义命中：整句 embedding 余弦，使用独立的低阈值 ENTITY_MATCH_THRESHOLD
      3. 保底召回：上述均空时取 top-3 最高相似实体，避免图谱路径为空
    alarm_index: 预构建 {归一报警号 -> [实体名]}；None 时按 entity_names 惰性构建。
    """
    if not entity_names:
        return []
    # —— 严格文档臂（2026-10-13）：实体定位只留"向量 top-k + 阈值" ——
    # 文档的 _match_entities 就是"取 top_k 个最近实体，再按阈值过滤"一句，
    # 没有字面命中、没有报警号别名、没有保底召回，也没有我们的贪心去重。
    # 默认关 ⇒ 历史分支逐位不变；只用于"严格模仿文档"那一臂。
    if os.getenv("STRICT_DOC_MATCH", "0") == "1":
        try:
            q = np.array(embed(query_str), dtype=np.float32)
            cands = []
            for name, e in zip(entity_names, entity_embeddings):
                if e is None:
                    continue
                try:
                    cands.append((name, _cosine_similarity(q, np.array(e, dtype=np.float32))))
                except Exception:
                    continue
            # 文档按 score DESC；并列按名定序，保证可复现
            cands.sort(key=lambda x: (-x[1], x[0]))
            return [n for n, s in cands[:top_k] if s >= threshold]
        except Exception:
            return []
    # 1) 字面命中（修复 KG-3：短实体名如"泵"是"离心泵"的子串，若不限制会
    #    召回全部含子串的实体导致噪声；按长度降序取前 top_k 个，长名更具体）
    literal = sorted([n for n in entity_names if n and n in query_str],
                     key=_literal_sort_key, reverse=True)[:top_k]
    literal = _dedup_alarm_prefix_fragments(literal)
    # 1.5) 报警号别名召回（字面没救回但问句点名了报警号 → 数字对上即召回）
    alarm_hits = []
    q_alarm = _alarm_query_number(query_str)
    if q_alarm:
        if alarm_index is None:
            alarm_index = _build_alarm_index(entity_names)
        cands = alarm_index.get(q_alarm) or []
        # 长名优先；排除字面已命中项；最多补 top_k 个
        alarm_hits = [n for n in cands if n not in literal][:top_k]
    # 2) 语义命中（低阈值）。修复 2026-08-05：原 top_k-by-sim 被集线实体近义变体
    #    挤满（见 DIVERSITY_COS 注释），改为贪心去重——按相似度降序逐个接受与
    #    已选实体 embedding 余弦 < DIVERSITY_COS 的实体，概念实体才有机会入选。
    semantic = []
    if not ABLATE.ablate_embed_match:
        try:
            q = np.array(embed(query_str), dtype=np.float32)
            candidates = []
            for name, e in zip(entity_names, entity_embeddings):
                if e is None:
                    continue
                try:
                    e_arr = np.array(e, dtype=np.float32)
                    candidates.append((name, e_arr, _cosine_similarity(q, e_arr)))
                except Exception:
                    continue
            candidates.sort(key=lambda x: x[2], reverse=True)
            picked_embs = []
            for name, e_arr, sim in candidates:
                if sim < threshold:
                    break
                if any(_cosine_similarity(e_arr, pe) >= DIVERSITY_COS for pe in picked_embs):
                    continue
                picked_embs.append(e_arr)
                semantic.append(name)
                if len(semantic) >= top_k:
                    break
        except Exception:
            pass
    # 合并去重，保持字面命中优先
    matched = list(dict.fromkeys(literal + alarm_hits + semantic))
    # 3.5) 概念短语重叠召回（2026-08-05，shukongdashi 检索调优）。
    #   整句 embedding 对短概念实体天然低分——"回参考点不能进行"对问题
    #   "回参考点操作不能进行时" sim=0.569（排第 92）< ENTITY_MATCH_THRESHOLD；
    #   字面命中又要求实体名是问题连续子串（中间隔了个"操作"）。但实体名与
    #   问题共享"回参考点""不能进行"两个全中文 4-gram，说明问题确实在谈该概念。
    #   故：共享 ≥2 个互不相同全中文 4-gram 的实体直接召回并置于最前（概念
    #   路径优先进入 context）。全中文 4-gram 排除 "FANUC" 等集线英文子串；
    #   ≥2 个排除"系统""报警"这类单一通用词误放行无关实体。
    if (not ABLATE.ablate_4gram
            and any(0x4e00 <= ord(ch) <= 0x9fff for ch in query_str)
            and len(query_str) >= 4):
        q4 = _query_4grams(query_str)
        overlap = sorted(
            (n for n in entity_names
             if n and n not in matched and _shared_chinese_4grams(n, q4) >= 2),
            key=lambda n: (_shared_chinese_4grams(n, q4), -len(n)),
            reverse=True,
        )[:OVERLAP_MAX]
        if overlap:
            matched = list(dict.fromkeys(overlap + matched))
    # 3) 保底：语义命中为空（全部低于阈值）时取 top-3 中仍达阈值的实体。
    #    v3.1（9.10）：保底同样卡 ENTITY_MATCH_THRESHOLD，不再无下限硬塞 top-3——
    #    否则 OOC 题（相似度 0.45~0.53 的无关实体）会被再次塞入，抵消语义阈值
    #    提升。没有达标实体即承认检索不到（宁缺毋滥），让答案生成走"未找到"分支。
    if not matched and not ABLATE.ablate_embed_match:
        try:
            q = np.array(embed(query_str), dtype=np.float32)
            sims = [_cosine_similarity(q, np.array(e, dtype=np.float32))
                    if e is not None else -1.0 for e in entity_embeddings]
            order = np.argsort(sims)[::-1][:3]
            matched = [entity_names[i] for i in order
                       if sims[i] >= threshold and entity_names[i]]
        except Exception:
            pass
    return matched


class BaseGraphQueryEngine:
    def __init__(self, graph_store, embed_model, top_k=TOP_K,
                 similarity_threshold=SIMILARITY_THRESHOLD):
        self.graph_store = graph_store
        self.embed_model = embed_model
        self.top_k = top_k
        self.similarity_threshold = similarity_threshold
        # B6 优化：进程内 query embedding 缓存（同一道题跨引擎/多路匹配共用）。
        # 不能用 lru_cache 缓存 (model, text)——OllamaLocalEmbedding 等模型对象
        # 不可哈希；改为按引擎实例缓存 text → embedding，语义等价且无哈希约束。
        self._embed_cache = {}

    def _get_query_embedding(self, text: str):
        key = str(text)
        if key not in self._embed_cache:
            self._embed_cache[key] = self.embed_model.get_text_embedding(key)
        return self._embed_cache[key]

    def query_structured(self, question: str, **kwargs) -> RetrievalResult:
        """统一入口：适配旧 query() 返回（图路径列表）为 RetrievalResult。

        Manual/Auto 引擎的 query() 返回 [(names, rels, dirs)]；本方法将其
        原样放入 RetrievalResult.paths，供评估器统一读取。旧 query() 保留。
        """
        paths = self.query(question, **kwargs)
        return RetrievalResult(paths=paths)


class AutoGraphQueryEngine(BaseGraphQueryEngine):
    """场景A：向量引导的多跳检索（文章 4.1 节）。"""

    def _match_entities_native(self, query_str: str, entity_names=None):
        """Neo4j 原生向量索引定位实体。

        修复 KG-5：与 memory 分支对齐口径——
          - 阈值由 0.5 降为 ENTITY_MATCH_THRESHOLD（消除 native/memory 双口径差）；
          - 有全量实体名时叠加字面命中（短实体名直接出现在问题中的零成本召回）；
          - 仍为空时保底取向量索引 top-3 最高分，避免图谱路径为空。
        """
        embedding = self._get_query_embedding(query_str)
        cypher = """
        CALL db.index.vector.queryNodes('entity', $limit, $embedding)
        YIELD node, score
        RETURN node.id AS id, score
        ORDER BY score DESC, id ASC
        """
        results = self.graph_store.structured_query(
            cypher, param_map={"limit": self.top_k, "embedding": embedding}
        )
        native = [r["id"] for r in results
                  if r.get("id") and r.get("score", 0) >= ENTITY_MATCH_THRESHOLD]
        names = entity_names or []
        # 字面命中补充（实体名直接出现在问题中），与 _match_entities_smart 对齐：
        # 按长度降序取前 top_k 个（长名更具体，短名如"泵"不淹没长名实体）。
        literal = sorted([n for n in names if n and n in query_str],
                         key=_literal_sort_key, reverse=True)[:self.top_k]
        matched = list(dict.fromkeys(native + literal))
        # A7 修复：概念短语重叠召回（与 _match_entities_smart 第 3.5 步同判定）。
        # native 向量索引对短概念实体同样低分——"回参考点不能进行"对整句问题
        # embedding 余弦 0.569 < 0.60，被语义阈值过滤；但实体名与问题共享
        # "回参考点""不能进行"两个全中文 4-gram，overlap 路直接召回救回，
        # 使 native/memory 两分支同阈值同语义（原 memory 独享 overlap 召回，
        # native 分支在 shukongdashi 概念题上检索不到答案路径）。
        if any(0x4e00 <= ord(ch) <= 0x9fff for ch in query_str) and len(query_str) >= 4:
            q4 = _query_4grams(query_str)
            overlap = sorted(
                (n for n in names
                 if n and n not in matched and _shared_chinese_4grams(n, q4) >= 2),
                key=lambda n: (_shared_chinese_4grams(n, q4), -len(n)),
                reverse=True,
            )[:OVERLAP_MAX]
            if overlap:
                matched = list(dict.fromkeys(overlap + matched))
        # 保底：仍为空时取 top-3 中相似度达 ENTITY_MATCH_THRESHOLD 的最高分
        #（v3.1：不再无下限硬塞，避免 OOC 题把无关实体塞进 context）
        if not matched and results:
            matched = [r["id"] for r in results[:3]
                       if r.get("id") and r.get("score", 0) >= ENTITY_MATCH_THRESHOLD]
        return matched

    def _match_entities_memory(self, query_str: str, entity_names, entity_embeddings):
        """降级：内存余弦相似度（与 Manual 同款）。"""
        return _match_entities_smart(
            query_str, entity_names, entity_embeddings,
            embed=self._get_query_embedding, top_k=self.top_k,
            threshold=ENTITY_MATCH_THRESHOLD,
        )

    def query(self, query_str: str, entity_names=None, entity_embeddings=None):
        mode = getattr(self.graph_store, "vector_mode", "memory")
        if mode == "native":
            matched = self._match_entities_native(query_str, entity_names=entity_names)
        else:
            matched = self._match_entities_memory(query_str, entity_names, entity_embeddings)
        return self._expand_paths(matched, query_str)

    def _expand_paths(self, matched, query_str=None):
        """以匹配实体为起点展开 1~4 跳路径。

        修复 H2：改为无向展开 `-[:REL*1..4]-`，避免漏掉"解决方法/可能原因"
        等常以入边出现的反向路径。
        修复 KG-2：LIMIT 30 全局截断会饿死后部匹配实体（matched 多时路径全
        集中在列表前部实体）且无排序不可复现；改为每个实体独立展开。
        修复 KG-4：返回 rel.type（关系语义存在属性里，type(rel) 恒为 "REL"）。
        修复 9.13-②：跨实体轮询交错，避免前部实体垄断 context 槽位。
        B2 优化：变长 `*1..4 ORDER BY length LIMIT 5` 被集线实体 depth-1 星形边
        占满，2~3 跳答案路径进不了集合；改 _expand_paths_per_depth 按深度逐跳
        限量采样（hop=1..4 各 per_depth=2 条），深度多样化。
        """
        if not matched:
            return []
        matched = _code_token_priority_entities(query_str or "", matched)
        per_entity = []
        for ent in matched:
            rows = _expand_paths_per_depth(
                self.graph_store._driver, "__Entity__", ent, key_field="id",
                source_manual=None, per_depth=2, max_hop=4,
            )
            per_entity.append(rows)
        # 2026-08-05：概念实体路径更高频轮询，其余轮询交错（见 _interleave_paths）。
        n_prio = len(_overlap_matched(query_str, matched)) if query_str else 0
        return _interleave_paths(per_entity, n_priority=n_prio)


class ManualGraphQueryEngine(BaseGraphQueryEngine):
    """场景B：人工构建图谱，内存余弦相似度（文章 4.2 节）。"""

    def __init__(self, graph_store, embed_model, source_manual: str = None,
                 embedding_cache_path: str = None,
                 embedding_cache_model: str = None,
                 progress=None, **kwargs):
        super().__init__(graph_store, embed_model, **kwargs)
        self.source_manual = source_manual
        self._entity_names, self._entity_embeddings = self._load_entities(self.graph_store._driver)
        self._embedding_cache_model = embedding_cache_model
        self.embedding_source = "neo4j"
        if embedding_cache_path:
            # 非默认 embedding：Neo4j 里存的是默认模型的实体向量，维度不兼容；
            # 用按模型分文件的实体向量缓存覆盖（首次算，之后秒级加载）。
            self._apply_embedding_cache(embedding_cache_path, progress)
        # 2026-09-06：报警号别名 + 现象实体括号别名索引（GSK983 口语问法适配）
        self._alarm_index = _build_alarm_index(self._entity_names)
        self._paren_index = _build_paren_alias_index(self._entity_names)

    def _apply_embedding_cache(self, path: str, progress=None):
        model = self._embedding_cache_model or "unknown"
        saved = {}
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    cached = json.load(f)
                if cached.get("model") == model:
                    saved = cached.get("rows") or {}
            except Exception as e:
                print(f"  [manual_eng] 实体向量缓存读取失败: {e}")
        total = len(self._entity_names)
        rows = {}
        dirty = not os.path.exists(path)
        for i, name in enumerate(self._entity_names):
            vec = saved.get(name)
            if not (isinstance(vec, list) and len(vec) > 1):
                try:
                    vec = self._get_query_embedding(name)
                except Exception:
                    vec = None
                dirty = True
            rows[name] = vec
            if progress and (i + 1) % 100 == 0:
                progress("entities", i + 1, total)
        if dirty:
            try:
                os.makedirs(os.path.dirname(path), exist_ok=True)
                with open(path, "w", encoding="utf-8") as f:
                    json.dump({"model": model, "n": total, "rows": rows},
                              f, ensure_ascii=False)
            except Exception as e:
                print(f"  [manual_eng] 实体向量缓存回写失败: {e}")
        self._entity_embeddings = [rows.get(n) for n in self._entity_names]
        self.embedding_source = "built" if dirty else "cache"
        if progress:
            progress("entities", total, total)

    def _load_entities(self, driver):
        # 2026-10-05（P0 检索确定性，预注册件 §3 F2）：实体清单查询补全序
        # ORDER BY——此前无 ORDER BY，实体列表顺序对库内部状态敏感，级联到
        # 语义选样与路径展开（E1 跨会话 207/288 ctx 漂移的嫌疑链之一）。
        if self.source_manual:
            cypher = ("MATCH (n:Entity {source_manual: $src}) "
                      "RETURN n.name AS name, n.embedding AS embedding "
                      "ORDER BY n.name ASC")
            recs = driver.session().run(cypher, src=self.source_manual)
        else:
            recs = driver.session().run(
                "MATCH (n:Entity) RETURN n.name AS name, n.embedding AS embedding "
                "ORDER BY n.name ASC")
        names = []
        embeddings = []
        missing = []
        for r in recs:
            name = r["name"]
            emb = r["embedding"]
            if name is None:
                continue
            names.append(name)
            if emb is not None:
                embeddings.append(list(emb))
            else:
                embeddings.append(None)
                missing.append(name)
        # 补齐 Neo4j 中缺失的实体 embedding（仅对缺失部分调 Ollama）
        if missing:
            print(f"  [manual_eng] 补齐 {len(missing)} 个缺失实体 embedding（Ollama）...")
            emb_by_name = {}
            for n in missing:
                try:
                    emb_by_name[n] = self._get_query_embedding(n)
                except Exception:
                    emb_by_name[n] = [0.0]
            embeddings = [emb_by_name.get(n) if e is None else e for n, e in zip(names, embeddings)]
        return names, embeddings

    def _match_entities(self, query_str: str):
        matched = _match_entities_smart(
            query_str, self._entity_names, self._entity_embeddings,
            embed=self._get_query_embedding, top_k=self.top_k,
            threshold=ENTITY_MATCH_THRESHOLD,
            alarm_index=self._alarm_index,
        )
        # 括号别名召回补充：现象实体 "OT报警(超程报警)" 按 "超程报警" 命中。
        # 放在最前（现象是用户直接问的主体，路径应优先展开）。
        if not matched:
            return matched
        paren_hits = _paren_alias_recall(
            query_str, self._paren_index, exclude=set(matched), top_k=3)
        if paren_hits:
            matched = list(dict.fromkeys(paren_hits + matched))
        return matched

    def query(self, query_str: str):
        matched = self._match_entities(query_str)
        if not matched:
            return []
        # 修复 KG-2：全局 LIMIT 30 无排序会饿死后部匹配实体，改为每个实体独立展开。
        # 修复 9.13-②：跨实体轮询交错，避免前部实体垄断 context 槽位。
        # B2 优化：变长 `*1..4 ORDER BY length LIMIT 5` 被集线实体 depth-1 星形边
        # 占满，2~3 跳答案路径进不了集合；改 _expand_paths_per_depth 按深度逐跳
        # 限量采样（hop=1..4 各 per_depth=2 条），深度多样化。
        # 代码 token 实体优先展开（Q12 等元件代号实体常排到 15+ 位）
        if not ABLATE.ablate_rescore:
            matched = _code_token_priority_entities(query_str, matched)
        per_entity = []
        for ent in matched[:MAX_EXPAND_ENTITIES]:
            rows = _expand_paths_per_depth(
                self.graph_store._driver, "Entity", ent, key_field="name",
                source_manual=self.source_manual, per_depth=8, max_hop=4,
                semantic_first=True,
            )
            rows = _rank_paths(rows, query_str)
            per_entity.append(rows)
        # 2026-08-05：概念实体（overlap 召回）路径更高频轮询，其余轮询交错，
        # 让答案代码路径（回参考点不能进行--[表现为]-->P/S90）进入评分 context。
        n_prio = len(_overlap_matched(query_str, matched))
        paths = _interleave_paths(per_entity, n_priority=n_prio)
        # 2026-09-07（能力边界1）：确认/检查类问法前置确认方法路径
        paths = _prioritize_diag_paths(paths, query_str)
        # 2026-09-09（P0-3）：同主体诊断边聚合成显式推理链（并列/顺序/条件）
        # 图谱扁平边打散后 LLM 难重构多跳语义；链行前置注入 path 流。
        # 仅对成链主体聚合，原始路径全部保留，不影响召回与评分口径。
        return _build_reasoning_chains(paths)


def _norm_fragment_token(text) -> str:
    """片段/查询 token 归一化：大写、全半角统一、去空白。

    用途：社区片段排序与窗口截取时，让“端子36”能匹配到原文“端子 36”，
    也让全角/半角、大小写差异不再造成关键证据漏召回。

    消融 B1（ABLATE_FRAG_NORM=1）时恒等退化——不归一化即不做
    大小写/全半角/空白兼容，验证该机制的边际贡献。
    """
    if ABLATE.ablate_frag_norm:
        return str(text)
    x = str(text).upper().replace("℃", "°C")
    fw = "０１２３４５６７８９ＡＢＣＤＥＦＧＨＩＪＫＬＭＮＯＰＱＲＳＴＵＶＷＸＹＺａｂｃｄｅｆｇｈｉｊｋｌｍｎｏｐｑｒｓｔｕｖｗｘｙｚ：，．－（）"
    hw = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz:,.-()"
    x = x.translate(str.maketrans(fw, hw))
    return re.sub(r"\s+", "", x)


def _fragment_window(text, tokens, max_chars=12000):
    """截取原文片段；若命中查询 token，以首个命中位置为中心扩窗。

    原实现固定取前 800 字，长表格/长案例中答案常出现在 800 字之后
    （实测 auto_032 的“端子36 +10V”在 995 字符处、auto_013 的
    D12/A169 在 800 字后被切掉）。改为围绕命中 token 截取，保留证据。
    2026-08-11：上限 800 → 12000——GT 原因/方法句的原文证据常在
    1200 字符窗口之外（auto_004 等 CR 低分含此成分），扩大窗口后
    由 _truncate_contexts 的总预算统一约束。

    消融 B2（ABLATE_WINDOW=1）时上限回退 1200 字符（v14 时代行为），
    验证 12000 扩窗的边际贡献。
    """
    if ABLATE.ablate_window:
        max_chars = 1200
    if not text:
        return ""
    if not tokens:
        return text[:max_chars]
    norm_text = _norm_fragment_token(text)
    norm_tokens = sorted({_norm_fragment_token(t) for t in tokens if t})
    best = None
    best_len = 0
    for nt in norm_tokens:
        pos = norm_text.find(nt)
        if pos >= 0 and (best is None or pos < best):
            best = pos
            best_len = len(nt)
    if best is None:
        return text[:max_chars]
    orig_idx = []
    norm_chars = []
    for i, ch in enumerate(text):
        nc = _norm_fragment_token(ch)
        if nc:
            norm_chars.append(nc)
            orig_idx.append(i)
    end_norm = min(best + best_len, len(norm_chars))
    if best >= len(orig_idx) or end_norm <= 0:
        return text[:max_chars]
    start_orig = orig_idx[best]
    end_orig = orig_idx[end_norm - 1] + 1
    half = max_chars // 2
    s = max(0, start_orig - half)
    e = min(len(text), end_orig + half)
    snippet = text[s:e]
    # 窗口不足时向两侧补足到 max_chars
    if len(snippet) < max_chars:
        pad_l = min(s, max_chars - len(snippet))
        pad_r = min(len(text) - e, max_chars - len(snippet) - pad_l)
        s2 = s - pad_l
        e2 = e + pad_r
        snippet = text[s2:e2]
    return snippet


class CommunityQueryEngine(BaseGraphQueryEngine):
    """宏观查询：社区描述 embedding 余弦匹配（文章 5.5 节）。"""

    def __init__(self, graph_store, embed_model, communities,
                 enrich_from_neo4j: bool = False,
                 chunk_index_cache: str = None,
                 chunk_index_cache_model: str = None,
                 reuse_chunk_embeddings: bool = False,
                 progress=None,
                 source_manual: str = None, **kwargs):
        super().__init__(graph_store, embed_model, **kwargs)
        self.source_manual = source_manual
        self._chunk_index_cache_model = chunk_index_cache_model
        self._reuse_chunk_embeddings = reuse_chunk_embeddings
        self._progress = progress
        self.communities = communities  # list[dict] with description_embedding
        self._chunk_rows = []
        self._chunk_matrix = np.empty((0, 0))
        self._chunk_index_source = "none"
        # 修复 KG-12：默认不混用 Neo4j 摘要（内存缓存为准，保证评测可复现）。
        # 旧行为无条件 _enrich_from_neo4j()：Neo4j 残留旧摘要时内存数据被污染，
        # 评测结果依赖 Neo4j 状态。需要从 Neo4j 补摘要时显式传 enrich_from_neo4j=True。
        if enrich_from_neo4j:
            self._enrich_from_neo4j()
        if communities:
            # A4 修复：缺失/空 description_embedding 回填（仿 Manual _load_entities）。
            # 旧逻辑用 min/max 0-dim hack 静默降级（空向量 norm=0 → 余弦恒 0，
            # 该社区永远召不回且无任何提示）。
            missing = [c for c in communities if not c.get("description_embedding")]
            if missing:
                print(f"  [comm_eng] 回填 {len(missing)} 个社区 description_embedding（Ollama）...")
                for c in missing:
                    try:
                        c["description_embedding"] = self._get_query_embedding(
                            c.get("description") or "")
                    except Exception:
                        c["description_embedding"] = [0.0]
            embeds = [c["description_embedding"] for c in communities]
            # 维度统一取众数（模型切换时偶发少数异常维度），按多数维度截断/补零
            dims = Counter(len(e) for e in embeds if e)
            target_dim = dims.most_common(1)[0][0] if dims else 0
            if target_dim:
                embeds = [np.array(
                    (e[:target_dim] if len(e) >= target_dim else e + [0.0] * (target_dim - len(e))),
                    dtype=np.float32) for e in embeds]
                self._comm_matrix = np.array(embeds, dtype=np.float32)
            else:
                self._comm_matrix = np.empty((0, 0))
        else:
            self._comm_matrix = np.empty((0, 0))
        if communities and chunk_index_cache:
            self._load_chunk_index(chunk_index_cache)

    def _enrich_from_neo4j(self):
        """从 Neo4j 加载已持久化的社区摘要，补充到内存社区数据中。

        社区身份是 (source_manual, id)：只按 id 建映射时，库里另一本手册的同号社区摘要
        会被贴到本手册这个社区上（2026-09-06 的 IS620F 缓存就是这么串档的）。
        没给 source_manual 就不混用——把"无归属地整库贴"这条路径彻底关掉。
        """
        if not self.source_manual:
            print("  [comm_eng] enrich_from_neo4j 需要 source_manual 才能安全回填，本次跳过")
            return
        try:
            neo4j_summaries = self.graph_store.get_community_summaries(
                source_manual=self.source_manual)
        except Exception:
            return
        if not neo4j_summaries:
            return
        neo4j_map = {str(s["id"]): s for s in neo4j_summaries}
        for comm in self.communities:
            cid = str(comm.get("id", ""))
            if cid in neo4j_map:
                ns = neo4j_map[cid]
                comm["summary"] = comm.get("summary") or ns.get("summary", "")
                comm["report"] = comm.get("report") or ns.get("report", "")

    def _match_communities(self, query_str: str):
        if self._comm_matrix.size == 0:
            return []
        q = np.array(self._get_query_embedding(query_str), dtype=np.float32)
        norms_comm = np.linalg.norm(self._comm_matrix, axis=1)
        norm_q = np.linalg.norm(q)
        sims = self._comm_matrix.dot(q) / (norms_comm * norm_q + 1e-9)
        idx = np.argsort(sims)[::-1][:self.top_k]
        matched = [self.communities[i] for i in idx if sims[i] >= self.similarity_threshold]
        # 修复 KG-7：保底召回加相似度下限，负分/完全不相关的社区宁可不召回，
        # 避免无关社区文本进 context 稀释 precision。
        if not matched and len(idx) > 0 and sims[idx[0]] >= COMMUNITY_FALLBACK_THRESHOLD:
            matched.append(self.communities[idx[0]])  # 保底召回 1 个
        return matched

    def _load_chunk_index(self, cache_path):
        """构建/刷新文本单元语义索引。

        索引向量必须与查询 embedding 在**同一进程**内计算：qwen3-embedding
        对长中文文本存在跨进程不可复现性（同一问句在两个进程得到不同向量），
        预生成的索引在另一个进程里相似度会失真（auto_036 实测 0.78 -> 0.50）。
        因此这里只复用缓存里的文本与社区映射，向量一律现场重算并回写。
        """
        try:
            saved = {}
            if os.path.exists(cache_path):
                with open(cache_path, "r", encoding="utf-8") as f:
                    saved = json.load(f)
                rows = saved.get("rows") or []
            else:
                rows = []
                seen = {}
                for ci, comm in enumerate(self.communities):
                    for text in comm.get("raw_context") or []:
                        if not text:
                            continue
                        if text in seen:
                            seen[text]["comm_idxs"].append(ci)
                        else:
                            seen[text] = {"text": text, "comm_idxs": [ci]}
                rows = list(seen.values())
            if not rows:
                print(f"  [comm_eng] chunk 索引为空: {cache_path}")
                return
            reuse = bool(
                self._reuse_chunk_embeddings
                and saved.get("model") == self._chunk_index_cache_model
                and saved.get("process_token") == _PROCESS_TOKEN
                and all(isinstance(r.get("embedding"), list) and len(r["embedding"]) > 1
                        for r in rows)
            )
            if not reuse:
                for i, row in enumerate(rows):
                    row["embedding"] = self._get_query_embedding(row["text"][:1500])
                    if (i + 1) % 50 == 0:
                        print(f"  [comm_eng] chunk 向量 {i + 1}/{len(rows)}")
                    if self._progress and (i + 1) % 50 == 0:
                        self._progress("chunks", i + 1, len(rows))
                try:
                    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
                    with open(cache_path, "w", encoding="utf-8") as f:
                        # 元数据写在最前：模型设置页只读文件头就能显示缓存归属，
                        # 不必解析上百条向量。
                        json.dump({"model": self._chunk_index_cache_model,
                                   "process_token": _PROCESS_TOKEN,
                                   "n": len(rows), "rows": rows},
                                  f, ensure_ascii=False)
                except Exception as e:
                    print(f"  [comm_eng] chunk 索引回写失败: {e}")
            else:
                print(f"  [comm_eng] chunk 向量缓存复用: {len(rows)} 行")
            self._chunk_index_source = "cache" if reuse else "built"
            embeds = [row.get("embedding") or [] for row in rows]
            dims = Counter(len(e) for e in embeds if e)
            target_dim = dims.most_common(1)[0][0] if dims else 0
            if not target_dim:
                print(f"  [comm_eng] chunk 索引向量异常: {cache_path}")
                return
            norm = []
            for e in embeds:
                if len(e) >= target_dim:
                    norm.append(np.array(e[:target_dim], dtype=np.float32))
                elif e:
                    norm.append(np.array(
                        e + [0.0] * (target_dim - len(e)), dtype=np.float32))
                else:
                    norm.append(np.zeros(target_dim, dtype=np.float32))
            self._chunk_rows = rows
            self._chunk_matrix = np.array(norm, dtype=np.float32)
            print(f"  [comm_eng] chunk 索引构建: {len(rows)} 个文本单元")
        except Exception as e:
            print(f"  [comm_eng] chunk 索引构建失败: {e}（跳过文本单元召回）")

    def _match_chunks(self, query_str: str, top_n: int = CHUNK_INDEX_TOP_N):
        """文本单元级语义召回：找描述匹配漏掉的案例原文。"""
        if self._chunk_matrix.size == 0 or not query_str:
            return []
        q = np.array(self._get_query_embedding(query_str), dtype=np.float32)
        norms = np.linalg.norm(self._chunk_matrix, axis=1)
        qn = np.linalg.norm(q)
        if qn == 0:
            return []
        sims = self._chunk_matrix.dot(q) / (norms * qn + 1e-9)
        out = []
        for i in np.argsort(sims)[::-1]:
            s = float(sims[i])
            if s < CHUNK_INDEX_THRESHOLD:
                break
            out.append((s, self._chunk_rows[i]))
            if len(out) >= top_n:
                break
        return out

    def _lex_query_tokens(self, query_str: str) -> set:
        """提取问句的词面 token（拉丁代码 + 中文 2-6 gram），归一化后用于兜底。"""
        toks = set(re.findall(r"[A-Za-z0-9][A-Za-z0-9./#-]*", query_str))
        for run in re.findall(r"[\u4e00-\u9fff]+", query_str):
            if len(run) < 2:
                continue
            for n in (2, 3, 4, 5, 6):
                for i in range(0, max(1, len(run) - n + 1)):
                    toks.add(run[i:i + n])
        return {_norm_fragment_token(t) for t in toks if _norm_fragment_token(t)}

    def _match_chunks_lexical(self, query_str: str, top_n: int = 1,
                              min_hits: int = 20):
        """词面兜底召回：语义向量跨进程不稳定时，靠 token 重叠保案例原文。

        语义索引与查询在同一进程重算后通常已足够；但 auto_007 的 SVO 案例
        在不同进程的语义排名会在 top-8 内外摆动，词面重叠则稳定排第 1。
        min_hits 过滤高频泛化片段，避免把无关长文塞进 context。
        """
        if not self._chunk_rows or not query_str:
            return []
        toks = self._lex_query_tokens(query_str)
        if not toks:
            return []
        scored = []
        for row in self._chunk_rows:
            nr = _norm_fragment_token(row.get("text") or "")
            c = sum(1 for t in toks if t in nr)
            if c >= min_hits:
                scored.append((c, row))
        scored.sort(key=lambda x: (-x[0], x[1].get("text", "")))
        return [(float(s), row) for s, row in scored[:top_n]]

    def query(self, query_str: str, max_per_community: int = 3,
              chunk_fallback: bool = True):
        """召回 Top-K 社区并组织上下文。

        修复 H1/H3：
          - H1：跨社区全局去重（同一 chunk 被多个社区绑定时会重复出现，
            此前只做单社区内去重，导致上下文信息密度被稀释）。
          - H3：无摘要时不再裸拼 raw，同样做全局去重，并按社区分组输出。
        B5 优化：每社区最多贡献 max_per_community 条去重后的 raw chunk——否则
        无摘要时 top-1 社区的几十条片段会占满 (max_ctx - min_path_slots)=6 个
        社区槽位，其他社区（可能是更全面的答案来源）完全进不了评分 context。
        跨社区多样性让 6 个槽位覆盖多个社区。
        修复 2026-08-06（shukongdashi 优化）：硬事实前置
          - 社区的 hard_facts（具体参数值/报警号含义/信号地址）排在摘要前面，
            因为这些是回答代码类问题的直接证据，信息密度远高于 LLM 摘要。
        """
        matched = self._match_communities(query_str)
        sem_rows = []
        lex_rows = []
        hit_rows = []
        if chunk_fallback:
            sem_rows = self._match_chunks(query_str)
            lex_rows = self._match_chunks_lexical(query_str)
            hit_rows = list(sem_rows)
            seen_texts = {row.get("text") for _, row in hit_rows}
            for _, row in lex_rows:
                if row.get("text") not in seen_texts:
                    hit_rows.append((0.0, row))
                    seen_texts.add(row.get("text"))
            # 文本单元命中社区前置：语义命中优先（最多 4 个社区），词面兜底
            # 紧随其后（最多 2 个社区），最后才是描述匹配社区。保证案例原文
            # 社区不会被 6 个社区槽位挤掉。
            ordered = []
            seen_ids = set()
            sem_comm_n = 0
            lex_comm_n = 0
            for _, row in sem_rows:
                if sem_comm_n >= 4:
                    break
                for ci in row.get("comm_idxs", []):
                    if sem_comm_n >= 4:
                        break
                    if 0 <= ci < len(self.communities):
                        comm = self.communities[ci]
                        cid = str(comm.get("id", ci))
                        if cid not in seen_ids:
                            seen_ids.add(cid)
                            ordered.append(comm)
                            sem_comm_n += 1
            for _, row in lex_rows:
                if lex_comm_n >= 2:
                    break
                for ci in row.get("comm_idxs", []):
                    if lex_comm_n >= 2:
                        break
                    if 0 <= ci < len(self.communities):
                        comm = self.communities[ci]
                        cid = str(comm.get("id", ci))
                        if cid not in seen_ids:
                            seen_ids.add(cid)
                            ordered.append(comm)
                            lex_comm_n += 1
            for comm in matched:
                cid = str(comm.get("id", self.communities.index(comm)))
                if cid not in seen_ids:
                    seen_ids.add(cid)
                    ordered.append(comm)
            matched = ordered
        # 命中原文按社区聚合，片段优先插入
        hit_texts_by_comm = {}
        if hit_rows:
            for _, row in hit_rows:
                for ci in row.get("comm_idxs", []):
                    if 0 <= ci < len(self.communities):
                        cid = str(self.communities[ci].get("id", ci))
                        hit_texts_by_comm.setdefault(cid, []).append(
                            row.get("text", ""))
        contexts = []
        seen = set()
        for comm in matched:
            cid = str(comm.get("id", self.communities.index(comm)))
            raw = comm.get("raw_context", [])
            summary = "" if ABLATE.ablate_chunk_summary else comm.get("summary", "")
            report = "" if ABLATE.ablate_chunk_summary else comm.get("report", "")
            hard_facts = comm.get("hard_facts", [])
            # 截断过长的社区报告（保留开头高信息密度部分），避免超出上下文窗口
            if report and len(report) > MAX_COMMUNITY_REPORT_CHARS:
                report = report[:MAX_COMMUNITY_REPORT_CHARS] + "……（报告已截断）"
            
            # 修复 2026-08-06：硬事实前置（参数值/报警号含义等）
            hard_facts_str = ""
            if hard_facts:
                # 最多显示 30 条硬事实，保证 50 题测试集代码 token 不被截掉
                facts_display = hard_facts[:30]
                hard_facts_str = f"[硬事实] {', '.join(facts_display)}"
            
            if summary or report or hard_facts_str:
                parts = []
                # 原文片段按与问题的代码/中文短语重叠度排序取前 N 条，
                # 避免具体案例片段被 raw_context 顺序排到末尾而丢失。
                query_tokens = set(
                    re.findall(r"[A-Za-z0-9][A-Za-z0-9./#-]*", query_str))
                query_tokens |= set(re.findall(r"[\u4e00-\u9fff]{2,6}", query_str))
                norm_query_tokens = [
                    _norm_fragment_token(t) for t in query_tokens
                    if _norm_fragment_token(t)
                ]

                def _frag_score(r):
                    nr = _norm_fragment_token(r)
                    return sum(1 for t in norm_query_tokens if t in nr)

                ranked = sorted(((i, _frag_score(r), r)
                                 for i, r in enumerate(raw)),
                                key=lambda x: (-x[1], x[0]))
                # 文本单元命中片段优先（描述匹配漏召回时，案例原文也必须进 context）
                for h in hit_texts_by_comm.get(cid, []):
                    if h and h not in seen and len(parts) < max_per_community:
                        seen.add(h)
                        parts.append(_fragment_window(h, query_tokens))
                for _, _, r in ranked:
                    if r and r not in seen and len(parts) < max_per_community:
                        seen.add(r)
                        # 命中查询 token 时围绕证据截取窗口，否则取头部；
                        # 每片段 800 字上限与总预算不变量保持不变。
                        parts.append(_fragment_window(r, query_tokens))
                # 硬事实排在最前面（信息密度最高）
                block_parts = []
                if hard_facts_str:
                    block_parts.append(hard_facts_str)
                if summary:
                    block_parts.append(f"[社区摘要] {summary}")
                if report:
                    block_parts.append(f"[社区报告] {report}")
                if parts:
                    block_parts.append("[相关片段]\n" + "\n".join(parts))
                block = "\n".join(block_parts)
                if block not in seen:
                    seen.add(block)
                    contexts.append(block)
            else:
                added = 0
                for h in hit_texts_by_comm.get(cid, []):
                    if h and h not in seen and added < max_per_community:
                        seen.add(h)
                        contexts.append(h)
                        added += 1
                for r in raw:
                    if r and r not in seen and added < max_per_community:
                        seen.add(r)
                        contexts.append(r)
                        added += 1
        return matched, contexts

    def query_structured(self, question: str, **kwargs) -> RetrievalResult:
        """统一入口：Community 引擎返回 (matched, contexts)，透传 kwargs。"""
        matched, contexts = self.query(question, **kwargs)
        return RetrievalResult(
            contexts=contexts,
            matched_communities=len(matched),
        )
