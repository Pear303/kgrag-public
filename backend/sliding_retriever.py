# -*- coding: utf-8 -*-
"""滑窗 chunk dense 检索器（方案 C + 方向 B）：
基于 parse_manual_chunks_sliding_window 的 363 个滑窗 chunk 建 embedding 索引，
提供 (query) -> [(score, row)] 接口。

方向 B 增强（2026-08-23）：query 含代码 token（DGN720-727 / G100.0 等）时，
先做符号匹配——精确命中含该 token 的 chunk 并优先返回（替代 dense 相似度，
因为 dense 对短代码 token 效果差）。无代码 token 时回退纯 dense。"""
import os, json, re, numpy as np

# 代码 token 正则（与 evaluator._QUERY_CODE_RE 对齐）
_CODE_RE = re.compile(
    r"DGN\d+[-\d]*|[GXYFKMOT]\d+[.#-]?\d*|P/S\d+|SV\d+|ALM-?\d+[A-Za-z0-9]*|"
    r"NC\d+[A-Za-z0-9]*|F\d+\.\d+|F\d{2,}|G\d+[.#-]\d+|M\d{2,}|T\d{2,}|"
    r"V[RW]\d|CH\d+|S\d{1,2}|CP\d+|KA\d+|QF\d+|SK\d+|PRM\d+|端子\d+|[A-Z]{2,}[0-9][0-9A-Z]*"
)

def _code_tokens(query_str):
    return list(dict.fromkeys(m.group(0) for m in _CODE_RE.finditer(query_str or "")))


# 中文式编号（2026-09-15 新增）：例147 / 例802A / 表2-4 / 图3-1 / 参数0074
# 依据 analysis/residual12_adjacency.py：残余低分题里 10/12 是"编号指代型"提问，
# 编号本身只命中 1–2 个 chunk（高度选择），但 gold 正文常位于**编号块之后 ±5 块内**
#（6/9 在后 5 块内、8/9 在双向 5 块内）——故命中编号后扩窗合并，而非直接返回编号块。
#
# OPT_ID_KINDS（2026-09-15 新增，默认 case）：
#   all  = 例/表/图/参数 全部触发
#   case = 仅"例N"触发。依据 analysis/id_expand_volume_control.py 的单题复核：
#          "表N"锚点会误命中（表5-6 的锚点与答案所在表不同），合并块占掉 1 个槽位后
#          把原本排在第 K 位的 gold 挤出（实测 g3_new_096 由命中转为不中）；
#          而残余题的收益 9/12 来自"例N"引用，收敛到 case 可保留收益、去掉该类误命中。
_CN_ID_KIND_RE = {
    'case': re.compile(r"例\s?\d+[A-Za-z]?"),
    'table': re.compile(r"表\s?\d+\s?[-–—]?\s?\d*"),
    'figure': re.compile(r"图\s?\d+\s?[-–—]?\s?\d*"),
    'param': re.compile(r"参数\s?\d+"),
}
_CN_ID_RE = re.compile(
    r"例\s?\d+[A-Za-z]?|表\s?\d+\s?[-–—]?\s?\d*|图\s?\d+\s?[-–—]?\s?\d*|参数\s?\d+")


def _cn_identifiers(query_str, kinds="all"):
    if kinds and kinds != "all" and kinds in _CN_ID_KIND_RE:
        pat = _CN_ID_KIND_RE[kinds]
    else:
        pat = _CN_ID_RE
    return list(dict.fromkeys(
        m.group(0).replace(" ", "").replace("\u3000", "")
        for m in pat.finditer(query_str or "")))


class SlidingWindowDenseRetriever:
    def __init__(self, chunk_texts, embed_fn, threshold=0.3, top_n=12):
        self._embed = embed_fn
        self._threshold = threshold
        self._top_n = top_n
        self._build(chunk_texts)

    def _build(self, chunk_texts):
        rows = []
        embeds = []
        fail_streak = 0
        embed_dim = 0  # 动态维度：取首个成功 embedding 的长度（勿硬编码，
        for i, t in enumerate(chunk_texts):  # qwen3-embedding 实际 1024 而非旧默认 384）
            rows.append({"text": t})
            try:
                vec = np.array(self._embed(t[:1500]), dtype=np.float32)
                if embed_dim == 0:
                    embed_dim = len(vec)
                embeds.append(vec)
                fail_streak = 0
            except Exception:
                fail_streak += 1
                # 失败占位：全零向量 norm=0 → 检索时 sims=0 被 threshold 过滤
                embeds.append(np.zeros(embed_dim or 1024, dtype=np.float32))
                # Ollama 挂起时每 chunk 180s（60s×3重试），连续失败快速放弃
                if fail_streak >= 3:
                    print(f"[SlidingRetriever] 连续 {fail_streak} 个 embedding 失败，"
                          f"Ollama 可能挂起，中止构建（已 {i+1}/{len(chunk_texts)}）")
                    self._rows = []
                    self._matrix = np.zeros((0, embed_dim or 1), dtype=np.float32)
                    self._norms = np.zeros(0)
                    return
            if (i + 1) % 60 == 0:
                print(f"[SlidingRetriever] embedding {i+1}/{len(chunk_texts)}")
        self._rows = rows
        if embeds:
            dim = max(len(e) for e in embeds)
            mat = []
            for e in embeds:
                if len(e) < dim:
                    e = np.concatenate([e, np.zeros(dim - len(e), dtype=np.float32)])
                mat.append(e)
            self._matrix = np.array(mat, dtype=np.float32)
            self._norms = np.linalg.norm(self._matrix, axis=1)
        else:
            self._matrix = np.zeros((0, embed_dim or 1), dtype=np.float32)
            self._norms = np.zeros(0)

    def __call__(self, query_str):
        """对外入口：原逻辑 + 可选的结构感知扩窗（OPT_ID_EXPAND=1 时启用）。

        开关关闭时（默认）返回值与历史实现逐位一致。
        """
        out = self._forward_base(query_str)
        if os.getenv("OPT_ID_EXPAND", "0") != "1":
            return out
        if not out or not query_str:
            return out
        ident = _cn_identifiers(query_str, os.getenv("OPT_ID_KINDS", "case"))
        if not ident:
            return out
        try:
            _w = int(os.getenv("OPT_ID_WINDOW", "3"))
            _slots = int(os.getenv("OPT_ID_SLOTS", "2"))
        except ValueError:
            _w, _slots = 3, 2
        # 1) 定位含编号的锚点块
        anchors = []
        for i, row in enumerate(self._rows):
            t = (row.get("text") or "").replace(" ", "").replace("\u3000", "")
            if any(k in t for k in ident):
                anchors.append(i)
        if not anchors:
            return out
        # 2) 每个锚点向两侧扩窗 _w 块并合并为一个上下文（同一结构块）
        blocks, covered = [], set()
        for a in anchors[:_slots]:
            lo, hi = max(0, a - _w), min(len(self._rows), a + _w + 1)
            idxs = [j for j in range(lo, hi) if j not in covered]
            if not idxs:
                continue
            covered.update(idxs)
            blocks.append({"text": " ".join(self._rows[j]["text"] for j in idxs),
                           "merged_from": idxs, "id_anchor": a})
        if not blocks:
            return out
        # 3) 扩窗块置于最前；原序中已被合并覆盖的单块去除（其内容已在合并块内，不丢召回）
        head = [(3.0 - 0.01 * k, b) for k, b in enumerate(blocks)]
        covered_texts = {self._rows[j]["text"] for j in covered}
        tail = [(s, r) for s, r in out if r.get("text") not in covered_texts]
        return (head + tail)[:self._top_n]

    def _forward_base(self, query_str):
        if self._matrix.size == 0 or not query_str:
            return []
        # 方向 B：符号优先
        tokens = _code_tokens(query_str)
        if tokens:
            sym = []
            rest_idx = []
            # 匹配时忽略 chunk 内空格（"DGN720 - 727" 与 "DGN720-727" 等价）
            for i, row in enumerate(self._rows):
                txt = row.get("text") or ""
                txt_ns = txt.replace(" ", "").replace("\u3000", "")
                if any(t in txt or t in txt_ns for t in tokens):
                    hits = sum(1 for t in tokens if t in txt or t in txt_ns)
                    sym.append((hits, i, row))
                else:
                    rest_idx.append(i)
            if sym:
                sym.sort(key=lambda x: -x[0])
                # OPT_SYM_CAP（2026-09-15 新增，默认 0 = 历史行为不变）：
                # 实测（analysis/verify_symbolic_demotion.py）漏检题的 gold 块
                # 在纯 dense 中位排名第 2、在"符号优先"序中被推到第 11；
                # 符号命中 ≥12 时把 12 个槽位全占满，dense 结果完全进不来
                # （30 道漏检题里 10 道如此，生产 top2 含 gold 的题 0/30）。
                # 置 cap>0 时：符号最多占 cap 个槽，其余槽用 dense 顺序填充
                # （dense 填充不再局限于"非符号块"，以保留符号块的次优命中）。
                try:
                    _cap = int(os.getenv("OPT_SYM_CAP", "0"))
                except ValueError:
                    _cap = 0
                # OPT_FUSE_LAMBDA（2026-10-16 新增，默认 0 = 历史行为不变；仅 cap>0 时生效）：
                # 历史符号优先给符号命中行写死 2.0+0.1*h >= 2.0，而 dense 余弦 <= 1.0，
                # 于是符号行恒压 dense，dense 永远上不来。置 lambda>0 时改为同尺度融合
                # s_i = cos_i + lambda * 1[第 i 行有符号命中]（所有命中行都加，即 scope=allh
                # 的字面读法），符号不再享有量纲特权，只在同一把尺上加一个小偏置，
                # 全体降序取 top_n，不再做"符号头 + dense 补足"两段拼接。
                # 零成本标定见 files/_abl_logs/_xline_fuselambda_20261016.json：
                # 全量 Δ 精确上界 +0.0166（地板 0.036），lambda>0.25 后收益单调变差，故限幅。
                try:
                    _fuse = float(os.getenv("OPT_FUSE_LAMBDA", "0"))
                except ValueError:
                    _fuse = 0.0
                if _cap > 0 and (_fuse < 0.0 or _fuse > 0.25):
                    raise ValueError(
                        "OPT_FUSE_LAMBDA 必须在 [0, 0.25] 内，实得 %r" % (_fuse,))
                if _cap > 0:
                    head = sym[:_cap]
                    out = [(2.0 + 0.1 * h, row) for h, _, row in head]
                    used = {i for _, i, _ in head}
                    q = np.array(self._embed(query_str[:1500]), dtype=np.float32)
                    qn = np.linalg.norm(q)
                    if qn > 0:
                        sims = self._matrix.dot(q) / (self._norms * qn + 1e-9)
                        if _fuse > 0.0:
                            # 显式 float64 累加：与标定件 files/_oneoff/xline/
                            # _xline_fuselambda_20261016.py 的 dtype 逐位对齐，
                            # 免掉 float32+λ 的尾数舍入造成的取序漂移。
                            fused = np.array(sims, dtype=np.float64)
                            for _, i, _ in sym:
                                fused[i] = fused[i] + _fuse
                            fout = []
                            for i in np.argsort(fused)[::-1]:
                                fout.append((float(fused[i]), self._rows[i]))
                                if len(fout) >= self._top_n:
                                    break
                            return fout[:self._top_n]
                        for i in np.argsort(sims)[::-1]:
                            if i in used:
                                continue
                            out.append((float(sims[i]), self._rows[i]))
                            used.add(i)
                            if len(out) >= self._top_n:
                                break
                    return out[:self._top_n]
                out = [(2.0 + 0.1 * h, row) for h, _, row in sym[:self._top_n]]
                # 不足 top_n 时用 dense 补
                if len(out) < self._top_n:
                    q = np.array(self._embed(query_str[:1500]), dtype=np.float32)
                    qn = np.linalg.norm(q)
                    if qn > 0:
                        sims = self._matrix.dot(q) / (self._norms * qn + 1e-9)
                        for i in np.argsort(sims)[::-1]:
                            if i in rest_idx:
                                out.append((float(sims[i]), self._rows[i]))
                            if len(out) >= self._top_n:
                                break
                return out[:self._top_n]
        # 无代码 token 或符号无命中：纯 dense
        q = np.array(self._embed(query_str[:1500]), dtype=np.float32)
        qn = np.linalg.norm(q)
        if qn == 0:
            return []
        sims = self._matrix.dot(q) / (self._norms * qn + 1e-9)
        out = []
        for i in np.argsort(sims)[::-1]:
            s = float(sims[i])
            if s < self._threshold:
                break
            out.append((s, self._rows[i]))
            if len(out) >= self._top_n:
                break
        return out
