# -*- coding: utf-8 -*-
"""BM25 词法检索器（零外部依赖，jieba 分词 + 标准 BM25）

用途：外部基线对比（reports/external_baseline_design.md 的 B1）。
**默认不接线到主流程**——与 OPT_ID_EXPAND 同样的"默认关闭"纪律，
避免影响任何既有臂的可复现性。

接口刻意与 SlidingWindowDenseRetriever 对齐：
    retrieve(query, top_k) -> List[Tuple[float, dict]]  # dict 含 'text'
"""
import math
from collections import Counter

import jieba


def default_tokenizer(s):
    """jieba 分词 + 去空白/纯标点；英文与数字保留。"""
    return [w.strip() for w in jieba.lcut(str(s))
            if w.strip() and not w.strip().isspace()]


class BM25Retriever:
    """标准 BM25（k1=1.2, b=0.75），对全量文档逐一打分后取 top_k。

    规模说明：本语料 363 个滑窗 chunk，逐文档打分完全可接受
    （单题 O(N·|q|)）。若日后用于更大语料需换倒排索引。
    """

    def __init__(self, texts, k1=1.2, b=0.75, tokenizer=None):
        self.texts = list(texts)
        self.k1 = k1
        self.b = b
        self.tok = tokenizer or default_tokenizer
        self.docs = [self.tok(t) for t in self.texts]
        self.N = len(self.docs)
        self.avgdl = (sum(len(d) for d in self.docs) / self.N) if self.N else 0.0
        self.tf = [Counter(d) for d in self.docs]
        df = Counter()
        for d in self.docs:
            for w in set(d):
                df[w] += 1
        # BM25 的 idf（带 +0.5 平滑，恒正）
        self.idf = {w: math.log(1 + (self.N - n + 0.5) / (n + 0.5))
                    for w, n in df.items()}
        self.n_terms = len(self.idf)

    def _score(self, qterms, i):
        f_d = self.tf[i]
        dl = len(self.docs[i])
        if dl == 0 or self.avgdl == 0:
            return 0.0
        denom_norm = self.k1 * (1 - self.b + self.b * dl / self.avgdl)
        s = 0.0
        for w in qterms:
            f = f_d.get(w, 0)
            if f == 0:
                continue
            idf = self.idf.get(w)
            if idf is None:
                continue
            s += idf * f * (self.k1 + 1) / (f + denom_norm)
        return s

    def retrieve(self, query, top_k=12):
        qterms = self.tok(query)
        scored = []
        for i in range(self.N):
            s = self._score(qterms, i)
            if s > 0:
                scored.append((s, {'text': self.texts[i], 'idx': i}))
        scored.sort(key=lambda x: -x[0])
        return scored[:top_k]

    def retrieve_scored(self, query, top_k=12):
        """返回 [(score, idx, text)]，供融合使用。"""
        qterms = self.tok(query)
        out = []
        for i in range(self.N):
            s = self._score(qterms, i)
            if s > 0:
                out.append((s, i, self.texts[i]))
        out.sort(key=lambda x: -x[0])
        return out[:top_k]

    def info(self):
        return {'type': 'BM25', 'k1': self.k1, 'b': self.b, 'N': self.N,
                'avgdl': round(self.avgdl, 1), 'n_terms': self.n_terms}


def _text_of(item):
    """兼容 (score, dict) 与 (score, idx, text) 与纯 str 三种返回形态。"""
    if isinstance(item, tuple):
        last = item[-1]
        if isinstance(last, dict):
            return last.get('text', '')
        return str(last)
    if isinstance(item, dict):
        return item.get('text', '')
    return str(item)


def rrf_fuse(hit_lists, k=60, top_k=12, weights=None):
    """Reciprocal Rank Fusion：score = Σ w_i / (k + rank_i)。

    hit_lists: List[List[...]]，每个元素为某一路召回的**已排序**结果。
    """
    if weights is None:
        weights = [1.0] * len(hit_lists)
    agg = {}
    for w, hits in zip(weights, hit_lists):
        for r, item in enumerate(hits):
            t = _text_of(item)
            if not t:
                continue
            agg[t] = agg.get(t, 0.0) + w / (k + r + 1)
    out = sorted(agg.items(), key=lambda x: -x[1])
    return [(s, {'text': t, 'rrf': s}) for t, s in out[:top_k]]
