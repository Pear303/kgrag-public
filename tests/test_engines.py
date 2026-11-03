"""engines 模块单元测试（仅测纯函数，不依赖 Neo4j）。"""
import numpy as np
import pytest

from backend.engines import _match_entities_smart


def _fake_embed(text):
    """确定性伪 embedding：按字符求和得到 32 维向量（保证同名文本同向量）。"""
    v = np.zeros(32, dtype=np.float32)
    for ch in str(text):
        v[ord(ch) % 32] += 1.0
    v = v / (np.linalg.norm(v) + 1e-9)
    return v.tolist()


def test_literal_match_hits_entity_in_question():
    """H4：实体名直接出现在问题中时，字面匹配应命中。"""
    names = ["气体未排净", "进口管路阻力过大", "叶轮磨损严重"]
    embeds = [_fake_embed(n) for n in names]
    matched = _match_entities_smart(
        "根据手册，泵启动前气体未排净会导致什么？",
        names, embeds, embed=_fake_embed, top_k=10, threshold=0.35,
    )
    assert "气体未排净" in matched


def test_semantic_fallback_with_low_threshold():
    """H4：字面未命中但语义相似（低阈值）时仍能召回，且不因 0.5 阈值漏掉。"""
    names = ["轴承温度过高", "机械密封泄漏量超标", "泵流量不足"]
    embeds = [_fake_embed(n) for n in names]
    # 问题与"轴承温度过高"共享"温度"等字符，语义相似度应高于低阈值
    matched = _match_entities_smart(
        "轴承过热是什么原因造成的？", names, embeds,
        embed=_fake_embed, top_k=3, threshold=0.35,
    )
    assert matched, "低阈值下不应空召回"


def test_fallback_requires_threshold_when_semantic_empty():
    """9.10：语义全部低于阈值时，保底不再无下限硬塞 top-3——宁缺毋滥。

    旧行为（H4）保底无条件取 top-3，会让 out-of-context 题（如"燃气轮机叶片
    裂纹"，实体相似度 0.45~0.53）被塞入无关实体 → 模型答非所问反而得分，
    是 OOC 指标口径不对称的根源之一。修复后保底同样卡 threshold。
    """
    names = ["压缩机转子不平衡", "风机叶轮积灰", "阀门密封面磨损"]
    embeds = [_fake_embed(n) for n in names]
    # threshold=0.99 高于所有实体相似度 → 保底也不得硬塞 → 返回空
    matched = _match_entities_smart(
        "完全不相关的宏观问题", names, embeds,
        embed=_fake_embed, top_k=3, threshold=0.99,
    )
    assert len(matched) <= 3
    assert matched == [], "保底不应无下限硬塞 top-3（宁缺毋滥，交给拒答分支）"

    # 保底内确有达标实体时仍可召回（模拟"语义分支因异常未执行"的兜底场景）
    matched2 = _match_entities_smart(
        "压缩机", names, embeds,
        embed=_fake_embed, top_k=3, threshold=0.0,
    )
    assert len(matched2) > 0, "保底中相似度达阈值时仍应召回"


def test_no_duplicates_and_preserves_literal_first():
    """H4：合并结果去重，且字面命中排在语义命中之前。"""
    names = ["润滑油位不足或径向载荷过大", "润滑油位不足"]
    embeds = [_fake_embed(n) for n in names]
    matched = _match_entities_smart(
        "润滑油位不足或径向载荷过大如何处理", names, embeds,
        embed=_fake_embed, top_k=10, threshold=0.35,
    )
    assert len(matched) == len(set(matched)), "不应有重复"
    assert matched[0] == "润滑油位不足或径向载荷过大", "字面命中应优先"


def test_empty_names_returns_empty():
    """空实体列表应返回空列表而非报错。"""
    assert _match_entities_smart("问题", [], [], embed=_fake_embed) == []


def test_literal_match_capped_and_long_first():
    """KG-3：字面命中按长度降序且受 top_k 限制。

    短实体名（"泵"）是长实体名（"离心泵叶轮"）的子串，若不限制会召回全部
    含子串的实体导致噪声路径挤占 LIMIT；修复后长名（更具体）优先且总数受限。
    """
    names = ["泵", "离心泵", "离心泵叶轮", "叶轮", "轴承"]
    embeds = [_fake_embed(n) for n in names]
    matched = _match_entities_smart(
        "离心泵叶轮磨损如何处理", names, embeds,
        embed=_fake_embed, top_k=2, threshold=0.35,
    )
    assert "离心泵叶轮" in matched, "最长字面命中应优先"
    assert len(matched) <= 2, "字面命中应受 top_k 限制"


def test_community_fallback_requires_similarity_floor():
    """KG-7：社区保底召回仅在相似度达下限时触发，完全不相关时不硬塞社区。"""
    import numpy as np
    from backend.engines import CommunityQueryEngine

    eng = CommunityQueryEngine.__new__(CommunityQueryEngine)
    eng.communities = [{"id": "a"}, {"id": "b"}, {"id": "c"}]
    eng.top_k = 3
    eng.similarity_threshold = 0.9
    # 单位向量：与 query=[1,0] 的余弦分别约为 0.866 / 0.309 / -0.866
    eng._comm_matrix = np.array(
        [[0.8660254, 0.5],
         [0.30901699, 0.9510565],
         [-0.8660254, 0.5]],
        dtype=np.float32,
    )

    q_emb = lambda text: [1.0, 0.0]
    eng._get_query_embedding = q_emb
    matched = eng._match_communities("问题")
    assert [c["id"] for c in matched] == ["a"], "0.866 >= 0.25 保底下限，应召回最相似社区"

    # 全部低于保底下限 → 宁可不召回
    eng._comm_matrix = np.array(
        [[0.2, 0.9798],
         [-0.2, 0.9798],
         [-0.8660254, 0.5]],
        dtype=np.float32,
    )
    matched = eng._match_communities("问题")
    assert matched == [], "相似度全部低于保底下限时不应硬塞社区"


# ════════════════════════════════════════════════════════════════════
# 补充：AutoGraphQueryEngine / CommunityQueryEngine 查询路径（fake store）
# ════════════════════════════════════════════════════════════════════

def _fixed_embed(text):
    """返回与文本无关的固定向量，方便精确断言。"""
    return [1.0, 0.0, 0.0, 0.0]


class _FakeEmbedModel:
    def get_text_embedding(self, text):
        return _fixed_embed(text)


class _FakeResult:
    """可迭代结果：支持 `for r in rows` 与 `rows.data()`。"""
    def __init__(self, rows):
        self._rows = rows
    def data(self):
        return self._rows
    def __iter__(self):
        return iter(self._rows)


class _FakeSession:
    def __init__(self, store):
        self._store = store
    def __enter__(self):
        return self
    def __exit__(self, *a):
        return False
    def run(self, cypher, **params):
        for key, rows in self._store.rows_by_cypher.items():
            if key in cypher:
                return _FakeResult(rows)
        return _FakeResult([])


class _FakeDriver:
    """B2 后路径展开走 driver.session().run()（替代 structured_query）。"""
    def __init__(self, store):
        self._store = store
    def session(self):
        return _FakeSession(self._store)


class _FakeGraphStore:
    """内存 fake：记录 structured_query 调用并返回可配置行。"""

    def __init__(self, vector_mode="memory", rows_by_cypher=None):
        self.vector_mode = vector_mode
        self.calls = []
        self.rows_by_cypher = rows_by_cypher or {}
        self._driver = _FakeDriver(self)

    def structured_query(self, cypher, param_map=None):
        self.calls.append((cypher, param_map))
        for key, rows in self.rows_by_cypher.items():
            if key in cypher:
                return rows
        return []


def test_cosine_similarity_basics():
    """余弦相似度：同向=1、正交=0、零向量=0。"""
    from backend.engines import _cosine_similarity
    import numpy as np
    assert _cosine_similarity(np.array([1.0, 0.0]), np.array([1.0, 0.0])) == 1.0
    assert _cosine_similarity(np.array([1.0, 0.0]), np.array([0.0, 1.0])) == 0.0
    assert _cosine_similarity(np.array([0.0, 0.0]), np.array([1.0, 0.0])) == 0.0


def test_auto_native_match_filters_by_threshold_and_literal():
    """Auto 原生匹配：分数 < ENTITY_MATCH_THRESHOLD 过滤 + 字面命中补充。"""
    from backend.engines import AutoGraphQueryEngine

    store = _FakeGraphStore(
        vector_mode="native",
        rows_by_cypher={"vector.queryNodes": [
            {"id": "叶轮", "score": 0.9},     # 达标
            {"id": "无关实体", "score": 0.5},  # 低于 0.60 过滤
        ]},
    )
    eng = AutoGraphQueryEngine(store, _FakeEmbedModel())
    matched = eng._match_entities_native("叶轮磨损怎么办", entity_names=["叶轮", "离心泵"])
    assert "叶轮" in matched, "达标 + 字面命中的实体必须命中"
    assert "无关实体" not in matched, "低于阈值不得命中"
    assert "离心泵" not in matched, "未出现在问题中的实体不得靠字面补充"


def test_auto_native_fallback_respects_threshold():
    """Auto 原生保底：全部低于阈值时不得硬塞 top-3。"""
    from backend.engines import AutoGraphQueryEngine

    store = _FakeGraphStore(
        vector_mode="native",
        rows_by_cypher={"vector.queryNodes": [
            {"id": "无关1", "score": 0.55},
            {"id": "无关2", "score": 0.4},
            {"id": "无关3", "score": 0.3},
        ]},
    )
    eng = AutoGraphQueryEngine(store, _FakeEmbedModel())
    matched = eng._match_entities_native("完全无关问题", entity_names=[])
    assert matched == [], "保底不得把低于 0.60 的实体硬塞回来"


def test_auto_native_overlap_recalls_concepts():
    """A7：native 分支提供 entity_names 时叠加 4-gram overlap 召回（与 memory 对齐）。

    概念实体（"回参考点不能进行"）在向量索引中 sim 0.569 < 0.60 被拒，但共享
    "回参考点""不能进行"两个全中文 4-gram，必须被 overlap 路召回救回。
    """
    from backend.engines import AutoGraphQueryEngine

    store = _FakeGraphStore(
        vector_mode="native",
        rows_by_cypher={"vector.queryNodes": [
            {"id": "无关实体", "score": 0.5},  # 低于 0.60 被过滤
        ]},
    )
    eng = AutoGraphQueryEngine(store, _FakeEmbedModel())
    matched = eng._match_entities_native(
        "回参考点操作不能进行时", entity_names=["回参考点不能进行", "叶轮", "离心泵"])
    assert "回参考点不能进行" in matched, "overlap 概念实体必须被召回"
    assert "叶轮" not in matched, "无共享 4-gram 的实体不得靠 overlap 误入"
    assert "离心泵" not in matched, "无共享 4-gram 的实体不得靠 overlap 误入"


def test_auto_expand_paths_interleaves_per_entity():
    """Auto 路径展开：跨实体轮询交错 + 按实体展开。"""
    from backend.engines import AutoGraphQueryEngine

    store = _FakeGraphStore(
        vector_mode="memory",
        rows_by_cypher={"MATCH path": [
            {"node_names": ["A", "B"], "rel_types": ["包含"]},
            {"node_names": ["A", "C"], "rel_types": ["可能原因"]},
        ]},
    )
    eng = AutoGraphQueryEngine(store, _FakeEmbedModel())
    paths = eng._expand_paths(["A", "B"])
    assert len(paths) >= 2, "匹配实体的路径必须全部展开（轮询交错）"


def test_auto_query_memory_mode_uses_smart_match():
    """Auto query 在 memory 模式下走 _match_entities_memory（智能匹配）。"""
    from backend.engines import AutoGraphQueryEngine

    store = _FakeGraphStore(vector_mode="memory", rows_by_cypher={"MATCH path": [
        {"node_names": ["叶轮", "流量下降"], "rel_types": ["可能原因"]},
    ]})
    eng = AutoGraphQueryEngine(store, _FakeEmbedModel())
    # query() 会先做实体匹配再展开；内存模式下不依赖真实 Neo4j
    paths = eng.query("叶轮", entity_names=["叶轮"], entity_embeddings=[[1.0, 0.0, 0.0, 0.0]])
    assert isinstance(paths, list)


def test_community_init_normalizes_embedding_dims():
    """A4：社区矩阵维度取众数（替代旧 min/max 截断），少数异常维度补齐不崩溃。"""
    from backend.engines import CommunityQueryEngine

    communities = [
        {"id": "c1", "description_embedding": [0.1, 0.2, 0.3]},
        {"id": "c2", "description_embedding": [0.4, 0.5]},       # 少数异常维度
        {"id": "c3", "description_embedding": [0.6, 0.7, 0.8]},
    ]
    eng = CommunityQueryEngine(_FakeGraphStore(), _FakeEmbedModel(), communities)
    assert eng._comm_matrix.shape == (3, 3), "众数维度 3（2 条），少数 2 维补齐"


def test_community_init_backfills_missing_embeddings():
    """A4：缺失 description_embedding 的社区用 embed 模型回填（不再静默 0 向量）。"""
    from backend.engines import CommunityQueryEngine

    communities = [
        {"id": "c1", "description_embedding": [1.0, 0.0, 0.0, 0.0],
         "description": "社区一描述"},
        {"id": "c2", "description_embedding": None, "description": "社区二描述"},
    ]
    eng = CommunityQueryEngine(_FakeGraphStore(), _FakeEmbedModel(), communities)
    assert eng._comm_matrix.shape == (2, 4), "回填的 c2 维度应与 c1 对齐（众数 4）"
    assert all(c["description_embedding"] for c in communities), "缺失 embedding 必须回填"


def test_community_init_empty_communities():
    """空社区列表 → 空矩阵，query 返回空不崩溃。"""
    from backend.engines import CommunityQueryEngine

    eng = CommunityQueryEngine(_FakeGraphStore(), _FakeEmbedModel(), [])
    assert eng._comm_matrix.size == 0
    matched, contexts = eng.query("问题")
    assert matched == [] and contexts == []


def test_community_query_dedup_and_block():
    """H1/H3：跨社区全局去重 + 有摘要时组织成块。"""
    from backend.engines import CommunityQueryEngine

    def _embed_2d(text):
        return [1.0, 0.0]

    class _Embed2D:
        def get_text_embedding(self, text):
            return _embed_2d(text)

    communities = [
        {"id": "c1", "description_embedding": [1.0, 0.0],
         "summary": "摘要1", "report": "报告1",
         "raw_context": ["片段A", "片段A"]},  # 社区内重复
        {"id": "c2", "description_embedding": [0.999, 0.001],
         "summary": "摘要2", "report": "报告2",
         "raw_context": ["片段A", "片段B"]},  # 与 c1 共享片段A（跨社区重复）
    ]
    eng = CommunityQueryEngine(_FakeGraphStore(), _Embed2D(), communities)
    matched, contexts = eng.query("问题")
    assert matched, "两个社区相似度都很高应都命中"
    total_pieces = "".join(contexts)
    assert total_pieces.count("片段A") == 1, "跨社区全局去重：片段A 只能出现一次"
    assert total_pieces.count("片段B") == 1
    assert "摘要1" in total_pieces and "报告1" in total_pieces, "有摘要应组织成块"


def test_community_query_no_summary_flat():
    """无摘要/报告时退化为裸拼片段（仍全局去重）。"""
    from backend.engines import CommunityQueryEngine

    class _Embed2D:
        def get_text_embedding(self, text):
            return [1.0, 0.0]

    communities = [
        {"id": "c1", "description_embedding": [1.0, 0.0],
         "summary": "", "report": "", "raw_context": ["原始片段1", "原始片段2"]},
    ]
    eng = CommunityQueryEngine(_FakeGraphStore(), _Embed2D(), communities)
    matched, contexts = eng.query("问题")
    assert matched
    flat = "".join(contexts)
    assert "原始片段1" in flat and "原始片段2" in flat


def test_community_query_max_per_community_cap():
    """B5：无摘要时每社区最多贡献 max_per_community 条，槽位跨社区分布。"""
    from backend.engines import CommunityQueryEngine

    class _Embed2D:
        def get_text_embedding(self, text):
            return [1.0, 0.0]

    communities = [
        {"id": "c1", "description_embedding": [1.0, 0.0],
         "summary": "", "report": "",
         "raw_context": ["C1片段1", "C1片段2", "C1片段3", "C1片段4"]},
        {"id": "c2", "description_embedding": [0.999, 0.001],
         "summary": "", "report": "",
         "raw_context": ["C2片段1", "C2片段2"]},
    ]
    eng = CommunityQueryEngine(_FakeGraphStore(), _Embed2D(), communities)
    matched, contexts = eng.query("问题")
    flat = "".join(contexts)
    assert contexts.count("C1片段1") == 1 and "C1片段1" in flat
    # top-1 社区被限制在 3 条，C1片段4 不得进入
    assert "C1片段4" not in flat, \
        "每社区 raw chunk 上限 3，top-1 社区不得占满全部槽位"
    # 第二社区仍有机会进入 context（跨社区多样性）
    assert "C2片段1" in flat, "6 个社区槽位应覆盖多个社区而非 top-1 独占"


def test_community_query_ranks_fragments_by_relevance():
    """2026-08-07：社区原文片段按与问题代码/短语重叠度排序，案例不被顺序挤出。"""
    from backend.engines import CommunityQueryEngine

    class _Embed2D:
        def get_text_embedding(self, text):
            return [1.0, 0.0]

    communities = [
        {"id": "c1", "description_embedding": [1.0, 0.0],
         "summary": "摘要", "report": "报告",
         "raw_context": [
             "无关片段A" * 10,
             "刀库气动冲击导致后位信号瞬间断开 X018-5 相关片段" * 2,
             "无关片段B" * 10,
             "无关片段C" * 10,
         ]},
    ]
    eng = CommunityQueryEngine(_FakeGraphStore(), _Embed2D(), communities)
    matched, contexts = eng.query("刀库气动冲击 X018-5")
    flat = "".join(contexts)
    assert "后位信号" in flat, "相关片段应按重叠度优先进入 context"


def test_community_query_chunk_index_priority(tmp_path):
    """2026-08-07：文本单元命中社区应前置，案例原文进入 context。"""
    import json
    from backend.engines import CommunityQueryEngine

    class _Embed2D:
        def get_text_embedding(self, text):
            # 含"案例"的文本映射到 [1,0]，其余映射到 [0,1]
            return [1.0, 0.0] if "案例" in str(text) else [0.0, 1.0]

    communities = [
        {"id": "c1", "description_embedding": [1.0, 0.0],
         "summary": "摘要1", "report": "报告1", "raw_context": ["无关1"]},
        {"id": "c2", "description_embedding": [0.0, 1.0],
         "summary": "摘要2", "report": "报告2",
         "raw_context": ["案例原文 105.1 内容"]},
    ]
    cache = tmp_path / "chunk_index.json"
    cache.write_text(json.dumps({
        "rows": [
            {"text": "无关1", "comm_idxs": [0], "embedding": [0.0, 1.0]},
            {"text": "案例原文 105.1 内容", "comm_idxs": [1],
             "embedding": [1.0, 0.0]},
        ]
    }, ensure_ascii=False), encoding="utf-8")

    eng = CommunityQueryEngine(
        _FakeGraphStore(), _Embed2D(), communities,
        chunk_index_cache=str(cache))
    matched, contexts = eng.query("案例原文")
    assert matched and str(matched[0].get("id")) == "c2", \
        "文本单元命中社区应排在描述匹配社区前"
    flat = "".join(contexts)
    assert "案例原文 105.1 内容" in flat, "命中原文片段必须进 context"


def test_community_query_lexical_fallback_rescues_case_chunk(tmp_path):
    """2026-08-07：语义漏召回时，词面重叠兜底仍能把案例社区带进 context。"""
    import json
    from backend.engines import CommunityQueryEngine

    class _Embed2D:
        def get_text_embedding(self, text):
            return [1.0, 0.0] if "语义命中" in str(text) else [0.0, 1.0]

    words = "".join("词%d" % i for i in range(30))
    q = "语义命中" + words
    communities = [
        {"id": "c1", "description_embedding": [1.0, 0.0],
         "summary": "摘要1", "report": "报告1",
         "raw_context": ["语义命中 无关片段"]},
        {"id": "c2", "description_embedding": [0.0, 1.0],
         "summary": "摘要2", "report": "报告2",
         "raw_context": [words]},
    ]
    cache = tmp_path / "chunk_index.json"
    cache.write_text(json.dumps({
        "rows": [
            {"text": "语义命中 无关片段", "comm_idxs": [0],
             "embedding": [1.0, 0.0]},
            {"text": words, "comm_idxs": [1], "embedding": [0.0, 1.0]},
        ]
    }, ensure_ascii=False), encoding="utf-8")

    eng = CommunityQueryEngine(
        _FakeGraphStore(), _Embed2D(), communities,
        chunk_index_cache=str(cache))
    matched, contexts = eng.query(q)
    flat = "".join(contexts)
    assert words in flat, "词面兜底案例原文必须进 context"
    assert any(str(c.get("id")) == "c2" for c in matched), \
        "词面兜底社区应被召回"


def test_community_query_truncates_long_report():
    """超长社区报告按 MAX_COMMUNITY_REPORT_CHARS 截断。"""
    from backend.engines import CommunityQueryEngine, MAX_COMMUNITY_REPORT_CHARS

    class _Embed2D:
        def get_text_embedding(self, text):
            return [1.0, 0.0]

    communities = [
        {"id": "c1", "description_embedding": [1.0, 0.0],
         "summary": "摘要", "report": "X" * (MAX_COMMUNITY_REPORT_CHARS + 500),
         "raw_context": ["片段"]},
    ]
    eng = CommunityQueryEngine(_FakeGraphStore(), _Embed2D(), communities)
    _, contexts = eng.query("问题")
    joined = "".join(contexts)
    assert "报告已截断" in joined, "超长报告必须带截断标记"


def test_community_enrich_from_neo4j_merges_summaries():
    """_enrich_from_neo4j：从 Neo4j 摘要补充到内存社区（不覆盖已有值）。

    社区身份是 (source_manual, id)：不传手册就不许回填，传了就必须按那本手册取。
    """
    from backend.engines import CommunityQueryEngine

    class _StoreWithSummaries:
        def get_community_summaries(self, source_manual=None):
            assert source_manual == '泵手册.md', "回填必须按手册取，不许整库扫"
            return [{"id": "c1", "summary": "neo4j摘要", "report": "neo4j报告"}]

        def __getattr__(self, _):
            raise AssertionError("不应访问其他属性")

    communities = [{"id": "c1", "description_embedding": [1.0, 0.0],
                    "summary": "", "report": ""}]
    eng = CommunityQueryEngine(_StoreWithSummaries(), _FakeEmbedModel(), communities,
                               enrich_from_neo4j=True, source_manual='泵手册.md')
    assert communities[0]["summary"] == "neo4j摘要", "缺失摘要应从 Neo4j 补充"
    assert communities[0]["report"] == "neo4j报告"

    # 已有值时不得被覆盖
    communities2 = [{"id": "c1", "description_embedding": [1.0, 0.0],
                     "summary": "内存已有", "report": "内存已有报告"}]
    CommunityQueryEngine(_StoreWithSummaries(), _FakeEmbedModel(), communities2,
                         enrich_from_neo4j=True, source_manual='泵手册.md')
    assert communities2[0]["summary"] == "内存已有", "已有摘要不得被 Neo4j 覆盖"

    # 没给 source_manual：一次库访问都不许发生（旧行为会整库按 id 贴，跨手册抄串）
    class _BoomStore:
        def get_community_summaries(self, source_manual=None):
            raise AssertionError("无手册时不得向 Neo4j 取摘要")

    communities3 = [{"id": "c1", "description_embedding": [1.0, 0.0],
                     "summary": "", "report": ""}]
    CommunityQueryEngine(_BoomStore(), _FakeEmbedModel(), communities3,
                         enrich_from_neo4j=True)
    assert communities3[0]["summary"] == "", "无 source_manual 时不得回填"


def test_interleave_paths_priority_emits_concept_paths_first():
    """2026-08-05：n_priority 前 N 个实体的路径整段优先输出，其余轮询交错。"""
    from backend.engines import _interleave_paths

    per_entity = [
        [("C1", "r"), ("C2", "r")],          # 概念实体（前 1 个），2 条路径
        [("A1", "r")],                        # 其余实体
        [("B1", "r"), ("B2", "r"), ("B3", "r")],
    ]
    # n_priority=0：维持 9.13-② 轮询（按列表序轮询：C1,A1,B1,C2,B2,B3）
    plain = [p[0] for p in _interleave_paths(per_entity)]
    assert plain == ["C1", "A1", "B1", "C2", "B2", "B3"], plain

    # n_priority=1：概念实体 C 的两条路径整段在前，其余 A/B 轮询
    prio = [p[0] for p in _interleave_paths(per_entity, n_priority=1)]
    assert prio == ["C1", "C2", "A1", "B1", "B2", "B3"], prio

    # B3：概念实体 >2 条路径时，2/轮高频轮询（不再整段前移，避免概念实体
    # 垄断 context 前 N 槽）——C 的 C3/C4 不再整体压过 A 的所有路径。
    per_entity3 = [
        [("C1", "r"), ("C2", "r"), ("C3", "r"), ("C4", "r")],  # 概念实体 4 条
        [("A1", "r"), ("A2", "r")],
    ]
    prio3 = [p[0] for p in _interleave_paths(per_entity3, n_priority=1)]
    assert prio3 == ["C1", "C2", "A1", "C3", "C4", "A2"], prio3


def test_expand_paths_per_depth_produces_multi_hop():
    """B2：逐跳采样产出 depth-2/3 路径（集线实体不再被 depth-1 星形边占满）。"""
    from backend.engines import _expand_paths_per_depth

    store = _FakeGraphStore(rows_by_cypher={
        "REL*1": [{"node_names": ["A", "B"], "rel_types": ["d1"]}],
        "REL*2": [{"node_names": ["A", "B", "C"], "rel_types": ["d2"]}],
        "REL*3": [{"node_names": ["A", "B", "C", "D"], "rel_types": ["d3"]}],
    })
    paths = _expand_paths_per_depth(store._driver, "Entity", "A", key_field="name",
                                    per_depth=1, max_hop=3)
    depths = sorted(len(p[0]) for p in paths)
    assert depths == [2, 3, 4], f"逐跳采样应覆盖 depth-1/2/3，got {depths}"
    # per_depth=1 → 每跳最多 1 条，共 3 条
    assert len(paths) == 3


def test_expand_paths_per_depth_preserves_direction():
    """P0-2：无向展开必须保留每条边的真实方向。"""
    from backend.engines import _expand_paths_per_depth

    store = _FakeGraphStore(rows_by_cypher={
        "REL*1": [{
            "node_names": ["A", "B"],
            "rel_edges": [["B", "A", "可能原因"]],
        }],
    })
    paths = _expand_paths_per_depth(store._driver, "Entity", "A",
                                    key_field="name", per_depth=1, max_hop=1)
    assert paths[0][1] == ["可能原因"]
    assert paths[0][2] == ["-"], "边方向为 B->A，沿 A->B 渲染应为反向"


def test_overlap_matched_identifies_concept_entities():
    """2026-08-05：_overlap_matched 识别与问题共享 ≥2 全中文 4-gram 的实体。"""
    from backend.engines import _overlap_matched

    q = "在FANUC 0系统中，当回参考点操作不能进行时，系统通常显示哪些报警号"
    matched = ["回参考点不能进行", "FANUC 0系统", "坐标轴需要手动回参考点操作"]
    overlap = _overlap_matched(q, matched)
    # 回参考点不能进行 与问题共享 回参考点 + 不能进行 两个 4-gram
    assert "回参考点不能进行" in overlap, "共享 回参考点+不能进行 应被识别"
    assert "FANUC 0系统" not in overlap, "集线实体（英文+数字子串）不得误入"
    # 坐标轴实体与问题共享 回参考点/参考点操/考点操作 ≥2 个，也应识别为概念实体
    assert "坐标轴需要手动回参考点操作" in overlap, "共享 ≥2 个 4-gram 应识别"
    # 只共享 1 个 4-gram 的实体（如"系统报警号"）不应命中
    matched2 = ["系统报警号", "FANUC 0系统"]
    assert _overlap_matched(q, matched2) == []

    # 空 matched / 纯英文问题 → 空
    assert _overlap_matched("FANUC 0 system", matched) == []
    assert _overlap_matched(q, []) == []


def test_query_4grams_filters_template_words():
    """2026-08-05：overlap 判定应剔除问句模板通用 4-gram。"""
    from backend.engines import _query_4grams

    q = "出了什么故障现象？可能是什么原因？应该怎么解决？FANUC 0MC系统出现ALM100报警"
    q4 = _query_4grams(q)
    for template in ("故障现象", "可能原因", "应该怎么", "解决排查"):
        assert template not in q4, f"模板词 {template} 不应参与 overlap"
    assert "了什么故" in q4, "非模板中文 4-gram 应保留"


def test_rank_paths_token_hit_first():
    """2026-08-05：含问题英数 token 的答案路径必须排在最前。"""
    from backend.engines import _rank_paths

    paths = [
        (["FANUC 0MC系统", "立式加工中心"], ["配套使用"]),
        (["ALM100报警", "DNC通信错误"], ["表现为"]),
        (["FANUC 0MC", "攻螺纹循环"], ["包含"]),
    ]
    ranked = _rank_paths(paths, "FANUC 0MC系统出现ALM100报警的含义是什么")
    assert ranked[0][0][0] == "ALM100报警", "含 ALM100 token 的路径应优先"


def test_rank_paths_short_semantic_over_long_weak_chain():
    """2026-08-05：无 token 命中时，短而语义强的路径优先于多跳弱关系链。"""
    from backend.engines import _rank_paths

    paths = [
        (["设备一", "设备二", "设备三", "设备四", "设备五"], ["包含"] * 4),
        (["故障A", "原因B"], ["可能原因"]),
        (["故障A", "现象C"], ["表现为"]),
    ]
    ranked = _rank_paths(paths, "故障发生的原因是什么")
    assert ranked[0][0][1] == "原因B", "可能原因短路径应优先"
    assert ranked[2][1][0] == "包含", "多跳弱关系链应排最后"


def test_code_token_priority_ignores_ubiquitous_model_tokens():
    """2026-09-09（gsk_037 根因）：型号弱 token 不得压制精确概念实体。

    GSK983 图谱几乎所有现象实体都以型号 "GSK983" 开头。问句含 "GSK983"
    时，_code_token_priority_entities 把 20 个含型号的现象实体全部前置，
    把不带型号前缀的精确实体 "存储行程限位报警(软件限位报警)" 压到第
    MAX_EXPAND_ENTITIES 位之外——其 212/213 处理方法路径永不进 context。

    修复：命中面过广的弱 token（占 matched ≥50%）无鉴别力，不参与重排。
    """
    from backend.engines import _code_token_priority_entities

    matched = [
        "存储行程限位报警(软件限位报警)",   # 精确概念实体，无 GSK983 前缀
        "GSK983故障现象:出现X/Y/Z正负限位报警",
        "GSK983故障现象:回零时410和411报警",
        "GSK983故障现象:系统不能自动执行程序不报警",
        "GSK983报警404",
        "GSK983故障现象:重复定位精度差(机械)",
        "GSK983报警000",
        "GSK983故障现象:回零有减速不能找到参考点",
        "GSK983报警221",
        "GSK983故障现象:DNC加工出现报警",
    ]
    q = "GSK983存储行程限位报警（软件限位报警，如212/213）发生后应如何复位处理？"
    out = _code_token_priority_entities(q, matched)
    assert out[0] == "存储行程限位报警(软件限位报警)", \
        "无鉴别力的型号 token(983)不应把精确概念实体压到末位: %s" % out


def test_code_token_priority_still_promotes_selective_code_entities():
    """2026-09-09：有鉴别力的代码 token（Q12/405）仍应前置命中实体。"""
    from backend.engines import _code_token_priority_entities

    matched = ["GSK983报警404", "GSK983报警405", "GSK983故障现象:主轴不转"]
    out = _code_token_priority_entities("报警405怎么处理", matched)
    assert out[0] == "GSK983报警405", "选择性代码 token(405) 应仍能前置: %s" % out


def test_community_fragment_window_keeps_late_evidence():
    """2026-08-07：长片段中答案位于 800 字之后时，窗口截取仍保留证据。

    实测 auto_032 的“端子36 +10V”在 995 字符处、auto_013 的 D12/A169
    在 800 字后被切掉；新逻辑围绕命中 token 扩窗。
    2026-08-11：上限 800→12000（配合 _truncate_contexts 总预算约束）。
    """
    from backend.engines import _fragment_window, _norm_fragment_token

    assert _norm_fragment_token("端子 36") == "端子36"
    assert _norm_fragment_token("X018－5") == "X018-5"
    text = "x" * 900 + "端子 36 驱动器内部 +10V 检测端 A2 +10V" + "y" * 300
    snippet = _fragment_window(text, {"36", "端子36", "SIMODRIVE"})
    assert len(snippet) <= 12000
    assert "+10V" in snippet, "命中 token 的窗口截取必须保留证据"


def test_community_query_fragment_window_used_for_ranked_raw():
    """2026-08-07：社区 query 用窗口截取替代固定前 800 字。"""
    from backend.engines import CommunityQueryEngine

    class _Embed2D:
        def get_text_embedding(self, text):
            return [1.0, 0.0]

    communities = [
        {"id": "c1", "description_embedding": [1.0, 0.0],
         "summary": "摘要", "report": "报告",
         "raw_context": [
             "无关前缀" * 500 + "端子 36 驱动器内部 +10V 检测端",
             "无关片段A" * 30,
         ]},
    ]
    eng = CommunityQueryEngine(_FakeGraphStore(), _Embed2D(), communities)
    _, contexts = eng.query("端子36检测哪一路内部电压")
    flat = "".join(contexts)
    assert "+10V" in flat, "排名片段命中 token 时应保留 token 附近证据"


# ════════════════════════════════════════════════════════════════════
# 报警号别名召回（2026-09-06，修复①③：GSK983 手册命名适配）
# ════════════════════════════════════════════════════════════════════

def test_alarm_query_number_extracts_digits_adjacent_to_alarm():
    """修复①：问句点名"报警+数字"时能提取归一报警号（去前导零）。"""
    from backend.engines import _alarm_query_number
    assert _alarm_query_number("报警405怎么处理") == "405"
    assert _alarm_query_number("GSK983报警090怎么排除") == "90"
    assert _alarm_query_number("GSK983报警090排除") == "90"
    assert _alarm_query_number("轴承过热怎么修") is None
    assert _alarm_query_number("主轴停止") is None


def test_build_alarm_index_groups_int_normalized_numbers():
    """修复①：报警号实体按 int 归一建索引（跨前导零变体同 key）。"""
    from backend.engines import _build_alarm_index
    names = ["GSK983报警405", "GSK983报警090", "GSK983报警90", "GSK983报警212"]
    idx = _build_alarm_index(names)
    assert "GSK983报警405" in idx["405"]
    assert set(idx["90"]) == {"GSK983报警090", "GSK983报警90"}, \
        "前导零变体应归一到同一 key"
    assert "212" in idx


def test_alarm_alias_recall_matches_when_literal_fails():
    """修复①：问"报警405"（无 GSK983 前缀）时别名命中完整实体。"""
    from backend.engines import _match_entities_smart, _build_alarm_index
    names = ["GSK983报警405", "GSK983报警404", "GSK983报警422",
             "GSK983故障现象:系统#401报警", "出现#085号报警"]
    embeds = [_fake_embed(n) for n in names]
    idx = _build_alarm_index(names)
    matched = _match_entities_smart(
        "报警405怎么处理", names, embeds, embed=_fake_embed,
        top_k=10, threshold=0.60, alarm_index=idx)
    assert "GSK983报警405" in matched, "别名召回应把 405 实体带进 matched"
    assert matched[0] == "GSK983报警405", "数字别名命中应排在最前（长名）"


def test_alarm_alias_lazy_index_when_none():
    """修复①：未传 alarm_index 时惰性构建（500cases 等无"报警N"命名不受影响）。"""
    from backend.engines import _match_entities_smart
    names = ["轴承温度过高", "机械密封泄漏量超标", "泵流量不足"]
    embeds = [_fake_embed(n) for n in names]
    matched = _match_entities_smart(
        "轴承过热是什么原因造成的？", names, embeds, embed=_fake_embed,
        top_k=3, threshold=0.35)
    assert matched, "低阈值下不应空召回（行为与修复前一致）"


def test_dedup_alarm_prefix_fragments_drops_short_variants():
    """修复③：字面命中集内 1/2 位碎片（同号/前缀）被长命名取代。"""
    from backend.engines import _dedup_alarm_prefix_fragments
    # ① 同号跨前导零变体：001 权威，1 剔除
    out = _dedup_alarm_prefix_fragments(
        ["GSK983报警001", "GSK983报警1", "GSK983报警010", "GSK983报警10"])
    assert out == ["GSK983报警001", "GSK983报警010"], out
    # ② 真前缀：405 权威，4 剔除；212 权威，2 剔除
    out = _dedup_alarm_prefix_fragments(
        ["GSK983报警405", "GSK983报警4", "GSK983报警212", "GSK983报警2"])
    assert out == ["GSK983报警405", "GSK983报警212"], out
    # ③ 无碎片集合保持原样
    out = _dedup_alarm_prefix_fragments(
        ["GSK983报警405", "GSK983报警404", "GSK983报警090"])
    assert out == ["GSK983报警405", "GSK983报警404", "GSK983报警090"], out
    # ④ 前导零不同号（001/010/100）不得互相误伤
    out = _dedup_alarm_prefix_fragments(
        ["GSK983报警001", "GSK983报警010", "GSK983报警100"])
    assert "GSK983报警001" in out and "GSK983报警010" in out and "GSK983报警100" in out


def test_paren_alias_segments_extracts_chinese_cores():
    """修复①-补充：括号现象实体提取纯中文别名段（≥4 字）。"""
    from backend.engines import _paren_alias_segments
    assert "超程报警" in _paren_alias_segments("OT报警(超程报警)")
    assert "微小偏差" in _paren_alias_segments("参考点停止位置与参考点不一致(微小偏差)")
    # 括号内太短不索引（(机械)/(软件) 等易误召）
    assert _paren_alias_segments("GSK983故障现象:上电轴暴走窜动(软件)") == []
    # 无括号实体不产生别名
    assert _paren_alias_segments("GSK983报警405") == []


def test_paren_alias_recall_matches_short_question():
    """修复①-补充：问"超程报警如何复位"能命中 "OT报警(超程报警)"。"""
    from backend.engines import _build_paren_alias_index, _paren_alias_recall
    names = ["OT报警(超程报警)", "GSK983故障现象:出现X/Y/Z正负限位报警", "GSK983报警210"]
    idx = _build_paren_alias_index(names)
    hits = _paren_alias_recall("GSK983机床超程报警如何复位", idx, exclude=set(), top_k=3)
    assert "OT报警(超程报警)" in hits, "括号别名应让现象实体被 '超程报警' 召回"


def test_paren_alias_does_not_fire_for_short_terms():
    """修复①-补充：泛短分类词（机械/软件）不单独产生别名入口，但纯中文长主体可以。"""
    from backend.engines import _paren_alias_segments, _build_paren_alias_index
    # 括号内 (机械)/(软件) 不足 4 字 → 不作为独立别名段
    assert _paren_alias_segments("重复定位精度差(机械)") == ["重复定位精度差"], \
        "短括号词不应是别名段；纯中文长主体（≥6字）应成为别名"
    names = ["重复定位精度差(机械)", "GSK983故障现象:重复定位精度差(软件)"]
    idx = _build_paren_alias_index(names)
    assert "机械" not in idx and "软件" not in idx, "泛短分类词不应是索引 key"
    assert "重复定位精度差" in idx, "纯中文长主体应可被单独问及"


def test_prioritize_diag_paths_puts_confirmation_first():
    """能力边界1（2026-09-07）：确认/检查类问法把确认方法路径前置。

    "怎么确认机床锁住(MLK)信号"此前确认方法路径排在轮询第 ~9 位，
    被 12 槽 context 挤出 LLM 注意力。确认动词问法应把首跳为
    确认方法/检查处理方法的路径稳定提到最前。
    """
    from backend.engines import _prioritize_diag_paths
    confirm = (["机床锁住(MLK)", "确认DGN#100的第7位为0"], ["确认方法"], ["+"])
    cause = (["机床锁住(MLK)", "不能进行JOG操作或JOG不正常"], ["故障原因"], ["-"])
    other = (["GSK983报警404", "伺服驱动器故障"], ["故障原因"], ["+"])
    out = _prioritize_diag_paths([cause, other, confirm],
                                 "GSK983怎么确认机床锁住(MLK)信号？")
    assert out[0] is confirm, "确认动词问法应把确认方法路径放到最前"
    assert cause in out and other in out, "其余路径应保持在场（仅重排不丢弃）"
    # 与问法字面相关的诊断路径应排在最前（其它现象的检查方法让位）
    other_check = (["GSK983故障现象:倍率开关程序锁开关无效", "开关故障"], ["检查处理方法"], ["+"])
    out3 = _prioritize_diag_paths([other_check, confirm],
                                  "GSK983怎么确认机床锁住(MLK)信号？")
    assert out3[0] is confirm, "问法所指实体的确认方法路径应最先"
    # 普通"原因/处理"问法不触发重排（保持原序，防止 500cases 回归）
    out2 = _prioritize_diag_paths([cause, confirm],
                                  "GSK983报警404是什么原因？")
    assert out2 == [cause, confirm], "非确认类问法不得重排"


def test_diag_re_matches_confirmation_phrasings():
    """能力边界1：确认/检查/怎么看 等问法措辞应触发诊断前置。"""
    from backend.engines import _DIAGNOSTIC_QUERY_RE
    for q in ["怎么确认机床锁住信号", "如何检查DGN#100", "怎么判断编码器故障",
              "如何检测位置偏差", "怎么核实限位开关", "查看DGN#820确认位置"]:
        assert _DIAGNOSTIC_QUERY_RE.search(q), q
    for q in ["GSK983报警212是什么原因", "报警700怎么处理", "超程报警如何复位"]:
        assert not _DIAGNOSTIC_QUERY_RE.search(q), q


# ════════════════════════════════════════════════════════════════════
# P0-3：推理链聚合（2026-09-09，多跳语义重构）
# ════════════════════════════════════════════════════════════════════

def test_build_reasoning_chains_groups_parallel_causes_and_methods():
    """P0-3: 同主体多条并列原因 + 方法边聚合为推理链行。"""
    from backend.engines import _build_reasoning_chains
    paths = [
        (["GSK983报警090", "返回参考点的起点离参考点太近"], ["故障原因"], ["+"]),
        (["GSK983报警090", "主机故障"], ["故障原因"], ["+"]),
        (["GSK983报警090", "更换脉冲编码器"], ["处理方法"], ["+"]),
        (["GSK983报警090", "增加进给速度"], ["处理方法"], ["+"]),
        (["GSK983报警404", "伺服驱动器故障"], ["故障原因"], ["+"]),
    ]
    out = _build_reasoning_chains(paths)
    # 链行应置于最前
    chain = [p for p in out if p[1] and str(p[1][0]) == "推理链"]
    assert chain, "应生成推理链行"
    chain_text = chain[0][0][0]
    assert "GSK983报警090" in chain_text
    assert "故障原因(2)" in chain_text, "并列原因应聚合计数: " + chain_text
    assert "起点离参考点太近" in chain_text and "主机故障" in chain_text
    assert "处理方法(2)" in chain_text
    # 原始路径全部保留在后
    assert len(out) == len(paths) + len(chain), "原始路径不得丢失"
    # 单条孤立的诊断边（404 只有 1 原因）不应成链
    for c in chain:
        assert "GSK983报警404" not in c[0][0], "单边主体不应成链"


def test_build_reasoning_chains_orders_conditional_last():
    """P0-3: 含"若/不能移动"的条件步骤排到无条件步骤后。"""
    from backend.engines import _build_reasoning_chains
    paths = [
        (["存储行程限位报警", "若不能移动按下急停按钮或执行G23指令"], ["处理方法"], ["+"]),
        (["存储行程限位报警", "在相反方向上移动机床"], ["处理方法"], ["+"]),
    ]
    out = _build_reasoning_chains(paths)
    chain = [p for p in out if p[1] and str(p[1][0]) == "推理链"]
    assert chain
    text = chain[0][0][0]
    # 无条件步骤"反向移动"应在条件步骤"若不能移动"之前
    assert text.index("在相反方向上移动机床") < text.index("若不能移动"), text


def test_build_reasoning_chains_skips_isolated_edge():
    """P0-3: 孤立单诊断边不聚合（避免无信息量链行）。"""
    from backend.engines import _build_reasoning_chains
    paths = [
        (["GSK983报警404", "伺服驱动器故障"], ["故障原因"], ["+"]),
        (["GSK983报警404", "检查信号线"], ["确认方法"], ["+"]),
    ]
    out = _build_reasoning_chains(paths)
    chain = [p for p in out if p[1] and str(p[1][0]) == "推理链"]
    assert not chain, "无成链价值（2 边不同关系）不应聚合"
    assert len(out) == len(paths), "路径应原样返回"
