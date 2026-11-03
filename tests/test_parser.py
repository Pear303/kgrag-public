"""parser 模块单元测试。

覆盖 parse_manual_chunks / parse_manual 的核心路径与边界情况。
不依赖 Neo4j 连接。
"""
from backend.parser import Chunk, normalize_entity, parse_manual, parse_manual_chunks


def test_parse_manual_chunks(test_manual_path):
    chunks = parse_manual_chunks(test_manual_path)
    assert len(chunks) > 0, "真实手册应产出至少一个 chunk"
    first = chunks[0]
    assert isinstance(first, Chunk)
    assert hasattr(first, "text")
    assert hasattr(first, "chunk_id")
    assert hasattr(first, "source")
    assert isinstance(first.text, str) and len(first.text) > 0
    assert isinstance(first.chunk_id, int)
    assert first.source == test_manual_path
    # 每个 chunk_id 唯一且连续（从 0 开始）
    ids = [c.chunk_id for c in chunks]
    assert ids == list(range(len(chunks)))


def test_parse_manual_triples(test_manual_path):
    triples = parse_manual(test_manual_path)
    assert isinstance(triples, list)
    assert len(triples) > 0, "真实手册应抽取至少一条三元组"
    for t in triples:
        assert isinstance(t, tuple)
        assert len(t) == 3
        assert all(isinstance(s, str) and len(s) > 0 for s in t), f"三元组每项应为非空字符串: {t}"


def test_empty_manual(tmp_path):
    """空文件：不应抛异常，返回空列表。"""
    empty = tmp_path / "empty.md"
    empty.write_text("", encoding="utf-8")
    triples = parse_manual(str(empty))
    chunks = parse_manual_chunks(str(empty))
    assert isinstance(triples, list)
    assert isinstance(chunks, list)
    assert triples == []
    assert chunks == []


def test_no_table_manual(tmp_path):
    """纯文本无表格：chunks 非空，triples 为空。"""
    md = tmp_path / "no_table.md"
    md.write_text("## 系统说明\n\n这是纯文本描述，没有任何表格。\n\n另一段文本。\n", encoding="utf-8")
    chunks = parse_manual_chunks(str(md))
    triples = parse_manual(str(md))
    assert len(chunks) > 0, "纯文本也应产出 chunk"
    assert isinstance(triples, list)
    assert triples == [], "无 Markdown 表格行时 triples 应为空"


def test_forbidden_relation_filtered(tmp_path):
    """FORBIDDEN_RELATIONS（如 MTBF）应被过滤，ALLOWED_RELATIONS（如 可能原因）应保留。"""
    from config import ALLOWED_RELATIONS, FORBIDDEN_RELATIONS

    md = tmp_path / "forbidden.md"
    md.write_text(
        "## 测试表格\n"
        "| 故障现象 | 故障关系 | 可能原因 |\n"
        "|---|---|---|\n"
        "| 温度过高 | 可能原因 | 润滑不足 |\n"
        "| 设备故障 | MTBF | 1000小时 |\n"
        "| 异响 | 环境温度 | 过高 |\n",
        encoding="utf-8"
    )

    assert "MTBF" in FORBIDDEN_RELATIONS
    assert "可能原因" in ALLOWED_RELATIONS

    triples = parse_manual(str(md))

    # 提取所有关系字段
    relations = {t[1] for t in triples}
    assert "可能原因" in relations, "ALLOWED 关系应被保留"
    assert "MTBF" not in relations, "FORBIDDEN 关系应被过滤"
    assert "环境温度" not in relations, "FORBIDDEN 关系应被过滤"
    assert len(triples) >= 1


def test_normalize_entity_full_vs_half_angle():
    """全角℃与半角°C应归一化为同一实体。"""
    assert normalize_entity("轴承座温度升高(>80℃)") == "轴承座温度升高(>80°C)"
    assert normalize_entity("轴承座温度升高(>80°C)") == "轴承座温度升高(>80°C)"
    # 全角括号也应转换
    assert normalize_entity("轴承座温度升高（>80℃）") == "轴承座温度升高(>80°C)"


def test_normalize_entity_strips_punctuation():
    """首尾标点和冗余空格应被去除。"""
    assert normalize_entity(",轴承,") == "轴承"
    assert normalize_entity("  轴承  ") == "轴承"
    assert normalize_entity("。轴承。") == "轴承"
    # 全角逗号转半角
    assert normalize_entity("轴承，密封") == "轴承,密封"


def test_normalize_entity_temperature_variants():
    """多种温度写法应统一。"""
    # ℃ 和 °C 等价
    assert normalize_entity("油池温度应低于82℃") == normalize_entity("油池温度应低于82°C")
    # 带空格的温度
    assert normalize_entity("温度 > 80 ℃") == "温度 > 80 °C"


def test_parse_manual_with_provenance_tracks_line_numbers(tmp_path):
    """parse_manual_with_provenance 应带 source_line 溯源，且过滤噪音关系。"""
    from backend.parser import parse_manual_with_provenance

    md = tmp_path / "prov.md"
    md.write_text(
        "| 故障现象 | 故障关系 | 可能原因 |\n"
        "|---|---|---|\n"
        "| 温度过高 | 可能原因 | 润滑不足 |\n"      # line 3
        "| 设备故障 | MTBF | 1000小时 |\n"          # line 4 过滤
        "| 轴承异响 | 解决方法 | 更换轴承 |\n",       # line 5
        encoding="utf-8"
    )
    results = parse_manual_with_provenance(str(md))
    rels = {r["r"] for r in results}
    assert "MTBF" not in rels, "FORBIDDEN 关系必须过滤"
    by_relation = {r["s"]: r for r in results if r["r"] == "可能原因"}
    assert by_relation["温度过高"]["source_line"] == 3, "行号溯源必须准确"
    assert by_relation["温度过高"]["source_manual"] == "prov.md"
    s = {r["s"] for r in results}
    assert "设备故障" not in s


def test_parse_manual_normalize_flag_disables_normalization(tmp_path):
    """do_normalize=False 时实体不做归一化（保留原样）。"""
    from backend.parser import parse_manual

    md = tmp_path / "nonorm.md"
    md.write_text(
        "| 故障现象 | 故障关系 | 可能原因 |\n"
        "|---|---|---|\n"
        "| 轴承（A） | 可能原因 | 润滑不足 |\n",
        encoding="utf-8"
    )
    triples = parse_manual(str(md), do_normalize=False)
    assert triples[0][0] == "轴承（A）", "不归一化应保留全角括号"


def test_seed_triples_delegates_to_graph_store(tmp_path):
    """seed_triples 应把三元组委托给 graph_store.seed_triples。"""
    from backend.parser import seed_triples

    class _FakeStore:
        def __init__(self):
            self.calls = []

        def seed_triples(self, triples, source_manual=None, source_line=None):
            self.calls.append((triples, source_manual, source_line))

    store = _FakeStore()
    triples = [("A", "包含", "B")]
    seed_triples(store, triples, source_manual="m.md", source_line=3)
    assert store.calls == [(triples, "m.md", 3)]


def test_sliding_window_short_text_single_chunk(tmp_path):
    """文本短于窗口时返回单个 chunk。"""
    from backend.parser import parse_manual_chunks_sliding_window

    md = tmp_path / "short.md"
    md.write_text("## 很短的手册\n\n只有一段。", encoding="utf-8")
    chunks = parse_manual_chunks_sliding_window(str(md), window_size=500, overlap_size=50)
    assert len(chunks) == 1
    assert chunks[0].chunk_id == 0


def test_sliding_window_empty_text_empty_chunks(tmp_path):
    """空文本 → 单块空 chunk（不抛异常）。"""
    from backend.parser import parse_manual_chunks_sliding_window

    md = tmp_path / "empty.md"
    md.write_text("", encoding="utf-8")
    chunks = parse_manual_chunks_sliding_window(str(md), window_size=500, overlap_size=50)
    assert len(chunks) == 1
    assert chunks[0].text == ""


def test_sliding_window_sentence_boundary_cut(tmp_path):
    """长文本应在句号处断开，且步长推进不产生死循环。"""
    from backend.parser import parse_manual_chunks_sliding_window

    sentences = "。".join(f"第{i}句描述离心泵故障现象" for i in range(20))
    md = tmp_path / "long.md"
    md.write_text("## 标题\n\n" + sentences, encoding="utf-8")
    chunks = parse_manual_chunks_sliding_window(str(md), window_size=120, overlap_size=20)
    assert len(chunks) >= 2, "长文本应产出多个窗口"
    ids = [c.chunk_id for c in chunks]
    assert ids == list(range(len(chunks))), "chunk_id 连续"
    assert all(c.text for c in chunks), "窗口文本不得为空"
