"""extractor 模块单元测试（纯逻辑，不触发真实 LLM/Ollama 调用）。"""
from backend.extractor import (
    _clean_token,
    _canonicalize_relation,
    _is_forbidden_relation,
    _split_chain,
    _try_parse_json,
    _dedup_semantic,
    industrial_parse_fn,
    normalize_entity,
)


def test_clean_token_strips_brackets_and_space():
    """_clean_token 去除边界括号/引号/空白。"""
    assert _clean_token(" 叶轮 ") == "叶轮"
    assert _clean_token("（叶轮）") == "叶轮"
    assert _clean_token('"叶轮"') == "叶轮"


def test_canonicalize_relation_maps_alias():
    """关系同义词应映射到标准关系（如 引起→可能原因）。"""
    assert _canonicalize_relation("引起") == "可能原因"
    assert _canonicalize_relation("导致") == "可能原因"
    assert _canonicalize_relation("包含") == "包含"  # 已是标准关系


def test_is_forbidden_relation():
    """FORBIDDEN 关系（MTBF）应被识别。"""
    assert _is_forbidden_relation("MTBF") is True
    assert _is_forbidden_relation("可能原因") is False


def test_split_chain_arrow_and_comma():
    """链条切分支持 A→B→C 与 A,B,C。"""
    assert _split_chain("A→B→C") == ["A", "B", "C"]
    assert _split_chain("A,B,C") == ["A", "B", "C"]
    assert _split_chain("A->B") == ["A", "B"]


def test_try_parse_json_array():
    """标准 JSON 数组可解析（含中英文字段别名）。"""
    out = _try_parse_json('[{"s":"叶轮","r":"包含","o":"泵"},{"s":"过热","r":"可能原因","o":"润滑不足"}]')
    assert ("叶轮", "包含", "泵", "none") in out
    assert ("过热", "可能原因", "润滑不足", "none") in out


def test_try_parse_json_meta_extraction():
    """P0-1: JSON 含 meta 字段时正确提取并校验枚举。"""
    out = _try_parse_json(
        '[{"s":"报警X","r":"解决方法","o":"反向移动","meta":"sequence"},'
        '{"s":"报警X","r":"解决方法","o":"若不能移动按急停","meta":"conditional"},'
        '{"s":"报警X","r":"解决方法","o":"修正设定值","meta":"非法值"}]'
    )
    assert ("报警X", "解决方法", "反向移动", "sequence") in out
    assert ("报警X", "解决方法", "若不能移动按急停", "conditional") in out
    assert ("报警X", "解决方法", "修正设定值", "none") in out, "非法 meta 归一到 none"


def test_industrial_parse_include_meta_flag():
    """P0-1: include_meta=True 返回 4 元组，默认 False 兼容 3 元组。"""
    out3 = industrial_parse_fn('[{"s":"报警090","r":"解决方法","o":"反向移动","meta":"sequence"}]')
    assert out3 == [("报警090", "解决方法", "反向移动")], "默认应剥离 meta 返回 3 元组"
    out4 = industrial_parse_fn(
        '[{"s":"报警090","r":"解决方法","o":"反向移动","meta":"sequence"}]', include_meta=True)
    assert out4 == [("报警090", "解决方法", "反向移动", "sequence")], "include_meta=True 应带 meta"


def test_try_parse_json_codeblock():
    """markdown 代码块包裹的 JSON 可解析。"""
    out = _try_parse_json('```json\n[{"s":"A","r":"包含","o":"B"}]\n```')
    assert ("A", "包含", "B", "none") in out


def test_try_parse_json_invalid_returns_empty():
    """非 JSON 或空输入返回空列表。"""
    assert _try_parse_json("不是JSON") == []
    assert _try_parse_json("") == []
    assert _try_parse_json("[{invalid json}]") == []
    # list-of-lists 形式不被 JSON 解析支持（仅 dict 数组），返回空
    assert _try_parse_json('[["A", "包含", "B"]]') == []


def test_industrial_parse_bracket_format():
    """括号格式 (A, 关系, B) 可解析且只保留 ALLOWED 关系。"""
    out = industrial_parse_fn("(叶轮流道, 包含, 进口管路), (叶轮流道, 引起, 振动异常)")
    rels = {r for _, r, _ in out}
    assert "包含" in rels, "ALLOWED 关系应保留"
    assert ("叶轮流道", "可能原因", "振动异常") in out, "引起→可能原因 映射"


def test_industrial_parse_arrow_format():
    """箭头格式 A-关系->B 可解析。"""
    out = industrial_parse_fn("离心泵叶轮--包含-->进口管路")
    assert ("离心泵叶轮", "包含", "进口管路") in out


def test_industrial_parse_dedup_and_noise_filter():
    """重复三元组去重，噪音实体过滤。"""
    out = industrial_parse_fn(
        '[{"s":"叶轮流道","r":"包含","o":"进口管路"},'
        '{"s":"叶轮流道","r":"包含","o":"进口管路"},'
        '{"s":"设备","r":"包含","o":"系统"}]'
    )
    assert len(out) == 1, "重复去重 + 噪音过滤后仅剩 1 条"
    assert out[0] == ("叶轮流道", "包含", "进口管路")


def test_dedup_semantic_merges_variants():
    """语义去重：归一化后相同的三元组合并，保留最长实体名。"""
    triples = [
        {"s": "轴承（A）", "r": "包含", "o": "滚动体", "source": "m.md", "chunk_id": 0},
        {"s": "轴承(A)", "r": "包含", "o": "滚动体", "source": "m.md", "chunk_id": 1},
    ]
    merged = _dedup_semantic(triples)
    assert len(merged) == 1, "全角/半角括号变体应合并"
    assert merged[0]["s"] == "轴承（A）", "保留最长/最完整原文"
    assert merged[0]["chunk_id"] == 0, "取最小 chunk_id"


def test_dedup_semantic_empty():
    """空输入原样返回。"""
    assert _dedup_semantic([]) == []


def test_normalize_entity_fullwidth():
    """实体归一化：全角括号/逗号、℃ 统一。"""
    assert normalize_entity("轴承（A），密封") == "轴承(A),密封"
    assert normalize_entity("温度>80℃") == "温度>80°C"
