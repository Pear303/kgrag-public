"""graph_store.py 中 Cypher 注入防护函数单元测试。

测试 _validate_cypher_identifier、_validate_label、_validate_rel_type
的白名单与正则校验逻辑。不依赖 Neo4j 连接。
"""
import sys
from unittest.mock import MagicMock

if "neo4j" not in sys.modules:
    mock_neo4j = MagicMock()
    sys.modules["neo4j"] = mock_neo4j

import pytest

from backend.graph_store import (
    _validate_cypher_identifier,
    _validate_label,
    _validate_rel_type,
    _VALID_LABELS,
    _VALID_REL_TYPES,
)


class TestValidateCypherIdentifier:

    def test_valid_simple_identifier(self):
        result = _validate_cypher_identifier("Entity", "标签")
        assert result == "Entity"

    def test_valid_identifier_with_underscores(self):
        result = _validate_cypher_identifier("__Entity__", "标签")
        assert result == "__Entity__"

    def test_valid_identifier_with_digits(self):
        result = _validate_cypher_identifier("Entity2", "标签")
        assert result == "Entity2"

    def test_valid_numeric_suffix(self):
        result = _validate_cypher_identifier("label_123", "标签")
        assert result == "label_123"

    def test_rejects_semicolon_injection(self):
        with pytest.raises(ValueError, match="非法 标签"):
            _validate_cypher_identifier("Entity; DROP TABLE", "标签")

    def test_rejects_dash_in_identifier(self):
        with pytest.raises(ValueError, match="非法 标签"):
            _validate_cypher_identifier("my-label", "标签")

    def test_rejects_space_in_identifier(self):
        with pytest.raises(ValueError, match="非法 标签"):
            _validate_cypher_identifier("my label", "标签")

    def test_rejects_slash_in_identifier(self):
        with pytest.raises(ValueError, match="非法 关系类型"):
            _validate_cypher_identifier("rel/type", "关系类型")

    def test_rejects_backtick_injection(self):
        with pytest.raises(ValueError, match="非法 标签"):
            _validate_cypher_identifier("`Label`", "标签")

    def test_rejects_closing_bracket(self):
        with pytest.raises(ValueError, match="非法 标签"):
            _validate_cypher_identifier("Label)", "标签")

    def test_rejects_non_string_input(self):
        with pytest.raises(ValueError, match="非法 标签 类型"):
            _validate_cypher_identifier(123, "标签")

    def test_rejects_none_input(self):
        with pytest.raises(ValueError, match="非法 标签 类型"):
            _validate_cypher_identifier(None, "标签")

    def test_rejects_empty_string(self):
        with pytest.raises(ValueError, match="非法 标签"):
            _validate_cypher_identifier("", "标签")

    def test_rejects_leading_digit(self):
        with pytest.raises(ValueError, match="非法 标签"):
            _validate_cypher_identifier("123Entity", "标签")

    def test_kind_shows_in_error_message(self):
        with pytest.raises(ValueError, match="非法 关系类型"):
            _validate_cypher_identifier("bad-re;l", "关系类型")


class TestValidateLabel:

    def test_all_valid_labels_pass(self):
        for label in _VALID_LABELS:
            assert _validate_label(label) == label

    def test_rejects_unknown_label(self):
        with pytest.raises(ValueError, match="未知标签"):
            _validate_label("HackedLabel")

    def test_rejects_empty_label(self):
        with pytest.raises(ValueError, match="未知标签"):
            _validate_label("")

    def test_valid_label_also_passes_cypher_check(self):
        result = _validate_label("__Entity__")
        assert result == "__Entity__"

    def test_sql_injection_blocked_by_label_whitelist(self):
        with pytest.raises(ValueError, match="未知标签"):
            _validate_label("Entity; DROP DATABASE")

    def test_case_sensitive_validation(self):
        with pytest.raises(ValueError, match="未知标签"):
            _validate_label("entity")


class TestValidateRelType:

    def test_valid_rel_type_passes(self):
        result = _validate_rel_type("REL")
        assert result == "REL"

    def test_rejects_unknown_rel_type(self):
        with pytest.raises(ValueError, match="未知关系类型"):
            _validate_rel_type("HackedRel")

    def test_rejects_empty_rel_type(self):
        with pytest.raises(ValueError, match="未知关系类型"):
            _validate_rel_type("")

    def test_rejects_injection_in_rel_type(self):
        with pytest.raises(ValueError, match="未知关系类型"):
            _validate_rel_type("REL; DROP TABLE")

    def test_rel_type_also_validates_identifier_syntax(self):
        with pytest.raises(ValueError, match="未知关系类型"):
            _validate_rel_type("bad-rel!")


class TestValidSetsIntegrity:

    def test_valid_labels_contains_expected(self):
        assert "Entity" in _VALID_LABELS
        assert "__Entity__" in _VALID_LABELS
        assert "__Entity__v2" in _VALID_LABELS

    def test_valid_rel_types_contains_expected(self):
        assert "REL" in _VALID_REL_TYPES

    def test_all_valid_labels_pass_identifier_check(self):
        for label in _VALID_LABELS:
            _validate_cypher_identifier(label, "标签")

    def test_all_valid_rel_types_pass_identifier_check(self):
        for rel_type in _VALID_REL_TYPES:
            _validate_cypher_identifier(rel_type, "关系类型")
