import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from backend.build_pipeline import (
    extract_path_for,
    communities_cache_for,
    chunk_index_cache_for,
    parsed_md_path_for,
)


def test_extract_path_for():
    assert extract_path_for("m1").endswith("llm_extract_m1.json")


def test_communities_cache_for():
    assert communities_cache_for("m1").endswith("communities_m1.json")


def test_chunk_index_cache_for():
    assert chunk_index_cache_for("m1").endswith("chunk_index_m1.json")


def test_parsed_md_path_for():
    assert parsed_md_path_for("m1").endswith("m1.md")


def test_paths_are_distinct_per_manual():
    a = extract_path_for("a")
    b = extract_path_for("b")
    assert a != b
