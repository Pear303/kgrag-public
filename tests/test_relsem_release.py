# -*- coding: utf-8 -*-
"""口语措辞放行票（relsem_release_hook）反向测试。

钉六条：默认关零行为改变；命中放行；通用缩写不计数；对象不一致票不撤；
非相关度门形状不撤；阈值 env 可调。环境快照隔离（test_table_row.py 同法）。
"""
import os
import sys

import pytest

_ENV_SNAPSHOT = dict(os.environ)

from backend.evaluator import relsem_latin_hits, relsem_release_hook  # noqa: E402

_CTX_FANUC = ['[推理链] FANUC 0T 系统开机后 CRT 无显示 ; 可能原因: 主板不良']
_CTX_PLC = ['[推理链] CNC与PLC通信受干扰 ; 可能原因: 通信电缆插头内线相碰']
_REL = {'relevance': 0.0, 'object_mismatch': False}


@pytest.fixture(autouse=True)
def _restore_env():
    os.environ.pop('OOC_RELSEM', None)
    os.environ.pop('OOC_RELSEM_LATIN_MIN', None)
    yield
    os.environ.pop('OOC_RELSEM', None)
    os.environ.pop('OOC_RELSEM_LATIN_MIN', None)


def test_default_off_no_behavior_change():
    os.environ['OOC_RELSEM'] = '0'
    rr, rec = relsem_release_hook('师傅，FANUC 车床 CRT 没显示', _CTX_FANUC, dict(_REL))
    assert rr == _REL and rec is None


def test_release_on_latin_hit():
    os.environ['OOC_RELSEM'] = '1'
    rr, rec = relsem_release_hook('师傅，这台 FANUC 车床 CRT 没显示咋回事',
                                  _CTX_FANUC, dict(_REL))
    assert rr is None
    assert rec['relsem_release']['latin_hits'] >= 1


def test_generic_acronym_never_counts():
    os.environ['OOC_RELSEM'] = '1'
    rr, rec = relsem_release_hook('PLC 与上位机通信频繁断连咋排查', _CTX_PLC, dict(_REL))
    assert rr == _REL and rec is None


def test_object_mismatch_not_released():
    os.environ['OOC_RELSEM'] = '1'
    rel = {'relevance': 0.0, 'object_mismatch': True}
    rr, rec = relsem_release_hook('师傅，FANUC 机床 EMG STOP', _CTX_FANUC, dict(rel))
    assert rr == rel and rec is None


def test_non_relevance_shape_not_released():
    os.environ['OOC_RELSEM'] = '1'
    for rr_in in ({'relevance': 0.0, 'sent_cov': 0.5},
                  {'joint_gate': {'score': 0.9}},
                  {'relevance': 0.0, 'vendor_gate': True}):
        rr, rec = relsem_release_hook('FANUC 机床问题', _CTX_FANUC, dict(rr_in))
        assert rr == rr_in and rec is None


def test_no_contexts_no_release():
    os.environ['OOC_RELSEM'] = '1'
    rr, rec = relsem_release_hook('FANUC 机床问题', [], dict(_REL))
    assert rr == _REL and rec is None


def test_min_threshold_env():
    os.environ['OOC_RELSEM'] = '1'
    os.environ['OOC_RELSEM_LATIN_MIN'] = '2'
    rr, rec = relsem_release_hook('FANUC 机床问题', _CTX_FANUC, dict(_REL))
    assert rr == _REL and rec is None


def test_latin_hits_unit():
    assert relsem_latin_hits('FANUC 的 EMG 报警', _CTX_FANUC) == 1
    assert relsem_latin_hits('fanuc 报警', _CTX_FANUC) == 1
    assert relsem_latin_hits('PLC 报警', _CTX_FANUC) == 0
    assert relsem_latin_hits('FANUC 报警', []) == 0
