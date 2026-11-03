"""管道包：将 main.py 中的单块逻辑拆分为独立可调用的阶段。

模块结构：
  context.py   — PipelineContext 数据类，阶段间共享状态
  stages.py    — 7 个独立阶段函数，每个可独立测试
  runner.py    — PipelineRunner 编排器，链式执行所有阶段
  extract.py   — 纯 LLM 抽取模式管道（v2 extract mode）
"""
from pipeline.context import PipelineContext
from pipeline.runner import PipelineRunner
from pipeline.extract import run_extract_pipeline
from pipeline.stages import (
    run_parse_stage,
    run_write_stage,
    run_extract_stage,
    run_community_stage,
    run_engine_setup_stage,
    run_verify_stage,
    run_visualize_stage,
)

__all__ = [
    "PipelineContext",
    "PipelineRunner",
    "run_extract_pipeline",
    "run_parse_stage",
    "run_write_stage",
    "run_extract_stage",
    "run_community_stage",
    "run_engine_setup_stage",
    "run_verify_stage",
    "run_visualize_stage",
]
