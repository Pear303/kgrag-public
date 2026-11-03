"""管道共享上下文：在各阶段间传递状态。"""
from dataclasses import dataclass, field
from typing import Any, Optional

from backend.parser import Chunk


@dataclass
class PipelineContext:
    """管道阶段间共享的运行状态。

    每个阶段从 context 读取所需输入，执行后将结果写回。
    """

    # === 输入 ===
    manual_path: str = ""

    # === Stage 1: Parse ===
    source_name: str = ""
    chunks: list[Chunk] = field(default_factory=list)
    triples_with_prov: list[dict] = field(default_factory=list)
    triples: list[tuple[str, str, str]] = field(default_factory=list)

    # === Stage 2: Write ===
    write_count: int = 0

    # === Setup (Stage 0) ===
    vector_mode: str = ""
    llm: Any = None
    embed: Any = None
    store: Any = None
    driver: Any = None

    # === Stage 3: LLM Extract ===
    all_llm_triples: list[tuple[str, str, str]] = field(default_factory=list)
    llm_compare_summary: Optional[dict] = None

    # === Stage 4: Communities ===
    communities: list[dict] = field(default_factory=list)
    community_count: int = 0

    # === Stage 5: Engines ===
    manual_eng: Any = None
    comm_eng: Any = None
    auto_eng: Any = None
    router: Any = None

    # === Stage 6: Verify ===
    micro_rows: list = field(default_factory=list)

    # === Stage 7: Visualize ===
    viz_output_path: str = ""
