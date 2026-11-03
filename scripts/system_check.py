# -*- coding: utf-8 -*-
"""系统状态自检：复用 setup_wizard 的检查逻辑，输出结构化 JSON。

只读：不安装、不启动、不写库、不改 .env；不返回任何密钥值。
命令行自检见 `python scripts/setup_wizard.py --check`，本模块只给 API/前端提供同一份结果。
"""
import contextlib
import io
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import setup_wizard as sw  # noqa: E402

_LOCK = threading.Lock()

_HINTS = (
    ("缺少 Python 依赖", "运行 python scripts/setup_wizard.py --install-deps"),
    ("依赖版本与 requirements.txt 不一致", "按 requirements.txt 对齐版本"),
    ("缺少模型", "运行 python scripts/setup_wizard.py --pull-models"),
    ("Ollama 未运行", "先安装/启动 Ollama（python scripts/setup_wizard.py --install-ollama）"),
    ("webui/dist 未构建", "运行 python scripts/setup_wizard.py --build-webui"),
    ("webui/node_modules 不存在", "重新构建时 python scripts/setup_wizard.py --build-webui 会自动 npm install"),
    ("webui/src 比 dist 新", "运行 python scripts/setup_wizard.py --build-webui"),
    ("/ui/ 不是新版前端", "重建 webui/dist 后重启 API"),
    ("Neo4j 连接/查询失败", "检查 Neo4j 服务、NEO4J_USER/NEO4J_PASSWORD 与 SERVING_NEO4J_URI"),
    ("Neo4j 端口不可达", "启动 Neo4j 服务，或把 SERVING_NEO4J_URI 指到已有实例"),
    ("服务 neo4j-pump 不存在", "安装 Neo4j 5.x，或改用已有实例"),
    (".env 不存在", "运行 python scripts/setup_wizard.py --init-env"),
    ("缺少关键配置键", "补齐 .env 中缺少的键"),
    ("LLM_PROVIDER", "生产答案必须使用本地模型：LLM_PROVIDER=local"),
    ("本地模型地址不是本机", "LOCAL_LLM_BASE_URL 指向 http://localhost:11434"),
    ("旧 Gradio", "运行 python scripts/service_control.py stop 清理旧前端"),
)


def _hint(name):
    for key, text in _HINTS:
        if key in name:
            return text
    return ""


def _clean_detail(detail):
    text = str(detail or "")
    root = str(sw.ROOT)
    for prefix in (root + "\\", root + "/"):
        text = text.replace(prefix, "")
    if text == root:
        text = "."
    return text


class JsonReport:
    def __init__(self):
        self.items = []
        self.section = "其他"

    def add(self, state, name, detail=""):
        status = str(state).lower()
        self.items.append({
            "section": self.section,
            "status": status,
            "name": str(name),
            "detail": _clean_detail(detail),
            "hint": "" if status == "ok" else _hint(str(name)),
        })

    def summary(self):
        return sum(1 for item in self.items if item["status"] == "fail")


def _group(items):
    sections = []
    by_name = {}
    for item in items:
        name = item["section"]
        if name not in by_name:
            entry = {"name": name, "items": []}
            by_name[name] = entry
            sections.append(entry)
        by_name[name]["items"].append({
            "status": item["status"],
            "name": item["name"],
            "detail": item["detail"],
            "hint": item["hint"],
        })
    return sections


def collect():
    started = time.time()
    rep = JsonReport()
    with _LOCK:
        buf = io.StringIO()
        try:
            with contextlib.redirect_stdout(buf):
                sw.check_all(rep)
        except Exception as e:
            rep.section = "自检"
            rep.add("FAIL", "自检执行失败", "%s: %s" % (type(e).__name__, str(e)[:300]))
    counts = {
        "ok": sum(1 for item in rep.items if item["status"] == "ok"),
        "warn": sum(1 for item in rep.items if item["status"] == "warn"),
        "fail": sum(1 for item in rep.items if item["status"] == "fail"),
    }
    return {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "elapsed_ms": int((time.time() - started) * 1000),
        "ready": counts["fail"] == 0,
        "counts": counts,
        "sections": _group(rep.items),
    }
