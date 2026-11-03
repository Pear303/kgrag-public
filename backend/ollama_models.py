# -*- coding: utf-8 -*-
"""Ollama 本地模型目录：列出可用模型并区分 embedding / 答案生成 LLM。

只做读操作（/api/tags、/api/show），不加载模型、不触发推理。
"""
import json
import re
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

DEFAULT_BASE_URL = "http://localhost:11434"

# 生产答案模型上限：本机登记 qwen3.5:4b 的 parameter_size 是 4.2B，
# 因此"4B 级"按 4.5B 判，不按字面 4.0 卡死现有生产模型。
ANSWER_PARAM_LIMIT_B = 4.5


class OllamaUnavailable(RuntimeError):
    pass


def _url(base_url: Optional[str], path: str) -> str:
    return (base_url or DEFAULT_BASE_URL).rstrip("/") + path


def _get(path: str, base_url: Optional[str] = None, timeout: float = 5.0) -> dict:
    try:
        with urllib.request.urlopen(_url(base_url, path), timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8"))
    except Exception as e:
        raise OllamaUnavailable("Ollama 不可访问：%s" % e)


def _post(path: str, payload: dict, base_url: Optional[str] = None,
          timeout: float = 20.0) -> dict:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        _url(base_url, path), data=data,
        headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8"))
    except Exception as e:
        raise OllamaUnavailable("Ollama %s 调用失败：%s" % (path, e))


def parse_params_b(text: str) -> Optional[float]:
    """把 '4.2B' / '566.70M' / '1.2K' 统一换算成十亿参数。"""
    if not text:
        return None
    m = re.match(r"\s*([\d.]+)\s*([BMK])\b", str(text), re.I)
    if not m:
        return None
    v = float(m.group(1))
    unit = m.group(2).upper()
    if unit == "B":
        return round(v, 3)
    if unit == "M":
        return round(v / 1000.0, 6)
    return round(v / 1_000_000.0, 9)


def _classify(name: str, caps: list, family: str) -> str:
    caps = [str(c).lower() for c in (caps or [])]
    if "embedding" in caps:
        return "embedding"
    if "completion" in caps:
        return "llm"
    # capabilities 拿不到时的保守回退：只认名字/家族，绝不把生成模型误当 embedding。
    low = (name or "").lower()
    if "embed" in low or family == "bert":
        return "embedding"
    return "unknown"


def list_models(base_url: Optional[str] = None, include_show: bool = True) -> list:
    """返回本地模型清单（按名称排序）。

    每项：name / kind(embedding|llm|unknown) / capabilities / parameter_size /
    params_b / answer_eligible / size_gb / family / modified_at。
    """
    tags = _get("/api/tags", base_url)
    rows = [m for m in (tags.get("models") or []) if m.get("name")]
    # /api/show 单次约 2s，串行 7 个模型会让设置页首开等 15s；
    # 并发探测后仍在内存缓存 5 分钟（见 api_server._CATALOG_TTL_S）。
    caps_by_name = {}
    if include_show and rows:
        def _caps(n):
            try:
                return _post("/api/show", {"model": n}, base_url,
                             timeout=20.0).get("capabilities") or []
            except Exception:
                return []

        with ThreadPoolExecutor(max_workers=min(8, len(rows))) as pool:
            for name, caps in zip([r["name"] for r in rows],
                                  pool.map(_caps, [r["name"] for r in rows])):
                caps_by_name[name] = caps
    out = []
    for m in rows:
        name = m["name"]
        details = m.get("details") or {}
        family = str(details.get("family") or "")
        caps = caps_by_name.get(name, []) if include_show else []
        kind = _classify(name, caps, family)
        param_text = str(details.get("parameter_size") or "")
        params_b = parse_params_b(param_text)
        out.append({
            "name": name,
            "kind": kind,
            "capabilities": [str(c) for c in caps],
            "family": family,
            "parameter_size": param_text,
            "params_b": params_b,
            "answer_eligible": (kind == "llm"
                                and (params_b is None
                                     or params_b <= ANSWER_PARAM_LIMIT_B)),
            "size_gb": round(float(m.get("size") or 0) / 1e9, 2),
            "modified_at": m.get("modified_at") or "",
        })
    out.sort(key=lambda x: (x["kind"] != "embedding", x["name"]))
    return out


def find_model(name: str, base_url: Optional[str] = None) -> Optional[dict]:
    for m in list_models(base_url=base_url, include_show=True):
        if m["name"] == name:
            return m
    return None
