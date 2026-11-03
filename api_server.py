# -*- coding: utf-8 -*-
"""REST API：KG-RAG 问答服务（FastAPI）。

端点：
- GET  /health              服务健康与数据概况
- GET  /system/check        只读系统自检（依赖/配置/数据/服务/前端）
- POST /query               微观问答（图谱路径 + 答案生成，含 OOC 拒答门）
- POST /query/macro         宏观问答（社区召回 + 总结）
- POST /query/compare       KG-RAG vs Vector RAG 对比

启动：
    uvicorn api_server:app --host 0.0.0.0 --port 8001
或：
    python api_server.py
"""
import os
import sys
import uuid
import json
import time
import threading
import logging
from collections import OrderedDict
from datetime import datetime
from typing import Literal, Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

logger = logging.getLogger("kgrag.api_server")

from fastapi import FastAPI, HTTPException, UploadFile, File, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
import uvicorn

from serve import (KGRAGService, default_embedding_model, default_llm_model,
                   model_cache_paths)
from backend.manual_registry import ManualRegistry
from backend.task_manager import TaskManager
from backend.document_loader import parse_document_to_md
from backend.build_pipeline import parsed_md_path_for
from backend.ollama_models import list_models, OllamaUnavailable
from backend.model_prefs import load_prefs, save_prefs, clear_prefs

app = FastAPI(title="KGRAG 工业故障推理引擎 API", version="1.0.0")


# ---------- 可选 API 令牌（2026-10-26 走查修复：默认关闭，行为不变） ----------
# .env 设 API_TOKEN=<随机串> 后，除健康检查/静态页外所有端点要求请求头
# X-API-Token。前端 401 时会弹输入框并把令牌存 localStorage。
@app.middleware("http")
async def _api_token_guard(request: Request, call_next):
    token = (os.getenv("API_TOKEN") or "").strip()
    path = request.url.path
    open_paths = ("/health", "/health/lite", "/ui", "/static", "/chain-view", "/docs",
                  "/openapi.json")
    if token and not any(path == p or path.startswith(p + "/") for p in open_paths):
        if request.headers.get("X-API-Token") != token:
            from fastapi.responses import JSONResponse
            return JSONResponse(status_code=401, content={"detail": "缺少或错误的 X-API-Token"})
    return await call_next(request)


# 挂载静态资源（cytoscape 本地库），供推理链可视化 iframe 使用
_STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
app.mount("/static", StaticFiles(directory=_STATIC_DIR), name="static")

# 挂载打包后的新版前端（webui/dist，Vue3+Element Plus）。
# 只在已构建时挂载，避免未 npm run build 时启动报错。
_WEBUI_DIST = os.path.join(os.path.dirname(os.path.abspath(__file__)), "webui", "dist")
if os.path.isdir(_WEBUI_DIST):
    app.mount("/ui", StaticFiles(directory=_WEBUI_DIST, html=True), name="webui")

# 推理链可视化独立页面（供 iframe src 加载，避免 srcdoc 转义问题）
_CHAIN_VIEW_TEMPLATE = """<!DOCTYPE html>
<html><head><meta charset="utf-8">
<script src="/static/vendor/cytoscape.min.js"></script>
<style>
 html,body{margin:0;font-family:'Microsoft YaHei',sans-serif;background:#fafafa}
 #cy{width:100%;height:580px}
 .detail-panel{padding:10px 14px;font-size:12px;color:#333;border-top:1px solid #eee;background:#fff;min-height:40px}
 .detail-panel .label{font-weight:bold;color:#555}
 .detail-panel .text{margin-top:4px;white-space:pre-wrap;word-break:break-all}
</style></head>
<body>
<div id="cy"></div>
<div class="detail-panel" id="detail"><span class="label">点击节点查看详情</span></div>
<script>
var DATA = __DATA__;
var cy = cytoscape({
  container: document.getElementById('cy'),
  elements: DATA.nodes.concat(DATA.edges),
  style: [
    { selector: 'node', style: {
        'label': 'data(label)', 'font-size': '12px', 'text-wrap': 'wrap',
        'text-max-width': '130px', 'text-valign': 'center', 'text-halign': 'center',
        'background-color': 'data(color)', 'border-width': 2, 'border-color': '#fff',
        'color': '#fff', 'text-outline-width': 2, 'text-outline-color': 'data(color)',
        'width': 'mapData(size, 20, 44, 28, 72)', 'height': 'mapData(size, 20, 44, 28, 72)'
    }},
    { selector: 'edge', style: {
        'width': 2, 'line-color': 'data(color)', 'target-arrow-shape': 'triangle',
        'target-arrow-color': 'data(color)', 'curve-style': 'bezier',
        'label': 'data(label)', 'font-size': '10px', 'color': '#555',
        'text-background-color': '#fff', 'text-background-opacity': 0.85,
        'text-background-padding': '2px'
    }}
  ],
  wheelSensitivity: 0.2,
  minZoom: 0.05,
  maxZoom: 5,
  layout: { name: 'breadthfirst', directed: true, spacingFactor: 1.1, padding: 20 }
});
var detailDiv = document.getElementById('detail');
cy.on('tap', 'node', function(e){
  var d = e.target.data();
  var txt = d.label + (d.text && d.text !== d.label ? '\\n\\n' + d.text : '');
  detailDiv.innerHTML = '<span class="label">' + d.label + '</span><div class="text">' + txt.replace(/\\n/g, '<br>') + '</div>';
});
cy.on('tap', function(e){ if (e.target === cy) { detailDiv.innerHTML = '<span class="label">点击节点查看详情</span>'; } });
setTimeout(function(){ cy.resize(); cy.fit(); cy.center(); }, 100);
window.addEventListener('resize', function(){ cy.resize(); cy.fit(); });
</script></body></html>"""


# ---------- 推理链元素暂存（前端先 POST 拿短 id，iframe 用 id 引用） ----------
# 旧做法把整张图的 JSON urlencode 进 iframe 的 URL：图一大 URL 就爆，浏览器每次都要
# 重解析一长串转义。改成服务端暂存 + 短 id，URL 只剩几十字节。
_CHAIN_CACHE = OrderedDict()
_CHAIN_CACHE_MAX = 200


@app.post("/chain/store")
async def chain_store(request: Request):
    """存一份 cytoscape elements，返回短 id 供 /chain-view?id= 引用。"""
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="body 必须是 JSON")
    cid = uuid.uuid4().hex[:12]
    _CHAIN_CACHE[cid] = json.dumps(body, ensure_ascii=False)
    while len(_CHAIN_CACHE) > _CHAIN_CACHE_MAX:
        _CHAIN_CACHE.popitem(last=False)
    return {"id": cid}


@app.get("/chain-view", response_class=HTMLResponse)
async def chain_view(id: str = "", data: str = ""):
    """返回独立推理链可视化页面。

    - id：服务端暂存元素的引用（推荐，URL 短）
    - data：urlencoded JSON（旧路径，保留兼容）
    """
    decoded = ""
    if id and id in _CHAIN_CACHE:
        decoded = _CHAIN_CACHE[id]
    elif data:
        import urllib.parse
        decoded = urllib.parse.unquote(data)
    try:
        json.loads(decoded)
    except Exception:
        decoded = '{"nodes": [], "edges": []}'
    return _CHAIN_VIEW_TEMPLATE.replace("__DATA__", decoded)

# 全局单例（进程内复用引擎，避免每次请求重建）
_service = None


def _build_service(manual: dict) -> KGRAGService:
    """根据 registry 手册条目构建对应 KGRAGService。

    prebuilt 手册（500案例/IS620F/GSK983 等）：按 round/extract 命名加载
    既有缓存与图谱；上传手册：以 manual_id 为 round 走标准产物路径。
    构建引擎实例前必须同步 SERVING_EXTRACT，避免 KGRAGService 仍从旧手册
    extract 加载 entity_chunk_map，导致证据回填错乱（跨手册污染）。
    """
    from serve import KGRAGService
    prebuilt = manual.get("prebuilt") or {}
    # env 是权威口径：服务器按 SERVING_MANUAL/SERVING_ROUND 启动时，UI 把同一本
    # 手册"设为当前"必须回到同一轮（2026-10-26 走查：曾从 closeA2/301 跳到
    # 注册表的 v15/302，顶栏数字跟着跳，运维无法判断哪份是权威）。
    env_manual = os.getenv("SERVING_MANUAL") or ""
    env_round = os.getenv("SERVING_ROUND") or ""
    serving_uri = manual.get("serving_uri") or None
    if prebuilt.get("round_name") and prebuilt.get("extract"):
        round_name = prebuilt["round_name"]
        if env_manual and env_round and manual["source_manual"] == env_manual:
            round_name = env_round
        os.environ["SERVING_EXTRACT"] = prebuilt["extract"]
        return KGRAGService(
            manual_name=prebuilt.get("manual_name") or manual["source_manual"],
            round_name=round_name,
            neo4j_uri=serving_uri,
        )
    manual_id = manual["manual_id"]
    os.environ["SERVING_EXTRACT"] = f"llm_extract_{manual_id}.json"
    from config import NEO4J_BUILD_URI
    return KGRAGService(manual_name=manual["source_manual"], round_name=manual_id,
                        neo4j_uri=serving_uri or NEO4J_BUILD_URI)


def get_service() -> KGRAGService:
    global _service
    if _service is None:
        # 优先读取环境变量指定的手册（部署/测试时显式指定）
        env_manual = os.getenv("SERVING_MANUAL")
        env_round = os.getenv("SERVING_ROUND")
        if env_manual and env_round:
            _service = KGRAGService(manual_name=env_manual, round_name=env_round)
        else:
            # 回退：恢复 registry 中"当前激活"手册
            current = _registry.get_current() if _registry is not None else None
            if current is not None:
                try:
                    _service = _build_service(current)
                except Exception as e:
                    logger.error("恢复 current 手册失败，回退默认: %s", e)
                    _service = KGRAGService()
            else:
                _service = KGRAGService()
    return _service


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000, description="故障诊断问题")
    max_ctx: int = Field(12, ge=1, le=20)


class MacroQueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    max_comm: int = Field(3, ge=1, le=10)


class CompareRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)


@app.on_event("startup")
async def _prewarm_service():
    """后台线程预热：不预热的话重启后 /health/lite 永远 ready=False，
    首个问题要同步等几分钟引擎构建，运维看着像挂了（2026-10-26 走查实测）。
    serve.py 有同款，但它挂在 serve 自己的 app 上，api_server 沾不到光。"""
    import threading

    def _build():
        try:
            get_service()
            logger.info("预热完成，服务就绪")
        except Exception as e:
            logger.error("预热失败（首次请求将重试构建）: %s", e)

    threading.Thread(target=_build, name="kgrag-api-prewarm", daemon=True).start()


@app.get("/health")
def health():
    return get_service().health()


@app.get("/health/lite")
def health_lite():
    """轻量健康检查：只读进程内状态 + .env，绝不实例化 KGRAGService。

    与 /health 的区别：/health 会 get_service()，而构建服务要建向量索引
    （VectorStoreIndex.from_documents ⇒ 走 Ollama embedding），数十秒且会与
    本机正在跑的答案生成抢 Ollama 单槽。前端页面加载只需要"当前是哪本手册"，
    不该为此触发一次模型构建。服务已构建时，顺带把它的 health 合并进来。
    """
    svc = _service
    resp = {
        "status": "ok",
        "ready": svc is not None,
        "manual": os.getenv("SERVING_MANUAL", ""),
        "round": os.getenv("SERVING_ROUND", ""),
        "neo4j_uri": os.environ.get("NEO4J_URI"),
    }
    if svc is not None:
        try:
            resp.update(svc.health())
        except Exception as e:
            resp["service_health_error"] = str(e)
    return resp


@app.get("/system/check")
def system_check():
    """只读系统自检：依赖、.env 键存在性、Neo4j、Ollama、缓存、前端构建、端口、工具。"""
    from scripts.system_check import collect
    return collect()


# ---------- 本地模型目录与切换 ----------
_model_switch_lock = threading.Lock()
_model_job = {
    "state": "idle",          # idle | building | ready | error
    "job_id": None,
    "target_embedding": None,
    "stage": None,
    "done": 0,
    "total": 0,
    "started_at": None,
    "finished_at": None,
    "error": None,
}
_catalog_cache = {"ts": 0.0, "data": None, "error": None}
_CATALOG_TTL_S = 300.0


class ModelChoice(BaseModel):
    model: str = Field(..., min_length=1, max_length=200)


def _job_snapshot() -> dict:
    with _model_switch_lock:
        return dict(_model_job)


def _job_update(**kw) -> None:
    with _model_switch_lock:
        _model_job.update(kw)


def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _persist_model_choice(embedding_model=None, llm_model=None) -> None:
    """只持久化"偏离 .env 默认"的选择；两者都默认时删除偏好文件。

    这样恢复默认后重启真正跟随 .env（以后改 .env 默认值不会被旧偏好钉住）。
    """
    emb = embedding_model or None
    llm = llm_model or None
    keep_emb = emb if emb and emb != default_embedding_model() else None
    keep_llm = llm if llm and llm != default_llm_model() else None
    clear_prefs()
    if keep_emb or keep_llm:
        save_prefs(embedding_model=keep_emb, llm_model=keep_llm)


def _catalog(force: bool = False) -> dict:
    now = time.time()
    if (not force and _catalog_cache["data"] is not None
            and now - _catalog_cache["ts"] < _CATALOG_TTL_S):
        return _catalog_cache["data"]
    models = list_models()
    data = {
        "models": models,
        "embedding_models": [m for m in models if m["kind"] == "embedding"],
        "answer_models": [m for m in models if m["answer_eligible"]],
        "fetched_at": _now_iso(),
    }
    _catalog_cache.update(ts=now, data=data, error=None)
    return data


def _selected_models(svc=None) -> dict:
    prefs = load_prefs()
    svc = _service if svc is None else svc
    if svc is not None:
        return {"embedding_model": svc.embedding_model,
                "llm_model": svc.llm_model,
                "source": "service"}
    return {
        "embedding_model": prefs.get("embedding_model") or default_embedding_model(),
        "llm_model": prefs.get("llm_model") or default_llm_model(),
        "source": "prefs",
    }


def _json_head(path: str, n: int = 400) -> dict:
    """只读 JSON 文件开头，取写在最前面的元数据（不解析大数组）。"""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return {"head": f.read(n)}
    except Exception:
        return {}


def _file_meta(path: str) -> dict:
    if not path:
        return {"path": None, "exists": False}
    try:
        st = os.stat(path)
        return {
            "path": path,
            "exists": True,
            "size_mb": round(st.st_size / 1048576.0, 2),
            "mtime": datetime.fromtimestamp(st.st_mtime).isoformat(timespec="seconds"),
        }
    except OSError:
        return {"path": path, "exists": False}


def _model_cache_report(round_name: str, model: str) -> dict:
    """按模型列出四类缓存是否已落盘（只读文件系统，不触发计算）。"""
    import re as _re
    paths = model_cache_paths(round_name, model)
    vector_meta_path = os.path.join(paths["vector"], "_meta.json")
    vector = _file_meta(vector_meta_path)
    if vector.get("exists"):
        try:
            with open(vector_meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            vector["embed_sig"] = meta.get("embed_sig")
            vector["n_docs"] = meta.get("n_docs")
        except Exception:
            pass

    def sidecar(path):
        info = _file_meta(path)
        if info.get("exists"):
            head = _json_head(path).get("head") or ""
            m = _re.search(r'"model"\s*:\s*"([^"]+)"', head)
            n = _re.search(r'"n"\s*:\s*(\d+)', head)
            if m:
                info["file_model"] = m.group(1)
            if n:
                info["n"] = int(n.group(1))
        return info

    chunk = sidecar(paths["chunk"])
    entity = sidecar(paths["entity"]) if paths["entity"] else {
        "path": None, "exists": True, "source": "neo4j（默认模型）"}
    communities = sidecar(paths["community_embeddings"] or paths["communities"])
    sliding = _file_meta(paths["sliding_embed"])
    opt_c = os.getenv("OPT_C", "0") == "1"
    complete = bool(
        vector.get("exists") and chunk.get("exists")
        and communities.get("exists")
        and (paths["is_default"] or entity.get("exists"))
        and (not opt_c or sliding.get("exists")))
    return {
        "model": model,
        "is_default": paths["is_default"],
        "vector": vector,
        "chunk": chunk,
        "entity": entity,
        "communities": communities,
        "sliding": dict(sliding, enabled=opt_c),
        "cache_complete": complete,
    }


def _start_embedding_switch(target: str) -> dict:
    if _service is None:
        raise HTTPException(status_code=409, detail="服务尚未就绪，无法切换 embedding")
    with _model_switch_lock:
        if _model_job["state"] == "building":
            raise HTTPException(
                status_code=409,
                detail="已有模型切换任务在构建中，请等待完成（job %s）"
                       % _model_job.get("job_id"))
        job_id = uuid.uuid4().hex[:8]
        _model_job.update(
            state="building", job_id=job_id, target_embedding=target,
            stage="queued", done=0, total=0,
            started_at=_now_iso(), finished_at=None, error=None)
    threading.Thread(target=_run_embedding_switch, args=(job_id, target),
                     name="model-switch-" + job_id, daemon=True).start()
    return _job_snapshot()


def _run_embedding_switch(job_id: str, target: str) -> None:
    global _service
    try:
        base = _service
        if base is None:
            raise RuntimeError("服务尚未就绪")
        logger.info("开始切换 embedding: %s -> %s", base.embedding_model, target)

        def progress(stage, done, total):
            _job_update(stage=stage, done=int(done), total=int(total))

        new_svc = KGRAGService(
            manual_name=base.manual_name,
            round_name=base.round_name,
            neo4j_uri=base._neo4j_uri,
            extract_path=base.extract_path,
            embedding_model=target,
            llm_model=base.llm_model,
            progress=progress,
        )
        # 构建期间答案 LLM 可能被热切过：以切换完成时的当前选择为准
        current = _service
        if current is not base:
            raise RuntimeError("构建期间服务已切换手册或重建，本次模型切换未应用")
        if current is not None and current.llm_model != new_svc.llm_model:
            new_svc.switch_llm(current.llm_model)
        _persist_model_choice(target, new_svc.llm_model)
        _service = new_svc
        logger.info("embedding 切换完成: %s（vector=%s, entity=%s, chunk=%s, comm=%s）",
                    target,
                    getattr(new_svc.vec_eng, "index_source", None),
                    getattr(new_svc.manual_eng, "embedding_source", None),
                    getattr(new_svc.comm_eng, "_chunk_index_source", None),
                    new_svc.community_embedding_source)
        _job_update(state="ready", stage="ready", done=1, total=1,
                    finished_at=_now_iso(), error=None)
    except Exception as e:
        logger.exception("embedding 切换失败: %s", target)
        _job_update(state="error", stage="error", finished_at=_now_iso(),
                    error=str(e))


@app.get("/models/local")
def models_local(refresh: bool = False):
    """本地 Ollama 模型目录 + 当前选择 + 切换任务状态。不实例化 KGRAGService。"""
    try:
        cat = _catalog(force=refresh)
        return {
            "ok": True,
            "ollama": {"reachable": True,
                       "base_url": os.getenv("LOCAL_LLM_BASE_URL",
                                             "http://localhost:11434")},
            **cat,
            "selected": _selected_models(),
            "defaults": {"embedding_model": default_embedding_model(),
                         "llm_model": default_llm_model()},
            "job": _job_snapshot(),
        }
    except OllamaUnavailable as e:
        return {
            "ok": False,
            "ollama": {"reachable": False,
                       "base_url": os.getenv("LOCAL_LLM_BASE_URL",
                                             "http://localhost:11434"),
                       "error": str(e)},
            "models": [], "embedding_models": [], "answer_models": [],
            "selected": _selected_models(),
            "defaults": {"embedding_model": default_embedding_model(),
                         "llm_model": default_llm_model()},
            "job": _job_snapshot(),
        }


@app.get("/models/status")
def models_status(refresh: bool = False):
    """当前服务模型 + 每个候选 embedding 的缓存状态 + 切换任务。"""
    # 一次请求只看一个服务快照：切换恰好发生在请求中途时，
    # selected/health/active_cache 仍保持同一代，不出现"旧选择+新缓存"。
    svc = _service
    round_name = (svc.round_name if svc is not None
                  else os.getenv("SERVING_ROUND") or "13_GSK983数控系统维修手册_中文")
    selected = _selected_models(svc)
    # 状态接口要快：目录探测（/api/show 约 2s）只在调用方显式 refresh 时做，
    # 否则只用已有缓存；前端拿到目录后会再拉一次状态补全缓存表。
    cat = _catalog_cache["data"]
    ollama_error = None
    if refresh:
        try:
            cat = _catalog(force=True)
        except OllamaUnavailable as e:
            cat = None
            ollama_error = str(e)
    if cat:
        emb_models = [m["name"] for m in cat["embedding_models"]]
    else:
        emb_models = [selected["embedding_model"]]
        if default_embedding_model() != selected["embedding_model"]:
            emb_models.append(default_embedding_model())
    caches = [_model_cache_report(round_name, m) for m in emb_models]
    resp = {
        "ok": True,
        "ready": _service is not None,
        "round": round_name,
        "selected": selected,
        "defaults": {"embedding_model": default_embedding_model(),
                     "llm_model": default_llm_model()},
        "caches": caches,
        "job": _job_snapshot(),
    }
    if ollama_error:
        resp["ollama_error"] = ollama_error
    if svc is not None:
        try:
            resp["health"] = svc.health()
            resp["active_cache"] = svc.cache_status()
        except Exception as e:
            resp["health_error"] = str(e)
    return resp


@app.post("/models/embedding")
def models_switch_embedding(req: ModelChoice):
    """切换检索 embedding（向量/图谱/社区/块索引统一换），后台构建并热替换。"""
    try:
        cat = _catalog()
    except OllamaUnavailable as e:
        raise HTTPException(status_code=503, detail=str(e))
    info = next((m for m in cat["embedding_models"] if m["name"] == req.model), None)
    if info is None:
        raise HTTPException(status_code=400,
                            detail="%s 不是可用的本地 embedding 模型" % req.model)
    if _service is not None and req.model == _service.embedding_model:
        return {"ok": True, "changed": False, "message": "已经是当前 embedding 模型",
                "job": _job_snapshot()}
    job = _start_embedding_switch(req.model)
    return {"ok": True, "changed": True, "job": job,
            "message": "已开始构建该模型的索引；构建期间旧模型继续服务"}


@app.post("/models/answer")
def models_switch_answer(req: ModelChoice):
    """热切换答案 LLM（秒级，不重建索引）。"""
    try:
        cat = _catalog()
    except OllamaUnavailable as e:
        raise HTTPException(status_code=503, detail=str(e))
    info = next((m for m in cat["answer_models"] if m["name"] == req.model), None)
    if info is None:
        raise HTTPException(
            status_code=400,
            detail="%s 不是可用的本地答案模型（必须是 completion 模型且不超过 4.5B 参数）"
                   % req.model)
    if _service is None:
        selected = _selected_models()
        _persist_model_choice(selected["embedding_model"], req.model)
        return {"ok": True, "applied": False,
                "message": "服务尚未就绪，已保存选择，服务构建后生效",
                "selected": _selected_models()}
    _service.switch_llm(req.model)
    _persist_model_choice(_service.embedding_model, req.model)
    return {"ok": True, "applied": True,
            "selected": _selected_models(), "health": _service.health()}


@app.post("/models/reset")
def models_reset():
    """恢复 .env 的出厂默认模型。embedding 需要重建时走后台任务。"""
    if _job_snapshot()["state"] == "building":
        raise HTTPException(status_code=409, detail="模型切换构建中，请完成后再恢复默认")
    before = _selected_models()
    clear_prefs()
    emb_default = default_embedding_model()
    llm_default = default_llm_model()
    resp = {"ok": True, "defaults": {"embedding_model": emb_default,
                                     "llm_model": llm_default},
            "changed_embedding": before["embedding_model"] != emb_default}
    if _service is None:
        resp["job"] = _job_snapshot()
        return resp
    if _service.llm_model != llm_default:
        try:
            _service.switch_llm(llm_default)
        except Exception as e:
            resp["llm_error"] = str(e)
    if _service.embedding_model != emb_default:
        resp["job"] = _start_embedding_switch(emb_default)
        resp["changed_embedding"] = True
    else:
        resp["changed_embedding"] = False
        resp["job"] = _job_snapshot()
    resp["selected"] = _selected_models()
    return resp


@app.post("/query")
def query(req: QueryRequest):
    try:
        return get_service().query(req.question, max_ctx=req.max_ctx)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"查询失败: {e}")


@app.post("/query/stream")
def query_stream(req: QueryRequest):
    """流式微观问答（SSE）：meta → answer_delta* → done。

    与 /query 同源同口径，只多一层"边生成边吐"。用同步生成器 ⇒ Starlette 会把它
    放到线程池里跑，事件循环不被阻塞（与 serve.py 的重活下沉同一思路）。
    """
    def _gen():
        try:
            svc = get_service()
            for ev in svc.stream_query(req.question, max_ctx=req.max_ctx):
                yield "data: " + json.dumps(ev, ensure_ascii=False) + "\n\n"
        except Exception as e:
            logger.exception("流式查询失败")
            yield "data: " + json.dumps(
                {"type": "error", "message": f"查询失败: {e}"},
                ensure_ascii=False) + "\n\n"

    return StreamingResponse(
        _gen(), media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.post("/query/macro")
def query_macro(req: MacroQueryRequest):
    try:
        return get_service().query_macro(req.question, max_comm=req.max_comm)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"宏观查询失败: {e}")


@app.post("/query/compare")
def query_compare(req: CompareRequest):
    try:
        return get_service().query_compare(req.question)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"对比查询失败: {e}")


# 手册与任务全局实例（进程内复用）
_registry = ManualRegistry()
_task_manager = TaskManager(_registry)

def _sweep_orphan_tasks() -> int:
    """服务重启后，进程内线程已不存在：把遗留的 queued/running 任务标记为失败，
    避免任务面板永远显示一条不会跑的"排队中"任务误导运维。"""
    n = 0
    from datetime import datetime
    for t in _task_manager.list_tasks():
        if t.get("status") in ("queued", "running"):
            t["status"] = "failed"
            t["error"] = "服务重启，任务线程已不存在（孤儿任务）。请重新提交。"
            t["finished_at"] = t.get("finished_at") or datetime.now().isoformat(timespec="seconds")
            _task_manager._write(t)
            n += 1
    return n

_sweep_orphan_tasks()

ALLOWED_EXT = {".md", ".markdown", ".txt", ".pdf", ".docx"}


@app.get("/manuals")
def list_manuals():
    return _registry.list_manuals()


@app.get("/manuals/{manual_id}")
def get_manual(manual_id: str):
    m = _registry.get_manual(manual_id)
    if m is None:
        raise HTTPException(status_code=404, detail=f"手册不存在: {manual_id}")
    return m


@app.post("/manuals")
async def upload_manual(file: UploadFile = File(...)):
    # 归一文件名，防止路径遍历（如 ../ 段越界写入）
    filename = os.path.basename(file.filename)
    ext = os.path.splitext(filename)[1].lower()
    if ext not in ALLOWED_EXT:
        raise HTTPException(status_code=400,
                            detail=f"不支持格式 {ext}，仅支持 {sorted(ALLOWED_EXT)}")
    from backend.manual_registry import _slugify
    manual_id = _slugify(filename)
    if _registry.get_manual(manual_id) is not None:
        manual_id = f"{manual_id}_{uuid.uuid4().hex[:6]}"
    # 保存上传原文
    from config import MANUALS_DIR
    os.makedirs(MANUALS_DIR, exist_ok=True)
    dest = os.path.join(MANUALS_DIR, filename)
    with open(dest, "wb") as f:
        f.write(await file.read())
    # 解析为统一 md
    md_path = parsed_md_path_for(manual_id)
    parse_document_to_md(dest, md_path)
    m = _registry.add_manual({
        "manual_id": manual_id,
        "source_manual": filename,
        "file_path": dest,
        "uploaded_at": _now(),
        "status": "calibrating",
    })
    # C1 流程（2026-10-03 签认）：上传先只做提示词校准（落盘冻结件），
    # 人工过目变更单后 POST /manuals/{id}/confirm-build 才真正建图。
    task_id = _task_manager.submit_calibrate(manual_id, md_path)
    return {"manual_id": manual_id, "task_id": task_id, "manual": m,
            "flow": "calibrate_first",
            "hint": "校准完成后请查看 /manuals/{id}/calibration 变更单，"
                    "确认后调用 POST /manuals/{id}/confirm-build 开始建图"}


@app.get("/manuals/{manual_id}/calibration")
def get_calibration(manual_id: str):
    """查看冻结校准件的变更单（C1 人工过目入口）。"""
    m = _registry.get_manual(manual_id)
    if m is None:
        raise HTTPException(status_code=404, detail=f"手册不存在: {manual_id}")
    from backend.prompt_calibrator import load_calibration
    md_path = parsed_md_path_for(manual_id)
    record = load_calibration(md_path, verify=False)
    if record is None:
        raise HTTPException(status_code=404,
                            detail=f"该手册还没有校准件（校准任务完成后可查）: {manual_id}")
    return record


@app.post("/manuals/{manual_id}/confirm-build")
def confirm_build(manual_id: str):
    """人工过目校准变更单后确认建图（C1 第二步；抽取用冻结校准件）。"""
    m = _registry.get_manual(manual_id)
    if m is None:
        raise HTTPException(status_code=404, detail=f"手册不存在: {manual_id}")
    from backend.prompt_calibrator import load_calibration
    md_path = parsed_md_path_for(manual_id)
    if load_calibration(md_path, verify=False) is None:
        raise HTTPException(status_code=400,
                            detail="还没有冻结校准件，无法确认构建（C1：先校准再过目再构建）")
    task_id = _task_manager.submit(manual_id, m["source_manual"], md_path,
                                   use_calibrated_prompt=True)
    return {"manual_id": manual_id, "task_id": task_id, "use_calibrated_prompt": True}


@app.post("/manuals/{manual_id}/activate")
def activate_manual(manual_id: str):
    m = _registry.get_manual(manual_id)
    if m is None:
        raise HTTPException(status_code=404, detail=f"手册不存在: {manual_id}")
    # 先构建引擎实例，成功后再切换 current，避免构建失败导致状态不一致
    global _service
    try:
        new_svc = _build_service(m)
    except Exception as e:
        raise HTTPException(status_code=500,
                            detail=f"手册构建引擎失败: {e}")
    _registry.set_current(manual_id)
    _service = new_svc
    resp = {"manual_id": manual_id, "current": True}
    # 空图警告：服务库（7690）里没有该手册的实体时，微观问答会退化为纯向量
    # 上下文、宏观无社区可召回。仍放行（向量检索可独立工作），但必须让运维看见。
    try:
        h = new_svc.health()
        probs = []
        if h.get("graph_entities", 0) == 0:
            probs.append("服务库中没有该手册的图谱实体（graph_entities=0），"
                         "微观问答将没有图谱路径，答案质量会明显下降")
        if h.get("communities", 0) == 0:
            probs.append("没有社区摘要缓存（宏观模式不可用；微观/对比不受影响）")
        if probs:
            resp["warning"] = f"已切换为「{manual_id}」，但：" + "；".join(probs) + "。"
            logger.warning("激活 %s 后存在缺失项: %s", manual_id, "; ".join(probs))
    except Exception:
        pass
    return resp


@app.delete("/manuals/{manual_id}")
def delete_manual(manual_id: str):
    m = _registry.get_manual(manual_id)
    if m is None:
        raise HTTPException(status_code=404, detail=f"手册不存在: {manual_id}")
    if m.get("prebuilt"):
        raise HTTPException(status_code=400, detail="预构建内置手册不可删除")
    if m.get("current"):
        raise HTTPException(status_code=400, detail="当前激活手册不可删除，请先切换")
    from backend.graph_store import GraphStore
    from config import NEO4J_BUILD_URI

    if not m.get("build"):
        # 空壳条目（status=uploaded、build 为空）：没有属于它的图谱与构建产物。
        # 按_manual_id 清产物会误删评测时代同名的派生缓存（2026-10-26 实测删掉
        # vector_index_SV660N），只删注册条目与上传原文。
        fp = m.get("file_path")
        if fp and os.path.exists(fp):
            try:
                os.remove(fp)
            except OSError:
                pass
        _registry.remove_manual(manual_id)
        return {"manual_id": manual_id, "deleted": True, "note": "空壳条目，仅移除注册记录"}

    from backend.build_pipeline import (
        _delete_manual_graph, extract_path_for, communities_cache_for,
        chunk_index_cache_for, parsed_md_path_for,
    )
    store = GraphStore(uri=NEO4J_BUILD_URI)
    try:
        _delete_manual_graph(store, m["source_manual"])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"删除图谱失败: {e}")
    finally:
        store.close()

    # 清理构建产物，避免删除后残留被后续同名手册误复用（交叉污染）
    import shutil
    from backend.build_pipeline import CACHE_DIR
    from backend.prompt_calibrator import calibration_dir_for
    parsed_md = parsed_md_path_for(manual_id)
    artifacts = [
        extract_path_for(manual_id),
        os.path.join(os.path.dirname(extract_path_for(manual_id)),
                     f"llm_extract_{manual_id}_provenance.json"),
        communities_cache_for(manual_id),
        chunk_index_cache_for(manual_id),
        os.path.join(CACHE_DIR, f"vector_index_{manual_id}"),
        parsed_md,
        calibration_dir_for(parsed_md),
    ]
    for p in artifacts:
        if os.path.isdir(p):
            try:
                shutil.rmtree(p)
            except OSError:
                pass
        elif os.path.exists(p):
            try:
                os.remove(p)
            except OSError:
                pass

    # 删除上传原文（不再被任何手册引用）
    fp = m.get("file_path")
    if fp and os.path.exists(fp):
        try:
            os.remove(fp)
        except OSError:
            pass

    # 清理 _processed 下的 .done 断点标记（失败不阻断删除主流程）
    try:
        import glob
        from config import PROJECT_ROOT
        processed_dir = os.path.join(PROJECT_ROOT, "data", "manuals", "_processed")
        for ckpt in glob.glob(os.path.join(processed_dir, f"{manual_id}.md_*.done")):
            os.remove(ckpt)
    except Exception:
        pass

    _registry.remove_manual(manual_id)
    return {"deleted": manual_id}


@app.get("/tasks")
def list_tasks():
    return _task_manager.list_tasks()


@app.get("/tasks/{task_id}")
def get_task(task_id: str):
    t = _task_manager.get_task(task_id)
    if t is None:
        raise HTTPException(status_code=404, detail=f"任务不存在: {task_id}")
    return t


# ===================== 历史会话（问答记录持久化） =====================
from backend.history_store import HistoryStore

_history_store = HistoryStore()


class CreateSessionRequest(BaseModel):
    title: str = Field("新对话", max_length=60)


class AppendMessageRequest(BaseModel):
    role: Literal["user", "assistant"] = Field(..., description="user / assistant")
    content: str = Field(..., description="消息文本")
    paths: list = Field([], description="KG 单行路径")
    path_detail: list = Field([], description="KG 逐跳结构化路径")
    refusal: bool = Field(False)
    refusal_detail: dict = Field({})
    n_contexts: int = Field(0)
    candidates: list = Field([], description="Vector 候选片段")
    chain_html: str = Field("", description="推理链可视化 HTML（历史回看时还原）")
    trace: Optional[dict] = Field(None, description="生成过程留痕（prompt 画像＋模型切分前原文）")


@app.post("/history/sessions")
def create_history_session(req: CreateSessionRequest):
    return _history_store.create_session(title=req.title)


@app.get("/history/sessions")
def list_history_sessions():
    return _history_store.list_sessions()


@app.get("/history/sessions/{session_id}")
def get_history_session(session_id: str):
    s = _history_store.get_session(session_id)
    if s is None:
        raise HTTPException(status_code=404, detail=f"会话不存在: {session_id}")
    return s


class RenameSessionRequest(BaseModel):
    title: str = Field(..., min_length=1, max_length=60)


@app.put("/history/sessions/{session_id}/title")
def rename_history_session(session_id: str, req: RenameSessionRequest):
    ok = _history_store.rename_session(session_id, req.title)
    if not ok:
        raise HTTPException(status_code=404, detail=f"会话不存在: {session_id}")
    return {"session_id": session_id, "title": req.title}


@app.delete("/history/sessions/{session_id}")
def delete_history_session(session_id: str):
    ok = _history_store.delete_session(session_id)
    if not ok:
        raise HTTPException(status_code=404, detail=f"会话不存在: {session_id}")
    return {"deleted": session_id}


@app.post("/history/sessions/{session_id}/messages")
def append_history_message(session_id: str, req: AppendMessageRequest):
    try:
        msg = _history_store.append_message(
            session_id, role=req.role, content=req.content,
            paths=req.paths, path_detail=req.path_detail,
            refusal=req.refusal, refusal_detail=req.refusal_detail,
            n_contexts=req.n_contexts, candidates=req.candidates,
            chain_html=req.chain_html,
            **({"trace": req.trace} if req.trace else {}),
        )
        return msg
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))


def _now() -> str:
    from datetime import datetime
    return datetime.now().isoformat(timespec="seconds")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("SERVING_PORT", "8001")))
