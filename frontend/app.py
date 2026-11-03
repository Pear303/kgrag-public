"""KG-RAG 工业故障推理引擎 Web 界面（Gradio）。

功能：
- Tab 1 问答：微观（图谱路径 + 答案生成，含 OOC 拒答）与宏观（社区总结），
  带历史对话记录（gr.Chatbot，会话内累积）
- Tab 2 对比：KG-RAG vs Vector RAG 并排，KG 展示逐跳多跳路径，
  Vector 展示候选片段列表（分数 + 文本可展开）
- Tab 3 手册管理：上传 → 自动构建 → 切换查询

数据源：评测隔离库 7690（shukongdashi_500cases.md，v5 全量 + 补抽），
经 serve.KGRAGService 提供（与评测链路完全一致的检索+答案逻辑）。

启动：
    python frontend/app.py          # 默认 7860 端口
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gradio as gr
import requests

# 问答统一走 8001 API：与 /health 报告的"当前手册"同一来源，
# 否则本进程自建的 KGRAGService 会和 API 进程各说各话（切手册后对话不生效）。
API = os.getenv("KGRAG_API_BASE") or (
    "http://localhost:%s" % os.getenv("SERVING_PORT", "8001"))


def _api_query(question: str, max_ctx: int = 12) -> dict:
    """微观查询（图谱推理 + 答案生成）。"""
    r = requests.post(f"{API}/query", json={"question": question, "max_ctx": max_ctx},
                      timeout=600)
    r.raise_for_status()
    return r.json()


def _api_stream_query(question: str, max_ctx: int = 12):
    """微观查询·流式：逐事件 yield dict（meta / reasoning / answer_delta / done / error）。

    服务端按 SSE（`data: {json}\\n\\n`）下发；这里按字节收再手动 UTF-8 解码，
    不用 requests 的 decode_unicode（它依赖响应头编码，中文易乱）。
    """
    with requests.post(f"{API}/query/stream",
                       json={"question": question, "max_ctx": max_ctx},
                       stream=True, timeout=600) as r:
        r.raise_for_status()
        for raw in r.iter_lines():
            if not raw:
                continue
            line = raw.decode("utf-8", "replace")
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if not payload:
                continue
            try:
                yield _json.loads(payload)
            except Exception:
                continue


def _api_query_macro(question: str, max_comm: int = 3) -> dict:
    """宏观查询（社区总结）。"""
    r = requests.post(f"{API}/query/macro", json={"question": question, "max_comm": max_comm},
                      timeout=600)
    r.raise_for_status()
    return r.json()


def _api_query_compare(question: str) -> dict:
    """对比查询（KG-RAG vs Vector RAG）。"""
    r = requests.post(f"{API}/query/compare", json={"question": question}, timeout=600)
    r.raise_for_status()
    return r.json()


def _current_manual_md() -> str:
    """从 /health 读当前手册（8001 进程激活的那本），失败时如实报未连上。"""
    try:
        h = requests.get(f"{API}/health", timeout=5).json()
        name = h.get("manual") or "（未知）"
        ent = h.get("graph_entities", "?")
        comm = h.get("communities", "?")
        return (f"📖 当前手册：**{name}**（{ent} 实体 / {comm} 社区；"
                f"来源：8001 API /health）")
    except Exception:
        return "⚠️ 无法连接 8001 API（问答不可用）——请先启动 api_server（端口 8001）"


# ============ 图谱路径渲染（逐跳展开） ============

def _render_path_detail(path_detail: list) -> str:
    """把逐跳结构化路径渲染成缩进文本，直观展示"哪步→哪步"。

    path_detail: [{"seq": [{"entity","rel","dir"},...], "text": "..."}, ...]
    每个 seq 的第 i 步（i>=1）含 rel（从上一实体到本实体的关系）。
    """
    if not path_detail:
        return "（无图谱路径命中）"
    lines = []
    for idx, p in enumerate(path_detail, start=1):
        seq = p.get("seq", [])
        lines.append(f"**路径 {idx}**：")
        for j, step in enumerate(seq):
            ent = step.get("entity", "")
            rel = step.get("rel")
            if j == 0:
                lines.append(f"  └ 起点：`{ent}`")
            else:
                arrow = "◀──" if step.get("dir") == "-" else "──▶"
                lines.append(f"  └ {arrow}[{rel}] → `{ent}`")
    return "\n".join(lines)


def _render_vector_candidates(candidates: list, max_chars: int = 200) -> str:
    """把 Vector 候选片段渲染成折叠列表：每候选显示 序号+分数+片段开头。

    candidates: [{"text": str, "score": float}, ...]
    文本超长截断并在末尾加省略标记（Gradio Markdown 无原生折叠，
    用 <details> 实现可展开/收起）。
    """
    if not candidates:
        return "（无候选片段）"
    parts = [f"共 {len(candidates)} 个候选片段（相似度从高到低）\n"]
    for idx, cand in enumerate(candidates, start=1):
        text = cand.get("text", "")
        score = cand.get("score", 0)
        preview = text[:max_chars]
        rest = text[max_chars:]
        # <details> 折叠：预览在前，全文在后
        parts.append(f"**候选 {idx}** | 相似度 `{score}`")
        parts.append(preview.replace("\n", " ") + ("…" if rest else ""))
        if rest:
            parts.append(
                f"<details><summary>展开全文（{len(text)}字）</summary>"
                f"\n\n{text}\n\n</details>"
            )
        parts.append("---")
    return "\n".join(parts)


# ============ 推理链可视化（cytoscape 内嵌） ============

import json as _json
import re as _re
import urllib.parse as _urllib

_CHAIN_REL_COLOR = {
    "故障原因": "#e74c3c", "可能原因": "#e74c3c",
    "处理方法": "#27ae60", "解决方法": "#27ae60",
    "确认方法": "#2980b9", "检查处理方法": "#2980b9",
    "报警含义": "#8e44ad", "报警类别": "#8e44ad",
    "表现为": "#f39c12", "包含": "#95a5a6", "配套使用": "#95a5a6",
    "推理链": "#34495e",
}
_CHAIN_DEFAULT_COLOR = "#7f8c8d"


def _build_chain_elements(path_detail: list) -> dict:
    """把 serve 的 path_detail 转成 cytoscape elements（nodes+edges）。

    - "[推理链]" 开头的项：解析为 主体 → 关系组 → 并列项 的树
    - 其余多跳路径：按 seq 逐跳连边
    """
    nodes, edges = [], []
    counter = [0]

    def _next(prefix):
        counter[0] += 1
        return f"{prefix}_{counter[0]}"

    for p in (path_detail or []):
        seq = p.get("seq", []) or []
        if not seq:
            continue
        head = str(seq[0].get("entity", ""))
        if head.startswith("[推理链]"):
            body = head[len("[推理链]"):].strip()
            segs = [s.strip() for s in body.split(";") if s.strip()]
            if not segs:
                continue
            subj = segs[0]
            subj_id = _next("subj")
            nodes.append({"data": {"id": subj_id, "label": subj[:60], "color": "#34495e",
                                   "size": 42, "rel": "主体", "text": body}})
            for seg in segs[1:]:
                if ":" not in seg:
                    continue
                rel_head, items_str = seg.split(":", 1)
                rel_head = rel_head.strip()
                m = _re.match(r"(.+?)\s*(?:\((\d+)\))?$", rel_head)
                rel_name = (m.group(1).strip() if m else rel_head)
                color = _CHAIN_REL_COLOR.get(rel_name, _CHAIN_DEFAULT_COLOR)
                rel_id = _next("rel")
                nodes.append({"data": {"id": rel_id, "label": rel_head, "color": color,
                                       "size": 32, "rel": rel_name, "text": ""}})
                edges.append({"data": {"source": subj_id, "target": rel_id,
                                       "label": "", "color": color}})
                for raw in items_str.split("|"):
                    item = raw.strip()
                    if not item:
                        continue
                    text = _re.sub(r"^\d+\.\s*", "", item)
                    it_id = _next("item")
                    nodes.append({"data": {"id": it_id, "label": text[:38], "color": color,
                                           "size": 26, "rel": rel_name, "text": text}})
                    edges.append({"data": {"source": rel_id, "target": it_id,
                                           "label": "", "color": color}})
        else:
            prev_id = None
            for step in seq:
                ent = str(step.get("entity", ""))
                rel = step.get("rel") or ""
                d = step.get("dir") or "?"
                cur_id = _next("n")
                nodes.append({"data": {"id": cur_id, "label": ent[:38],
                                       "color": _CHAIN_REL_COLOR.get(rel, _CHAIN_DEFAULT_COLOR),
                                       "size": 28, "rel": rel, "text": ent}})
                if prev_id is not None:
                    arrow = "◀ " if d == "-" else "▶ "
                    edges.append({"data": {"source": prev_id, "target": cur_id,
                                           "label": arrow + rel,
                                           "color": _CHAIN_REL_COLOR.get(rel, _CHAIN_DEFAULT_COLOR)}})
                prev_id = cur_id
    return {"nodes": nodes, "edges": edges}


def _render_chain_from_result(r: dict, max_paths: int = 10, height: int = 640) -> str:
    """根据查询结果构造推理链图 → 返回 iframe HTML。

    先把 elements 存到服务端（POST /chain/store）拿一个短 id，再让 iframe 用
    `/chain-view?id=...` 引用。旧做法是把整张图的 JSON urlencode 进 URL，图一大
    URL 就爆；暂存失败时退回旧 data= 路径，功能不丢。
    """
    path_detail = (r.get("path_detail") or [])[:max_paths]
    elements = _build_chain_elements(path_detail)
    if not elements["nodes"]:
        return ""
    src = f"{API}/chain-view"
    try:
        resp = requests.post(f"{API}/chain/store", json=elements, timeout=5)
        if resp.status_code == 200:
            cid = (resp.json() or {}).get("id")
            if cid:
                src = f"{API}/chain-view?id={cid}"
    except Exception:
        pass
    if src.endswith("/chain-view"):  # 暂存失败：退回旧路径
        data_enc = _urllib.quote(_json.dumps(elements, ensure_ascii=False), safe="")
        src = f"{API}/chain-view?data={data_enc}"
    n_nodes = len(elements["nodes"])
    n_edges = len(elements["edges"])
    return (
        f'<div style="font-size:12px;color:#666;margin-bottom:6px">'
        f'推理链图：共 {n_nodes} 个节点 / {n_edges} 条边（点击节点查看全文）</div>'
        f'<iframe src="{src}" '
        f'style="width:100%;height:{height}px;border:1px solid #e0e0e0;border-radius:8px;"></iframe>'
    )


def _render_chain_html(question: str, max_paths: int = 10) -> str:
    """查询 → 构造推理链图 → 返回内嵌 cytoscape 的 iframe HTML。"""
    if not question or not question.strip():
        return "<div style='padding:20px;color:#888'>请输入问题后点击「生成推理链」。</div>"
    try:
        svc_r = _api_query(question)
    except Exception as e:
        return f"<div style='padding:20px;color:#c00'>查询失败: {e}</div>"
    html = _render_chain_from_result(svc_r, max_paths=max_paths)
    if not html:
        return "<div style='padding:20px;color:#888'>未检索到可展示的图谱路径。</div>"
    return html


# ============ Tab1：问答（会话式历史，后端持久化） ============

def _refusal_why(rd: dict) -> str:
    """拒答有三道不同的闸门，成因不一样，话术必须分开。

    把三道合并成一句"内容不在知识库里"是对用户说谎：2026-10-17 实测厂商门拦的 5 道题，
    答案逐字在被喂的手册里，图谱却没抽出那个型号词条（见
    files/_oneoff/vendorgate/_vendorgate_forensics_20261017.py）。
    """
    rd = rd or {}
    if rd.get("vendor_gate"):
        return rd.get("reason") or "图谱里未收录问题提到的厂商或型号词条"
    if rd.get("sent_cov") is not None:
        return "问题与这本手册的领域语义相似度偏低（相似度 %s）" % rd.get("sent_cov")
    if "relevance" in rd or "object_mismatch" in rd:
        return ("检索到的内容与问题对不上（相关度 %s，对象不一致 %s）"
                % (rd.get("relevance"), rd.get("object_mismatch")))
    return "未检索到可作答的内容"


def _assistant_content(r, mode: str, refusal: bool) -> tuple:
    """根据查询结果拼 assistant 展示文本，返回 (text, detail)。

    detail 是存后端的结构化字段（KG路径/Vector候选/拒答详情），回看时据此还原。
    """
    if mode.startswith("微观"):
        text = r["answer"]
        detail = {
            "mode": mode,
            "paths": r.get("paths", []),
            "path_detail": r.get("path_detail", []),
            "refusal": refusal,
            "refusal_detail": r.get("refusal_detail"),
            "n_contexts": r.get("n_contexts", 0),
        }
        parts = [text]
        if refusal:
            rd = r.get("refusal_detail") or {}
            parts.append(f"\n\n[系统判定：{_refusal_why(rd)}，已拒答]")
        else:
            detail_txt = _render_path_detail(r.get("path_detail", []))
            if detail_txt and "无图谱路径" not in detail_txt:
                parts.append("\n\n**知识图谱多跳路径**\n\n" + detail_txt)
        return "\n".join(parts), detail
    elif mode.startswith("对比"):
        kg = r.get("kg", {})
        vec = r.get("vector", {})
        detail = {
            "mode": mode,
            "kg_paths": kg.get("paths", []),
            "kg_path_detail": kg.get("path_detail", []),
            "refusal": kg.get("refusal", False),
            "kg_n_contexts": kg.get("n_contexts", 0),
            "vec_candidates": vec.get("candidates", []),
            "vec_answer": vec.get("answer", ""),
            "vec_n_contexts": vec.get("n_contexts", 0),
        }
        parts = []
        parts.append("### 知识图谱（KG-RAG）答案\n" + kg.get("answer", ""))
        if kg.get("refusal"):
            parts.append("[系统判定：%s，已拒答]" % _refusal_why(kg.get("refusal_detail")))
        pd = _render_path_detail(kg.get("path_detail", []))
        if pd and "无图谱路径" not in pd:
            parts.append("**知识图谱多跳路径**\n\n" + pd)
        parts.append("### 向量检索（Vector RAG）答案\n" + vec.get("answer", ""))
        vc = _render_vector_candidates(vec.get("candidates", []))
        if "无候选" not in vc:
            parts.append("**Vector 候选片段**\n\n" + vc)
        return "\n\n".join(parts), detail
    else:  # 宏观
        text = r["answer"]
        detail = {"mode": mode,
                  "n_communities": len(r.get("communities", []))}
        if r["communities"]:
            text += f"\n\n【命中社区】{len(r['communities'])} 个"
        return text, detail


def answer_cur(question: str, mode: str, session_id: str, msgs: list):
    """问答 → 追加到当前会话（写后端历史）→ 返回 (最新消息, 会话id, 空输入, 推理链HTML)。

    首次无会话时自动新建；问题保存 user 消息、结果保存 assistant(带detail)。
    后端持久化保证浏览器刷新/服务重启后历史仍在。
    """
    if not question.strip():
        return msgs, session_id, "", ""
    chain_html = ""
    if mode.startswith("对比"):
        r = _api_query_compare(question)
        refusal = r.get("kg", {}).get("refusal", False)
        chain_html = _render_chain_from_result(r.get("kg", {}), height=420)
    elif mode.startswith("微观"):
        r = _api_query(question)
        refusal = r["refusal"]
        chain_html = _render_chain_from_result(r, height=520)
    else:  # 宏观
        r = _api_query_macro(question)
        refusal = False
    full, detail = _assistant_content(r, mode, refusal)

    # 确保有会话
    if not session_id:
        sess = _api_new_session()
        session_id = sess["session_id"] if sess else None
    if session_id:
        _api_append_msg(session_id, "user", question)
        detail["chain_html"] = chain_html
        _api_append_msg(session_id, "assistant", full, **detail)

    # 追加到当前显示列表
    msgs = list(msgs) + [
        {"role": "user", "content": question},
        {"role": "assistant", "content": full},
    ]
    return msgs, session_id, "", chain_html


def _snapshot(msgs: list) -> list:
    """Chatbot 消息列表的浅拷贝。

    流式时反复 yield 同一个 list 对象，Gradio 可能按值比较判定"没变化"而跳过刷新；
    每次给一个新对象最稳。
    """
    return [dict(m) for m in msgs]


def _send_stream(q, mode, sid, msgs):
    """发送（流式）：微观走 /query/stream 边生成边显示；宏观/对比沿用原实现。

    生成器 ⇒ Gradio 逐步刷新。事件序：meta（拒答判定 + 图谱路径）→ reasoning
    （推理进度）→ answer_delta（逐段答案）→ done；结束后写后端历史。
    """
    q = (q or "").strip()
    banner = _current_manual_md()
    base = list(msgs or [])
    if not q:
        yield base, sid, "", "", gr.update(), banner
        return

    # 宏观 / 对比：后端暂无流式端点，沿用原一次性实现（后续可另开流式）
    if not mode.startswith("微观"):
        yield base + [{"role": "user", "content": q}], sid, "", "", gr.update(), banner
        out, sid_out, q_out, chain_out = answer_cur(q, mode, sid, base)
        choices = (gr.update(choices=_session_choices(), value=sid_out)
                   if (not sid and sid_out) else gr.update())
        yield out, sid_out, q_out, chain_out, choices, _current_manual_md()
        return

    # 微观：流式（若后端尚未提供 /query/stream，退回非流式，避免问答直接不可用）
    msgs = base + [
        {"role": "user", "content": q},
        {"role": "assistant", "content": "⏳ 正在检索图谱并推理…"},
    ]
    yield _snapshot(msgs), sid, "", "", gr.update(), banner

    from itertools import chain as _chain
    try:
        ev_iter = _api_stream_query(q)
        first_ev = next(ev_iter, None)
    except Exception:
        ev_iter, first_ev = None, None

    if ev_iter is None:  # 旧后端：退回一次性查询
        out, sid_out, q_out, chain_out = answer_cur(q, mode, sid, base)
        choices = (gr.update(choices=_session_choices(), value=sid_out)
                   if (not sid and sid_out) else gr.update())
        yield out, sid_out, q_out, chain_out, choices, _current_manual_md()
        return

    meta, answer, chain_html = None, "", ""
    try:
        for ev in _chain([first_ev] if first_ev else [], ev_iter):
            t = ev.get("type")
            if t == "meta":
                meta = ev
                msgs[-1]["content"] = "⏳ 已命中图谱路径，正在生成答案…"
                yield _snapshot(msgs), sid, "", chain_html, gr.update(), banner
            elif t == "reasoning":
                msgs[-1]["content"] = f"⏳ 正在推理…（{ev.get('chars', 0)} 字）"
                yield _snapshot(msgs), sid, "", chain_html, gr.update(), banner
            elif t == "answer_delta":
                answer += ev.get("text", "")
                msgs[-1]["content"] = answer
                yield _snapshot(msgs), sid, "", chain_html, gr.update(), banner
            elif t == "error":
                msgs[-1]["content"] = f"❌ {ev.get('message', '查询失败')}"
                yield _snapshot(msgs), sid, "", chain_html, gr.update(), banner
                return
            elif t == "done":
                answer = ev.get("answer") or answer
                meta = ev
    except Exception as e:
        msgs[-1]["content"] = f"❌ 查询失败：{e}"
        yield _snapshot(msgs), sid, "", chain_html, gr.update(), banner
        return

    meta = meta or {}
    r = {
        "answer": answer,
        "refusal": bool(meta.get("refusal")),
        "refusal_detail": meta.get("refusal_detail"),
        "n_contexts": meta.get("n_contexts", 0),
        "paths": meta.get("paths", []),
        "path_detail": meta.get("path_detail", []),
    }
    full, detail = _assistant_content(r, mode, r["refusal"])
    chain_html = _render_chain_from_result(r, height=520)
    msgs[-1]["content"] = full
    yield _snapshot(msgs), sid, "", chain_html, gr.update(), banner

    # 落后端历史（刷新/重启后仍在）
    if not sid:
        sess = _api_new_session()
        sid = sess["session_id"] if sess else None
    if sid:
        _api_append_msg(sid, "user", q)
        detail["chain_html"] = chain_html
        _api_append_msg(sid, "assistant", full, **detail)
    choices = gr.update(choices=_session_choices(), value=sid) if sid else gr.update()
    yield _snapshot(msgs), sid, "", chain_html, choices, banner


# ============ 手册管理辅助 ============

# ---- 历史会话 HTTP 辅助（后端持久化） ----

def _api_new_session(title: str = "新对话"):
    try:
        r = requests.post(f"{API}/history/sessions",
                          json={"title": title}, timeout=5)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return None


def _api_list_sessions():
    try:
        data = requests.get(f"{API}/history/sessions", timeout=5).json()
        return data if isinstance(data, list) else []
    except Exception:
        return []


def _api_get_session(session_id: str):
    try:
        r = requests.get(f"{API}/history/sessions/{session_id}", timeout=5)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return None


def _api_delete_session(session_id: str):
    try:
        requests.delete(f"{API}/history/sessions/{session_id}", timeout=5)
    except Exception:
        pass


def _api_rename_session(session_id: str, title: str):
    try:
        requests.put(f"{API}/history/sessions/{session_id}/title",
                     json={"title": title}, timeout=5)
    except Exception:
        pass


def _api_append_msg(session_id: str, role: str, content: str, **detail):
    """把一条消息写入后端历史。detail 为结构化字段（paths/path_detail 等）。"""
    payload = {
        "role": role,
        "content": content,
        "paths": detail.get("paths", []) or detail.get("kg_paths", []),
        "path_detail": (detail.get("path_detail", [])
                        or detail.get("kg_path_detail", [])),
        "refusal": bool(detail.get("refusal", False)),
        "refusal_detail": detail.get("refusal_detail") or {},
        "n_contexts": int(detail.get("n_contexts", 0)
                          or detail.get("kg_n_contexts", 0)),
        "candidates": detail.get("vec_candidates", [])
        or detail.get("candidates", []),
        "chain_html": detail.get("chain_html", "") or "",
    }
    try:
        requests.post(f"{API}/history/sessions/{session_id}/messages",
                      json=payload, timeout=5)
    except Exception:
        pass


def _session_choices():
    """历史会话 → Dropdown choices [(label, session_id)]。"""
    sessions = _api_list_sessions()
    if not sessions:
        return []
    choices = []
    for s in sessions:
        label = (s.get("title") or s.get("session_id") or "")[:24]
        n = s.get("n_messages", 0)
        ts = (s.get("updated_at") or "")[5:16]  # MM-DD HH:MM
        choices.append((f"{label}（{n}条 {ts}）", s["session_id"]))
    return choices


def _new_conversation_with_choices() -> tuple:
    """新建会话：返回 (空消息, 新session_id, 空消息状态, 空推理链HTML, 最新会话列表)。"""
    sess = _api_new_session()
    sid = sess["session_id"] if sess else None
    if sid:
        return [], sid, [], "", gr.update(choices=_session_choices(), value=sid)
    return [], sid, [], "", gr.update(choices=_session_choices())


def _rename_current(current_sid: str, new_title: str, chatbot_msgs: list) -> tuple:
    """重命名当前会话：更新后端标题，返回刷新后的会话列表 + 提示状态。"""
    if not current_sid:
        return gr.update(), "⚠️ 请先创建或选中一个会话"
    if not new_title or not new_title.strip():
        return gr.update(), "⚠️ 标题不能为空"
    _api_rename_session(current_sid, new_title.strip())
    return gr.update(choices=_session_choices(), value=current_sid), \
        f"✅ 已重命名为「{new_title.strip()}」"


def _delete_current(current_sid: str, chatbot_msgs: list, choices: list) -> tuple:
    """删除当前会话，并自动切换到下一个会话（避免误删后留在空白/错误状态）。"""
    if not current_sid:
        return [], None, [], "", gr.update(choices=_session_choices()), \
            "⚠️ 没有可删除的会话"
    # 收集删除前的顺序，便于删除后自动切到相邻会话
    before = [c[1] for c in _session_choices()]
    _api_delete_session(current_sid)
    after = [c[1] for c in _session_choices()]
    if current_sid in before:
        pos = before.index(current_sid)
    else:
        pos = 0
    target = None
    # 优先选当前会话后面那位，没有了就选前一位，至少保证别空白
    if after:
        if pos < len(after):
            target = after[pos]
        else:
            target = after[-1]
    if target:
        msgs, sid, chain_html = _load_history_into_chatbot(target)
        return msgs, sid, msgs, "", \
            gr.update(choices=_session_choices(), value=target), \
            f"🗑️ 已删除，已切换到「{target}」"
    return [], None, [], "", gr.update(choices=_session_choices()), \
        "🗑️ 已删除最后一个会话"


def _load_history_into_chatbot(session_id: str) -> tuple:
    """回看某会话：把它渲染成 Chatbot 消息，返回 (msgs, sid, chain_html)。

    chain_html 取最后一条 assistant 消息存下的推理链图，还原到显示区。
    兼容多种可能的 key 名。
    """
    if not session_id:
        return [], session_id, ""
    sess = _api_get_session(session_id)
    if sess is None:
        return [], session_id, ""
    msgs = []
    last_chain = ""
    for m in sess.get("messages", []):
        if m.get("role") == "assistant":
            d = m.get("detail") or {}
            # 兼容多种可能的 key 名
            chain_val = (d.get("chain_html")
                         or d.get("chain_viz")
                         or d.get("recommend_chain")
                         or "")
            if chain_val:
                last_chain = chain_val
            content = m["content"]
            # 若存了结构化 detail，附加完整路径/候选展示（未存则原样显示）
            parts = [content]
            if d.get("path_detail"):
                pd = _render_path_detail(d.get("path_detail"))
                if pd and "无图谱路径" not in pd:
                    parts.append("\n\n**知识图谱多跳路径**\n\n" + pd)
            elif d.get("kg_path_detail"):
                pd = _render_path_detail(d.get("kg_path_detail"))
                if pd and "无图谱路径" not in pd:
                    parts.append("\n\n**知识图谱多跳路径**\n\n" + pd)
            if d.get("candidates"):
                vc = _render_vector_candidates(d.get("candidates"))
                if "无候选" not in vc:
                    parts.append("**向量检索候选片段**\n\n" + vc)
            msgs.append({"role": "assistant", "content": "\n\n".join(parts)})
        else:
            msgs.append({"role": m.get("role", "user"),
                         "content": m.get("content", "")})
    return msgs, session_id, last_chain


def _new_conversation() -> tuple:
    """新建会话：返回 (空消息, 新session_id, 空消息状态, 空推理链HTML, 最新会话列表)。"""
    return _new_conversation_with_choices()


def _refresh_session_choices() -> list:
    return _session_choices()


def _api_list_manuals():
    try:
        data = requests.get(f"{API}/manuals", timeout=5).json()
        return data if isinstance(data, list) else []
    except Exception:
        return []


def _api_list_tasks():
    try:
        data = requests.get(f"{API}/tasks", timeout=5).json()
        return data if isinstance(data, list) else []
    except Exception:
        return []


def _api_upload(fileobj):
    """上传手册（C1 第一步：只提交提示词校准任务，不直接建图）。"""
    if fileobj is None:
        return "请先选择文件", ""
    with open(fileobj.name, "rb") as f:
        files = {"file": (os.path.basename(fileobj.name), f)}
        try:
            r = requests.post(f"{API}/manuals", files=files, timeout=10)
            if r.status_code == 200:
                j = r.json()
                mid = j.get("manual_id", "")
                return (f"上传成功，校准任务 {j['task_id']} 已入队。\n"
                        f"手册ID: {mid}\n"
                        f"校准完成后点【查看校准结果】过目变更单，认可后点【确认并开始构建】。", mid)
            return f"上传失败: {r.text}", ""
        except Exception as e:
            return f"上传异常: {e}", ""


def _render_calibration(mid):
    """拉取并渲染冻结校准件的变更单（C1 人工过目）。"""
    if not mid:
        return "（先上传手册）"
    try:
        r = requests.get(f"{API}/manuals/{mid}/calibration", timeout=10)
        if r.status_code != 200:
            return f"暂无校准件: {r.text}"
        rec = r.json()
        lines = [f"采样模式: {rec.get('mode')}（手册 {rec.get('manual_chars')} 字 → 送入 "
                 f"{rec.get('selected_chars')} 字 / {rec.get('n_windows')} 窗）",
                 f"适配版提示词 sha8: {rec.get('adapted_prompt_sha8')}",
                 f"变更 {len(rec.get('updates', {}))} 处:", ""]
        for ch in rec.get("changes", []):
            lines.append(f"**[{ch.get('slot')}]** {ch.get('reason')}")
        updates = rec.get("updates", {})
        if updates:
            lines.append("")
            lines.append("<details><summary>新槽值全文</summary>")
            lines.append("")
            lines.append("```")
            for k, v in updates.items():
                lines.append(f"### {k}")
                lines.append(v)
            lines.append("```")
            lines.append("</details>")
        return "\n".join(lines)
    except Exception as e:
        return f"查询异常: {e}"


def _api_confirm_build(mid):
    """C1 第二步：人工过目后确认建图（抽取用冻结校准件）。"""
    if not mid:
        return "（先上传手册并完成校准）"
    try:
        r = requests.post(f"{API}/manuals/{mid}/confirm-build", timeout=10)
        if r.status_code == 200:
            j = r.json()
            return f"已确认，构建任务 {j['task_id']} 已入队（抽取使用校准适配版提示词）。"
        return f"确认失败: {r.text}"
    except Exception as e:
        return f"确认异常: {e}"


def _api_activate(manual_id):
    try:
        r = requests.post(f"{API}/manuals/{manual_id}/activate", timeout=60)
        if r.status_code == 200:
            return f"已切换到 {manual_id}"
        return f"切换失败: {r.text}"
    except Exception as e:
        return f"切换异常: {e}"


def _api_delete(manual_id):
    try:
        r = requests.delete(f"{API}/manuals/{manual_id}", timeout=30)
        if r.status_code == 200:
            return f"已删除 {manual_id}"
        return f"删除失败: {r.text}"
    except Exception as e:
        return f"删除异常: {e}"


def _render_manuals():
    manuals = _api_list_manuals()
    if not manuals:
        return "（暂无手册）"
    lines = ["| 手册ID | 状态 | 三元组 | 社区 | 当前 |",
             "|---|---|---|---|---|"]
    for m in manuals:
        b = m.get("build") or {}
        cur = "★" if m.get("current") else ""
        lines.append(f"| {m['manual_id']} | {m['status']} | "
                     f"{b.get('triples', '-')} | {b.get('communities', '-')} | {cur} |")
    return "\n".join(lines)


def _render_tasks():
    tasks = _api_list_tasks()
    if not tasks:
        return "（暂无任务）"
    lines = ["| 任务ID | 类型 | 状态 | 步骤 | 进度 |",
             "|---|---|---|---|---|"]
    for t in tasks:
        kind = {"calibrate": "校准", "build": "构建"}.get(t.get("kind"), "构建")
        lines.append(f"| {t['task_id']} | {kind} | {t['status']} | {t.get('step', 0)} | "
                     f"{t.get('pct', 0)}% |")
    return "\n".join(lines)


# ============ UI ============

_CSS = """
/* ---- 全局：禁止水平溢出 ---- */
.gradio-container { overflow-x: hidden !important; }
#sidebar-panel {
    border-right: 1px solid #eee;
    padding: 0 8px 0 0 !important;
    min-height: 0 !important;
    max-height: none !important;
    overflow-y: auto !important;
    overflow-x: hidden !important;
}
#sidebar-panel .md h3 { margin-top: 2px; font-size: 14px; }
#main-panel { padding-left: 6px; }

/* ---- 会话列表：Codex / DeepSeek 风格竖排 ---- */
#session-list {
    gap: 2px !important;
    overflow: visible !important;
}
#session-list .wrap {
    width: 100% !important;
    overflow: visible !important;
}
#session-list label.item {
    display: block !important;
    width: 100% !important;
    margin: 0 !important;
    padding: 6px 10px !important;
    border-radius: 6px !important;
    cursor: pointer !important;
    border: 1px solid transparent !important;
    transition: background 0.15s ease;
    font-size: 13px !important;
    line-height: 1.4 !important;
    white-space: normal !important;
    overflow: hidden !important;
    text-overflow: ellipsis !important;
}
#session-list label.item:hover {
    background: rgba(0, 0, 0, 0.04) !important;
}
#session-list label.item.selected,
#session-list label.item:has(input:checked) {
    background: rgba(59, 130, 246, 0.12) !important;
    border-color: rgba(59, 130, 246, 0.35) !important;
    font-weight: 600 !important;
}
#session-list input[type="radio"] {
    display: none !important;
    width: 0 !important;
    min-width: 0 !important;
}

/* ---- 侧边栏内所有子组件不要撑宽 ---- */
#sidebar-panel > *,
#sidebar-panel .form,
#sidebar-panel .textbox,
#sidebar-panel .radio-group {
    min-width: 0 !important;
    max-width: 100% !important;
    overflow: hidden !important;
}

/* ---- Row 不溢出 ---- */
.gradio-row { min-width: 0 !important; overflow: hidden !important; }

/* ---- 查询模式下拉框：文字与箭头不要重叠 ---- */
#main-panel .dropdown, #main-panel .wrap, #main-panel .input {
    min-width: 0 !important;
}
#main-panel .dropdown span {
    max-width: 80% !important;
    overflow: hidden !important;
    text-overflow: ellipsis !important;
    white-space: nowrap !important;
    padding-right: 20px !important;
}
"""

with gr.Blocks(title="KGRAG 工业故障推理引擎", fill_width=True) as demo:
    gr.Markdown("# KGRAG 工业故障推理引擎")

    with gr.Tabs():
        with gr.TabItem("问答"):
            with gr.Row(equal_height=False):
                # ============ 左侧会话侧边栏 ============
                with gr.Column(scale=1, min_width=180, elem_id="sidebar-panel"):
                    gr.Markdown("### 💬 会话列表")
                    with gr.Row():
                        new_btn = gr.Button("＋ 新建", variant="primary", size="sm", scale=1)
                        refresh_btn = gr.Button("↻ 刷新", variant="secondary", size="sm", scale=1)
                    session_list = gr.Radio(
                        label="💬 会话列表",
                        choices=[], interactive=True,
                        container=False, min_width=0,
                        elem_id="session-list",
                    )
                    rename_input = gr.Textbox(
                        label="重命名标题", lines=1,
                        placeholder="输入新标题后点「确认重命名」",
                    )
                    with gr.Row():
                        rename_btn = gr.Button("重命名", variant="secondary", size="sm", scale=1)
                        delete_btn = gr.Button("删除会话", variant="stop", size="sm", scale=1)
                    rename_status = gr.Markdown(
                        "支持：点击会话直接进入；重命名/删除当前会话。",
                        elem_classes="hint",
                    )
                    session_state = gr.State(None)   # 当前 session_id
                    msgs_state = gr.State([])        # 当前会话的消息列表

                # ============ 右侧聊天主区 ============
                with gr.Column(scale=3, min_width=0, elem_id="main-panel"):
                    manual_banner = gr.Markdown(_current_manual_md())
                    chatbot = gr.Chatbot(
                        label="对话（当前会话）",
                        height=520,
                    )
                    chain_viz = gr.HTML(
                        "<div style='padding:16px;color:#888;border:1px dashed #ddd;"
                        "border-radius:8px;'>"
                        "提问后，这里将显示回答内嵌的 KG 推理链图"
                        "（可点击节点查看全文）。</div>"
                    )
                    question = gr.Textbox(
                        label="问题",
                        placeholder="如：三菱M64系统中，若使用变频器作为主轴驱动系统，参数P145应设定为什么值？",
                        lines=2,
                    )
                    with gr.Accordion("高级：查询模式（默认 微观·图谱推理）", open=False):
                        mode = gr.Dropdown(
                            choices=["微观(图谱推理)", "宏观(社区总结)", "对比(知识图谱 vs 向量检索)"],
                            value="微观(图谱推理)", label="查询模式",
                            interactive=True,
                        )
                    btn = gr.Button("发送", variant="primary")

            # ============ 事件绑定 ============

            btn.click(_send_stream, inputs=[question, mode, session_state, msgs_state],
                      outputs=[chatbot, session_state, question, chain_viz, session_list,
                               manual_banner])
            question.submit(_send_stream, inputs=[question, mode, session_state, msgs_state],
                            outputs=[chatbot, session_state, question, chain_viz, session_list,
                                     manual_banner])

            # --- 点选会话 → 直接加载（不再需要点「加载」按钮）---
            def _on_session_select(choice):
                if not choice:
                    return gr.update(), None, [], "", gr.update()
                msgs, sid, chain_html = _load_history_into_chatbot(choice)
                return msgs, sid, msgs, chain_html, gr.update()

            session_list.change(
                _on_session_select,
                inputs=[session_list],
                outputs=[chatbot, session_state, msgs_state, chain_viz, rename_status],
            )

            # --- 新建会话（自动出现在列表并选中）---
            new_btn.click(
                _new_conversation,
                outputs=[chatbot, session_state, msgs_state, chain_viz, session_list],
            )

            # --- 刷新会话列表 ---
            refresh_btn.click(
                lambda: gr.update(choices=_session_choices()),
                outputs=[session_list],
            )

            # --- 重命名 ---
            rename_btn.click(
                _rename_current,
                inputs=[session_state, rename_input, msgs_state],
                outputs=[session_list, rename_status],
            )

            # --- 删除会话（自动切到相邻会话，不留在空白页）---
            delete_btn.click(
                _delete_current,
                inputs=[session_state, msgs_state, session_list],
                outputs=[chatbot, session_state, msgs_state, chain_viz,
                         session_list, rename_status],
            )

            # --- 启动时刷新会话列表与当前手册横幅 ---
            demo.load(
                lambda: [gr.update(choices=_session_choices()), _current_manual_md()],
                outputs=[session_list, manual_banner],
            )

        with gr.TabItem("手册管理"):
            gr.Markdown("## 手册管理（上传 → 提示词校准 → 人工过目 → 确认构建 → 切换查询）")
            with gr.Row():
                upload_file = gr.File(label="上传手册", file_types=[".md", ".txt", ".pdf", ".docx"])
                upload_btn = gr.Button("上传并校准提示词", variant="primary")
            upload_state = gr.State("")  # 最近一次上传的 manual_id
            upload_status = gr.Textbox(label="上传状态", lines=3, interactive=False)
            upload_btn.click(_api_upload, inputs=[upload_file],
                             outputs=[upload_status, upload_state])

            gr.Markdown("### 校准变更单（C1：须人工过目）")
            with gr.Row():
                view_calib_btn = gr.Button("查看校准结果")
                confirm_build_btn = gr.Button("确认并开始构建", variant="primary")
            calibration_md = gr.Markdown("（暂无校准结果）")
            view_calib_btn.click(_render_calibration, inputs=[upload_state],
                                 outputs=[calibration_md])
            confirm_build_btn.click(_api_confirm_build, inputs=[upload_state],
                                    outputs=[upload_status])

            gr.Markdown("### 任务进度")
            task_table = gr.Markdown(_render_tasks())
            refresh_tasks_btn = gr.Button("刷新任务")
            refresh_tasks_btn.click(lambda: _render_tasks(), outputs=[task_table])

            gr.Markdown("### 手册列表")
            manual_table = gr.Markdown(_render_manuals())
            refresh_manuals_btn = gr.Button("刷新手册")
            refresh_manuals_btn.click(lambda: _render_manuals(), outputs=[manual_table])

            with gr.Row():
                manual_id_box = gr.Textbox(label="手册ID（用于切换/删除）", lines=1)
                activate_btn = gr.Button("设为当前")
                delete_btn = gr.Button("删除", variant="stop")
            op_status = gr.Textbox(label="操作结果", lines=2, interactive=False)
            activate_btn.click(
                lambda mid: (_api_activate(mid), _current_manual_md()),
                inputs=[manual_id_box],
                outputs=[op_status, manual_banner],
            )
            delete_btn.click(_api_delete, inputs=[manual_id_box], outputs=[op_status])


if __name__ == "__main__":
    # 生成器流式必须有队列；并发上限防止一次长生成把界面锁死
    demo.queue(default_concurrency_limit=8)
    demo.launch(
        server_port=int(os.getenv("UI_PORT", "7860")),
        css=_CSS,
        theme=gr.themes.Soft(),
    )
