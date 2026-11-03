// 统一接口层：同源相对路径。
// 开发时由 vite.config.js 的 proxy 转发到 8001；生产由 FastAPI 同源提供。
const BASE = ''

async function req(path, { method = 'GET', body, params, _retried } = {}) {
  let url = BASE + path
  if (params) {
    const q = new URLSearchParams(params).toString()
    if (q) url += (url.includes('?') ? '&' : '?') + q
  }
  const headers = {}
  if (body) headers['Content-Type'] = 'application/json'
  const token = localStorage.getItem('api_token')
  if (token) headers['X-API-Token'] = token
  const res = await fetch(url, { method, headers, body: body ? JSON.stringify(body) : undefined })
  if (res.status === 401 && !_retried) {
    // 可选令牌鉴权开启时的补票：输入一次，存 localStorage，重放本次请求
    const t = window.prompt('本服务已开启 API 令牌校验，请输入 X-API-Token：')
    if (t) {
      localStorage.setItem('api_token', t)
      return req(path, { method, body, params, _retried: true })
    }
  }
  if (!res.ok) {
    let detail = `HTTP ${res.status}`
    try {
      const j = await res.json()
      // FastAPI 422 的 detail 是校验错误数组，逐条取 msg，别让 UI 显示 [object Object]
      if (Array.isArray(j.detail)) {
        detail = j.detail.map(d => (d.loc ? d.loc.join('.') + ': ' : '') + (d.msg || '')).join('；') || detail
      } else {
        detail = j.detail || j.error || detail
      }
    } catch (e) { /* 非 JSON 响应，保留状态码 */ }
    throw new Error(detail)
  }
  const ct = res.headers.get('content-type') || ''
  return ct.includes('application/json') ? res.json() : res.text()
}

export const api = {
  health: () => req('/health'),
  // 轻量健康：只读手册名/环境，不触发后端构建 KGRAGService（构建会走 Ollama）。
  // 页面加载/横幅用它；/health 留给需要完整服务状态的场合。
  healthLite: () => req('/health/lite'),
  // 只读系统自检：依赖/.env 键/Neo4j/Ollama/缓存/前端/端口/工具
  systemCheck: () => req('/system/check'),

  // ---- 本地模型设置 ----
  modelsLocal: (refresh = false) =>
    req('/models/local' + (refresh ? '?refresh=true' : '')),
  modelsStatus: (refresh = false) =>
    req('/models/status' + (refresh ? '?refresh=true' : '')),
  switchEmbedding: (model) =>
    req('/models/embedding', { method: 'POST', body: { model } }),
  switchAnswer: (model) =>
    req('/models/answer', { method: 'POST', body: { model } }),
  resetModels: () => req('/models/reset', { method: 'POST' }),

  query: (question, max_ctx = 12) => req('/query', { method: 'POST', body: { question, max_ctx } }),
  queryMacro: (question, max_comm = 3) =>
    req('/query/macro', { method: 'POST', body: { question, max_comm } }),
  queryCompare: (question) => req('/query/compare', { method: 'POST', body: { question } }),

  // ---- 历史会话（后端持久化） ----
  listSessions: () => req('/history/sessions'),
  createSession: (title = '新对话') => req('/history/sessions', { method: 'POST', body: { title } }),
  getSession: (id) => req(`/history/sessions/${id}`),
  renameSession: (id, title) =>
    req(`/history/sessions/${id}/title`, { method: 'PUT', body: { title } }),
  deleteSession: (id) => req(`/history/sessions/${id}`, { method: 'DELETE' }),
  appendMessage: (id, msg) =>
    req(`/history/sessions/${id}/messages`, { method: 'POST', body: msg }),

  // ---- 手册管理 ----
  listManuals: () => req('/manuals'),
  listTasks: () => req('/tasks'),
  getCalibration: (id) => req(`/manuals/${id}/calibration`),
  confirmBuild: (id) => req(`/manuals/${id}/confirm-build`, { method: 'POST' }),
  activateManual: (id) => req(`/manuals/${id}/activate`, { method: 'POST' }),
  deleteManual: (id) => req(`/manuals/${id}`, { method: 'DELETE' }),
  async uploadManual(file) {
    const fd = new FormData()
    fd.append('file', file)
    const res = await fetch(BASE + '/manuals', { method: 'POST', body: fd })
    if (!res.ok) {
      let d = `HTTP ${res.status}`
      try { d = (await res.json()).detail || d } catch (e) { /* keep */ }
      throw new Error(d)
    }
    return res.json()
  },
}

/**
 * 流式微观问答：POST /query/stream（SSE）。
 * EventSource 只能 GET，所以这里用 fetch + ReadableStream 手解 SSE 帧。
 * onEvent(ev) 会依次收到 meta / reasoning / answer_delta / done / error。
 */
export async function streamQuery(question, { maxCtx = 12, onEvent, signal } = {}) {
  const sHeaders = { 'Content-Type': 'application/json' }
  const sToken = localStorage.getItem('api_token')
  if (sToken) sHeaders['X-API-Token'] = sToken
  const res = await fetch(`${BASE}/query/stream`, {
    method: 'POST',
    headers: sHeaders,
    body: JSON.stringify({ question, max_ctx: maxCtx }),
    signal,
  })
  if (!res.ok || !res.body) throw new Error(`HTTP ${res.status}`)
  const reader = res.body.getReader()
  const decoder = new TextDecoder('utf-8')
  let buf = ''
  for (;;) {
    const { value, done } = await reader.read()
    if (done) break
    buf += decoder.decode(value, { stream: true })
    let idx
    while ((idx = buf.indexOf('\n\n')) >= 0) {
      const frame = buf.slice(0, idx)
      buf = buf.slice(idx + 2)
      for (const line of frame.split('\n')) {
        if (!line.startsWith('data:')) continue
        const payload = line.slice(5).trim()
        if (!payload) continue
        try {
          onEvent && onEvent(JSON.parse(payload))
        } catch (e) {
          // 坏帧跳过，不让单帧解析失败中断整条流
        }
      }
    }
  }
}
