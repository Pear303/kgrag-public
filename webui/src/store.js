import { reactive } from 'vue'
import { api } from './api'

export const store = reactive({
  manualText: '',
  connected: false,
  sessions: [],
  currentSessionId: null,
  messages: [],   // [{role, content, pathDetail, candidates, refusal, refusalDetail, nContexts}]
  streaming: false,

  async refreshHealth() {
    try {
      // 用 /health/lite：只读手册名与环境，不触发 get_service() 的模型构建。
      // 真机 Ollama 单槽，构建会与本机正在跑的答案生成抢资源。
      const h = await api.healthLite()
      this.connected = true
      const hasCounts = h.graph_entities !== undefined && h.communities !== undefined
      const stat = hasCounts
        ? `（${h.graph_entities} 实体 / ${h.communities} 社区）`
        : '（未预热：首次提问后加载）'
      this.manualText = `当前手册：${h.manual || '（未知）'}${stat}`
    } catch (e) {
      this.connected = false
      this.manualText = '⚠️ 无法连接 8001 API（问答不可用）'
    }
  },

  async loadSessions() {
    try {
      this.sessions = (await api.listSessions()) || []
    } catch (e) {
      this.sessions = []
    }
  },

  async newSession() {
    const s = await api.createSession()
    await this.loadSessions()
    this.currentSessionId = s.session_id
    this.messages = []
    return s
  },

  async openSession(id) {
    const sess = await api.getSession(id)
    this.currentSessionId = id
    this.messages = (sess.messages || []).map((m) => {
      if (m.role !== 'assistant') return { role: m.role, content: m.content }
      const d = m.detail || {}
      return {
        role: 'assistant',
        content: m.content,
        // 后端存了 path_detail（微观）或 kg_path_detail（对比）——回看时据此重画链图，
        // 不再依赖旧版那串 chain_html。
        pathDetail: d.path_detail || d.kg_path_detail || [],
        candidates: d.candidates || d.vec_candidates || [],
        refusal: !!d.refusal,
        refusalDetail: d.refusal_detail || null,
        nContexts: d.n_contexts ?? d.kg_n_contexts ?? 0,
        trace: d.trace || null,
      }
    })
  },

  async renameSession(id, title) {
    await api.renameSession(id, title)
    await this.loadSessions()
  },

  async deleteSession(id) {
    // 删除后的落点对齐旧版（frontend/app.py 的 _delete_current）：
    // 以「删除前的显示顺序」定位，删完取 after[pos]（即它后面那个）；
    // 越界则取最后一个；顺序里找不到该 id 则退化为第 0 个。
    const before = (this.sessions || []).map((s) => s.session_id)
    await api.deleteSession(id)
    await this.loadSessions()
    if (this.currentSessionId !== id) return
    const after = (this.sessions || []).map((s) => s.session_id)
    if (!after.length) {
      this.currentSessionId = null
      this.messages = []
      return
    }
    let pos = before.indexOf(id)
    if (pos < 0) pos = 0
    const target = pos < after.length ? after[pos] : after[after.length - 1]
    await this.openSession(target)
  },

  /** 把一轮问答写进后端历史（首轮自动建会话）。 */
  async persist(question, msg) {
    if (!this.currentSessionId) {
      const s = await api.createSession()
      this.currentSessionId = s.session_id
    }
    await api.appendMessage(this.currentSessionId, { role: 'user', content: question })
    await api.appendMessage(this.currentSessionId, {
      role: 'assistant',
      content: msg.content || '',
      paths: msg.paths || [],
      path_detail: msg.pathDetail || [],
      refusal: !!msg.refusal,
      refusal_detail: msg.refusalDetail || {},
      n_contexts: msg.nContexts || 0,
      candidates: msg.candidates || [],
      chain_html: '',
      trace: msg.trace || null,
    })
    await this.loadSessions()
  },
})
