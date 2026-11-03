<template>
  <div class="chat-wrap">
    <div ref="scrollEl" class="chat-scroll">
      <div v-if="!store.messages.length" style="color:#909399;font-size:14px;line-height:2">
        <div>输入故障问题开始。默认「微观」＝ 图谱路径推理 + 答案生成。</div>
        <div style="color:#b0b3b8">
          例：三菱M64系统中，若使用变频器作为主轴驱动系统，参数P145应设定为什么值？
        </div>
      </div>

      <div v-for="(m, i) in store.messages" :key="i" class="msg" :class="m.role">
        <div class="bubble">
          <template v-if="m.role === 'user'">{{ m.content }}</template>
          <template v-else>
            <div v-if="m.refusal" class="refusal-card">
              <div class="refusal-card-head">已拒答 · {{ refusalWhy(m.refusalDetail || {}) }}</div>
              <div class="refusal-card-body" v-html="md(m.content)"></div>
            </div>
            <div v-else v-html="md(m.content)"></div>

            <div v-if="hasChain(m)" class="chain-box">
              <div class="chain-cap">
                推理链图：{{ nodeCount(m) }} 个节点 / {{ edgeCount(m) }} 条边（点节点看原文）
              </div>
              <ChainGraph :elements="els(m)" :contexts="m.trace?.contexts || []" :height="420" />
            </div>

            <div v-if="m.candidates && m.candidates.length" class="chain-box">
              <div class="chain-cap">向量检索候选片段（{{ m.candidates.length }} 条）</div>
              <details v-for="(c, ci) in m.candidates" :key="ci">
                <summary>候选 {{ ci + 1 }} · 相似度 {{ c.score }}</summary>
                <div class="chain-detail">{{ c.text }}</div>
              </details>
            </div>

            <div v-if="m.trace" class="chain-box">
              <details>
                <summary>
                  生成过程留痕 · 送进模型 {{ m.trace.ctx_profile.length }} 块
                  （路径碎片 {{ m.trace.n_path }} / 手册原文 {{ m.trace.n_text }}，
                  共 {{ m.trace.ctx_chars_total }} 字）· 提示词 {{ m.trace.prompt_chars }} 字
                  · 模型原样输出 {{ m.trace.raw_chars }} 字
                </summary>
                <div class="chain-detail">
                  <div v-for="c in m.trace.ctx_profile" :key="c.i">
                    {{ c.i }}｜{{ c.kind }}｜{{ c.chars }} 字｜{{ c.head }}
                  </div>
                  <pre class="raw-out">{{ m.trace.raw_output }}{{
                    m.trace.raw_truncated ? '\n…（超出留痕上限，已截断）' : '' }}</pre>
                </div>
              </details>
            </div>
          </template>
        </div>
      </div>
    </div>

    <div class="chat-input">
      <div class="row">
        <el-input
          v-model="question"
          type="textarea"
          :rows="2"
          resize="none"
          placeholder="如：三菱M64系统中，若使用变频器作为主轴驱动系统，参数P145应设定为什么值？"
          @keydown.enter.exact.prevent="send"
        />
        <el-button
          v-if="!(store.streaming && mode === 'micro')"
          type="primary"
          size="large"
          :loading="store.streaming"
          @click="send"
        >
          发送
        </el-button>
        <el-button v-else type="danger" size="large" @click="stop">
          停止
        </el-button>
      </div>
      <div class="row" style="margin-top: 8px">
        <el-radio-group v-model="mode" size="small">
          <el-radio-button value="micro">微观（图谱推理）</el-radio-button>
          <el-radio-button value="macro">宏观（社区总结）</el-radio-button>
          <el-radio-button value="compare">对比（KG vs 向量）</el-radio-button>
        </el-radio-group>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, reactive, nextTick } from 'vue'
import { ElMessage } from 'element-plus'
import ChainGraph from '../components/ChainGraph.vue'
import { api, streamQuery } from '../api'
import { store } from '../store'
import { buildChainElements, renderPathDetail } from '../chain'
import { md } from '../md'

const question = ref('')
const mode = ref('micro')
const scrollEl = ref(null)

// 停止按钮只在微观（流式）这条路上有效：实测依据只覆盖了它（客户端断线后下一题不必等
// 剩余生成，但下一题仍贵约 4.7 秒），宏观/对比是非流式接口，没有可中断的中间态。
let ctrl = null
let waitTimer = null
function stop() {
  if (ctrl) ctrl.abort()
}

// 每条 assistant 消息的链图 elements 只算一次，缓存到消息对象上
const _cache = new WeakMap()
function els(m) {
  if (!_cache.has(m)) _cache.set(m, buildChainElements(m.pathDetail))
  return _cache.get(m)
}
const hasChain = (m) => (m.pathDetail || []).length > 0 && els(m).nodes.length > 0
const nodeCount = (m) => els(m).nodes.length
const edgeCount = (m) => els(m).edges.length

function scrollToBottom() {
  nextTick(() => {
    const el = scrollEl.value
    if (el) el.scrollTop = el.scrollHeight
  })
}

function composeMicro(am, answer) {
  let out = answer || ''
  if (am.refusal) {
    out += `\n\n> 系统判定：已拒答（${refusalWhy(am.refusalDetail || {})}）`
  } else if (am.pathDetail && am.pathDetail.length) {
    const pd = renderPathDetail(am.pathDetail)
    if (pd) out += `\n\n<details><summary>知识图谱多跳路径（文本）</summary>\n\n${pd}\n\n</details>`
  }
  return out
}

// 拒答有三道不同的闸门，成因不一样，话术必须分开：把三道合并成一句"内容不在知识库里"
// 是在对用户说谎（2026-10-17 实测：厂商门拦的 5 道题，答案逐字在被喂的手册里）。
function refusalWhy(rd) {
  if (rd.vendor_gate) {
    return rd.reason || '图谱里未收录问题提到的厂商或型号词条'
  }
  if (rd.joint_gate) {
    const j = rd.joint_gate
    return `综合校验判定超出本手册覆盖范围（综合得分 ${j.score ?? '-'}，判线 ${j.theta ?? '-'}）`
  }
  if (rd.sent_cov !== undefined && rd.sent_cov !== null) {
    return `问题与这本手册的领域语义相似度偏低（相似度 ${rd.sent_cov}）`
  }
  if (rd.relevance !== undefined || rd.object_mismatch !== undefined) {
    const bits = []
    if (rd.relevance !== undefined && rd.relevance !== null) bits.push(`相关度 ${rd.relevance}`)
    if (rd.object_mismatch) bits.push('提问对象与手册内容不匹配')
    return `检索到的内容与问题对不上（${bits.join('；') || '依据不足'}）`
  }
  return '未检索到可作答的内容'
}

function composeCompare(kg, vec) {
  const parts = []
  parts.push('### 知识图谱（KG-RAG）答案\n' + (kg.answer || ''))
  if (kg.refusal) {
    // 拒答依据原样带上（相关度/对象不匹配/联合门得分…），泛化成
    // "未检索到可作答的内容"会让运维没法判断是被哪道校验拦下的
    parts.push('> [系统判定：已拒答（' + refusalWhy(kg.refusal_detail || {}) + '）]')
    parts.push('> 注意：向量检索通道没有拒答门，下方答案可能基于弱相关片段，请对照其相似度分值判读。')
  }
  const pd = renderPathDetail(kg.path_detail || [])
  if (pd) parts.push('<details><summary>知识图谱多跳路径（文本）</summary>\n\n' + pd + '\n\n</details>')
  parts.push('### 向量检索（Vector RAG）答案\n' + (vec.answer || ''))
  return parts.join('\n\n')
}

// 必须是响应式对象：直接改裸对象不会触发模板重渲染
function newAssistantMsg() {
  return reactive({
    role: 'assistant',
    content: '⏳ 正在检索图谱并推理…',
    pathDetail: [],
    candidates: [],
    refusal: false,
    refusalDetail: null,
    nContexts: 0,
    trace: null,
    stopped: false,
  })
}

async function send() {
  const q = question.value.trim()
  if (!q || store.streaming) return
  if (!store.currentSessionId) {
    try { await store.newSession() } catch (e) { /* 后端不可用时仍允许本地展示 */ }
  }

  store.messages.push({ role: 'user', content: q })
  question.value = ''
  const am = newAssistantMsg()
  store.messages.push(am)
  store.streaming = true
  scrollToBottom()

  try {
    if (mode.value === 'micro') {
      let meta = null
      let answer = ''
      let gotError = false
      let errMsg = ''
      ctrl = new AbortController()
      // 动态等待提示（2026-10-24 产品化）：本地生成 20~40s，每 0.5s 刷新等待秒数，
      // 首 delta / error / done 到达即停表，避免"死屏感"
      // waitTimer 提到 try 外：中止（AbortError）走 catch 时也必须停表——
      // 否则它每 0.5s 把"⏹ 已停止"覆盖回"⏳ 正在检索…（Ns）"且永不清理
      // （2026-10-26 补测实测：中止后消息永远显示检索中）。
      const t0 = Date.now()
      const startWaitTimer = () => {
        waitTimer = setInterval(() => {
          if (answer || gotError) { clearInterval(waitTimer); return }
          const sec = Math.round((Date.now() - t0) / 1000)
          am.content = meta
            ? `⏳ 已命中 ${meta.n_contexts || 0} 条上下文，正在生成答案…（已等待 ${sec}s，通常 20~40s）`
            : `⏳ 正在检索图谱并构建上下文…（${sec}s）`
        }, 500)
      }
      startWaitTimer()
      await streamQuery(q, {
        signal: ctrl.signal,
        onEvent: (ev) => {
          if (ev.type === 'meta') {
            meta = ev
            if (!answer) am.content = `⏳ 已命中 ${ev.n_contexts || 0} 条上下文，正在生成答案…`
          } else if (ev.type === 'reasoning') {
            am.content = `⏳ 正在推理…（${ev.chars || 0} 字）`
          } else if (ev.type === 'answer_delta') {
            if (!answer) { clearInterval(waitTimer); am.content = '' }
            answer += ev.text || ''
            am.content = answer
          } else if (ev.type === 'error') {
            gotError = true
            errMsg = ev.message || '查询失败'
            am.content = '❌ ' + errMsg
          } else if (ev.type === 'done') {
            answer = ev.answer || answer
            meta = ev
          }
          scrollToBottom()
        },
      })
      clearInterval(waitTimer)
      if (meta) {
        am.refusal = !!meta.refusal
        am.refusalDetail = meta.refusal_detail || null
        am.nContexts = meta.n_contexts || 0
        am.pathDetail = meta.path_detail || []
        am.trace = meta.trace || null
      }
      // 兜底（2026-10-24）：error 后 composeMicro 不再无条件覆盖错误消息——
      // 此前 227 行重写会抹掉 onEvent 里设置的 ❌ 提示，用户看到的是"空回答"
      if (gotError) {
        am.content = '⚠ 生成失败：' + errMsg
          + '\n\n常见原因：本地模型服务暂时不可用。请稍后重试；若持续失败，请检查 Ollama 与后端服务状态。'
          + (meta && meta.path_detail && meta.path_detail.length
            ? '\n\n（本次检索已完成，共 ' + (meta.n_contexts || 0) + ' 条上下文——问题本身可答，是生成环节失败。）' : '')
      } else {
        am.content = composeMicro(am, answer)
      }
    } else if (mode.value === 'macro') {
      const r = await api.queryMacro(q)
      am.content = r.answer || ''
      if (r.communities && r.communities.length) {
        const lines = r.communities.map(c => {
          const head = (c.summary || '').replace(/\s+/g, ' ').slice(0, 160)
          const nodes = (c.top_nodes || []).slice(0, 6).join('、')
          return `- **社区 ${c.id}**（${c.nodes} 节点）${nodes ? '：' + nodes : ''}${head ? '\n  ' + head : ''}`
        })
        // 宏观没有拒答门：答不上时会保底召回相似度最近的社区。此时把它们标成
        // "命中"会与拒答正文自相矛盾（2026-10-26 补测：超纲题列出无关社区）。
        const caveat = /无法总结|未检索到相关社区/.test(r.answer || '')
          ? '\n> 注：本问题未匹配到可靠相关的社区，以下仅为相似度最近的保底召回，可能与问题无关。' : ''
        am.content += `\n\n【命中社区】${r.communities.length} 个${caveat}\n` + lines.join('\n')
      }
    } else {
      const r = await api.queryCompare(q)
      const kg = r.kg || {}
      const vec = r.vector || {}
      am.content = composeCompare(kg, vec)
      am.pathDetail = kg.path_detail || []
      am.candidates = vec.candidates || []
      am.refusal = !!kg.refusal
      am.refusalDetail = kg.refusal_detail || null
      am.nContexts = kg.n_contexts || 0
      am.trace = kg.trace || null
    }
  } catch (e) {
    if (e && e.name === 'AbortError') {
      // 用户按了停止：保留已经收到的内容，既不伪装成失败，也不承诺"立刻恢复全速"
      // （实测中止后下一道题仍比空闲贵约 4.7 秒）
      const body = am.content || ''
      am.stopped = true
      am.content = body.startsWith('⏳')
        ? '⏹ 已停止：这条回答还没吐出正文就被中止了。'
        : body + '\n\n> ⏹ 已停止，以上是中止前收到的部分内容。'
    } else {
      am.content = '❌ 查询失败：' + e.message
    }
  } finally {
    ctrl = null
    if (waitTimer) clearInterval(waitTimer)
    store.streaming = false
    scrollToBottom()
    try {
      await store.persist(q, {
        content: am.content,
        pathDetail: am.pathDetail,
        candidates: am.candidates,
        refusal: am.refusal,
        refusalDetail: am.refusalDetail,
        nContexts: am.nContexts,
        trace: am.trace,
      })
    } catch (e) {
      ElMessage.warning('历史写入失败：' + e.message)
    }
  }
}
</script>
