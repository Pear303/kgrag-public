<template>
  <div class="models-view">
    <div class="sys-head">
      <div>
        <h2>模型设置</h2>
        <div class="sub">
          检索 Embedding 与答案 LLM 都只允许本机 Ollama 模型。切换后按模型各用一份缓存，
          来回切换不会每次重建。
        </div>
      </div>
      <el-button plain :loading="loading" @click="load">↻ 刷新</el-button>
    </div>

    <el-alert
      v-if="error"
      type="error"
      :title="'读取失败：' + error"
      :closable="false"
      show-icon
    />

    <el-alert
      v-if="catalog && !catalog.ollama.reachable"
      type="error"
      title="Ollama 不可访问，无法切换模型"
      :description="catalog.ollama.error || '请先确认本机 Ollama 已启动（默认 11434 端口）。'"
      :closable="false"
      show-icon
    />

    <template v-if="status">
      <div class="card current-card">
        <div class="card-head">
          <h3>当前生效</h3>
          <el-tag :type="status.ready ? 'success' : 'warning'" size="small" effect="plain">
            {{ status.ready ? '服务已就绪' : '服务构建中/未就绪' }}
          </el-tag>
        </div>
        <div class="current-grid">
          <div class="current-item">
            <div class="current-label">检索 Embedding</div>
            <div class="current-value">{{ status.selected.embedding_model }}</div>
            <div class="current-meta">
              <span>向量索引：{{ sourceText(h.vector_index) }}</span>
              <span>图谱实体：{{ entitySourceText(h.embedding_source) }}</span>
              <span>社区描述：{{ sourceText(h.community_embedding_source) }}</span>
              <span>块索引：{{ sourceText(h.chunk_index_source) }}</span>
            </div>
          </div>
          <div class="current-item">
            <div class="current-label">答案 LLM</div>
            <div class="current-value">{{ status.selected.llm_model }}</div>
            <div class="current-meta">
              <span>生成答案只走本机模型</span>
              <span>切换答案模型：秒级，不重建索引</span>
            </div>
          </div>
        </div>
        <div class="current-foot">
          出厂默认：embedding <code>{{ status.defaults.embedding_model }}</code> ·
          LLM <code>{{ status.defaults.llm_model }}</code>
          <el-button
            link
            type="primary"
            size="small"
            :disabled="job.state === 'building'"
            @click="resetModels"
          >恢复默认</el-button>
        </div>
      </div>

      <el-alert
        v-if="job.state === 'building'"
        type="info"
        :closable="false"
        show-icon
        class="job-alert"
      >
        <template #title>
          正在构建 {{ job.target_embedding }} 的索引：{{ stageText(job.stage) }}
          <span v-if="job.total">（{{ job.done }} / {{ job.total }}）</span>
        </template>
        <el-progress
          :percentage="jobPercent"
          :indeterminate="job.total <= 1"
          :stroke-width="10"
          style="margin-top: 8px"
        />
        <div class="job-hint">
          构建在后台进行，旧模型继续回答，问答页不会中断。完成后自动切换。
        </div>
      </el-alert>

      <el-alert
        v-else-if="job.state === 'error'"
        type="error"
        title="上一次模型切换失败，当前仍在用旧模型"
        :description="job.error || ''"
        :closable="false"
        show-icon
        class="job-alert"
      />

      <div class="card">
        <div class="card-head">
          <h3>检索 Embedding</h3>
          <span class="hint">向量检索、图谱实体匹配、社区召回、块召回统一使用</span>
        </div>
        <div v-if="!catalog" class="hint" style="margin-bottom:8px">正在读取本地模型目录…</div>
        <el-radio-group v-model="chosenEmbedding" class="model-list">
          <div
            v-for="m in embeddingModels"
            :key="m.name"
            class="model-row"
            :class="{ selected: chosenEmbedding === m.name }"
            @click="selectEmbedding(m.name)"
          >
            <el-radio :value="m.name" :disabled="job.state === 'building'">
              <span class="model-name">{{ m.name }}</span>
            </el-radio>
            <div class="model-tags">
              <el-tag v-if="m.name === status.selected.embedding_model" size="small" type="success" effect="plain">
                当前
              </el-tag>
              <el-tag v-if="m.name === status.defaults.embedding_model" size="small" effect="plain">
                默认
              </el-tag>
              <el-tag v-if="m.name.includes('4b')" size="small" type="warning" effect="plain">
                实验
              </el-tag>
              <span class="model-param">{{ m.parameter_size || '—' }}</span>
              <span class="model-param">{{ m.size_gb }} GB</span>
            </div>
            <div class="model-cache">
              <span
                v-for="b in cacheBadges(m.name)"
                :key="b.key"
                class="cache-badge"
                :class="{ ready: b.ready }"
                :title="b.title"
              >
                {{ b.ready ? '✓' : '○' }} {{ b.label }}
              </span>
            </div>
            <div v-if="m.name.includes('4b')" class="model-note">
              288 题答案级 A/B 结论：整体没有变好（+1.78pp，p=0.69）；表格题更强、
              精确编号/报警题更弱。可在本页切换体验，但不作为生产默认。
            </div>
          </div>
        </el-radio-group>
        <div class="model-actions">
          <el-button
            type="primary"
            :loading="job.state === 'building'"
            :disabled="!catalog || !catalog.ollama.reachable || chosenEmbedding === status.selected.embedding_model"
            @click="applyEmbedding"
          >
            切换检索模型
          </el-button>
          <span class="hint">
            首次切到某模型会构建四类缓存（数分钟）；之后切回直接复用。
          </span>
        </div>
      </div>

      <div class="card">
        <div class="card-head">
          <h3>答案 LLM</h3>
          <span class="hint">必须是本机 completion 模型，且不超过 4.5B 参数</span>
        </div>
        <div v-if="!catalog" class="hint" style="margin-bottom:8px">正在读取本地模型目录…</div>
        <el-radio-group v-model="chosenAnswer" class="model-list">
          <div
            v-for="m in answerModels"
            :key="m.name"
            class="model-row"
            :class="{ selected: chosenAnswer === m.name }"
            @click="chosenAnswer = m.name"
          >
            <el-radio :value="m.name">
              <span class="model-name">{{ m.name }}</span>
            </el-radio>
            <div class="model-tags">
              <el-tag v-if="m.name === status.selected.llm_model" size="small" type="success" effect="plain">
                当前
              </el-tag>
              <el-tag v-if="m.name === status.defaults.llm_model" size="small" effect="plain">
                默认
              </el-tag>
              <span class="model-param">{{ m.parameter_size || '—' }}</span>
              <span class="model-param">{{ m.size_gb }} GB</span>
            </div>
            <div class="model-note">
              {{ m.capabilities.join(' / ') }}
            </div>
          </div>
        </el-radio-group>
        <div class="model-actions">
          <el-button
            type="primary"
            :disabled="!catalog || !catalog.ollama.reachable || chosenAnswer === status.selected.llm_model"
            @click="applyAnswer"
          >
            切换答案模型
          </el-button>
          <span class="hint">只替换生成器，检索与索引不动，切换后立即生效。</span>
        </div>
      </div>

      <div class="card">
        <div class="card-head">
          <h3>按模型缓存</h3>
          <span class="hint">切换来回不覆盖、不互相串用</span>
        </div>
        <el-table :data="status.caches" size="small" border>
          <el-table-column prop="model" label="Embedding 模型" min-width="210">
            <template #default="{ row }">
              <span>{{ row.model }}</span>
              <el-tag v-if="row.model === status.selected.embedding_model" size="small" type="success" effect="plain" style="margin-left:6px">
                当前
              </el-tag>
            </template>
          </el-table-column>
          <el-table-column label="向量索引" width="105" align="center">
            <template #default="{ row }">
              <el-tooltip :content="row.vector.path || ''" placement="top">
                <el-tag :type="row.vector.exists ? 'success' : 'info'" size="small" effect="plain">
                  {{ row.vector.exists ? '已有' : '待建' }}
                </el-tag>
              </el-tooltip>
            </template>
          </el-table-column>
          <el-table-column label="图谱实体" width="105" align="center">
            <template #default="{ row }">
              <el-tooltip :content="row.entity.path || 'Neo4j 中随图存储（默认模型）'" placement="top">
                <el-tag :type="row.entity.exists ? 'success' : 'info'" size="small" effect="plain">
                  {{ row.entity.exists ? '已有' : '待建' }}
                </el-tag>
              </el-tooltip>
            </template>
          </el-table-column>
          <el-table-column label="社区描述" width="105" align="center">
            <template #default="{ row }">
              <el-tooltip :content="row.communities.path || ''" placement="top">
                <el-tag :type="row.communities.exists ? 'success' : 'info'" size="small" effect="plain">
                  {{ row.communities.exists ? '已有' : '待建' }}
                </el-tag>
              </el-tooltip>
            </template>
          </el-table-column>
          <el-table-column label="块索引" width="105" align="center">
            <template #default="{ row }">
              <el-tooltip :content="row.chunk.path || ''" placement="top">
                <el-tag :type="row.chunk.exists ? 'success' : 'info'" size="small" effect="plain">
                  {{ row.chunk.exists ? '已有' : '待建' }}
                </el-tag>
              </el-tooltip>
            </template>
          </el-table-column>
          <el-table-column label="滑窗通道" width="105" align="center">
            <template #default="{ row }">
              <el-tooltip :content="row.sliding.path || ''" placement="top">
                <el-tag
                  :type="row.sliding.exists ? 'success' : (row.sliding.enabled ? 'info' : 'info')"
                  size="small"
                  effect="plain"
                >
                  {{ row.sliding.enabled ? (row.sliding.exists ? '已有' : '待建') : '未启用' }}
                </el-tag>
              </el-tooltip>
            </template>
          </el-table-column>
          <el-table-column label="缓存完整" width="95" align="center">
            <template #default="{ row }">
              <el-tag :type="row.cache_complete ? 'success' : 'warning'" size="small" effect="plain">
                {{ row.cache_complete ? '是' : '否' }}
              </el-tag>
            </template>
          </el-table-column>
        </el-table>
        <div class="cache-foot">
          块索引向量按设计只在本进程内复用（qwen3-embedding 长文本跨进程不可复现），
          API 重启后第一次会重算一次；实体/向量/社区/滑窗缓存跨进程可用。
        </div>
      </div>

      <div class="sys-foot">
        本页只切换本机模型；不改 .env、不调用外部接口。切换记录写入
        <code>data/runtime/model_prefs.json</code>，重启后继续生效。
      </div>
    </template>

    <el-skeleton v-else-if="loading" :rows="12" animated style="margin-top:16px" />
  </div>
</template>

<script setup>
import { computed, onBeforeUnmount, onMounted, ref } from 'vue'
import { ElMessage, ElMessageBox } from 'element-plus'
import { api } from '../api'

const catalog = ref(null)
const status = ref(null)
const loading = ref(false)
const error = ref('')
const chosenEmbedding = ref('')
const chosenAnswer = ref('')
const confirming = ref(false)
let timer = null

const job = computed(() => status.value?.job || { state: 'idle' })
const h = computed(() => status.value?.health || {})
const embeddingModels = computed(() => catalog.value?.embedding_models || [])
const answerModels = computed(() => catalog.value?.answer_models || [])

const jobPercent = computed(() => {
  const j = job.value
  if (!j.total) return 0
  return Math.min(100, Math.round((j.done / j.total) * 100))
})

const STAGES = {
  queued: '排队中',
  connect: '连接图库',
  embedding_init: '加载 embedding 模型',
  communities: '计算社区描述向量',
  entities: '计算图谱实体向量',
  chunks: '计算块索引向量',
  vector: '构建向量索引',
  sliding: '构建滑窗检索器',
  graph: '加载图谱实体',
  ready: '完成',
  error: '失败',
}

function stageText(s) {
  return STAGES[s] || s || '准备中'
}

function sourceText(v) {
  const MAP = {
    cache: '复用缓存',
    built: '本次构建',
    neo4j: '随图存储',
    off: '未启用',
    none: '无',
    ready: '已就绪',
    'cache-file': '社区缓存文件',
  }
  return MAP[v] || (v == null ? '—' : v)
}

function entitySourceText(v) {
  return v === 'neo4j' ? '随图存储（默认模型）' : sourceText(v)
}

function cacheBadges(name) {
  const c = (status.value?.caches || []).find(x => x.model === name)
  if (!c) {
    return [
      { key: 'v', label: '向量', ready: false, title: '缓存状态未知' },
      { key: 'e', label: '实体', ready: false, title: '缓存状态未知' },
      { key: 'c', label: '社区', ready: false, title: '缓存状态未知' },
      { key: 'k', label: '块', ready: false, title: '缓存状态未知' },
    ]
  }
  return [
    { key: 'v', label: '向量', ready: !!c.vector.exists, title: c.vector.path || '' },
    { key: 'e', label: '实体', ready: !!c.entity.exists, title: c.entity.path || 'Neo4j 中随图存储（默认模型）' },
    { key: 'c', label: '社区', ready: !!c.communities.exists, title: c.communities.path || '' },
    { key: 'k', label: '块', ready: !!c.chunk.exists, title: c.chunk.path || '' },
    { key: 's', label: '滑窗', ready: !c.sliding?.enabled || !!c.sliding?.exists, title: c.sliding?.path || '' },
  ]
}

function friendlyError(e) {
  const m = e?.message || String(e)
  if (m.includes('404') || m.toLowerCase().includes('not found')) {
    return '后端还未加载模型设置接口（当前是旧进程）。重启 API 后本页即可使用。'
  }
  return m
}

function syncChoices(s) {
  if (!chosenEmbedding.value) chosenEmbedding.value = s.selected.embedding_model
  if (!chosenAnswer.value) chosenAnswer.value = s.selected.llm_model
}

async function load() {
  loading.value = true
  error.value = ''
  try {
    // 状态接口约 0.02s，先渲染当前模型；模型目录要探测 Ollama（约 2s），
    // 不阻塞整页，否则用户会盯着整页骨架屏等目录。
    const s = await api.modelsStatus()
    status.value = s
    syncChoices(s)
    if (s.job?.state === 'building') startPolling()
    api.modelsLocal().then(c => {
      catalog.value = c
      const emb = c.embedding_models || []
      const ans = c.answer_models || []
      if (!emb.some(m => m.name === chosenEmbedding.value)) {
        chosenEmbedding.value = s.selected.embedding_model
      }
      if (!ans.some(m => m.name === chosenAnswer.value)) {
        chosenAnswer.value = s.selected.llm_model
      }
      // 目录已热：再拉一次状态，把全部候选模型的缓存行补全（此时约 0.02s）
      return api.modelsStatus().then(s2 => { status.value = s2 })
    }).catch(e => {
      if (!error.value) error.value = '模型目录读取失败：' + friendlyError(e)
    })
  } catch (e) {
    error.value = friendlyError(e)
  } finally {
    loading.value = false
  }
}

function selectEmbedding(name) {
  if (job.value.state !== 'building') chosenEmbedding.value = name
}

function startPolling() {
  if (timer) return
  timer = setInterval(async () => {
    try {
      const s = await api.modelsStatus()
      status.value = s
      const j = s.job || {}
      if (j.state === 'building') return
      stopPolling()
      if (j.state === 'ready') {
        ElMessage.success('模型切换完成')
      } else if (j.state === 'error') {
        ElMessage.error('模型切换失败：' + (j.error || '未知错误'))
      }
      await load()
    } catch (e) {
      stopPolling()
      ElMessage.error('轮询切换状态失败：' + e.message)
    }
  }, 2000)
}

function stopPolling() {
  if (timer) {
    clearInterval(timer)
    timer = null
  }
}

async function applyEmbedding() {
  const target = chosenEmbedding.value
  if (!target || confirming.value) return
  const cache = (status.value?.caches || []).find(c => c.model === target)
  const cached = !!cache?.cache_complete
  const detail = cached
    ? `该模型的缓存已在盘上，切换时直接加载，不重算向量。构建期间旧模型继续服务。`
    : `该模型缓存不完整，首次切换会构建向量索引、图谱实体向量、社区描述向量、` +
      `块索引${cache?.sliding?.enabled ? '与滑窗检索器' : ''}（数分钟）；` +
      `构建期间旧模型继续服务，完成后自动切换。`
  confirming.value = true
  try {
    await ElMessageBox.confirm(
      `将把检索 Embedding 切换为 ${target}。${detail}`,
      '切换检索模型',
      { type: 'warning', confirmButtonText: '开始切换', cancelButtonText: '取消' }
    )
  } catch (e) {
    confirming.value = false
    return
  }
  confirming.value = false
  try {
    const r = await api.switchEmbedding(target)
    if (!r.changed) {
      ElMessage.info(r.message || '已经是当前模型')
      await load()
      return
    }
    ElMessage.info(r.message || '已开始构建')
    status.value = { ...(status.value || {}), job: r.job }
    startPolling()
  } catch (e) {
    ElMessage.error('切换失败：' + e.message)
  }
}

async function applyAnswer() {
  const target = chosenAnswer.value
  if (!target) return
  try {
    const r = await api.switchAnswer(target)
    ElMessage.success(r.message || '答案模型已切换')
    await load()
  } catch (e) {
    ElMessage.error('切换失败：' + e.message)
  }
}

async function resetModels() {
  if (confirming.value) return
  confirming.value = true
  try {
    await ElMessageBox.confirm(
      '将恢复 .env 里的出厂默认模型。若当前不是默认 embedding，会触发一次后台重建。',
      '恢复默认模型',
      { type: 'warning', confirmButtonText: '恢复默认', cancelButtonText: '取消' }
    )
  } catch (e) {
    confirming.value = false
    return
  }
  confirming.value = false
  try {
    const r = await api.resetModels()
    if (r.job?.state === 'building') {
      ElMessage.info('已开始构建默认模型索引')
      status.value = { ...(status.value || {}), job: r.job }
      startPolling()
    } else {
      ElMessage.success('已恢复默认模型')
      await load()
    }
  } catch (e) {
    ElMessage.error('恢复失败：' + e.message)
  }
}

onMounted(load)
onBeforeUnmount(stopPolling)
</script>

<style scoped>
.models-view { padding: 18px 22px; overflow-y: auto; height: 100%; }
.models-view h2 { margin: 0 0 4px; font-size: 18px; }
.models-view .sub { color: #909399; font-size: 13px; margin-bottom: 0; }

.current-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 14px; }
@media (max-width: 980px) { .current-grid { grid-template-columns: 1fr; } }
.current-item {
  border: 1px solid #ebeef5; border-radius: 8px; padding: 12px 14px;
  background: #fafbfc;
}
.current-label { font-size: 12px; color: #909399; }
.current-value { font-size: 16px; font-weight: 600; margin: 4px 0 8px; word-break: break-all; }
.current-meta { display: flex; flex-wrap: wrap; gap: 6px 14px; font-size: 12px; color: #606266; }
.current-foot { margin-top: 12px; font-size: 12px; color: #909399; }
.current-foot code { color: #606266; }

.job-alert { margin-bottom: 16px; }
.job-hint { font-size: 12px; color: #606266; margin-top: 6px; }

.hint { font-size: 12px; color: #909399; }
.model-list {
  display: flex; flex-direction: column; width: 100%; gap: 8px;
  align-items: stretch;
}
.model-row {
  display: block; width: 100%; border: 1px solid #e4e7ed; border-radius: 8px;
  padding: 10px 12px; cursor: pointer; transition: border-color .15s, background .15s;
}
.model-row:hover { border-color: #c0c4cc; }
.model-row.selected { border-color: var(--kg-accent); background: #f5f9ff; }
.model-name { font-weight: 600; color: #303133; }
.model-tags { display: flex; align-items: center; gap: 6px; margin: 6px 0 0 24px; flex-wrap: wrap; }
.model-param { font-size: 12px; color: #909399; }
.model-cache { display: flex; gap: 6px; margin: 8px 0 0 24px; flex-wrap: wrap; }
.cache-badge {
  font-size: 12px; color: #a8abb2; border: 1px solid #e4e7ed;
  border-radius: 10px; padding: 0 8px; line-height: 20px;
}
.cache-badge.ready { color: #529b2e; border-color: #b3e19d; background: #f0f9eb; }
.model-note { font-size: 12px; color: #909399; margin: 6px 0 0 24px; line-height: 1.6; }
.model-actions { display: flex; align-items: center; gap: 12px; margin-top: 14px; }
.cache-foot { font-size: 12px; color: #909399; margin-top: 10px; line-height: 1.6; }
</style>
