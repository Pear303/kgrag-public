<template>
  <div class="manuals">
    <h2>手册管理</h2>
    <div class="sub">
      上传 → 提示词校准 → 人工过目变更单 → 确认构建 → 设为当前查询
    </div>

    <!-- 上传与校准 -->
    <div class="card">
      <h3>1. 上传手册（自动进入提示词校准）</h3>
      <div style="display:flex;gap:12px;align-items:flex-start;flex-wrap:wrap">
        <el-upload
          :auto-upload="false"
          :limit="1"
          :file-list="fileList"
          accept=".md,.markdown,.txt,.pdf,.docx"
          :on-change="onFileChange"
          :on-remove="() => (fileList = [])"
        >
          <el-button>选择文件</el-button>
        </el-upload>
        <el-button type="primary" :loading="uploading" @click="doUpload">上传并校准</el-button>
        <span v-if="uploadMsg" style="font-size:13px;color:#606266">{{ uploadMsg }}</span>
      </div>
    </div>

    <!-- 校准变更单 -->
    <div class="card">
      <h3>2. 校准变更单（C1：须人工过目）</h3>
      <div style="display:flex;gap:10px;align-items:center;margin-bottom:10px">
        <el-input v-model="manualId" placeholder="手册ID" style="width:260px" clearable />
        <el-button @click="loadCalibration">查看校准结果</el-button>
        <el-button type="primary" :disabled="!manualId" @click="doConfirmBuild">
          确认并开始构建
        </el-button>
      </div>
      <div v-if="calibration" class="calib-changes">
        <div>
          采样模式：<b>{{ calibration.mode }}</b>（手册 {{ calibration.manual_chars }} 字 →
          送入 {{ calibration.selected_chars }} 字 / {{ calibration.n_windows }} 窗）
        </div>
        <div>适配版提示词 sha8：<code>{{ calibration.adapted_prompt_sha8 }}</code></div>
        <div style="margin-top:6px">
          变更 {{ (calibration.changes || []).length }} 处：
        </div>
        <ul>
          <li v-for="(ch, i) in calibration.changes || []" :key="i">
            <b>[{{ ch.slot }}]</b> {{ ch.reason }}
          </li>
        </ul>
        <details v-if="calibration.updates">
          <summary>新槽值全文</summary>
          <pre style="white-space:pre-wrap;font-size:12px">{{ Object.entries(calibration.updates).map(([k, v]) => `### ${k}\n${v}`).join('\n\n') }}</pre>
        </details>
      </div>
      <div v-else style="color:#909399;font-size:13px">（暂无校准结果）</div>
    </div>

    <!-- 任务进度 -->
    <div class="card">
      <div class="card-head">
        <h3>3. 任务进度</h3>
        <el-button link size="small" @click="refreshTasks">↻ 刷新</el-button>
      </div>
      <el-table :data="tasks" size="small" style="width:100%">
        <el-table-column prop="task_id" label="任务ID" width="150" />
        <el-table-column label="类型" width="100">
          <template #default="{ row }">
            {{ { calibrate: '校准', build: '构建' }[row.kind] || '构建' }}
          </template>
        </el-table-column>
        <el-table-column prop="status" label="状态" width="120" />
        <el-table-column prop="step" label="步骤" width="90" />
        <el-table-column label="进度">
          <template #default="{ row }">
            <el-progress :percentage="Number(row.pct || 0)" :stroke-width="12" />
          </template>
        </el-table-column>
      </el-table>
    </div>

    <!-- 手册列表 -->
    <div class="card">
      <div class="card-head">
        <h3>4. 手册列表</h3>
        <el-button link size="small" @click="refreshManuals">↻ 刷新</el-button>
      </div>
      <el-table :data="manuals" size="small" style="width:100%">
        <el-table-column prop="manual_id" label="手册ID" min-width="200" />
        <el-table-column prop="status" label="状态" width="120" />
        <el-table-column label="三元组" width="100">
          <template #default="{ row }">{{ row.build?.triples ?? '-' }}</template>
        </el-table-column>
        <el-table-column label="社区" width="90">
          <template #default="{ row }">{{ row.build?.communities ?? '-' }}</template>
        </el-table-column>
        <el-table-column label="当前" width="80">
          <template #default="{ row }">
            <el-tag v-if="row.current" type="success" size="small">当前</el-tag>
          </template>
        </el-table-column>
        <el-table-column label="操作" width="180">
          <template #default="{ row }">
            <!-- busy：一次"设为当前"要在服务端整份重建检索器（几十秒），期间全表操作按钮置灰，
                 否则连点会触发多次全量重建，每次都白等一遍 -->
            <el-button link size="small" :disabled="busy || row.current"
                       :loading="activatingId === row.manual_id"
                       @click="doActivate(row)">
              设为当前
            </el-button>
            <el-button link size="small" type="danger" :disabled="busy || row.current"
                       @click="doDelete(row.manual_id)">
              删除
            </el-button>
          </template>
        </el-table-column>
      </el-table>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted, onBeforeUnmount } from 'vue'
import { ElMessage, ElMessageBox, ElLoading } from 'element-plus'
import { api } from '../api'
import { store } from '../store'

const manuals = ref([])
const tasks = ref([])
const fileList = ref([])
const uploading = ref(false)
const uploadMsg = ref('')
const manualId = ref('')
const calibration = ref(null)
const busy = ref(false)
const activatingId = ref('')
let timer = null

async function loadManuals() {
  try { manuals.value = (await api.listManuals()) || [] } catch (e) { manuals.value = [] }
}
async function loadTasks() {
  try { tasks.value = (await api.listTasks()) || [] } catch (e) { tasks.value = [] }
}

// 手动刷新（对齐旧版「刷新任务」/「刷新手册」两个按钮）：
// 任务进度另有 8s 自动轮询兜底；手册列表没有轮询，构建/激活完成时靠这个按钮更新。
async function refreshTasks() {
  await loadTasks()
  ElMessage.success(`已刷新（${tasks.value.length} 个任务）`)
}
async function refreshManuals() {
  await loadManuals()
  ElMessage.success(`已刷新（${manuals.value.length} 本手册）`)
}

function onFileChange(f) {
  fileList.value = [f]
}

async function doUpload() {
  const f = fileList.value[0]
  if (!f || !f.raw) { ElMessage.warning('请先选择文件'); return }
  uploading.value = true
  uploadMsg.value = ''
  try {
    const j = await api.uploadManual(f.raw)
    manualId.value = j.manual_id || ''
    uploadMsg.value = `已入队校准任务 ${j.task_id}；手册ID：${j.manual_id}`
    ElMessage.success('上传成功，已进入校准')
    fileList.value = []
    await Promise.all([loadTasks(), loadManuals()])
  } catch (e) {
    uploadMsg.value = '上传失败：' + e.message
    ElMessage.error('上传失败：' + e.message)
  } finally {
    uploading.value = false
  }
}

async function loadCalibration() {
  if (!manualId.value) { ElMessage.warning('请先填手册ID或上传手册'); return }
  try {
    calibration.value = await api.getCalibration(manualId.value)
  } catch (e) {
    calibration.value = null
    ElMessage.warning('暂无校准件：' + e.message)
  }
}

async function doConfirmBuild() {
  try {
    await ElMessageBox.confirm('确认该手册的校准变更单已过目，开始建图？', '确认构建', {
      type: 'warning', confirmButtonText: '开始构建', cancelButtonText: '取消',
    })
  } catch (e) { return }
  try {
    const j = await api.confirmBuild(manualId.value)
    ElMessage.success('已入队构建任务 ' + j.task_id)
    await loadTasks()
  } catch (e) {
    ElMessage.error('确认失败：' + e.message)
  }
}

async function doActivate(row) {
  if (busy.value) return
  const id = row.manual_id
  // 表格第一列显示的是手册ID，提示只报 display_name 会对不上行；
  // 两者不一样时并列给出，保证"点名的那本"能在表里找到。
  const dn = row.display_name || row.source_manual || ''
  const name = dn && dn !== id ? `${dn}（${id}）` : id
  busy.value = true
  activatingId.value = id
  // activate 是同步接口：服务端要在请求里重建向量索引和社区缓存（实测几十秒），
  // 期间必须给出进行中的画面，否则看着像"点了没反应"。
  const loading = ElLoading.service({
    text: `正在按「${name}」重建检索器与社区缓存，约需几十秒，请稍候…`,
    background: 'rgba(255, 255, 255, .75)',
  })
  try {
    const r = await api.activateManual(id)
    if (r && r.warning) {
      ElMessage.warning({ message: `已切换为「${name}」，但：${r.warning}`, duration: 10000, showClose: true })
    } else {
      ElMessage.success(`已把当前手册切换为「${name}」`)
    }
    await Promise.all([loadManuals(), store.refreshHealth()])
  } catch (e) {
    ElMessage.error(`切换到「${name}」失败：${e.message}`)
  } finally {
    loading.close()
    activatingId.value = ''
    busy.value = false
  }
}

async function doDelete(id) {
  try {
    await ElMessageBox.confirm(`删除手册「${id}」及其图谱产物？`, '确认删除', {
      type: 'warning', confirmButtonText: '删除', cancelButtonText: '取消',
    })
  } catch (e) { return }
  try {
    await api.deleteManual(id)
    ElMessage.success('已删除')
    await loadManuals()
  } catch (e) {
    ElMessage.error('删除失败：' + e.message)
  }
}

onMounted(async () => {
  await Promise.all([loadManuals(), loadTasks()])
  timer = setInterval(loadTasks, 8000)   // /tasks 很轻，只刷新任务进度
})
onBeforeUnmount(() => { if (timer) clearInterval(timer) })
</script>
