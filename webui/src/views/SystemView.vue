<template>
  <div class="system-view">
    <div class="sys-head">
      <div>
        <h2>系统状态</h2>
        <div class="sub">
          只读自检：依赖、配置键存在性、Neo4j、Ollama、缓存、前端构建、端口与工具。不会显示密钥值。
        </div>
      </div>
      <el-button type="primary" plain :loading="loading" @click="load">↻ 重新检查</el-button>
    </div>

    <el-alert
      v-if="error"
      type="error"
      :title="'检查失败：' + error"
      :closable="false"
      show-icon
    />

    <template v-if="data">
      <el-alert
        :type="overall.type"
        :title="overall.title"
        :description="summaryText"
        :closable="false"
        show-icon
      />

      <div v-for="sec in data.sections" :key="sec.name" class="card sys-card">
        <div class="card-head">
          <h3>{{ sec.name }}</h3>
          <span class="sys-sec-count">{{ sec.items.length }} 项</span>
        </div>
        <div v-for="item in sec.items" :key="item.name" class="sys-item">
          <el-tag :type="tagType(item.status)" size="small" effect="plain" class="sys-tag">
            {{ statusText(item.status) }}
          </el-tag>
          <div class="sys-item-body">
            <div class="sys-item-name">{{ item.name }}</div>
            <div v-if="item.detail" class="sys-item-detail">{{ item.detail }}</div>
            <div v-if="item.hint" class="sys-item-hint">建议：{{ item.hint }}</div>
          </div>
        </div>
      </div>

      <div class="sys-foot">
        检查时间：{{ data.generated_at }}。本页只读，不会安装依赖、启动服务或修改配置。
      </div>
    </template>

    <el-skeleton v-else-if="loading" :rows="10" animated style="margin-top:16px" />
  </div>
</template>

<script setup>
import { computed, onMounted, ref } from 'vue'
import { api } from '../api'

const data = ref(null)
const loading = ref(false)
const error = ref('')

async function load() {
  loading.value = true
  error.value = ''
  try {
    data.value = await api.systemCheck()
  } catch (e) {
    error.value = e.message
  } finally {
    loading.value = false
  }
}

onMounted(load)

const counts = computed(() => data.value?.counts || { ok: 0, warn: 0, fail: 0 })

const overall = computed(() => {
  if (counts.value.fail > 0) {
    return { type: 'error', title: `环境未就绪：${counts.value.fail} 项失败` }
  }
  if (counts.value.warn > 0) {
    return { type: 'warning', title: `基本就绪：${counts.value.warn} 项警告` }
  }
  return { type: 'success', title: '环境就绪' }
})

const summaryText = computed(() => {
  if (!data.value) return ''
  return `${counts.value.ok} 项通过 · ${counts.value.warn} 项警告 · ${counts.value.fail} 项失败 · 耗时 ${(data.value.elapsed_ms / 1000).toFixed(1)} 秒`
})

const STATUS = {
  ok: { type: 'success', text: '正常' },
  warn: { type: 'warning', text: '警告' },
  fail: { type: 'danger', text: '失败' },
}

function tagType(status) {
  return (STATUS[status] || STATUS.fail).type
}

function statusText(status) {
  return (STATUS[status] || STATUS.fail).text
}
</script>
