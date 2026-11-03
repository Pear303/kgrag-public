<template>
  <div class="session-wrap">
    <div class="side-head">
      <span class="side-head-title">会话</span>
      <el-button link size="small" @click="refresh">↻ 刷新</el-button>
    </div>
    <div class="session-list">
      <div v-if="!store.sessions.length" class="empty" style="color:#8b98a8;font-size:13px;padding:8px 10px">
        （暂无会话）
      </div>
      <div
        v-for="s in store.sessions"
        :key="s.session_id"
        class="session-item"
        :class="{ active: s.session_id === store.currentSessionId }"
        @click="open(s.session_id)"
      >
        <span class="t">{{ s.title || s.session_id }}</span>
        <span class="ops" @click.stop>
          <el-button link size="small" @click="rename(s)">改名</el-button>
          <el-button link size="small" type="danger" @click="remove(s)">删</el-button>
        </span>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ElMessage, ElMessageBox } from 'element-plus'
import { store } from '../store'

async function refresh() {
  await store.loadSessions()
  ElMessage.success(`已刷新（${store.sessions.length} 个会话）`)
}

async function open(id) {
  try {
    await store.openSession(id)
  } catch (e) {
    ElMessage.error('打开失败：' + e.message)
  }
}

async function rename(s) {
  try {
    const { value } = await ElMessageBox.prompt('输入新的会话标题', '重命名', {
      inputValue: s.title || '',
      confirmButtonText: '确定',
      cancelButtonText: '取消',
    })
    if (value && value.trim()) {
      await store.renameSession(s.session_id, value.trim())
      ElMessage.success('已重命名')
    }
  } catch (e) { /* 用户取消 */ }
}

async function remove(s) {
  try {
    await ElMessageBox.confirm(`删除会话「${s.title || s.session_id}」？`, '确认删除', {
      type: 'warning',
      confirmButtonText: '删除',
      cancelButtonText: '取消',
    })
    await store.deleteSession(s.session_id)
    ElMessage.success('已删除')
  } catch (e) { /* 用户取消 */ }
}
</script>
