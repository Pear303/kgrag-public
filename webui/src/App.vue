<template>
  <el-container class="app-shell">
    <el-aside width="272px" class="app-side">
      <div class="brand">KGRAG 工业故障推理</div>

      <div class="app-nav">
        <el-menu :default-active="route.path" router>
          <el-menu-item index="/">问答</el-menu-item>
          <el-menu-item index="/manuals">手册管理</el-menu-item>
          <el-menu-item index="/system">系统状态</el-menu-item>
          <el-menu-item index="/models">模型设置</el-menu-item>
        </el-menu>
      </div>

      <div class="side-sep"></div>
      <SessionSidebar />

      <div class="side-foot">
        <el-button type="primary" size="small" style="width: 100%" @click="onNew">
          新建对话
        </el-button>
      </div>
    </el-aside>

    <el-container>
      <el-header class="topbar">
        <span class="dot" :style="{ background: store.connected ? '#67c23a' : '#f56c6c' }"></span>
        <span>{{ store.manualText }}</span>
      </el-header>
      <el-main class="main-pane">
        <router-view />
      </el-main>
    </el-container>
  </el-container>
</template>

<script setup>
import { onMounted } from 'vue'
import { useRoute } from 'vue-router'
import { ElMessage } from 'element-plus'
import SessionSidebar from './components/SessionSidebar.vue'
import { store } from './store'

const route = useRoute()

async function onNew() {
  try {
    await store.newSession()
    ElMessage.success('已新建对话')
  } catch (e) {
    ElMessage.error('新建失败：' + e.message)
  }
}

onMounted(async () => {
  await Promise.all([store.refreshHealth(), store.loadSessions()])
})
</script>
