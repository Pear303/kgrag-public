import { createApp } from 'vue'
import { createRouter, createWebHashHistory } from 'vue-router'
import ElementPlus from 'element-plus'
import 'element-plus/dist/index.css'
import zhCn from 'element-plus/es/locale/lang/zh-cn'

import App from './App.vue'
import ChatView from './views/ChatView.vue'
import ManualsView from './views/ManualsView.vue'
import SystemView from './views/SystemView.vue'
import ModelsView from './views/ModelsView.vue'
import './style.css'

const router = createRouter({
  // hash 路由：挂到任意子路径（/app、/ui…）都不需要服务端重写规则
  history: createWebHashHistory(),
  routes: [
    { path: '/', name: 'chat', component: ChatView },
    { path: '/manuals', name: 'manuals', component: ManualsView },
    { path: '/system', name: 'system', component: SystemView },
    { path: '/models', name: 'models', component: ModelsView },
  ],
})

createApp(App).use(router).use(ElementPlus, { locale: zhCn }).mount('#app')
