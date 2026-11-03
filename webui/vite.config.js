import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'

// api_server 没有挂 CORSMiddleware，浏览器直连 8001 会被 CORS 拦。
// 开发期统一走 Vite 代理（同源），生产期由 FastAPI 同源提供 dist，两边都不需要改后端配置。
const API = 'http://127.0.0.1:8001'
const proxyPrefixes = [
  '/query', '/manuals', '/tasks', '/history', '/chain', '/health', '/static', '/test',
]

export default defineConfig({
  plugins: [vue()],
  base: './', // 相对路径：挂到 /app 或任意子路径都能加载资源
  server: {
    port: 5173,
    proxy: Object.fromEntries(
      proxyPrefixes.map(p => [p, { target: API, changeOrigin: true }]),
    ),
  },
  build: { outDir: 'dist', emptyOutDir: true },
})
