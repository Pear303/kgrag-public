<template>
  <div>
    <div ref="host" class="chain-graph" :style="{ height: height + 'px' }"></div>
    <div class="chain-detail">
      <template v-if="detail">
        <span class="lb">{{ detail.label }}</span>
        <div v-if="detail.text && detail.text !== detail.label">{{ detail.text }}</div>
        <div v-for="(h, i) in detail.evidence" :key="i" class="chain-evidence">…{{ h }}…</div>
      </template>
      <template v-else>
        <span class="lb">点节点查看实体与所在上下文原文</span>
      </template>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted, onBeforeUnmount, watch, nextTick } from 'vue'
import cytoscape from 'cytoscape'
import dagre from 'cytoscape-dagre'

cytoscape.use(dagre)

const props = defineProps({
  elements: { type: Object, default: () => ({ nodes: [], edges: [] }) },
  height: { type: Number, default: 420 },
  // 送进模型的上下文全文（trace.contexts）：点节点时现场子串检索，展示原文出处
  contexts: { type: Array, default: () => [] },
})

const host = ref(null)
const detail = ref(null)
let cy = null

function findEvidence(term) {
  const t = String(term || '').trim()
  if (!t || t.length < 2) return []
  const out = []
  for (const c of props.contexts || []) {
    const s = String(c || '')
    const i = s.indexOf(t)
    if (i < 0) continue
    const start = Math.max(0, i - 60)
    out.push(s.slice(start, i + t.length + 120).replace(/\s+/g, ' '))
    if (out.length >= 3) break
  }
  return out
}

const STYLE = [
  {
    selector: 'node',
    style: {
      label: 'data(label)',
      'font-size': '12px',
      'text-wrap': 'wrap',
      'text-max-width': '130px',
      'text-valign': 'center',
      'text-halign': 'center',
      'background-color': 'data(color)',
      'border-width': 2,
      'border-color': '#fff',
      color: '#fff',
      'text-outline-width': 2,
      'text-outline-color': 'data(color)',
      width: 'mapData(size, 20, 44, 28, 72)',
      height: 'mapData(size, 20, 44, 28, 72)',
    },
  },
  {
    selector: 'edge',
    style: {
      width: 2,
      'line-color': 'data(color)',
      'target-arrow-shape': 'triangle',
      'target-arrow-color': 'data(color)',
      'curve-style': 'bezier',
      label: 'data(label)',
      'font-size': '10px',
      color: '#555',
      'text-background-color': '#fff',
      'text-background-opacity': 0.85,
      'text-background-padding': '2px',
    },
  },
]

function render() {
  if (!host.value) return
  if (cy) { cy.destroy(); cy = null }
  detail.value = null
  const els = [...(props.elements?.nodes || []), ...(props.elements?.edges || [])]
  if (!els.length) return
  cy = cytoscape({
    container: host.value,
    elements: els,
    style: STYLE,
    // 不设 wheelSensitivity：cytoscape 明确警告自定义该值会让主流鼠标缩放异常
    minZoom: 0.05,
    maxZoom: 5,
    layout: { name: 'dagre', rankDir: 'LR', nodeSep: 22, rankSep: 60, padding: 16, animate: false },
  })
  cy.on('tap', 'node', (e) => {
    const d = e.target.data()
    detail.value = { label: d.label, text: d.text, evidence: findEvidence(d.text || d.label) }
  })
  cy.on('tap', (e) => { if (e.target === cy) detail.value = null })
  setTimeout(() => { if (cy) { cy.resize(); cy.fit(undefined, 24) } }, 60)
}

onMounted(() => nextTick(render))
watch(() => props.elements, () => nextTick(render), { deep: true })
onBeforeUnmount(() => { if (cy) { cy.destroy(); cy = null } })
</script>
