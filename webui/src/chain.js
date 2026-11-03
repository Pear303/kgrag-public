// 推理链转换：从后端 path_detail 生成 cytoscape elements。
// 逻辑对齐旧 Gradio 版的 _build_chain_elements（保持同样的关系配色与"[推理链]"树形展开），
// 但改成在前端算，省掉一次 /chain/store 往返，也不再需要 iframe。

const REL_COLOR = {
  故障原因: '#e74c3c', 可能原因: '#e74c3c',
  处理方法: '#27ae60', 解决方法: '#27ae60',
  确认方法: '#2980b9', 检查处理方法: '#2980b9',
  报警含义: '#8e44ad', 报警类别: '#8e44ad',
  表现为: '#f39c12', 包含: '#95a5a6', 配套使用: '#95a5a6',
  推理链: '#34495e',
}
const DEFAULT_COLOR = '#7f8c8d'
const colorOf = (rel) => REL_COLOR[rel] || DEFAULT_COLOR

export function buildChainElements(pathDetail) {
  const nodes = []
  const edges = []
  let n = 0
  const next = (p) => `${p}_${++n}`

  for (const p of pathDetail || []) {
    const seq = (p && p.seq) || []
    if (!seq.length) continue
    const head = String(seq[0].entity ?? '')
    if (head.startsWith('[推理链]')) {
      const body = head.slice('[推理链]'.length).trim()
      const segs = body.split(';').map(s => s.trim()).filter(Boolean)
      if (!segs.length) continue
      const subj = segs[0]
      const subjId = next('subj')
      nodes.push({ data: { id: subjId, label: subj.slice(0, 60), color: '#34495e', size: 42, rel: '主体', text: body } })
      for (const seg of segs.slice(1)) {
        const ci = seg.indexOf(':')
        if (ci < 0) continue
        const relHead = seg.slice(0, ci).trim()
        const itemsStr = seg.slice(ci + 1)
        const m = relHead.match(/^(.+?)\s*(?:\((\d+)\))?$/)
        const relName = m ? m[1].trim() : relHead
        const color = colorOf(relName)
        const relId = next('rel')
        nodes.push({ data: { id: relId, label: relHead, color, size: 32, rel: relName, text: '' } })
        edges.push({ data: { source: subjId, target: relId, label: '', color } })
        for (const raw of itemsStr.split('|')) {
          const item = raw.trim()
          if (!item) continue
          const text = item.replace(/^\d+\.\s*/, '')
          const itId = next('item')
          nodes.push({ data: { id: itId, label: text.slice(0, 38), color, size: 26, rel: relName, text } })
          edges.push({ data: { source: relId, target: itId, label: '', color } })
        }
      }
    } else {
      let prevId = null
      for (const step of seq) {
        const ent = String(step.entity ?? '')
        const rel = step.rel || ''
        const d = step.dir || '?'
        const curId = next('n')
        nodes.push({ data: { id: curId, label: ent.slice(0, 38), color: colorOf(rel), size: 28, rel, text: ent } })
        if (prevId !== null) {
          const arrow = d === '-' ? '◀ ' : '▶ '
          edges.push({ data: { source: prevId, target: curId, label: arrow + rel, color: colorOf(rel) } })
        }
        prevId = curId
      }
    }
  }
  return { nodes, edges }
}

/** 逐跳文本（对齐旧版 _render_path_detail 的展示口径）。 */
export function renderPathDetail(pathDetail) {
  if (!pathDetail || !pathDetail.length) return ''
  const lines = []
  pathDetail.forEach((p, idx) => {
    lines.push(`**路径 ${idx + 1}**：`)
    const seq = p.seq || []
    seq.forEach((step, j) => {
      const ent = step.entity ?? ''
      if (j === 0) lines.push(`  - 起点：\`${ent}\``)
      else lines.push(`  - ${step.dir === '-' ? '◀──' : '──▶'}[${step.rel}]${step.dir === '-' ? '──' : ' →'} \`${ent}\``)
    })
  })
  return lines.join('\n')
}
