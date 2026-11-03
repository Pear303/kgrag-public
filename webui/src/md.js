import { marked } from 'marked'
import DOMPurify from 'dompurify'

marked.setOptions({ gfm: true, breaks: true })

/** Markdown → 安全 HTML。答案/路径来自本项目后端，仍做一次 XSS 清洗。 */
export function md(text) {
  if (!text) return ''
  return DOMPurify.sanitize(marked.parse(String(text)))
}
