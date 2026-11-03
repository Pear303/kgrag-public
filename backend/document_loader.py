# backend/document_loader.py
"""文档格式统一层：pdf/docx/md/txt → 纯文本 md。

职责单一：把上传的任意支持格式转换为统一 UTF-8 纯文本。
不涉及 chunk 化（parser 负责）、不涉及 LLM（extractor 负责）。
"""
import os

SUPPORTED_EXTENSIONS = {".md", ".markdown", ".txt", ".pdf", ".docx"}


def _ext(path: str) -> str:
    return os.path.splitext(path)[1].lower()


def load_document_to_text(path: str) -> str:
    """读取文档为纯文本字符串。"""
    e = _ext(path)
    if e in {".md", ".markdown", ".txt"}:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    if e == ".docx":
        return _load_docx(path)
    if e == ".pdf":
        return _load_pdf(path)
    raise ValueError(f"不支持的文件格式: {e}（仅支持 {sorted(SUPPORTED_EXTENSIONS)}）")


def _load_docx(path: str) -> str:
    import docx
    doc = docx.Document(path)
    parts = []
    for para in doc.paragraphs:
        t = para.text.strip()
        if not t:
            continue
        if para.style.name.startswith("Heading"):
            parts.append(f"# {t}")
        else:
            parts.append(t)
    return "\n\n".join(parts)


def _load_pdf(path: str) -> str:
    import pdfplumber
    parts = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            txt = page.extract_text() or ""
            if txt:
                parts.append(txt)
    return "\n\n".join(parts)


def parse_document_to_md(path: str, out_md_path: str) -> str:
    """解析文档并写出统一 UTF-8 .md 文件，返回输出路径。"""
    text = load_document_to_text(path)
    os.makedirs(os.path.dirname(out_md_path), exist_ok=True)
    with open(out_md_path, "w", encoding="utf-8") as f:
        f.write(text)
    return out_md_path
