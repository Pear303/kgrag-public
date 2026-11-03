# tests/test_document_loader.py
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.document_loader import load_document_to_text, parse_document_to_md


def test_load_md_returns_raw_text(tmp_path):
    p = tmp_path / "a.md"
    p.write_text("# 标题\n\n正文内容", encoding="utf-8")
    assert load_document_to_text(str(p)) == "# 标题\n\n正文内容"


def test_load_txt_returns_raw_text(tmp_path):
    p = tmp_path / "a.txt"
    p.write_text("纯文本", encoding="utf-8")
    assert load_document_to_text(str(p)) == "纯文本"


def test_load_docx_returns_text(tmp_path):
    try:
        import docx
    except ImportError:
        import pytest
        pytest.skip("python-docx 未安装")
    from docx import Document
    p = tmp_path / "a.docx"
    d = Document()
    d.add_paragraph("第一段")
    d.add_heading("标题", level=1)
    d.save(str(p))
    text = load_document_to_text(str(p))
    assert "第一段" in text
    assert "标题" in text


def test_load_pdf_returns_text(tmp_path, monkeypatch):
    # 不依赖真实 PDF 文件：mock 底层 _load_pdf，验证 .pdf 分派与返回值透传
    import backend.document_loader as dl
    monkeypatch.setattr(dl, "_load_pdf", lambda path: "PDF文本内容")
    p = tmp_path / "a.pdf"
    p.write_bytes(b"%PDF-1.4 fake")
    assert load_document_to_text(str(p)) == "PDF文本内容"


def test_unsupported_extension_raises(tmp_path):
    p = tmp_path / "a.xyz"
    p.write_text("x", encoding="utf-8")
    import pytest
    with pytest.raises(ValueError):
        load_document_to_text(str(p))


def test_parse_document_to_md_writes_file(tmp_path):
    p = tmp_path / "a.txt"
    p.write_text("hello", encoding="utf-8")
    out = tmp_path / "out.md"
    result = parse_document_to_md(str(p), str(out))
    assert result == str(out)
    assert out.exists()
    assert out.read_text(encoding="utf-8") == "hello"
