"""
Extracts plain text from uploaded files: .docx, .pdf, .txt
"""

import io
from docx import Document
from pypdf import PdfReader


def extract_text_from_docx(file_bytes: bytes) -> str:
    doc = Document(io.BytesIO(file_bytes))
    return "\n".join(para.text for para in doc.paragraphs if para.text.strip())


def extract_text_from_pdf(file_bytes: bytes) -> str:
    reader = PdfReader(io.BytesIO(file_bytes))
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text.strip()


def extract_text_from_txt(file_bytes: bytes) -> str:
    return file_bytes.decode("utf-8", errors="ignore").strip()


def extract_text(filename: str, file_bytes: bytes) -> str:
    """Dispatch based on file extension."""
    ext = filename.rsplit(".", 1)[-1].lower()

    if ext == "docx":
        return extract_text_from_docx(file_bytes)
    elif ext == "pdf":
        return extract_text_from_pdf(file_bytes)
    elif ext == "txt":
        return extract_text_from_txt(file_bytes)
    else:
        raise ValueError(f"Unsupported file type: .{ext}. Supported: .docx, .pdf, .txt")
