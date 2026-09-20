"""Turn an uploaded file into plain text.

Extraction happens in memory and the bytes are dropped as soon as the text is
out — nothing here writes to disk.
"""

from __future__ import annotations

import io
import re


class ExtractionError(Exception):
    pass


def _clean(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace(" ", " ").replace("•", "- ").replace("●", "- ")
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def from_pdf(data: bytes) -> str:
    try:
        from pypdf import PdfReader
    except ImportError as e:
        raise ExtractionError("PDF support needs pypdf installed on the server.") from e
    try:
        reader = PdfReader(io.BytesIO(data))
        pages = [(p.extract_text() or "") for p in reader.pages]
    except Exception as e:
        raise ExtractionError(f"Could not read that PDF ({type(e).__name__}).") from e
    text = _clean("\n\n".join(pages))
    if len(text) < 120:
        # Almost certainly a scan or an image-only export. Worth saying so
        # precisely, because it is also exactly why an ATS would reject it.
        raise ExtractionError(
            "That PDF has almost no extractable text — it looks like a scan or an "
            "image-only export. An ATS would read it the same way, as blank. "
            "Export a text-based PDF, or paste the text instead."
        )
    return text


def from_docx(data: bytes) -> str:
    try:
        import docx
    except ImportError as e:
        raise ExtractionError("DOCX support needs python-docx installed on the server.") from e
    try:
        d = docx.Document(io.BytesIO(data))
    except Exception as e:
        raise ExtractionError(f"Could not read that Word file ({type(e).__name__}).") from e
    parts = [p.text for p in d.paragraphs]
    for table in d.tables:                      # CVs in tables are common and ATS-hostile
        for row in table.rows:
            cells = [c.text.strip() for c in row.cells if c.text.strip()]
            if cells:
                parts.append("\t".join(cells))
    return _clean("\n".join(parts))


def from_upload(filename: str, data: bytes) -> tuple[str, str]:
    """Returns (text, source) where source is pdf | docx | text.

    The source is carried through to the ATS audit, which judges a real PDF
    differently from pasted text — it can only assess the formatting a parser
    would choke on if it knows a file was involved at all.
    """
    name = (filename or "").lower()
    if name.endswith(".pdf") or data[:5] == b"%PDF-":
        return from_pdf(data), "pdf"
    if name.endswith(".docx") or data[:2] == b"PK":
        return from_docx(data), "docx"
    if name.endswith(".doc"):
        raise ExtractionError("Legacy .doc isn't supported — save as PDF or .docx, or paste the text.")
    try:
        return _clean(data.decode("utf-8")), "text"
    except UnicodeDecodeError:
        try:
            return _clean(data.decode("latin-1")), "text"
        except Exception as e:
            raise ExtractionError("Couldn't read that file as text.") from e
