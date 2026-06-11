"""Catalog of the NCERT textbook PDFs in books/, keyed by NCERT code.

Each chapter PDF is named by its code (e.g. books/Phy 12/leph101.pdf), which is
exactly the `code` a neet-rag search hit returns. This lets the viewer serve a
chapter at /books/<code> and the assistant link a citation straight to the page
(/books/<code>#page=<pdf_page>)."""
import threading
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BOOKS_DIR = PROJECT_ROOT / "books"

_lock = threading.Lock()
_catalog: dict[str, Path] | None = None


def catalog() -> dict[str, Path]:
    """code (lowercase) -> absolute PDF path. Built once from the books tree."""
    global _catalog
    with _lock:
        if _catalog is None:
            _catalog = {}
            if BOOKS_DIR.is_dir():
                for pdf in BOOKS_DIR.rglob("*.pdf"):
                    _catalog[pdf.stem.lower()] = pdf
        return _catalog


def resolve(code: str) -> Path | None:
    return catalog().get((code or "").strip().lower())
