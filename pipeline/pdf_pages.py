"""Page rendering, content-page filtering, and column splitting."""
import sys
from dataclasses import dataclass

import fitz
import numpy as np

from . import config

# Reuse the validated divider finder from the original script.
sys.path.insert(0, str(config.PROJECT_ROOT))
from split_columns import find_divider_x  # noqa: E402


@dataclass
class Column:
    side: str            # "left" | "right"
    arr: np.ndarray      # H x W grayscale (uint8)
    x_offset_px: int     # column's x origin in full-page pixels


@dataclass
class PageRender:
    pno: int             # 0-based page index
    page: fitz.Page
    gray: np.ndarray     # full page grayscale
    divider_px: int
    columns: list[Column]


def render_gray(page: fitz.Page) -> np.ndarray:
    pix = page.get_pixmap(matrix=fitz.Matrix(config.ZOOM, config.ZOOM))
    arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    return arr[:, :, :3].mean(axis=2).astype(np.uint8)


def is_content_page(page: fitz.Page, gray: np.ndarray) -> tuple[bool, str]:
    """Return (keep, reason). Skips covers and 'Space for Rough Work' filler."""
    h, w = gray.shape
    body = gray[int(h * 0.08): int(h * 0.95), int(w * 0.05): int(w * 0.95)]
    ink_frac = float((body < config.INK_LEVEL).mean())
    if ink_frac < config.MIN_PAGE_INK_FRAC:
        return False, f"blank page (ink {ink_frac:.4f})"
    text = page.get_text().lower()
    words = text.split()
    if "rough work" in text and len(words) < 40:
        return False, "rough-work page"
    # front-matter: numbered exam instructions would poison question numbering
    if "instructions" in text and any(k in text for k in
                                      ("invigilator", "answer sheet", "test booklet")):
        return False, "instructions page"
    return True, ""


def split_columns(pno: int, page: fitz.Page, gray: np.ndarray) -> PageRender:
    """Split a rendered page at its column divider (the user-specified
    split-first flow: all question detection runs on these halves)."""
    rgb = np.stack([gray] * 3, axis=2)  # find_divider_x expects 3 channels
    div = find_divider_x(rgb)
    pad = 4
    left = Column("left", gray[:, : max(1, div - pad)], 0)
    right = Column("right", gray[:, min(gray.shape[1] - 1, div + pad):], div + pad)
    return PageRender(pno, page, gray, div, [left, right])
