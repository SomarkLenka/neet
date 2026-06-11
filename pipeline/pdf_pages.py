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


def find_section_breaks(gray: np.ndarray, divider: int) -> list[int]:
    """y positions of mid-page subject headers (PHYSICS / CHEMISTRY / BOTANY /
    ZOOLOGY) that split a page into reading bands: a centered header makes the
    columns restart left->right below it.

    Restricted to the page's middle band (15%-85%): the masthead/logo rules at
    the top and the page-number rule at the bottom also cross the divider but
    are page furniture, not section transitions. Clusters of nearby crossings
    (a header's text rows plus its box border) collapse to one break.
    """
    h, w = gray.shape
    x0, x1 = max(0, divider - 150), min(w, divider + 150)
    # strict darkness + width: section headers are broad solid-black bands at
    # the divider; watermarks are grey and divider rules are only a few px wide
    window = gray[:, x0:x1] < 100
    width = window.sum(axis=1)
    crossing = width > 120
    raw, start = [], None
    for y in range(int(h * 0.15), int(h * 0.85)):
        if crossing[y] and start is None:
            start = y
        elif not crossing[y] and start is not None:
            if 10 <= y - start <= 250:
                raw.append((start + y) // 2)
            start = None
    # collapse crossings within ~400px (one header spans several text rows)
    breaks: list[int] = []
    for y in raw:
        if breaks and y - breaks[-1] <= 400:
            continue
        breaks.append(y)
    return breaks


def split_columns(pno: int, page: fitz.Page, gray: np.ndarray) -> PageRender:
    """Split a rendered page at its column divider (the user-specified
    split-first flow: all question detection runs on these halves)."""
    rgb = np.stack([gray] * 3, axis=2)  # find_divider_x expects 3 channels
    div = find_divider_x(rgb)
    pad = 4
    left = Column("left", gray[:, : max(1, div - pad)], 0)
    right = Column("right", gray[:, min(gray.shape[1] - 1, div + pad):], div + pad)
    return PageRender(pno, page, gray, div, [left, right])
