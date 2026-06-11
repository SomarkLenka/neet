"""Question-start detection: ink blobs in a column's number gutter.

The gutter is the band left of the body-text indent where question numbers
("1.", "46.") are printed. Body text occupies most rows at its indent, while
numbers are sparse — so the indent is found from the per-x ink-row fraction,
and digit-height blobs inside the gutter mark question starts.
"""
from dataclasses import dataclass, field

import numpy as np

from . import config


@dataclass
class Blob:
    y0: int
    y1: int
    ink: int
    crop: np.ndarray            # grayscale crop around the blob (for OCR)
    number: int | None = None   # filled by numbering.resolve
    source: str = "none"        # text | ocr | inferred
    ocr_value: int | None = None
    text_value: int | None = None
    x_ink: int = 0              # column x of the mark's first ink pixel
    is_question: bool = False   # OCR/text read "N." (question style)
    is_option: bool = False     # OCR/text read "N)" or "(N" (option marker)
    x_dist: float = 1e9         # px distance from the question-number column


def estimate_body_indent(col: np.ndarray) -> int:
    """x (px) where dense body text begins in this column.

    Body text occupies a sizable fraction of rows at its left edge; question
    numbers and stray marks occupy only a few. The first 20px are skipped
    (residue of the column-divider line after the split).
    """
    h, w = col.shape
    band = col[int(h * config.HEADER_FRAC * 2): int(h * config.FOOTER_FRAC), :]
    ink = band < config.INK_LEVEL
    frac = ink.mean(axis=0)  # fraction of rows with ink, per x
    frac[:20] = 0
    thresh = max(0.06, 0.35 * float(frac.max()))
    dense = frac > thresh
    run = 0
    for x in range(w):
        run = run + 1 if dense[x] else 0
        if run >= 8:
            return x - run + 1
    return int(w * 0.15)  # fallback: typical indent


def find_question_blobs(col: np.ndarray) -> tuple[list[Blob], tuple[int, int]]:
    """Return (blobs, (gutter_x0, gutter_x1)) for one column image."""
    h, w = col.shape
    indent = estimate_body_indent(col)
    band_px = int(config.GUTTER_BAND_PT * config.ZOOM)
    gx0, gx1 = max(20, indent - band_px), max(21, indent - 4)
    strip = col[:, gx0:gx1] < config.INK_LEVEL

    rows = strip.sum(axis=1)
    on = rows >= 2
    y_top, y_bot = int(h * config.HEADER_FRAC), int(h * config.FOOTER_FRAC)

    blobs: list[Blob] = []
    start = None
    last_end = -10**9
    for y in range(h + 1):
        v = on[y] if y < h else False
        if v and start is None:
            # merge with previous blob across a small gap
            if blobs and y - last_end <= config.BLOB_GAP_PX:
                start = blobs.pop().y0
            else:
                start = y
        elif not v and start is not None:
            blobs.append(Blob(start, y, 0, None))
            last_end = y
            start = None

    keep: list[Blob] = []
    lo, hi = config.BLOB_H_RANGE
    for b in blobs:
        hgt = b.y1 - b.y0
        if not (lo <= hgt <= hi):
            continue
        if b.y0 < y_top or b.y1 > y_bot:
            continue
        b.ink = int(strip[b.y0:b.y1].sum())
        if b.ink < config.BLOB_MIN_INK:
            continue
        xs = np.nonzero(strip[b.y0:b.y1].any(axis=0))[0]
        if xs[-1] - xs[0] > 0.85 * (gx1 - gx0):
            continue  # ink spans the whole gutter: a rule/header line, not a number
        b.x_ink = gx0 + int(xs[0])
        m = 10
        b.crop = col[max(0, b.y0 - m): b.y1 + m, max(0, gx0 - m): gx1 + m].copy()
        keep.append(b)
    return keep, (gx0, gx1)
