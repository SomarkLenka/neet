"""Question-number resolution.

Three signals, in decreasing trust:
  1. PDF text-layer number words found in the gutter band ("text")
  2. Tesseract digit OCR of each detected gutter blob ("ocr")
  3. Strict sequential numbering 1..180 across columns and pages ("inferred")

Anchored blobs are filtered to a consistent increasing chain (weighted LIS,
text anchors outweigh OCR); the gaps between anchors are filled sequentially.
Disagreements and gaps are reported, never raised.
"""
import re

import fitz
import numpy as np
import pytesseract
from PIL import Image

from . import config
from .detect import Blob

pytesseract.pytesseract.tesseract_cmd = config.TESSERACT_EXE

_NUM_WORD = re.compile(r"^\(?(\d{1,3})([.,)]?)$")


def classify_token(txt: str) -> tuple[str | None, int | None]:
    """Question numbers are always 'N.'; option markers are 'N)' or '(N)'.
    Returns (kind, value) with kind in question|option|bare."""
    m = _NUM_WORD.match(txt.strip())
    if not m:
        return None, None
    n = int(m.group(1))
    if not (1 <= n <= config.MAX_QUESTION):
        return None, None
    if txt.lstrip().startswith("(") or m.group(2) == ")":
        return "option", n
    if m.group(2) in (".", ","):
        return "question", n
    return "bare", n


def ocr_blob(crop: np.ndarray | None) -> tuple[str | None, int | None]:
    if crop is None or crop.size == 0:
        return None, None
    base = Image.fromarray(crop).convert("L")
    base = base.resize((base.width * 3, base.height * 3), Image.LANCZOS)
    # hard-binarized variant strips light watermarks behind the digits
    binar = base.point(lambda v: 0 if v < 110 else 255)
    for img in (base, binar):
        for psm in (7, 8, 6):
            try:
                txt = pytesseract.image_to_string(
                    img, config=f"--psm {psm} -c tessedit_char_whitelist=0123456789.()"
                ).strip()
            except Exception:
                return None, None
            kind, val = classify_token(txt)
            if kind:
                return kind, val
    return None, None


def text_anchors(page: fitz.Page, x_offset_px: int, gutter: tuple[int, int]) -> list[tuple[str, int, float, float, float]]:
    """Number words from the PDF text layer inside this column's gutter band.

    Returns [(kind, value, y0_px, y1_px, x0_px)] in column pixel coordinates.
    """
    gx0, gx1 = gutter
    out = []
    for w in page.get_text("words"):
        kind, n = classify_token(w[4])
        if kind is None:
            continue
        x0_px = w[0] * config.ZOOM - x_offset_px
        if gx0 - 12 <= x0_px <= gx1 + 12:
            out.append((kind, n, w[1] * config.ZOOM, w[3] * config.ZOOM, x0_px))
    return out


def attach_anchors(blobs: list[Blob], anchors: list) -> list[Blob]:
    """Match text-layer number words to blobs by y overlap.

    'N.' words are strong question anchors (unmatched ones become synthetic
    blobs — the text layer saw a number where ink detection missed). 'N)'/'(N'
    words mark the blob as an option marker. Bare digits are weak: used only
    as an OCR-grade hint on an existing blob.
    """
    blobs = sorted(blobs, key=lambda b: b.y0)
    for kind, n, ay0, ay1, ax0 in anchors:
        ac = (ay0 + ay1) / 2
        hit = next((b for b in blobs if b.y0 - 20 <= ac <= b.y1 + 20), None)
        if kind == "question":
            if hit is not None:
                hit.text_value = n
                hit.is_question = True
            else:
                blobs.append(Blob(int(ay0), int(ay1), 0, None, text_value=n,
                                  x_ink=int(ax0), is_question=True))
        elif hit is not None:
            if kind == "option":
                hit.is_option = True
            elif hit.ocr_value is None:
                hit.ocr_value = n
    return sorted(blobs, key=lambda b: b.y0)


def is_bilingual(page_column_values: list[dict[str, set[int]]]) -> bool:
    """True when left/right columns repeat the same numbers (Hindi | English)."""
    dup = tot = 0
    for cols in page_column_values:
        left, right = cols.get("left", set()), cols.get("right", set())
        if len(left) >= 2 and len(right) >= 2:
            tot += 1
            if len(left & right) / min(len(left), len(right)) >= 0.5:
                dup += 1
    return tot >= 2 and dup / tot > 0.5


def _weighted_lis(items: list[tuple[int, int, int]]) -> list[int]:
    """items: (position_index, value, weight). Returns indices into `items`
    forming the max-weight strictly-increasing-by-value chain."""
    n = len(items)
    best = [0] * n
    prev = [-1] * n
    for i in range(n):
        best[i] = items[i][2]
        for j in range(i):
            if items[j][1] < items[i][1] and best[j] + items[i][2] > best[i]:
                best[i] = best[j] + items[i][2]
                prev[i] = j
    if not n:
        return []
    i = max(range(n), key=lambda k: best[k])
    chain = []
    while i != -1:
        chain.append(i)
        i = prev[i]
    return chain[::-1]


def resolve(blobs: list[Blob], warnings: list[str]) -> list[Blob]:
    """Assign question numbers to reading-ordered blobs. Returns blobs that
    received a number (extras dropped); mutates blob.number/source."""
    anchored = []
    for idx, b in enumerate(blobs):
        if b.text_value is not None:
            anchored.append((idx, b.text_value, 3))
        elif b.ocr_value is not None:
            anchored.append((idx, b.ocr_value, 2 if b.is_question else 1))
    chain = _weighted_lis(anchored)
    anchors = [(anchored[k][0], anchored[k][1]) for k in chain]

    dropped_anchor_blobs = {anchored[k][0] for k in range(len(anchored))} - {a[0] for a in anchors}
    if dropped_anchor_blobs:
        warnings.append(f"{len(dropped_anchor_blobs)} number reading(s) contradicted the sequence and were ignored")

    if not anchors:
        warnings.append("no readable question numbers anywhere; numbering assumed sequential from 1")
        for i, b in enumerate(blobs):
            if i >= config.MAX_QUESTION:
                break
            b.number, b.source = i + 1, "inferred"
        return [b for b in blobs if b.number]

    def set_num(b: Blob, n: int, _anchor: bool = False):
        b.number = n
        b.source = "text" if b.text_value == n else ("ocr" if b.ocr_value == n else "inferred")

    # segment before the first anchor: count backwards
    first_idx, first_val = anchors[0]
    for off, i in enumerate(range(first_idx - 1, -1, -1), start=1):
        n = first_val - off
        if n < 1:
            warnings.append(f"{first_idx - (first_val - 1)} leading blob(s) before question 1 dropped")
            break
        set_num(blobs[i], n, False)

    for (i0, v0), (i1, v1) in zip(anchors, anchors[1:]):
        set_num(blobs[i0], v0, True)
        between = list(range(i0 + 1, i1))
        gap = v1 - v0 - 1
        if len(between) > gap:
            # false positives among them: prefer question-style marks, then
            # marks x-aligned with the number column (header/body fragments
            # are often inkier than a real digit, so ink ranks last)
            ranked = sorted(between, key=lambda k: (
                not blobs[k].is_question, blobs[k].x_dist, -blobs[k].ink))
            keep = sorted(ranked[:gap])
            warnings.append(f"dropped {len(between) - gap} spurious mark(s) between q{v0} and q{v1}")
            between = keep
        if len(between) < gap:
            warnings.append(f"{gap - len(between)} question(s) between q{v0} and q{v1} not found")
        for off, k in enumerate(between, start=1):
            set_num(blobs[k], v0 + off, False)
    set_num(blobs[anchors[-1][0]], anchors[-1][1], True)

    # segment after the last anchor
    last_idx, last_val = anchors[-1]
    n = last_val
    for i in range(last_idx + 1, len(blobs)):
        n += 1
        if n > config.MAX_QUESTION:
            warnings.append(f"{len(blobs) - i} trailing blob(s) beyond question {config.MAX_QUESTION} dropped")
            break
        set_num(blobs[i], n, False)

    out, seen = [], set()
    for b in blobs:
        if b.number and b.number not in seen:
            seen.add(b.number)
            out.append(b)
    return out
