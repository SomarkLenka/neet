#!/usr/bin/env python3
"""
split_columns.py — turn a 2-column PDF into a clean single-column PDF.

For every source page it finds the central vertical divider, cuts the page into a
LEFT half and a RIGHT half, and writes them as consecutive pages in a new PDF:

    src p1 -> out p1 (p1-left), out p2 (p1-right)
    src p2 -> out p3 (p2-left), out p4 (p2-right)
    ...

Reading order is preserved (left column, then right column, page by page), so a
downstream OCR pass sees an ordinary single column and never crosses the gutter.

Usage:
    python3 split_columns.py input.pdf output.pdf [--dpi 300] [--quality 85]

Notes:
  * Divider is found from pixels (the printed vertical line), with a midpoint
    fallback if no strong line exists — so it also works on whitespace-gutter papers.
  * Output is image-only (no text layer), which is exactly what you want before
    running `ocrmypdf` / Tesseract fresh:  ocrmypdf --force-ocr output.pdf final.pdf
"""
import sys, io, argparse
import fitz                      # PyMuPDF
import numpy as np
from PIL import Image


def find_divider_x(arr):
    """Return the x (in pixels) of the column divider for one rendered page.

    Scans the central band for the vertical column with the most continuous 'ink'
    (the printed divider line). Falls back to the page midpoint if nothing is
    strong enough (e.g. a paper that separates columns with whitespace only).
    """
    h, w, _ = arr.shape
    # 'ink' = clearly non-white: dark OR saturated (catches a coloured rule too)
    ink = ((arr.max(2).astype(int) - arr.min(2).astype(int) > 40) |
           (arr.mean(2) < 120))
    y0, y1 = int(h * 0.15), int(h * 0.90)        # ignore header/footer bands
    best_x, best_cov = w // 2, 0.0
    for x in range(int(w * 0.40), int(w * 0.60)):
        cov = ink[y0:y1, x].mean()               # fraction of height that is ink
        if cov > best_cov:
            best_cov, best_x = cov, x
    # require a convincing line; otherwise assume a whitespace gutter at the middle
    return best_x if best_cov >= 0.45 else w // 2


def split_pdf(in_path, out_path, dpi=300, quality=85, pad=2):
    src = fitz.open(in_path)
    out = fitz.open()
    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)

    for pno in range(src.page_count):
        page = src[pno]
        pix = page.get_pixmap(matrix=mat)
        arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
        if pix.n == 4:                            # drop alpha if present
            arr = arr[:, :, :3]
        full = Image.frombytes("RGB", [pix.width, pix.height], arr[:, :, :3].tobytes())

        gx = find_divider_x(arr[:, :, :3])
        halves = [
            full.crop((0, 0, max(1, gx - pad), pix.height)),          # LEFT
            full.crop((min(pix.width - 1, gx + pad), 0, pix.width, pix.height)),  # RIGHT
        ]

        for half in halves:
            buf = io.BytesIO()
            half.save(buf, format="JPEG", quality=quality)
            buf.seek(0)
            # page size in points so the chosen DPI is preserved
            w_pt = half.width * 72.0 / dpi
            h_pt = half.height * 72.0 / dpi
            newpage = out.new_page(width=w_pt, height=h_pt)
            newpage.insert_image(newpage.rect, stream=buf.read())

    out.save(out_path, deflate=True, garbage=4)
    print(f"{src.page_count} source pages -> {out.page_count} single-column pages")
    print(f"saved: {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("input")
    ap.add_argument("output")
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--quality", type=int, default=85)
    ap.add_argument("--pad", type=int, default=2, help="px trimmed around the divider")
    a = ap.parse_args()
    split_pdf(a.input, a.output, a.dpi, a.quality, a.pad)
