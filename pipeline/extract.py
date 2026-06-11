"""Question cropping and text extraction."""
import fitz
import numpy as np
import pytesseract
from PIL import Image

from . import config


def crop_region(col: np.ndarray, y0: int, y1: int) -> np.ndarray | None:
    """Crop [y0, y1) from a column, trimming trailing whitespace only."""
    h, w = col.shape
    y0 = max(0, y0 - config.CROP_TOP_MARGIN)
    y1 = min(h, y1)
    if y1 - y0 < 20:
        return None
    crop = col[y0:y1]
    ink_rows = np.nonzero((crop < config.INK_LEVEL).sum(axis=1) >= 2)[0]
    if ink_rows.size == 0:
        return None
    return crop[: min(crop.shape[0], ink_rows[-1] + config.CROP_BOTTOM_MARGIN)]


def save_png(crop: np.ndarray, path) -> None:
    Image.fromarray(crop).save(path, format="PNG")


def region_text(page: fitz.Page, x_offset_px: int, col_width_px: int,
                y0_px: int, y1_px: int) -> tuple[str, str]:
    """Question text from the PDF text layer clipped to the region; falls back
    to Tesseract when the layer is empty/thin (some papers' OCR skipped it).
    Returns (text, source)."""
    z = config.ZOOM
    rect = fitz.Rect(x_offset_px / z, max(0, y0_px - config.CROP_TOP_MARGIN) / z,
                     (x_offset_px + col_width_px) / z, y1_px / z)
    text = " ".join(page.get_text(clip=rect).split())
    if len(text) >= config.MIN_PDF_TEXT_CHARS:
        return text, "pdf"
    return "", "none"


def tesseract_text(crop: np.ndarray) -> str:
    try:
        raw = pytesseract.image_to_string(Image.fromarray(crop), config="--psm 6")
        return " ".join(raw.split())
    except Exception:
        return ""
