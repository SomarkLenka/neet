"""Pipeline constants and shared helpers.

Values marked "validated" were tuned empirically against the papers in papers/
(see docs in run_pipeline.py); change them only with --debug overlay evidence.
"""
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PAPERS_DIR = PROJECT_ROOT / "papers"
EXTRACTED_DIR = PROJECT_ROOT / "extracted"

DPI = 300
ZOOM = DPI / 72.0

# Not an exam paper (user-confirmed exclusion). Case-insensitive name match.
EXCLUDE_NAMES = {"allen_success_mantra_biology-1_ocr.pdf"}

# Subject ranges by question number (NEET 180-question format).
CATEGORY_BOUNDS = [
    (1, 45, "physics"),
    (46, 90, "chemistry"),
    (91, 135, "botany"),
    (136, 180, "zoology"),
]
CATEGORIES = [c for _, _, c in CATEGORY_BOUNDS]
MAX_QUESTION = 180

# --- gutter blob detection (validated at 300dpi) ---
# Question numbers sit in a band left of the body-text indent. The body indent
# is estimated per column from the ink profile; the gutter band extends this
# many PDF points left of it.
GUTTER_BAND_PT = 75
# Digit-line height range in px at 300dpi ("1." rendered ~65px in all samples).
BLOB_H_RANGE = (38, 110)
# Rows closer together than this merge into one blob (anti-aliasing gaps).
BLOB_GAP_PX = 8
# Minimum ink pixels for a blob to count (rejects specks / faint bleed).
BLOB_MIN_INK = 60
# Header/footer exclusion bands as fractions of page height.
HEADER_FRAC = 0.045
FOOTER_FRAC = 0.965

# Ink threshold on grayscale 0-255 (below = ink).
INK_LEVEL = 150

# Pages whose body ink fraction is below this are blank/rough-work pages.
MIN_PAGE_INK_FRAC = 0.005

# Question crop margins (px at 300dpi).
CROP_TOP_MARGIN = 12
CROP_BOTTOM_MARGIN = 14

# If the PDF text layer yields fewer characters than this for a question
# region, re-OCR the crop with Tesseract instead.
MIN_PDF_TEXT_CHARS = 40

TESSERACT_EXE = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


def category_for(num: int) -> str | None:
    for lo, hi, cat in CATEGORY_BOUNDS:
        if lo <= num <= hi:
            return cat
    return None


def slugify(pdf_name: str) -> str:
    """'KOTA RE-NEET MAJOR-02 QN PAPER_ocr.pdf' -> 'kota_re_neet_major_02_qn_paper'."""
    stem = re.sub(r"\.pdf$", "", pdf_name, flags=re.I)
    stem = re.sub(r"_ocr$", "", stem, flags=re.I)
    stem = re.sub(r"[^a-z0-9]+", "_", stem.lower())
    return stem.strip("_")


def paper_title(pdf_name: str) -> str:
    """Human-readable title from the file name."""
    stem = re.sub(r"\.pdf$", "", pdf_name, flags=re.I)
    return re.sub(r"_ocr$", "", stem, flags=re.I).strip()
