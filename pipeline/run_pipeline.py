"""Batch extraction: papers/*.pdf -> extracted/<slug>/q###.png + manifests.

Usage:
    python -m pipeline.run_pipeline [--papers NAME ...] [--debug]

Two passes per paper (a 300dpi page render is ~120MB, so pages are never all
held in memory): pass 1 detects question-start blobs and resolves numbering;
pass 2 re-renders page by page to crop and save each question.
"""
import argparse
import json
import os
import sys
import tempfile
import traceback
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone

import fitz
from PIL import Image, ImageDraw

from . import config, detect, extract, numbering, pdf_pages


_OCR_POOL = ThreadPoolExecutor(max_workers=8)


@dataclass
class Placed:
    pno: int
    side: str
    x_offset: int
    col_width: int
    blob: detect.Blob
    band: int = 0   # section band on the page (reading order restarts per band)


def write_json(path, obj) -> None:
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path), suffix=".tmp")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=1)
    os.replace(tmp, path)


def detect_paper(doc, debug_dir, warnings):
    """Pass 1: blobs + numbering signals for every content page/column."""
    placed: list[Placed] = []
    skipped, page_vals = [], []
    for pno in range(doc.page_count):
        page = doc[pno]
        gray = pdf_pages.render_gray(page)
        keep, reason = pdf_pages.is_content_page(page, gray)
        if not keep:
            skipped.append({"page": pno + 1, "reason": reason})
            continue
        pr = pdf_pages.split_columns(pno, page, gray)
        breaks = pdf_pages.find_section_breaks(gray, pr.divider_px)
        vals: dict[str, set] = {}
        overlays = []
        for col in pr.columns:
            blobs, gutter = detect.find_question_blobs(col.arr)
            # drop marks on a section-header row, e.g. the BIOLOGY box edge
            blobs = [b for b in blobs if not any(b.y0 - 40 <= y <= b.y1 + 40 for y in breaks)]
            anchors = numbering.text_anchors(page, col.x_offset_px, gutter)
            blobs = numbering.attach_anchors(blobs, anchors)
            # tesseract runs as one subprocess per blob: parallelize across threads
            todo = [b for b in blobs if b.text_value is None and not b.is_option]
            for b, (kind, val) in zip(todo, _OCR_POOL.map(
                    lambda b: numbering.ocr_blob(b.crop), todo)):
                if kind == "option":
                    b.is_option = True
                elif kind == "question":
                    b.ocr_value, b.is_question = val, True
                elif kind == "bare" and b.ocr_value is None:
                    b.ocr_value = val
            vals[col.side] = {b.text_value or b.ocr_value for b in blobs} - {None}
            placed += [Placed(pno, col.side, col.x_offset_px, col.arr.shape[1], b,
                              band=sum(1 for y in breaks if b.y0 > y)) for b in blobs]
            overlays.append((col, gutter, blobs))
        page_vals.append(vals)
        if debug_dir:
            _debug_overlay(gray, pr.divider_px, overlays, debug_dir, pno)
    return placed, skipped, page_vals


def _debug_overlay(gray, divider, overlays, debug_dir, pno):
    img = Image.fromarray(gray).convert("RGB")
    d = ImageDraw.Draw(img)
    d.line([(divider, 0), (divider, img.height)], fill=(0, 0, 255), width=5)
    for col, (gx0, gx1), blobs in overlays:
        for b in blobs:
            color = (0, 160, 0) if b.text_value else (220, 0, 0)
            d.rectangle([col.x_offset_px + gx0, b.y0, col.x_offset_px + gx1, b.y1],
                        outline=color, width=4)
    img.resize((img.width // 4, img.height // 4)).save(
        os.path.join(debug_dir, f"page{pno + 1:02d}.jpg"), quality=70)


def crop_paper(doc, placed: list[Placed], out_dir, warnings):
    """Pass 2: crop every numbered blob's region and extract its text."""
    by_page: dict[int, list[Placed]] = {}
    for p in placed:
        by_page.setdefault(p.pno, []).append(p)
    questions = []
    for pno in sorted(by_page):
        page = doc[pno]
        gray = pdf_pages.render_gray(page)
        pr = pdf_pages.split_columns(pno, page, gray)
        cols = {c.side: c for c in pr.columns}
        items = by_page[pno]
        for p in items:
            col = cols[p.side]
            same_col = sorted((q for q in items if q.side == p.side), key=lambda q: q.blob.y0)
            nxt = next((q for q in same_col if q.blob.y0 > p.blob.y0), None)
            y_end = nxt.blob.y0 if nxt else int(col.arr.shape[0] * config.FOOTER_FRAC)
            crop = extract.crop_region(col.arr, p.blob.y0, y_end)
            if crop is None:
                warnings.append(f"q{p.blob.number}: empty crop on page {pno + 1}")
                continue
            fname = f"q{p.blob.number:03d}.png"
            extract.save_png(crop, os.path.join(out_dir, fname))
            text, source = extract.region_text(page, col.x_offset_px, col.arr.shape[1],
                                               p.blob.y0, y_end)
            if source == "none":
                text = extract.tesseract_text(crop)
                source = "tesseract" if text else "none"
            questions.append({
                "number": p.blob.number,
                "category": config.category_for(p.blob.number),
                "image": fname,
                "text": text,
                "text_source": source,
                "number_source": p.blob.source,
                "page": pno + 1,
                "column": p.side,
            })
    return sorted(questions, key=lambda q: q["number"])


def prefilter_marks(placed: list[Placed], warnings) -> list[Placed]:
    """Drop option markers ('N)' / '(N)') and marks not x-aligned with the
    question-number column. Question numbers all share one x position per
    column; option markers sit noticeably to the right of it."""
    n0 = len(placed)
    placed = [p for p in placed if not p.blob.is_option]
    dropped_opt = n0 - len(placed)
    dropped_x = 0
    tol = int(30 * config.ZOOM)  # 30pt: options are outdented ~60pt further right
    for side in ("left", "right"):
        qx = sorted(p.blob.x_ink for p in placed
                    if p.side == side and p.blob.is_question)
        if len(qx) < 3:
            continue
        med = qx[len(qx) // 2]
        for p in placed:
            if p.side == side:
                p.blob.x_dist = abs(p.blob.x_ink - med)
        keep = [p for p in placed if p.side != side or p.blob.is_question
                or p.blob.x_dist <= tol]
        dropped_x += len(placed) - len(keep)
        placed = keep
    if dropped_opt or dropped_x:
        warnings.append(f"filtered {dropped_opt} option marker(s) and "
                        f"{dropped_x} misaligned mark(s) from the gutter")
    return placed


def process_paper(pdf_path, out_root, debug=False):
    name = os.path.basename(pdf_path)
    slug = config.slugify(name)
    out_dir = os.path.join(out_root, slug)
    os.makedirs(out_dir, exist_ok=True)
    debug_dir = None
    if debug:
        debug_dir = os.path.join(out_dir, "debug")
        os.makedirs(debug_dir, exist_ok=True)

    warnings: list[str] = []
    doc = fitz.open(pdf_path)
    placed, skipped, page_vals = detect_paper(doc, debug_dir, warnings)

    bilingual = numbering.is_bilingual(page_vals)
    if bilingual:
        placed = [p for p in placed if p.side == "right"]
    placed = prefilter_marks(placed, warnings)

    placed.sort(key=lambda p: (p.pno, p.band, p.side == "right", p.blob.y0))
    ordered_blobs = [p.blob for p in placed]
    kept = set(map(id, numbering.resolve(ordered_blobs, warnings)))
    placed = [p for p in placed if id(p.blob) in kept and p.blob.number]

    questions = crop_paper(doc, placed, out_dir, warnings)
    doc.close()

    nums = {q["number"] for q in questions}
    missing = [n for n in range(1, (max(nums) if nums else 0) + 1) if n not in nums]
    manifest = {
        "paper": slug,
        "title": config.paper_title(name),
        "source_pdf": f"papers/{name}",
        "bilingual": bilingual,
        "dpi": config.DPI,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "pages_total": len(page_vals) + len(skipped),
        "pages_skipped": skipped,
        "questions": questions,
        "missing": missing,
        "warnings": warnings,
    }
    write_json(os.path.join(out_dir, "manifest.json"), manifest)
    return manifest


def update_index(manifest) -> None:
    """Merge one paper's results into index.json (fail-open: every finished
    paper is immediately visible to the viewer, even mid-batch)."""
    path = os.path.join(config.EXTRACTED_DIR, "index.json")
    try:
        with open(path, encoding="utf-8") as f:
            index = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        index = {"categories": config.CATEGORIES, "papers": [], "questions": []}
    slug = manifest["paper"]
    index["papers"] = [p for p in index["papers"] if p["slug"] != slug] + [{
        "slug": slug, "title": manifest["title"], "bilingual": manifest["bilingual"],
        "question_count": len(manifest["questions"]), "missing": manifest["missing"],
    }]
    index["papers"].sort(key=lambda p: p["title"].lower())
    index["questions"] = [q for q in index["questions"] if q["paper"] != slug] + [{
        "paper": slug, "number": q["number"], "category": q["category"],
        "image": f"{slug}/{q['image']}", "has_text": bool(q["text"]),
    } for q in manifest["questions"]]
    index["generated_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    write_json(path, index)


def update_qa(entry) -> None:
    path = os.path.join(config.EXTRACTED_DIR, "qa_report.json")
    try:
        with open(path, encoding="utf-8") as f:
            qa = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        qa = []
    qa = [e for e in qa if e.get("paper") != entry["paper"]] + [entry]
    write_json(path, qa)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--papers", nargs="*", help="only these PDF file names")
    ap.add_argument("--debug", action="store_true", help="write detection overlay images")
    args = ap.parse_args()

    pdfs = sorted(p for p in config.PAPERS_DIR.iterdir()
                  if p.suffix.lower() == ".pdf" and p.name.lower() not in config.EXCLUDE_NAMES)
    if args.papers:
        want = {w.lower() for w in args.papers}
        pdfs = [p for p in pdfs if p.name.lower() in want]
        if not pdfs:
            sys.exit(f"no papers matched {args.papers}")

    os.makedirs(config.EXTRACTED_DIR, exist_ok=True)
    done = 0
    for pdf in pdfs:
        print(f"=== {pdf.name}", flush=True)
        try:
            m = process_paper(str(pdf), str(config.EXTRACTED_DIR), debug=args.debug)
        except Exception:
            print(traceback.format_exc(), flush=True)
            update_qa({"paper": pdf.name, "error": traceback.format_exc(limit=3)})
            continue
        update_index(m)
        update_qa({
            "paper": pdf.name, "slug": m["paper"], "bilingual": m["bilingual"],
            "found": len(m["questions"]), "missing": m["missing"],
            "low_confidence": sum(1 for q in m["questions"] if q["number_source"] == "inferred"),
            "warnings": m["warnings"], "pages_skipped": m["pages_skipped"],
        })
        done += 1
        print(f"    {len(m['questions'])} questions, {len(m['missing'])} missing, "
              f"bilingual={m['bilingual']}, warnings={len(m['warnings'])}", flush=True)
    print(f"\nbatch complete: {done}/{len(pdfs)} papers indexed", flush=True)


if __name__ == "__main__":
    main()
