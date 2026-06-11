"""Pre-bake support-bubble answers into local files for offline serving.

    python -m viewer.bake [--papers SLUG ...] [--questions N ...]
                          [--generate] [--force] [--workers K]

Default pass writes the STRUCTURE only: data/baked/<slug>/<qnum>.json with every
bubble-tree node present, status "empty", answer null — NO LLM calls. This is
what ships until prompts (and the RAG source) are finalized.

--generate (built, run later, intentionally not part of the default flow) fills
empty nodes by calling claude once per node via assistant.collect(). Resumable:
already-answered nodes are skipped unless --force.
"""
import argparse
import json
import os
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

from . import assistant, bubbles

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EXTRACTED = PROJECT_ROOT / "extracted"
BAKED_DIR = PROJECT_ROOT / "data" / "baked"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def baked_path(slug: str, num: int) -> Path:
    return BAKED_DIR / slug / f"{num}.json"


def load_baked(slug: str, num: int) -> dict:
    p = baked_path(slug, num)
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass
    return {"paper": slug, "number": num, "generated_at": None,
            "model": None, "nodes": {}}


def save_baked(doc: dict) -> None:
    p = baked_path(doc["paper"], doc["number"])
    p.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=p.parent, suffix=".tmp")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump(doc, f, ensure_ascii=False, indent=1)
    os.replace(tmp, p)


def ensure_structure(doc: dict, nodes: list[dict]) -> bool:
    """Add any missing node with status 'empty'. Returns True if changed."""
    changed = False
    for node in nodes:
        if node["id"] not in doc["nodes"]:
            entry = {"kind": node["kind"], "status": "empty", "answer": None}
            if node["kind"] == "rag":
                entry["sources"] = []
            doc["nodes"][node["id"]] = entry
            changed = True
    return changed


def question_prompt(meta: dict, node_prompt: str) -> str:
    """Self-contained prompt: full question context + this node's ask."""
    return (
        f"This is question {meta['number']} ({meta['category']}) from the NEET paper "
        f"\"{meta['title']}\".\n\n"
        f"Read the question image at this absolute path using your Read tool:\n"
        f"{meta['image_path']}\n\n"
        f"Extracted OCR text (may contain errors; the image is authoritative):\n"
        f"---\n{meta.get('text') or '(no text extracted)'}\n---\n\n"
        f"Task: {node_prompt}"
    )


def generate_node(meta: dict, node: dict, cfg: dict) -> dict:
    """Call claude once for one node. Returns a node entry dict."""
    prompt = question_prompt(meta, node["prompt"])
    res = assistant.collect(prompt, kind=node["kind"], cfg=cfg)
    entry = {"kind": node["kind"], "answer": res["text"] or None,
             "generated_at": _now(), "cost_usd": res.get("cost_usd")}
    if node["kind"] == "rag":
        entry["sources"] = res.get("sources") or []
        # no MCP configured -> the model could not retrieve: mark as stub
        entry["status"] = "stub" if not cfg.get("mcp_config") else (
            "ready" if res["text"] else "error")
    else:
        entry["status"] = "ready" if res["text"] else "error"
    if res.get("error"):
        entry["error"] = res["error"]
    return entry


def iter_questions(index: dict, papers: set[str] | None, qnums: set[int] | None):
    by_paper: dict[str, list[int]] = {}
    for q in index["questions"]:
        if papers and q["paper"] not in papers:
            continue
        if qnums and q["number"] not in qnums:
            continue
        by_paper.setdefault(q["paper"], []).append(q["number"])
    return by_paper


def manifest_for(slug: str) -> dict:
    return json.loads((EXTRACTED / slug / "manifest.json").read_text(encoding="utf-8"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--papers", nargs="*", help="only these paper slugs")
    ap.add_argument("--questions", nargs="*", type=int, help="only these question numbers")
    ap.add_argument("--generate", action="store_true",
                    help="fill empty nodes via claude (LLM calls); default writes structure only")
    ap.add_argument("--force", action="store_true", help="re-generate already-answered nodes")
    ap.add_argument("--workers", type=int, default=4, help="parallel claude calls when --generate")
    args = ap.parse_args()

    index_path = EXTRACTED / "index.json"
    if not index_path.exists():
        sys.exit("no extracted/index.json - run the pipeline first")
    index = json.loads(index_path.read_text(encoding="utf-8"))

    nodes = bubbles.flatten()
    node_by_id = {n["id"]: n for n in nodes}
    by_paper = iter_questions(index, set(args.papers or []) or None,
                              set(args.questions or []) or None)
    cfg = assistant.load_config() if args.generate else {}

    total_q = sum(len(v) for v in by_paper.values())
    print(f"{total_q} questions x {len(nodes)} nodes "
          f"({'generate' if args.generate else 'structure-only'})", flush=True)

    structure_made = filled = skipped = errored = 0
    for slug, qnums in by_paper.items():
        manifest = manifest_for(slug) if args.generate else None
        mq = {q["number"]: q for q in manifest["questions"]} if manifest else {}
        for num in sorted(qnums):
            doc = load_baked(slug, num)
            if ensure_structure(doc, nodes):
                structure_made += 1

            if args.generate:
                q = mq.get(num)
                meta = {
                    "number": num, "category": q["category"], "title": manifest["title"],
                    "image_path": str(EXTRACTED / slug / q["image"]), "text": q.get("text", ""),
                }
                todo = [nid for nid, e in doc["nodes"].items()
                        if args.force or e.get("status") in ("empty", "error")]
                if todo:
                    with ThreadPoolExecutor(max_workers=args.workers) as pool:
                        results = pool.map(
                            lambda nid: (nid, generate_node(meta, node_by_id[nid], cfg)), todo)
                        for nid, entry in results:
                            doc["nodes"][nid] = entry
                            if entry["status"] == "error":
                                errored += 1
                            else:
                                filled += 1
                    doc["generated_at"] = _now()
                    doc["model"] = cfg.get("model")
                else:
                    skipped += 1
            save_baked(doc)
        print(f"  {slug}: {len(qnums)} questions", flush=True)

    print(f"\nstructure written/updated: {structure_made} question files", flush=True)
    if args.generate:
        print(f"nodes filled: {filled}, errored: {errored}, "
              f"questions skipped (already done): {skipped}", flush=True)


if __name__ == "__main__":
    main()
