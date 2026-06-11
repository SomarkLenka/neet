"""NEET question viewer + Claude assistant. Run: python -m viewer.app"""
import json
import queue
import threading
from pathlib import Path

from flask import Flask, Response, abort, jsonify, request, send_from_directory

from . import assistant, bubbles, chats

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EXTRACTED = PROJECT_ROOT / "extracted"
BAKED = PROJECT_ROOT / "data" / "baked"
STATIC = Path(__file__).resolve().parent / "static"

app = Flask(__name__, static_folder=str(STATIC), static_url_path="/static")

_index_cache = {"data": None, "mtime": None}
_index_lock = threading.Lock()
_manifest_cache: dict[str, tuple[float, dict]] = {}


def load_index(reload=False):
    """Cached by file mtime, so papers appear as the (still running)
    pipeline batch finishes them."""
    with _index_lock:
        p = EXTRACTED / "index.json"
        mtime = p.stat().st_mtime if p.exists() else None
        if mtime != _index_cache["mtime"] or reload:
            _index_cache["data"] = json.loads(p.read_text(encoding="utf-8")) if p.exists() else None
            _index_cache["mtime"] = mtime
            _manifest_cache.clear()
        return _index_cache["data"]


def load_manifest(slug: str) -> dict:
    p = EXTRACTED / slug / "manifest.json"
    if not p.exists():
        abort(404, "unknown paper")
    mtime = p.stat().st_mtime
    cached = _manifest_cache.get(slug)
    if cached is None or cached[0] != mtime:
        _manifest_cache[slug] = (mtime, json.loads(p.read_text(encoding="utf-8")))
    return _manifest_cache[slug][1]


def get_question(slug: str, num: int) -> dict:
    q = next((q for q in load_manifest(slug)["questions"] if q["number"] == num), None)
    if not q:
        abort(404, "unknown question")
    return q


@app.get("/")
def home():
    return send_from_directory(STATIC, "index.html")


@app.get("/api/index")
def api_index():
    data = load_index(reload=request.args.get("reload") == "1")
    if data is None:
        return jsonify({"error": "no extracted data - run: python -m pipeline.run_pipeline"}), 503
    return jsonify(data)


@app.get("/api/papers/<slug>/questions/<int:num>")
def api_question(slug, num):
    m = load_manifest(slug)
    q = get_question(slug, num)
    return jsonify({**q, "paper": slug, "paper_title": m["title"]})


@app.get("/img/<slug>/<path:fname>")
def img(slug, fname):
    return send_from_directory(EXTRACTED / slug, fname)


# ---- support bubbles (pre-baked answers) ---------------------------------

_bubbles_cache = {"tree": None, "mtime": None}


def bubble_tree_view():
    """Client-safe bubble tree, cached by file mtime so edits to
    assistant_bubbles.json show up without a restart."""
    p = bubbles.BUBBLES_PATH
    mtime = p.stat().st_mtime if p.exists() else None
    if mtime != _bubbles_cache["mtime"]:
        _bubbles_cache["tree"] = bubbles.client_view()
        _bubbles_cache["mtime"] = mtime
    return _bubbles_cache["tree"]


def baked_doc(slug: str, num: int) -> dict:
    p = BAKED / slug / f"{num}.json"
    if not p.exists():
        return {"paper": slug, "number": num, "nodes": {}}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {"paper": slug, "number": num, "nodes": {}}


@app.get("/api/bubbles")
def api_bubbles():
    try:
        return jsonify({"bubbles": bubble_tree_view()})
    except bubbles.BubblesError as e:
        return jsonify({"error": str(e)}), 500


@app.get("/api/baked/<slug>/<int:num>")
def api_baked(slug, num):
    return jsonify(baked_doc(slug, num))


def generate_and_cache(slug: str, num: int, node: dict) -> tuple[dict | None, str | None]:
    """Generate one bubble node's answer live via claude and cache it into the
    baked file (status 'ready', so the offline bake later skips it). Returns
    (entry, error)."""
    from .bake import question_prompt, save_baked, load_baked
    m = load_manifest(slug)
    q = get_question(slug, num)
    meta = {"number": num, "category": q["category"], "title": m["title"],
            "image_path": str(EXTRACTED / slug / q["image"]), "text": q.get("text", "")}
    res = assistant.collect(question_prompt(meta, node["prompt"]), kind=node["kind"])
    if res.get("error") and not res["text"]:
        return None, res["error"]
    entry = {"kind": node["kind"], "answer": res["text"],
             "status": "ready" if node["kind"] == "direct" else "stub",
             "sources": res.get("sources") or [], "cost_usd": res.get("cost_usd")}
    doc = load_baked(slug, num)
    doc["nodes"][node["id"]] = entry
    save_baked(doc)
    return entry, None


@app.post("/api/baked/<slug>/<int:num>/click")
def api_baked_click(slug, num):
    """Student action: record the clicked bubble in chat history and return its
    answer. A pre-baked answer is served instantly (no LLM). An empty node that
    is marked on_demand (e.g. the Answer button) is generated live and cached
    on first use; other empty nodes just report their status."""
    body = request.get_json(silent=True) or {}
    node_id = (body.get("node_id") or "").strip()
    label = (body.get("label") or "").strip()
    if not node_id:
        return jsonify({"error": "node_id required"}), 400
    node = baked_doc(slug, num)["nodes"].get(node_id)
    chat = chats.load_chat(slug, num)
    chats.append_message(chat, "user", label or node_id, node_id=node_id, bubble=True)

    if not node or node.get("status") == "empty" or not node.get("answer"):
        tree_node = next((n for n in bubbles.flatten() if n["id"] == node_id), None)
        if tree_node and tree_node.get("on_demand"):
            entry, err = generate_and_cache(slug, num, tree_node)
            if err:
                return jsonify({"node_id": node_id, "status": "error", "error": err}), 502
            chats.append_message(chat, "assistant", entry["answer"],
                                 node_id=node_id, sources=entry.get("sources") or [])
            return jsonify({"node_id": node_id, **entry})
        status = node.get("status", "missing") if node else "missing"
        return jsonify({"node_id": node_id, "status": status, "answer": None})

    chats.append_message(chat, "assistant", node["answer"],
                         node_id=node_id, sources=node.get("sources") or [])
    return jsonify({"node_id": node_id, "status": node.get("status", "ready"),
                    "answer": node["answer"], "sources": node.get("sources") or []})


@app.post("/api/baked/<slug>/<int:num>/generate")
def api_baked_generate(slug, num):
    """Admin/preview: generate one node's answer live and cache it. Same path
    the offline bake uses; exposed so prompts can be previewed before a bake."""
    body = request.get_json(silent=True) or {}
    node_id = (body.get("node_id") or "").strip()
    node = next((n for n in bubbles.flatten() if n["id"] == node_id), None)
    if not node:
        return jsonify({"error": f"unknown bubble '{node_id}'"}), 404
    entry, err = generate_and_cache(slug, num, node)
    if err:
        return jsonify({"error": err}), 502
    return jsonify({"node_id": node_id, **entry})


# ---- chat ----------------------------------------------------------------

@app.get("/api/chat/<slug>/<int:num>")
def chat_history(slug, num):
    chat = chats.load_chat(slug, num)
    running = assistant.REGISTRY.running_for(slug, num)
    return jsonify({**chat, "streaming": running.id if running else None})


@app.delete("/api/chat/<slug>/<int:num>")
def chat_reset(slug, num):
    running = assistant.REGISTRY.running_for(slug, num)
    if running:
        running.stop()
    chats.reset_chat(slug, num)
    return jsonify({"ok": True})


@app.post("/api/chat/<slug>/<int:num>/message")
def chat_message(slug, num):
    body = request.get_json(silent=True) or {}
    message = (body.get("message") or "").strip()
    if not message:
        return jsonify({"error": "empty message"}), 400

    m = load_manifest(slug)
    q = get_question(slug, num)
    chat = chats.load_chat(slug, num)

    meta = {
        "number": num, "category": q["category"], "title": m["title"],
        "image_path": str(EXTRACTED / slug / q["image"]),
        "text": q.get("text", ""),
    }
    fresh_prompt = assistant.first_turn_prompt(meta, message)
    prompt = message if chat.get("session_id") else fresh_prompt

    try:
        turn = assistant.REGISTRY.create(slug, num, chat)
    except assistant.AssistantError as e:
        return jsonify({"error": str(e)}), 409

    chats.append_message(chat, "user", message)
    turn.start(prompt, fresh_prompt)
    return jsonify({"stream_id": turn.id})


@app.post("/api/chat/<slug>/<int:num>/stop")
def chat_stop(slug, num):
    running = assistant.REGISTRY.running_for(slug, num)
    if running:
        running.stop()
        return jsonify({"ok": True})
    return jsonify({"ok": False, "error": "nothing streaming"}), 404


@app.get("/api/chat/stream/<stream_id>")
def chat_stream(stream_id):
    turn = assistant.REGISTRY.get(stream_id)
    if not turn:
        abort(404, "unknown stream")

    def gen():
        while True:
            try:
                kind, payload = turn.events.get(timeout=15)
            except queue.Empty:
                if turn.finished.is_set():
                    break
                yield ": ping\n\n"
                continue
            data = json.dumps(payload if isinstance(payload, dict) else {"text": payload},
                              ensure_ascii=False)
            yield f"event: {kind}\ndata: {data}\n\n"
            if kind in ("done", "error"):
                break
        assistant.REGISTRY.discard(turn)

    return Response(gen(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, threaded=True, debug=False)
