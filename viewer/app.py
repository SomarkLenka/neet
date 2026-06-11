"""NEET question viewer + Claude assistant. Run: python -m viewer.app"""
import json
import queue
import threading
from pathlib import Path

from flask import Flask, Response, abort, jsonify, request, send_from_directory

from . import assistant, chats

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EXTRACTED = PROJECT_ROOT / "extracted"
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
