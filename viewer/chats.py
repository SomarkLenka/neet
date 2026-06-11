"""Per-question chat persistence: data/chats/<paper>/<qnum>.json."""
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

CHATS_DIR = Path(__file__).resolve().parent.parent / "data" / "chats"


def _path(slug: str, num: int) -> Path:
    return CHATS_DIR / slug / f"{num}.json"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def load_chat(slug: str, num: int) -> dict:
    p = _path(slug, num)
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            # corrupted file: rename aside and start fresh rather than 500
            try:
                p.rename(p.with_suffix(".corrupt.json"))
            except OSError:
                pass
    return {"paper": slug, "number": num, "session_id": None,
            "created_at": _now(), "updated_at": _now(), "messages": []}


def save_chat(chat: dict) -> None:
    p = _path(chat["paper"], chat["number"])
    p.parent.mkdir(parents=True, exist_ok=True)
    chat["updated_at"] = _now()
    fd, tmp = tempfile.mkstemp(dir=p.parent, suffix=".tmp")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump(chat, f, ensure_ascii=False, indent=1)
    os.replace(tmp, p)


def append_message(chat: dict, role: str, content: str, **extra) -> None:
    chat["messages"].append({"role": role, "content": content, "ts": _now(), **extra})
    save_chat(chat)


def reset_chat(slug: str, num: int) -> None:
    p = _path(slug, num)
    if p.exists():
        p.unlink()
