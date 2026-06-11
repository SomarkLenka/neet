"""Support-chat bubble tree: load, validate, flatten, and client-safe view.

The tree lives in assistant_bubbles.json at the project root. Each node:
    {id, label, kind: "direct"|"rag", prompt, followups: [node, ...]}
Node ids are dotted paths ("solution.why_wrong"); they are the stable storage
keys in the baked answer files, so renaming an id orphans its baked answers.
"""
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BUBBLES_PATH = PROJECT_ROOT / "assistant_bubbles.json"
MAX_DEPTH = 3  # guard against an accidentally deep / cyclic authored tree


class BubblesError(Exception):
    pass


def load_tree() -> list[dict]:
    """Return the top-level bubble list, validated. Raises BubblesError."""
    try:
        data = json.loads(BUBBLES_PATH.read_text(encoding="utf-8"))
    except FileNotFoundError as e:
        raise BubblesError(f"{BUBBLES_PATH.name} not found") from e
    except json.JSONDecodeError as e:
        raise BubblesError(f"{BUBBLES_PATH.name} is invalid JSON: {e}") from e
    bubbles = data.get("bubbles")
    if not isinstance(bubbles, list) or not bubbles:
        raise BubblesError("'bubbles' must be a non-empty list")
    seen: set[str] = set()
    for node in bubbles:
        _validate(node, depth=1, seen=seen)
    return bubbles


def _validate(node: dict, depth: int, seen: set[str]) -> None:
    if depth > MAX_DEPTH:
        raise BubblesError(f"bubble tree exceeds MAX_DEPTH={MAX_DEPTH} at '{node.get('id')}'")
    for field in ("id", "label", "kind", "prompt"):
        if not node.get(field):
            raise BubblesError(f"bubble node missing '{field}': {node!r}")
    if node["kind"] not in ("direct", "rag"):
        raise BubblesError(f"bubble '{node['id']}' has unknown kind '{node['kind']}'")
    if node["id"] in seen:
        raise BubblesError(f"duplicate bubble id '{node['id']}'")
    seen.add(node["id"])
    for child in node.get("followups", []) or []:
        _validate(child, depth + 1, seen)


def flatten(tree: list[dict] | None = None) -> list[dict]:
    """Depth-first list of every node (for baking — one answer per node)."""
    tree = load_tree() if tree is None else tree
    out: list[dict] = []

    def walk(node):
        out.append(node)
        for child in node.get("followups", []) or []:
            walk(child)

    for node in tree:
        walk(node)
    return out


def client_view(tree: list[dict] | None = None) -> list[dict]:
    """Tree with prompts stripped — safe to send to the browser. Preserves
    id/label/kind/followups so the UI can branch without ever seeing prompts."""
    tree = load_tree() if tree is None else tree

    def strip(node):
        return {
            "id": node["id"],
            "label": node["label"],
            "kind": node["kind"],
            "followups": [strip(c) for c in node.get("followups", []) or []],
        }

    return [strip(n) for n in tree]
