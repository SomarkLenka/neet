"""Claude CLI subprocess manager.

Each chat turn runs `claude -p --output-format stream-json`; the first turn for
a question feeds the question image + OCR text, follow-ups use --resume with
the stored session id. Events are pushed onto a per-turn queue that the SSE
route drains. assistant_config.json is reloaded on every turn, so model, MCP
servers (--mcp-config) and allowed tools can change without a restart.
"""
import json
import os
import queue
import shutil
import subprocess
import threading
import time
import uuid
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

from . import chats

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "assistant_config.json"
DEBUG_LOG = PROJECT_ROOT / "data" / "assistant.log"
_log_lock = threading.Lock()


def debug_log(event: str, **fields) -> None:
    """Append a JSON line to data/assistant.log (every claude invocation,
    every lifecycle event, full error context). Never raises."""
    rec = {"ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
           "event": event, **fields}
    try:
        DEBUG_LOG.parent.mkdir(parents=True, exist_ok=True)
        with _log_lock, open(DEBUG_LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except OSError:
        pass

# Effective defaults. The shipped assistant_config.json restates these
# explicitly so every knob is user-editable in one place; this dict is only a
# fallback for keys a hand-edited config happens to omit.
DEFAULT_CONFIG = {
    "claude_command": None,    # full argv prefix to invoke claude, e.g. ["claude"]
    "claude_path": None,       # path to a claude executable (if claude_command unset)
    "model": None,
    "allowed_tools": ["Read"],
    "system_prompt": None,             # replaces the default system prompt (--system-prompt)
    "append_system_prompt": None,      # or append instead (--append-system-prompt)
    "exclude_dynamic_system_prompt_sections": False,  # strip per-machine prompt sections
    "mcp_config": None,                # --mcp-config <file>
    "strict_mcp_config": False,        # ignore all other MCP config (--strict-mcp-config)
    "settings": None,                  # isolated settings file (--settings <file>)
    "config_dir": None,                # isolated CLAUDE_CONFIG_DIR (no system hooks/skills/plugins)
    "extra_args": [],
    "timeout_seconds": 300,
}


class AssistantError(Exception):
    pass


def load_config() -> dict:
    cfg = dict(DEFAULT_CONFIG)
    try:
        cfg.update(json.loads(CONFIG_PATH.read_text(encoding="utf-8")))
    except FileNotFoundError:
        pass
    except json.JSONDecodeError as e:
        raise AssistantError(f"assistant_config.json is invalid JSON: {e}")
    return cfg


def subprocess_env(cfg: dict) -> dict:
    """Environment for the claude subprocess. When config_dir is set, point
    CLAUDE_CONFIG_DIR at an isolated home that carries the live credentials but
    none of the system Claude's hooks/skills/plugins/MCP — fully isolating this
    Claude from the developer's environment. Credentials are re-copied each
    spawn so a token refresh in the real home propagates."""
    env = os.environ.copy()
    iso = cfg.get("config_dir")
    if not iso:
        return env
    iso_path = Path(iso) if os.path.isabs(iso) else (PROJECT_ROOT / iso)
    iso_path.mkdir(parents=True, exist_ok=True)
    real = Path(os.environ.get("CLAUDE_CONFIG_DIR") or (Path.home() / ".claude"))
    try:
        cred = real / ".credentials.json"
        if cred.exists():
            shutil.copy2(cred, iso_path / ".credentials.json")
        settings_src = PROJECT_ROOT / "claude_settings.json"
        if settings_src.exists():
            shutil.copy2(settings_src, iso_path / "settings.json")
    except OSError as e:
        debug_log("config_dir_setup_failed", error=str(e), dir=str(iso_path))
    env["CLAUDE_CONFIG_DIR"] = str(iso_path)
    return env


def resolve_claude(cfg: dict) -> list[str]:
    """Base argv to invoke the claude CLI. `claude_command` (a list) overrides
    everything — use it to wrap the CLI (e.g. ["wsl","claude"] or
    ["node","C:/path/cli.js"]). Otherwise fall back to `claude_path` or PATH."""
    cmd = cfg.get("claude_command")
    if cmd:
        return list(cmd) if isinstance(cmd, (list, tuple)) else [str(cmd)]
    exe = cfg.get("claude_path") or shutil.which("claude")
    if not exe:
        raise AssistantError(
            "claude CLI not found. Install it, set claude_path, or set claude_command "
            "in assistant_config.json.")
    if exe.lower().endswith((".cmd", ".bat")):
        return ["cmd", "/c", exe]
    return [exe]


def build_command(cfg: dict, session_id: str | None, *, stream: bool = True,
                  with_mcp: bool = True) -> list[str]:
    cmd = resolve_claude(cfg) + ["-p", "--output-format", "stream-json", "--verbose"]
    if stream:
        cmd += ["--include-partial-messages"]
    if cfg.get("model"):
        cmd += ["--model", cfg["model"]]
    if cfg.get("allowed_tools"):
        cmd += ["--allowedTools", ",".join(cfg["allowed_tools"])]
    if cfg.get("system_prompt"):
        cmd += ["--system-prompt", cfg["system_prompt"]]
    if cfg.get("append_system_prompt"):
        cmd += ["--append-system-prompt", cfg["append_system_prompt"]]
    if cfg.get("exclude_dynamic_system_prompt_sections"):
        cmd += ["--exclude-dynamic-system-prompt-sections"]
    if cfg.get("settings"):
        cmd += ["--settings", cfg["settings"]]
    if with_mcp and cfg.get("mcp_config"):
        cmd += ["--mcp-config", cfg["mcp_config"]]
        if cfg.get("strict_mcp_config"):
            cmd += ["--strict-mcp-config"]
    if session_id:
        cmd += ["--resume", session_id]
    cmd += cfg.get("extra_args") or []
    return cmd


def collect(prompt: str, *, kind: str = "direct", cfg: dict | None = None,
            timeout: int | None = None) -> dict:
    """One-shot, non-streaming claude run for the baker. Returns
    {"text", "sources", "cost_usd", "session_id", "error"}.

    `kind == "rag"` keeps the configured MCP/tools so the model can retrieve
    reference material; with no mcp_config set it simply answers without it
    (the caller marks such answers as stubs). Never raises — failures come
    back in "error" so a batch bake never dies on one bad question."""
    if cfg is None:
        try:
            cfg = load_config()
        except AssistantError as e:
            return {"text": "", "sources": [], "cost_usd": None,
                    "session_id": None, "error": str(e)}
    try:
        # only RAG nodes need the (slow-to-start) neet-rag MCP server attached;
        # direct reasoning nodes skip it entirely to avoid per-spawn cold-start
        cmd = build_command(cfg, None, stream=False, with_mcp=(kind == "rag"))
    except AssistantError as e:
        return {"text": "", "sources": [], "cost_usd": None, "session_id": None, "error": str(e)}

    t0 = time.monotonic()
    debug_log("collect_start", kind=kind, cmd=cmd, prompt=prompt)
    try:
        proc = subprocess.run(
            cmd, cwd=str(PROJECT_ROOT), input=prompt, env=subprocess_env(cfg),
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            encoding="utf-8", errors="replace",
            timeout=timeout or (cfg.get("timeout_seconds") or 300) + 30,
        )
    except subprocess.TimeoutExpired:
        debug_log("collect_timeout", kind=kind)
        return {"text": "", "sources": [], "cost_usd": None, "session_id": None,
                "error": "claude timed out"}
    except OSError as e:
        return {"text": "", "sources": [], "cost_usd": None, "session_id": None,
                "error": f"could not start claude: {e}"}

    final_text, session_id, cost, result_err = "", None, None, None
    for line in proc.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            continue
        t = msg.get("type")
        if t == "system" and msg.get("subtype") == "init":
            session_id = msg.get("session_id") or session_id
        elif t == "assistant":
            blocks = ((msg.get("message") or {}).get("content")) or []
            texts = [b.get("text", "") for b in blocks if b.get("type") == "text"]
            if texts:
                final_text = "\n".join(texts)
        elif t == "result":
            session_id = msg.get("session_id") or session_id
            cost = msg.get("total_cost_usd")
            if msg.get("is_error"):
                result_err = msg.get("result") or "claude reported an error"
            elif not final_text:
                final_text = msg.get("result", "")

    err = result_err
    if proc.returncode != 0 and not final_text:
        tail = "\n".join(proc.stderr.splitlines()[-10:]).strip()
        err = err or tail or f"claude exited with code {proc.returncode}"
    debug_log("collect_end", kind=kind, exit_code=proc.returncode,
              duration_s=round(time.monotonic() - t0, 1), cost_usd=cost,
              text_chars=len(final_text), error=err)
    return {"text": final_text, "sources": [], "cost_usd": cost,
            "session_id": session_id, "error": err if not final_text else None}


def first_turn_prompt(meta: dict, user_message: str) -> str:
    return (
        f"This is question {meta['number']} ({meta['category']}) from the NEET paper "
        f"\"{meta['title']}\".\n\n"
        f"Read the question image at this absolute path using your Read tool:\n"
        f"{meta['image_path']}\n\n"
        f"Extracted OCR text (may contain errors; the image is authoritative):\n"
        f"---\n{meta.get('text') or '(no text extracted)'}\n---\n\n"
        f"User's question: {user_message}"
    )


class Turn:
    """One in-flight claude turn. Events on .events:
    ("delta", text) | ("status", text) | ("done", info) | ("error", message)."""

    def __init__(self, slug: str, num: int, chat: dict):
        self.id = uuid.uuid4().hex
        self.slug, self.num, self.chat = slug, num, chat
        self.events: queue.Queue = queue.Queue()
        self.proc: subprocess.Popen | None = None
        self.text_parts: list[str] = []
        self.final_text: str | None = None
        self.stopped = False
        self.finished = threading.Event()
        self._lock = threading.Lock()

    # -- public ----------------------------------------------------------
    def start(self, prompt: str, fresh_prompt: str | None):
        """fresh_prompt: full context prompt to use if --resume fails and the
        turn is retried as a new session."""
        threading.Thread(target=self._run, args=(prompt, fresh_prompt), daemon=True).start()

    def stop(self):
        self.stopped = True
        with self._lock:
            if self.proc and self.proc.poll() is None:
                self.proc.kill()

    # -- internals ---------------------------------------------------------
    def _run(self, prompt: str, fresh_prompt: str | None):
        try:
            cfg = load_config()
            ok, err = self._run_once(cfg, prompt, self.chat.get("session_id"))
            if not ok and not self.stopped and self.chat.get("session_id") and \
                    ("No conversation found" in err or "session" in err.lower()):
                # stored session was cleaned up: restart fresh with full context
                self.chat["session_id"] = None
                self.text_parts.clear()
                self.events.put(("status", "previous session lost - restarting with question context"))
                ok, err = self._run_once(cfg, fresh_prompt or prompt, None)
            if not ok:
                self._fail(err)
        except AssistantError as e:
            self._fail(str(e))
        except Exception as e:
            self._fail(f"assistant backend error: {e!r}")

    def _run_once(self, cfg: dict, prompt: str, session_id: str | None) -> tuple[bool, str]:
        cmd = build_command(cfg, session_id)
        t0 = time.monotonic()
        debug_log("turn_start", turn=self.id, paper=self.slug, question=self.num,
                  resume=session_id, cmd=cmd, prompt=prompt)
        try:
            with self._lock:
                self.proc = subprocess.Popen(
                    cmd, cwd=str(PROJECT_ROOT), env=subprocess_env(cfg),
                    stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                    encoding="utf-8", errors="replace",
                )
        except OSError as e:
            debug_log("spawn_failed", turn=self.id, error=str(e))
            return False, f"could not start claude: {e}"
        proc = self.proc

        stderr_tail: deque = deque(maxlen=50)
        threading.Thread(target=lambda: stderr_tail.extend(proc.stderr),
                         daemon=True).start()

        idle = cfg.get("timeout_seconds") or 300
        watchdog = [None]

        def arm():
            if watchdog[0]:
                watchdog[0].cancel()
            watchdog[0] = threading.Timer(idle, proc.kill)
            watchdog[0].daemon = True
            watchdog[0].start()

        arm()
        try:
            proc.stdin.write(prompt)
            proc.stdin.close()
        except OSError:
            pass

        got_result = False
        result_info: dict = {}
        for line in proc.stdout:
            arm()
            line = line.strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                continue
            t = msg.get("type")
            if t == "system":
                debug_log("claude_system", turn=self.id, subtype=msg.get("subtype"),
                          session_id=msg.get("session_id"))
            if t == "system" and msg.get("subtype") == "init":
                sid = msg.get("session_id")
                if sid:
                    self.chat["session_id"] = sid
                    chats.save_chat(self.chat)
            elif t == "stream_event":
                ev = msg.get("event") or {}
                if ev.get("type") == "content_block_delta":
                    delta = (ev.get("delta") or {})
                    if delta.get("type") == "text_delta" and delta.get("text"):
                        self.text_parts.append(delta["text"])
                        self.events.put(("delta", delta["text"]))
                elif ev.get("type") == "content_block_start":
                    cb = ev.get("content_block") or {}
                    if cb.get("type") == "tool_use":
                        self.events.put(("status", f"using tool: {cb.get('name', '?')}"))
            elif t == "assistant":
                blocks = ((msg.get("message") or {}).get("content")) or []
                texts = [b.get("text", "") for b in blocks if b.get("type") == "text"]
                if texts:
                    self.final_text = "\n".join(texts)
            elif t == "result":
                got_result = True
                result_info = msg
        if watchdog[0]:
            watchdog[0].cancel()
        code = proc.wait()
        debug_log("turn_end", turn=self.id, exit_code=code, got_result=got_result,
                  is_error=result_info.get("is_error"), stopped=self.stopped,
                  duration_s=round(time.monotonic() - t0, 1),
                  cost_usd=result_info.get("total_cost_usd"),
                  text_chars=len(self.final_text or "".join(self.text_parts)),
                  stderr_tail="".join(list(stderr_tail)[-10:]).strip() or None)

        if self.stopped:
            self._finish_partial("stopped")
            return True, ""
        if got_result and not result_info.get("is_error"):
            text = self.final_text or "".join(self.text_parts) or result_info.get("result", "")
            sid = result_info.get("session_id")
            if sid:
                self.chat["session_id"] = sid
            chats.append_message(self.chat, "assistant", text)
            self.events.put(("done", {
                "session_id": self.chat.get("session_id"),
                "full_text": text,
                "cost_usd": result_info.get("total_cost_usd"),
            }))
            self.finished.set()
            return True, ""
        err = result_info.get("result") or "\n".join(stderr_tail).strip() or f"claude exited with code {code}"
        return False, err

    def _finish_partial(self, reason: str):
        partial = "".join(self.text_parts)
        if partial:
            chats.append_message(self.chat, "assistant", partial, stopped=True)
        self.events.put(("done", {"session_id": self.chat.get("session_id"),
                                  "full_text": partial, "stopped": True}))
        self.finished.set()

    def _fail(self, message: str):
        debug_log("turn_failed", turn=self.id, paper=self.slug, question=self.num,
                  error=message)
        partial = "".join(self.text_parts)
        if partial:
            chats.append_message(self.chat, "assistant", partial, error=message)
        self.events.put(("error", message))
        self.finished.set()


class TurnRegistry:
    """In-memory registry of running turns; one per question at a time."""

    def __init__(self):
        self._lock = threading.Lock()
        self._by_id: dict[str, Turn] = {}
        self._by_q: dict[tuple, Turn] = {}

    def create(self, slug: str, num: int, chat: dict) -> Turn:
        key = (slug, num)
        with self._lock:
            running = self._by_q.get(key)
            if running and not running.finished.is_set():
                raise AssistantError("a response is already streaming for this question")
            turn = Turn(slug, num, chat)
            self._by_id[turn.id] = turn
            self._by_q[key] = turn
            return turn

    def get(self, stream_id: str) -> Turn | None:
        return self._by_id.get(stream_id)

    def running_for(self, slug: str, num: int) -> Turn | None:
        t = self._by_q.get((slug, num))
        return t if t and not t.finished.is_set() else None

    def discard(self, turn: Turn):
        with self._lock:
            self._by_id.pop(turn.id, None)


REGISTRY = TurnRegistry()
