"""Lifecycle management for the long-lived neet-rag HTTP server.

The viewer spawns the warm RAG server as a child on startup and guarantees it
dies with the viewer: on Windows the child is placed in a Job Object with
KILL_ON_JOB_CLOSE (so even a force-kill of the viewer takes the server down),
with an atexit/signal handler as the graceful fallback. If a server is already
listening on the configured port (e.g. started by hand), it is reused and left
untouched.

Config lives under the "rag_server" key of assistant_config.json:
    {enabled, command[], cwd, env{}, host, port, ready_timeout}
"""
import atexit
import os
import signal
import socket
import subprocess
import sys
import threading
import time

from .assistant import load_config, debug_log

_proc: subprocess.Popen | None = None
_job = None              # keep the Job Object handle alive for the process lifetime
_owned = False
_lock = threading.Lock()


def _port_open(host: str, port: int, timeout: float = 1.0) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def _kill_on_close_job(pid: int):
    """Windows: put `pid` in a job that kills it when the job handle closes
    (i.e. when this viewer process exits, however it exits)."""
    import ctypes
    from ctypes import wintypes as w

    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x2000
    PROCESS_TERMINATE, PROCESS_SET_QUOTA = 0x0001, 0x0100
    JobObjectExtendedLimitInformation = 9

    class BASIC(ctypes.Structure):
        _fields_ = [("PerProcessUserTimeLimit", ctypes.c_int64),
                    ("PerJobUserTimeLimit", ctypes.c_int64),
                    ("LimitFlags", w.DWORD),
                    ("MinimumWorkingSetSize", ctypes.c_size_t),
                    ("MaximumWorkingSetSize", ctypes.c_size_t),
                    ("ActiveProcessLimit", w.DWORD),
                    ("Affinity", ctypes.c_size_t),
                    ("PriorityClass", w.DWORD),
                    ("SchedulingClass", w.DWORD)]

    class IO(ctypes.Structure):
        _fields_ = [(n, ctypes.c_uint64) for n in
                    ("Read", "Write", "Other", "ReadT", "WriteT", "OtherT")]

    class EXT(ctypes.Structure):
        _fields_ = [("BasicLimitInformation", BASIC), ("IoInfo", IO),
                    ("ProcessMemoryLimit", ctypes.c_size_t),
                    ("JobMemoryLimit", ctypes.c_size_t),
                    ("PeakProcessMemoryUsed", ctypes.c_size_t),
                    ("PeakJobMemoryUsed", ctypes.c_size_t)]

    k32.CreateJobObjectW.restype = w.HANDLE
    k32.OpenProcess.restype = w.HANDLE
    hjob = k32.CreateJobObjectW(None, None)
    if not hjob:
        return None
    info = EXT()
    info.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
    if not k32.SetInformationJobObject(hjob, JobObjectExtendedLimitInformation,
                                       ctypes.byref(info), ctypes.sizeof(info)):
        return None
    hproc = k32.OpenProcess(PROCESS_TERMINATE | PROCESS_SET_QUOTA, False, pid)
    if not hproc:
        return None
    ok = k32.AssignProcessToJobObject(hjob, hproc)
    k32.CloseHandle(hproc)
    return hjob if ok else None


def ensure_started() -> None:
    """Start the managed RAG server if enabled and not already running."""
    global _proc, _job, _owned
    cfg = (load_config().get("rag_server") or {})
    if not cfg.get("enabled"):
        return
    host, port = cfg.get("host", "127.0.0.1"), int(cfg.get("port", 8077))
    with _lock:
        if _proc and _proc.poll() is None:
            return
        if _port_open(host, port):
            debug_log("rag_server_reused", host=host, port=port)
            print(f"[rag] reusing existing neet-rag server on {host}:{port}", flush=True)
            return
        cmd = cfg.get("command")
        if not cmd:
            return
        env = os.environ.copy()
        env.update({k: str(v) for k, v in (cfg.get("env") or {}).items()})
        try:
            _proc = subprocess.Popen(cmd, cwd=cfg.get("cwd"), env=env,
                                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except OSError as e:
            debug_log("rag_server_spawn_failed", error=str(e))
            print(f"[rag] could not start neet-rag server: {e}", flush=True)
            return
        _owned = True
        if sys.platform == "win32":
            try:
                _job = _kill_on_close_job(_proc.pid)
            except Exception as e:  # job is best-effort; atexit still covers graceful exit
                debug_log("rag_server_job_failed", error=str(e))
        atexit.register(stop)
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                signal.signal(sig, lambda *_: (stop(), sys.exit(0)))
            except (ValueError, OSError):
                pass  # not in main thread / unsupported
        debug_log("rag_server_started", pid=_proc.pid, cmd=cmd, port=port)
        print(f"[rag] starting neet-rag server (pid {_proc.pid}); warming up in background...", flush=True)

    # warm up off the request path so the viewer serves immediately
    threading.Thread(target=_wait_ready, args=(host, port, int(cfg.get("ready_timeout", 180))),
                     daemon=True).start()


def _wait_ready(host: str, port: int, timeout: int) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if _proc and _proc.poll() is not None:
            print("[rag] server exited during warmup", flush=True)
            return
        if _port_open(host, port):
            print(f"[rag] neet-rag server warm on {host}:{port}", flush=True)
            return
        time.sleep(2)
    print(f"[rag] warmup timed out after {timeout}s (continuing; chat RAG may be slow)", flush=True)


def stop() -> None:
    global _proc
    with _lock:
        if _proc and _owned and _proc.poll() is None:
            try:
                _proc.terminate()
                _proc.wait(timeout=10)
            except Exception:
                try:
                    _proc.kill()
                except Exception:
                    pass
        _proc = None
