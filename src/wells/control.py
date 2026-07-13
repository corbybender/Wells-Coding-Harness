"""Run control shared between the TUI (main thread) and worker threads.

Three concerns, one module:

  * **Cooperative cancellation** — the TUI sets a flag (Escape); the executor
    and the graph stream loop check it between steps and stop cleanly. Thread
    workers cannot be killed, so this is the only correct way to cancel.
  * **Live activity** — the executor publishes what it is doing right now
    (round, step, current tool); the status bar polls it.
  * **UI event hook** — when a listener is registered (the TUI), executor
    output goes through typed events instead of stdout capture. Without a
    listener (plain CLI / tests), :func:`emit` returns False and callers fall
    back to ``print``.

Everything here must stay import-light and thread-safe: it is touched from
the Textual event loop, worker threads, and library code.
"""

from __future__ import annotations

import os
import platform
import subprocess
import threading
import time
from dataclasses import dataclass, field
from typing import Callable

_ON_WINDOWS = platform.system() == "Windows"


class RunCancelled(Exception):
    """Raised inside a run when the user cancelled it."""


def kill_process_tree(proc: subprocess.Popen) -> None:
    """Best-effort hard-kill of *proc* and everything it spawned.

    ``Popen.kill()`` only signals the direct child (pwsh.exe / sh) — whatever
    it launched (npm, a test runner, a build) keeps running as an orphan.
    Used by the shell-command tool's cancellation poll and by ``/stop``'s
    hard-kill path.
    """
    try:
        if _ON_WINDOWS:
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                capture_output=True, timeout=10,
            )
        else:
            import signal
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except Exception:
        pass
    try:
        proc.kill()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# UI events
# ---------------------------------------------------------------------------


@dataclass
class UIEvent:
    """One displayable event from a run.

    kind: run_note | llm_text | round | tool_line | warn | error
    text: preformatted Rich markup, ready to render.
    data: structured extras (tool name, args, ok, …) for richer UIs.
    """

    kind: str
    text: str = ""
    data: dict = field(default_factory=dict)


class RunControl:
    """Process-wide control channel for the active run."""

    def __init__(self) -> None:
        self._cancel = threading.Event()
        self._lock = threading.Lock()
        self._activity: str = ""
        self._activity_at: float = 0.0
        self._listener: Callable[[UIEvent], None] | None = None
        self._steers: list[str] = []
        # Ordered per-stage progress: label -> (current_step, cap). cap 0 = no limit.
        self._progress: dict[str, tuple[int, int]] = {}
        # Ordered pipeline stages: name -> {"status": run|done|fail, "t0", "secs"}.
        self._stages: dict[str, dict] = {}
        # Live subprocesses spawned by tool calls, for /stop's immediate hard-kill.
        self._procs: set[subprocess.Popen] = set()

    # -- cancellation --------------------------------------------------------

    def reset(self) -> None:
        """Call at the start of every run."""
        self._cancel.clear()
        self.set_activity("")
        with self._lock:
            self._steers.clear()
            self._progress.clear()
            self._stages.clear()

    # -- per-stage progress (drives the info panel) ---------------------------

    def set_progress(self, label: str, current: int, cap: int) -> None:
        with self._lock:
            self._progress[label] = (current, cap)

    def progress(self) -> list[tuple[str, int, int]]:
        """Stages in start order: (label, current_step, cap). cap 0 = no limit."""
        with self._lock:
            return [(k, v[0], v[1]) for k, v in self._progress.items()]

    # -- pipeline stages (orchestrate breadcrumb: indexer → planner → …) -------

    def stage_start(self, name: str) -> None:
        with self._lock:
            # Re-entering a stage (loop back to coder) restarts its clock.
            self._stages.pop(name, None)
            self._stages[name] = {"status": "run", "t0": time.monotonic(), "secs": 0.0}

    def stage_end(self, name: str, ok: bool = True) -> None:
        with self._lock:
            st = self._stages.get(name)
            if st is None:
                return
            st["secs"] = time.monotonic() - st["t0"]
            st["status"] = "done" if ok else "fail"

    def stages(self) -> list[tuple[str, str, float]]:
        """Pipeline stages in start order: (name, status, elapsed_seconds)."""
        now = time.monotonic()
        with self._lock:
            return [
                (k, v["status"],
                 v["secs"] if v["status"] != "run" else now - v["t0"])
                for k, v in self._stages.items()
            ]

    # -- mid-run steering ------------------------------------------------------

    def add_steer(self, text: str) -> None:
        """Queue a user instruction to inject into the agent's next round."""
        with self._lock:
            self._steers.append(text)

    def drain_steers(self) -> list[str]:
        with self._lock:
            out = self._steers[:]
            self._steers.clear()
            return out

    def pending_steers(self) -> int:
        with self._lock:
            return len(self._steers)

    def cancel(self) -> None:
        self._cancel.set()

    def cancelled(self) -> bool:
        return self._cancel.is_set()

    def checkpoint(self) -> None:
        """Raise :class:`RunCancelled` if a cancel was requested."""
        if self._cancel.is_set():
            raise RunCancelled()

    # -- hard kill (/stop) ----------------------------------------------------
    # Escape/cancel() is cooperative — it asks running code to stop at its next
    # checkpoint (a polling loop). /stop does not ask: it kills every tracked
    # subprocess right now, on top of setting the cancel flag. Python cannot
    # force-kill a thread, so a blocked-on-network-I/O worker may still be
    # finishing a single call in the background — the run generation counter
    # in the TUI is what makes that outcome get discarded instead of reviving
    # the UI.

    def track_proc(self, proc: subprocess.Popen) -> None:
        with self._lock:
            self._procs.add(proc)

    def untrack_proc(self, proc: subprocess.Popen) -> None:
        with self._lock:
            self._procs.discard(proc)

    def kill_tracked_procs(self) -> int:
        """Hard-kill every live tracked subprocess (and its tree). Returns count."""
        with self._lock:
            procs = list(self._procs)
        n = 0
        for p in procs:
            if p.poll() is None:
                kill_process_tree(p)
                n += 1
            self.untrack_proc(p)
        return n

    # -- live activity -------------------------------------------------------

    def set_activity(self, text: str) -> None:
        with self._lock:
            if text != self._activity:
                self._activity_at = time.monotonic()
            self._activity = text

    def activity(self) -> str:
        with self._lock:
            return self._activity

    def activity_info(self) -> tuple[str, float]:
        """Current activity and seconds since it was last changed."""
        with self._lock:
            if not self._activity:
                return "", 0.0
            return self._activity, time.monotonic() - self._activity_at

    # -- UI events -----------------------------------------------------------

    def set_listener(self, fn: Callable[[UIEvent], None] | None) -> None:
        self._listener = fn

    def emit(self, kind: str, text: str = "", **data) -> bool:
        """Send an event to the registered listener.

        Returns True when a listener consumed it; False means the caller
        should fall back to printing (plain CLI mode).
        """
        fn = self._listener
        if fn is None:
            return False
        try:
            fn(UIEvent(kind=kind, text=text, data=data))
        except Exception:
            return False
        return True


CONTROL = RunControl()


def ui(kind: str, text: str = "", **data) -> None:
    """Emit a UI event, falling back to print when no listener is registered."""
    if not CONTROL.emit(kind, text, **data):
        print(text)
