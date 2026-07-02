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

import threading
from dataclasses import dataclass, field
from typing import Callable


class RunCancelled(Exception):
    """Raised inside a run when the user cancelled it."""


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
        self._listener: Callable[[UIEvent], None] | None = None

    # -- cancellation --------------------------------------------------------

    def reset(self) -> None:
        """Call at the start of every run."""
        self._cancel.clear()
        self.set_activity("")

    def cancel(self) -> None:
        self._cancel.set()

    def cancelled(self) -> bool:
        return self._cancel.is_set()

    def checkpoint(self) -> None:
        """Raise :class:`RunCancelled` if a cancel was requested."""
        if self._cancel.is_set():
            raise RunCancelled()

    # -- live activity -------------------------------------------------------

    def set_activity(self, text: str) -> None:
        with self._lock:
            self._activity = text

    def activity(self) -> str:
        with self._lock:
            return self._activity

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
