"""Full-screen Textual TUI for Wells coding harness.

Replaces the prompt_toolkit REPL with a proper layout:
  ┌─────────────────────────────────────────────┐
  │  RichLog  — scrollable output               │
  ├─────────────────────────────────────────────┤
  │  PromptInput — multi-line user prompt       │
  ├─────────────────────────────────────────────┤
  │  StatusBar — always visible, refreshes live │
  └─────────────────────────────────────────────┘

Run output arrives through three channels, in order of preference:
  1. Typed UI events from ``control.CONTROL`` (executor tool lines, warnings).
  2. A Rich console proxy (Rich markup / tables / panels from cli.py).
  3. Captured sys.stdout (stray prints from agents / libraries).
All three funnel into :meth:`WellsApp.write_log`, which also records a
transcript for ``/export``.

Cancellation is cooperative: Escape sets ``CONTROL.cancel()`` and the
executor / graph loop stop at the next step boundary. Thread workers cannot
be killed, so the input stays disabled until the worker actually exits.
"""

from __future__ import annotations

import io
import json
import sys
import threading
import time as _time
from pathlib import Path
from typing import Any

from textual import work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.message import Message
from textual.widgets import OptionList, RichLog, Static, TextArea
from textual.widgets._option_list import Option

from coding_harness import chat, config
from coding_harness.control import CONTROL, UIEvent
from coding_harness.tokens import LEDGER

_HISTORY_FILE = Path.home() / ".wells" / "history.json"
_HISTORY_MAX = 200

# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------

_CSS = """
Screen {
    background: $background;
}

#output {
    height: 1fr;
    border: none;
    padding: 0 1;
    scrollbar-gutter: stable;
}

#bottom {
    height: auto;
    dock: bottom;
}

#command-list {
    display: none;
    height: auto;
    max-height: 14;
    border: solid $accent 50%;
    background: $surface-darken-1;
    dock: bottom;
    margin-bottom: 4;
}

#input {
    height: auto;
    max-height: 8;
    border: tall $accent 40%;
    background: $surface;
}

#input:focus {
    border: tall $accent;
}

StatusBar {
    height: 1;
    background: #1a1a2e;
    padding: 0 1;
}
"""


# ---------------------------------------------------------------------------
# Status bar — always visible, refreshes every 250 ms
# ---------------------------------------------------------------------------

class StatusBar(Static):
    """Persistent status bar: workspace | model | tokens | activity | mode."""

    def on_mount(self) -> None:
        self.set_interval(0.25, self._refresh)

    def _refresh(self) -> None:
        self.update(self._build())

    def _build(self) -> str:
        try:
            totals = LEDGER.totals()
            used = totals["input"] + totals["output"]
            saved = totals["saved_trim"] + totals["saved_summary"]
        except Exception:
            used = saved = 0

        try:
            model = config.model_name_for_task("coding")
        except Exception:
            model = "?"

        wd = config.WORKSPACE_ROOT
        if len(wd) > 36:
            wd = "…" + wd[-35:]

        saved_s = f"  [dim]saved: {saved:,}[/dim]" if saved else ""
        tokens_s = f"tokens: {used:,}{saved_s}"

        try:
            import coding_harness.cli as _cli
            state = _cli._REPL_STATE
            force = state.get("force_mode")
            busy_since = state.get("busy_since")
        except Exception:
            force = None
            busy_since = None

        if force == "task":
            mode = "[bold magenta]mode: orchestrate[/bold magenta]"
        else:
            mode = "[dim]mode: auto[/dim]"

        if busy_since is not None:
            secs = int(_time.monotonic() - busy_since)
            if secs >= 60:
                elapsed = f"[yellow]{secs // 60}m {secs % 60:02d}s[/yellow]"
            else:
                elapsed = f"[dim]{secs}s[/dim]"
            activity = CONTROL.activity()
            act_s = f"  [magenta]{activity}[/magenta]" if activity else ""
            elapsed_s = f"  {elapsed}{act_s}  [dim]esc: cancel[/dim]"
        else:
            elapsed_s = ""

        return (
            f"[dim]{wd}[/dim]  "
            f"[green]{model}[/green]  "
            f"[cyan]{tokens_s}[/cyan]{elapsed_s}  "
            f"{mode}"
        )


# ---------------------------------------------------------------------------
# Multi-line prompt input
# ---------------------------------------------------------------------------

class PromptInput(TextArea):
    """Multi-line prompt. Enter submits; Shift+Enter / Ctrl+J inserts a newline.

    Up on the first line / Down on the last line scroll the prompt history
    (handled by the app via :class:`HistoryScroll`).
    """

    class Submitted(Message):
        def __init__(self, value: str) -> None:
            self.value = value
            super().__init__()

    class HistoryScroll(Message):
        def __init__(self, direction: int) -> None:
            self.direction = direction  # -1 older, +1 newer
            super().__init__()

    async def _on_key(self, event) -> None:
        key = event.key

        if key == "enter":
            event.stop()
            event.prevent_default()
            self.post_message(self.Submitted(self.text))
            return

        if key in ("shift+enter", "ctrl+j"):
            event.stop()
            event.prevent_default()
            self.insert("\n")
            return

        if key == "escape":
            # Bubble to the app: closes the command popup / cancels the task.
            return

        if key == "up" and self.cursor_location[0] == 0:
            event.stop()
            event.prevent_default()
            self.post_message(self.HistoryScroll(-1))
            return

        if key == "down":
            popup = getattr(self.app, "_cmdlist", None)
            if popup is not None and popup.display:
                return  # bubble: the app moves focus into the command list
            if self.cursor_location[0] == self.document.line_count - 1:
                event.stop()
                event.prevent_default()
                self.post_message(self.HistoryScroll(1))
                return

        await super()._on_key(event)


# ---------------------------------------------------------------------------
# I/O capture helpers
# ---------------------------------------------------------------------------

class _TUIStdout:
    """Redirect sys.stdout → RichLog (used for streaming tokens + bare print())."""

    def __init__(self, app: "WellsApp") -> None:
        self._app = app
        self._buf = ""

    def write(self, text: str) -> int:
        if text:
            # Buffer partial lines; flush on newline so Rich sees full lines.
            self._buf += text
            if "\n" in self._buf:
                lines = self._buf.split("\n")
                for line in lines[:-1]:
                    if line:
                        self._app.call_from_thread(self._app.write_log, line)
                self._buf = lines[-1]
        return len(text)

    def flush(self) -> None:
        if self._buf:
            self._app.call_from_thread(self._app.write_log, self._buf)
            self._buf = ""

    def fileno(self) -> int:
        raise OSError("_TUIStdout has no file descriptor")

    def isatty(self) -> bool:
        return False


class _TUIConsole:
    """Drop-in for Rich Console inside cli.py; routes markup to the log.

    thread_safe=True  → use call_from_thread (worker threads)
    thread_safe=False → call widget directly   (asyncio main thread)
    """

    def __init__(self, app: "WellsApp", *, thread_safe: bool = True) -> None:
        self._app = app
        self._thread_safe = thread_safe

    def _write(self, renderable: Any) -> None:
        if self._thread_safe:
            self._app.call_from_thread(self._app.write_log, renderable)
        else:
            self._app.write_log(renderable)

    def print(self, *args, **kwargs) -> None:
        if not args:
            self._write("")
            return
        # Single arg: pass renderables (Table, Panel, etc.) directly.
        if len(args) == 1:
            arg = args[0]
            if isinstance(arg, str):
                self._write((arg))
            else:
                self._write(arg)  # Rich renderable
        else:
            # Multiple args: join as markup string.
            self._write((" ".join(str(a) for a in args)))

    def status(self, message: str = "", **kwargs) -> "_NullStatus":
        return _NullStatus(message, self)

    def log(self, *args, **kwargs) -> None:
        self.print(*args, **kwargs)


class _NullStatus:
    """Replaces console.status() context manager (spinners can't run in threads)."""

    def __init__(self, msg: str, console: _TUIConsole) -> None:
        self._msg = msg
        self._console = console

    def __enter__(self) -> "_NullStatus":
        if self._msg:
            self._console.print(f"[cyan]{self._msg}[/cyan]")
        return self

    def __exit__(self, *args: Any) -> None:
        pass


# ---------------------------------------------------------------------------
# Main Textual application
# ---------------------------------------------------------------------------

class WellsApp(App[None]):
    """Full-screen TUI for the Wells coding harness."""

    CSS = _CSS

    BINDINGS = [
        Binding("ctrl+c", "quit", "Quit", priority=True),
        Binding("escape", "cancel_task", "Cancel task"),
        Binding("ctrl+l", "clear_log", "Clear output"),
        # Scroll bindings — priority=True so they fire even when Input has focus.
        # mouse=False hands mouse-wheel events to the terminal (for copy-paste),
        # so keyboard is the only scroll path inside the TUI.
        Binding("pageup",    "scroll_up",     "Scroll up",      show=False, priority=True),
        Binding("pagedown",  "scroll_down",   "Scroll down",    show=False, priority=True),
        Binding("ctrl+home", "scroll_top",    "Scroll to top",  show=False, priority=True),
        Binding("ctrl+end",  "scroll_bottom", "Scroll to end",  show=False, priority=True),
    ]

    def __init__(self, resume_context: str | None = None) -> None:
        super().__init__()
        self._resume_context = resume_context
        self._graph_app: Any = None
        self._agent_state: dict = {}
        self._busy = False
        # Pending interactive command waiting for a follow-up reply.
        # Format: {"kind": "resume_select"|"sessions_clear"|"approval", ...}
        self._pending: dict | None = None
        # Prompt history (persisted) + browse state.
        self._history: list[str] = []
        self._hist_idx: int | None = None
        self._hist_draft: str = ""
        # Everything ever written to the log, for /export.
        self._transcript: list[Any] = []

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------

    def compose(self) -> ComposeResult:
        yield RichLog(
            id="output",
            markup=True,
            highlight=True,
            wrap=True,
        )
        yield OptionList(id="command-list")
        with Vertical(id="bottom"):
            yield PromptInput(
                id="input",
                placeholder="Ask a question or give a task… (/ commands, Shift+Enter newline)",
                show_line_numbers=False,
            )
            yield StatusBar()

    # ------------------------------------------------------------------
    # Startup
    # ------------------------------------------------------------------

    def on_mount(self) -> None:
        from coding_harness import safety
        from coding_harness.cli import _REPL_STATE
        from coding_harness.graph import build_graph

        self._log: RichLog = self.query_one("#output", RichLog)
        self._input: PromptInput = self.query_one("#input", PromptInput)
        self._cmdlist: OptionList = self.query_one("#command-list", OptionList)

        # Initialize shared REPL state.
        _REPL_STATE["memory"] = chat.ConversationMemory()
        _REPL_STATE["force_mode"] = None
        _REPL_STATE["last_state"] = {}
        _REPL_STATE["resume_context"] = self._resume_context
        _REPL_STATE["resume_session_id"] = None

        # Typed UI events from the executor render directly (no stdout hop).
        CONTROL.set_listener(self._on_ui_event)
        # Under HARNESS_SAFETY=approve, destructive tool calls ask the user here.
        safety.set_approver(self._tui_approver)

        self._history = self._history_load()

        # Build the LangGraph (may take a moment).
        self._graph_app = build_graph()
        self._agent_state = {
            "iteration": 0,
            "max_iterations": config.MAX_ITERATIONS,
            "workspace_root": config.WORKSPACE_ROOT,
            "safety": config.HARNESS_SAFETY,
            "plan_mode": config.PLAN_MODE,
            "messages": [],
            "executor_messages": [],
        }

        self._print_welcome()
        if self._resume_context:
            self.write_log(
                "[dim]Session context loaded — next task continues from previous session.[/dim]"
            )

        self._ensure_repo_index()
        self._input.focus()

    def on_unmount(self) -> None:
        CONTROL.set_listener(None)

    # ------------------------------------------------------------------
    # Output helpers
    # ------------------------------------------------------------------

    def write_log(self, renderable: Any) -> None:
        """Write to the RichLog from the asyncio event loop thread."""
        self._transcript.append(renderable)
        self._log.write(renderable)

    def _on_ui_event(self, ev: UIEvent) -> None:
        """Render a typed executor event (called from worker threads)."""
        try:
            self.call_from_thread(self.write_log, ev.text)
        except Exception:
            # Already on the app thread (or app shutting down).
            try:
                self.write_log(ev.text)
            except Exception:
                pass

    def _print_welcome(self) -> None:
        logo_shown = False
        try:
            from coding_harness.logo import logo_lines
            width = self.size.width or 80
            lines = logo_lines(max_width=width)
            self._log.write("")
            for line in lines:
                self._log.write(line)  # not transcripted — /export stays clean
            logo_shown = bool(lines)
        except Exception:
            pass
        if not logo_shown:
            # Narrow terminal: plain title instead of the glyph lockup.
            self.write_log("\n[bold blue]Wells Coding Harness[/bold blue]")
        self.write_log(f"[dim]Model:[/dim] [green]{config.model_name_for_task('coding')}[/green]")
        self.write_log(
            f"[dim]Workspace:[/dim] [bold]{config.WORKSPACE_ROOT}[/bold]"
            f"  [dim](safety: {config.HARNESS_SAFETY})[/dim]"
        )
        self.write_log(
            "Ask anything — questions, edits, tasks. "
            "Use [bold]/orchestrate[/bold] for complex multi-component work. "
            "Type [bold]/[/bold] for all commands. "
            "[dim]Shift+Enter: newline · ↑/↓: history · Esc: cancel run[/dim]\n"
        )

    def _ensure_repo_index(self) -> None:
        try:
            from coding_harness import config as _cfg
            if not _cfg.INDEX_AUTO_UPDATE:
                return
            from coding_harness.index_tools import ensure_index
            max_age = float(__import__("os").environ.get("INDEX_MAX_AGE_HOURS", "24"))
            result = ensure_index(_cfg.WORKSPACE_ROOT, max_age_hours=max_age, auto_build=True)
            if result.startswith("index-built"):
                self.write_log(f"[green]{result[:120]}[/green]")
            elif result.startswith("index-failed"):
                self.write_log(f"[yellow]Index: {result}[/yellow]")
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Prompt history
    # ------------------------------------------------------------------

    def _history_load(self) -> list[str]:
        try:
            data = json.loads(_HISTORY_FILE.read_text(encoding="utf-8"))
            return [str(x) for x in data][-_HISTORY_MAX:]
        except Exception:
            return []

    def _history_save(self) -> None:
        try:
            _HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
            _HISTORY_FILE.write_text(
                json.dumps(self._history[-_HISTORY_MAX:], ensure_ascii=False),
                encoding="utf-8",
            )
        except Exception:
            pass

    def _history_add(self, text: str) -> None:
        if text and (not self._history or self._history[-1] != text):
            self._history.append(text)
            self._history_save()
        self._hist_idx = None
        self._hist_draft = ""

    def _set_input_text(self, text: str) -> None:
        self._input.load_text(text)
        self._input.move_cursor(self._input.document.end)

    def on_prompt_input_history_scroll(self, event: PromptInput.HistoryScroll) -> None:
        if not self._history:
            return
        if self._hist_idx is None:
            if event.direction > 0:
                return  # nothing newer than the live draft
            self._hist_draft = self._input.text
            self._hist_idx = len(self._history) - 1
        else:
            self._hist_idx += event.direction

        if self._hist_idx >= len(self._history):
            # Scrolled past the newest entry — restore the draft.
            self._hist_idx = None
            self._set_input_text(self._hist_draft)
            return

        self._hist_idx = max(0, self._hist_idx)
        self._set_input_text(self._history[self._hist_idx])

    # ------------------------------------------------------------------
    # Input handling
    # ------------------------------------------------------------------

    def on_text_area_changed(self, event: TextArea.Changed) -> None:
        """Show/hide the command popup as the user types."""
        from coding_harness.cli import SLASH_COMMANDS
        value = self._input.text
        if value.startswith("/") and "\n" not in value:
            matches = [
                (cmd, short) for cmd, short, _ in SLASH_COMMANDS
                if cmd.startswith(value)
            ]
            self._cmdlist.clear_options()
            for cmd, short in matches:
                self._cmdlist.add_option(Option(f"{cmd}  [dim]{short}[/dim]", id=cmd))
            self._cmdlist.display = bool(matches)
        else:
            self._cmdlist.display = False

    def on_key(self, event) -> None:
        """Arrow-down from input moves focus into the command list."""
        if event.key == "down" and self._cmdlist.display and self.focused is self._input:
            self._cmdlist.focus()
            event.stop()
        elif event.key == "escape" and self._cmdlist.display:
            self._cmdlist.display = False
            self._input.focus()
            event.stop()

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        """Fill the input with the selected command and return focus."""
        cmd = event.option_id or ""
        if cmd:
            self._set_input_text(cmd + " ")
        self._cmdlist.display = False
        self._input.focus()

    def on_prompt_input_submitted(self, event: PromptInput.Submitted) -> None:
        self._cmdlist.display = False
        text = event.value.strip()
        self._input.load_text("")
        if not text:
            return

        # Handle pending interactive confirmations first.
        if self._pending:
            self._handle_pending_reply(text)
            return

        self._history_add(text)

        if text.startswith("/"):
            self._dispatch_slash(text)
            return

        if self._busy:
            self.write_log("[yellow]Still working… please wait.[/yellow]")
            return

        self._start_run(text)

    # ------------------------------------------------------------------
    # Slash commands
    # ------------------------------------------------------------------

    def _dispatch_slash(self, command: str) -> None:
        """Route slash commands on the asyncio event loop thread.

        Commands that call input() internally — /resume and /sessions clear —
        are intercepted HERE, before handle_slash_command is ever called.
        Letting them reach handle_slash_command would block the event loop
        forever because Textual owns stdin and input() never returns.
        """
        parts = command.strip().split()
        cmd = parts[0].lower() if parts else ""
        args = parts[1:]
        arg = args[0] if args else ""

        # -- Intercept blocking / TUI-only commands first ----------------------
        if cmd == "/resume":
            self._start_resume_picker(arg)
            return

        if cmd == "/export":
            self._export_transcript(" ".join(args))
            return

        if cmd == "/undo":
            from coding_harness.cli import undo_preview
            sha, stat = undo_preview()
            if not sha:
                self.write_log(
                    "[yellow]No checkpoint to undo (no run yet, or not a git repo).[/yellow]"
                )
                return
            if not stat:
                self.write_log(
                    "[dim]Working tree already matches the last checkpoint — nothing to undo.[/dim]"
                )
                return
            self.write_log(
                f"\n[bold]Reverting to pre-run checkpoint {sha[:8]}:[/bold]\n{stat}\n"
                "[dim]Type [bold]y[/bold] to revert or anything else to cancel.[/dim]"
            )
            self._pending = {"kind": "undo_confirm", "sha": sha}
            return

        if cmd == "/sessions" and arg.lower() == "clear":
            all_ws = "--all" in [a.lower() for a in args]
            scope = "ALL workspaces" if all_ws else "this workspace"
            self.write_log(
                f"[yellow]Delete all sessions for {scope}?[/yellow] "
                "[dim]Type [bold]y[/bold] to confirm or anything else to cancel.[/dim]"
            )
            self._pending = {"kind": "sessions_clear", "all_ws": all_ws}
            return

        # -- All other commands are safe to run synchronously -----------------
        import coding_harness.cli as cli_mod
        orig = cli_mod.console
        cli_mod.console = _TUIConsole(self, thread_safe=False)
        try:
            keep_running = cli_mod.handle_slash_command(command)
        finally:
            cli_mod.console = orig

        if not keep_running:
            self.exit()

    def _export_transcript(self, arg: str) -> None:
        """Write the full session log to a plain-text/markdown file."""
        from rich.console import Console as _RichConsole

        name = (arg or "").strip() or _time.strftime("wells-transcript-%Y%m%d-%H%M%S.md")
        path = Path(name).expanduser()
        if not path.is_absolute():
            path = Path(config.WORKSPACE_ROOT) / path

        buf = io.StringIO()
        rc = _RichConsole(file=buf, width=100, force_terminal=False)
        for item in self._transcript:
            try:
                rc.print(item)
            except Exception:
                rc.print(str(item))
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(buf.getvalue(), encoding="utf-8")
            self.write_log(f"[green]Transcript exported → {path}[/green]")
        except Exception as e:
            self.write_log(f"[red]Export failed: {e}[/red]")

    def _start_resume_picker(self, arg: str) -> None:
        from coding_harness.sessions import (
            build_resume_context, format_age, is_session_id,
            list_sessions, load_session,
        )

        if arg and is_session_id(arg):
            session = load_session(arg)
            if not session:
                self.write_log(f"[red]Session not found: {arg}[/red]")
                return
            self._load_resume_session(session)
            return

        sessions = list_sessions(workspace=config.WORKSPACE_ROOT, limit=10)
        if not sessions:
            self.write_log("[yellow]No sessions for this workspace.[/yellow]")
            return

        self.write_log("\n[bold]Recent sessions:[/bold]")
        for i, s in enumerate(sessions, 1):
            age = format_age(s.get("created_at", ""))
            status = s.get("status", "?")
            color = "green" if status == "COMPLETE" else "yellow"
            goal = (s.get("goal") or "")[:60]
            self.write_log(
                f"  [cyan]{i}.[/cyan] [{age}] [{color}]{status}[/{color}] {goal!r}"
            )
            self.write_log(f"     [dim]{s['id']}[/dim]")
        self.write_log(
            "\n[dim]Type the session number to load, or anything else to cancel.[/dim]"
        )
        self._pending = {"kind": "resume_select", "sessions": sessions}

    def _handle_pending_reply(self, text: str) -> None:
        assert self._pending is not None
        kind = self._pending["kind"]

        if kind == "resume_select":
            sessions = self._pending["sessions"]
            self._pending = None
            try:
                session = sessions[int(text) - 1]
                self._load_resume_session(session)
            except (ValueError, IndexError):
                self.write_log("[dim]Cancelled.[/dim]")

        elif kind == "sessions_clear":
            all_ws = self._pending["all_ws"]
            self._pending = None
            if text.lower() in ("y", "yes"):
                from coding_harness.sessions import clear_sessions
                workspace = None if all_ws else config.WORKSPACE_ROOT
                n = clear_sessions(workspace=workspace)
                self.write_log(f"[green]Deleted {n} session(s).[/green]")
            else:
                self.write_log("[dim]Cancelled.[/dim]")

        elif kind == "undo_confirm":
            sha = self._pending["sha"]
            self._pending = None
            if text.lower() in ("y", "yes"):
                from coding_harness.cli import undo_apply
                ok, msg = undo_apply(sha)
                self.write_log(f"[green]{msg}[/green]" if ok else f"[red]{msg}[/red]")
            else:
                self.write_log("[dim]Cancelled.[/dim]")

        elif kind == "approval":
            pend = self._pending
            self._pending = None
            ok = text.lower() in ("y", "yes")
            pend["holder"]["ok"] = ok
            self.write_log(
                "[green]Approved.[/green]" if ok else "[dim]Denied.[/dim]"
            )
            if self._busy:
                # Hand the input back to the running worker.
                self._input.disabled = True
                self._input.placeholder = "Working…"
            pend["event"].set()

    def _load_resume_session(self, session: dict) -> None:
        from coding_harness.sessions import build_resume_context
        import coding_harness.cli as cli_mod

        cli_mod._REPL_STATE["resume_context"] = build_resume_context(session)
        cli_mod._REPL_STATE["resume_session_id"] = session["id"]
        self.write_log(f"\n[green]Session loaded: {session['id']}[/green]")
        self.write_log(
            f"[dim]Previous goal: {(session.get('goal') or '')[:70]}[/dim]"
        )
        self.write_log(
            "[dim]Context injected — your next task will continue from this session.[/dim]\n"
        )

    # ------------------------------------------------------------------
    # Safety approval (HARNESS_SAFETY=approve)
    # ------------------------------------------------------------------

    def _tui_approver(self, action: str, detail: str) -> bool:
        """Ask the user to approve a destructive action.

        Called from a worker thread while a run is in flight; blocks that
        thread until the user answers (or the run is cancelled).
        """
        if threading.current_thread() is threading.main_thread():
            # Cannot block the UI thread; deny → safety degrades to dry-run.
            return False

        ev = threading.Event()
        holder = {"ok": False}

        def _ask() -> None:
            self.write_log(
                f"\n[bold yellow]Approval needed:[/bold yellow] {action}"
                f"\n  [dim]{detail}[/dim]"
                "\n[dim]Type [bold]y[/bold] to approve, anything else to deny.[/dim]"
            )
            self._pending = {"kind": "approval", "event": ev, "holder": holder}
            self._input.disabled = False
            self._input.placeholder = "Approve? y/N"
            self._input.focus()

        self.call_from_thread(_ask)

        while not ev.wait(0.5):
            if CONTROL.cancelled():
                return False
        return holder["ok"]

    # ------------------------------------------------------------------
    # Task / chat runner (worker thread)
    # ------------------------------------------------------------------

    def _start_run(self, text: str) -> None:
        import coding_harness.cli as _cli
        CONTROL.reset()
        self._busy = True
        self._input.disabled = True
        self._input.placeholder = "Working…"
        _cli._REPL_STATE["busy_since"] = _time.monotonic()
        self.write_log(f"\n[bold cyan]>[/bold cyan] {text}\n")
        self._run_input(text)

    @work(thread=True)
    def _run_input(self, text: str) -> None:
        """Run chat or task in a worker thread with redirected I/O."""
        import coding_harness.cli as cli_mod
        from coding_harness.cli import (
            _REPL_STATE, _run_task, _summarize_run,
        )

        tui_console = _TUIConsole(self, thread_safe=True)
        tui_stdout = _TUIStdout(self)

        orig_cli_console = cli_mod.console
        orig_stdout = sys.stdout
        orig_stderr = sys.stderr

        cli_mod.console = tui_console
        sys.stdout = tui_stdout
        sys.stderr = tui_stdout  # capture stray prints from agents

        try:
            force = _REPL_STATE.get("force_mode")
            if force:
                intent = force
                _REPL_STATE["force_mode"] = None
            else:
                intent = chat.classify_intent(text)

            from coding_harness.cli import StreamingCallback, _run_auto
            callbacks = [StreamingCallback()]

            if intent in ("task", "orchestrate"):
                _run_task(text, self._agent_state, self._graph_app, callbacks)
                _REPL_STATE["memory"].set_run_summary(
                    _summarize_run(_REPL_STATE.get("last_state", {}))
                )
            else:
                # "auto" (default) — direct executor, handles Q&A and tasks.
                _run_auto(text, self._agent_state, callbacks)
        except Exception as e:
            self.call_from_thread(
                self.write_log, f"[bold red]Error:[/bold red] {e}"
            )
        finally:
            # Flush any buffered output, then restore I/O — but only if this
            # worker still owns the redirect (a cancelled worker must never
            # clobber the redirect installed by a newer run).
            tui_stdout.flush()
            if sys.stdout is tui_stdout:
                sys.stdout = orig_stdout
            if sys.stderr is tui_stdout:
                sys.stderr = orig_stderr
            if cli_mod.console is tui_console:
                cli_mod.console = orig_cli_console

            self._busy = False
            self.call_from_thread(self._restore_input)

    def _restore_input(self) -> None:
        import coding_harness.cli as _cli
        _cli._REPL_STATE["busy_since"] = None
        CONTROL.set_activity("")
        self._input.disabled = False
        self._input.placeholder = (
            "Ask a question or give a task… (/ commands, Shift+Enter newline)"
        )
        self._input.focus()

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

    def action_quit(self) -> None:
        self.exit()

    def action_cancel_task(self) -> None:
        # A pending approval blocks the worker — deny it so the cancel lands.
        if self._pending and self._pending.get("kind") == "approval":
            pend = self._pending
            self._pending = None
            pend["holder"]["ok"] = False
            pend["event"].set()

        if self._busy and not CONTROL.cancelled():
            CONTROL.cancel()
            CONTROL.set_activity("cancelling…")
            self.write_log(
                "\n[yellow]Cancelling — stops after the current step…[/yellow]"
            )

    def action_clear_log(self) -> None:
        self._log.clear()

    def action_scroll_up(self) -> None:
        self._log.scroll_page_up(animate=False)

    def action_scroll_down(self) -> None:
        self._log.scroll_page_down(animate=False)

    def action_scroll_top(self) -> None:
        self._log.scroll_home(animate=False)

    def action_scroll_bottom(self) -> None:
        self._log.scroll_end(animate=False)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_tui(resume_context: str | None = None) -> None:
    """Launch the Textual TUI. Called by run_repl()."""
    try:
        from coding_harness import setup
        setup.first_run_setup()
    except Exception:
        pass

    from coding_harness.main import _ensure_model_configured
    if not _ensure_model_configured():
        return

    WellsApp(resume_context=resume_context).run(mouse=False)
