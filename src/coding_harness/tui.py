"""Full-screen Textual TUI for Wells coding harness.

Replaces the prompt_toolkit REPL with a proper layout:
  ┌─────────────────────────────────────────────┐
  │  RichLog  — scrollable output               │
  ├─────────────────────────────────────────────┤
  │  Input    — user prompt                     │
  ├─────────────────────────────────────────────┤
  │  StatusBar — always visible, refreshes live │
  └─────────────────────────────────────────────┘

Output is captured from both sys.stdout (bare print / streaming tokens)
and a Rich console proxy (Rich markup / tables / panels from cli.py).
Both paths write to the RichLog widget via call_from_thread, keeping
the status bar visible at all times regardless of what's running.
"""

from __future__ import annotations

import sys
import time as _time
from typing import Any

from textual import work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.widgets import Input, OptionList, RichLog, Static
from textual.widgets._option_list import Option

from coding_harness import chat, config
from coding_harness.tokens import LEDGER

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
    height: 4;
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
    height: 3;
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
    """Persistent status bar: workspace | model | tokens | mode."""

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
            force = _cli._REPL_STATE.get("force_mode")
        except Exception:
            force = None

        if force == "chat":
            mode = "[bold yellow]mode: chat[/bold yellow]"
        elif force == "task":
            mode = "[bold magenta]mode: task[/bold magenta]"
        else:
            mode = "[dim]mode: auto[/dim]"

        return (
            f"[dim]{wd}[/dim]  "
            f"[green]{model}[/green]  "
            f"[cyan]{tokens_s}[/cyan]  "
            f"{mode}"
        )


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
                        self._app.call_from_thread(self._app._log.write, line)
                self._buf = lines[-1]
        return len(text)

    def flush(self) -> None:
        if self._buf:
            self._app.call_from_thread(self._app._log.write, self._buf)
            self._buf = ""

    def fileno(self) -> int:
        raise OSError("_TUIStdout has no file descriptor")

    def isatty(self) -> bool:
        return False


class _TUIConsole:
    """Drop-in for Rich Console inside cli.py; routes markup to RichLog.

    thread_safe=True  → use call_from_thread (worker threads)
    thread_safe=False → call widget directly   (asyncio main thread)
    """

    def __init__(self, app: "WellsApp", *, thread_safe: bool = True) -> None:
        self._app = app
        self._thread_safe = thread_safe

    def _write(self, renderable: Any) -> None:
        if self._thread_safe:
            self._app.call_from_thread(self._app._log.write, renderable)
        else:
            self._app._log.write(renderable)

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
        Binding("ctrl+d", "quit", "Quit"),
        Binding("escape", "cancel_task", "Cancel task"),
        Binding("ctrl+l", "clear_log", "Clear output"),
    ]

    def __init__(self, resume_context: str | None = None) -> None:
        super().__init__()
        self._resume_context = resume_context
        self._graph_app: Any = None
        self._agent_state: dict = {}
        self._busy = False
        # Pending interactive command waiting for a follow-up reply.
        # Format: {"kind": "resume_select"|"sessions_clear", "data": ...}
        self._pending: dict | None = None

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
            yield Input(
                placeholder="Ask a question or give a task… (/ for commands, Ctrl+C to quit)",
                id="input",
            )
            yield StatusBar()

    # ------------------------------------------------------------------
    # Startup
    # ------------------------------------------------------------------

    def on_mount(self) -> None:
        from coding_harness.cli import _REPL_STATE
        from coding_harness.graph import build_graph

        self._log: RichLog = self.query_one("#output", RichLog)
        self._input: Input = self.query_one("#input", Input)
        self._cmdlist: OptionList = self.query_one("#command-list", OptionList)

        # Initialize shared REPL state.
        _REPL_STATE["memory"] = chat.ConversationMemory()
        _REPL_STATE["force_mode"] = None
        _REPL_STATE["last_state"] = {}
        _REPL_STATE["resume_context"] = self._resume_context
        _REPL_STATE["resume_session_id"] = None

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

    # ------------------------------------------------------------------
    # Output helpers
    # ------------------------------------------------------------------

    def write_log(self, renderable: Any) -> None:
        """Write to the RichLog from the asyncio event loop thread."""
        self._log.write(renderable)

    def _print_welcome(self) -> None:
        self.write_log("\n[bold blue]Wells Coding Harness[/bold blue]")
        self.write_log(f"[dim]Model:[/dim] [green]{config.model_name_for_task('coding')}[/green]")
        self.write_log(
            f"[dim]Workspace:[/dim] [bold]{config.WORKSPACE_ROOT}[/bold]"
            f"  [dim](safety: {config.HARNESS_SAFETY})[/dim]"
        )
        self.write_log(
            "Ask a question [dim](auto chat)[/dim] or give a task "
            "[dim](auto agent)[/dim]. Type [bold]/[/bold] for commands.\n"
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
    # Input handling
    # ------------------------------------------------------------------

    def on_input_changed(self, event: Input.Changed) -> None:
        """Show/hide the command popup as the user types."""
        from coding_harness.cli import SLASH_COMMANDS
        value = event.value
        if value.startswith("/"):
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
            self._input.value = cmd + " "
            self._input.cursor_position = len(self._input.value)
        self._cmdlist.display = False
        self._input.focus()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        self._cmdlist.display = False
        text = event.value.strip()
        self._input.clear()
        if not text:
            return

        # Handle pending interactive confirmations first.
        if self._pending:
            self._handle_pending_reply(text)
            return

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
        """Run slash commands synchronously on the asyncio thread."""
        import coding_harness.cli as cli_mod

        # Swap console to TUI console (main-thread variant, no call_from_thread).
        orig = cli_mod.console
        cli_mod.console = _TUIConsole(self, thread_safe=False)
        try:
            keep_running = cli_mod.handle_slash_command(command)
        finally:
            cli_mod.console = orig

        if not keep_running:
            self.exit()
            return

        # Special handling for interactive subcommands that need user input.
        self._intercept_interactive(command)

    def _intercept_interactive(self, command: str) -> None:
        """Replace raw input() prompts with a pending-reply state machine."""
        parts = command.strip().lower().split()
        cmd = parts[0] if parts else ""
        sub = parts[1] if len(parts) > 1 else ""

        if cmd == "/resume" and not self._pending:
            # show_resume_list was already called by handle_slash_command's
            # _handle_resume_cmd — but that hit input() which did nothing in TUI.
            # Re-trigger our TUI-native picker.
            self._start_resume_picker(parts[1] if len(parts) > 1 else "")
        elif cmd == "/sessions" and sub == "clear":
            scope = "ALL workspaces" if "--all" in parts else "this workspace"
            self.write_log(
                f"[yellow]Delete all sessions for {scope}?[/yellow] "
                "[dim]Type [bold]y[/bold] to confirm or anything else to cancel.[/dim]"
            )
            self._pending = {
                "kind": "sessions_clear",
                "all_ws": "--all" in parts,
            }

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
    # Task / chat runner (worker thread)
    # ------------------------------------------------------------------

    def _start_run(self, text: str) -> None:
        self._busy = True
        self._input.disabled = True
        self._input.placeholder = "Working…"
        self.write_log(f"\n[bold cyan]>[/bold cyan] {text}\n")
        self._run_input(text)

    @work(thread=True)
    def _run_input(self, text: str) -> None:
        """Run chat or task in a worker thread with redirected I/O."""
        import coding_harness.cli as cli_mod
        from coding_harness.cli import (
            _REPL_STATE, _run_chat, _run_task, _summarize_run,
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

            from coding_harness.cli import StreamingCallback
            callbacks = [StreamingCallback()]

            if intent == "chat":
                _run_chat(text, callbacks)
            else:
                _run_task(text, self._agent_state, self._graph_app, callbacks)
                _REPL_STATE["memory"].set_run_summary(
                    _summarize_run(_REPL_STATE.get("last_state", {}))
                )
        except Exception as e:
            self.call_from_thread(
                self.write_log, f"[bold red]Error:[/bold red] {e}"
            )
        finally:
            # Flush any buffered output before restoring.
            tui_stdout.flush()
            cli_mod.console = orig_cli_console
            sys.stdout = orig_stdout
            sys.stderr = orig_stderr

            self._busy = False
            self.call_from_thread(self._restore_input)

    def _restore_input(self) -> None:
        self._input.disabled = False
        self._input.placeholder = (
            "Ask a question or give a task… (/ for commands, Ctrl+C to quit)"
        )
        self._input.focus()

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

    def action_quit(self) -> None:
        self.exit()

    def action_cancel_task(self) -> None:
        if self._busy:
            self.workers.cancel_all()
            self.write_log("\n[yellow]Task cancelled.[/yellow]")
            self._busy = False
            self._restore_input()

    def action_clear_log(self) -> None:
        self._log.clear()


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

    WellsApp(resume_context=resume_context).run()
