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
from textual.containers import Horizontal, Vertical
from textual.message import Message
from textual.screen import ModalScreen
from textual.widgets import Input, OptionList, RichLog, Static, TextArea
from textual.widgets._option_list import Option

from wells import chat, config
from wells.control import CONTROL, UIEvent
from wells.tokens import LEDGER

_HISTORY_FILE = Path.home() / ".wells" / "history.json"
_HISTORY_MAX = 200
_UI_PREFS_FILE = Path.home() / ".wells" / "ui.json"

# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------

_CSS = """
Screen {
    background: $background;
}

#main {
    layout: horizontal;
    height: 1fr;
}

#log-col {
    width: 1fr;
    height: 100%;
}

#output {
    width: 100%;
    height: 1fr;
    border: none;
    padding: 0 1;
    scrollbar-gutter: stable;
}

#live {
    display: none;
    height: auto;
    max-height: 8;
    padding: 0 1;
    color: $text-muted;
}

InfoPanel {
    width: 34;
    height: 100%;
    border-left: solid $accent 40%;
    background: $surface-darken-1;
    padding: 0 1;
    overflow-y: auto;
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
        self._liab_cache: tuple[float, int] = (0.0, 0)  # (checked_at, count)
        self._last_build = 0.0
        self.set_interval(0.25, self._refresh)

    def _refresh(self) -> None:
        # Busy: full 4 Hz so elapsed/activity visibly tick. Idle: 1 Hz is plenty.
        try:
            import wells.cli as _cli
            busy = _cli._REPL_STATE.get("busy_since") is not None
        except Exception:
            busy = False
        now = _time.monotonic()
        if not busy and now - self._last_build < 1.0:
            return
        self._last_build = now
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

        saved_s = f" [dim]saved {saved:,}[/dim]" if saved else ""
        cost_s = ""
        try:
            from wells import pricing
            c = pricing.fmt(pricing.run_cost())
            if c:
                cost_s = f" [bold]{c}[/bold]"
        except Exception:
            pass
        tokens_s = f"tokens {used:,}{cost_s}{saved_s}"

        try:
            import wells.cli as _cli
            state = _cli._REPL_STATE
            force = state.get("force_mode")
            busy_since = state.get("busy_since")
            pinned = len(state.get("pinned") or [])
        except Exception:
            force = None
            busy_since = None
            pinned = 0

        # Operating mode: plan (read-only) / approve (confirm) / dryrun / auto.
        if config.PLAN_MODE:
            mode = "[bold yellow]mode: plan[/bold yellow]"
        elif config.HARNESS_SAFETY == "approve":
            mode = "[bold cyan]mode: approve[/bold cyan]"
        elif config.HARNESS_SAFETY == "dryrun":
            mode = "[bold magenta]mode: dryrun[/bold magenta]"
        else:
            mode = "[green]mode: auto[/green]"

        route_s = "  [bold magenta]route: orchestrate[/bold magenta]" if force == "task" else ""
        pin_s = f"  [yellow]📌{pinned}[/yellow]" if pinned else ""
        try:
            nq = len(self.app._queue)  # type: ignore[attr-defined]
            queue_s = f"  [yellow]⏳{nq}[/yellow]" if nq else ""
        except Exception:
            queue_s = ""

        # Open rule liabilities (e.g. a rented GPU still running) — always red.
        # Cached: the check reads a file, and this bar rebuilds up to 4×/s.
        liab_s = ""
        try:
            now = _time.monotonic()
            checked_at, n_liab = self._liab_cache
            if now - checked_at >= 5.0:
                from wells import rules as _rules
                n_liab = len(_rules.engine_for(config.WORKSPACE_ROOT).open_liabilities())
                self._liab_cache = (now, n_liab)
            if n_liab:
                liab_s = f"  [bold red blink]⚠{n_liab} LIABILITY[/bold red blink]"
        except Exception:
            pass

        if busy_since is not None:
            secs = int(_time.monotonic() - busy_since)
            if secs >= 60:
                elapsed = f"[yellow]{secs // 60}m {secs % 60:02d}s[/yellow]"
            else:
                elapsed = f"[dim]{secs}s[/dim]"
            activity, act_age = CONTROL.activity_info()
            # Ticking per-activity age = constant "still alive" signal even
            # when the model produces no output for a while.
            age_s = f" [dim]· {int(act_age)}s[/dim]" if act_age >= 3 else ""
            act_s = f"  [magenta]{activity}[/magenta]{age_s}" if activity else ""
            elapsed_s = f"  {elapsed}{act_s}  [dim]esc: cancel[/dim]"
        else:
            elapsed_s = ""

        return (
            f"[dim]{wd}[/dim]  "
            f"[green]{model}[/green]  "
            f"[cyan]{tokens_s}[/cyan]{elapsed_s}  "
            f"{mode}{route_s}{pin_s}{queue_s}{liab_s}"
        )


# ---------------------------------------------------------------------------
# Right-side info panel (F2 toggles; the bottom bar takes over when hidden)
# ---------------------------------------------------------------------------

class InfoPanel(Static):
    """Always-visible session dashboard: workspace, model, mode, tokens/$,
    activity, pinned files, MCP servers, index, git, liabilities."""

    def on_mount(self) -> None:
        self._slow_cache: dict = {}   # 30s-cached expensive lookups
        self._slow_at = 0.0
        self._rules_cache: tuple[float, list] = (0.0, [])  # 5s TTL (file read)
        self.set_interval(1.0, self._refresh)
        self._refresh()

    def _open_liabilities(self) -> list:
        now = _time.monotonic()
        checked_at, items = self._rules_cache
        if now - checked_at >= 5.0:
            from wells import rules as _rules
            items = _rules.engine_for(config.WORKSPACE_ROOT).open_liabilities()
            self._rules_cache = (now, items)
        return items

    def _refresh(self) -> None:
        try:
            self.update(self._build())
        except Exception:
            pass

    # -- cached expensive lookups (index stats, git branch) -----------------

    def _slow(self) -> dict:
        if _time.monotonic() - self._slow_at < 30:
            return self._slow_cache
        out: dict = {}
        try:
            from wells.index_tools import INDEXER_AVAILABLE, index_status
            if INDEXER_AVAILABLE:
                st = index_status(config.WORKSPACE_ROOT)
                if st.get("exists"):
                    out["index"] = (
                        f"{st['total_symbols']:,} syms / {st['total_files']:,} files"
                    )
                else:
                    out["index"] = "[yellow]not built[/yellow]"
        except Exception:
            pass
        try:
            from wells import gitops
            ok, branch = gitops._git(
                config.WORKSPACE_ROOT, "rev-parse", "--abbrev-ref", "HEAD",
                timeout=5,
            )
            if ok and branch.strip():
                out["branch"] = branch.strip().splitlines()[-1][:24]
        except Exception:
            pass
        self._slow_cache = out
        self._slow_at = _time.monotonic()
        return out

    def _build(self) -> str:
        import wells.cli as _cli
        from wells import pricing

        L: list[str] = []
        rule = "[dim]" + "─" * 30 + "[/dim]"

        def row(label: str, value: str) -> None:
            L.append(f"[dim]{label:<9}[/dim]{value}")

        # -- identity ---------------------------------------------------------
        L.append("[bold blue]Wells[/bold blue]")
        L.append(rule)
        ws = config.WORKSPACE_ROOT
        if len(ws) > 22:
            ws = "…" + ws[-21:]
        row("dir", f"[white]{ws}[/white]")
        slow = self._slow()
        if slow.get("branch"):
            row("branch", f"[white]{slow['branch']}[/white]")
        try:
            row("model", f"[green]{config.model_name_for_task('coding')}[/green]")
        except Exception:
            pass

        mode_map = {
            "plan": "[bold yellow]plan[/bold yellow]",
            "approve": "[bold cyan]approve[/bold cyan]",
            "dryrun": "[bold magenta]dryrun[/bold magenta]",
            "auto": "[green]auto[/green]",
        }
        row("mode", mode_map.get(_cli.current_mode(), _cli.current_mode()))
        if _cli._REPL_STATE.get("force_mode") == "task":
            row("route", "[bold magenta]orchestrate[/bold magenta]")

        # -- usage --------------------------------------------------------------
        L.append(rule)
        try:
            t = LEDGER.totals()
            used = t["input"] + t["output"]
            row("tokens", f"[cyan]{used:,}[/cyan]")
            cost = pricing.fmt(pricing.run_cost())
            if cost:
                row("cost", f"[bold]{cost}[/bold]")
            saved = t["saved_trim"] + t["saved_summary"]
            if saved:
                row("saved", f"[green]{saved:,}[/green]")
            if t["cache_read"]:
                row("cache", f"[dim]{t['cache_read']:,}[/dim]")
        except Exception:
            pass

        # -- activity -------------------------------------------------------------
        busy_since = _cli._REPL_STATE.get("busy_since")
        if busy_since is not None:
            L.append(rule)
            secs = int(_time.monotonic() - busy_since)
            elapsed = f"{secs // 60}m {secs % 60:02d}s" if secs >= 60 else f"{secs}s"
            row("elapsed", f"[yellow]{elapsed}[/yellow]")
            act, act_age = CONTROL.activity_info()
            if act:
                age_s = f" [dim]{int(act_age)}s[/dim]" if act_age >= 3 else ""
                L.append(f"[magenta]{act[:31]}[/magenta]{age_s}")
            L.append("[dim]esc: cancel · /steer · /btw[/dim]")

        # -- orchestrate pipeline breadcrumb ---------------------------------------
        stages = CONTROL.stages()
        if stages:
            L.append(rule)
            L.append("[bold]pipeline[/bold]")
            for name, status, secs_f in stages[-8:]:
                if status == "run":
                    L.append(f"[yellow]▶[/yellow] {name[:14]:<14}[yellow]{int(secs_f)}s[/yellow]")
                elif status == "fail":
                    L.append(f"[red]✗[/red] [dim]{name[:14]:<14}{int(secs_f)}s[/dim]")
                else:
                    L.append(f"[green]✓[/green] [dim]{name[:14]:<14}{int(secs_f)}s[/dim]")

        # -- per-stage step progress (cap 0 = No Limit) ----------------------------
        prog = CONTROL.progress()
        if prog:
            L.append(rule)
            L.append("[bold]progress[/bold]")
            for label, cur, cap in prog[-8:]:
                cap_disp = "[green]No Limit[/green]" if cap == 0 else str(cap)
                current = label == prog[-1][0] and busy_since is not None
                marker = "[yellow]▶[/yellow] " if current else "  "
                L.append(f"{marker}[dim]{label[:13]:<13}[/dim]{cur} [dim]/[/dim] {cap_disp}")
        try:
            queued = len(self.app._queue)  # type: ignore[attr-defined]
        except Exception:
            queued = 0
        if queued:
            row("queued", f"[yellow]⏳ {queued}[/yellow]")

        # -- context ---------------------------------------------------------------
        pinned = _cli._REPL_STATE.get("pinned") or []
        mcp_live: dict = {}
        try:
            from wells import mcp_client
            mcp_live = mcp_client.connected()
        except Exception:
            pass
        if pinned or mcp_live or slow.get("index"):
            L.append(rule)
            if pinned:
                row("pinned", f"[yellow]📌 {len(pinned)}[/yellow]")
                for p in pinned[:4]:
                    L.append(f"  [dim]{p[-27:]}[/dim]")
                if len(pinned) > 4:
                    L.append(f"  [dim]… +{len(pinned) - 4}[/dim]")
            if mcp_live:
                row("mcp", f"[green]{len(mcp_live)} connected[/green]")
                for name, tools_ in list(mcp_live.items())[:4]:
                    L.append(f"  [dim]{name} ({len(tools_)})[/dim]")
            if slow.get("index"):
                row("index", f"[dim]{slow['index']}[/dim]")

        # -- liabilities -----------------------------------------------------------
        try:
            open_l = self._open_liabilities()
            if open_l:
                L.append(rule)
                L.append(f"[bold red]⚠ {len(open_l)} OPEN LIABILITY[/bold red]")
                for l in open_l[:3]:
                    L.append(f"  [red]{l['rule_id'][:28]}[/red]")
                L.append("[dim]/rules to resolve[/dim]")
        except Exception:
            pass

        L.append(rule)
        L.append("[dim]F2: hide panel[/dim]")
        return "\n".join(L)


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
# Settings modal (/config) — replaces the stdin menu, which would deadlock
# the event loop (Textual owns stdin, so input() never returns).
# ---------------------------------------------------------------------------

class SettingsScreen(ModalScreen["dict | None"]):
    """Modal settings editor over the ``settings.SETTINGS`` schema.

    Enter on a row edits it; Enter commits the value; Escape backs out of the
    edit, then closes the panel. Dismisses with the staged {key: value}
    changes (or None) — the app persists them to .env and reloads config.
    """

    CSS = """
    SettingsScreen {
        align: center middle;
    }
    #settings-panel {
        width: 96;
        max-width: 96%;
        height: 80%;
        border: thick $accent;
        background: $surface;
        padding: 1 2;
    }
    #settings-title {
        height: 1;
        text-style: bold;
    }
    #settings-list {
        height: 1fr;
        border: none;
    }
    #settings-help {
        height: 2;
        color: $text-muted;
    }
    #settings-input {
        display: none;
        height: 3;
    }
    """

    BINDINGS = [Binding("escape", "back", "Back / close", priority=True)]

    def __init__(self) -> None:
        super().__init__()
        self._staged: dict[str, str] = {}
        self._editing = None  # settings.Setting | None

    def compose(self) -> ComposeResult:
        with Vertical(id="settings-panel"):
            yield Static("Wells Settings  [dim](Enter: edit · Esc: close & save)[/dim]",
                         id="settings-title", markup=True)
            yield OptionList(id="settings-list")
            yield Static("", id="settings-help", markup=True)
            yield Input(id="settings-input")

    def on_mount(self) -> None:
        self._populate()
        self.query_one("#settings-list", OptionList).focus()

    def _display_value(self, s) -> str:
        from wells import settings as settings_mod
        v = self._staged.get(s.key, settings_mod.current_value(s))
        shown = settings_mod.mask(v) if s.secret else (v or "(default)")
        if len(shown) > 42:
            shown = shown[:41] + "…"
        return shown

    def _populate(self) -> None:
        from wells import settings as settings_mod

        lst = self.query_one("#settings-list", OptionList)
        highlighted = lst.highlighted
        lst.clear_options()
        seen_cat = None
        for s in settings_mod.SETTINGS:
            if s.category != seen_cat:
                seen_cat = s.category
                lst.add_option(Option(f"[bold cyan]── {s.category} ──[/bold cyan]",
                                      id=f"_cat_{s.category}", disabled=True))
            star = "[yellow]*[/yellow]" if s.key in self._staged else " "
            lst.add_option(Option(
                f"{star}{s.key:<28} [green]{self._display_value(s)}[/green]  "
                f"[dim]{s.label}[/dim]",
                id=s.key,
            ))
        if highlighted is not None:
            lst.highlighted = min(highlighted, lst.option_count - 1)

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        from wells import settings as settings_mod

        event.stop()  # don't bubble to the app's command-popup handler
        key = event.option_id or ""
        s = settings_mod.SETTINGS_BY_KEY.get(key)
        if s is None:
            return
        self._editing = s
        help_widget = self.query_one("#settings-help", Static)
        choices = f"  [bold]choices:[/bold] {', '.join(s.choices)}" if s.choices else ""
        help_widget.update(f"[bold]{s.key}[/bold] — {s.help}{choices}")
        box = self.query_one("#settings-input", Input)
        current = self._staged.get(key, settings_mod.current_value(s))
        box.value = "" if s.secret else current
        box.placeholder = (
            "(hidden — type a new value, Enter to keep current)" if s.secret
            else "new value (Enter to commit, Esc to cancel)"
        )
        box.display = True
        box.focus()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        from wells import settings as settings_mod

        event.stop()
        s = self._editing
        if s is None:
            return
        value = event.value.strip()
        if s.choices and value and value not in s.choices:
            self.query_one("#settings-help", Static).update(
                f"[red]{s.key} must be one of: {', '.join(s.choices)}[/red]"
            )
            return
        if value and value != settings_mod.current_value(s):
            self._staged[s.key] = value
        self._close_editor()
        self._populate()

    def _close_editor(self) -> None:
        self._editing = None
        box = self.query_one("#settings-input", Input)
        box.display = False
        self.query_one("#settings-help", Static).update("")
        self.query_one("#settings-list", OptionList).focus()

    def action_back(self) -> None:
        if self._editing is not None:
            self._close_editor()
        else:
            self.dismiss(self._staged or None)


# ---------------------------------------------------------------------------
# MCP server manager (/mcp) — same interaction model as the settings modal.
# ---------------------------------------------------------------------------

class MCPAddScreen(ModalScreen["dict | None"]):
    """Small form to add an MCP server: name, command, args, env."""

    CSS = """
    MCPAddScreen { align: center middle; }
    #mcp-add-panel {
        width: 80; max-width: 96%; height: auto;
        border: thick $accent; background: $surface; padding: 1 2;
    }
    #mcp-add-panel Static { height: auto; }
    #mcp-add-panel Input { margin-bottom: 1; }
    """

    BINDINGS = [Binding("escape", "cancel", "Cancel", priority=True)]

    def compose(self) -> ComposeResult:
        with Vertical(id="mcp-add-panel"):
            yield Static("[bold]Add MCP server[/bold]  [dim](Enter moves to next field; "
                         "Enter on the last field saves · Esc cancels)[/dim]", markup=True)
            yield Static("Name [dim](e.g. fetch)[/dim]", markup=True)
            yield Input(id="mcp-name", placeholder="server name")
            yield Static("Command [dim](e.g. uvx or npx)[/dim]", markup=True)
            yield Input(id="mcp-command", placeholder="executable")
            yield Static("Arguments [dim](space separated, e.g. -y @modelcontextprotocol/server-filesystem C:/data)[/dim]", markup=True)
            yield Input(id="mcp-args", placeholder="(optional)")
            yield Static("Environment [dim](KEY=VALUE pairs, space separated)[/dim]", markup=True)
            yield Input(id="mcp-env", placeholder="(optional, e.g. GITHUB_PERSONAL_ACCESS_TOKEN=ghp_…)")
            yield Static("", id="mcp-add-error", markup=True)

    def on_mount(self) -> None:
        self.query_one("#mcp-name", Input).focus()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        event.stop()
        order = ["mcp-name", "mcp-command", "mcp-args", "mcp-env"]
        cur = event.input.id or ""
        if cur in order[:-1]:
            self.query_one(f"#{order[order.index(cur) + 1]}", Input).focus()
            return
        # Last field — validate and dismiss with the spec.
        name = self.query_one("#mcp-name", Input).value.strip()
        command = self.query_one("#mcp-command", Input).value.strip()
        if not name or not command:
            self.query_one("#mcp-add-error", Static).update(
                "[red]Name and command are required.[/red]"
            )
            return
        spec: dict = {"command": command}
        args = self.query_one("#mcp-args", Input).value.split()
        if args:
            spec["args"] = args
        env_pairs = self.query_one("#mcp-env", Input).value.split()
        env = {k: v for k, _, v in (p.partition("=") for p in env_pairs) if k and v}
        if env:
            spec["env"] = env
        self.dismiss({"name": name, "spec": spec})

    def action_cancel(self) -> None:
        self.dismiss(None)


class MCPScreen(ModalScreen[None]):
    """Modal MCP server manager: list, add, enable/disable, test, remove."""

    CSS = """
    MCPScreen { align: center middle; }
    #mcp-panel {
        width: 100; max-width: 96%; height: 75%;
        border: thick $accent; background: $surface; padding: 1 2;
    }
    #mcp-title { height: 2; text-style: bold; }
    #mcp-list { height: 1fr; border: none; }
    #mcp-status { height: 2; color: $text-muted; }
    """

    BINDINGS = [
        Binding("escape", "close", "Close", priority=True),
        Binding("a", "add", "Add"),
        Binding("e", "toggle", "Enable/Disable"),
        Binding("t", "test", "Test"),
        Binding("d", "remove", "Remove"),
    ]

    def compose(self) -> ComposeResult:
        with Vertical(id="mcp-panel"):
            yield Static(
                "MCP Servers  [dim](a: add · e: enable/disable · t: test connection · "
                "d: remove · Esc: close)[/dim]",
                id="mcp-title", markup=True,
            )
            yield OptionList(id="mcp-list")
            yield Static("", id="mcp-status", markup=True)

    def on_mount(self) -> None:
        self._populate()
        self.query_one("#mcp-list", OptionList).focus()
        from wells import mcp_client as mc
        if mc.env_override_active():
            self._status(
                "[yellow]MCP_SERVERS env var is set — it overrides mcp.json edits.[/yellow]"
            )

    # -- rendering ---------------------------------------------------------

    def _status(self, text: str) -> None:
        self.query_one("#mcp-status", Static).update(text)

    def _row_state(self, name: str) -> str:
        """State of the highlighted row: connected|enabled|disabled|example."""
        from wells import mcp_client as mc
        data = mc.read_file_config()
        if name in mc.connected():
            return "connected"
        if name in data and not name.startswith("_"):
            return "enabled"
        if name in (data.get("_disabled") or {}):
            return "disabled"
        return "example"

    def _populate(self) -> None:
        from wells import mcp_client as mc

        lst = self.query_one("#mcp-list", OptionList)
        highlighted = lst.highlighted
        lst.clear_options()
        data = mc.read_file_config()
        live = mc.connected()
        active = {k: v for k, v in data.items()
                  if not k.startswith("_") and isinstance(v, dict)}
        disabled = data.get("_disabled") or {}
        examples = data.get("_examples") or {}

        def _cmd(spec: dict) -> str:
            return (f"{spec.get('command', '?')} " + " ".join(spec.get("args") or []))[:52]

        def _header(label: str) -> None:
            lst.add_option(Option(f"[bold cyan]── {label} ──[/bold cyan]",
                                  id=f"_hdr_{label}", disabled=True))

        if active:
            _header("Configured")
            for name, spec in active.items():
                if name in live:
                    state = f"[green]connected · {len(live[name])} tool(s)[/green]"
                else:
                    state = "[yellow]enabled (not connected)[/yellow]"
                lst.add_option(Option(
                    f"{name:<14} {state}  [dim]{_cmd(spec)}[/dim]", id=name))
        if disabled:
            _header("Disabled")
            for name, spec in disabled.items():
                lst.add_option(Option(
                    f"[dim]{name:<14} disabled  {_cmd(spec)}[/dim]", id=name))
        shown = set(active) | set(disabled)
        ex = {k: v for k, v in examples.items() if k not in shown}
        if ex:
            _header("Examples (press e to enable)")
            for name, spec in ex.items():
                lst.add_option(Option(
                    f"[dim]{name:<14} example   {_cmd(spec)}[/dim]", id=name))
        if lst.option_count == 0:
            lst.add_option(Option("[dim]No servers configured — press a to add one.[/dim]",
                                  id="_hdr_empty", disabled=True))
        if highlighted is not None and lst.option_count:
            lst.highlighted = min(highlighted, lst.option_count - 1)

    def _selected(self) -> str | None:
        lst = self.query_one("#mcp-list", OptionList)
        if lst.highlighted is None:
            return None
        opt = lst.get_option_at_index(lst.highlighted)
        name = opt.id or ""
        return None if name.startswith("_hdr_") else name

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        event.stop()
        self.action_test()  # Enter on a row = test its connection

    # -- actions (connects run in threads; UI stays live) --------------------

    def _bg(self, fn) -> None:
        threading.Thread(target=fn, daemon=True).start()

    def action_close(self) -> None:
        self.dismiss(None)

    def action_add(self) -> None:
        def _done(result: dict | None) -> None:
            if not result:
                return
            from wells import mcp_client as mc
            name, spec = result["name"], result["spec"]
            mc.add_server(name, spec)
            self._populate()
            self._status(f"[dim]Added '{name}' — connecting…[/dim]")

            def _connect() -> None:
                ok, msg, names = mc.connect_server(name, spec)
                text = (
                    f"[green]{name}: {msg}[/green]" if ok
                    else f"[yellow]{name}: saved, but connect failed — {msg}[/yellow]"
                )
                self.app.call_from_thread(self._status, text)
                self.app.call_from_thread(self._populate)

            self._bg(_connect)

        self.app.push_screen(MCPAddScreen(), _done)

    def action_toggle(self) -> None:
        from wells import mcp_client as mc
        name = self._selected()
        if not name:
            return
        state = self._row_state(name)
        if state in ("connected", "enabled"):
            mc.disconnect_server(name)
            ok, msg = mc.set_enabled(name, False)
            self._status(f"[dim]{name}: {msg if ok else msg}[/dim]")
            self._populate()
            return
        # disabled or example -> enable + connect
        ok, msg = mc.set_enabled(name, True)
        if not ok:
            self._status(f"[yellow]{msg}[/yellow]")
            return
        self._populate()
        self._status(f"[dim]{name}: enabled — connecting…[/dim]")
        spec = mc.load_config().get(name) or {}

        def _connect() -> None:
            ok2, msg2, _names = mc.connect_server(name, spec)
            text = (
                f"[green]{name}: {msg2}[/green]" if ok2
                else f"[yellow]{name}: connect failed — {msg2}[/yellow]"
            )
            self.app.call_from_thread(self._status, text)
            self.app.call_from_thread(self._populate)

        self._bg(_connect)

    def action_test(self) -> None:
        from wells import mcp_client as mc
        name = self._selected()
        if not name:
            return
        data = mc.read_file_config()
        spec = (
            (data.get(name) if not name.startswith("_") else None)
            or (data.get("_disabled") or {}).get(name)
            or (data.get("_examples") or {}).get(name)
        )
        if not isinstance(spec, dict):
            self._status(f"[yellow]No spec found for {name}.[/yellow]")
            return
        self._status(f"[dim]Testing '{name}'…[/dim]")

        def _test() -> None:
            ok, msg, names = mc.connect_server(name, spec)
            if ok:
                tools_s = ", ".join(n.removeprefix(f"mcp_{name}_") for n in names[:8])
                more = f" +{len(names) - 8} more" if len(names) > 8 else ""
                text = f"[green]{name}: {msg}[/green]  [dim]{tools_s}{more}[/dim]"
            else:
                text = f"[red]{name}: {msg}[/red]"
            self.app.call_from_thread(self._status, text)
            self.app.call_from_thread(self._populate)

        self._bg(_test)

    def action_remove(self) -> None:
        from wells import mcp_client as mc
        name = self._selected()
        if not name:
            return
        if self._row_state(name) == "example":
            self._status("[dim]Examples can't be removed — they're just templates.[/dim]")
            return
        mc.disconnect_server(name)
        spec = mc.remove_server(name)
        if spec is None:
            self._status(f"[yellow]Not found: {name}[/yellow]")
        else:
            import json as _json
            self._status(
                f"[green]Removed '{name}'.[/green] [dim]Was: {_json.dumps(spec)[:70]}[/dim]"
            )
        self._populate()


# ---------------------------------------------------------------------------
# Skills manager (/skills) — list / view / add / edit / remove SKILL.md files.
# Same interaction model as the MCP modal.
# ---------------------------------------------------------------------------

class SkillEditScreen(ModalScreen["dict | None"]):
    """Form to create or edit a skill: name, description, body.

    When ``existing`` is provided, the fields are pre-filled for editing
    (and the name field is read-only). Otherwise it's a new-skill form.
    """

    CSS = """
    SkillEditScreen { align: center middle; }
    #skill-edit-panel {
        width: 90; max-width: 96%; height: auto; max-height: 85%;
        border: thick $accent; background: $surface; padding: 1 2;
    }
    #skill-edit-panel Static { height: auto; }
    #skill-edit-panel Input { margin-bottom: 1; }
    #skill-body { height: 12; margin-bottom: 1; }
    #skill-edit-error { height: auto; }
    """

    BINDINGS = [
        Binding("escape", "cancel", "Cancel", priority=True),
        Binding("ctrl+s", "save", "Save", priority=True),
    ]

    def __init__(self, existing: dict | None = None) -> None:
        self._existing = existing  # {"name", "description", "body"} or None
        super().__init__()

    def compose(self) -> ComposeResult:
        is_edit = self._existing is not None
        title = "[bold]Edit skill[/bold]" if is_edit else "[bold]Add skill[/bold]"
        with Vertical(id="skill-edit-panel"):
            yield Static(
                f"{title}  [dim](Ctrl+S saves · Esc cancels)[/dim]",
                markup=True,
            )
            yield Static("Name [dim](lowercase, digits, hyphens)[/dim]", markup=True)
            name_val = self._existing["name"] if self._existing else ""
            name_input = Input(id="skill-name", value=name_val)
            if is_edit:
                name_input.disabled = True
            yield name_input
            yield Static("Description [dim](one line shown to the agent)[/dim]", markup=True)
            desc_val = self._existing.get("description", "") if self._existing else ""
            yield Input(id="skill-desc", value=desc_val)
            yield Static("Body [dim](markdown instructions — what the agent loads)[/dim]", markup=True)
            body_val = self._existing.get("body", "") if self._existing else ""
            yield TextArea(body_val, id="skill-body", language="markdown")
            yield Static("", id="skill-edit-error", markup=True)

    def on_mount(self) -> None:
        if self._existing:
            self.query_one("#skill-desc", Input).focus()
        else:
            self.query_one("#skill-name", Input).focus()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        event.stop()
        order = ["skill-name", "skill-desc"]
        cur = event.input.id or ""
        if cur in order and cur != order[-1]:
            self.query_one(f"#{order[order.index(cur) + 1]}", Input).focus()
        elif cur == "skill-desc":
            self.query_one("#skill-body", TextArea).focus()

    def action_save(self) -> None:
        name = self.query_one("#skill-name", Input).value.strip()
        desc = self.query_one("#skill-desc", Input).value.strip()
        body = self.query_one("#skill-body", TextArea).text
        if self._existing is None:
            from wells import skills as sk
            err = sk.validate_name(name)
            if err:
                self.query_one("#skill-edit-error", Static).update(f"[red]{err}[/red]")
                return
        self.dismiss({"name": name.lower(), "description": desc, "body": body})

    def action_cancel(self) -> None:
        self.dismiss(None)


class SkillsScreen(ModalScreen[None]):
    """Modal skills manager: list, view, add, edit, remove."""

    CSS = """
    SkillsScreen { align: center middle; }
    #skills-panel {
        width: 100; max-width: 96%; height: 75%;
        border: thick $accent; background: $surface; padding: 1 2;
    }
    #skills-title { height: 2; text-style: bold; }
    #skills-list { height: 1fr; border: none; }
    #skills-status { height: 2; color: $text-muted; }
    #skills-view { display: none; height: 1fr; border: solid $accent; padding: 0 1; }
    """

    BINDINGS = [
        Binding("escape", "close", "Close", priority=True),
        Binding("a", "add", "Add"),
        Binding("e", "edit", "Edit"),
        Binding("d", "remove", "Remove"),
        Binding("enter", "view", "View"),
    ]

    _VIEWING = False  # toggle between list view and skill-body view

    def compose(self) -> ComposeResult:
        with Vertical(id="skills-panel"):
            yield Static(
                "Agent Skills  [dim](Enter: view · a: add · e: edit · "
                "d: remove · Esc: close)[/dim]",
                id="skills-title", markup=True,
            )
            yield OptionList(id="skills-list")
            yield Static("", id="skills-view", markup=True, disabled=True)
            yield Static("", id="skills-status", markup=True)

    def on_mount(self) -> None:
        self._populate()
        self.query_one("#skills-list", OptionList).focus()

    # -- rendering ---------------------------------------------------------

    def _status(self, text: str) -> None:
        self.query_one("#skills-status", Static).update(text)

    def _populate(self) -> None:
        from wells import skills as sk

        lst = self.query_one("#skills-list", OptionList)
        highlighted = lst.highlighted
        lst.clear_options()
        idx = sk.skills_for(config.WORKSPACE_ROOT)
        if idx.is_empty():
            lst.add_option(Option(
                "[dim]No skills yet — press a to add one.[/dim]",
                id="_hdr_empty", disabled=True,
            ))
        else:
            for s in idx.skills:
                desc = s.description or "(no description)"
                lst.add_option(Option(f"{s.name}  [dim]— {desc}[/dim]", id=s.name))
        if highlighted is not None and lst.option_count:
            lst.highlighted = min(highlighted, lst.option_count - 1)

    def _selected(self) -> str | None:
        lst = self.query_one("#skills-list", OptionList)
        if lst.highlighted is None:
            return None
        opt = lst.get_option_at_index(lst.highlighted)
        name = opt.id or ""
        return None if name.startswith("_hdr_") else name

    # -- actions -----------------------------------------------------------

    def action_close(self) -> None:
        self.dismiss(None)

    def action_view(self) -> None:
        name = self._selected()
        if not name:
            return
        from wells import skills as sk
        ok, raw = sk.read_skill_raw(name, config.WORKSPACE_ROOT)
        if not ok:
            self._status(f"[red]{raw}[/red]")
            return
        view = self.query_one("#skills-view", Static)
        view.update(f"[bold cyan]── {name} ──[/bold cyan]\n{raw}")
        # Show the full path too for reference.
        path = sk.skill_file_path(name, config.WORKSPACE_ROOT)
        self._status(f"[dim]{path}[/dim]" if path else "")

    def action_add(self) -> None:
        def _done(result: dict | None) -> None:
            if not result:
                return
            from wells import skills as sk
            ok, msg = sk.create_skill(
                result["name"], result["description"], result["body"],
                config.WORKSPACE_ROOT,
            )
            self._populate()
            (self._status(f"[green]{msg}[/green]")
             if ok else self._status(f"[red]{msg}[/red]"))

        self.app.push_screen(SkillEditScreen(existing=None), _done)

    def action_edit(self) -> None:
        name = self._selected()
        if not name:
            self._status("[dim]Select a skill to edit.[/dim]")
            return
        from wells import skills as sk
        skill = sk.skills_for(config.WORKSPACE_ROOT).by_name(name)
        if skill is None:
            self._status(f"[red]Unknown skill: {name}[/red]")
            return
        existing = {"name": skill.name, "description": skill.description, "body": skill.body}

        def _done(result: dict | None) -> None:
            if not result:
                return
            ok, msg = sk.update_skill(
                result["name"], config.WORKSPACE_ROOT,
                description=result["description"] or None,
                body=result["body"],
            )
            self._populate()
            (self._status(f"[green]{msg}[/green]")
             if ok else self._status(f"[red]{msg}[/red]"))

        self.app.push_screen(SkillEditScreen(existing=existing), _done)

    def action_remove(self) -> None:
        name = self._selected()
        if not name:
            self._status("[dim]Select a skill to remove.[/dim]")
            return
        from wells import skills as sk
        ok, msg = sk.delete_skill(name, config.WORKSPACE_ROOT)
        self._populate()
        (self._status(f"[green]{msg}[/green]")
         if ok else self._status(f"[red]{msg}[/red]"))


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
        Binding("f2", "toggle_panel", "Info panel", priority=True),
        # Scroll bindings — priority=True so they fire even when Input has focus.
        # Mouse capture is enabled (mouse=True) so the scroll wheel works in the
        # output panel. Hold Shift while selecting to use the terminal's native
        # copy-paste instead of Textual's mouse capture.
        Binding("pageup",    "scroll_up",     "Scroll up",      show=False, priority=True),
        Binding("pagedown",  "scroll_down",   "Scroll down",    show=False, priority=True),
        Binding("ctrl+home", "scroll_top",    "Scroll to top",  show=False, priority=True),
        Binding("ctrl+end",  "scroll_bottom", "Scroll to end",  show=False, priority=True),
    ]

    def __init__(self, resume_context: str | None = None) -> None:
        super().__init__()
        self._resume_context = resume_context
        self._graph_app: Any = None
        self._graph_lock = threading.Lock()
        self._agent_state: dict = {}
        self._busy = False
        # Bumped by every run and by /stop — see _run_input's finally block.
        self._run_gen = 0
        # Live streaming region state (in-progress model output).
        self._live_buf = ""
        self._live_at = 0.0
        # Pending interactive command waiting for a follow-up reply.
        # Format: {"kind": "resume_select"|"sessions_clear"|"approval", ...}
        self._pending: dict | None = None
        # Messages typed during a run — executed in order when it finishes.
        self._queue: list[str] = []
        # /exit was requested during a run: leave as soon as the worker stops.
        self._exit_after_run = False
        # /btw side-conversation history (session-lived).
        self._btw_history: list[tuple[str, str]] = []
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
        with Horizontal(id="main"):
            with Vertical(id="log-col"):
                yield RichLog(
                    id="output",
                    markup=True,
                    highlight=True,
                    wrap=True,
                )
                # Live streaming region: in-progress model output lands here
                # character-by-character, then commits to the log line-by-line.
                yield Static(id="live")
            yield InfoPanel(id="info-panel")
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
        from wells import safety
        from wells.cli import _REPL_STATE

        self._log: RichLog = self.query_one("#output", RichLog)
        self._live: Static = self.query_one("#live", Static)
        self._input: PromptInput = self.query_one("#input", PromptInput)
        self._cmdlist: OptionList = self.query_one("#command-list", OptionList)
        self._panel: InfoPanel = self.query_one("#info-panel", InfoPanel)
        self._statusbar: StatusBar = self.query_one(StatusBar)
        self._apply_panel_visibility(self._load_panel_pref())

        # Initialize shared REPL state.
        _REPL_STATE["memory"] = chat.ConversationMemory()
        _REPL_STATE["force_mode"] = None
        _REPL_STATE["last_state"] = {}
        _REPL_STATE["resume_context"] = self._resume_context
        _REPL_STATE["resume_session_id"] = None

        # Typed UI events from the executor render directly (no stdout hop).
        CONTROL.set_listener(self._on_ui_event)
        # Seed the global rules file so every workspace gets the defaults.
        try:
            from wells import rules as _rules
            _rules.ensure_global_template()
            if _rules.engine_for(config.WORKSPACE_ROOT).open_liabilities():
                self.write_log(
                    "[bold red]⚠ OPEN LIABILITIES from a previous session — "
                    "run /rules to see them. This may be costing money.[/bold red]"
                )
        except Exception:
            pass
        # Under HARNESS_SAFETY=approve, destructive tool calls ask the user here.
        safety.set_approver(self._tui_approver)

        self._history = self._history_load()

        # Build the LangGraph in the background — it's only needed for
        # orchestrate runs, and building it on the event loop delays first paint.
        threading.Thread(
            target=self._ensure_graph, name="wells-graph-build", daemon=True
        ).start()
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
        self._connect_mcp_servers()
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
            if ev.kind == "llm_chunk":
                self.call_from_thread(self._llm_chunk, ev.text)
            elif ev.kind == "llm_done":
                self.call_from_thread(self._llm_done)
            else:
                self.call_from_thread(self.write_log, ev.text)
        except Exception:
            # Already on the app thread (or app shutting down).
            try:
                self.write_log(ev.text)
            except Exception:
                pass

    # -- live streaming region (in-progress model output) --------------------

    def _llm_chunk(self, text: str) -> None:
        """Append a streamed chunk: full lines commit to the log immediately,
        the partial tail renders in the live region below it (throttled)."""
        from rich.text import Text
        self._live_buf += text
        if "\n" in self._live_buf:
            *done, self._live_buf = self._live_buf.split("\n")
            for line in done:
                self.write_log(Text(line))
        now = _time.monotonic()
        if not self._live_buf:
            self._live.update("")
        elif now - self._live_at >= 0.05:
            self._live_at = now
            self._live.display = True
            self._live.update(Text(self._live_buf[-600:] + "▌"))

    def _llm_done(self) -> None:
        from rich.text import Text
        if self._live_buf:
            self.write_log(Text(self._live_buf))
            self._live_buf = ""
        self._live.update("")
        self._live.display = False

    def _ensure_graph(self) -> Any:
        """Build the LangGraph once (thread-safe); return the compiled app."""
        with self._graph_lock:
            if self._graph_app is None:
                from wells.graph import build_graph
                self._graph_app = build_graph()
            return self._graph_app

    def _print_welcome(self) -> None:
        logo_shown = False
        try:
            from wells.logo import logo_lines
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
            self.write_log("\n[bold blue]Wells[/bold blue]")
        self.write_log(f"[dim]Model:[/dim] [green]{config.model_name_for_task('coding')}[/green]")
        self.write_log(
            f"[dim]Workspace:[/dim] [bold]{config.WORKSPACE_ROOT}[/bold]"
            f"  [dim](safety: {config.HARNESS_SAFETY})[/dim]"
        )
        if getattr(config, "WORKSPACE_ROOT_INVALID", ""):
            self.write_log(
                f"[bold red]⚠ Configured WORKSPACE_ROOT does not exist:[/bold red] "
                f"{config.WORKSPACE_ROOT_INVALID}\n"
                f"[yellow]Falling back to the directory above. Fix it with "
                f"/working-dir <path> or edit WORKSPACE_ROOT in .env.[/yellow]"
            )
        self.write_log(
            "Ask anything — questions, edits, tasks. "
            "Use [bold]/orchestrate[/bold] for complex multi-component work. "
            "Type [bold]/[/bold] for all commands. "
            "[dim]Shift+Enter: newline · ↑/↓: history · Esc: cancel run · F2: info panel[/dim]\n"
        )

    def _ensure_repo_index(self) -> None:
        try:
            from wells import config as _cfg
            if not _cfg.INDEX_AUTO_UPDATE:
                return
            from wells.index_tools import ensure_index
            max_age = float(__import__("os").environ.get("INDEX_MAX_AGE_HOURS", "24"))
            result = ensure_index(_cfg.WORKSPACE_ROOT, max_age_hours=max_age, auto_build=True)
            if result.startswith("index-built"):
                self.write_log(f"[green]{result[:120]}[/green]")
            elif result.startswith("index-failed"):
                self.write_log(f"[yellow]Index: {result}[/yellow]")
        except Exception:
            pass

    def _connect_mcp_servers(self) -> None:
        """Connect configured MCP servers in the background (never blocks UI)."""
        from wells.mcp_client import ensure_template, load_config

        ensure_template()  # first run: write ~/.wells/mcp.json with samples
        if not load_config():
            return

        def _connect() -> None:
            try:
                from wells.mcp_client import register_mcp_tools
                names = register_mcp_tools()
                if names:
                    self.call_from_thread(
                        self.write_log,
                        f"[green]MCP tools connected:[/green] [dim]{', '.join(names)}[/dim]",
                    )
            except Exception as e:
                try:
                    self.call_from_thread(
                        self.write_log, f"[yellow]MCP connect failed: {e}[/yellow]"
                    )
                except Exception:
                    pass

        threading.Thread(target=_connect, name="wells-mcp-connect", daemon=True).start()

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
        from wells.cli import SLASH_COMMANDS
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

        # /stop is a hard kill switch — it must work even mid-approval-prompt
        # or mid-anything-else, so it's checked before the pending-reply
        # intercept below (which would otherwise swallow it as a y/N answer).
        if text.lower() == "/stop":
            self._hard_stop()
            return

        # Handle pending interactive confirmations first.
        if self._pending:
            self._handle_pending_reply(text)
            return

        self._history_add(text)

        # /btw: concurrent side conversation — works busy or idle.
        if text.lower().startswith("/btw"):
            self._start_btw(text[4:].strip())
            return

        if text.startswith("/"):
            cmd = text.split()[0].lower()
            # Quit must never queue: cancel the run, exit when it stops.
            if cmd in ("/exit", "/quit"):
                if not self._busy:
                    self.exit()
                    return
                if self._exit_after_run:
                    import os as _os
                    _os._exit(0)  # second /exit = force-quit immediately
                self._exit_after_run = True
                CONTROL.cancel()
                CONTROL.set_activity("cancelling…")
                self.write_log(
                    "[yellow]Cancelling the running task — exiting when it "
                    "stops. Type /exit again to force-quit now.[/yellow]"
                )
                return
            # Every other slash command runs immediately, busy or not — only
            # plain messages queue. (Settings/mode changes made mid-run apply
            # from the next run; the in-flight run keeps its snapshot.)
            self._dispatch_slash(text)
            return

        if self._busy:
            self._queue.append(text)
            self.write_log(
                f"[yellow]⏳ queued #{len(self._queue)}[/yellow] [dim]{text[:80]}"
                f" — runs when the current task finishes. "
                f"(/btw <msg> for a side chat now)[/dim]"
            )
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
        if cmd == "/stop":
            self._hard_stop()
            return

        if cmd == "/resume":
            self._start_resume_picker(arg)
            return

        if cmd == "/config":
            # settings.interactive_menu() blocks on input() — deadlocks under
            # Textual. Open the modal settings panel instead.
            self._open_settings()
            return

        if cmd == "/export":
            self._export_transcript(" ".join(args))
            return

        if cmd == "/mcp":
            if not args:
                # No subcommand: open the visual manager (like /config).
                from wells.mcp_client import ensure_template
                ensure_template()
                self.push_screen(MCPScreen())
                return
            # Subcommands (power path) run in a worker thread — server
            # connects can take many seconds (npx downloads, etc.).
            self._run_mcp_command(" ".join(args))
            return

        if cmd == "/skills":
            sub = args[0].lower() if args else ""
            if not sub:
                # No subcommand: open the visual manager (like /config, /mcp).
                self.push_screen(SkillsScreen())
                return
            if sub in ("add", "edit", "update") and not (len(args) >= 2):
                # Bare /skills add or /skills edit with no name → use the modal
                # (the CLI handler calls input() which deadlocks under Textual).
                if sub == "add":
                    self.push_screen(SkillEditScreen(existing=None))
                else:
                    self.write_log(
                        "[dim]Usage: /skills edit <name> (or open the modal with bare /skills)[/dim]"
                    )
                return
            # Non-blocking subcommands (list, show, remove) delegate to the CLI
            # handler via the synchronous fallback below.

        if cmd == "/steer":
            note = " ".join(args)
            if not note:
                self.write_log("[dim]Usage: /steer <instruction> — injected into "
                               "the running agent's next reasoning round.[/dim]")
                return
            if not self._busy:
                self.write_log("[yellow]Nothing is running — just type your "
                               "message normally.[/yellow]")
                return
            CONTROL.add_steer(note)
            self.write_log(
                f"[bold cyan]⮕ steer queued:[/bold cyan] {note[:90]} "
                f"[dim](lands at the agent's next step)[/dim]"
            )
            return

        if cmd == "/queue":
            if arg.lower() == "clear":
                n = len(self._queue)
                self._queue.clear()
                self.write_log(f"[green]Cleared {n} queued message(s).[/green]")
            elif not self._queue:
                self.write_log("[dim]Queue is empty. Messages typed during a "
                               "run are queued automatically.[/dim]")
            else:
                self.write_log(f"[bold]Queued ({len(self._queue)}):[/bold]")
                for i, q in enumerate(self._queue, 1):
                    self.write_log(f"  [yellow]{i}.[/yellow] {q[:90]}")
                self.write_log("[dim]/queue clear to discard.[/dim]")
            return

        if cmd == "/undo":
            if self._busy:
                # Restoring the tree while the agent writes to it would
                # corrupt both — this is a refusal, not a queue.
                self.write_log(
                    "[yellow]/undo can't run while a task is writing to the "
                    "workspace — press Esc to cancel it first.[/yellow]"
                )
                return
            from wells.cli import undo_preview
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

        # -- All other commands run in a worker thread: some (/doctor,
        # /index build) block for seconds and used to freeze the whole UI.
        self._run_slash_worker(command)

    @work(thread=True)
    def _run_slash_worker(self, command: str) -> None:
        import wells.cli as cli_mod

        orig_console = cli_mod.console
        orig_stdout = sys.stdout
        # If a run is in flight, its worker already redirected console/stdout
        # to the TUI — reuse that instead of clobbering its redirect.
        swap = not isinstance(cli_mod.console, _TUIConsole)
        shim = _TUIStdout(self)
        if swap:
            cli_mod.console = _TUIConsole(self, thread_safe=True)
            sys.stdout = shim  # bare print() in handlers must reach the log too
        try:
            keep_running = cli_mod.handle_slash_command(command)
        except Exception as e:
            keep_running = True
            self.call_from_thread(
                self.write_log, f"[red]{command.split()[0]} failed: {e}[/red]"
            )
        finally:
            if swap:
                shim.flush()
                if sys.stdout is shim:
                    sys.stdout = orig_stdout
                if isinstance(cli_mod.console, _TUIConsole):
                    cli_mod.console = orig_console

        if not keep_running:
            self.call_from_thread(self.exit)

    @work(thread=True)
    def _run_mcp_command(self, arg: str) -> None:
        """Run /mcp in a worker thread (connects can block for seconds)."""
        import wells.cli as cli_mod

        orig = cli_mod.console
        cli_mod.console = _TUIConsole(self, thread_safe=True)
        try:
            cli_mod._handle_mcp(arg)
        except Exception as e:
            self.call_from_thread(self.write_log, f"[red]/mcp failed: {e}[/red]")
        finally:
            if isinstance(cli_mod.console, _TUIConsole):
                cli_mod.console = orig

    def _open_settings(self) -> None:
        def _done(staged: dict | None) -> None:
            if not staged:
                self.write_log("[dim]Settings closed — no changes.[/dim]")
                return
            from pathlib import Path as _Path

            from wells import settings as settings_mod
            from wells.main import _reload_module_config

            settings_mod.apply_changes(staged)
            try:
                settings_mod.update_env_file(_Path(".env"), staged)
                where = "applied + saved to .env"
            except Exception as e:
                where = f"applied live (could not write .env: {e})"
            _reload_module_config()
            self.write_log(
                f"[green]Settings {where}:[/green] "
                f"[dim]{', '.join(sorted(staged))}[/dim]"
            )

        self.push_screen(SettingsScreen(), _done)

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
        from wells.sessions import (
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
                from wells.sessions import clear_sessions
                workspace = None if all_ws else config.WORKSPACE_ROOT
                n = clear_sessions(workspace=workspace)
                self.write_log(f"[green]Deleted {n} session(s).[/green]")
            else:
                self.write_log("[dim]Cancelled.[/dim]")

        elif kind == "undo_confirm":
            sha = self._pending["sha"]
            self._pending = None
            if text.lower() in ("y", "yes"):
                from wells.cli import undo_apply
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
                self._input.placeholder = (
                    "Working… type to queue · /steer redirects · /btw side chat · Esc cancels · /stop kills it"
                )
            pend["event"].set()

    def _load_resume_session(self, session: dict) -> None:
        from wells.sessions import build_resume_context
        import wells.cli as cli_mod

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
        import wells.cli as _cli
        CONTROL.reset()
        # Pick up /mode and /working-dir changes made since the last run.
        self._agent_state["safety"] = config.HARNESS_SAFETY
        self._agent_state["plan_mode"] = config.PLAN_MODE
        self._agent_state["workspace_root"] = config.WORKSPACE_ROOT
        self._busy = True
        # Bumped on every run and on every /stop — a worker only gets to
        # touch shared UI state (busy flag, input restore) if its generation
        # still matches when it finishes. /stop bumps this immediately so an
        # abandoned worker's eventual completion (Python can't force-kill a
        # thread) is silently discarded instead of reviving the "busy" UI.
        self._run_gen += 1
        my_gen = self._run_gen
        # Input stays ENABLED: messages typed mid-run are queued, /btw chats
        # on the side, and the *_BUSY_SAFE_SLASH* commands run immediately.
        self._input.placeholder = (
            "Working… type to queue · /steer redirects · /btw side chat · Esc cancels · /stop kills it"
        )
        _cli._REPL_STATE["busy_since"] = _time.monotonic()
        self.write_log(f"\n[bold cyan]>[/bold cyan] {text}\n")
        self._run_input(text, my_gen)

    @work(thread=True)
    def _run_input(self, text: str, run_gen: int) -> None:
        """Run chat or task in a worker thread with redirected I/O."""
        import wells.cli as cli_mod
        from wells.cli import (
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
                via = "forced"
            else:
                CONTROL.set_activity("classifying request…")
                intent = chat.classify_intent(
                    text, use_llm_fallback=config.INTENT_LLM_FALLBACK
                )
                via = "classifier"

            # Always say which pipeline this message got — the single most
            # common "where am I?" question.
            if intent in ("task", "orchestrate"):
                route_line = (
                    "[dim]route:[/dim] [bold magenta]orchestrate[/bold magenta] "
                    f"[dim]({via}) — indexer → planner → architect → coder → "
                    "tester → reviewer → finisher[/dim]"
                )
            else:
                route_line = (
                    "[dim]route:[/dim] [green]auto[/green] "
                    "[dim]— direct executor (/orchestrate forces the full pipeline)[/dim]"
                )
            self.call_from_thread(self.write_log, route_line)

            from wells.cli import StreamingCallback, _run_auto
            callbacks = [StreamingCallback()]

            if intent in ("task", "orchestrate"):
                _run_task(text, self._agent_state, self._ensure_graph(), callbacks)
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

            if run_gen == self._run_gen:
                self._busy = False
                self.call_from_thread(self._restore_input)
            # else: /stop already abandoned this generation and restored the
            # UI itself — a newer run may already be in flight, so touching
            # _busy/input here would corrupt its state.

    def _restore_input(self) -> None:
        import wells.cli as _cli
        _cli._REPL_STATE["busy_since"] = None
        CONTROL.set_activity("")
        self._llm_done()  # flush a stream cancelled mid-line
        if self._exit_after_run:
            self.exit()
            return
        self._input.disabled = False

        # If the just-finished run was cut off by a cap or cancelled (not a
        # clean finish, not an error), the cli.py side set 'suggest_continue'.
        # Pre-fill the prompt with a one-Enter resume hint so the user can
        # continue with zero typing. One-shot: popped on first read.
        suggest = bool(_cli._REPL_STATE.pop("suggest_continue", False))
        if suggest and not self._queue:
            self._input.text = "continue from where you stopped"
            self._input.placeholder = (
                "↑ Enter to continue · type to replace · Esc-then-Delete to clear"
            )
            # Select all so the first keystroke replaces the suggestion
            # instead of appending to it.
            try:
                self._input.action_select_all()
            except Exception:
                pass
        else:
            self._input.placeholder = (
                "Ask a question or give a task… (/ commands, Shift+Enter newline)"
            )
        self._input.focus()

        # Drain the mid-run queue: next message starts automatically.
        if self._queue and not self._busy:
            nxt = self._queue.pop(0)
            self.write_log(
                f"\n[yellow]▶ from queue ({len(self._queue)} left):[/yellow] {nxt[:80]}"
            )
            if nxt.startswith("/") and not nxt.lower().startswith("/btw"):
                self._dispatch_slash(nxt)
                # Slash commands are synchronous — keep draining.
                if self._queue and not self._busy:
                    self.call_later(self._restore_input)
            else:
                self._start_run(nxt)

    # ------------------------------------------------------------------
    # /btw — concurrent side conversation (never touches the main run)
    # ------------------------------------------------------------------

    def _start_btw(self, msg: str) -> None:
        if not msg:
            self.write_log("[dim]Usage: /btw <message> — side chat that runs "
                           "even while a task is working.[/dim]")
            return
        self.write_log(f"\n[bold cyan]btw ▸[/bold cyan] {msg}")
        self._btw_history.append(("user", msg))
        # Snapshot context on the UI thread (widget access isn't thread-safe).
        try:
            tail = "\n".join(s.text for s in self._log.lines[-30:])[-3000:]
        except Exception:
            tail = ""
        threading.Thread(
            target=self._btw_worker, args=(msg, tail),
            name="wells-btw", daemon=True,
        ).start()

    def _btw_worker(self, msg: str, log_tail: str) -> None:
        import wells.cli as cli_mod
        from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

        try:
            goal = (cli_mod._REPL_STATE.get("last_state") or {}).get("goal", "")
            activity = CONTROL.activity()
            mem = cli_mod._REPL_STATE["memory"].last_run_summary
            system = (
                "You are Wells' side-channel assistant (/btw). The user is talking "
                "to you WHILE a main agent task may be running in parallel. You "
                "have no tools and must not interfere with the main task — you "
                "observe and answer. Be brief and direct.\n\n"
                f"Main task goal: {goal or '(none currently)'}\n"
                f"Current activity: {activity or '(idle)'}\n"
                + (f"\nLast run summary:\n{mem}\n" if mem else "")
                + (f"\nRecent session log tail:\n{log_tail}\n" if log_tail else "")
            )
            msgs: list = [SystemMessage(content=system)]
            for role, content in self._btw_history[-8:]:
                msgs.append(HumanMessage(content=content) if role == "user"
                            else AIMessage(content=content))
            # Cheap profile; independent of the main run's model/loop.
            llm = config.get_llm_for_task("classification", temperature=0.3)
            resp = config._invoke_with_retry(llm, msgs)
            text = (resp.content or "").strip() or "(no reply)"
            self._btw_history.append(("assistant", text))
            self.call_from_thread(
                self.write_log, f"[cyan]btw ◂[/cyan] {text}\n"
            )
        except Exception as e:
            self.call_from_thread(
                self.write_log, f"[red]btw failed: {type(e).__name__}: {e}[/red]"
            )

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

    def _hard_stop(self) -> None:
        """`/stop` — force-terminate whatever is running, right now.

        Escape/action_cancel_task is cooperative: it sets a flag that running
        code checks at its next poll point (every ~250ms since the shell/LLM
        cancellation fixes, but still a *wait*). /stop does not wait — it
        kills every tracked subprocess immediately and stops the UI from
        waiting on the worker thread at all. Python cannot force-kill a
        thread, so an abandoned worker may still be finishing a single
        blocked call somewhere in the background; the run-generation counter
        (see _run_input's finally block) makes sure that outcome is discarded
        instead of reviving the busy UI.
        """
        import wells.cli as _cli

        # A blocked approval prompt is moot after a hard stop — deny it.
        if self._pending and self._pending.get("kind") == "approval":
            pend = self._pending
            self._pending = None
            pend["holder"]["ok"] = False
            pend["event"].set()
        else:
            self._pending = None

        was_busy = self._busy
        CONTROL.cancel()
        n = CONTROL.kill_tracked_procs()

        if not was_busy and not n:
            self.write_log("[dim]Nothing is running.[/dim]")
            return

        self._run_gen += 1  # abandon the current worker generation
        self._busy = False
        _cli._REPL_STATE["busy_since"] = None
        CONTROL.set_activity("")

        killed_s = f"killed {n} process(es); " if n else ""
        self.write_log(
            f"[bold red]⛔ HARD STOP[/bold red] — {killed_s}abandoned the "
            f"running task. Ready for new input."
        )
        self._restore_input()

    def action_clear_log(self) -> None:
        self._log.clear()

    # -- info panel ----------------------------------------------------------

    def _load_panel_pref(self) -> bool:
        try:
            return bool(json.loads(_UI_PREFS_FILE.read_text(encoding="utf-8"))
                        .get("info_panel", True))
        except Exception:
            return True

    def _save_panel_pref(self, visible: bool) -> None:
        try:
            _UI_PREFS_FILE.parent.mkdir(parents=True, exist_ok=True)
            _UI_PREFS_FILE.write_text(
                json.dumps({"info_panel": visible}), encoding="utf-8"
            )
        except Exception:
            pass

    def _apply_panel_visibility(self, visible: bool) -> None:
        # The panel and the compact bottom bar carry the same info — show
        # exactly one so the data is always somewhere on screen.
        self._panel.display = visible
        self._statusbar.display = not visible

    def action_toggle_panel(self) -> None:
        visible = not self._panel.display
        self._apply_panel_visibility(visible)
        self._save_panel_pref(visible)

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
        from wells import setup
        setup.first_run_setup()
    except Exception:
        pass

    from wells.main import _ensure_model_configured
    if not _ensure_model_configured():
        return

    try:
        WellsApp(resume_context=resume_context).run(mouse=True)
    except KeyboardInterrupt:
        # Windows' ProactorEventLoop can raise a SECOND KeyboardInterrupt
        # while it closes (asyncio.run's own cleanup polls the IOCP again),
        # if Ctrl+C is pressed more than once in quick succession. The app
        # has already exited by the time this surfaces — swallow the ugly
        # traceback instead of dumping it on the user.
        print("\n[wells] interrupted.")
