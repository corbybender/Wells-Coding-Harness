"""Interactive REPL CLI for the Wells coding harness."""

import os
import sys
from pathlib import Path

from prompt_toolkit import PromptSession
from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.styles import Style
from prompt_toolkit.formatted_text import HTML
from rich.console import Console
from langchain_core.callbacks import BaseCallbackHandler

from coding_harness import chat, config, settings
from coding_harness.graph import build_graph
from coding_harness.tokens import LEDGER
from coding_harness.main import _print_final_summary, _print_info, _reload_module_config

console = Console()

style = Style.from_dict(
    {
        "prompt": "#00aa00 bold",
    }
)


# ---------------------------------------------------------------------------
# Slash command catalog (single source of truth for help + autocomplete).
# ---------------------------------------------------------------------------
# (command, short description, long help shown in /help).
SLASH_COMMANDS: list[tuple[str, str, str]] = [
    ("/help", "Show available commands", "List all slash commands."),
    ("/quit", "Exit the REPL", "Quit the Wells session (also: /exit)."),
    ("/exit", "Exit the REPL", "Quit the Wells session (also: /quit)."),
    (
        "/config",
        "Open interactive settings menu",
        "Edit model, provider, safety, budgets, ...",
    ),
    (
        "/info",
        "Print effective configuration",
        "Show resolved profiles, workspace, knobs.",
    ),
    ("/plan", "Toggle plan mode", "When ON, coder plans edits without applying them."),
    (
        "/working-dir",
        "View/change working directory",
        "Show or set the workspace root tools are confined to.",
    ),
    (
        "/status",
        "Show status panel",
        "Print working dir, model, and token usage/savings.",
    ),
    (
        "/chat",
        "Force chat mode (next message)",
        "Answer your next message directly without the agent loop.",
    ),
    (
        "/task",
        "Force task mode (next message)",
        "Run your next message through the full agent loop.",
    ),
    (
        "/clear",
        "Clear conversation history",
        "Forget prior chat context (keeps agent-run summary).",
    ),
    (
        "/index",
        "Manage repository index",
        "Build/update repo index (wells-index). Subcommands: /index (build), /index status, /index clear.",
    ),
]


class SlashCompleter(Completer):
    """Autocomplete slash commands. Typing '/' lists every command."""

    def get_completions(self, document, complete_event):
        text = document.text_before_cursor
        if not text.startswith("/"):
            return
        # Suggest all commands whose name starts with the typed text.
        for cmd, short, _long in SLASH_COMMANDS:
            if cmd.startswith(text):
                yield Completion(
                    cmd,
                    start_position=-len(text),
                    display=cmd,
                    display_meta=short,
                )


class StreamingCallback(BaseCallbackHandler):
    """Streams LLM tokens to the console."""

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        if config.STREAM_OUTPUT:
            sys.stdout.write(token)
            sys.stdout.flush()

    def on_llm_end(self, response, **kwargs) -> None:
        if config.STREAM_OUTPUT:
            sys.stdout.write("\n")
            sys.stdout.flush()


def print_welcome() -> None:
    console.print("\n[bold blue]Wells Coding Harness[/bold blue]")
    console.print(f"Model: {config.model_name_for_task('coding')}")
    console.print(
        f"Workspace: {config.WORKSPACE_ROOT}  (safety: {config.HARNESS_SAFETY})"
    )
    console.print(
        "Ask a question (auto chat) or give a task (auto agent). "
        "Type [bold]/[/bold] for commands."
    )


def handle_slash_command(command: str) -> bool:
    """Handles slash commands. Returns False if REPL should exit, True otherwise."""
    # Split into command + optional argument (e.g. "/working-dir Q:\proj").
    parts = command.strip().split(None, 1)
    cmd = parts[0].lower()
    arg = parts[1].strip() if len(parts) > 1 else ""

    if cmd in ("/quit", "/exit"):
        return False
    elif cmd == "/help":
        _print_help()
    elif cmd == "/config":
        settings.interactive_menu(Path(".env"))
        _reload_module_config()
    elif cmd == "/info":
        _reload_module_config()
        _print_info()
    elif cmd == "/plan":
        current = os.environ.get("PLAN_MODE", "0")
        new_val = "0" if current not in ("0", "false", "no", "") else "1"
        os.environ["PLAN_MODE"] = new_val
        _reload_module_config()
        console.print(
            f"\nPlan mode is now: [bold]{'ON' if config.PLAN_MODE else 'OFF'}[/bold]\n"
        )
    elif cmd == "/working-dir":
        _handle_working_dir(arg)
    elif cmd == "/status":
        _print_status_panel()
    elif cmd == "/chat":
        _set_force_mode("chat")
    elif cmd == "/task":
        _set_force_mode("task")
    elif cmd == "/clear":
        _REPL_STATE["memory"].clear()
        console.print("[green]Conversation history cleared.[/green]")
    elif cmd == "/index":
        _handle_index(arg)
    else:
        console.print(f"[red]Unknown command: {command}[/red]")
        console.print("[dim]Type / for a list of commands.[/dim]")
    return True


def _print_help() -> None:
    """Print the full slash-command catalog."""
    console.print("\n[bold]Available Commands:[/bold]")
    for cmd, short, long in SLASH_COMMANDS:
        console.print(f"  [cyan]{cmd:<15}[/cyan] [dim]-[/dim] {short}")
    console.print()


def _handle_working_dir(arg: str) -> None:
    """Show or change the working directory (WORKSPACE_ROOT).

    With no argument: prints the current working directory.
    With a path argument: validates it exists and is a directory, then updates
    WORKSPACE_ROOT in os.environ (live) and persists it to .env.
    """
    if not arg:
        console.print(
            f"\nWorking directory: [bold green]{config.WORKSPACE_ROOT}[/bold green]\n"
        )
        return

    path = Path(arg).expanduser()
    if not path.is_absolute():
        path = (Path.cwd() / path).resolve()
    if not path.exists():
        console.print(f"[red]Path does not exist: {path}[/red]")
        return
    if not path.is_dir():
        console.print(f"[red]Not a directory: {path}[/red]")
        return

    # Update live + persist to .env so it survives restarts.
    os.environ["WORKSPACE_ROOT"] = str(path)
    try:
        settings.update_env_file(Path(".env"), {"WORKSPACE_ROOT": str(path)})
    except Exception:
        pass
    _reload_module_config()
    console.print(
        f"\nWorking directory set to: [bold green]{config.WORKSPACE_ROOT}[/bold green]\n"
    )


def _print_status_panel() -> None:
    """Print a status panel (working dir, model, token usage/savings)."""
    from rich.panel import Panel
    from rich.table import Table

    table = Table.grid(padding=(0, 2))
    table.add_column(style="bold cyan", no_wrap=True)
    table.add_column()

    table.add_row("Working dir", config.WORKSPACE_ROOT)
    table.add_row("Model", config.model_name_for_task("coding"))
    table.add_row("Safety", config.HARNESS_SAFETY)
    table.add_row("Plan mode", "ON" if config.PLAN_MODE else "OFF")

    totals = LEDGER.totals()
    used = totals["input"] + totals["output"]
    saved = totals["saved_trim"] + totals["saved_summary"]
    table.add_row("Calls", str(totals["calls"]))
    table.add_row("Tokens used", f"{used:,}")
    if saved:
        table.add_row("Tokens saved", f"[bold green]{saved:,}[/bold green]")
    if totals["cache_read"]:
        table.add_row("Cache hits", f"{totals['cache_read']:,}")

    console.print(Panel(table, title="[bold]Wells Status[/bold]", border_style="blue"))


def _handle_index(arg: str) -> None:
    """Handle /index command (build, status, clear)."""
    from coding_harness import index_tools
    from coding_harness.tools import ToolContext

    if not index_tools.INDEXER_AVAILABLE:
        console.print(
            "[red]Error: Index engine not available. Install: pip install wells-index[/red]"
        )
        return

    ctx = ToolContext(workspace=config.WORKSPACE_ROOT)

    if not arg or arg in ("build", "update"):
        console.print("[cyan]Indexing repository...[/cyan]")
        result = index_tools.index_workspace(ctx)
        if result.ok:
            console.print(f"[green]{result.output}[/green]")
        else:
            console.print(f"[red]Error: {result.error or result.output}[/red]")
    elif arg == "status":
        console.print("[cyan]Repository index statistics:[/cyan]")
        result = index_tools.list_symbols(ctx, "")
        if result.ok:
            console.print(result.output)
        else:
            console.print(f"[red]Error: {result.error or result.output}[/red]")
    elif arg == "clear":
        console.print("[cyan]Clearing index...[/cyan]")
        try:
            from wells_index import IndexEngine

            engine = IndexEngine(config.WORKSPACE_ROOT)
            engine.clear()
            console.print("[green]Index cleared.[/green]")
        except Exception as e:
            console.print(f"[red]Error: Could not clear index: {e}[/red]")
    else:
        console.print(f"[red]Unknown /index subcommand: {arg}[/red]")
        console.print("[dim]Usage: /index [build|status|clear][/dim]")


def _bottom_toolbar():
    """Persistent status bar shown beneath the prompt (like opencode's panel).

    Returns prompt_toolkit-formatted text. Called on every render, so it stays
    live as token usage grows and as the working dir / model change.
    """
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
    # Shorten the working dir for narrow terminals.
    if len(wd) > 32:
        wd = "..." + wd[-29:]

    saved_str = f" | saved: {saved:,}" if saved else ""

    # Show forced mode if active.
    force = _REPL_STATE.get("force_mode")
    mode_str = ""
    if force == "chat":
        mode_str = '<style fg="#ffaa00">| FORCE:chat</style>'
    elif force == "task":
        mode_str = '<style fg="#ff44ff">| FORCE:task</style>'
    else:
        mode_str = '<style fg="#666666">| auto</style>'

    return HTML(
        f'<style fg="#888888">{wd}</style>'
        f' <style fg="#00aa00">| {model}</style>'
        f' <style fg="#5588ff">| tokens: {used:,}{saved_str}</style>'
        f" {mode_str}"
    )


def run_repl() -> None:
    # Auto-setup on first run
    try:
        from coding_harness import setup
        setup.first_run_setup()
    except Exception:
        pass  # Setup is optional

    if not _ensure_model_configured():
        return

    print_welcome()

    # Session-scoped state shared with slash commands and the router.
    _REPL_STATE["memory"] = chat.ConversationMemory()
    _REPL_STATE["force_mode"] = None  # "chat" | "task" | None (auto)
    _REPL_STATE["last_state"] = {}

    session = PromptSession(
        completer=SlashCompleter(),
        bottom_toolbar=_bottom_toolbar,
    )

    app = build_graph()

    # Maintain conversational state across loops
    agent_state = {
        "iteration": 0,
        "max_iterations": config.MAX_ITERATIONS,
        "workspace_root": config.WORKSPACE_ROOT,
        "safety": config.HARNESS_SAFETY,
        "plan_mode": config.PLAN_MODE,
        "messages": [],
        "executor_messages": [],
    }

    callbacks = [StreamingCallback()]

    while True:
        try:
            text = session.prompt(
                HTML("<prompt>Wells&gt;</prompt> "), style=style
            ).strip()
        except KeyboardInterrupt:
            continue
        except EOFError:
            break

        if not text:
            continue

        if text.startswith("/"):
            if not handle_slash_command(text):
                break
            continue

        # ---- Route: chat vs task -----------------------------------------
        force = _REPL_STATE.get("force_mode")
        if force:
            intent = force
            _REPL_STATE["force_mode"] = None  # one-shot override
        else:
            intent = chat.classify_intent(text)

        if intent == "chat":
            _run_chat(text, callbacks)
        else:
            _run_task(text, agent_state, app, callbacks)
            # Feed the agent's result back into the conversation memory so
            # follow-up questions ("did it work?") have context.
            _REPL_STATE["memory"].set_run_summary(
                _summarize_run(_REPL_STATE.get("last_state", {}))
            )


def _run_chat(text: str, callbacks) -> None:
    """Answer ``text`` directly without the agent loop (streamed)."""
    console.print()  # blank line before streamed output
    try:
        chat.conversational_reply(
            text,
            _REPL_STATE["memory"],
            on_token=_stream_token if config.STREAM_OUTPUT else None,
        )
    except Exception as e:
        console.print(f"\n[bold red]Error during chat:[/bold red] {e}")
    console.print()


def _stream_token(token: str) -> None:
    sys.stdout.write(token)
    sys.stdout.flush()


def _run_task(text: str, agent_state: dict, app, callbacks) -> None:
    """Run ``text`` through the full agentic graph."""
    # Update goal
    agent_state["goal"] = text
    agent_state["iteration"] = 0
    LEDGER.reset()

    console.print(f"\n[bold cyan]Executing:[/bold cyan] {text}\n")

    try:
        # We use stream to get node updates
        for update in app.stream(
            agent_state, config={"callbacks": callbacks}, stream_mode="updates"
        ):
            for node_name, node_state in update.items():
                console.print(
                    f"\n[bold magenta]>> {node_name.upper()} <<[/bold magenta]"
                )
                if not config.STREAM_OUTPUT:
                    console.print(f"Completed step: {node_name}")

                # Merge node_state back into our persistent agent_state
                for k, v in node_state.items():
                    agent_state[k] = v

        _REPL_STATE["last_state"] = dict(agent_state)
        _print_final_summary(agent_state)
        console.print("\n" + LEDGER.format_report())
    except Exception as e:
        console.print(f"\n[bold red]Error during execution:[/bold red] {e}")


def _summarize_run(state: dict) -> str:
    """Compact summary of the last agentic run for chat context."""
    if not state:
        return ""
    status = "COMPLETE" if state.get("review_complete") else "INCOMPLETE"
    review = (state.get("review_result") or "").strip()
    steps = (state.get("implementation_steps") or "").strip()
    parts = [
        f"Goal: {state.get('goal', '')}",
        f"Status: {status}",
        f"Iterations: {state.get('iteration', 0)}",
    ]
    if steps:
        parts.append(f"Implementation:\n{steps[:600]}")
    if review:
        parts.append(f"Review:\n{review[:600]}")
    return "\n".join(parts)


# Session-scoped state for the REPL (memory, force-mode, last agent state).
_REPL_STATE: dict = {
    "memory": chat.ConversationMemory(),
    "force_mode": None,
    "last_state": {},
}


def _set_force_mode(mode: str) -> None:
    _REPL_STATE["force_mode"] = mode
    label = (
        "[bold cyan]chat[/bold cyan]"
        if mode == "chat"
        else "[bold magenta]task[/bold magenta]"
    )
    console.print(
        f"Next message will be handled in {label} mode "
        f"[dim](auto-routing resumes after)[/dim]."
    )


def _ensure_model_configured() -> bool:
    from coding_harness.main import _ensure_model_configured as check

    return check()


if __name__ == "__main__":
    run_repl()
