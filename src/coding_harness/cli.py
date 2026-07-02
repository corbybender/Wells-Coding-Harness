"""Interactive REPL CLI for the Wells coding harness."""

import os
import sys
import time as _time
from pathlib import Path

from rich.console import Console
from langchain_core.callbacks import BaseCallbackHandler

from coding_harness import chat, config, settings
from coding_harness.graph import build_graph
from coding_harness.tokens import LEDGER
from coding_harness.main import _print_final_summary, _print_info, _reload_module_config

console = Console()


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
        "/orchestrate",
        "Force full orchestration (next message)",
        "Run next message through the full planner→coder→tester→reviewer loop.",
    ),
    (
        "/task",
        "Alias for /orchestrate",
        "Run your next message through the full agent loop (alias for /orchestrate).",
    ),
    (
        "/auto",
        "Reset to auto-routing",
        "Let Wells classify each message automatically (auto / orchestrate).",
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
    (
        "/sessions",
        "Browse session history",
        "List, delete, or clear sessions. Usage: /sessions [delete ID|clear|--all]",
    ),
    (
        "/resume",
        "Resume a previous session",
        "Pick a session and load its context for the next task. Usage: /resume [SESSION_ID]",
    ),
    (
        "/export",
        "Export session transcript",
        "Write the session log to a file. Usage: /export [path] (default: wells-transcript-<ts>.md)",
    ),
    (
        "/undo",
        "Revert everything the last run changed",
        "Restore the working tree to the automatic pre-run checkpoint (git repos only).",
    ),
]


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
    elif cmd in ("/orchestrate", "/task"):
        _set_force_mode("task")
    elif cmd == "/auto":
        _REPL_STATE["force_mode"] = None
        console.print("[dim]Routing reset to [bold]auto[/bold] — Wells will classify each message.[/dim]")
    elif cmd == "/clear":
        _REPL_STATE["memory"].clear()
        console.print("[green]Conversation history cleared.[/green]")
    elif cmd == "/index":
        _handle_index(arg)
    elif cmd == "/sessions":
        _handle_sessions(arg)
    elif cmd == "/resume":
        _handle_resume_cmd(arg)
    elif cmd == "/export":
        # Intercepted by the TUI (which owns the transcript) before reaching here.
        console.print("[yellow]/export is only available inside the TUI.[/yellow]")
    elif cmd == "/undo":
        _handle_undo()
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

    # Index status
    try:
        from coding_harness.index_tools import index_status

        idx = index_status(config.WORKSPACE_ROOT)
        if idx["available"]:
            if idx["exists"]:
                age = idx["age_hours"]
                age_str = f" (updated {age:.0f}h ago)" if age is not None else ""
                table.add_row("Index", f"{idx['total_symbols']:,} symbols, "
                              f"{idx['total_files']:,} files{age_str}")
            else:
                table.add_row("Index", "[yellow]not built[/yellow]")
        else:
            table.add_row("Index", "[dim]not available[/dim]")
    except Exception:
        pass

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

    force = arg == "force"
    if not arg or arg in ("build", "update") or force:
        import time as _time

        if force:
            # Delete the DB so next run re-parses every file from scratch.
            import shutil
            db_path = Path(config.WORKSPACE_ROOT) / ".wells_index"
            if db_path.exists():
                shutil.rmtree(db_path)
            console.print("[yellow]Index cleared — rebuilding from scratch...[/yellow]")

        t0 = _time.time()
        with console.status("[cyan]Indexing repository...[/cyan]", spinner="dots"):
            result = index_tools.index_workspace(ctx)
        elapsed = _time.time() - t0
        mins, secs = divmod(int(elapsed), 60)
        console.print(f"[dim]Done in {mins:02d}:{secs:02d}[/dim]")

        if result.ok:
            console.print(f"[green]{result.output}[/green]")
            _REPL_STATE["memory"].set_run_summary(
                f"User ran /index on workspace {config.WORKSPACE_ROOT}.\n"
                + result.output.strip()
            )
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
        console.print("[dim]Usage: /index [build|update|force|status|clear][/dim]")


def run_repl(resume_context: str | None = None) -> None:
    """Launch the full-screen Textual TUI (replaces the old prompt_toolkit REPL)."""
    from coding_harness.tui import run_tui
    run_tui(resume_context=resume_context)


# Prepended to executor system prompt in auto mode — sets conversational tone
# and clarifies Q&A vs action behaviour. Environment context (OS, shell, tools)
# is already injected by executor._system_prompt via _build_env_context().
_AUTO_SYSTEM_PREFIX = (
    "You are Wells, a concise coding assistant.\n\n"
    "- For questions or explanations: answer directly in your response. "
    "Use tools only when you need to look something up to be accurate "
    "(e.g. read a file to verify a detail). Do not invent answers.\n"
    "- For any action (edit a file, run a command, deploy, push, etc.): "
    "use run_command and the other tools to actually do it. "
    "Do the minimum necessary. Do not add unrequested changes.\n"
    "- Be brief."
)


def _run_auto(text: str, agent_state: dict, callbacks) -> None:
    """Run ``text`` via the direct executor — handles Q&A and tasks alike."""
    from coding_harness.executor import run_executor
    from coding_harness.sessions import new_session_id, save_session, session_from_final_state
    from coding_harness.tools import ToolContext

    resume_ctx: str | None = _REPL_STATE.pop("resume_context", None)
    _REPL_STATE.pop("resume_session_id", None)

    original_goal = text
    effective_task = text
    if resume_ctx:
        effective_task = f"{resume_ctx}\n\nCURRENT REQUEST:\n{text}"

    # Inject last_run_summary so follow-up questions have context.
    memory = _REPL_STATE["memory"]
    if memory.last_run_summary and not resume_ctx:
        effective_task = (
            f"Context from previous action:\n{memory.last_run_summary}\n\n"
            f"Current request:\n{text}"
        )

    LEDGER.reset()
    session_id = new_session_id()
    t0 = _time.time()
    _save_undo_checkpoint()

    if resume_ctx:
        console.print("[dim]Continuing from previous session...[/dim]")

    ctx = ToolContext(
        workspace=agent_state.get("workspace_root", config.WORKSPACE_ROOT),
        plan_mode=agent_state.get("plan_mode", config.PLAN_MODE),
        safety=agent_state.get("safety", config.HARNESS_SAFETY),
    )

    # Ensure the repo index is current before every executor run so that
    # find_symbol / search_symbols return real results (not empty).
    if config.INDEX_AUTO_UPDATE:
        from coding_harness import index_tools
        index_tools.ensure_index(ctx.workspace, auto_build=True)

    # Repo map in the (stable) system prefix: the model starts knowing where
    # things live, and the stable position is prompt-cache friendly.
    from coding_harness.repomap import repo_map_block
    system_prefix = _AUTO_SYSTEM_PREFIX + repo_map_block(ctx.workspace)

    try:
        result = run_executor(
            task=effective_task,
            ctx=ctx,
            system_prefix=system_prefix,
            stream=config.STREAM_OUTPUT,
        )

        console.print()
        if result.summary and not result.streamed:
            # Streamed answers already appeared live — don't print them twice.
            console.print(result.summary)
        console.print()

        t = LEDGER.totals()
        total = t["input"] + t["output"]
        console.print(
            f"[dim]{result.steps_taken} step(s) · {total:,} tokens "
            f"({t['input']:,} in / {t['output']:,} out)[/dim]"
        )

        try:
            final_state = {
                "review_complete": result.stopped_reason == "done",
                "implementation_steps": result.summary,
                "review_result": result.stopped_reason,
                "iteration": 1,
                "git_summary": "",
            }
            data = session_from_final_state(
                session_id, original_goal, final_state,
                workspace=config.WORKSPACE_ROOT,
                tokens_in=t["input"],
                tokens_out=t["output"],
                duration_seconds=int(_time.time() - t0),
                resumed_from=resume_ctx[:80] if resume_ctx else None,
            )
            save_session(session_id, data)
            console.print(f"[dim][session: {session_id}][/dim]")
        except Exception:
            pass

        memory.set_run_summary(
            f"Goal: {original_goal}\nResult: {result.summary[:400]}"
        )

    except Exception as e:
        from coding_harness.logger import log_error
        log_error(f"_run_auto failed: {type(e).__name__}: {e}", e)
        console.print(f"\n[bold red]Error:[/bold red] {e}")


def _run_task(text: str, agent_state: dict, app, callbacks) -> None:
    """Run ``text`` through the full agentic graph."""
    from coding_harness.sessions import (
        build_resume_context, new_session_id, save_session, session_from_final_state,
    )

    # Consume resume context (one-shot: cleared after first use).
    resume_ctx: str | None = _REPL_STATE.pop("resume_context", None)
    _REPL_STATE.pop("resume_session_id", None)

    original_goal = text
    effective_goal = (
        f"{resume_ctx}\n\nCONTINUED GOAL:\n{text}" if resume_ctx else text
    )
    agent_state["goal"] = effective_goal
    agent_state["iteration"] = 0
    LEDGER.reset()
    session_id = new_session_id()
    t0 = _time.time()
    _save_undo_checkpoint()

    _NODE_LABELS = {
        "planner":   "[bold blue]Planning…[/bold blue]",
        "architect": "[bold blue]Architecting…[/bold blue]",
        "coder":     "[bold green]Coding…[/bold green]",
        "tester":    "[bold yellow]Testing…[/bold yellow]",
        "reviewer":  "[bold cyan]Reviewing…[/bold cyan]",
        "finisher":  "[bold cyan]Finishing…[/bold cyan]",
        "indexer":   "[dim]Indexing…[/dim]",
    }

    if resume_ctx:
        console.print("[dim]Continuing from previous session...[/dim]")
    console.print(f"\n[bold]Goal:[/bold] {text}\n")

    from coding_harness.control import CONTROL, RunCancelled

    try:
        for update in app.stream(
            agent_state, config={"callbacks": callbacks}, stream_mode="updates"
        ):
            if CONTROL.cancelled():
                raise RunCancelled()
            if config.MAX_RUN_TOKENS:
                t_ = LEDGER.totals()
                used_ = t_["input"] + t_["output"]
                if used_ >= config.MAX_RUN_TOKENS:
                    console.print(
                        f"\n[bold red]Token budget reached ({used_:,}/"
                        f"{config.MAX_RUN_TOKENS:,}) — stopping the run.[/bold red]"
                    )
                    break
            for node_name, node_state in update.items():
                label = _NODE_LABELS.get(node_name, f"[bold]{node_name.title()}…[/bold]")
                console.print(f"\n{label}")
                for k, v in node_state.items():
                    agent_state[k] = v

            # Checkpoint after every node: a crash/kill mid-run loses at most
            # one node's work, and /resume can continue from the last state.
            try:
                t_ = LEDGER.totals()
                save_session(session_id, session_from_final_state(
                    session_id, original_goal, agent_state,
                    workspace=config.WORKSPACE_ROOT,
                    tokens_in=t_["input"],
                    tokens_out=t_["output"],
                    duration_seconds=int(_time.time() - t0),
                    resumed_from=resume_ctx[:80] if resume_ctx else None,
                    in_progress=True,
                ))
            except Exception:
                pass

        _REPL_STATE["last_state"] = dict(agent_state)
        _print_final_summary(agent_state)

        t = LEDGER.totals()
        total = t["input"] + t["output"]
        console.print(
            f"\n[dim][tokens] {total:,} total "
            f"({t['input']:,} in / {t['output']:,} out) "
            f"across {t['calls']} calls[/dim]"
        )

        # Save session.
        try:
            data = session_from_final_state(
                session_id, original_goal, agent_state,
                workspace=config.WORKSPACE_ROOT,
                tokens_in=t["input"],
                tokens_out=t["output"],
                duration_seconds=int(_time.time() - t0),
                resumed_from=resume_ctx[:80] if resume_ctx else None,
            )
            save_session(session_id, data)
            console.print(f"[dim][session: {session_id}][/dim]")
        except Exception as e:
            console.print(f"[dim][session save failed: {e}][/dim]")

    except RunCancelled:
        console.print("\n[yellow]Task cancelled by user.[/yellow]")
    except Exception as e:
        from coding_harness.logger import log_error
        log_error(f"_run_task failed: {type(e).__name__}: {e}", e)
        console.print(f"\n[bold red]Error during execution:[/bold red] {e}")


def _save_undo_checkpoint() -> None:
    """Snapshot the working tree before a run so /undo can revert it."""
    try:
        from coding_harness import gitops
        sha = gitops.snapshot_worktree(config.WORKSPACE_ROOT)
        _REPL_STATE["undo_checkpoint"] = sha or None
    except Exception:
        _REPL_STATE["undo_checkpoint"] = None


def undo_preview() -> tuple[str, str]:
    """Return (checkpoint_sha, diff_stat vs now). Empty sha = nothing to undo."""
    from coding_harness import gitops
    sha = _REPL_STATE.get("undo_checkpoint") or ""
    if not sha:
        # Fall back to the persisted ref (survives restarts).
        ok, out = gitops._git(config.WORKSPACE_ROOT, "rev-parse", "--verify",
                              gitops._UNDO_REF)
        sha = out.strip() if ok else ""
    if not sha:
        return "", ""
    return sha, gitops.snapshot_diff_stat(config.WORKSPACE_ROOT, sha)


def undo_apply(sha: str) -> tuple[bool, str]:
    """Restore the working tree to ``sha`` (the pre-run checkpoint)."""
    from coding_harness import gitops
    return gitops.restore_snapshot(config.WORKSPACE_ROOT, sha)


def _handle_undo() -> None:
    """Plain-CLI /undo (the TUI intercepts this command with its own confirm)."""
    sha, stat = undo_preview()
    if not sha:
        console.print("[yellow]No checkpoint to undo (no run yet, or not a git repo).[/yellow]")
        return
    if not stat:
        console.print("[dim]Working tree already matches the last checkpoint — nothing to undo.[/dim]")
        return
    console.print(f"\n[bold]Reverting to pre-run checkpoint {sha[:8]}:[/bold]\n{stat}\n")
    try:
        confirm = input("Revert these changes? [y/N] ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        return
    if confirm in ("y", "yes"):
        ok, msg = undo_apply(sha)
        console.print(f"[green]{msg}[/green]" if ok else f"[red]{msg}[/red]")
    else:
        console.print("[dim]Cancelled.[/dim]")


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


def _handle_sessions(arg: str) -> None:
    """Handle /sessions [list|delete ID|clear] [--all]."""
    from coding_harness.sessions import (
        clear_sessions, delete_session, format_age, list_sessions,
    )
    from rich.table import Table

    parts = arg.strip().split() if arg.strip() else []
    all_ws = "--all" in parts
    sub_parts = [p for p in parts if p != "--all"]
    subcmd = sub_parts[0] if sub_parts else "list"
    workspace = None if all_ws else config.WORKSPACE_ROOT

    if subcmd in ("list", ""):
        sessions = list_sessions(workspace=workspace, limit=25)
        if not sessions:
            scope = "any workspace" if all_ws else "this workspace"
            console.print(f"[yellow]No sessions found for {scope}.[/yellow]")
            return
        table = Table(show_header=True, header_style="bold cyan", expand=False)
        table.add_column("Session ID", style="dim", width=26, no_wrap=True)
        table.add_column("Age", width=10)
        table.add_column("Status", width=10)
        table.add_column("Tokens", justify="right", width=9)
        table.add_column("Goal")
        for s in sessions:
            age = format_age(s.get("created_at", ""))
            status = s.get("status", "?")
            color = "green" if status == "COMPLETE" else "yellow"
            tok = (s.get("tokens_in") or 0) + (s.get("tokens_out") or 0)
            tok_s = f"{tok:,}" if tok else "?"
            goal = (s.get("goal") or "")[:55]
            table.add_row(
                s["id"], age, f"[{color}]{status}[/{color}]", tok_s, goal
            )
        console.print(table)
        scope = "all workspaces" if all_ws else "this workspace"
        console.print(
            f"[dim]{len(sessions)} session(s) — {scope}. "
            f"Use /sessions --all for all workspaces.[/dim]\n"
        )

    elif subcmd == "delete" and len(sub_parts) >= 2:
        sid = sub_parts[1]
        if delete_session(sid):
            console.print(f"[green]Deleted: {sid}[/green]")
        else:
            console.print(f"[red]Not found: {sid}[/red]")

    elif subcmd == "clear":
        scope = "ALL workspaces" if all_ws else "this workspace"
        try:
            confirm = input(f"Delete all sessions for {scope}? [y/N] ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            return
        if confirm in ("y", "yes"):
            n = clear_sessions(workspace=workspace)
            console.print(f"[green]Deleted {n} session(s).[/green]")
        else:
            console.print("[dim]Cancelled.[/dim]")

    else:
        console.print("[red]Usage: /sessions [list|delete SESSION_ID|clear] [--all][/red]")


def _handle_resume_cmd(arg: str) -> None:
    """Handle /resume [SESSION_ID] — load a previous session's context."""
    from coding_harness.sessions import (
        build_resume_context, format_age, is_session_id,
        list_sessions, load_session,
    )

    sid = arg.strip()
    if sid and is_session_id(sid):
        session = load_session(sid)
        if not session:
            console.print(f"[red]Session not found: {sid}[/red]")
            return
    else:
        sessions = list_sessions(workspace=config.WORKSPACE_ROOT, limit=10)
        if not sessions:
            console.print("[yellow]No sessions for this workspace.[/yellow]")
            return
        console.print("\n[bold]Recent sessions:[/bold]")
        for i, s in enumerate(sessions, 1):
            age = format_age(s.get("created_at", ""))
            status = s.get("status", "?")
            color = "green" if status == "COMPLETE" else "yellow"
            goal = (s.get("goal") or "")[:60]
            console.print(
                f"  [cyan]{i}.[/cyan] [{age}] [{color}]{status}[/{color}] {goal!r}"
            )
            console.print(f"     [dim]{s['id']}[/dim]")
        console.print()
        try:
            choice = input("Select session number (or Enter to cancel): ").strip()
        except (EOFError, KeyboardInterrupt):
            return
        if not choice:
            console.print("[dim]Cancelled.[/dim]")
            return
        try:
            session = sessions[int(choice) - 1]
        except (ValueError, IndexError):
            console.print("[red]Invalid selection.[/red]")
            return

    _REPL_STATE["resume_context"] = build_resume_context(session)
    _REPL_STATE["resume_session_id"] = session["id"]
    console.print(f"\n[green]Session loaded: {session['id']}[/green]")
    console.print(f"[dim]Previous goal : {(session.get('goal') or '')[:70]}[/dim]")
    console.print(
        "[dim]Context injected — your next task will continue from this session.[/dim]\n"
    )


# Session-scoped state for the REPL (memory, force-mode, last agent state).
_REPL_STATE: dict = {
    "memory": chat.ConversationMemory(),
    "force_mode": None,
    "last_state": {},
    "resume_context": None,
    "resume_session_id": None,
    "busy_since": None,   # monotonic timestamp set while a run is in progress
}


def _set_force_mode(mode: str) -> None:
    _REPL_STATE["force_mode"] = mode
    labels = {
        "auto": "[bold green]auto[/bold green] [dim](direct executor)[/dim]",
        "task": "[bold magenta]orchestrate[/bold magenta] [dim](full planning loop)[/dim]",
    }
    label = labels.get(mode, f"[bold]{mode}[/bold]")
    console.print(
        f"Next message: {label} [dim](auto-routing resumes after)[/dim]."
    )


def _ensure_model_configured() -> bool:
    from coding_harness.main import _ensure_model_configured as check

    return check()


if __name__ == "__main__":
    run_repl()
