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
    (
        "/mode",
        "Switch operating mode",
        "Usage: /mode [plan|approve|auto|dryrun]. plan = read-only, approve = confirm "
        "each write/command, auto = full autonomy, dryrun = simulate everything.",
    ),
    (
        "/add",
        "Pin a file into every prompt",
        "Usage: /add <path>. Pinned files are injected into the context of every run "
        "until dropped — guaranteed context instead of hoping the agent reads them.",
    ),
    (
        "/drop",
        "Unpin a file (or all)",
        "Usage: /drop <path>|all.",
    ),
    (
        "/context",
        "Show pinned files and their token cost",
        "Lists files added with /add, with per-file token estimates.",
    ),
    (
        "/doctor",
        "Diagnose the environment",
        "Checks model reachability, API key, TLS, repo index health, git, and fast checkers.",
    ),
    (
        "/mcp",
        "Manage MCP servers",
        "Usage: /mcp [list] | add <name> <command> [args…] | remove <name> | "
        "enable <name> | disable <name> | test <name>. Backed by ~/.wells/mcp.json.",
    ),
    (
        "/rules",
        "Operating rules + open liabilities",
        "Usage: /rules [list|reload|discharge <id>]. Rules are enforced at the "
        "tool boundary (.wells/rules.yaml) and injected from RULES.md.",
    ),
    (
        "/btw",
        "Side chat while a task runs",
        "Usage: /btw <message>. Independent conversation that works even mid-"
        "orchestration — no tools, aware of the running task's goal and log.",
    ),
    (
        "/queue",
        "Messages queued during a run",
        "Messages typed while a task runs are queued and executed in order "
        "when it finishes. Usage: /queue [clear].",
    ),
    (
        "/steer",
        "Redirect the RUNNING agent",
        "Usage: /steer <instruction>. Injected into the running agent's next "
        "reasoning round — changes course mid-task without cancelling.",
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
    elif cmd == "/mode":
        _handle_mode(arg)
    elif cmd == "/add":
        _handle_add(arg)
    elif cmd == "/drop":
        _handle_drop(arg)
    elif cmd == "/context":
        _handle_context()
    elif cmd == "/doctor":
        _handle_doctor()
    elif cmd == "/mcp":
        _handle_mcp(arg)
    elif cmd == "/rules":
        _handle_rules(arg)
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


def reply_timestamp() -> str:
    """Plain-text timestamp for the start of every reply (RULES R0 lineage):
    '7:23:41am 7-4-2026'."""
    t = _time.localtime()
    hour = t.tm_hour % 12 or 12
    ampm = "am" if t.tm_hour < 12 else "pm"
    return f"{hour}:{t.tm_min:02d}:{t.tm_sec:02d}{ampm} {t.tm_mon}-{t.tm_mday}-{t.tm_year}"


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

    # User-pinned files (/add) are guaranteed context.
    pin_block = _pinned_context_block()
    if pin_block:
        effective_task = f"{pin_block}\n{effective_task}"

    LEDGER.reset()
    session_id = new_session_id()
    t0 = _time.time()
    _save_undo_checkpoint()

    console.print(f"[bold #39FF14]{reply_timestamp()}[/bold #39FF14]")
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

    # Repo map in the system prefix: the model starts knowing where things
    # live, ranked by relevance to this goal.
    from coding_harness.repomap import repo_map_block
    system_prefix = _AUTO_SYSTEM_PREFIX + repo_map_block(ctx.workspace, goal=text)

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
        from coding_harness import pricing
        cost_s = pricing.fmt(pricing.run_cost())
        cost_part = f" · {cost_s}" if cost_s else ""
        console.print(
            f"[dim]{result.steps_taken} step(s) · {total:,} tokens "
            f"({t['input']:,} in / {t['output']:,} out){cost_part}[/dim]"
        )

        # A run with an undischarged liability (e.g. a still-running rented
        # GPU) is NOT complete, whatever the model says.
        liabilities_clear = _enforce_liabilities(ctx.workspace)

        try:
            final_state = {
                "review_complete": result.stopped_reason == "done" and liabilities_clear,
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

        if config.AUTO_COMMIT and result.stopped_reason == "done":
            _maybe_auto_commit(original_goal, result.summary)

    except Exception as e:
        from coding_harness.logger import log_error
        log_error(f"_run_auto failed: {type(e).__name__}: {e}", e)
        console.print(f"\n[bold red]Error:[/bold red] {e}")


def _maybe_auto_commit(goal: str, summary: str) -> None:
    """Commit the run's changes with an LLM-generated conventional message."""
    from coding_harness import gitops

    ckpt = _REPL_STATE.get("undo_checkpoint") or ""
    stat = gitops.snapshot_diff_stat(config.WORKSPACE_ROOT, ckpt) if ckpt else ""
    if ckpt and not stat:
        return  # run changed nothing

    message = ""
    try:
        prompt = (
            "Write a git commit message for the change below. Conventional Commits "
            "format (type(scope): subject). Subject line <= 65 chars, imperative "
            "mood. Add a 1-2 line body ONLY if the why isn't obvious. Reply with "
            "the message only — no fences, no commentary.\n\n"
            f"GOAL:\n{goal[:400]}\n\nWHAT WAS DONE:\n{summary[:600]}\n\n"
            f"DIFF STAT:\n{stat[:500]}"
        )
        from langchain_core.messages import HumanMessage
        llm = config.get_llm_for_task("summarization", temperature=0.1)
        resp = config._invoke_with_retry(llm, [HumanMessage(content=prompt)])
        message = (resp.content or "").strip().strip("`")
    except Exception:
        pass
    if not message:
        message = f"chore(wells): {goal.strip()[:60]}"

    ok, sha = gitops.auto_commit(config.WORKSPACE_ROOT, message)
    if ok and sha:
        console.print(f"[dim]auto-committed {sha}: {message.splitlines()[0][:70]}[/dim]")
    elif not ok:
        console.print(f"[yellow]auto-commit failed: {sha}[/yellow]")


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
    pin_block = _pinned_context_block()
    if pin_block:
        effective_goal = f"{pin_block}\n{effective_goal}"
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

    console.print(f"[bold #39FF14]{reply_timestamp()}[/bold #39FF14]")
    if resume_ctx:
        console.print("[dim]Continuing from previous session...[/dim]")
    console.print(f"\n[bold]Goal:[/bold] {text}\n")

    from coding_harness.control import CONTROL, RunCancelled

    try:
        node_t0 = _time.monotonic()
        for update in app.stream(
            agent_state, config={"callbacks": callbacks}, stream_mode="updates"
        ):
            node_secs = _time.monotonic() - node_t0
            node_t0 = _time.monotonic()
            if node_secs >= 5:
                console.print(f"[dim]  ({node_secs:.0f}s)[/dim]")
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
                CONTROL.set_progress(
                    "iteration",
                    agent_state.get("iteration", 0),
                    agent_state.get("max_iterations", config.MAX_ITERATIONS),
                )

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

        # Liability enforcement before the run may be called complete.
        if not _enforce_liabilities(config.WORKSPACE_ROOT):
            agent_state["review_complete"] = False
            agent_state["review_result"] = (
                (agent_state.get("review_result") or "")
                + "\n\n[RULES: run has UNDISCHARGED liabilities — see /rules]"
            ).strip()

        _REPL_STATE["last_state"] = dict(agent_state)
        _print_final_summary(agent_state)

        t = LEDGER.totals()
        total = t["input"] + t["output"]
        from coding_harness import pricing
        cost_s = pricing.fmt(pricing.run_cost())
        cost_part = f" · {cost_s}" if cost_s else ""
        console.print(
            f"\n[dim][tokens] {total:,} total "
            f"({t['input']:,} in / {t['output']:,} out) "
            f"across {t['calls']} calls{cost_part}[/dim]"
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


_MODES = {
    "plan":    ("1", None),        # PLAN_MODE, HARNESS_SAFETY (None = leave as-is)
    "approve": ("0", "approve"),
    "auto":    ("0", "auto"),
    "dryrun":  ("0", "dryrun"),
}


def current_mode() -> str:
    if config.PLAN_MODE:
        return "plan"
    return config.HARNESS_SAFETY if config.HARNESS_SAFETY in ("approve", "dryrun") else "auto"


def _handle_mode(arg: str) -> None:
    """Switch operating mode: plan | approve | auto | dryrun."""
    want = arg.strip().lower()
    if not want:
        console.print(
            f"\nOperating mode: [bold]{current_mode()}[/bold]\n"
            "[dim]  plan    — read-only: investigate and describe, never change\n"
            "  approve — apply changes, but confirm each write/command\n"
            "  auto    — full autonomy inside the workspace\n"
            "  dryrun  — simulate every mutation\n"
            "Usage: /mode <plan|approve|auto|dryrun>[/dim]\n"
        )
        return
    if want not in _MODES:
        console.print(f"[red]Unknown mode: {want}[/red] [dim](plan|approve|auto|dryrun)[/dim]")
        return
    plan, safety_v = _MODES[want]
    os.environ["PLAN_MODE"] = plan
    if safety_v:
        os.environ["HARNESS_SAFETY"] = safety_v
    _reload_module_config()
    labels = {
        "plan": "[bold yellow]plan[/bold yellow] — read-only",
        "approve": "[bold cyan]approve[/bold cyan] — confirm each write/command",
        "auto": "[bold green]auto[/bold green] — full autonomy",
        "dryrun": "[bold magenta]dryrun[/bold magenta] — simulate everything",
    }
    console.print(f"Operating mode: {labels[want]}")


def _pinned() -> list[str]:
    return _REPL_STATE.setdefault("pinned", [])


def _handle_add(arg: str) -> None:
    """Pin a file into the context of every run."""
    if not arg.strip():
        console.print("[red]Usage: /add <path>[/red]")
        return
    rel = arg.strip().strip('"')
    p = Path(config.WORKSPACE_ROOT) / rel
    if not p.is_file():
        console.print(f"[red]Not a file: {rel}[/red]")
        return
    pinned = _pinned()
    if rel in pinned:
        console.print(f"[dim]{rel} is already pinned.[/dim]")
        return
    pinned.append(rel)
    from coding_harness.tokens import estimate_tokens
    try:
        tok = estimate_tokens(p.read_text(encoding="utf-8", errors="replace"))
    except Exception:
        tok = 0
    console.print(
        f"[green]Pinned {rel}[/green] [dim](~{tok:,} tokens — injected into every run "
        f"until /drop)[/dim]"
    )


def _handle_drop(arg: str) -> None:
    pinned = _pinned()
    rel = arg.strip().strip('"')
    if not rel:
        console.print("[red]Usage: /drop <path>|all[/red]")
        return
    if rel.lower() == "all":
        n = len(pinned)
        pinned.clear()
        console.print(f"[green]Dropped {n} pinned file(s).[/green]")
        return
    if rel in pinned:
        pinned.remove(rel)
        console.print(f"[green]Dropped {rel}[/green]")
    else:
        console.print(f"[yellow]Not pinned: {rel}[/yellow] [dim](see /context)[/dim]")


def _handle_context() -> None:
    from rich.table import Table
    from coding_harness.tokens import estimate_tokens

    pinned = _pinned()
    if not pinned:
        console.print("[dim]No pinned files. Use /add <path> to pin one.[/dim]")
        return
    table = Table(show_header=True, header_style="bold cyan", expand=False)
    table.add_column("Pinned file")
    table.add_column("Tokens", justify="right")
    total = 0
    for rel in pinned:
        p = Path(config.WORKSPACE_ROOT) / rel
        try:
            tok = estimate_tokens(p.read_text(encoding="utf-8", errors="replace"))
        except Exception:
            tok = 0
        total += tok
        table.add_row(rel, f"{tok:,}")
    table.add_row("[bold]total[/bold]", f"[bold]{total:,}[/bold]")
    console.print(table)


_PIN_FILE_TOKEN_CAP = 4000     # per-file trim threshold
_PIN_TOTAL_TOKEN_WARN = 10000  # warn when pinned block gets expensive


def _pinned_context_block() -> str:
    """Render pinned files as a prompt block ('' when nothing pinned)."""
    from coding_harness.tokens import estimate_tokens

    pinned = _pinned()
    if not pinned:
        return ""
    parts = ["PINNED CONTEXT FILES (user pinned these; treat as authoritative):"]
    total = 0
    for rel in pinned:
        p = Path(config.WORKSPACE_ROOT) / rel
        try:
            body = p.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue
        tok = estimate_tokens(body)
        if tok > _PIN_FILE_TOKEN_CAP:
            lines = body.splitlines()
            keep = max(50, int(len(lines) * _PIN_FILE_TOKEN_CAP / tok))
            body = "\n".join(lines[:keep]) + f"\n… (trimmed; {len(lines) - keep} more lines — read_file for the rest)"
            tok = estimate_tokens(body)
        total += tok
        parts.append(f"\n--- {rel} ---\n{body}")
    if total > _PIN_TOTAL_TOKEN_WARN:
        console.print(
            f"[yellow]⚠ pinned context is ~{total:,} tokens per run — "
            f"consider /drop for files you no longer need.[/yellow]"
        )
    return "\n".join(parts) + "\n"


def _handle_doctor() -> None:
    """Environment diagnostic: model, key, TLS, index, git, checkers."""
    import shutil as _shutil
    from rich.table import Table

    table = Table(show_header=True, header_style="bold cyan", expand=False)
    table.add_column("Check", no_wrap=True)
    table.add_column("Status", no_wrap=True)
    table.add_column("Detail")
    OK = "[green]✓[/green]"
    WARN = "[yellow]⚠[/yellow]"
    FAIL = "[red]✗[/red]"

    # Provider profile + API key
    profile = config.providers.load_profile(config.ACTIVE_PROFILE)
    if profile is None:
        table.add_row("model profile", FAIL, f"profile {config.ACTIVE_PROFILE!r} not configured")
    else:
        key_s = "key set" if profile.api_key else "[yellow]no API key[/yellow]"
        table.add_row("model profile", OK, f"{profile.label()} ({key_s})")

    # Model reachability (one tiny call)
    with console.status("[cyan]Pinging model…[/cyan]"):
        try:
            t0 = _time.monotonic()
            llm = config.get_llm_for_task("classification")
            from langchain_core.messages import HumanMessage
            resp = llm.invoke([HumanMessage(content="Reply with the word: pong")])
            ms = int((_time.monotonic() - t0) * 1000)
            ok_ping = bool((resp.content or "").strip())
            table.add_row("model reachable", OK if ok_ping else WARN, f"{ms} ms round-trip")
        except Exception as e:
            table.add_row("model reachable", FAIL, f"{type(e).__name__}: {str(e)[:80]}")

    # TLS trust
    try:
        import truststore  # noqa: F401
        table.add_row("tls trust", OK, "truststore (OS certificate store)")
    except Exception:
        bundle = os.environ.get("SSL_CERT_FILE", "")
        table.add_row(
            "tls trust", OK if bundle else WARN,
            f"SSL_CERT_FILE={bundle}" if bundle else "no truststore; certifi fallback",
        )

    # Repo index health
    try:
        from coding_harness.index_tools import INDEXER_AVAILABLE, index_status
        if not INDEXER_AVAILABLE:
            table.add_row("repo index", WARN, "wells-index not installed — grep fallback")
        else:
            st = index_status(config.WORKSPACE_ROOT)
            if st.get("error"):
                table.add_row("repo index", FAIL, st["error"][:80])
            elif not st["exists"]:
                table.add_row("repo index", WARN, "not built yet — /index to build")
            elif st["total_files"] > 0 and st["total_symbols"] == 0:
                from coding_harness.setup import repair_index_core
                fixed, msg = repair_index_core()
                table.add_row(
                    "repo index", WARN if fixed else FAIL,
                    f"stale native core (0 symbols) — {msg}",
                )
            else:
                age = st.get("age_hours")
                age_s = f", {age:.0f}h old" if age is not None else ""
                table.add_row(
                    "repo index", OK,
                    f"{st['total_symbols']:,} symbols / {st['total_files']:,} files{age_s}",
                )
    except Exception as e:
        table.add_row("repo index", FAIL, str(e)[:80])

    # Git + checkpointing
    if _shutil.which("git"):
        from coding_harness import gitops
        ok_repo, out = gitops._git(config.WORKSPACE_ROOT, "rev-parse", "--is-inside-work-tree")
        in_repo = ok_repo and "true" in out.lower()
        table.add_row(
            "git", OK if in_repo else WARN,
            "workspace is a git repo (/undo available)" if in_repo
            else "workspace not a git repo — /undo disabled",
        )
    else:
        table.add_row("git", WARN, "git not on PATH — checkpoints and /undo disabled")

    # Fast checkers (self-heal)
    found = [c for c in ("ruff", "node") if _shutil.which(c)]
    detail = ", ".join(found) if found else "none — python falls back to py_compile"
    table.add_row("fast checkers", OK if found else WARN, detail)

    # Pinned files still exist
    missing = [r for r in _pinned() if not (Path(config.WORKSPACE_ROOT) / r).is_file()]
    if missing:
        table.add_row("pinned files", WARN, f"missing: {', '.join(missing[:5])}")

    # Rules engine + open liabilities
    try:
        from coding_harness import rules as rules_mod
        eng = rules_mod.engine_for(config.WORKSPACE_ROOT)
        open_l = eng.open_liabilities()
        if open_l:
            table.add_row(
                "rules", FAIL,
                f"{len(eng.rules)} rules; [bold red]{len(open_l)} OPEN "
                f"LIABILITY(IES)[/bold red] — /rules",
            )
        else:
            table.add_row("rules", OK, f"{len(eng.rules)} enforced, no open liabilities")
    except Exception as e:
        table.add_row("rules", WARN, str(e)[:80])

    console.print(table)


def _handle_mcp(arg: str) -> None:
    """Manage MCP servers: list / add / remove / enable / disable / test."""
    from coding_harness import mcp_client as mc

    parts = arg.strip().split()
    sub = parts[0].lower() if parts else "list"

    if mc.env_override_active() and sub != "list":
        console.print(
            "[yellow]MCP_SERVERS env var is set — it overrides mcp.json, so file "
            "edits won't take effect until you unset it.[/yellow]"
        )

    if sub == "list":
        _mcp_list(mc)
    elif sub == "add":
        if len(parts) < 3:
            console.print("[red]Usage: /mcp add <name> <command> [args…][/red]")
            return
        name, command, args = parts[1], parts[2], parts[3:]
        spec: dict = {"command": command}
        if args:
            spec["args"] = args
        mc.add_server(name, spec)
        console.print(f"[green]Added '{name}' to mcp.json.[/green] [dim]Connecting…[/dim]")
        ok, msg, names = mc.connect_server(name, spec)
        if ok:
            console.print(f"[green]{name}: {msg}[/green] [dim]{', '.join(names)}[/dim]")
        else:
            console.print(
                f"[yellow]{name}: saved, but connect failed — {msg}[/yellow]\n"
                "[dim]Fix the command/env in ~/.wells/mcp.json, then /mcp test "
                f"{name}.[/dim]"
            )
    elif sub in ("remove", "rm", "delete"):
        if len(parts) < 2:
            console.print("[red]Usage: /mcp remove <name>[/red]")
            return
        name = parts[1]
        mc.disconnect_server(name)
        spec = mc.remove_server(name)
        if spec is None:
            console.print(f"[yellow]Not found: {name}[/yellow]")
        else:
            import json as _json
            console.print(
                f"[green]Removed '{name}'.[/green] "
                f"[dim]Was: {_json.dumps(spec)} — /mcp add to restore.[/dim]"
            )
    elif sub == "disable":
        if len(parts) < 2:
            console.print("[red]Usage: /mcp disable <name>[/red]")
            return
        name = parts[1]
        mc.disconnect_server(name)
        ok, msg = mc.set_enabled(name, False)
        console.print(f"[green]{name}: {msg}[/green]" if ok else f"[yellow]{msg}[/yellow]")
    elif sub == "enable":
        if len(parts) < 2:
            console.print("[red]Usage: /mcp enable <name>[/red]")
            return
        name = parts[1]
        ok, msg = mc.set_enabled(name, True)
        if not ok:
            console.print(f"[yellow]{msg}[/yellow]")
            return
        spec = mc.load_config().get(name) or mc.read_file_config().get(name) or {}
        console.print(f"[green]{name}: enabled.[/green] [dim]Connecting…[/dim]")
        ok, msg, names = mc.connect_server(name, spec)
        if ok:
            console.print(f"[green]{name}: {msg}[/green] [dim]{', '.join(names)}[/dim]")
        else:
            console.print(f"[yellow]{name}: connect failed — {msg}[/yellow]")
    elif sub == "test":
        if len(parts) < 2:
            console.print("[red]Usage: /mcp test <name>[/red]")
            return
        name = parts[1]
        cfg = mc.load_config()
        spec = cfg.get(name) or (mc.read_file_config().get("_disabled") or {}).get(name) \
            or (mc.read_file_config().get("_examples") or {}).get(name)
        if not spec:
            console.print(f"[yellow]Not found: {name}[/yellow]")
            return
        console.print(f"[dim]Connecting to '{name}'…[/dim]")
        ok, msg, names = mc.connect_server(name, spec)
        if ok:
            console.print(f"[green]{name}: {msg}[/green]")
            for n in names:
                console.print(f"  [dim]{n}[/dim]")
        else:
            console.print(f"[red]{name}: {msg}[/red]")
    else:
        console.print(
            "[red]Usage: /mcp [list] | add <name> <command> [args…] | remove <name> | "
            "enable <name> | disable <name> | test <name>[/red]"
        )


def _mcp_list(mc) -> None:
    from rich.table import Table

    data = mc.read_file_config()
    live = mc.connected()
    active = {k: v for k, v in data.items() if not k.startswith("_") and isinstance(v, dict)}
    disabled = data.get("_disabled") or {}
    examples = data.get("_examples") or {}

    table = Table(show_header=True, header_style="bold cyan", expand=False)
    table.add_column("Server", no_wrap=True)
    table.add_column("State", no_wrap=True)
    table.add_column("Command")
    table.add_column("Tools", justify="right")

    def _cmd(spec: dict) -> str:
        return (f"{spec.get('command', '?')} " + " ".join(spec.get("args") or []))[:60]

    for name, spec in active.items():
        state = "[green]connected[/green]" if name in live else "[yellow]enabled[/yellow]"
        table.add_row(name, state, _cmd(spec), str(len(live.get(name, []))) if name in live else "-")
    for name, spec in disabled.items():
        table.add_row(name, "[dim]disabled[/dim]", f"[dim]{_cmd(spec)}[/dim]", "-")
    for name, spec in examples.items():
        if name not in active and name not in disabled:
            table.add_row(f"[dim]{name}[/dim]", "[dim]example[/dim]", f"[dim]{_cmd(spec)}[/dim]", "-")

    console.print(table)
    if mc.env_override_active():
        console.print("[yellow]Note: MCP_SERVERS env var is set and overrides this file.[/yellow]")
    console.print(
        "[dim]/mcp add <name> <command> [args…] · enable/disable/test/remove <name> — "
        "file: ~/.wells/mcp.json[/dim]"
    )


def _enforce_liabilities(workspace: str) -> bool:
    """Run-end liability enforcement. Returns True when all clear.

    If open liabilities remain (e.g. a rented GPU was started but never
    terminated), optionally runs ONE bounded follow-up agent pass to discharge
    them, then re-checks. Still open → loud red warning; the run must not be
    treated as complete.
    """
    from coding_harness import rules as rules_mod

    if not config.RULES_ENFORCE:
        return True
    try:
        eng = rules_mod.engine_for(workspace)
    except Exception:
        return True
    open_l = eng.open_liabilities()
    if not open_l:
        return True

    summary = eng.liability_summary()
    console.print(
        f"\n[bold red]⚠ OPEN LIABILITIES ({len(open_l)}):[/bold red] {summary}"
    )

    if config.RULES_AUTODISCHARGE and config.HARNESS_SAFETY != "dryrun" and not config.PLAN_MODE:
        console.print("[yellow]Attempting automatic discharge…[/yellow]")
        try:
            from coding_harness.executor import run_executor
            from coding_harness.tools import ToolContext
            lines = "\n".join(
                f"- [{l['rule_id']}] opened {l['opened_at']}: {l['detail']}"
                for l in open_l
            )
            run_executor(
                task=(
                    "MANDATORY CLEANUP — the run cannot end until these open "
                    "liabilities are discharged. For each one, run the "
                    "termination/close command, then VERIFY with a status "
                    "check and quote the evidence:\n" + lines
                ),
                ctx=ToolContext(workspace=workspace, safety=config.HARNESS_SAFETY),
                max_steps=10,
                step_label="liability-discharge",
            )
        except Exception as e:
            console.print(f"[yellow]Auto-discharge failed: {e}[/yellow]")
        open_l = eng.open_liabilities()

    if open_l:
        console.print(
            f"[bold red]⚠ STILL OPEN ({len(open_l)}): {eng.liability_summary()}[/bold red]\n"
            "[red]This may be COSTING MONEY RIGHT NOW. Close it manually, then "
            "run [bold]/rules discharge <id>[/bold] to acknowledge — or ask me "
            "to terminate it.[/red]"
        )
        return False
    console.print("[green]All liabilities discharged and verified.[/green]")
    return True


def _handle_rules(arg: str) -> None:
    """Handle /rules [list|reload|discharge <id>]."""
    from rich.table import Table
    from coding_harness import rules as rules_mod

    eng = rules_mod.engine_for(config.WORKSPACE_ROOT)
    parts = arg.strip().split()
    sub = parts[0].lower() if parts else "list"

    if sub == "reload":
        rules_mod.reload_all()
        console.print(f"[green]Rules reloaded ({len(eng.rules)} active).[/green]")
        return
    if sub == "discharge":
        if len(parts) < 2:
            console.print("[red]Usage: /rules discharge <rule-id>[/red]")
            return
        n = eng.discharge(parts[1])
        console.print(
            f"[green]Discharged {n} liability(ies) for '{parts[1]}'.[/green]"
            if n else f"[yellow]No open liabilities for '{parts[1]}'.[/yellow]"
        )
        return

    # list
    table = Table(show_header=True, header_style="bold cyan", expand=False)
    table.add_column("Rule", no_wrap=True)
    table.add_column("Severity", no_wrap=True)
    table.add_column("Trigger")
    sev_color = {"block": "red", "confirm": "yellow", "warn": "cyan",
                 "liability": "magenta"}
    for r in eng.rules:
        c = sev_color.get(r.severity, "white")
        trig = (f"open: {r.open[:44]}…" if r.severity == "liability"
                else f"{r.tool or 'any'}: {r.pattern[:44]}")
        table.add_row(r.id, f"[{c}]{r.severity}[/{c}]", f"[dim]{trig}[/dim]")
    console.print(table)

    open_l = eng.open_liabilities()
    if open_l:
        console.print(f"\n[bold red]⚠ Open liabilities ({len(open_l)}):[/bold red]")
        for l in open_l:
            console.print(
                f"  [red]{l['rule_id']}[/red] opened {l['opened_at']} — "
                f"[dim]{l['detail'][:80]}[/dim]"
            )
        console.print(
            "[dim]Close the resource, then /rules discharge <id> to acknowledge.[/dim]"
        )
    else:
        console.print("\n[green]No open liabilities.[/green]")
    console.print(
        "[dim]Enforced rules: .wells/rules.yaml (workspace) + ~/.wells/rules.yaml "
        "(global). Prompt/audit layer: RULES.md.[/dim]"
    )


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
