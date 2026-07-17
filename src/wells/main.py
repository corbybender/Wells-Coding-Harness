"""Entry point for the Wells agentic coding harness.

Usage:
    wells "<your development goal>"        # run the harness
    wells --workspace /path "fix the bug"  # run against another project
    wells config                           # interactive settings menu
    wells info                             # show effective config
    wells --plan "<goal>"                  # plan mode (no edits)
    wells --version                        # show version
    wells "<goal>" MAX_ITERATIONS=5        # inline setting overrides

    # Headless / scriptable (CI, wrapping Wells from other tooling):
    wells -p --output-format json "<goal>"      # one JSON object on stdout, exit 0/1
    echo "<goal>" | wells -p --output-format json  # task piped in via stdin
"""

from __future__ import annotations

import json
import os
import sys
import time as _time
from datetime import datetime
from pathlib import Path

_STARTUP_T0 = _time.perf_counter()  # as close to process start as we control


def _startup_mark(label: str) -> None:
    """Append an elapsed-since-process-start timing line when profiling is on.

    Set WELLS_STARTUP_PROFILE=1 to enable. Written to a file, not stdout —
    the TUI takes over the terminal (alternate screen buffer) almost
    immediately, so print()/stderr output from this phase is invisible in a
    real launch. Near-zero cost when disabled: one os.environ.get() call.
    """
    if os.environ.get("WELLS_STARTUP_PROFILE") not in ("1", "true", "yes"):
        return
    try:
        elapsed = (_time.perf_counter() - _STARTUP_T0) * 1000
        log = Path.home() / ".wells" / "startup-profile.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open("a", encoding="utf-8") as f:
            f.write(f"{elapsed:8.1f}ms  {label}\n")
    except Exception:
        pass


_startup_mark("main.py: module start")

from wells import __version__, config, settings  # noqa: E402 — timed deliberately

_startup_mark("main.py: core imports done (wells.config, settings)")


def _print_section(title: str, body: str) -> None:
    line = "=" * 70
    print(f"\n{line}\n{title}\n{line}\n{body or '(empty)'}")


def _print_final_summary(state: dict) -> None:
    complete = state.get("review_complete", False)
    iterations = state.get("iteration", 0)
    max_iter = state.get("max_iterations", config.MAX_ITERATIONS)
    if state.get("review_error"):
        # The reviewer's own LLM call failed — this was never a real verdict,
        # so don't report it as if the reviewer judged the work and disagreed.
        status = "ERROR (reviewer could not run — see notes below, check API key/connectivity)"
    else:
        status = "COMPLETE" if complete else f"INCOMPLETE (stopped after {iterations}/{max_iter} iterations)"

    bar = "=" * 70
    print(f"\n{bar}")
    print(f"  {status}")
    print(bar)

    # Git result (set by finisher node)
    git = state.get("git_summary", "")
    if git:
        print(f"\n  Changes: {git}")

    # What the coder actually did — first 800 chars is usually enough
    impl = (state.get("implementation_steps") or "").strip()
    if impl:
        preview = impl[:800] + (" …(truncated)" if len(impl) > 800 else "")
        print(f"\n  What was done:\n{_indent(preview, 4)}")

    # Reviewer feedback — only meaningful lines, capped at 15
    review = (state.get("review_result") or "").strip()
    if review:
        lines = [ln for ln in review.splitlines() if ln.strip()]
        shown = "\n".join(lines[:15])
        if len(lines) > 15:
            shown += f"\n  … ({len(lines) - 15} more lines)"
        print(f"\n  Reviewer notes:\n{_indent(shown, 4)}")

    print(bar)


def _indent(text: str, spaces: int) -> str:
    pad = " " * spaces
    return "\n".join(pad + ln for ln in text.splitlines())


def _print_info() -> None:
    """Print the effective configuration (resolved profiles + run knobs)."""
    from wells import providers

    bar = "=" * 64
    print(f"\n{bar}\n Wells harness — effective configuration\n{bar}")
    print(f"  Active profile : {config.ACTIVE_PROFILE}")
    try:
        prof = providers.load_profile(config.ACTIVE_PROFILE)
        print(f"  Model          : {prof.label() if prof else '(not configured)'}")
        if prof:
            print(f"  Provider kind  : {prof.kind}")
            print(f"  Base URL       : {prof.base_url or '(provider default)'}")
            print(f"  API key set    : {bool(prof.api_key)}")
    except Exception as e:
        print(f"  Model          : (error resolving: {e})")

    cheap = config.cheap_profile_name()
    if cheap != config.ACTIVE_PROFILE:
        cprof = providers.load_profile(cheap)
        print(f"  Cheap profile  : {cheap} -> {cprof.label() if cprof else '?'}")

    print(f"\n  Available profiles : {config.MODEL_PROFILES}")
    print(f"  Workspace root     : {config.WORKSPACE_ROOT}")
    print(f"  Safety policy      : {config.HARNESS_SAFETY}")
    print(f"  Plan mode          : {'on' if config.PLAN_MODE else 'off'}")
    print(f"  Max iterations     : {config.MAX_ITERATIONS}")
    print(f"  Max tool steps     : {config.MAX_TOOL_STEPS}")
    print(
        f"  Token budget/call  : {config.BUDGET.max_input_tokens} "
        f"(reserved out {config.BUDGET.reserved_output_tokens}) — "
        f"this is the enforced context-trim ceiling for the executor's "
        f"safety-drop pipeline, not just informational"
    )
    print(
        f"  Small/local budget : {config.SMALL_BUDGET.max_input_tokens} "
        f"(reserved out {config.SMALL_BUDGET.reserved_output_tokens}) — "
        f"used automatically instead of the above for profiles that look "
        f"like local Ollama"
    )
    print(
        f"  Summarize on loop  : {'on' if config.SUMMARIZE_ON_LOOP else 'off'} "
        f"(threshold {config.SUMMARIZE_THRESHOLD})"
    )
    print(
        f"  Ollama num_ctx     : {config.OLLAMA_NUM_CTX or 'off'} "
        "(native-API context-window warm-up for local Ollama profiles)"
    )
    # Show the active principles source so users know which AGENT.md is in effect.
    try:
        from wells import principles
        print(f"  Principles         : {principles.source_label(config.WORKSPACE_ROOT)}")
    except Exception:
        pass
    print(bar)


def _run_goal(
    goal: str, *, resume_context: str | None = None, output_format: str = "text"
) -> None:
    """Build and invoke the harness graph for ``goal``.

    ``output_format="json"`` is the headless/scriptable path (``wells -p
    --output-format json "task"``, CI pipelines, wrapping Wells from other
    tooling): every progress/debug print the graph and its agents scatter
    across stdout is redirected to stderr for the duration of the run, then
    exactly one JSON object describing the outcome is written to stdout.
    Exit code reflects success (0) vs incomplete/error (1) so a CI step can
    branch on it without scraping text.
    """
    import contextlib
    import io

    from wells.graph import build_graph
    from wells.sessions import new_session_id, save_session, session_from_final_state
    from wells.tokens import LEDGER

    json_mode = output_format == "json"

    if not _ensure_model_configured():
        if json_mode:
            print(json.dumps({"status": "error", "error": "model not configured"}))
        sys.exit(1)

    LEDGER.reset()
    session_id = new_session_id()
    t0 = _time.time()

    def _say(line: str) -> None:
        if not json_mode:
            print(line)

    _say(f"Model: {config.model_name_for_task('coding')}")
    _say(f"Workspace: {config.WORKSPACE_ROOT}  (safety: {config.HARNESS_SAFETY})")
    if config.WORKSPACE_ROOT_INVALID:
        _say(
            f"WARNING: configured WORKSPACE_ROOT does not exist: "
            f"{config.WORKSPACE_ROOT_INVALID} — falling back to the directory "
            f"above. Fix WORKSPACE_ROOT in .env (or pass --workspace)."
        )
    if config.PLAN_MODE:
        _say("Plan mode: ON (coder will plan edits without applying them)")
    _say(f"Max coder<->reviewer iterations: {config.MAX_ITERATIONS}")
    _say(f"Goal: {goal}")
    if resume_context:
        _say("[Continuing from previous session — context injected]")
    _say("-" * 70)

    app = build_graph()
    effective_goal = f"{resume_context}\n\nCONTINUED GOAL:\n{goal}" if resume_context else goal
    initial_state = {
        "goal": effective_goal,
        "iteration": 0,
        "max_iterations": config.MAX_ITERATIONS,
        "workspace_root": config.WORKSPACE_ROOT,
        "safety": config.HARNESS_SAFETY,
        "plan_mode": config.PLAN_MODE,
        "messages": [],
    }

    # Every agent/graph node prints its own progress chatter directly to
    # stdout (print("[coder] ...") etc.) — there is no single choke point to
    # silence instead. Redirecting the whole stream to stderr for the
    # invoke's duration keeps stdout pure JSON without touching every call
    # site; nothing here reads stdin, so it's a safe blanket swap.
    stdout_guard = (
        contextlib.redirect_stdout(sys.stderr) if json_mode else contextlib.nullcontext()
    )
    with stdout_guard:
        final_state = app.invoke(initial_state)
    duration = int(_time.time() - t0)

    t = LEDGER.totals()
    total = t["input"] + t["output"]

    if json_mode:
        from wells import pricing
        payload = {
            "status": (
                "error" if final_state.get("review_error")
                else "complete" if final_state.get("review_complete")
                else "incomplete"
            ),
            "goal": goal,
            "session_id": session_id,
            "workspace": config.WORKSPACE_ROOT,
            "iterations": final_state.get("iteration", 0),
            "max_iterations": final_state.get("max_iterations", config.MAX_ITERATIONS),
            "summary": (final_state.get("implementation_steps") or "").strip(),
            "review_result": (final_state.get("review_result") or "").strip(),
            "git_summary": final_state.get("git_summary", ""),
            "tokens": {
                "input": t["input"], "output": t["output"], "total": total,
                "calls": t["calls"], "cache_read": t["cache_read"],
            },
            "cost_usd": pricing.run_cost(),
            "duration_seconds": duration,
        }
        exit_code = 0 if payload["status"] == "complete" else 1
    else:
        _print_final_summary(final_state)
        print(
            f"\n[tokens] {total:,} total "
            f"({t['input']:,} in / {t['output']:,} out) across {t['calls']} calls"
            + (f", {t['cache_read']:,} cache hits" if t["cache_read"] else "")
            + (" — set WELLS_TOKEN_REPORT=1 for full breakdown" if total > 50_000 else "")
        )
        if os.environ.get("WELLS_TOKEN_REPORT") == "1":
            print("\n" + LEDGER.format_report())
        exit_code = None

    # Persist session for later resume/history.
    try:
        data = session_from_final_state(
            session_id, goal, final_state,
            workspace=config.WORKSPACE_ROOT,
            tokens_in=t["input"],
            tokens_out=t["output"],
            duration_seconds=duration,
            resumed_from=resume_context[:80] if resume_context else None,
        )
        save_session(session_id, data)
        _say(f"[session: {session_id}]")
    except Exception as e:
        _say(f"[session save failed: {e}]")

    if json_mode:
        print(json.dumps(payload, ensure_ascii=False, default=str))
        sys.exit(exit_code)


def _ensure_model_configured_fast() -> bool:
    """Check the active profile resolves — no client construction.

    ``providers.get_chat_model()`` (below) imports the provider SDK package
    and builds the client: for langchain_openai that's ~3s of import alone,
    plus ~1s building SSL-verified httpx clients and ~1s of pydantic model
    construction — around 5s, measured. This half of the check is what's
    actually needed before the TUI can usefully show a prompt (a missing
    MODEL_<profile> is a real "can't do anything" error); the expensive
    client build is deferred to :func:`warm_chat_model`, run in a background
    thread by ``run_tui`` while the app is mounting.
    """
    from wells import providers

    try:
        prof = providers.load_profile(config.ACTIVE_PROFILE)
    except Exception:
        prof = None
    if prof is None or not prof.model:
        print(
            f"ERROR: active profile {config.ACTIVE_PROFILE!r} has no model configured."
        )
        print(
            "Run `wells config` to set it up, or set "
            f"MODEL_{config.ACTIVE_PROFILE}=<model> in your environment."
        )
        return False
    return True


def warm_chat_model() -> None:
    """Construct (and cache) the active chat model client.

    ``providers.get_chat_model`` is ``@lru_cache``d, so this just does the
    ~5s import+construction cost once, off the startup path — the first real
    LLM call then hits a warm cache instead of paying it. Any failure here
    (missing provider package, bad config) is swallowed; it surfaces
    naturally with the same RuntimeError on the first real call, which is
    where the fully-synchronous one-shot CLI path (_ensure_model_configured)
    still checks it eagerly, since that path is about to make a call anyway.
    """
    from wells import providers

    try:
        providers.get_chat_model(config.ACTIVE_PROFILE)
    except Exception:
        pass


def _ensure_model_configured() -> bool:
    """Full check used by the one-shot CLI path: profile resolves AND the
    client actually builds. Worth doing eagerly here — a one-shot run is
    about to need the client immediately, so there's no "meanwhile" for a
    background warm to hide behind (see _ensure_model_configured_fast for
    the TUI's non-blocking equivalent)."""
    if not _ensure_model_configured_fast():
        return False
    from wells import providers

    try:
        providers.get_chat_model(config.ACTIVE_PROFILE)
        return True
    except RuntimeError as e:
        print(f"ERROR: {e}")
        return False


def _reload_module_config() -> None:
    """Re-import config values that may have changed via the menu/overrides.

    Several modules captured values at import time; after the menu mutates the
    environment we refresh the ones that matter for a run.
    """
    import importlib

    importlib.reload(config)


def _run_index_cmd(args: list[str]) -> None:
    """Handle `wells index` subcommand (build/update/status/clear)."""
    from wells import index_tools
    from wells.tools import ToolContext

    if not index_tools.INDEXER_AVAILABLE:
        print("ERROR: Index engine not available. Install: pip install wells-index")
        sys.exit(1)

    ctx = ToolContext(workspace=config.WORKSPACE_ROOT)

    if not args or args[0] in ("build", "update", ""):
        # Build/update the index
        print(f"Indexing {config.WORKSPACE_ROOT}...")
        result = index_tools.index_workspace(ctx)
        if result.ok:
            print(result.output)
        else:
            print(f"ERROR: {result.error or result.output}")
            sys.exit(1)
    elif args[0] == "--status":
        # Show index statistics
        print("Repository index statistics:")
        result = index_tools.list_symbols(ctx, "")
        if result.ok:
            print(result.output)
        else:
            print(f"ERROR: {result.error or result.output}")
            sys.exit(1)
    elif args[0] == "--clear":
        # Clear the index
        print(f"Clearing index at {config.WORKSPACE_ROOT}...")
        try:
            from wells_index import IndexEngine
            engine = IndexEngine(config.WORKSPACE_ROOT)
            engine.clear()
            print("Index cleared.")
        except Exception as e:
            print(f"ERROR: Could not clear index: {e}")
            sys.exit(1)
    else:
        print(f"ERROR: Unknown index subcommand: {args[0]}")
        print("Usage: wells index [--status|--clear]")
        sys.exit(2)


def _run_sessions_cmd(args: list[str]) -> None:
    """Handle `wells sessions [list|delete|clear] [--all]` subcommand."""
    from wells.sessions import (
        clear_sessions, delete_session, format_age, list_sessions,
    )

    all_ws = "--all" in args
    sub_args = [a for a in args if a != "--all"]
    subcmd = sub_args[0] if sub_args else "list"
    workspace = None if all_ws else config.WORKSPACE_ROOT

    if subcmd in ("list", ""):
        sessions = list_sessions(workspace=workspace, limit=50)
        if not sessions:
            ws_note = "any workspace" if all_ws else f"workspace: {workspace}"
            print(f"No sessions found ({ws_note}).")
            return
        print(f"\n{'SESSION ID':<26}  {'AGE':<10}  {'STATUS':<12}  {'TOKENS':>8}   GOAL")
        print("-" * 92)
        for s in sessions:
            age = format_age(s.get("created_at", ""))
            status = s.get("status", "?")
            tok = (s.get("tokens_in") or 0) + (s.get("tokens_out") or 0)
            tok_s = f"{tok:,}" if tok else "?"
            goal = (s.get("goal") or "")[:48]
            print(f"{s['id']:<26}  {age:<10}  {status:<12}  {tok_s:>8}   {goal}")
        scope = "all workspaces" if all_ws else "this workspace"
        print(f"\n{len(sessions)} session(s) — {scope}. "
              f"Add --all to see every workspace.\n")

    elif subcmd == "delete":
        if len(sub_args) < 2:
            print("ERROR: sessions delete requires a SESSION_ID")
            print("Usage: wells sessions delete SESSION_ID")
            sys.exit(2)
        if delete_session(sub_args[1]):
            print(f"Deleted: {sub_args[1]}")
        else:
            print(f"Not found: {sub_args[1]}")
            sys.exit(1)

    elif subcmd == "clear":
        ws_note = "ALL workspaces" if all_ws else f"workspace: {workspace}"
        try:
            confirm = input(f"Delete all sessions for {ws_note}? [y/N] ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            return
        if confirm in ("y", "yes"):
            n = clear_sessions(workspace=workspace)
            print(f"Deleted {n} session(s).")
        else:
            print("Cancelled.")

    else:
        print(f"ERROR: Unknown sessions subcommand: {subcmd!r}")
        print("Usage: wells sessions [list|delete SESSION_ID|clear] [--all]")
        sys.exit(2)


def _handle_resume_flag(flag_value: str, goal_args: list[str]) -> tuple[str, str | None]:
    """Resolve -r/--resume into (goal, resume_context).

    ``flag_value`` is either "" (interactive picker) or a specific session ID.
    ``goal_args`` are the remaining CLI goal words.
    Returns (goal_to_run, resume_context_or_None).
    """
    from wells.sessions import (
        build_resume_context, format_age, is_session_id,
        list_sessions, load_session,
    )

    if flag_value and is_session_id(flag_value):
        session = load_session(flag_value)
        if not session:
            print(f"ERROR: Session not found: {flag_value}")
            sys.exit(2)
    else:
        # Interactive picker
        sessions = list_sessions(limit=10)
        if not sessions:
            print("No previous sessions found — starting fresh.")
            return " ".join(goal_args), None

        print("\nRecent sessions:")
        for i, s in enumerate(sessions, 1):
            age = format_age(s.get("created_at", ""))
            status_sym = "✓" if s.get("status") == "COMPLETE" else "~"
            goal_prev = (s.get("goal") or "")[:58]
            tok = (s.get("tokens_in") or 0) + (s.get("tokens_out") or 0)
            tok_s = f"{tok // 1000}K" if tok >= 1000 else str(tok)
            print(f"  {i}. [{age}] {status_sym} {goal_prev!r}  [{tok_s} tok]")
            print(f"     {s['id']}")
        print()
        try:
            choice = input("Select [1-N] or Enter to start fresh: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return " ".join(goal_args), None

        if not choice:
            return " ".join(goal_args), None

        try:
            session = sessions[int(choice) - 1]
        except (ValueError, IndexError):
            print(f"Invalid selection: {choice!r} — starting fresh.")
            return " ".join(goal_args), None

    # Show what we're resuming
    print(f"\nResuming: {session['id']}")
    print(f"Previous goal: {session.get('goal', '')}")
    print(f"Status: {session.get('status', '?')}")
    if session.get("git_summary"):
        print(f"Changes: {session['git_summary']}")
    print()

    # Goal is whatever the user passed on the command line.
    # Do NOT fall back to the session's previous goal — the caller uses an
    # empty goal to decide whether to launch the TUI or run a one-shot task.
    # The resume context already contains the previous goal as readable text.
    goal = " ".join(goal_args).strip()
    return goal, build_resume_context(session)


def _print_usage() -> None:
    print(__doc__)
    print(
        "\nFlags:\n"
        "  -w, --workspace PATH   operate on PATH instead of the current dir\n"
        "  -s, --safety MODE      auto | approve | dryrun\n"
        "  -r, --resume [SID]     resume a previous session (interactive or by ID)\n"
        "      --plan             plan mode (describe edits, don't apply)\n"
        "  -p, --print            headless one-shot; with no goal arg, reads the\n"
        "                         task from stdin instead of launching the REPL\n"
        "      --output-format F  text (default) | json — json implies --print;\n"
        "                         exit code 0 on COMPLETE, 1 otherwise\n"
        "      --json             shorthand for --output-format json\n"
        "      --version          show version and exit\n"
        "  -h, --help             show this help\n"
        "\nSubcommands:\n"
        "  sessions               list session history\n"
        "  sessions delete SID    delete one session\n"
        "  sessions clear         delete all sessions for current workspace\n"
        "  sessions --all         operate across all workspaces\n"
        "  traces                 list recorded run traces for this workspace\n"
        "  replay PATH|N|latest   re-run the harness over a recorded trace and\n"
        "                         report divergence (harness regression check)\n"
    )


def _run_traces_cmd(args: list[str]) -> None:
    from wells import traces

    ws = config.WORKSPACE_ROOT
    found = traces.list_traces(ws)
    if not found:
        print(f"No traces recorded under {ws}\\.wells\\traces "
              f"(runs record automatically; WELLS_TRACE=0 disables).")
        return
    print(f"\nRecorded traces in {ws} (newest last):")
    for i, p in enumerate(found, 1):
        try:
            import json as _json
            head = _json.loads(p.read_text(encoding="utf-8"))
            task_prev = " ".join((head.get("task") or "").split())[:60]
            info = (f"[{head.get('stopped_reason', '?')}, "
                    f"{head.get('steps_taken', '?')} steps] {task_prev!r}")
        except Exception:
            info = "(unreadable)"
        print(f"  {i}. {p.name}  {info}")
    print("\nReplay one with: wells replay <N|latest|path>")


def _run_replay_cmd(args: list[str]) -> None:
    from wells import traces

    if not args:
        print("usage: wells replay <N|latest|path-to-trace.json>")
        sys.exit(2)
    sel = args[0]
    ws = config.WORKSPACE_ROOT
    path = None
    if sel == "latest":
        found = traces.list_traces(ws)
        path = found[-1] if found else None
    elif sel.isdigit():
        found = traces.list_traces(ws)
        idx = int(sel) - 1
        path = found[idx] if 0 <= idx < len(found) else None
    else:
        p = Path(sel)
        path = p if p.is_file() else None
    if path is None:
        print(f"No trace matching {sel!r}. Run `wells traces` to list them.")
        sys.exit(2)

    print(f"Replaying {path.name} (recorded model outputs, stubbed tools) ...")
    report = traces.replay(path)
    print(f"\n  stop reason : recorded={report['recorded_stopped_reason']!r}  "
          f"replayed={report['stopped_reason']!r}")
    print(f"  steps       : recorded={report['recorded_steps']}  "
          f"replayed={report['steps_taken']}")
    print(f"  tool calls  : recorded={report['recorded_calls']}")
    print(f"                replayed={report['calls']}")
    if report["match"]:
        print("\n  MATCH — the harness makes the same decisions it made live.")
    else:
        print("\n  DIVERGED — harness behavior changed for this recorded run.")
        sys.exit(1)


def main() -> None:
    # Set uv link mode to avoid hardlink warning
    os.environ["UV_LINK_MODE"] = "copy"

    # Auto-setup on first run (build indexer, prompt for workspace, index)
    # This runs silently in background on first use
    try:
        from wells import setup
        setup.first_run_setup()
    except Exception:
        # Setup failures are non-fatal; system works without indexer (falls back to grep)
        pass

    argv = list(sys.argv[1:])

    # --version short-circuits everything.
    if "--version" in argv or "-V" in argv:
        print(f"wells {__version__}")
        return

    # ---- Pass 1: strip global flags (--workspace, --safety, --plan, --resume) ----
    workspace_override: str | None = None
    safety_override: str | None = None
    plan_flag = False
    print_flag = False   # -p/--print: headless one-shot (already the default for a goal arg)
    output_format = "text"
    resume_flag: str | None = None  # None = no resume; "" = interactive; "ID" = specific
    remaining: list[str] = []
    i = 0
    while i < len(argv):
        a = argv[i]
        if a in ("-p", "--print"):
            print_flag = True
        elif a == "--output-format":
            i += 1
            if i < len(argv):
                output_format = argv[i]
            else:
                print("ERROR: --output-format requires a value (text|json)")
                sys.exit(2)
        elif a.startswith("--output-format="):
            output_format = a.split("=", 1)[1]
        elif a == "--json":
            output_format = "json"
        elif a in ("-w", "--workspace"):
            i += 1
            if i < len(argv):
                workspace_override = argv[i]
            else:
                print("ERROR: --workspace requires a PATH argument")
                sys.exit(2)
        elif a.startswith("--workspace="):
            workspace_override = a.split("=", 1)[1]
        elif a in ("-s", "--safety"):
            i += 1
            if i < len(argv):
                safety_override = argv[i]
            else:
                print("ERROR: --safety requires a MODE argument")
                sys.exit(2)
        elif a.startswith("--safety="):
            safety_override = a.split("=", 1)[1]
        elif a == "--plan":
            plan_flag = True
        elif a in ("-r", "--resume"):
            from wells.sessions import is_session_id
            # Peek ahead: if next arg is a session ID, consume it
            if i + 1 < len(argv) and is_session_id(argv[i + 1]):
                i += 1
                resume_flag = argv[i]
            else:
                resume_flag = ""  # interactive picker
        else:
            remaining.append(a)
        i += 1

    # ---- Apply workspace/safety/plan overrides to the environment ----
    # We do NOT os.chdir() — that would break `uv run`'s project detection.
    # The harness passes workspace_root into the graph state, and the tool layer
    # uses it as the cwd for every subprocess it spawns.
    if workspace_override:
        ws = str(Path(workspace_override).resolve())
        if not Path(ws).is_dir():
            print(f"ERROR: workspace path does not exist or is not a directory: {ws}")
            sys.exit(2)
        os.environ["WORKSPACE_ROOT"] = ws
    if safety_override:
        os.environ["HARNESS_SAFETY"] = safety_override
    if plan_flag:
        os.environ["PLAN_MODE"] = "1"
    if workspace_override or safety_override or plan_flag:
        _reload_module_config()

    if output_format not in ("text", "json"):
        print(f"ERROR: --output-format must be 'text' or 'json', got {output_format!r}")
        sys.exit(2)
    if output_format == "json":
        print_flag = True  # JSON output only makes sense for a one-shot headless run

    # ---- Pass 2: subcommand detection on what's left ----
    if remaining and remaining[0] in ("-h", "--help", "help"):
        _print_usage()
        return
    if remaining and remaining[0] == "config":
        settings.interactive_menu(Path(".env"))
        return
    if remaining and remaining[0] == "info":
        # Apply any inline KEY=VALUE overrides first, then show.
        settings.parse_argv_settings(remaining[1:])
        _reload_module_config()
        _print_info()
        return
    if remaining and remaining[0] == "principles":
        # Show which AGENT.md principles are active (and where they come from).
        from wells import principles
        ws = config.WORKSPACE_ROOT
        print(f"\nPrinciples source: {principles.source_label(ws)}")
        print("-" * 60)
        print(principles.principles_text(ws))
        return
    if remaining and remaining[0] == "index":
        _run_index_cmd(remaining[1:])
        return
    if remaining and remaining[0] == "sessions":
        _run_sessions_cmd(remaining[1:])
        return
    if remaining and remaining[0] == "traces":
        _run_traces_cmd(remaining[1:])
        return
    if remaining and remaining[0] == "replay":
        _run_replay_cmd(remaining[1:])
        return

    # ---- Pass 3: a goal run — separate goal args from KEY=VALUE overrides ----
    overrides = [a for a in remaining if _looks_like_override(a)]
    goal_args = [a for a in remaining if not _looks_like_override(a)]

    if overrides:
        settings.parse_argv_settings(overrides)
        _reload_module_config()

    if not goal_args and print_flag and resume_flag is None:
        # -p/--print (or --output-format json) with no goal argument: read the
        # task from stdin, the standard scripting convention (echo "task" |
        # wells -p --output-format json). Ambiguous/empty stdin is a hard
        # error rather than silently falling through to the interactive REPL,
        # which would hang a CI job waiting on a TTY that will never answer.
        piped = sys.stdin.read().strip() if not sys.stdin.isatty() else ""
        if not piped:
            print("ERROR: -p/--print with no goal argument requires a task on stdin.")
            sys.exit(2)
        _run_goal(piped, output_format=output_format)
        return

    if not goal_args and resume_flag is None:
        # No goal and no resume flag — launch the interactive REPL.
        # Start the file-system watcher before entering the TUI so the index
        # stays live while the dev works (changes indexed within ~1.5s).
        if config.INDEX_AUTO_UPDATE:
            try:
                from wells import index_watcher, index_tools
                ws = config.WORKSPACE_ROOT
                started = index_watcher.start(ws)
                if started:
                    # Build index on first launch if missing; watcher handles
                    # all subsequent updates automatically.
                    index_tools.ensure_index(ws, auto_build=True)
            except Exception:
                pass  # watcher is optional — Wells works without it
        from wells.cli import run_repl
        run_repl()
        return

    if resume_flag is not None:
        goal, resume_ctx = _handle_resume_flag(resume_flag, goal_args)
        if goal:
            # User supplied a new goal on the CLI: run it once with context.
            _run_goal(goal, resume_context=resume_ctx, output_format=output_format)
        else:
            # No new goal: enter the TUI with the session context preloaded.
            from wells.cli import run_repl
            run_repl(resume_context=resume_ctx)
        return

    goal = " ".join(goal_args).strip()
    _run_goal(goal, output_format=output_format)


def _looks_like_override(arg: str) -> bool:
    """True if ``arg`` looks like ``KEY=VALUE`` (a settings override, not a goal)."""
    if "=" not in arg:
        return False
    key = arg.split("=", 1)[0]
    return key.isidentifier() or key.replace("_", "").isalnum()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        # Defense in depth: run_tui() already catches this around the TUI's
        # own asyncio.run(), but the one-shot goal path (_run_goal) and any
        # exception during asyncio's own shutdown can still surface here.
        print("\n[wells] interrupted.")
        sys.exit(130)
