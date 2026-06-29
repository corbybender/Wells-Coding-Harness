"""Entry point for the Wells agentic coding harness.

Usage:
    coding-harness "<your development goal>"        # run the harness
    coding-harness --workspace /path "fix the bug"  # run against another project
    coding-harness config                           # interactive settings menu
    coding-harness info                             # show effective config
    coding-harness --plan "<goal>"                  # plan mode (no edits)
    coding-harness --version                        # show version
    coding-harness "<goal>" MAX_ITERATIONS=5        # inline setting overrides
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from coding_harness import __version__, config, settings


def _print_section(title: str, body: str) -> None:
    line = "=" * 70
    print(f"\n{line}\n{title}\n{line}\n{body or '(empty)'}")


def _print_final_summary(state: dict) -> None:
    _print_section("DEVELOPMENT PLAN", state.get("development_plan", ""))
    _print_section("ARCHITECTURE PROPOSAL", state.get("architecture", ""))
    _print_section("IMPLEMENTATION STEPS", state.get("implementation_steps", ""))
    _print_section("TEST PLAN", state.get("test_plan", ""))
    _print_section("REVIEW RESULT", state.get("review_result", ""))

    status = "COMPLETE" if state.get("review_complete") else "INCOMPLETE"
    summary = (
        f"Goal: {state.get('goal', '')}\n"
        f"Status: {status}\n"
        f"Iterations used: {state.get('iteration', 0)} / "
        f"{state.get('max_iterations', config.MAX_ITERATIONS)}\n"
        f"Model: {config.model_name_for_task('coding')}"
    )
    _print_section("FINAL SUMMARY", summary)


def _print_info() -> None:
    """Print the effective configuration (resolved profiles + run knobs)."""
    from coding_harness import providers

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
        f"(reserved out {config.BUDGET.reserved_output_tokens})"
    )
    print(
        f"  Summarize on loop  : {'on' if config.SUMMARIZE_ON_LOOP else 'off'} "
        f"(threshold {config.SUMMARIZE_THRESHOLD})"
    )
    # Show the active principles source so users know which AGENT.md is in effect.
    try:
        from coding_harness import principles
        print(f"  Principles         : {principles.source_label(config.WORKSPACE_ROOT)}")
    except Exception:
        pass
    print(bar)


def _run_goal(goal: str) -> None:
    """Build and invoke the harness graph for ``goal``."""
    from coding_harness.graph import build_graph
    from coding_harness.tokens import LEDGER

    if not _ensure_model_configured():
        sys.exit(1)

    LEDGER.reset()
    print(f"Model: {config.model_name_for_task('coding')}")
    print(f"Workspace: {config.WORKSPACE_ROOT}  (safety: {config.HARNESS_SAFETY})")
    if config.PLAN_MODE:
        print("Plan mode: ON (coder will plan edits without applying them)")
    print(f"Max coder<->reviewer iterations: {config.MAX_ITERATIONS}")
    print(f"Goal: {goal}")
    print("-" * 70)

    app = build_graph()
    initial_state = {
        "goal": goal,
        "iteration": 0,
        "max_iterations": config.MAX_ITERATIONS,
        "workspace_root": config.WORKSPACE_ROOT,
        "safety": config.HARNESS_SAFETY,
        "plan_mode": config.PLAN_MODE,
        "messages": [],
    }

    final_state = app.invoke(initial_state)
    _print_final_summary(final_state)
    print("\n" + LEDGER.format_report())


def _ensure_model_configured() -> bool:
    """Check the active profile resolves + the provider package is installed."""
    from coding_harness import providers

    try:
        prof = providers.load_profile(config.ACTIVE_PROFILE)
    except Exception:
        prof = None
    if prof is None or not prof.model:
        print(
            f"ERROR: active profile {config.ACTIVE_PROFILE!r} has no model configured."
        )
        print(
            "Run `coding-harness config` to set it up, or set "
            f"MODEL_{config.ACTIVE_PROFILE}=<model> in your environment."
        )
        return False
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
    from coding_harness import index_tools
    from coding_harness.tools import ToolContext

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


def _print_usage() -> None:
    print(__doc__)
    print(
        "\nFlags:\n"
        "  -w, --workspace PATH   operate on PATH instead of the current dir\n"
        "  -s, --safety MODE      auto | approve | dryrun\n"
        "      --plan             plan mode (describe edits, don't apply)\n"
        "      --version          show version and exit\n"
        "  -h, --help             show this help\n"
    )


def main() -> None:
    argv = list(sys.argv[1:])

    # --version short-circuits everything.
    if "--version" in argv or "-V" in argv:
        print(f"wells {__version__}")
        return

    # ---- Pass 1: strip global flags (--workspace, --safety, --plan) ----
    # We pull these out FIRST so a subcommand like `info` or `config` can appear
    # after them (e.g. `--workspace /path info`) without being mistaken for a goal.
    workspace_override: str | None = None
    safety_override: str | None = None
    plan_flag = False
    remaining: list[str] = []
    i = 0
    while i < len(argv):
        a = argv[i]
        if a in ("-w", "--workspace"):
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
        from coding_harness import principles
        ws = config.WORKSPACE_ROOT
        print(f"\nPrinciples source: {principles.source_label(ws)}")
        print("-" * 60)
        print(principles.principles_text(ws))
        return
    if remaining and remaining[0] == "index":
        # Build or manage the repository index.
        _run_index_cmd(remaining[1:])
        return

    # ---- Pass 3: a goal run — separate goal args from KEY=VALUE overrides ----
    overrides = [a for a in remaining if _looks_like_override(a)]
    goal_args = [a for a in remaining if not _looks_like_override(a)]

    if overrides:
        settings.parse_argv_settings(overrides)
        _reload_module_config()

    if not goal_args:
        # No goal given — launch the interactive REPL.
        from coding_harness.cli import run_repl
        run_repl()
        return

    goal = " ".join(goal_args).strip()
    _run_goal(goal)


def _looks_like_override(arg: str) -> bool:
    """True if ``arg`` looks like ``KEY=VALUE`` (a settings override, not a goal)."""
    if "=" not in arg:
        return False
    key = arg.split("=", 1)[0]
    return key.isidentifier() or key.replace("_", "").isalnum()


if __name__ == "__main__":
    main()
