"""MCP server interface for the Wells coding-agent harness.

Exposes the harness capabilities as Model Context Protocol tools so external
agent clients (Claude Code, OpenCode, Codex-style CLIs, Gemini CLI, etc.) can
call into the harness via stdio transport.

Start the server:

    coding-harness-mcp          # console script
    python -m coding_harness.mcp_server

The MCP layer is deliberately thin: every tool delegates to the existing
harness core (graph, agents, executor, tools, gitops, memory). No orchestration
logic is duplicated here.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import mcp.server
import mcp.server.models
import mcp.server.stdio
import mcp.types as types

from coding_harness import config, gitops, memory
from coding_harness.compress import compress_output
from coding_harness.config import (
    HARNESS_SAFETY,
    MAX_ITERATIONS,
    MAX_TOOL_STEPS,
    PLAN_MODE,
    WORKSPACE_ROOT,
)
from coding_harness.graph import build_graph
from coding_harness.tokens import LEDGER
from coding_harness.tools import ToolContext, dispatch, registry

server = mcp.server.Server("coding-harness")


# ---------------------------------------------------------------------------
# Tool definitions
# ---------------------------------------------------------------------------


@server.list_tools()
async def handle_list_tools() -> list[types.Tool]:
    core = [
        types.Tool(
            name="run_agent_task",
            description="Run the full Wells harness: planner -> architect -> coder -> tester -> "
            "reviewer (up to max_iterations), then finalize (memory + optional git/PR). "
            "The coder/tester/reviewer are autonomous tool-using agents that actually "
            "edit files and run tests in the workspace.",
            inputSchema={
                "type": "object",
                "properties": {
                    "goal": {
                        "type": "string",
                        "description": "Software development goal",
                    },
                    "workspace": {
                        "type": "string",
                        "description": f"Workspace root (default {WORKSPACE_ROOT})",
                    },
                    "max_iterations": {"type": "integer", "default": MAX_ITERATIONS},
                    "safety": {
                        "type": "string",
                        "enum": ["auto", "approve", "dryrun"],
                        "default": HARNESS_SAFETY,
                    },
                    "plan_mode": {
                        "type": "boolean",
                        "default": False,
                        "description": "If true, plan edits without applying",
                    },
                },
                "required": ["goal"],
            },
        ),
        types.Tool(
            name="plan_task",
            description="Run only the planner and architect stages (fast). Reads AGENTS.md memory. "
            "Returns a development plan + architecture proposal.",
            inputSchema={
                "type": "object",
                "properties": {
                    "goal": {"type": "string"},
                    "workspace": {"type": "string", "default": WORKSPACE_ROOT},
                },
                "required": ["goal"],
            },
        ),
        types.Tool(
            name="review_code",
            description="Run the reviewer on provided implementation context. Returns DECISION "
            "(COMPLETE/INCOMPLETE) + detailed review.",
            inputSchema={
                "type": "object",
                "properties": {
                    "goal": {"type": "string"},
                    "plan": {"type": "string"},
                    "architecture": {"type": "string"},
                    "implementation_steps": {"type": "string"},
                    "test_plan": {"type": "string"},
                },
                "required": ["goal", "implementation_steps"],
            },
        ),
        types.Tool(
            name="run_executor",
            description="Run a single autonomous executor loop for an arbitrary task in the workspace. "
            "This is the claude-like agent: it can read/edit files and run commands via the "
            "safety policy. Returns the model's final summary.",
            inputSchema={
                "type": "object",
                "properties": {
                    "task": {
                        "type": "string",
                        "description": "What the executor should accomplish",
                    },
                    "workspace": {"type": "string", "default": WORKSPACE_ROOT},
                    "max_steps": {"type": "integer", "default": MAX_TOOL_STEPS},
                    "safety": {
                        "type": "string",
                        "enum": ["auto", "approve", "dryrun"],
                        "default": HARNESS_SAFETY,
                    },
                    "plan_mode": {"type": "boolean", "default": False},
                    "read_only": {
                        "type": "boolean",
                        "default": False,
                        "description": "Restrict to read tools",
                    },
                },
                "required": ["task"],
            },
        ),
        types.Tool(
            name="spawn_subagent",
            description="Spawn a focused research (read-only) or fix (can-edit) subagent for a scoped "
            "task. Returns a compact summary the caller can merge.",
            inputSchema={
                "type": "object",
                "properties": {
                    "task": {"type": "string"},
                    "role": {
                        "type": "string",
                        "enum": ["research", "fix"],
                        "default": "research",
                    },
                    "workspace": {"type": "string", "default": WORKSPACE_ROOT},
                    "max_steps": {"type": "integer", "default": 8},
                },
                "required": ["task"],
            },
        ),
        types.Tool(
            name="search_repo",
            description="Search a workspace via glob and/or grep. Read-only, always allowed.",
            inputSchema={
                "type": "object",
                "properties": {
                    "glob": {
                        "type": "string",
                        "description": "Filename glob, e.g. **/*.py",
                    },
                    "pattern": {
                        "type": "string",
                        "description": "Content regex (optional)",
                    },
                    "workspace": {"type": "string", "default": WORKSPACE_ROOT},
                    "path": {"type": "string", "default": "."},
                },
            },
        ),
        types.Tool(
            name="read_file",
            description="Read a file from the workspace (with line numbers). Read-only.",
            inputSchema={
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "workspace": {"type": "string", "default": WORKSPACE_ROOT},
                    "offset": {"type": "integer", "default": 1},
                    "limit": {"type": "integer", "default": 2000},
                },
                "required": ["path"],
            },
        ),
        types.Tool(
            name="run_command",
            description="Run a shell command in the workspace (confined, blocklisted, safety-gated). "
            "Returns compressed stdout/stderr + exit code.",
            inputSchema={
                "type": "object",
                "properties": {
                    "command": {"type": "string"},
                    "workspace": {"type": "string", "default": WORKSPACE_ROOT},
                    "safety": {
                        "type": "string",
                        "enum": ["auto", "approve", "dryrun"],
                        "default": HARNESS_SAFETY,
                    },
                },
                "required": ["command"],
            },
        ),
        types.Tool(
            name="git_status",
            description="Return git status + a short diff stat for the workspace (read-only).",
            inputSchema={
                "type": "object",
                "properties": {
                    "workspace": {"type": "string", "default": WORKSPACE_ROOT},
                },
            },
        ),
        types.Tool(
            name="get_memory",
            description="Read the project memory (AGENTS.md) accumulated across runs.",
            inputSchema={
                "type": "object",
                "properties": {
                    "workspace": {"type": "string", "default": WORKSPACE_ROOT},
                },
            },
        ),
        types.Tool(
            name="compress_logs",
            description="Compress a shell/test/build log blob (strips ANSI, dedupes, preserves errors).",
            inputSchema={
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "tail_lines": {"type": "integer", "default": 160},
                },
                "required": ["text"],
            },
        ),
        types.Tool(
            name="get_harness_info",
            description="Return the effective harness configuration: active profile, model, "
            "endpoint, safety policy, workspace, iteration caps, token budgets.",
            inputSchema={"type": "object", "properties": {}},
        ),
        types.Tool(
            name="get_principles",
            description="Return the active harness operating principles (AGENT.md) that govern "
            "every agent's behavior, regardless of which model is configured. Shows the "
            "source file and full principle text.",
            inputSchema={
                "type": "object",
                "properties": {
                    "workspace": {"type": "string", "default": WORKSPACE_ROOT},
                },
            },
        ),
        types.Tool(
            name="index_repo",
            description="Build or update the structural repository index (if wells-index is installed). "
            "Uses tree-sitter to extract symbols, create a knowledge graph, and store in SQLite. "
            "Only re-parses changed files. Transparent and fast on incremental runs.",
            inputSchema={
                "type": "object",
                "properties": {
                    "workspace": {"type": "string", "default": WORKSPACE_ROOT},
                },
            },
        ),
        types.Tool(
            name="find_symbol",
            description="Find definition location(s) for a symbol by exact name. "
            "Uses the repo index for fast, precise lookups (much faster than grep). "
            "Returns file path, line numbers, and symbol kind.",
            inputSchema={
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Symbol name to find"},
                    "workspace": {"type": "string", "default": WORKSPACE_ROOT},
                },
                "required": ["name"],
            },
        ),
        types.Tool(
            name="find_references",
            description="Find all files and lines that reference or call a symbol. "
            "Includes direct references, function calls, and inheritance relationships.",
            inputSchema={
                "type": "object",
                "properties": {
                    "symbol": {"type": "string", "description": "Symbol to find references for"},
                    "workspace": {"type": "string", "default": WORKSPACE_ROOT},
                },
                "required": ["symbol"],
            },
        ),
        types.Tool(
            name="find_callers",
            description="Find all functions and methods that call a given function. "
            "Useful for understanding how a function is used across the codebase.",
            inputSchema={
                "type": "object",
                "properties": {
                    "symbol": {"type": "string", "description": "Function/method name"},
                    "workspace": {"type": "string", "default": WORKSPACE_ROOT},
                },
                "required": ["symbol"],
            },
        ),
        types.Tool(
            name="search_symbols",
            description="Prefix/substring search for symbols across the index. "
            "Returns matching names and their locations.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Prefix or substring to search for"},
                    "limit": {"type": "integer", "default": 20},
                    "workspace": {"type": "string", "default": WORKSPACE_ROOT},
                },
                "required": ["query"],
            },
        ),
    ]
    return core


# ---------------------------------------------------------------------------
# Tool dispatch
# ---------------------------------------------------------------------------


def _format_output(**fields: Any) -> str:
    return json.dumps(fields, indent=2, default=str)


def _text_content(text: str) -> list[types.TextContent]:
    return [types.TextContent(type="text", text=text)]


def _ctx(args: dict) -> ToolContext:
    """Build a ToolContext from MCP args (workspace/safety/plan_mode overrides)."""
    return ToolContext(
        workspace=args.get("workspace") or WORKSPACE_ROOT,
        safety=args.get("safety") or HARNESS_SAFETY,
        plan_mode=bool(args.get("plan_mode", PLAN_MODE)),
    )


@server.call_tool()
async def handle_call_tool(name: str, arguments: dict) -> list[types.TextContent]:
    try:
        if name == "run_agent_task":
            return await _run_agent_task(arguments)
        if name == "plan_task":
            return await _plan_task(arguments)
        if name == "review_code":
            return await _review_code(arguments)
        if name == "run_executor":
            return await _run_executor(arguments)
        if name == "spawn_subagent":
            return await _spawn_subagent(arguments)
        if name == "search_repo":
            return _search_repo(arguments)
        if name == "read_file":
            return _read_file(arguments)
        if name == "run_command":
            return _run_command(arguments)
        if name == "git_status":
            return _git_status(arguments)
        if name == "get_memory":
            return _get_memory(arguments)
        if name == "compress_logs":
            return _compress_logs(arguments)
        if name == "get_harness_info":
            return _get_harness_info(arguments)
        if name == "get_principles":
            return _get_principles(arguments)
        if name == "index_repo":
            return _index_repo(arguments)
        if name == "find_symbol":
            return _find_symbol(arguments)
        if name == "find_references":
            return _find_references(arguments)
        if name == "find_callers":
            return _find_callers(arguments)
        if name == "search_symbols":
            return _search_symbols(arguments)
        return _text_content(_format_output(error=f"Unknown tool: {name}"))
    except Exception as exc:
        return _text_content(_format_output(error=f"{type(exc).__name__}: {exc}"))


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------


def _run_graph_sync(
    goal: str, max_iters: int, workspace: str, safety: str, plan_mode: bool
) -> dict:
    """Build + invoke the harness graph synchronously (runs in a thread)."""
    LEDGER.reset()
    app = build_graph()
    initial = {
        "goal": goal,
        "iteration": 0,
        "max_iterations": max_iters,
        "workspace_root": workspace,
        "safety": safety,
        "plan_mode": plan_mode,
        "messages": [],
    }
    return app.invoke(initial)


async def _run_agent_task(args: dict) -> list[types.TextContent]:
    goal = args["goal"]
    max_iters = args.get("max_iterations", MAX_ITERATIONS)
    workspace = args.get("workspace") or WORKSPACE_ROOT
    safety = args.get("safety") or HARNESS_SAFETY
    plan_mode = bool(args.get("plan_mode", False))

    final = await asyncio.to_thread(
        _run_graph_sync, goal, max_iters, workspace, safety, plan_mode
    )

    status = "COMPLETE" if final.get("review_complete") else "INCOMPLETE"
    extra: dict[str, Any] = {"status": status}
    for k in (
        "development_plan",
        "architecture",
        "implementation_steps",
        "test_plan",
        "review_result",
        "git_summary",
        "pr_url",
        "memory_written",
    ):
        if final.get(k):
            extra[k] = final[k]
    extra["iterations_used"] = final.get("iteration", 0)
    extra["max_iterations"] = final.get("max_iterations", max_iters)
    extra["token_usage"] = LEDGER.totals()
    return _text_content(_format_output(**extra))


async def _plan_task(args: dict) -> list[types.TextContent]:
    from coding_harness.agents.architect import architect
    from coding_harness.agents.planner import planner

    workspace = args.get("workspace") or WORKSPACE_ROOT
    state: dict = {
        "goal": args["goal"],
        "iteration": 0,
        "max_iterations": 1,
        "workspace_root": workspace,
        "safety": "dryrun",
        "messages": [],
    }

    def _run() -> dict:
        LEDGER.reset()
        state.update(planner(state))
        if state.get("development_plan"):
            state.update(architect(state))
        return state

    final = await asyncio.to_thread(_run)
    return _text_content(
        _format_output(
            development_plan=final.get("development_plan", ""),
            architecture=final.get("architecture", ""),
        )
    )


async def _review_code(args: dict) -> list[types.TextContent]:
    from coding_harness.agents.reviewer import reviewer

    state: dict = {
        "goal": args["goal"],
        "development_plan": args.get("plan", ""),
        "architecture": args.get("architecture", ""),
        "implementation_steps": args["implementation_steps"],
        "test_plan": args.get("test_plan", ""),
        "iteration": 1,
        "max_iterations": 1,
        "workspace_root": WORKSPACE_ROOT,
        "safety": "dryrun",
        "messages": [],
    }
    final = await asyncio.to_thread(lambda: reviewer(state))
    return _text_content(
        _format_output(
            review_result=final.get("review_result", ""),
            review_complete=final.get("review_complete", False),
        )
    )


async def _run_executor(args: dict) -> list[types.TextContent]:
    from coding_harness.executor import run_executor

    ctx = _ctx(args)
    toolset = registry(include_mutating=False) if args.get("read_only") else None
    result = await asyncio.to_thread(
        lambda: run_executor(
            task=args["task"],
            ctx=ctx,
            toolset=toolset,
            max_steps=args.get("max_steps", MAX_TOOL_STEPS),
        )
    )
    return _text_content(
        _format_output(
            summary=result.summary,
            steps_taken=result.steps_taken,
            stopped_reason=result.stopped_reason,
            tool_calls=[
                {k: v for k, v in c.items() if k != "output_preview"}
                for c in result.tool_calls
            ],
            token_usage=LEDGER.totals(),
        )
    )


async def _spawn_subagent(args: dict) -> list[types.TextContent]:
    from coding_harness import subagents

    ctx = _ctx(args)
    role = args.get("role", "research")
    task = args["task"]
    max_steps = args.get("max_steps", 8)
    spec = (subagents.fix_subagent if role == "fix" else subagents.research_subagent)(
        "mcp", task, max_steps=max_steps
    )
    report = await asyncio.to_thread(lambda: subagents.run_subagent(spec, ctx))
    return _text_content(
        _format_output(
            name=report.name,
            ok=report.ok,
            steps_taken=report.steps_taken,
            summary=report.summary,
        )
    )


def _search_repo(args: dict) -> list[types.TextContent]:
    ctx = _ctx(args)
    out: list[str] = []
    if args.get("glob"):
        r = dispatch(
            "glob", {"pattern": args["glob"], "path": args.get("path", ".")}, ctx
        )
        out.append(f"-- glob {args['glob']} --\n{r.output if r.ok else r.error}")
    if args.get("pattern"):
        r = dispatch(
            "grep", {"pattern": args["pattern"], "path": args.get("path", ".")}, ctx
        )
        out.append(f"-- grep {args['pattern']!r} --\n{r.output if r.ok else r.error}")
    if not out:
        return _text_content(_format_output(error="Provide `glob` and/or `pattern`"))
    return _text_content("\n\n".join(out))


def _read_file(args: dict) -> list[types.TextContent]:
    ctx = _ctx(args)
    r = dispatch(
        "read_file",
        {
            "path": args["path"],
            "offset": args.get("offset", 1),
            "limit": args.get("limit", 2000),
        },
        ctx,
    )
    return _text_content(r.output if r.ok else _format_output(error=r.error))


def _run_command(args: dict) -> list[types.TextContent]:
    ctx = _ctx(args)
    r = dispatch("run_command", {"command": args["command"]}, ctx)
    return _text_content(
        r.output if r.ok else _format_output(error=r.error, output=r.output)
    )


def _git_status(args: dict) -> list[types.TextContent]:
    ctx = _ctx(args)
    status = gitops._run(ctx, "git status --short --branch")
    diff = gitops.diff_summary(ctx)
    return _text_content(
        _format_output(
            status=status[1].strip() if status[0] else "(not a git repo)",
            diff_stat=diff.strip(),
        )
    )


def _get_memory(args: dict) -> list[types.TextContent]:
    mem = memory.load(args.get("workspace") or WORKSPACE_ROOT)
    return _text_content(
        _format_output(
            exists=mem.exists,
            path=str(mem.path) if mem.path else "",
            memory=mem.section_for_context(max_chars=8000) or "(empty)",
        )
    )


def _compress_logs(args: dict) -> list[types.TextContent]:
    return _text_content(
        compress_output(args["text"], tail_lines=args.get("tail_lines", 160))
    )


def _get_harness_info(args: dict) -> list[types.TextContent]:
    from coding_harness import providers

    info: dict[str, Any] = {
        "package": "coding-harness",
        "version": "0.2.0",
        "active_profile": config.ACTIVE_PROFILE,
        "model": config.model_name_for_task("coding"),
        "available_profiles": config.MODEL_PROFILES,
        "workspace_root": WORKSPACE_ROOT,
        "safety_policy": HARNESS_SAFETY,
        "plan_mode": PLAN_MODE,
        "max_iterations": MAX_ITERATIONS,
        "max_tool_steps": MAX_TOOL_STEPS,
        "token_budget_max_input": config.BUDGET.max_input_tokens,
        "summarize_on_loop": config.SUMMARIZE_ON_LOOP,
        "api_key_configured": False,
        "transport": "stdio",
    }
    try:
        from coding_harness import principles
        info["principles_source"] = principles.source_label(WORKSPACE_ROOT)
    except Exception:
        pass
    try:
        prof = providers.load_profile(config.ACTIVE_PROFILE)
        if prof:
            info["provider_kind"] = prof.kind
            info["base_url"] = prof.base_url or "(provider default)"
            info["api_key_configured"] = bool(prof.api_key)
    except Exception:
        pass
    return _text_content(_format_output(**info))


def _get_principles(args: dict) -> list[types.TextContent]:
    """Return the active operating principles (AGENT.md) + their source."""
    from coding_harness import principles
    ws = args.get("workspace") or WORKSPACE_ROOT
    return _text_content(_format_output(
        source=principles.source_label(ws),
        principles=principles.principles_text(ws),
    ))


def _index_repo(args: dict) -> list[types.TextContent]:
    """Build or update the repository index."""
    from coding_harness import index_tools

    if not index_tools.INDEXER_AVAILABLE:
        return _text_content(_format_output(
            error="Index engine not available. Install: pip install wells-index"
        ))

    ctx = _ctx(args)
    result = index_tools.index_workspace(ctx)
    return _text_content(_format_output(
        ok=result.ok,
        output=result.output,
        error=result.error,
    ))


def _find_symbol(args: dict) -> list[types.TextContent]:
    """Find definition(s) for a symbol by exact name."""
    from coding_harness import index_tools

    if not index_tools.INDEXER_AVAILABLE:
        return _text_content(_format_output(error="Index engine not available"))

    ctx = _ctx(args)
    result = index_tools.find_symbol(ctx, args["name"])
    return _text_content(_format_output(
        ok=result.ok,
        output=result.output,
        error=result.error,
    ))


def _find_references(args: dict) -> list[types.TextContent]:
    """Find all references to a symbol."""
    from coding_harness import index_tools

    if not index_tools.INDEXER_AVAILABLE:
        return _text_content(_format_output(error="Index engine not available"))

    ctx = _ctx(args)
    result = index_tools.find_references(ctx, args["symbol"])
    return _text_content(_format_output(
        ok=result.ok,
        output=result.output,
        error=result.error,
    ))


def _find_callers(args: dict) -> list[types.TextContent]:
    """Find all callers of a function."""
    from coding_harness import index_tools

    if not index_tools.INDEXER_AVAILABLE:
        return _text_content(_format_output(error="Index engine not available"))

    ctx = _ctx(args)
    result = index_tools.find_callers(ctx, args["symbol"])
    return _text_content(_format_output(
        ok=result.ok,
        output=result.output,
        error=result.error,
    ))


def _search_symbols(args: dict) -> list[types.TextContent]:
    """Search for symbols by prefix/substring."""
    from coding_harness import index_tools

    if not index_tools.INDEXER_AVAILABLE:
        return _text_content(_format_output(error="Index engine not available"))

    ctx = _ctx(args)
    limit = args.get("limit", 20)
    result = index_tools.search_symbols(ctx, args["query"], limit)
    return _text_content(_format_output(
        ok=result.ok,
        output=result.output,
        error=result.error,
    ))


# ---------------------------------------------------------------------------
# Server entry point
# ---------------------------------------------------------------------------


async def start_server() -> None:
    """Run the MCP server over stdio transport."""
    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            mcp.server.models.InitializationOptions(
                server_name="coding-harness",
                server_version="0.2.0",
                capabilities=types.ServerCapabilities(),
            ),
        )


def main() -> None:
    asyncio.run(start_server())


if __name__ == "__main__":
    main()
