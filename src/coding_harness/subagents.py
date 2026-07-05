"""Subagents: focused executor runs spawned from a parent agent.

This is what gives the harness claude-style parallelism: instead of one giant
context, the parent coder can spawn a *research* subagent to investigate a
specific area in parallel (read-only) and a *fix* subagent to make a small
scoped change — each with its own step budget and toolset, returning a compact
report that the parent merges.

Why it matters for token efficiency:
  * Each subagent carries only the slice of context relevant to its task, so the
    parent's context stays small.
  * Read-only subagents can't mutate state, so parallel research is safe.
  * Subagent results are compressed summaries — the parent never sees the
    subagent's full tool-call history.

Implementation:
  * Subagents are just :func:`executor.run_executor` calls with a constrained
    toolset and a short task description. No new orchestration framework — the
    same loop, smaller scope.
  * Parallelism is via :mod:`concurrent.futures` (threads, since the LLM calls
    are I/O-bound). The order of results is stable (sorted by subagent name).
  * The parent's :class:`ToolContext` is shared (same workspace/safety), so
    confinement and the safety policy apply uniformly.

Spawned subagents are surfaced through ``spawn_subagents`` and ``dispatch_subagents``
so they can be exposed as a tool (``spawn_subagent``) the model itself can call.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace

from coding_harness import executor, tools
from coding_harness.executor import ExecutorResult
from coding_harness.tools import ToolContext


# ---------------------------------------------------------------------------
# Subagent definitions
# ---------------------------------------------------------------------------


@dataclass
class SubagentSpec:
    """A subagent task: name, role, toolset, prompt."""

    name: str
    role: str  # research | fix | review (informational)
    task: str  # focused natural-language task
    toolset: str = "readonly"  # readonly | exec | full
    max_steps: int | None = None  # None -> config.SUBAGENT_MAX_STEPS (0 = no limit)
    temperature: float = 0.1


@dataclass
class SubagentReport:
    """Result of one subagent run, ready to hand back to the parent."""

    name: str
    ok: bool
    summary: str
    steps_taken: int
    tool_calls: list[dict] = field(default_factory=list)
    error: str = ""

    def as_context_block(self) -> str:
        """Compact text for the parent's context."""
        status = "ok" if self.ok else "ERROR"
        head = f"### Subagent [{self.name}] — {status} ({self.steps_taken} steps)"
        body = self.error if not self.ok and self.error else self.summary
        return f"{head}\n{body}"


# ---------------------------------------------------------------------------
# Toolset selection
# ---------------------------------------------------------------------------


_NO_NESTING = ("spawn_subagent", "parallel_research")


def _resolve_toolset(name: str) -> list[tools.ToolDef]:
    """Toolset for a subagent — never includes the spawning tools themselves
    (recursion guard; ctx.subagent enforces the same rule at dispatch time)."""
    if name == "readonly":
        base = tools.registry(include_mutating=False)
    elif name == "exec":
        base = tools.registry(include_mutating=False) + [
            t for t in tools.EXEC_TOOLS if t.name in ("run_tests", "run_command")
        ]
    else:  # full
        base = tools.ALL_TOOLS
    return [t for t in base if t.name not in _NO_NESTING]


# ---------------------------------------------------------------------------
# Run + spawn
# ---------------------------------------------------------------------------


def run_subagent(
    spec: SubagentSpec,
    ctx: ToolContext,
    *,
    profile: str | None = None,
    quiet: bool = False,
) -> SubagentReport:
    """Run a single subagent synchronously and return its report.

    ``quiet=True`` suppresses the subagent's per-step output (used for parallel
    fan-out where concurrent runs would interleave in the log).
    """
    toolset = _resolve_toolset(spec.toolset)
    sub_ctx = replace(ctx, subagent=True)
    if spec.max_steps is None:
        from coding_harness import config as _config
        cap = _config.SUBAGENT_MAX_STEPS  # 0 = no limit
    else:
        cap = spec.max_steps
    try:
        result: ExecutorResult = executor.run_executor(
            task=spec.task,
            ctx=sub_ctx,
            toolset=toolset,
            max_steps=cap,
            profile=profile,
            temperature=spec.temperature,
            step_label=f"subagent-{spec.name}",
            quiet=quiet,
        )
    except Exception as e:
        return SubagentReport(
            name=spec.name,
            ok=False,
            summary="",
            steps_taken=0,
            error=f"{type(e).__name__}: {e}",
        )

    return SubagentReport(
        name=spec.name,
        ok=result.stopped_reason != "error",
        summary=result.summary,
        steps_taken=result.steps_taken,
        tool_calls=result.tool_calls,
        error="" if result.stopped_reason != "error" else result.summary,
    )


def dispatch_subagents(
    specs: list[SubagentSpec],
    ctx: ToolContext,
    *,
    max_workers: int = 4,
    profile: str | None = None,
) -> list[SubagentReport]:
    """Run several subagents in parallel; return reports in input order.

    Parallelism is via threads (LLM calls are I/O-bound). Each subagent gets
    its own constrained toolset; all share the parent's ToolContext so workspace
    confinement and the safety policy still hold.
    """
    if not specs:
        return []
    if len(specs) == 1:
        return [run_subagent(specs[0], ctx, profile=profile)]

    workers = max(1, min(max_workers, len(specs)))
    with ThreadPoolExecutor(
        max_workers=workers, thread_name_prefix="wells-subagent"
    ) as pool:
        # quiet=True: concurrent runs must not interleave in the log.
        futures = {
            pool.submit(run_subagent, s, ctx, profile=profile, quiet=True): s.name
            for s in specs
        }
        by_name: dict[str, SubagentReport] = {}
        for fut in futures:
            by_name[futures[fut]] = fut.result()
    return [by_name[s.name] for s in specs]  # preserve input order


# ---------------------------------------------------------------------------
# Convenience builders for common subagent roles
# ---------------------------------------------------------------------------


def research_subagent(name: str, question: str, *, max_steps: int | None = None) -> SubagentSpec:
    """A read-only investigator. Cannot mutate the workspace."""
    task = (
        f"Research task (read-only). Investigate the codebase to answer this question precisely:\n"
        f"{question}\n\n"
        f"Use read_file / list_dir / glob / grep to find the relevant code. Do NOT edit anything.\n"
        f"Reply with a focused answer: the key files/symbols, how they fit together, and anything "
        f"the parent agent needs to know to act on this area."
    )
    return SubagentSpec(
        name=name, role="research", task=task, toolset="readonly", max_steps=max_steps
    )


def fix_subagent(name: str, change: str, *, max_steps: int | None = None) -> SubagentSpec:
    """A scoped editor. Makes a single, well-defined change and verifies it."""
    task = (
        f"Scoped edit task. Make exactly this change in the workspace, then verify it:\n"
        f"{change}\n\n"
        f"Read the affected file(s), make the minimal edit, run any quick verification (re-read, "
        f"or a targeted test), and reply with a one-paragraph summary of what you changed."
    )
    return SubagentSpec(
        name=name, role="fix", task=task, toolset="full", max_steps=max_steps
    )
