"""Planner: investigates the codebase and produces a concrete implementation plan.

The planner is an *agentic* step — it uses read-only tools (find_symbol, grep,
read_file, list_dir, glob) to actually explore the repo before writing a plan.
This grounds every downstream node in reality: the coder receives exact file
paths, function names, and line numbers rather than vague prose, so it can
execute with minimal additional discovery.

For less-capable models this is especially important: a concrete plan acts as
rails that guide the model step-by-step rather than leaving it to rediscover
the codebase from scratch inside the coder loop.
"""

from coding_harness import memory
from coding_harness.executor import run_executor
from coding_harness.tools import ToolContext
import coding_harness.tools as _tools


_PLANNER_PREFIX = """\
You are a senior software investigator and planner.
Your ONLY job right now is to INVESTIGATE the codebase and produce a CONCRETE plan.
Do NOT implement anything. Do NOT write code. Just read and plan.

Investigation strategy:
1. If the goal spans MULTIPLE independent areas (different modules, layers, or
   concerns), start with ONE parallel_research call — 2-4 focused questions, one
   per area. They run concurrently, so this is far faster than investigating
   each area yourself sequentially. For single-area goals, skip it.
2. Use find_symbol / search_symbols to locate relevant functions, classes, or modules.
3. Read only the specific sections you need — use offset/limit to avoid reading whole files.
4. Use grep when you need to find all usages or wiring of a pattern.
5. Stop reading once you have enough to write a specific plan.

Output format (use these exact headings):

COMPLEXITY: SIMPLE or COMPLEX
(SIMPLE = a few localized edits in existing files — small fix, rename, add a
button/flag/log line. COMPLEX = new modules, new APIs, data-model or schema
changes, cross-component wiring, migrations. When unsure, say COMPLEX.)

## Summary
One sentence: what will be done.

## Affected files
- relative/path/to/file.ext — what changes and roughly where (line numbers if known)

## Implementation steps
Numbered, specific steps. Each step must name the exact file and what to change.
Reference line numbers wherever possible.
Example: "3. Edit admin/index.html line 147 — add <button id='cancel'>Cancel</button> after the Save button div."

## Verification
What to run or check to confirm the task is complete.

## Risks / gotchas
Non-obvious constraints, naming collisions, side-effects to watch for.
"""


def planner(state: dict) -> dict:
    print("[planner] investigating codebase and drafting concrete plan ...")

    goal = state.get("goal", "")
    workspace = state.get("workspace_root")

    # Repo map: start the planner knowing where things live instead of
    # spending tool steps on discovery.
    from coding_harness.repomap import repo_map_block
    map_block = repo_map_block(workspace or "", goal=goal)

    # Weave in project memory so the planner knows established facts
    # about this repo (file layout, conventions, prior gotchas).
    task = memory.inject_into_prompt(
        f"Investigate the codebase and write a concrete implementation plan for:\n\n{goal}"
        f"\n{map_block}",
        workspace,
    )

    ctx = ToolContext.from_state(state)

    result = run_executor(
        task=task,
        ctx=ctx,
        toolset=_tools.READ_TOOLS,   # find_symbol, grep, read, list, glob — no writes, no shell
        system_prefix=_PLANNER_PREFIX,
        max_steps=25,                # enough to explore; not enough to drift
        step_label="planner",
    )

    plan = result.summary or "(planner produced no output)"
    complexity = _parse_complexity(plan)
    print(f"[planner] plan ready ({len(plan)} chars, complexity: {complexity})")
    return {"development_plan": plan, "plan_complexity": complexity}


def _parse_complexity(plan: str) -> str:
    """Read the COMPLEXITY marker from the plan head; default to complex.

    Complex is the safe default — it only costs one extra (cheap) architect
    call, whereas mis-labelling a complex task as simple skips design review.
    """
    for line in plan.strip().splitlines()[:6]:
        up = line.upper()
        if "COMPLEXITY" in up:
            return "simple" if "SIMPLE" in up else "complex"
    return "complex"
