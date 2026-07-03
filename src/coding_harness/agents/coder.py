"""Coder: makes real edits to the workspace via the agentic executor.

In v1 the coder only *described* changes. Now it drives the tool-calling
executor (:mod:`coding_harness.executor`) to actually read, edit, create files
and run lint/build inside the workspace — confined by the safety policy.

Behaviour by mode:
  * ``plan_mode`` on  -> the executor runs read-only and describes the edits it
    *would* make (mutating tools simulate). The returned summary is the plan.
  * otherwise         -> the executor applies edits, runs verification commands,
    and returns a summary of what changed + the verification result.

On loop iterations (>=2) the coder is seeded with the compressed review feedback
so it addresses the reviewer's specific concerns rather than starting over.
"""

from coding_harness import memory
from coding_harness.executor import run_executor
from coding_harness.tools import ToolContext


CODER_TASK_TEMPLATE = """You are implementing a pre-researched plan in the workspace at {workspace}.

GOAL:
{goal}

CONCRETE PLAN (produced by a planner that already read the codebase):
{context}

REVIEW FEEDBACK TO ADDRESS (if any):
{feedback}

{memory}
{repo_map}

The plan above already names the exact files, functions, and line numbers you need.
Follow it step by step:
1. Read ONLY the specific sections called out in the plan (use offset/limit — do not
   re-read whole files the planner already investigated).
2. Make the changes exactly as described.
3. Verify each change (re-read the edited section; run the verification step from the plan).
4. When done, reply with a concise summary of every file changed and the verification result.

Do not call tools further once you are done. Do not re-discover code the plan already identified.
"""


def _build_context_block(state: dict) -> str:
    """Compact durable context: prefer the rolling summary on loop iterations."""
    iteration = state.get("iteration", 0)
    task_summary = state.get("task_summary", "")
    if iteration >= 1 and task_summary:
        return task_summary
    plan = state.get("development_plan", "")
    arch = state.get("architecture", "")
    parts = []
    if plan:
        parts.append(f"Development plan:\n{plan}")
    if arch:
        parts.append(f"Architecture:\n{arch}")
    return "\n\n".join(parts) or "(none)"


def coder(state: dict) -> dict:
    iteration = state.get("iteration", 0) + 1
    print(f"[coder] iteration {iteration} - driving executor to implement the goal ...")

    ctx = ToolContext.from_state(state)
    feedback = state.get("review_result", "") or "(none)"
    mem_slice = memory.load(ctx.workspace).section_for_context(max_chars=2000)
    mem_block = (
        f"PROJECT MEMORY (AGENTS.md — established facts about this repo):\n{mem_slice}"
        if mem_slice
        else ""
    )
    from coding_harness.repomap import repo_map_block

    task = CODER_TASK_TEMPLATE.format(
        workspace=ctx.workspace,
        goal=state.get("goal", ""),
        context=_build_context_block(state),
        feedback=feedback,
        memory=mem_block,
        repo_map=repo_map_block(ctx.workspace, goal=state.get("goal", "")),
    )

    result = run_executor(
        task=task,
        ctx=ctx,
        step_label=f"coder-{iteration}",
        seed_messages=list(state.get("executor_messages") or []) or None,
    )

    print(
        f"[coder] executor done: {result.steps_taken} steps, reason={result.stopped_reason}"
    )

    return {
        "implementation_steps": result.summary,
        "code_changes": result.summary,
        "iteration": iteration,
        "executor_messages": result.messages,
    }
