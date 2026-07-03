"""Reviewer: verifies the real work product and emits COMPLETE/INCOMPLETE.

In v1 the reviewer judged text. Now it independently verifies the coder's work
by inspecting the actual workspace: it can read changed files, run the test
suite, and run lint/build — then decide whether the goal is genuinely met.

The reviewer runs with read+exec tools only (no writes), so it cannot "fix"
things to look good. Its decision still drives the graph loop: COMPLETE ends
the run; INCOMPLETE routes back through the summarizer to the coder with the
reviewer's specific feedback attached.
"""

from coding_harness.executor import run_executor
from coding_harness.tools import EXEC_TOOLS, ToolContext, registry

REVIEWER_TASK_TEMPLATE = """You are a strict senior reviewer. Independently verify whether the goal below
has been correctly and completely implemented in the workspace at {workspace}.

GOAL:
{goal}

CODER'S CLAIM (what it says it did):
{changes}

TESTER'S REPORT (test results):
{tests}

VERIFY BY DOING, not by trusting the claims:
1. Read the files the coder claims to have changed; confirm the changes exist
   and are correct.
2. If a test suite is available, run it. If running commands fails (no runner,
   browser-only UI, missing runtime), skip command verification and verify by
   reading the code. DO NOT mark INCOMPLETE just because run_command failed.
3. Check edge cases the coder may have missed.

Mark COMPLETE when the changed code plausibly implements the goal, even if you
cannot run it. Mark INCOMPLETE only when code is genuinely missing or wrong.

Then reply in EXACTLY this format on your FIRST line:
    DECISION: COMPLETE
or
    DECISION: INCOMPLETE
followed by a blank line and then your detailed review with specific, actionable
feedback.
"""


def _parse_decision(text: str) -> bool:
    """Return True if the reviewer marked the work COMPLETE (first non-empty line)."""
    first = ""
    for line in (text or "").strip().splitlines():
        if line.strip():
            first = line.upper()
            break
    if "INCOMPLETE" in first:
        return False
    if "COMPLETE" in first:
        return True
    return False


def reviewer(state: dict) -> dict:
    print("[reviewer] independently verifying the work via executor ...")
    ctx = ToolContext.from_state(state)
    toolset = registry(include_mutating=False) + [
        t for t in EXEC_TOOLS if t.name in ("run_tests", "run_command")
    ]

    task = REVIEWER_TASK_TEMPLATE.format(
        workspace=ctx.workspace,
        goal=state.get("goal", ""),
        changes=state.get("implementation_steps", "") or "(none)",
        tests=state.get("test_results") or state.get("test_plan", "") or "(not run)",
    )

    from coding_harness import config as _config
    result = run_executor(
        task=task,
        ctx=ctx,
        toolset=toolset,
        max_steps=10,
        step_label="reviewer",
        temperature=0.0,
        # Verification is judgment-light — route to the cheap profile when set.
        profile=_config.cheap_profile_name() if _config.CHEAP_VERIFY else None,
    )

    text = result.summary
    complete = _parse_decision(text)
    print(
        f"[reviewer] decision: {'COMPLETE' if complete else 'INCOMPLETE'} "
        f"({result.steps_taken} verify steps)"
    )

    return {"review_result": text, "review_complete": complete}
