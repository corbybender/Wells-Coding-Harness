"""Tester: runs the real test suite and interprets results.

Two layers:

1. **Deterministic gate** — when the repo has a recognizable test setup
   (pytest config, package.json test script, Cargo.toml, go.mod), the suite is
   run directly and its exit code is recorded as ``tests_passed`` ground truth.
   The model cannot claim "tests pass" when they don't, and the graph can
   fail-fast back to the coder without an LLM call when they clearly fail.
2. **Agentic interpretation** — an LLM pass characterizes failures (which
   assertion, which file:line) so the coder gets actionable feedback. Skipped
   entirely when the deterministic gate passes: a green suite needs no prose.

The tester runs with read+exec tools but NOT write tools, so it cannot alter
source — it only runs commands and inspects output.
"""

from pathlib import Path

from coding_harness import tools as _tools
from coding_harness.executor import run_executor
from coding_harness.tools import EXEC_TOOLS, ToolContext, registry


TESTER_TASK_TEMPLATE = """Run the project's tests for the workspace at {workspace} and report the result.

GOAL (for context):
{goal}

RECENTLY MADE CHANGES (coder summary):
{changes}
{ground_truth}
STEPS:
1. Inspect the repo layout (list_dir / read_file on the test config) to confirm
   the correct test command, unless one is obvious.
2. Run the test suite (run_tests, or run_command with the specific command).
3. If tests fail, read the failing test + the relevant source to characterize the
   failure precisely. Do NOT fix anything — that is the coder's job.
4. Reply with a concise report: PASS/FAIL, the command run, a one-line summary,
   and (if failing) the specific failing assertions with file:line references.
"""

_GROUND_TRUTH_TEMPLATE = """
GROUND TRUTH (the harness already ran the suite — do not dispute this):
The test suite was executed and FAILED. Output tail:
{output}

Skip step 1-2; the run above is authoritative. Focus on characterizing the
failures (step 3) and reporting them (step 4).
"""


def _has_test_setup(workspace: str) -> bool:
    """True when the repo has a recognizable, runnable test configuration."""
    root = Path(workspace)
    if any(
        (root / f).exists()
        for f in ("pyproject.toml", "pytest.ini", "setup.cfg", "Cargo.toml", "go.mod")
    ):
        return True
    if (root / "tests").is_dir():
        return True
    pkg = root / "package.json"
    if pkg.exists():
        try:
            import json
            test_script = (json.loads(pkg.read_text(encoding="utf-8"))
                           .get("scripts", {}).get("test", ""))
            return bool(test_script) and "no test specified" not in test_script
        except Exception:
            return False
    return False


def _run_deterministic_gate(ctx: ToolContext) -> tuple[bool | None, str]:
    """Run the detected test suite directly; return (passed, report).

    passed is None when the gate could not produce ground truth (no test setup,
    dry-run/plan mode, or the run was blocked by policy).
    """
    if not _has_test_setup(ctx.workspace):
        return None, ""
    result = _tools.dispatch("run_tests", {}, ctx)
    if result.simulated:
        return None, ""
    # Keep the tail — that's where test runners put the failure summary.
    lines = (result.output or result.error or "").strip().splitlines()
    tail = "\n".join(lines[-40:])
    return result.ok, tail


def tester(state: dict) -> dict:
    ctx = ToolContext.from_state(state)

    # ── Layer 1: deterministic gate ─────────────────────────────────────────
    print("[tester] running deterministic test gate ...")
    passed, gate_output = _run_deterministic_gate(ctx)

    if passed is True:
        report = f"PASS (deterministic run by harness)\n{gate_output}"
        print("[tester] suite green — skipping LLM interpretation.")
        out: dict = {
            "test_plan": report,
            "test_results": report,
            "tests_passed": True,
        }
        # Fast path: simple plan + deterministically green suite needs no
        # reviewer pass — auto-approve and save an entire agent run.
        if state.get("plan_complexity") == "simple":
            print("[tester] simple plan + green suite — auto-approving (reviewer skipped).")
            out["review_complete"] = True
            out["review_result"] = (
                "AUTO-APPROVED: simple plan and the full test suite passed "
                "deterministically (harness-run, exit code 0)."
            )
        return out

    # ── Layer 2: agentic interpretation ─────────────────────────────────────
    print("[tester] interpreting via executor ...")
    # Read+exec only — the tester must not edit source.
    toolset = registry(include_mutating=False) + [
        t for t in EXEC_TOOLS if t.name in ("run_tests", "run_command")
    ]

    ground_truth = (
        _GROUND_TRUTH_TEMPLATE.format(output=gate_output) if passed is False else ""
    )
    task = TESTER_TASK_TEMPLATE.format(
        workspace=ctx.workspace,
        goal=state.get("goal", ""),
        changes=state.get("implementation_steps", "") or "(none)",
        ground_truth=ground_truth,
    )

    from coding_harness import config as _config
    result = run_executor(
        task=task,
        ctx=ctx,
        toolset=toolset,
        max_steps=8,
        step_label="tester",
        temperature=0.0,
        # Verification is judgment-light — route to the cheap profile when set.
        profile=_config.cheap_profile_name() if _config.CHEAP_VERIFY else None,
    )

    print(f"[tester] done: {result.steps_taken} steps, reason={result.stopped_reason}")
    out: dict = {
        "test_plan": result.summary,  # keep the field name for backward compat
        "test_results": result.summary,
    }
    if passed is False:
        out["tests_passed"] = False
        # Feed the failure report to the coder through the feedback channel the
        # reviewer normally fills — the fail-fast route skips the reviewer.
        out["review_result"] = (
            f"AUTOMATED TEST GATE FAILED — fix the failing tests.\n\n{result.summary}"
        )
        out["review_complete"] = False
    return out
