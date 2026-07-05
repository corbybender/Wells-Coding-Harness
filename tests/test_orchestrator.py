"""Tests for the dynamic graph routing and the deterministic test gate."""

from __future__ import annotations

from pathlib import Path

from coding_harness.agents.planner import _parse_complexity
from coding_harness.agents import tester
from coding_harness.graph import (
    _route_after_plan,
    _route_after_review,
    _route_after_tests,
    build_graph,
)
from coding_harness import tools


# ---------------------------------------------------------------------------
# Planner complexity marker
# ---------------------------------------------------------------------------


def test_parse_complexity_simple():
    assert _parse_complexity("COMPLEXITY: SIMPLE\n## Summary\nAdd a button.") == "simple"


def test_parse_complexity_complex():
    assert _parse_complexity("COMPLEXITY: COMPLEX\n## Summary\nNew module.") == "complex"


def test_parse_complexity_defaults_to_complex():
    assert _parse_complexity("## Summary\nNo marker present.") == "complex"
    assert _parse_complexity("") == "complex"


# ---------------------------------------------------------------------------
# Graph routing
# ---------------------------------------------------------------------------


def test_route_after_plan():
    assert _route_after_plan({"plan_complexity": "simple"}) == "code"
    assert _route_after_plan({"plan_complexity": "complex"}) == "design"
    assert _route_after_plan({}) == "design"


def test_route_after_tests_fail_fast():
    state = {"tests_passed": False, "iteration": 1, "max_iterations": 3}
    assert _route_after_tests(state) == "loop"


def test_route_after_tests_cap_goes_to_reviewer():
    state = {"tests_passed": False, "iteration": 3, "max_iterations": 3}
    assert _route_after_tests(state) == "review"


def test_route_after_tests_pass_or_unknown_goes_to_reviewer():
    assert _route_after_tests({"tests_passed": True}) == "review"
    assert _route_after_tests({}) == "review"


def test_route_after_review():
    assert _route_after_review({"review_complete": True}) == "finalize"
    assert (
        _route_after_review({"review_complete": False, "iteration": 3, "max_iterations": 3})
        == "finalize"
    )
    assert (
        _route_after_review({"review_complete": False, "iteration": 1, "max_iterations": 3})
        == "loop"
    )


def test_graph_compiles_with_conditional_edges():
    assert build_graph() is not None


# ---------------------------------------------------------------------------
# Deterministic test gate
# ---------------------------------------------------------------------------


def test_has_test_setup_detects_pyproject(tmp_path: Path):
    (tmp_path / "pyproject.toml").write_text("[project]\nname='x'\n")
    assert tester._has_test_setup(str(tmp_path)) is True


def test_has_test_setup_empty_dir(tmp_path: Path):
    assert tester._has_test_setup(str(tmp_path)) is False


def test_has_test_setup_npm_default_stub_ignored(tmp_path: Path):
    (tmp_path / "package.json").write_text(
        '{"scripts": {"test": "echo \\"Error: no test specified\\" && exit 1"}}'
    )
    assert tester._has_test_setup(str(tmp_path)) is False


def test_has_test_setup_npm_real_script(tmp_path: Path):
    (tmp_path / "package.json").write_text('{"scripts": {"test": "vitest run"}}')
    assert tester._has_test_setup(str(tmp_path)) is True


def test_deterministic_gate_none_without_setup(tmp_path: Path):
    ctx = tools.ToolContext(workspace=str(tmp_path), safety="auto")
    passed, report = tester._run_deterministic_gate(ctx)
    assert passed is None
    assert report == ""


def test_deterministic_gate_simulated_in_plan_mode(tmp_path: Path):
    (tmp_path / "pyproject.toml").write_text("[project]\nname='x'\n")
    ctx = tools.ToolContext(workspace=str(tmp_path), safety="auto", plan_mode=True)
    passed, _ = tester._run_deterministic_gate(ctx)
    assert passed is None  # simulated runs are not ground truth


# ---------------------------------------------------------------------------
# Parallel research fan-out
# ---------------------------------------------------------------------------

import threading
import time as _t
from unittest.mock import patch

from coding_harness import subagents
from coding_harness.subagents import SubagentReport
from coding_harness.tools import _parallel_research, get_tool


def _fake_report(spec, ctx, *, profile=None, quiet=False):
    _t.sleep(0.05)
    return SubagentReport(
        name=spec.name,
        ok=True,
        summary=f"answer for {spec.name} on {threading.current_thread().name}",
        steps_taken=2,
    )


def test_parallel_research_merges_reports_in_order(tmp_path):
    ctx = tools.ToolContext(workspace=str(tmp_path), safety="auto")
    with patch.object(subagents, "run_subagent", side_effect=_fake_report):
        result = _parallel_research(
            ctx, questions=["how does auth work?", "where are routes?", "db layer?"]
        )
    assert result.ok
    assert result.output.index("research-1") < result.output.index("research-2")
    assert result.output.index("research-2") < result.output.index("research-3")


def test_parallel_research_runs_concurrently(tmp_path):
    ctx = tools.ToolContext(workspace=str(tmp_path), safety="auto")
    threads_seen: set[str] = set()

    def record(spec, ctx, *, profile=None, quiet=False):
        threads_seen.add(threading.current_thread().name)
        _t.sleep(0.05)
        return SubagentReport(name=spec.name, ok=True, summary="x", steps_taken=1)

    with patch.object(subagents, "run_subagent", side_effect=record):
        _parallel_research(ctx, questions=["a?", "b?", "c?"])
    assert len(threads_seen) > 1  # actually fanned out across threads


def test_parallel_research_blocks_recursion(tmp_path):
    ctx = tools.ToolContext(workspace=str(tmp_path), safety="auto", subagent=True)
    result = _parallel_research(ctx, questions=["q?"])
    assert not result.ok
    assert "cannot spawn" in result.error


def test_parallel_research_requires_questions(tmp_path):
    ctx = tools.ToolContext(workspace=str(tmp_path), safety="auto")
    result = _parallel_research(ctx, questions=[])
    assert not result.ok


def test_parallel_research_registered_and_readonly():
    tool = get_tool("parallel_research")
    assert tool is not None and tool.mutating is False
    # Available in the read-only registry (planner/reviewer/tester toolsets).
    assert any(t.name == "parallel_research" for t in tools.registry(include_mutating=False))


def test_subagent_toolsets_exclude_spawning_tools():
    for kind in ("readonly", "exec", "full"):
        names = {t.name for t in subagents._resolve_toolset(kind)}
        assert "parallel_research" not in names
        assert "spawn_subagent" not in names


def test_quiet_executor_emits_no_ui_events(tmp_path):
    from langchain_core.messages import AIMessage
    from coding_harness import config, executor
    from coding_harness.control import CONTROL
    from coding_harness.tokens import LEDGER

    LEDGER.reset()
    CONTROL.reset()
    seen: list = []
    CONTROL.set_listener(lambda ev: seen.append(ev))
    script = iter([
        AIMessage(content='<tool_call>{"name": "list_dir", "args": {}}</tool_call>'),
        AIMessage(content="done"),
    ])
    ctx = tools.ToolContext(workspace=str(tmp_path), safety="auto")
    try:
        with (
            patch.object(config, "_invoke_with_retry", side_effect=lambda l, m: next(script)),
            patch.object(executor, "_try_bind_tools", return_value=None),
        ):
            result = executor.run_executor(
                task="x", ctx=ctx, max_steps=3, step_label="t", quiet=True
            )
    finally:
        CONTROL.set_listener(None)
    assert result.stopped_reason == "done"
    assert seen == []  # quiet run emitted nothing


# ---------------------------------------------------------------------------
# Latency cuts: ops routing, auto-approve fast path, /steer injection
# ---------------------------------------------------------------------------


def test_ops_verbs_route_to_auto():
    from coding_harness.chat import classify_intent
    for text in [
        "push the previous changes up to our azure app service",
        "deploy the site to production",
        "publish the package to npm",
        "restart the api service",
        "please push the build to the platform",
    ]:
        assert classify_intent(text, use_llm_fallback=False) == "auto", text


def test_route_after_tests_auto_approve_finalizes():
    state = {"tests_passed": True, "review_complete": True}
    assert _route_after_tests(state) == "finalize"
    # Green but complex (tester didn't auto-approve) still goes to review.
    assert _route_after_tests({"tests_passed": True}) == "review"


def test_tester_auto_approves_simple_green(tmp_path: Path):
    with patch.object(tester, "_run_deterministic_gate", return_value=(True, "1 passed")):
        out = tester.tester({
            "workspace_root": str(tmp_path),
            "plan_complexity": "simple",
        })
    assert out["tests_passed"] is True
    assert out["review_complete"] is True
    with patch.object(tester, "_run_deterministic_gate", return_value=(True, "1 passed")):
        out2 = tester.tester({
            "workspace_root": str(tmp_path),
            "plan_complexity": "complex",
        })
    assert "review_complete" not in out2  # complex still gets reviewed


def test_steer_injected_into_next_round(tmp_path: Path):
    from langchain_core.messages import AIMessage, HumanMessage

    from coding_harness import config, executor
    from coding_harness.control import CONTROL
    from coding_harness.tokens import LEDGER

    LEDGER.reset()
    CONTROL.reset()
    seen_steer_at_call = []

    responses = iter([
        AIMessage(content='<tool_call>{"name": "list_dir", "args": {}}</tool_call>'),
        AIMessage(content="done"),
    ])

    def fake_invoke(llm, messages):
        seen_steer_at_call.append(any(
            isinstance(m, HumanMessage) and "USER STEER" in (m.content or "")
            for m in messages
        ))
        return next(responses)

    ctx = tools.ToolContext(workspace=str(tmp_path), safety="auto")
    CONTROL.add_steer("switch to SQLite instead")
    with (
        patch.object(config, "_invoke_with_retry", side_effect=fake_invoke),
        patch.object(executor, "_try_bind_tools", return_value=None),
    ):
        executor.run_executor(task="x", ctx=ctx, max_steps=4, step_label="t")
    # The steer was present in the messages of the first (and later) calls.
    assert seen_steer_at_call and seen_steer_at_call[0] is True
    assert CONTROL.pending_steers() == 0


def test_iteration_cap_zero_never_finalizes_by_cap():
    state = {"review_complete": False, "iteration": 99, "max_iterations": 0}
    assert _route_after_review(state) == "loop"
    state_tests = {"tests_passed": False, "iteration": 99, "max_iterations": 0}
    assert _route_after_tests(state_tests) == "loop"
