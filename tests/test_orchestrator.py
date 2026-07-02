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
