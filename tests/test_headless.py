"""Tests for headless/scriptable mode: -p/--print, --output-format json."""

from __future__ import annotations

import json
import sys
from unittest.mock import MagicMock, patch

import pytest

from wells import config, main


class _FakeGraph:
    def __init__(self, final_state: dict):
        self._final_state = final_state

    def invoke(self, initial_state):
        return self._final_state


def _patched(final_state: dict):
    """Context managers needed to run _run_goal() without touching real infra."""
    return (
        patch.object(main, "_ensure_model_configured", return_value=True),
        patch("wells.graph.build_graph", return_value=_FakeGraph(final_state)),
        patch("wells.sessions.new_session_id", return_value="20260101-000000-abcdef"),
        patch("wells.sessions.save_session"),
        patch("wells.sessions.session_from_final_state", return_value={}),
        patch("wells.pricing.run_cost", return_value=0.0123),
    )


def _run_json(capsys, final_state: dict, **kw) -> tuple[dict, int]:
    patchers = _patched(final_state)
    with patchers[0], patchers[1], patchers[2], patchers[3], patchers[4], patchers[5]:
        with pytest.raises(SystemExit) as ei:
            main._run_goal("do the thing", output_format="json", **kw)
    out = capsys.readouterr().out
    lines = [ln for ln in out.splitlines() if ln.strip()]
    assert len(lines) == 1, f"expected exactly one stdout line, got: {lines!r}"
    return json.loads(lines[0]), ei.value.code


# ---------------------------------------------------------------------------
# _run_goal(output_format="json")
# ---------------------------------------------------------------------------


def test_json_mode_emits_single_json_line_on_complete(capsys):
    payload, code = _run_json(capsys, {
        "review_complete": True, "iteration": 2, "max_iterations": 3,
        "implementation_steps": "did the thing", "review_result": "LGTM",
        "git_summary": "1 file changed",
    })
    assert payload["status"] == "complete"
    assert payload["summary"] == "did the thing"
    assert payload["session_id"] == "20260101-000000-abcdef"
    assert payload["cost_usd"] == 0.0123
    assert "tokens" in payload and "duration_seconds" in payload
    assert code == 0


def test_json_mode_incomplete_exits_nonzero(capsys):
    payload, code = _run_json(capsys, {
        "review_complete": False, "iteration": 3, "max_iterations": 3,
    })
    assert payload["status"] == "incomplete"
    assert code == 1


def test_json_mode_reviewer_error_reported_as_error_status(capsys):
    payload, code = _run_json(capsys, {"review_error": True, "review_result": "boom"})
    assert payload["status"] == "error"
    assert code == 1


def test_json_mode_suppresses_chatter_from_stdout(capsys):
    """Progress prints ('Model:', 'Goal:', etc.) must not pollute stdout —
    only the final JSON line belongs there."""
    payload, _ = _run_json(capsys, {"review_complete": True})
    err = capsys.readouterr()  # second call drains what's left (session line etc.)
    combined_out = err.out
    assert "Model:" not in combined_out
    assert "Goal:" not in combined_out


def test_text_mode_unchanged_prints_human_readable(capsys):
    patchers = _patched({"review_complete": True, "implementation_steps": "done"})
    with patchers[0], patchers[1], patchers[2], patchers[3], patchers[4], patchers[5]:
        main._run_goal("do the thing", output_format="text")
    out = capsys.readouterr().out
    assert "Goal: do the thing" in out
    assert "COMPLETE" in out
    # Must NOT be JSON — no attempt to parse the whole thing as one object.
    with pytest.raises(json.JSONDecodeError):
        json.loads(out)


# ---------------------------------------------------------------------------
# CLI flag parsing (main())
# ---------------------------------------------------------------------------


def test_output_format_json_implies_print_and_reads_stdin(monkeypatch, capsys):
    """--output-format json with no goal arg must read the task from stdin,
    not fall through to the interactive REPL (which would hang CI)."""
    monkeypatch.setattr(sys, "argv", ["wells", "--output-format", "json"])
    monkeypatch.setattr(sys.stdin, "isatty", lambda: False)
    monkeypatch.setattr(sys.stdin, "read", lambda: "fix the bug\n")
    captured = {}

    def _fake_run_goal(goal, **kw):
        captured["goal"] = goal
        captured["output_format"] = kw.get("output_format")

    with (
        patch("wells.setup.first_run_setup"),
        patch.object(main, "_run_goal", side_effect=_fake_run_goal),
    ):
        main.main()
    assert captured == {"goal": "fix the bug", "output_format": "json"}


def test_print_flag_with_no_goal_and_no_stdin_errors_instead_of_hanging(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["wells", "-p"])
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)  # real terminal, no pipe
    with patch("wells.setup.first_run_setup"):
        with pytest.raises(SystemExit) as ei:
            main.main()
    assert ei.value.code == 2


def test_invalid_output_format_rejected(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["wells", "--output-format", "yaml", "goal text"])
    with patch("wells.setup.first_run_setup"):
        with pytest.raises(SystemExit) as ei:
            main.main()
    assert ei.value.code == 2


def test_goal_arg_with_json_flag_bypasses_stdin(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["wells", "--json", "fix the bug"])
    captured = {}

    def _fake_run_goal(goal, **kw):
        captured["goal"] = goal
        captured["output_format"] = kw.get("output_format")

    with (
        patch("wells.setup.first_run_setup"),
        patch.object(main, "_run_goal", side_effect=_fake_run_goal),
    ):
        main.main()
    assert captured == {"goal": "fix the bug", "output_format": "json"}
