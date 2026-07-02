"""Tests for the agentic executor loop (Layer 2).

Uses a scripted mock model so the loop logic is verified without live API calls.
Covers text-fallback parsing, native tool_calls, the step cap, and error paths.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
from langchain_core.messages import AIMessage

from coding_harness import config, executor, tools
from coding_harness.tokens import LEDGER


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    (tmp_path / "maths.py").write_text("def add(a, b):\n    return a - b\n")
    return tmp_path


@pytest.fixture
def ctx(workspace: Path) -> tools.ToolContext:
    return tools.ToolContext(workspace=str(workspace), safety="auto")


def _scripted(responses):
    """Return a fake _invoke_with_retry that yields the scripted AIMessages."""
    it = iter(responses)

    def _fake(llm, messages):
        try:
            return next(it)
        except StopIteration:
            return AIMessage(content="(done)")

    return _fake


# ---------------------------------------------------------------------------
# Text tool-call parsing
# ---------------------------------------------------------------------------


def test_parse_single_text_call():
    calls = executor.parse_text_tool_calls(
        '<tool_call>{"name": "read_file", "args": {"path": "x.py"}}</tool_call>'
    )
    assert calls == [{"name": "read_file", "args": {"path": "x.py"}}]


def test_parse_multiple_text_calls():
    text = '<tool_call>{"name": "a", "args": {}}</tool_call>\n text \n<tool_call>{"name": "b", "args": {"k": 1}}</tool_call>'
    calls = executor.parse_text_tool_calls(text)
    assert [c["name"] for c in calls] == ["a", "b"]
    assert calls[1]["args"] == {"k": 1}


def test_parse_arguments_alias():
    # Some models emit "arguments" instead of "args".
    calls = executor.parse_text_tool_calls(
        '<tool_call>{"name": "x", "arguments": {"k": 1}}</tool_call>'
    )
    assert calls[0]["args"] == {"k": 1}


def test_parse_malformed_skipped():
    calls = executor.parse_text_tool_calls("<tool_call>{not json}</tool_call>")
    assert len(calls) == 1
    assert "_parse_error" in calls[0]


# ---------------------------------------------------------------------------
# Call extraction (native vs text)
# ---------------------------------------------------------------------------


def test_extract_native_calls():
    msg = AIMessage(
        content="thinking",
        tool_calls=[
            {"name": "read_file", "args": {"path": "x"}, "id": "c1"},
        ],
    )
    calls = executor._extract_calls(msg, native_tools=True)
    assert calls == [{"name": "read_file", "args": {"path": "x"}, "id": "c1"}]


def test_extract_falls_back_to_text_when_no_native():
    msg = AIMessage(
        content='<tool_call>{"name": "grep", "args": {"pattern": "x"}}</tool_call>'
    )
    calls = executor._extract_calls(msg, native_tools=True)
    assert calls == [{"name": "grep", "args": {"pattern": "x"}}]


def test_extract_text_mode():
    msg = AIMessage(
        content='sure\n<tool_call>{"name": "list_dir", "args": {}}</tool_call>'
    )
    calls = executor._extract_calls(msg, native_tools=False)
    assert calls == [{"name": "list_dir", "args": {}}]


# ---------------------------------------------------------------------------
# Full loop (text fallback)
# ---------------------------------------------------------------------------


def test_loop_text_mode_edits_and_completes(ctx: tools.ToolContext, workspace: Path):
    LEDGER.reset()
    script = [
        AIMessage(
            content='<tool_call>{"name": "read_file", "args": {"path": "maths.py"}}</tool_call>'
        ),
        AIMessage(
            content='<tool_call>{"name": "edit_file", "args": {"path": "maths.py", "old_string": "return a - b", "new_string": "return a + b"}}</tool_call>'
        ),
        AIMessage(content="Done: fixed add() to return a + b."),
    ]
    with (
        patch.object(config, "_invoke_with_retry", side_effect=_scripted(script)),
        patch.object(executor, "_try_bind_tools", return_value=None),
    ):
        result = executor.run_executor(
            task="fix add()", ctx=ctx, max_steps=6, step_label="t"
        )
    assert result.stopped_reason == "done"
    assert result.steps_taken == 2
    assert {c["name"] for c in result.tool_calls} == {"read_file", "edit_file"}
    assert (workspace / "maths.py").read_text() == "def add(a, b):\n    return a + b\n"


def test_loop_native_mode(ctx: tools.ToolContext):
    LEDGER.reset()
    script = [
        AIMessage(
            content="listing",
            tool_calls=[{"name": "list_dir", "args": {"path": "."}, "id": "c1"}],
        ),
        AIMessage(content="All done."),
    ]
    # In native mode, _try_bind_tools returns the llm unchanged (truthy) so the
    # executor treats tool_calls as authoritative.
    with (
        patch.object(config, "_invoke_with_retry", side_effect=_scripted(script)),
        patch.object(executor, "_try_bind_tools", return_value=object()),
    ):
        result = executor.run_executor(
            task="list files", ctx=ctx, max_steps=4, step_label="t"
        )
    assert result.stopped_reason == "done"
    assert result.steps_taken == 1
    assert result.tool_calls[0]["name"] == "list_dir"


def test_loop_hits_step_cap(ctx: tools.ToolContext):
    LEDGER.reset()
    # Every response is another tool call -> never terminates naturally.
    repeat = AIMessage(
        content='<tool_call>{"name": "list_dir", "args": {}}</tool_call>'
    )
    script = [repeat] * 20
    with (
        patch.object(config, "_invoke_with_retry", side_effect=_scripted(script)),
        patch.object(executor, "_try_bind_tools", return_value=None),
    ):
        result = executor.run_executor(
            task="loop forever", ctx=ctx, max_steps=3, step_label="t"
        )
    assert result.stopped_reason == "max_steps"
    assert result.steps_taken == 3


def test_loop_handles_invoke_error(ctx: tools.ToolContext):
    LEDGER.reset()

    def boom(llm, messages):
        raise RuntimeError("network down")

    with (
        patch.object(config, "_invoke_with_retry", side_effect=boom),
        patch.object(executor, "_try_bind_tools", return_value=None),
    ):
        result = executor.run_executor(task="x", ctx=ctx, max_steps=3, step_label="t")
    assert result.stopped_reason == "error"
    assert "network down" in result.summary


def test_loop_records_token_usage(ctx: tools.ToolContext):
    LEDGER.reset()
    script = [
        AIMessage(content='<tool_call>{"name": "list_dir", "args": {}}</tool_call>'),
        AIMessage(content="done"),
    ]
    with (
        patch.object(config, "_invoke_with_retry", side_effect=_scripted(script)),
        patch.object(executor, "_try_bind_tools", return_value=None),
    ):
        executor.run_executor(task="x", ctx=ctx, max_steps=3, step_label="t")
    totals = LEDGER.totals()
    assert totals["calls"] >= 2  # at least the two scripted responses
    assert totals["input"] > 0


def test_loop_plan_mode_simulates_writes(ctx: tools.ToolContext, workspace: Path):
    LEDGER.reset()
    ctx.plan_mode = True
    script = [
        AIMessage(
            content='<tool_call>{"name": "edit_file", "args": {"path": "maths.py", "old_string": "a - b", "new_string": "a + b"}}</tool_call>'
        ),
        AIMessage(content="Plan: would change a-b to a+b."),
    ]
    with (
        patch.object(config, "_invoke_with_retry", side_effect=_scripted(script)),
        patch.object(executor, "_try_bind_tools", return_value=None),
    ):
        result = executor.run_executor(
            task="plan the fix", ctx=ctx, max_steps=3, step_label="t"
        )
    assert result.stopped_reason == "done"
    # The edit was simulated, not applied.
    assert result.tool_calls[0]["simulated"] is True
    assert "return a - b" in (workspace / "maths.py").read_text()


# ---------------------------------------------------------------------------
# Cooperative cancellation + per-run token budget
# ---------------------------------------------------------------------------


def test_loop_stops_on_cancel(ctx: tools.ToolContext):
    from coding_harness.control import CONTROL

    LEDGER.reset()
    CONTROL.reset()

    def cancelling(llm, messages):
        # First model call succeeds but the user cancels during it.
        CONTROL.cancel()
        return AIMessage(
            content='<tool_call>{"name": "list_dir", "args": {}}</tool_call>'
        )

    try:
        with (
            patch.object(config, "_invoke_with_retry", side_effect=cancelling),
            patch.object(executor, "_try_bind_tools", return_value=None),
        ):
            result = executor.run_executor(
                task="x", ctx=ctx, max_steps=10, step_label="t"
            )
    finally:
        CONTROL.reset()
    assert result.stopped_reason == "cancelled"
    # The tool call queued in the same response is skipped once cancelled.
    assert result.steps_taken == 0


def test_loop_stops_on_token_budget(ctx: tools.ToolContext):
    from coding_harness.control import CONTROL

    LEDGER.reset()
    CONTROL.reset()
    repeat = AIMessage(
        content='<tool_call>{"name": "list_dir", "args": {}}</tool_call>'
    )
    with (
        patch.object(config, "_invoke_with_retry", side_effect=_scripted([repeat] * 20)),
        patch.object(executor, "_try_bind_tools", return_value=None),
        patch.object(config, "MAX_RUN_TOKENS", 1),
    ):
        result = executor.run_executor(task="x", ctx=ctx, max_steps=10, step_label="t")
    # First round runs (usage 0 < budget), second round trips the cap.
    assert result.stopped_reason == "budget"
    assert "budget" in result.summary
