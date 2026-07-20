"""Tests for trace v2: per-call usage_log capture + cache-efficiency analysis.

Covers:
  * ExecutorResult.usage_log round-trips into the trace JSON
  * analyze_trace returns per-call stats and flags cache-break points
  * analyze_traces aggregates across a corpus
  * v1 backward-compat: traces with no usage_log degrade gracefully
  * the live token report includes the new CACHE% column
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

from langchain_core.messages import AIMessage

from wells import traces
from wells.executor import ExecutorResult
from wells.tokens import LEDGER, StepUsage, TokenLedger


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _trace_with_usage(path: Path, *, usage: list[dict], **extra) -> Path:
    """Write a synthetic v2 trace file with the given per-call usage_log."""
    data = {
        "version": 2,
        "recorded_at": "2026-07-20 12:00:00",
        "task": extra.get("task", "synthetic task"),
        "step_label": "t",
        "stopped_reason": "done",
        "steps_taken": len(usage),
        "summary": "ok",
        "tool_calls": [],
        "messages": [],
        "usage_log": usage,
    }
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Trace v2 schema: usage_log round-trips
# ---------------------------------------------------------------------------


def test_record_run_writes_usage_log(tmp_path: Path):
    """ExecutorResult.usage_log must land in the trace JSON."""
    usage = [
        {
            "round": 1,
            "model": "test-model",
            "input": 1000,
            "output": 50,
            "reasoning": 0,
            "cache_read": 0,
            "cache_creation": 0,
            "mask_saved": 0,
            "drop_saved": 0,
            "ctx_tokens_at_call": 1000,
        },
        {
            "round": 2,
            "model": "test-model",
            "input": 200,
            "output": 60,
            "reasoning": 0,
            "cache_read": 800,
            "cache_creation": 0,
            "mask_saved": 0,
            "drop_saved": 0,
            "ctx_tokens_at_call": 1000,
        },
    ]
    r = ExecutorResult(summary="x", stopped_reason="done", usage_log=usage)
    p = traces.record_run(
        task="t",
        workspace=str(tmp_path),
        step_label="t",
        result=r,
    )
    assert p is not None
    data = json.loads(p.read_text(encoding="utf-8"))
    assert data["version"] == 2
    assert data["usage_log"] == usage


def test_record_run_handles_missing_usage_log(tmp_path: Path):
    """An ExecutorResult with the default empty usage_log writes an empty list."""
    r = ExecutorResult(summary="x", stopped_reason="done")
    p = traces.record_run(
        task="t",
        workspace=str(tmp_path),
        step_label="t",
        result=r,
    )
    data = json.loads(p.read_text(encoding="utf-8"))
    assert data["usage_log"] == []


# ---------------------------------------------------------------------------
# analyze_trace
# ---------------------------------------------------------------------------


def test_analyze_trace_returns_per_call_breakdown(tmp_path: Path):
    usage = [
        {
            "round": 1,
            "model": "m",
            "input": 1000,
            "output": 50,
            "reasoning": 0,
            "cache_read": 0,
            "cache_creation": 0,
            "mask_saved": 0,
            "drop_saved": 0,
            "ctx_tokens_at_call": 1000,
        },
        {
            "round": 2,
            "model": "m",
            "input": 200,
            "output": 60,
            "reasoning": 0,
            "cache_read": 800,
            "cache_creation": 0,
            "mask_saved": 0,
            "drop_saved": 0,
            "ctx_tokens_at_call": 1000,
        },
    ]
    p = _trace_with_usage(tmp_path / "t.json", usage=usage)
    r = traces.analyze_trace(p)
    assert r["calls"] == 2
    assert r["per_call"][0]["cache_efficiency"] == 0.0
    assert r["per_call"][1]["cache_efficiency"] == 0.8  # 800 / (200+800)
    assert r["totals"]["cache_read"] == 800
    # Aggregate efficiency = sum(cache_read) / (sum(input) + sum(cache_read))
    assert r["totals"]["cache_efficiency"] == 0.4  # 800 / (1200 + 800)


def test_analyze_trace_flags_cache_breaks(tmp_path: Path):
    """A round that drops from ~100% cache hit to <50% must be flagged."""
    usage = [
        {
            "round": 1,
            "model": "m",
            "input": 100,
            "output": 10,
            "reasoning": 0,
            "cache_read": 900,
            "cache_creation": 0,
            "mask_saved": 0,
            "drop_saved": 0,
            "ctx_tokens_at_call": 1000,
        },
        {
            "round": 2,
            "model": "m",
            "input": 800,
            "output": 10,
            "reasoning": 0,
            "cache_read": 200,
            "cache_creation": 0,
            "mask_saved": 500,
            "drop_saved": 0,
            "ctx_tokens_at_call": 1000,
        },
    ]
    p = _trace_with_usage(tmp_path / "t.json", usage=usage)
    r = traces.analyze_trace(p)
    assert any("round 2" in n and "dropped" in n for n in r["notes"])
    # And the masking ctx_op is included in the note.
    assert any("mask-500" in n for n in r["notes"])


def test_analyze_trace_handles_v1_format(tmp_path: Path):
    """v1 traces have no usage_log — analyze must degrade gracefully."""
    data = {
        "version": 1,
        "task": "old",
        "stopped_reason": "done",
        "steps_taken": 3,
        "tool_calls": [],
        "messages": [],
    }
    p = tmp_path / "v1.json"
    p.write_text(json.dumps(data), encoding="utf-8")
    r = traces.analyze_trace(p)
    assert r["calls"] == 0
    assert r["totals"]["cache_read"] == 0
    assert any("v1" in n.lower() for n in r["notes"])


def test_analyze_trace_notes_when_provider_has_no_cache(tmp_path: Path):
    """If cache_read is zero everywhere, the report should say so."""
    usage = [
        {
            "round": 1,
            "model": "m",
            "input": 500,
            "output": 50,
            "reasoning": 0,
            "cache_read": 0,
            "cache_creation": 0,
            "mask_saved": 0,
            "drop_saved": 0,
            "ctx_tokens_at_call": 500,
        },
    ]
    p = _trace_with_usage(tmp_path / "t.json", usage=usage)
    r = traces.analyze_trace(p)
    assert any("no cache hits" in n.lower() for n in r["notes"])


# ---------------------------------------------------------------------------
# analyze_traces (aggregate)
# ---------------------------------------------------------------------------


def test_analyze_traces_aggregates_and_ranks_breaks(tmp_path: Path):
    """Two traces: analyze_traces sums totals and surfaces the worst break."""
    # Trace A: round 2 has a 60-point drop.
    a_usage = [
        {
            "round": 1,
            "model": "m",
            "input": 100,
            "output": 10,
            "reasoning": 0,
            "cache_read": 900,
            "cache_creation": 0,
            "mask_saved": 0,
            "drop_saved": 0,
            "ctx_tokens_at_call": 1000,
        },
        {
            "round": 2,
            "model": "m",
            "input": 700,
            "output": 10,
            "reasoning": 0,
            "cache_read": 300,
            "cache_creation": 0,
            "mask_saved": 0,
            "drop_saved": 0,
            "ctx_tokens_at_call": 1000,
        },
    ]
    # Trace B: no cache activity at all.
    b_usage = [
        {
            "round": 1,
            "model": "m",
            "input": 500,
            "output": 50,
            "reasoning": 0,
            "cache_read": 0,
            "cache_creation": 0,
            "mask_saved": 0,
            "drop_saved": 0,
            "ctx_tokens_at_call": 500,
        },
    ]
    pa = _trace_with_usage(tmp_path / "a.json", usage=a_usage, task="A")
    pb = _trace_with_usage(tmp_path / "b.json", usage=b_usage, task="B")

    agg = traces.analyze_traces([pa, pb])
    assert agg["traces"] == 2
    assert agg["totals"]["calls"] == 3
    assert agg["totals"]["cache_read"] == 1200  # 900 + 300 + 0
    # The worst break is trace A's round 2 (~0.6 drop).
    assert agg["worst_breaks"][0][0] == str(pa)
    assert agg["worst_breaks"][0][1] == 2


def test_format_analysis_renders_table(tmp_path: Path):
    usage = [
        {
            "round": 1,
            "model": "m",
            "input": 100,
            "output": 10,
            "reasoning": 0,
            "cache_read": 0,
            "cache_creation": 0,
            "mask_saved": 0,
            "drop_saved": 0,
            "ctx_tokens_at_call": 100,
        },
        {
            "round": 2,
            "model": "m",
            "input": 50,
            "output": 10,
            "reasoning": 0,
            "cache_read": 50,
            "cache_creation": 0,
            "mask_saved": 200,
            "drop_saved": 0,
            "ctx_tokens_at_call": 100,
        },
    ]
    p = _trace_with_usage(tmp_path / "t.json", usage=usage)
    out = traces.format_analysis(traces.analyze_trace(p))
    assert "CACHE-EFFICIENCY REPORT" in out
    assert "mask-200" in out  # ctx_op column rendered
    assert "ROUND" in out and "EFF" in out  # header present


# ---------------------------------------------------------------------------
# Live token report: CACHE% column
# ---------------------------------------------------------------------------


def test_token_report_has_cache_percent_column():
    """format_report should include a CACHE% column header."""
    ledger = TokenLedger()
    ledger.steps.append(
        StepUsage(
            step="t",
            task_type="executor",
            model="m",
            input_tokens=500,
            output_tokens=50,
            reasoning_tokens=0,
            cache_read_tokens=500,
            category_tokens={},
            saved_by_trim=0,
            saved_by_summary=0,
        )
    )
    report = ledger.format_report()
    assert "CACHE%" in report
    assert "50%" in report  # 500 / (500+500)


# ---------------------------------------------------------------------------
# End-to-end: a real (scripted) executor run populates usage_log
# ---------------------------------------------------------------------------


def test_executor_run_populates_usage_log(tmp_path: Path, monkeypatch):
    """A scripted run should produce a non-empty usage_log on its result and
    in the recorded trace."""
    from wells import config, executor, tools

    monkeypatch.setenv("WELLS_TRACE", "1")
    LEDGER.reset()

    def _fake_invoke(llm, messages):
        # Simulate a provider reporting usage_metadata with cache_read.
        msg = AIMessage(content="done")
        msg.usage_metadata = {  # type: ignore[attr-defined]
            "input_tokens": 1000,
            "output_tokens": 50,
            "input_token_details": {"cache_read": 800},
        }
        return msg

    ctx = tools.ToolContext(workspace=str(tmp_path), safety="auto")
    with (
        patch.object(config, "_invoke_with_retry", side_effect=_fake_invoke),
        patch.object(config, "STRUCTURED_OUTPUTS", False),
        patch.object(executor, "_try_bind_tools", return_value=None),
    ):
        result = executor.run_executor(
            task="t",
            ctx=ctx,
            max_steps=1,
            step_label="t",
            quiet=True,
        )

    assert len(result.usage_log) >= 1
    entry = result.usage_log[0]
    assert entry["input"] == 1000
    assert entry["cache_read"] == 800
    assert entry["model"]  # populated

    # And the trace has it too.
    found = traces.list_traces(str(tmp_path))
    assert found
    data = json.loads(found[-1].read_text(encoding="utf-8"))
    assert data["usage_log"]
    assert data["usage_log"][0]["cache_read"] == 800
