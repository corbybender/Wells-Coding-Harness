"""Run traces: record executor runs, replay them as harness regression tests.

Every live failure so far had to be reconstructed by hand (SSH sessions,
copy-pasted TUI output, guesswork about what the model actually said).
Instead: each executor run is recorded — task, every model reply, every tool
call with its result, the stop reason — as one JSON file under
``.wells/traces/``. ``wells replay <trace>`` then re-runs the *harness* over
the recorded model outputs with tool dispatch stubbed to the recorded
results, and reports whether the harness still makes the same decisions
(same tool-call sequence, same stop reason).

That turns any failure in the wild into a one-command, permanent regression
fixture: change the parser/nudges/loop detectors, replay the trace corpus,
see immediately which recorded behaviors changed. The model itself is never
called during replay — this tests the harness, not the model.

Recording is best-effort and never breaks a run. ``WELLS_TRACE=0`` disables;
the newest ``WELLS_TRACE_KEEP`` traces are kept per workspace (default 20).
"""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path

TRACE_VERSION = 2
TRACE_SUBDIR = Path(".wells") / "traces"

# Monotonic per-process counter: same-second runs (stepwise mode, parallel
# subagents) must never collide on filename. id()-based suffixes don't work —
# CPython reuses freed object ids within a loop.
_SEQ = 0


def _next_seq() -> int:
    global _SEQ
    _SEQ += 1
    return _SEQ


def enabled() -> bool:
    return os.environ.get("WELLS_TRACE", "1").strip().lower() not in (
        "0",
        "false",
        "no",
        "off",
    )


def _keep() -> int:
    try:
        return max(1, int(os.environ.get("WELLS_TRACE_KEEP", "20")))
    except ValueError:
        return 20


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------


def _serialize_message(m) -> dict:
    from langchain_core.messages import (
        AIMessage,
        HumanMessage,
        SystemMessage,
        ToolMessage,
    )

    content = getattr(m, "content", "") or ""
    if isinstance(content, list):
        content = " ".join(b.get("text", "") for b in content if isinstance(b, dict))
    if isinstance(m, ToolMessage):
        mtype = "tool"
    elif isinstance(m, AIMessage):
        mtype = "ai"
    elif isinstance(m, SystemMessage):
        mtype = "system"
    elif isinstance(m, HumanMessage):
        mtype = "human"
    else:
        mtype = "other"
    out: dict = {"type": mtype, "content": str(content)}
    tcs = getattr(m, "tool_calls", None) or []
    if tcs:
        out["tool_calls"] = [
            {"name": tc.get("name"), "args": tc.get("args") or {}, "id": tc.get("id")}
            for tc in tcs
        ]
    if isinstance(m, ToolMessage):
        out["name"] = m.name
        out["tool_call_id"] = m.tool_call_id
    return out


# ---------------------------------------------------------------------------
# Record
# ---------------------------------------------------------------------------


def record_run(
    *, task: str, workspace: str | None, step_label: str, result
) -> Path | None:
    """Write one run's trace under <workspace>/.wells/traces. Never raises."""
    if not enabled():
        return None
    try:
        root = Path(workspace or ".")
        if not root.is_dir():
            return None
        d = root / TRACE_SUBDIR
        d.mkdir(parents=True, exist_ok=True)
        ts = time.strftime("%Y%m%d-%H%M%S")
        slug = re.sub(r"[^\w-]+", "-", step_label or "run").strip("-")[:40] or "run"
        path = d / f"{ts}-{slug}-{os.getpid()}-{_next_seq():04d}.json"
        data = {
            "version": TRACE_VERSION,
            "recorded_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "task": task,
            "step_label": step_label,
            "stopped_reason": result.stopped_reason,
            "steps_taken": result.steps_taken,
            "summary": result.summary,
            "tool_calls": result.tool_calls,
            "messages": [_serialize_message(m) for m in result.messages],
            # v2: per-LLM-call usage log for cache-efficiency analysis
            # (see analyze_trace). Empty list for runs that never made an
            # LLM call; absent for v1 traces (analyze_trace handles both).
            "usage_log": list(getattr(result, "usage_log", []) or []),
        }
        path.write_text(
            json.dumps(data, ensure_ascii=False, indent=1, default=str),
            encoding="utf-8",
        )
        _rotate(d)
        return path
    except Exception:
        return None


def _rotate(d: Path) -> None:
    try:
        files = sorted(p for p in d.glob("*.json") if p.is_file())
        for p in files[: max(0, len(files) - _keep())]:
            p.unlink(missing_ok=True)
    except Exception:
        pass


def list_traces(workspace: str | None = None) -> list[Path]:
    d = Path(workspace or ".") / TRACE_SUBDIR
    if not d.is_dir():
        return []
    return sorted(p for p in d.glob("*.json") if p.is_file())


# ---------------------------------------------------------------------------
# Replay
# ---------------------------------------------------------------------------


def replay(trace_path: str | Path) -> dict:
    """Re-run the harness over a recorded trace; report behavioral divergence.

    The recorded AI replies are fed back as the scripted model; tool dispatch
    is stubbed to return each call's recorded result (matched by tool name,
    in order). No model and no real tools run. The report compares the
    replayed tool-call sequence and stop reason against the recording —
    ``match`` is True when the harness made the same decisions it made live.
    """
    import tempfile
    from unittest.mock import patch

    from langchain_core.messages import AIMessage

    from wells import config, executor, tools
    from wells.control import CONTROL
    from wells.tokens import LEDGER
    from wells.tools import ToolContext, ToolResult

    trace = json.loads(Path(trace_path).read_text(encoding="utf-8"))

    ai_script: list[AIMessage] = []
    for m in trace.get("messages", []):
        if m.get("type") != "ai":
            continue
        tcs = [
            {
                "name": tc.get("name"),
                "args": tc.get("args") or {},
                "id": tc.get("id") or f"replay_{i}",
            }
            for i, tc in enumerate(m.get("tool_calls") or [])
        ]
        ai_script.append(AIMessage(content=m.get("content", ""), tool_calls=tcs))
    script_it = iter(ai_script)

    def _scripted_invoke(llm, messages):
        try:
            return next(script_it)
        except StopIteration:
            return AIMessage(content="(replay script exhausted)")

    recorded = list(trace.get("tool_calls") or [])
    pending = list(recorded)

    def _recorded_dispatch(name, args, ctx):
        for i, rc in enumerate(pending):
            if rc.get("name") == name:
                rc = pending.pop(i)
                ok = bool(rc.get("ok", True))
                out = rc.get("output_preview", "") or ""
                return ToolResult(
                    ok, out if ok else "", "" if ok else (out or "recorded failure")
                )
        return ToolResult(True, "(no recorded result for this call)", "")

    LEDGER.reset()
    CONTROL.reset()
    with (
        tempfile.TemporaryDirectory() as td,
        patch.object(config, "_invoke_with_retry", side_effect=_scripted_invoke),
        patch.object(config, "STRUCTURED_OUTPUTS", False),
        patch.object(config, "STREAM_GUARD", False),
        patch.object(config, "ESCALATION_PROFILE", ""),
        # Report native tool support so recorded native tool_calls replay;
        # text-format calls still parse via the fallback path.
        patch.object(executor, "_try_bind_tools", side_effect=lambda llm, ts: llm),
        patch.object(tools, "dispatch", side_effect=_recorded_dispatch),
        patch.dict(os.environ, {"WELLS_TRACE": "0"}),
    ):
        result = executor.run_executor(
            task=trace.get("task", ""),
            ctx=ToolContext(workspace=td, safety="auto"),
            max_steps=0,
            step_label="replay",
        )

    replayed_names = [c.get("name") for c in result.tool_calls]
    recorded_names = [c.get("name") for c in recorded]
    return {
        "trace": str(trace_path),
        "recorded_stopped_reason": trace.get("stopped_reason"),
        "stopped_reason": result.stopped_reason,
        "recorded_steps": trace.get("steps_taken"),
        "steps_taken": result.steps_taken,
        "recorded_calls": recorded_names,
        "calls": replayed_names,
        "match": (
            result.stopped_reason == trace.get("stopped_reason")
            and replayed_names == recorded_names
        ),
    }


# ---------------------------------------------------------------------------
# Cache-efficiency analysis
# ---------------------------------------------------------------------------


def _cache_efficiency(entry: dict) -> float:
    """0..1 — share of the prompt that was a cache hit at the provider."""
    cache_read = entry.get("cache_read", 0) or 0
    inp = entry.get("input", 0) or 0
    denom = inp + cache_read
    if denom <= 0:
        return 0.0
    return cache_read / denom


def analyze_trace(trace_path: str | Path) -> dict:
    """Per-call cache-efficiency report for one trace.

    Returns ``{trace, task, stopped_reason, calls, totals, per_call, notes}``.

    ``per_call`` is the round-by-round breakdown: input/output/cache_read/
    cache_efficiency/context_op for every LLM call. ``totals`` aggregates
    across the run; ``notes`` flags cache-efficiency drops (likely
    cache-break points) — the exact signal the masking-batch fix targets.

    Returns ``calls: 0`` for v1 traces (which predate usage_log) so callers
    can fall back gracefully.
    """
    trace = json.loads(Path(trace_path).read_text(encoding="utf-8"))
    usage = trace.get("usage_log") or []

    per_call = []
    cache_effs: list[float] = []
    for e in usage:
        eff = _cache_efficiency(e)
        cache_effs.append(eff)
        ctx_op = ""
        if e.get("mask_saved"):
            ctx_op += f"mask-{e['mask_saved']}"
        if e.get("drop_saved"):
            ctx_op += ("/" if ctx_op else "") + f"drop-{e['drop_saved']}"
        per_call.append(
            {
                "round": e.get("round", 0),
                "model": e.get("model", ""),
                "input": e.get("input", 0),
                "output": e.get("output", 0),
                "reasoning": e.get("reasoning", 0),
                "cache_read": e.get("cache_read", 0),
                "cache_creation": e.get("cache_creation", 0),
                "cache_efficiency": round(eff, 3),
                "ctx_tokens_at_call": e.get("ctx_tokens_at_call", 0),
                "ctx_op": ctx_op,
            }
        )

    totals = {
        "calls": len(usage),
        "input": sum(e.get("input", 0) for e in usage),
        "output": sum(e.get("output", 0) for e in usage),
        "reasoning": sum(e.get("reasoning", 0) for e in usage),
        "cache_read": sum(e.get("cache_read", 0) for e in usage),
        "cache_creation": sum(e.get("cache_creation", 0) for e in usage),
        "mask_saved": sum(e.get("mask_saved", 0) for e in usage),
        "drop_saved": sum(e.get("drop_saved", 0) for e in usage),
    }
    totals["cache_efficiency"] = round(
        totals["cache_read"] / max(1, totals["input"] + totals["cache_read"]), 3
    )

    # Notes: flag rounds where cache efficiency dropped sharply — likely
    # cache-break points. Heuristic: a 25+ point drop from the running max.
    notes: list[str] = []
    if cache_effs:
        running_max = cache_effs[0]
        for i, eff in enumerate(cache_effs):
            if i == 0:
                continue
            if running_max - eff >= 0.25 and eff < 0.5:
                notes.append(
                    f"round {per_call[i]['round']}: cache efficiency dropped "
                    f"{running_max:.0%} → {eff:.0%}"
                    + (f" ({per_call[i]['ctx_op']})" if per_call[i]["ctx_op"] else "")
                )
            running_max = max(running_max, eff)
    if not usage:
        notes.append("trace has no usage_log (v1 format); upgrade needed")
    elif totals["cache_read"] == 0:
        notes.append("no cache hits recorded — provider may not support prompt caching")

    return {
        "trace": str(trace_path),
        "task": trace.get("task", ""),
        "stopped_reason": trace.get("stopped_reason"),
        "steps_taken": trace.get("steps_taken", 0),
        "calls": len(usage),
        "totals": totals,
        "per_call": per_call,
        "notes": notes,
    }


def analyze_traces(trace_paths: list[str | Path]) -> dict:
    """Aggregate cache-efficiency across many traces.

    Returns ``{traces, calls, totals, worst_breaks}`` — worst_breaks is the
    list of (trace, round, drop) tuples for the largest cache-efficiency
    drops across the whole corpus, useful for ranking where masking layout
    hurts most.
    """
    agg = {
        "input": 0,
        "output": 0,
        "reasoning": 0,
        "cache_read": 0,
        "cache_creation": 0,
        "mask_saved": 0,
        "drop_saved": 0,
        "calls": 0,
    }
    worst_breaks: list[tuple[str, int, float]] = []
    for p in trace_paths:
        try:
            r = analyze_trace(p)
        except Exception:
            continue
        for k in (
            "input",
            "output",
            "reasoning",
            "cache_read",
            "cache_creation",
            "mask_saved",
            "drop_saved",
            "calls",
        ):
            agg[k] += r["totals"].get(k, 0)
        # Track the single worst drop in this trace.
        prev_eff = None
        for c in r["per_call"]:
            eff = c["cache_efficiency"]
            if prev_eff is not None and (prev_eff - eff) >= 0.25 and eff < 0.5:
                worst_breaks.append((r["trace"], c["round"], round(prev_eff - eff, 3)))
            prev_eff = eff

    worst_breaks.sort(key=lambda t: -t[2])
    agg["cache_efficiency"] = round(
        agg["cache_read"] / max(1, agg["input"] + agg["cache_read"]), 3
    )
    return {
        "traces": len(trace_paths),
        "totals": agg,
        "worst_breaks": worst_breaks[:20],
    }


def format_analysis(report: dict) -> str:
    """Render analyze_trace's report as a readable table for the CLI."""
    lines: list[str] = []
    bar = "=" * 92
    lines.append(bar)
    lines.append(f"CACHE-EFFICIENCY REPORT — {report['trace']}")
    lines.append(bar)
    lines.append(f"task: {report['task'][:80]!r}")
    lines.append(
        f"stopped: {report['stopped_reason']}  steps: {report['steps_taken']}  "
        f"calls: {report['calls']}"
    )
    lines.append("")
    t = report["totals"]
    lines.append(
        f"totals: input={t['input']:,}  output={t['output']:,}  "
        f"cache_read={t['cache_read']:,}  ({t['cache_efficiency']:.0%} hit)  "
        f"mask_saved={t['mask_saved']:,}  drop_saved={t['drop_saved']:,}"
    )
    if t["cache_read"]:
        # Rough $ saved at typical Anthropic rates (cache read = 10% of input).
        lines.append(
            f"  ~{t['cache_read'] // 10:,} input-equivalent tokens saved at provider"
        )
    lines.append("")
    if report["per_call"]:
        header = (
            f"{'ROUND':>5} {'IN':>7} {'OUT':>6} {'CACHE':>7} "
            f"{'EFF':>5} {'CTX@CALL':>9} {'CTX_OP':<14} {'MODEL':<24}"
        )
        lines.append(header)
        lines.append("-" * 92)
        for c in report["per_call"]:
            lines.append(
                f"{c['round']:>5} {c['input']:>7,} {c['output']:>6,} "
                f"{c['cache_read']:>7,} {c['cache_efficiency']:>4.0%} "
                f"{c['ctx_tokens_at_call']:>9,} {c['ctx_op'] or '-':<14} "
                f"{c['model'][:24]:<24}"
            )
        lines.append("-" * 92)
    if report["notes"]:
        lines.append("")
        lines.append("notes:")
        for n in report["notes"]:
            lines.append(f"  - {n}")
    lines.append(bar)
    return "\n".join(lines)
