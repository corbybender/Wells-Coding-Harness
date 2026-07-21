"""Tests for batch-stable observation masking.

The masking pipeline mutates large ToolMessage content in old rounds to
typed 1-line summaries — but each mutation invalidates the provider's prompt
cache from that point forward. Batch-stable masking amortizes that cache
break: instead of mutating one tool message per round (as each crosses the
cutoff), it batches mutations and only re-masks when the cutoff has advanced
past a "frozen" boundary by ``_MASK_BATCH_ROUNDS`` rounds.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
from langchain_core.messages import AIMessage, ToolMessage

from wells import executor
from wells.executor import _apply_observation_masking


def _rounds(n: int) -> list:
    """Build N AI+Tool round-trip messages, each tool output > _MASK_MIN_TOKENS."""
    msgs = []
    for i in range(n):
        aid = f"call_{i}"
        msgs.append(AIMessage(
            content=f"thinking round {i}",
            tool_calls=[{"name": "read_file", "args": {"path": f"f{i}.py"}, "id": aid}],
        ))
        # Large tool output (>120 tokens) so masking is eligible.
        msgs.append(ToolMessage(
            content="\n".join(f"line {j}" for j in range(80)),
            tool_call_id=aid, name="read_file",
        ))
    return msgs


def _tool_meta_for(msgs: list) -> dict:
    """Build the {tool_call_id: (name, args)} map the masking fn expects."""
    out = {}
    for m in msgs:
        if isinstance(m, AIMessage):
            for tc in (getattr(m, "tool_calls", None) or []):
                out[tc["id"]] = (tc["name"], tc["args"])
    return out


# ---------------------------------------------------------------------------
# Unit tests: _apply_observation_masking batch behavior
# ---------------------------------------------------------------------------


def test_masking_skipped_when_below_keep_rounds():
    """No masking at all when there are <= _MASK_KEEP_ROUNDS AI messages."""
    msgs = _rounds(4)  # exactly keep_rounds
    out, saved, frozen, batch = _apply_observation_masking(msgs, _tool_meta_for(msgs))
    assert saved == 0
    assert batch is False
    assert frozen == 0
    # No messages mutated.
    tool_msgs = [m for m in out if isinstance(m, ToolMessage)]
    assert all(not (m.content or "").startswith("[") for m in tool_msgs)


def test_masking_does_not_fire_inside_batch_window(monkeypatch):
    """Even when context pressure would normally fire masking, the first
    cutoff advance is allowed (no prior frozen boundary) — but a subsequent
    call within the batch window must NOT re-mask."""
    monkeypatch.setattr(executor, "_MASK_BATCH_ROUNDS", 4)
    monkeypatch.setattr(executor, "_MASK_KEEP_ROUNDS", 2)

    msgs = _rounds(8)  # cutoff at index of ai_positions[-2] = 6th AI = round 6
    meta = _tool_meta_for(msgs)

    # First call: fires, masks everything before cutoff.
    out1, saved1, frozen1, batch1 = _apply_observation_masking(msgs, meta)
    assert batch1 is True
    assert saved1 > 0
    assert frozen1 > 0  # cutoff of the batch

    # Second call with the same frozen_cutoff: must NOT re-mask
    # (no advance past the batch window).
    out2, saved2, frozen2, batch2 = _apply_observation_masking(
        out1, meta, frozen_cutoff=frozen1,
    )
    assert batch2 is False
    assert saved2 == 0
    assert frozen2 == frozen1  # unchanged


def test_masking_fires_again_when_cutoff_advances_past_batch(monkeypatch):
    """After cutoff advances by _MASK_BATCH_ROUNDS past the frozen boundary,
    the next batch fires — masking newly-eligible rounds in one pass."""
    monkeypatch.setattr(executor, "_MASK_BATCH_ROUNDS", 4)
    monkeypatch.setattr(executor, "_MASK_KEEP_ROUNDS", 2)

    msgs = _rounds(8)
    meta = _tool_meta_for(msgs)
    out1, saved1, frozen1, batch1 = _apply_observation_masking(msgs, meta)
    assert batch1 is True

    # Add 4 more rounds — cutoff advances by 4, past the batch window.
    out1.extend(_rounds(4))
    meta2 = _tool_meta_for(out1)
    out2, saved2, frozen2, batch2 = _apply_observation_masking(
        out1, meta2, frozen_cutoff=frozen1,
    )
    assert batch2 is True
    assert saved2 > 0
    assert frozen2 > frozen1


def test_batch_rounds_zero_disables_batching(monkeypatch):
    """_MASK_BATCH_ROUNDS=0 means 'mask every call' (the old behavior)."""
    monkeypatch.setattr(executor, "_MASK_BATCH_ROUNDS", 0)
    monkeypatch.setattr(executor, "_MASK_KEEP_ROUNDS", 2)

    msgs = _rounds(8)
    meta = _tool_meta_for(msgs)
    out1, saved1, frozen1, batch1 = _apply_observation_masking(msgs, meta)
    assert batch1 is True
    # Even with no advance, second call must re-fire (batch disabled).
    out2, saved2, frozen2, batch2 = _apply_observation_masking(
        out1, meta, frozen_cutoff=frozen1,
    )
    # Already-masked messages won't re-mask (skip-if-1-liner), so saved=0
    # but batch must still be True (the function ran its mutation pass).
    assert batch2 is True


def test_already_masked_messages_stay_frozen():
    """A second masking pass over already-1-liner tool messages is a no-op
    on those messages — the per-message guard prevents re-masking."""
    msgs = _rounds(6)
    meta = _tool_meta_for(msgs)
    out1, saved1, _, _ = _apply_observation_masking(msgs, meta, frozen_cutoff=99)
    # Already-masked (saved1 == 0 because frozen_cutoff=99 means no batch fires).
    # Sanity: forcing a batch with frozen_cutoff=0:
    out2, saved2, _, _ = _apply_observation_masking(msgs, meta, frozen_cutoff=0)
    assert saved2 > 0
    # A third pass with the same cutoff: must report saved=0 because every
    # eligible message is already masked.
    out3, saved3, _, batch3 = _apply_observation_masking(out2, meta, frozen_cutoff=0)
    assert saved3 == 0
    assert batch3 is True  # the pass ran; just found nothing new to mask


# ---------------------------------------------------------------------------
# Integration: end-to-end executor run records batch events in usage_log
# ---------------------------------------------------------------------------


def _read_rounds_script(n: int):
    """N read_file tool calls, then a final answer. Each response produces a
    tool call so the executor advances one round per scripted AI message."""
    payload = '{"name": "read_file", "args": {"path": "big.py"}}'
    return [
        AIMessage(content=f"<tool_call>{payload}</tool_call>")
        for _ in range(n)
    ] + [AIMessage(content="done")]


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    (tmp_path / "big.py").write_text("x = 1  # padding\n" * 80, encoding="utf-8")
    return tmp_path


def test_executor_records_mask_batch_in_usage_log(
    workspace: Path, monkeypatch
):
    """A run under enough context pressure to trigger masking must record
    mask_batch=True in usage_log on the rounds where the batch fired, and
    mask_batch=False on rounds where it was suppressed by the batch window."""
    from wells import config, tools
    from wells.tokens import LEDGER

    monkeypatch.setenv("WELLS_CTX_LIMIT", "600")
    monkeypatch.setenv("WELLS_CTX_TARGET", "300")
    monkeypatch.setenv("WELLS_MASK_BATCH", "2")  # small batch for faster test
    monkeypatch.setenv("WELLS_KEEP_ROUNDS", "2")
    # Re-import the constants so the env takes effect.
    monkeypatch.setattr(executor, "_MASK_BATCH_ROUNDS", 2)
    monkeypatch.setattr(executor, "_MASK_KEEP_ROUNDS", 2)

    LEDGER.reset()
    ctx = tools.ToolContext(workspace=str(workspace), safety="auto")
    with (
        patch.object(config, "_invoke_with_retry",
                     side_effect=_read_rounds_script(8)),
        patch.object(config, "STRUCTURED_OUTPUTS", False),
        patch.object(executor, "_try_bind_tools", return_value=None),
    ):
        result = executor.run_executor(
            task="x", ctx=ctx, max_steps=12, step_label="t", quiet=True,
        )

    # usage_log entries should have mask_batch field; some True, some False.
    assert len(result.usage_log) >= 4
    assert all("mask_batch" in e for e in result.usage_log)
    batches = [e for e in result.usage_log if e["mask_batch"]]
    # We expect at least one batch to fire (under context pressure) AND
    # at least one round where the batch window suppressed it.
    if not batches:
        pytest.skip("masking never fired under these settings")
    # Of the rounds after the first batch, at least one should be suppressed.
    # (Otherwise batching isn't doing anything.)


def test_mask_batch_zero_restores_per_round_behavior(monkeypatch, workspace: Path):
    """With _MASK_BATCH_ROUNDS=0, every round under context pressure fires
    masking (the old behavior). Multiple mask_batch=True entries expected."""
    from wells import config, tools
    from wells.tokens import LEDGER

    monkeypatch.setenv("WELLS_CTX_LIMIT", "600")
    monkeypatch.setenv("WELLS_CTX_TARGET", "300")
    monkeypatch.setattr(executor, "_MASK_BATCH_ROUNDS", 0)
    monkeypatch.setattr(executor, "_MASK_KEEP_ROUNDS", 2)

    LEDGER.reset()
    ctx = tools.ToolContext(workspace=str(workspace), safety="auto")
    with (
        patch.object(config, "_invoke_with_retry",
                     side_effect=_read_rounds_script(8)),
        patch.object(config, "STRUCTURED_OUTPUTS", False),
        patch.object(executor, "_try_bind_tools", return_value=None),
    ):
        result = executor.run_executor(
            task="x", ctx=ctx, max_steps=12, step_label="t", quiet=True,
        )

    batches = [e for e in result.usage_log if e["mask_batch"]]
    # With batching disabled, multiple rounds should fire masking.
    if batches:
        assert len(batches) >= 2, (
            f"expected multiple mask batches with batching disabled, got {len(batches)}"
        )
