"""Agent step runner: assembles context, invokes the model, accounts for tokens.

Every agent call goes through :func:`run_step`. It:
  * builds the prompt via :class:`ContextManager` (categorized, budget-trimmed),
  * sends a stable ``SystemMessage`` prefix + variable ``HumanMessage`` body
    (prompt-cache friendly),
  * reads the real ``usage_metadata`` from the response (actuals, not estimates),
  * calibrates the estimator and records everything in the global :data:`LEDGER`.

A single failing call never aborts the graph run: we record estimated usage and
return a clearly-marked failure note instead.
"""

from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage

from coding_harness.config import (
    BUDGET,
    _invoke_with_retry,
    get_llm_for_task,
    model_name_for_task,
)
from coding_harness.context import ContextManager
from coding_harness.control import CONTROL
from coding_harness.principles import inject_into_prompt as inject_principles
from coding_harness.tokens import LEDGER, calibrate, estimate_tokens


def run_step(
    *,
    step: str,
    task_type: str,
    system: str,
    chunks: dict,
    budget=None,
    saved_by_summary: int = 0,
    workspace: str | None = None,
) -> tuple[str, object]:
    """Run one agent step and return ``(response_text, ContextReport)``.

    The harness operating principles (AGENT.md) are always prepended to the
    system prompt, so every agent — regardless of which model is configured —
    is governed by the same behavioral constitution.
    """
    # Abort between agent steps when the user cancelled the run (TUI Escape).
    CONTROL.checkpoint()
    CONTROL.set_activity(f"{step} · thinking")

    cm = ContextManager(budget or BUDGET)
    body, report = cm.build_context(chunks)
    system = inject_principles(system, workspace)

    messages = [SystemMessage(content=system), HumanMessage(content=body)]
    llm = get_llm_for_task(task_type)
    model = model_name_for_task(task_type)
    full_text = f"{system}\n{body}"

    try:
        resp = _invoke_with_retry(llm, messages)
        content = (resp.content or "").strip()
        um = getattr(resp, "usage_metadata", None) or {}
        input_tokens = um.get("input_tokens") or estimate_tokens(full_text)
        output_tokens = um.get("output_tokens") or 0
        reasoning_tokens = ((um.get("output_token_details") or {}).get("reasoning")) or 0
        cache_read_tokens = ((um.get("input_token_details") or {}).get("cache_read")) or 0
        calibrate(full_text, input_tokens)
    except Exception as err:  # transient exhaustion / non-transient -> degrade gracefully
        content = f"[{step} LLM call failed: {type(err).__name__}: {str(err)[:160]}]"
        input_tokens = estimate_tokens(full_text)
        output_tokens = reasoning_tokens = cache_read_tokens = 0
        print(f"[{step}] {content}")

    LEDGER.record(
        step=step,
        task_type=task_type,
        model=model,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        reasoning_tokens=reasoning_tokens,
        cache_read_tokens=cache_read_tokens,
        category_tokens=report.category_tokens,
        saved_by_trim=report.saved_by_trim,
        saved_by_summary=saved_by_summary,
    )
    return content, report
