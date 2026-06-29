"""Typed LangGraph state shared by every agent node."""

from typing import TypedDict


class AgentState(TypedDict, total=False):
    """Typed LangGraph state shared by every agent node.

    Although declared ``total=False`` (so any subset is constructible), every
    run seeds ``goal`` — it is the one field every node reads. Treating it as
    effectively-required keeps call sites readable without forcing callers to
    populate every optional field.
    """

    goal: str
    iteration: int
    max_iterations: int

    # Workspace / execution context (Layer 1/2).
    workspace_root: str
    safety: str  # auto | approve | dryrun
    plan_mode: bool
    index_ready: bool  # True when repo index is built/up-to-date

    development_plan: str
    architecture: str
    implementation_steps: str
    code_changes: str
    test_plan: str
    test_results: str
    review_result: str
    review_complete: bool

    # Per-node executor message history (lets the coder resume across iterations).
    executor_messages: list

    # Token optimization: rolling task-state summary used on loop iterations.
    task_summary: str

    summary: str
    messages: list[str]

    # Finisher outputs (post-run git/memory write-back).
    finalized: bool
    git_summary: str
    pr_url: str
    memory_written: str
