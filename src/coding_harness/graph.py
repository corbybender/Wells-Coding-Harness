"""Wires the planner -> architect -> coder -> tester -> reviewer workflow.

Loop rule: when the reviewer is not satisfied, control runs the ``summarizer``
(condenses durable context for cheaper re-use) and then returns to ``coder``.
The loop is bounded by ``max_iterations`` (default 3) to avoid runaway runs.

After the loop ends (COMPLETE, or the cap was hit), a ``finisher`` node runs to
write the project-memory lesson and (optionally) create a git branch/commit/PR.
"""

from langgraph.graph import END, START, StateGraph

from coding_harness.agents.architect import architect
from coding_harness.agents.coder import coder
from coding_harness.agents.planner import planner
from coding_harness.agents.reviewer import reviewer
from coding_harness.agents.tester import tester
from coding_harness.config import MAX_ITERATIONS, INDEX_AUTO_UPDATE
from coding_harness.finisher import finisher
from coding_harness.state import AgentState
from coding_harness.summarize import summarizer_node
from coding_harness.tools import ToolContext


def indexer_node(state: AgentState) -> AgentState:
    """Build or update the structural repository index (if available).

    Runs transparently before planning. Sets index_ready=True when complete.
    If wells-index is not installed or INDEX_AUTO_UPDATE is disabled, this is a no-op.
    """
    # Import here to avoid circular dependency and late-bind the availability check
    from coding_harness import index_tools

    if not INDEX_AUTO_UPDATE or not index_tools.INDEXER_AVAILABLE:
        return {"index_ready": False}

    try:
        ctx = ToolContext.from_state(state)
        result = index_tools.index_workspace(ctx)
        return {"index_ready": result.ok}
    except Exception:
        # If indexing fails, continue anyway (graceful degradation)
        return {"index_ready": False}


def _route_after_review(state: AgentState) -> str:
    """Conditional edge after the reviewer: loop or finalize."""
    if state.get("review_complete"):
        return "finalize"

    iteration = state.get("iteration", 0)
    cap = state.get("max_iterations", MAX_ITERATIONS)
    if iteration >= cap:
        print(f"[graph] reached max iterations ({cap}); finalizing.")
        return "finalize"

    print(f"[graph] iteration {iteration} incomplete -> summarizer -> coder.")
    return "loop"


def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("indexer", indexer_node)
    graph.add_node("planner", planner)
    graph.add_node("architect", architect)
    graph.add_node("coder", coder)
    graph.add_node("tester", tester)
    graph.add_node("reviewer", reviewer)
    graph.add_node("summarizer", summarizer_node)
    graph.add_node("finisher", finisher)

    graph.add_edge(START, "indexer")
    graph.add_edge("indexer", "planner")
    graph.add_edge("planner", "architect")
    graph.add_edge("architect", "coder")
    graph.add_edge("coder", "tester")
    graph.add_edge("tester", "reviewer")
    # On INCOMPLETE: condense context, then iterate.
    graph.add_conditional_edges(
        "reviewer",
        _route_after_review,
        {"finalize": "finisher", "loop": "summarizer"},
    )
    graph.add_edge("summarizer", "coder")
    graph.add_edge("finisher", END)

    return graph.compile()
