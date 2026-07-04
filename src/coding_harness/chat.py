"""Intent classification and conversation memory for Wells.

Two modes only:
  - ``"auto"``        — default; direct executor handles questions AND tasks.
  - ``"orchestrate"`` — full planner→coder→tester→reviewer graph for complex
                        multi-component work.

The classifier is intentionally biased toward ``"auto"``: the executor can do
everything auto can plus answer questions via tool-augmented LLM. Orchestrate is
reserved for requests where a planning pass genuinely adds value.
"""

from __future__ import annotations

import re
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Literal

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from coding_harness.config import (
    _invoke_with_retry,
    get_llm_for_task,
)

Intent = Literal["auto", "orchestrate"]


# ---------------------------------------------------------------------------
# Heuristic classification — free, no model call
# ---------------------------------------------------------------------------

# Verbs that, combined with an architectural scope noun, signal orchestrate.
_ORCHESTRATE_VERBS = {
    "build", "implement", "create", "design", "architect",
    "set up", "setup", "rewrite", "develop", "scaffold",
}

# Scope nouns that represent multi-component systems.
_ORCHESTRATE_SCOPE = {
    "system", "service", "api", "framework", "platform",
    "authentication", "auth", "database", "pipeline",
    "microservice", "backend", "frontend", "oauth", "jwt",
    "workflow", "architecture",
}


# Ops/deploy verbs: these are execution tasks, not design tasks. A planning
# pass adds minutes and zero value — the direct executor handles them.
_OPS_VERBS = {
    "push", "deploy", "publish", "upload", "release", "restart", "install",
    "commit", "merge", "sync", "run", "start", "stop", "ship",
}


def _heuristic_classify(text: str) -> Intent | None:
    """Return ``"orchestrate"`` for clearly complex work, ``None`` when ambiguous.

    Everything that isn't obviously orchestrate-level returns None so the LLM
    fallback can decide — or just falls through to ``"auto"`` (the safe default).
    """
    if not text.strip():
        return "auto"

    lower = " " + text.strip().lower() + " "

    # Ops tasks route straight to the executor, even when they mention
    # architectural nouns ("push to the azure app SERVICE" is not a design
    # task). Check the first meaningful word.
    words = [w for w in text.strip().lower().split() if w not in ("please", "can", "you", "now")]
    if words and words[0] in _OPS_VERBS:
        return "auto"

    # Orchestrate: big-creation verb AND architectural scope noun.
    has_orch_verb = any(
        f" {v} " in lower or lower.lstrip().startswith(v + " ")
        for v in _ORCHESTRATE_VERBS
    )
    has_orch_scope = any(
        f" {n} " in lower or lower.rstrip().endswith(f" {n}")
        for n in _ORCHESTRATE_SCOPE
    )
    if has_orch_verb and has_orch_scope:
        return "orchestrate"

    return None  # let LLM or caller decide


# ---------------------------------------------------------------------------
# LLM-based classification — fallback for ambiguous inputs
# ---------------------------------------------------------------------------

_CLASSIFIER_SYSTEM = (
    "Reply with exactly one word: AUTO or ORCHESTRATE.\n\n"
    "AUTO — the default; handles questions, edits, deployments, running commands,\n"
    "reading files, or any task completable in a handful of steps:\n"
    "  'what does X do?', 'change the height in style.css', 'deploy to Azure',\n"
    "  'read publishing.md and push', 'fix the login bug', 'add error handling'\n\n"
    "ORCHESTRATE — only for complex multi-component work needing a planning pass:\n"
    "  'build a login system with OAuth2, JWT, and database schema'\n"
    "  'refactor the entire auth module across all files'\n"
    "  'implement a complete REST API with authentication and tests'\n\n"
    "When in doubt: AUTO. Orchestrate only when scope is clearly architectural."
)

_CLASSIFIER_CACHE: dict[str, Intent] = {}


def _llm_classify(text: str) -> Intent:
    """Classify via cheap model. Cached; defaults to ``"auto"`` on any error."""
    key = text.strip().lower()[:200]
    if key in _CLASSIFIER_CACHE:
        return _CLASSIFIER_CACHE[key]
    try:
        llm = get_llm_for_task("classification")
        resp = _invoke_with_retry(
            llm,
            [SystemMessage(content=_CLASSIFIER_SYSTEM), HumanMessage(content=text)],
        )
        answer = (resp.content or "").strip().upper()
        intent: Intent = "orchestrate" if answer.startswith("ORCHESTRATE") else "auto"
    except Exception:
        intent = "auto"  # conservative: auto can do everything
    _CLASSIFIER_CACHE[key] = intent
    return intent


def classify_intent(text: str, *, use_llm_fallback: bool = True) -> Intent:
    """Classify ``text`` as ``"auto"`` or ``"orchestrate"``.

    Heuristic handles the obvious orchestrate cases for free. Ambiguous inputs
    fall back to a cheap model call (unless ``use_llm_fallback=False``, in
    which case ambiguous → ``"auto"``).
    """
    direct = _heuristic_classify(text)
    if direct is not None:
        return direct
    if use_llm_fallback:
        return _llm_classify(text)
    return "auto"


def clear_classifier_cache() -> None:
    _CLASSIFIER_CACHE.clear()


# ---------------------------------------------------------------------------
# Conversation memory
# ---------------------------------------------------------------------------


@dataclass
class ConversationMemory:
    """Lightweight context store: last_run_summary feeds context into auto runs.

    ``last_run_summary`` is updated after each auto or orchestrate run so that
    follow-up questions ("did it work?", "what did you change?") have context
    without re-running anything.
    """

    max_turns: int = 12
    turns: Deque[tuple[str, str]] = field(default_factory=lambda: deque(maxlen=12))
    last_run_summary: str = ""

    def __post_init__(self) -> None:
        self.turns = deque(self.turns, maxlen=self.max_turns)

    def add(self, role: str, content: str) -> None:
        self.turns.append((role, content))

    def clear(self) -> None:
        self.turns.clear()
        self.last_run_summary = ""

    def set_run_summary(self, summary: str) -> None:
        self.last_run_summary = (summary or "").strip()

    def as_messages(self, system: str) -> list:
        """Build a LangChain message list: system + run-summary + history."""
        msgs: list = [SystemMessage(content=system)]
        if self.last_run_summary:
            msgs.append(
                SystemMessage(
                    content=(
                        "Context — the most recent action in this session:\n"
                        + self.last_run_summary
                    )
                )
            )
        for role, content in self.turns:
            if role == "user":
                msgs.append(HumanMessage(content=content))
            else:
                msgs.append(AIMessage(content=content))
        return msgs
