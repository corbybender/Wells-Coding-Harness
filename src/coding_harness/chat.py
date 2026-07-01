"""Conversational mode: routes plain chat past the agentic loop.

The Wells REPL sends every message through the planner->architect->coder->
tester->reviewer graph by default. That is the right thing for a real
development task ("add a login page", "fix the bug in parser.py"), but it is
absurdly expensive for a quick question ("did you actually make that change?",
"what does this file do?", "explain your last run").

This module provides:

* :func:`classify_intent` — decides whether a message is a conversational
  question (``"chat"``) or a real development task (``"task"``). A fast
  heuristic layer handles the obvious cases for free; ambiguous inputs fall
  back to a tiny classifier call against the cheap model.

* :func:`conversational_reply` — streams a direct LLM reply to a question,
  with full conversation history + a summary of the most recent agent run so
  follow-up questions like "did it work?" have context.

* :class:`ConversationMemory` — a bounded, role-tagged history of the chat so
  the conversation is coherent across turns.

The router is intentionally conservative: when in doubt, it routes to the
agentic loop. A wrong "chat" decision wastes the user's time; a wrong "task"
decision only costs a little extra compute.
"""

from __future__ import annotations

import re
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Literal

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from coding_harness.config import (
    LLM_BACKOFF_BASE,
    LLM_MAX_RETRIES,
    _invoke_with_retry,
    _is_transient,
    get_llm_for_task,
    model_name_for_task,
)
from coding_harness.tokens import LEDGER, calibrate, estimate_tokens

Intent = Literal["chat", "simple", "task"]


# ---------------------------------------------------------------------------
# Heuristic intent classification (free, instant, no model call)
# ---------------------------------------------------------------------------

# Strongly conversational openings. A message starting with one of these is
# almost always a question aimed at the assistant, not a build instruction.
_QUESTION_STARTERS = (
    "what ",
    "why ",
    "how ",
    "when ",
    "where ",
    "who ",
    "did you",
    "do you",
    "are you",
    "is the",
    "is it",
    "is there",
    "can you",
    "could you",
    "would you",
    "will you",
    "should i",
    "explain",
    "tell me",
    "describe",
    "summarize",
    "what's",
    "whats",
    "how's",
    "hows",
    "where's",
    "show me",
    "help me understand",
    "what did",
    "what does",
    "what is",
    "what was",
    "what were",
    "which ",
    "whose ",
    "whom ",
)

# Greetings / acknowledgements / filler that is never a dev task.
_CHITCHAT = (
    "hi",
    "hello",
    "hey",
    "yo",
    "sup",
    "howdy",
    "thanks",
    "thank you",
    "thx",
    "ty",
    "cool",
    "nice",
    "got it",
    "ok",
    "okay",
    "k",
    "lol",
    "haha",
    "sure",
    "right",
    "yes",
    "no",
    "agreed",
    "makes sense",
    "understood",
    "sounds good",
)

# ---------------------------------------------------------------------------
# Simple-task signals — scoped atomic edits that don't need orchestration
# ---------------------------------------------------------------------------

# Atomic "point-fix" verbs: imply a small, located change to something known.
_SIMPLE_VERBS = (
    "change", "update", "set", "fix", "rename", "delete", "remove",
    "replace", "edit", "correct", "adjust", "tweak", "bump", "increase",
    "decrease", "toggle", "enable", "disable", "comment out", "uncomment",
    "add a line", "add one", "remove the", "delete the", "change the",
    "update the", "set the", "fix the", "rename the",
)

# File-like targets: a specific file or element is named, scoping the task.
_FILE_EXTENSION_RE = re.compile(
    r"\b\w+\.(html?|css|js|ts|tsx|jsx|py|rs|go|java|json|ya?ml|toml|md|txt|sh|env)\b",
    re.IGNORECASE,
)
_SPECIFIC_TARGET_RE = re.compile(
    r"\b(the\s+\w+\s+(div|class|function|method|variable|field|property|line|"
    r"file|component|element|style|attribute|tag|button|header|footer|"
    r"section|column|row|width|height|color|font|margin|padding|border|"
    r"value|key|string|text|label|name|path|url|import|export))\b",
    re.IGNORECASE,
)

# These words in the request push it toward full orchestration, not simple.
_COMPLEX_BLOCKERS = (
    "build", "create", "implement", "set up", "design", "architect",
    "add a", "add the", "new ", "entire", "all ", "across", "system",
    "service", "api", "auth", "database", "deploy", "refactor",
    "restructure", "migrate",
)


def _is_simple_task(text: str) -> bool:
    """Return True when the request looks like a scoped, single-file atomic edit."""
    lower = text.strip().lower()
    words = lower.split()
    if len(words) > 30:
        return False  # long requests are rarely simple

    # Any complex-blocker present → not simple
    if any(b in lower for b in _COMPLEX_BLOCKERS):
        return False

    has_simple_verb = any(lower.startswith(v) or f" {v} " in f" {lower} " for v in _SIMPLE_VERBS)
    has_file_target = bool(_FILE_EXTENSION_RE.search(text))
    has_specific_target = bool(_SPECIFIC_TARGET_RE.search(text))

    return has_simple_verb and (has_file_target or has_specific_target)


# ---------------------------------------------------------------------------
# Verbs that signal a request to *change the codebase* — a real task.
# If any appear, lean toward "task" even when the message is phrased as a
# question ("can you fix the login bug?").
_TASK_SIGNALS = (
    "implement",
    "create",
    "build",
    "add",
    "write",
    "generate",
    "make",
    "fix",
    "repair",
    "patch",
    "resolve",
    "debug",
    "refactor",
    "rename",
    "reorganize",
    "restructure",
    "move",
    "delete",
    "remove",
    "drop",
    "clean up",
    "cleanup",
    "update",
    "upgrade",
    "modify",
    "change",
    "edit",
    "replace",
    "install",
    "deploy",
    "migrate",
    "port",
    "convert",
    "optimize",
    "speed up",
    "improve",
    "enhance",
    "set up",
    "setup",
    "configure",
    # Verbs commonly missed that imply action on files/systems:
    "send",
    "push",
    "publish",
    "upload",
    "submit",
    "run",
    "execute",
    "launch",
    "start",
    "stop",
    "restart",
    "open",
    "read and",   # "read X and do Y" is always a task
    "check and",  # "check X and do Y"
)

# Reference to prior work — strong signal this is a follow-up question, not a
# brand-new task.
_FOLLOWUP_SIGNALS = (
    "previous",
    "earlier",
    "last",
    "just",
    "your output",
    "your run",
    "the fix",
    "the change",
    "the patch",
    "the result",
    "the summary",
    "did you",
    "have you",
    "you said",
    "you mentioned",
    "you wrote",
    "that ",
    "this ",
    "it ",
    "above",
    "below",
    # Past-tense / present-perfect: user describing what they already did.
    "i used",
    "i ran",
    "i've",
    "i have",
    "i did",
    "i created",
    "i made",
    "i built",
    "i added",
    "i removed",
    "i updated",
    "i set",
    "i ran",
    "have been",
    "has been",
)

# Patterns that make a "task" word actually a question ("how do i fix X" is a
# question, not an instruction to fix X).
_QUESTION_HINTS_RE = re.compile(
    r"\b(how (do|does|to|can|could)|why (do|does|is|are|was|were|can't|cannot)"
    r"|what (is|are|do|does|was|were|does the)|where (is|are|do i|does)"
    r"|can (i|you) (use|run|call|do|see|check|tell|verify|confirm|show|access|read|view|find|know|detect|access)"
    r"|is (it|there|this|that) (a |an |the )?"
    r"|explain |tell me |describe )\b",
    re.IGNORECASE,
)

# Word-boundary regex for task signals — prevents "fix" matching "fixed",
# "create" matching "created", "add" matching "added", etc.
_TASK_SIGNAL_RE = re.compile(
    r"\b(?:" + "|".join(re.escape(s) for s in _TASK_SIGNALS) + r")\b",
    re.IGNORECASE,
)


def _word_count(text: str) -> int:
    return len(text.split())


def _heuristic_classify(text: str) -> Intent | None:
    """Return ``"chat"``/``"task"`` for obvious cases, ``None`` if ambiguous."""
    stripped = text.strip()
    if not stripped:
        return "chat"
    lower = " " + stripped.lower() + " "
    words = _word_count(stripped)

    # Very short chitchat / greetings → chat.
    if words <= 4:
        bare = stripped.lower().strip("?!.,")
        first = bare.split(None, 1)[0] if bare else ""
        if bare in _CHITCHAT or first in _CHITCHAT:
            return "chat"

    has_task_signal = bool(_TASK_SIGNAL_RE.search(lower))
    has_followup = any(sig in lower for sig in _FOLLOWUP_SIGNALS)
    has_question_mark = "?" in stripped
    starts_with_question = lower[1:].lstrip().startswith(_QUESTION_STARTERS)
    looks_like_question = bool(_QUESTION_HINTS_RE.search(stripped))

    # A question about prior work ("did you make the fix?") → chat.
    # Exception: "can you please send/push/deploy this" has a followup word ("this")
    # but is clearly a task — let the task-signal check win in that case.
    if has_followup and (
        has_question_mark or starts_with_question or looks_like_question
    ):
        if not (has_task_signal and _imperative_after_please(stripped)):
            return "chat"

    # "how do I fix X" / "why does X fail" — asking, not instructing.
    if looks_like_question and not _imperative(stripped):
        return "chat"

    # Pure question with no task verb → chat.
    if starts_with_question and not has_task_signal:
        return "chat"

    # Imperative task verb present — check if it's a simple scoped edit first.
    if has_task_signal and not starts_with_question and not looks_like_question:
        if _is_simple_task(stripped):
            return "simple"
        return "task"

    # Task verb phrased as a question ("can you fix the bug?") → check scope.
    if (
        has_task_signal
        and (starts_with_question or has_question_mark)
        and not has_followup
    ):
        if _imperative_after_please(stripped):
            if _is_simple_task(stripped):
                return "simple"
            return "task"

    return None  # ambiguous


def _imperative(text: str) -> bool:
    """True if the sentence reads as a command (starts with a task verb)."""
    first = text.strip().split(None, 1)[0].lower().rstrip(",.:!")
    return first in _TASK_SIGNALS


_POLITE_PREFIXES = (
    "can you please ", "can you ", "could you please ", "could you ",
    "would you please ", "would you ", "will you ", "please ",
)


def _imperative_after_please(text: str) -> bool:
    """True when a polite task request like 'can you [please] X' contains a task verb.

    Checks both the immediate first verb AND any task verb anywhere in the
    message, because requests like 'can you please read X and send Y' have
    the task verb ("send") after a non-task opener ("read").
    """
    lower = text.lower().lstrip()
    for lead in _POLITE_PREFIXES:
        if lower.startswith(lead):
            rest = lower[len(lead):].lstrip()
            if rest.startswith("please "):
                rest = rest[7:].lstrip()
            # Check first verb directly after the prefix.
            first = rest.split(None, 1)[0].rstrip(",.:!") if rest else ""
            if first in _TASK_SIGNALS:
                return True
            # Also check anywhere in the full message — handles "can you read X
            # and send Y" where the task verb comes after a non-task opener.
            if _TASK_SIGNAL_RE.search(lower):
                return True
    return False


# ---------------------------------------------------------------------------
# LLM-based intent classification (fallback for ambiguous inputs)
# ---------------------------------------------------------------------------

_CLASSIFIER_SYSTEM = (
    "You are a fast intent router for a coding assistant REPL. Classify the "
    "user's message into exactly one of three categories.\n\n"
    "Reply with exactly one word: CHAT, SIMPLE, or TASK.\n\n"
    "CHAT — conversational; answer directly, no file edits needed:\n"
    "  'explain what you did', 'what does X do?', 'did the fix work?'\n\n"
    "SIMPLE — a scoped, atomic file edit with a clear target; no planning needed:\n"
    "  'change the div height to 200px in index.html'\n"
    "  'rename the function foo to bar in utils.py'\n"
    "  'update the color in main.css to #ff0000'\n"
    "  'fix the typo on line 42 of config.py'\n\n"
    "TASK — new feature, multi-file change, unclear scope, or architecture work:\n"
    "  'add a login page', 'build a REST API', 'refactor the auth system'\n"
    "  'fix the bug' (no file specified), 'implement dark mode'\n\n"
    "Rules:\n"
    "- If a specific file AND a specific small change are both named → SIMPLE\n"
    "- If it requires planning, multiple files, or creating new things → TASK\n"
    "- Questions or explanations → CHAT\n"
    "- When unsure between SIMPLE and TASK, prefer TASK.\n"
)

_CLASSIFIER_CACHE: dict[str, Intent] = {}


def _llm_classify(text: str) -> Intent:
    """Classify via the cheap model. Cached + fault-tolerant (defaults to task)."""
    key = text.strip().lower()[:200]
    if key in _CLASSIFIER_CACHE:
        return _CLASSIFIER_CACHE[key]
    try:
        llm = get_llm_for_task("classification")
        resp = _invoke_with_retry(
            llm, [SystemMessage(content=_CLASSIFIER_SYSTEM), HumanMessage(content=text)]
        )
        answer = (resp.content or "").strip().upper()
        if answer.startswith("CHAT"):
            intent: Intent = "chat"
        elif answer.startswith("SIMPLE"):
            intent = "simple"
        else:
            intent = "task"
    except Exception:
        intent = "task"  # conservative fallback
    _CLASSIFIER_CACHE[key] = intent
    return intent


def classify_intent(text: str, *, use_llm_fallback: bool = True) -> Intent:
    """Decide whether ``text`` is a conversational question or a dev task.

    A fast heuristic handles the obvious cases for free. Ambiguous inputs fall
    back to a one-word classifier call against the cheap model (unless
    ``use_llm_fallback`` is False, in which case ambiguous -> task).
    """
    direct = _heuristic_classify(text)
    if direct is not None:
        return direct
    if use_llm_fallback:
        return _llm_classify(text)
    return "task"


def clear_classifier_cache() -> None:
    _CLASSIFIER_CACHE.clear()


# ---------------------------------------------------------------------------
# Conversation memory
# ---------------------------------------------------------------------------


@dataclass
class ConversationMemory:
    """Bounded chat history + the most recent agent-run summary.

    ``last_run_summary`` is set by the REPL after each agentic run so that
    follow-up questions ("did it work?", "what did you change?") have context
    without re-running the whole graph.
    """

    max_turns: int = 12
    turns: Deque[tuple[str, str]] = field(default_factory=lambda: deque(maxlen=12))
    last_run_summary: str = ""

    def __post_init__(self) -> None:
        # Respect max_turns if passed to the constructor.
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
                        "Context — the most recent agent run in this session:\n"
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


# ---------------------------------------------------------------------------
# Conversational reply
# ---------------------------------------------------------------------------

# Marker the chat LLM outputs when it recognises the request needs tool use.
# The calling code detects this and auto-escalates to task mode.
ESCALATE_MARKER = "<<NEEDS_TASK>>"

_CHAT_SYSTEM_BASE = (
    "You are Wells, a concise, helpful coding assistant chatting with the user "
    "in an interactive REPL. Answer the user's question directly and briefly. "
    "You are NOT in agentic mode — you cannot run tools, read files, or edit code "
    "in this turn.\n\n"
    "IMPORTANT — if the user is asking you to DO something that requires reading "
    "files, running commands, editing code, deploying, pushing, building, or any "
    "other action in the workspace, output ONLY this marker on the very first line "
    "of your response, then give a one-sentence description of what you will do:\n"
    f"    {ESCALATE_MARKER}\n"
    "The system will automatically re-run the request in task mode — do NOT tell "
    "the user to retype their message or switch modes manually.\n\n"
    "For actual questions, explanations, or follow-ups about past runs: answer "
    "directly without the marker. Be honest about what the last agent run did or "
    "did not accomplish based on the provided context."
)


def needs_escalation(reply: str) -> bool:
    """True when the chat LLM signalled that this request needs task mode."""
    return ESCALATE_MARKER in reply


def _build_chat_system() -> str:
    """Build the chat system prompt with live workspace and index context."""
    try:
        from coding_harness import config as _cfg
        workspace = _cfg.WORKSPACE_ROOT
    except Exception:
        workspace = ""

    lines = [_CHAT_SYSTEM_BASE, "", f"Workspace: {workspace}"]

    try:
        from coding_harness.index_tools import INDEXER_AVAILABLE, index_status
        if INDEXER_AVAILABLE:
            status = index_status(workspace)
            if status.get("exists"):
                age = status.get("age_hours")
                age_str = f", last updated {age:.1f}h ago" if age is not None else ""
                lines.append(
                    f"Repository index: BUILT — {status['total_files']} files, "
                    f"{status['total_symbols']} symbols indexed{age_str}. "
                    "The agent (task mode) can query it with find_symbol, "
                    "find_references, find_callers, and search_symbols tools."
                )
            else:
                lines.append(
                    "Repository index: NOT YET BUILT. "
                    "User can run /index to build it."
                )
        else:
            lines.append("Repository index: not available (wells-index not installed).")
    except Exception:
        pass

    return "\n".join(lines)


def _stream_with_retry(llm, messages):
    """Stream a response with the same transient-retry policy as _invoke_with_retry.

    Each retry re-runs the whole stream from the start (mid-stream resume is
    not possible over HTTP). Yields content chunks; raises the last error if
    every retry fails.
    """
    import time

    last_err: Exception | None = None
    for attempt in range(1, LLM_MAX_RETRIES + 1):
        try:
            collected: list[str] = []
            for chunk in llm.stream(messages):
                piece = chunk.content or ""
                if piece:
                    collected.append(piece)
                    yield piece
            return  # success
        except Exception as err:
            last_err = err
            if not _is_transient(err) or attempt == LLM_MAX_RETRIES:
                raise
            backoff = min(LLM_BACKOFF_BASE**attempt, 30.0)
            print(
                f"[chat] transient {type(err).__name__} on stream attempt "
                f"{attempt}/{LLM_MAX_RETRIES}; retrying in {backoff:.1f}s ..."
            )
            time.sleep(backoff)
    assert last_err is not None
    raise last_err


def conversational_reply(
    text: str,
    memory: ConversationMemory,
    *,
    on_token=None,
) -> str:
    """Stream a direct conversational reply to ``text``.

    Records the exchange in ``memory`` and accounts tokens to the global
    :data:`LEDGER` under the ``chat`` step. ``on_token(token)`` is called for
    each streamed token (if the model streams).
    """
    system_prompt = _build_chat_system()
    messages = memory.as_messages(system_prompt)
    messages.append(HumanMessage(content=text))
    llm = get_llm_for_task("chat")
    model = model_name_for_task("chat")
    full_text = system_prompt + "\n" + text

    try:
        # Stream if a callback was supplied (with transient retry).
        if on_token is not None:
            collected: list[str] = []
            for piece in _stream_with_retry(llm, messages):
                collected.append(piece)
                on_token(piece)
            content = "".join(collected).strip()
            # usage_metadata isn't reliably on the final stream chunk; estimate.
            input_tokens = estimate_tokens(full_text)
            output_tokens = estimate_tokens(content) if content else 0
            reasoning_tokens = 0
            cache_read_tokens = 0
        else:
            resp = _invoke_with_retry(llm, messages)
            content = (resp.content or "").strip()
            um = getattr(resp, "usage_metadata", None) or {}
            input_tokens = um.get("input_tokens") or estimate_tokens(full_text)
            output_tokens = um.get("output_tokens") or 0
            reasoning_tokens = (
                (um.get("output_token_details") or {}).get("reasoning")
            ) or 0
            cache_read_tokens = (
                (um.get("input_token_details") or {}).get("cache_read")
            ) or 0
        calibrate(full_text, input_tokens)
    except Exception as err:
        from coding_harness.logger import log_error, log_path
        log_error(f"conversational_reply failed: {type(err).__name__}: {err}", err)
        content = f"(chat call failed: {type(err).__name__}: {str(err)[:160]})"
        input_tokens = estimate_tokens(full_text)
        output_tokens = reasoning_tokens = cache_read_tokens = 0
        print(f"[chat] {content}")
        print(f"[chat] Full error logged to: {log_path()}")

    LEDGER.record(
        step="chat",
        task_type="chat",
        model=model,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        reasoning_tokens=reasoning_tokens,
        cache_read_tokens=cache_read_tokens,
        category_tokens={"chat": input_tokens},
        saved_by_trim=0,
        saved_by_summary=0,
    )

    memory.add("user", text)
    memory.add("assistant", content or "(no response)")
    return content
