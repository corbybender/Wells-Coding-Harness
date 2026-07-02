"""Tests for the conversational intent router and chat memory.

These cover the heuristic classifier (the free, instant path) on a spread of
realistic REPL inputs, plus ConversationMemory bookkeeping. The LLM-fallback
classifier and the live conversational_reply are integration-tested manually.
"""

from __future__ import annotations

import pytest

from coding_harness import chat
from coding_harness.chat import ConversationMemory, classify_intent


# ---------------------------------------------------------------------------
# Heuristic classifier (no LLM calls)
# ---------------------------------------------------------------------------

# Everything that isn't clearly architectural routes to "auto" — the direct
# executor answers questions AND performs small tasks.
AUTO_INPUTS = [
    "explain your previous output in a simple summary. Did you actually make the fix I asked for?",
    "Did you actually make the fix?",
    "What does this file do?",
    "hello",
    "thanks!",
    "why does the test fail?",
    "how do I fix the login bug?",
    "What did you change?",
    "ok",
    "what's the status of the last run?",
    "fix the bug in parser.py",
    "add a login page",
    "write a test for the parser",
    "delete the old config file",
    "update the README to include install steps",
    "please add a button that loads the selected page",
]

# Big-creation verb + architectural scope noun → full orchestration graph.
ORCHESTRATE_INPUTS = [
    "build a login system with OAuth2, JWT, and database schema",
    "implement a complete REST API with authentication and tests",
    "create an authentication service for the platform",
    "design a data pipeline for ingesting events",
    "rewrite the backend as a microservice",
    "set up a CI/CD pipeline for this repo",
]


@pytest.mark.parametrize("text", AUTO_INPUTS)
def test_classify_auto(text):
    assert classify_intent(text, use_llm_fallback=False) == "auto", text


@pytest.mark.parametrize("text", ORCHESTRATE_INPUTS)
def test_classify_orchestrate(text):
    assert classify_intent(text, use_llm_fallback=False) == "orchestrate", text


def test_empty_is_auto():
    assert classify_intent("", use_llm_fallback=False) == "auto"


def test_ambiguous_defaults_to_auto_without_llm():
    # A single neutral noun like "dashboard" is ambiguous -> auto (safe default).
    assert classify_intent("dashboard", use_llm_fallback=False) == "auto"
    assert classify_intent("something", use_llm_fallback=False) == "auto"


def test_classifier_cache_cleared():
    chat._CLASSIFIER_CACHE["stub"] = "auto"
    chat.clear_classifier_cache()
    assert chat._CLASSIFIER_CACHE == {}


# ---------------------------------------------------------------------------
# ConversationMemory
# ---------------------------------------------------------------------------


def test_memory_add_and_messages():
    mem = ConversationMemory(max_turns=4)
    mem.add("user", "hello")
    mem.add("assistant", "hi there")
    msgs = mem.as_messages("SYSTEM")
    assert msgs[0].content == "SYSTEM"
    assert msgs[1].content == "hello"
    assert msgs[2].content == "hi there"


def test_memory_run_summary_injected():
    mem = ConversationMemory()
    mem.set_run_summary("Goal: fix bug\nStatus: INCOMPLETE")
    msgs = mem.as_messages("SYSTEM")
    # system + run-summary-system + (no turns) = 2 messages
    assert len(msgs) == 2
    assert "fix bug" in msgs[1].content


def test_memory_bounded():
    mem = ConversationMemory(max_turns=2)
    mem.add("user", "a")
    mem.add("assistant", "b")
    mem.add("user", "c")  # should evict oldest
    msgs = mem.as_messages("S")
    # system + 2 turns
    assert len(msgs) == 3
    contents = [m.content for m in msgs]
    assert "a" not in contents  # evicted


def test_memory_clear():
    mem = ConversationMemory()
    mem.add("user", "x")
    mem.set_run_summary("summary")
    mem.clear()
    assert mem.turns == [] or len(mem.turns) == 0
    assert mem.last_run_summary == ""
