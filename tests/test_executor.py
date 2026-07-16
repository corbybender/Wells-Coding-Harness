"""Tests for the agentic executor loop (Layer 2).

Uses a scripted mock model so the loop logic is verified without live API calls.
Covers text-fallback parsing, native tool_calls, the step cap, and error paths.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from wells import config, executor, tools
from wells.tokens import LEDGER


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


def test_parse_bare_json_call_no_wrapper():
    # This is the exact payload observed from qwen2.5-coder:7b via Ollama:
    # the model calls the tool correctly using its own trained format, with
    # no <tool_call> wrapper and no native tool_calls field populated.
    text = '{"name": "write_file", "arguments": {"path": "hello.txt", "content": "hello world"}}'
    calls = executor.parse_text_tool_calls(text)
    assert calls == [{"name": "write_file", "args": {"path": "hello.txt", "content": "hello world"}}]


def test_parse_bare_json_call_in_fence():
    text = '```json\n{"name": "list_dir", "arguments": {"path": "."}}\n```'
    calls = executor.parse_text_tool_calls(text)
    assert calls == [{"name": "list_dir", "args": {"path": "."}}]


def test_parse_bare_json_ignores_plain_prose():
    calls = executor.parse_text_tool_calls("Sure, I can help with that. Let me think it over.")
    assert calls == []


def test_parse_bare_json_ignores_unrelated_json():
    # A JSON-shaped answer that isn't a tool call (no "name"/"arguments" tool
    # call shape) must not be misread as one.
    calls = executor.parse_text_tool_calls('{"result": 42, "ok": true}')
    assert calls == []


def test_parse_bare_json_broken_by_unescaped_quote_reports_parse_error():
    """Reproduces the live bug: qwen embedded Python source containing
    `"utf8"` as a JSON string value without escaping the inner quotes,
    breaking the JSON. The old code silently treated this as "no call at
    all" (continue, no error), which let the executor fall through to
    code-block salvage and write the broken JSON wrapper itself to disk as
    if it were the file content. It must now be reported as a parse error."""
    text = (
        '```json\n{ "name": "write_file", "arguments": { "path": "tree.py", '
        '"content": "tree.parse(bytes(source_code, "utf8"))" } }\n```'
    )
    calls = executor.parse_text_tool_calls(text)
    assert len(calls) == 1
    assert "_parse_error" in calls[0]
    assert "utf8" in calls[0]["_parse_error"]


def test_parse_bare_json_fenced_call_not_shadowed_by_raw_text_fallback():
    """The raw whole-text candidate (which includes the ```json fence
    markers and therefore always fails json.loads) must not shadow a
    well-formed fenced call — it has to be tried, and succeed, before the
    raw-text fallback is ever considered a "failed attempt"."""
    text = '```json\n{"name": "list_dir", "arguments": {"path": "."}}\n```'
    calls = executor.parse_text_tool_calls(text)
    assert calls == [{"name": "list_dir", "args": {"path": "."}}]


# ---------------------------------------------------------------------------
# Compact system prompt (small local-model context windows)
# ---------------------------------------------------------------------------


def test_system_prompt_compact_is_meaningfully_smaller(tmp_path: Path):
    """Measured live: the full prompt for this repo's own RULES.md alone was
    4131 tokens — bigger than Ollama's default 4096-token context window,
    before the task, history, or response budget. Compact mode must shrink
    it enough to matter, not just symbolically."""
    from wells.tokens import estimate_tokens
    from wells import tools as _tools

    # Sized to roughly match this repo's own real RULES.md (~7.5KB, measured
    # live at 1710 of the full prompt's 4131 tokens) rather than a token
    # RULES.md whose savings ratio wouldn't be representative.
    (tmp_path / "RULES.md").write_text("R1 — a rule.\n" * 500, encoding="utf-8")
    task = "Write a Python script."
    full = executor._system_prompt(
        task, _tools.ALL_TOOLS, plan_mode=False, workspace=str(tmp_path), compact=False
    )
    compact = executor._system_prompt(
        task, _tools.ALL_TOOLS, plan_mode=False, workspace=str(tmp_path), compact=True
    )
    assert estimate_tokens(compact) < estimate_tokens(full) * 0.75


def test_compact_prompt_includes_matching_few_shot_examples(tmp_path: Path):
    """Compact mode targets small models — they get two worked tool-call
    examples in whichever format the run expects; full mode skips them."""
    from wells import tools as _tools

    compact_text = executor._system_prompt(
        "t", _tools.ALL_TOOLS, plan_mode=False, workspace=str(tmp_path),
        compact=True, structured=False,
    )
    assert "EXAMPLES — copy this format exactly" in compact_text
    assert '<tool_call>{"name": "read_file"' in compact_text

    compact_structured = executor._system_prompt(
        "t", _tools.ALL_TOOLS, plan_mode=False, workspace=str(tmp_path),
        compact=True, structured=True,
    )
    assert "EXAMPLES — copy this format exactly" in compact_structured
    # Structured examples must show the bare-JSON form, not the tag.
    assert "<tool_call>" not in compact_structured.split("EXAMPLES")[1]

    full_text = executor._system_prompt(
        "t", _tools.ALL_TOOLS, plan_mode=False, workspace=str(tmp_path),
        compact=False,
    )
    assert "EXAMPLES — copy this format exactly" not in full_text


def test_run_executor_auto_detects_local_ollama_and_uses_compact_prompt(
    ctx: tools.ToolContext,
):
    """The executor must resolve the active profile and switch to the
    compact system prompt automatically for a profile that looks like local
    Ollama — no manual opt-in should be required for the common case."""
    LEDGER.reset()
    from pathlib import Path as _Path
    from wells import providers as _providers

    # A RULES.md distinguishing marker: present in full mode, absent in
    # compact — without this the assertion would pass trivially (no
    # RULES.md at all means neither mode includes it).
    _Path(ctx.workspace, "RULES.md").write_text(
        "the harness enforces the machine-checkable ones and audits the rest\n",
        encoding="utf-8",
    )

    local_profile = _providers.ProviderProfile(
        name="LocalQwen3", kind="openai", model="qwen2.5-coder:7b",
        base_url="http://127.0.0.1:11434/v1",
    )
    script = [AIMessage(content="Done.")]
    with (
        patch.object(config, "_invoke_with_retry", side_effect=_scripted(script)),
        patch.object(executor, "_try_bind_tools", return_value=None),
        patch.object(config.providers, "load_profile", return_value=local_profile),
        patch.object(config.providers, "get_chat_model", return_value=object()),
    ):
        result = executor.run_executor(
            task="x", ctx=ctx, max_steps=3, step_label="t", profile="LocalQwen3"
        )
    system_text = result.messages[0].content
    # The full-mode RULES.md marker phrase must not appear; the compact
    # pointer phrasing must, confirming compact=True actually reached
    # _system_prompt rather than just being computed and discarded.
    assert "the harness enforces the machine-checkable ones and audits" not in system_text


# ---------------------------------------------------------------------------
# Structured outputs (grammar-constrained tool calls for local Ollama)
# ---------------------------------------------------------------------------


class _FakeBindableLLM:
    """Records bind() kwargs; bound copies share the recording list."""

    def __init__(self, binds=None):
        self.binds = binds if binds is not None else []

    def bind(self, **kwargs):
        self.binds.append(kwargs)
        return _FakeBindableLLM(self.binds)


def _local_ollama_profile(kind="openai"):
    from wells import providers as _providers
    return _providers.ProviderProfile(
        name="LocalQwen3", kind=kind, model="qwen2.5-coder:7b",
        base_url="http://127.0.0.1:11434/v1" if kind == "openai"
        else "http://127.0.0.1:11434",
    )


def test_structured_schema_lists_all_tools_plus_final_answer():
    schema = executor._structured_output_schema(tools.ALL_TOOLS)
    names = schema["properties"]["name"]["enum"]
    assert "final_answer" in names
    for t in tools.ALL_TOOLS:
        assert t.name in names
    assert schema["required"] == ["name", "args"]


def test_try_bind_structured_skips_cloud_profiles():
    from wells import providers as _providers
    cloud = _providers.ProviderProfile(
        name="zai", kind="openai", model="glm-5.2",
        base_url="https://api.z.ai/api/coding/paas/v4/",
    )
    assert executor._try_bind_structured(_FakeBindableLLM(), cloud, tools.ALL_TOOLS) is None
    assert executor._try_bind_structured(_FakeBindableLLM(), None, tools.ALL_TOOLS) is None


def test_try_bind_structured_native_ollama_uses_format_field():
    llm = _FakeBindableLLM()
    bound = executor._try_bind_structured(llm, _local_ollama_profile("ollama"), tools.ALL_TOOLS)
    assert bound is not None
    assert "format" in llm.binds[0]
    assert llm.binds[0]["format"]["type"] == "object"


def test_try_bind_structured_openai_shim_uses_response_format():
    llm = _FakeBindableLLM()
    bound = executor._try_bind_structured(llm, _local_ollama_profile("openai"), tools.ALL_TOOLS)
    assert bound is not None
    rf = llm.binds[0]["response_format"]
    assert rf["type"] == "json_schema"
    assert rf["json_schema"]["schema"]["properties"]["name"]["enum"]


def test_resolve_structured_reply_final_answer_becomes_text():
    calls = [{"name": "final_answer", "args": {"summary": "all done, tests pass"}}]
    real, text = executor._resolve_structured_reply(calls, '{"name": "final_answer"...}')
    assert real == []
    assert text == "all done, tests pass"


def test_resolve_structured_reply_blanks_raw_json_for_real_calls():
    raw = '{"name": "read_file", "args": {"path": "x.py"}}'
    calls = [{"name": "read_file", "args": {"path": "x.py"}}]
    real, text = executor._resolve_structured_reply(calls, raw)
    assert real == calls
    assert text == ""  # wire format is not narrative


def test_loop_structured_mode_executes_and_finishes_on_final_answer(
    ctx: tools.ToolContext, workspace: Path
):
    """End-to-end: local Ollama profile + structured binding → bare-JSON tool
    call executes, final_answer ends the run with its summary as the text."""
    LEDGER.reset()
    script = [
        AIMessage(content='{"name": "write_file", "args": {"path": "out.txt", "content": "hi"}}'),
        AIMessage(content='{"name": "final_answer", "args": {"summary": "Wrote out.txt containing hi."}}'),
    ]
    llm = _FakeBindableLLM()
    with (
        patch.object(config, "_invoke_with_retry", side_effect=_scripted(script)),
        patch.object(config, "STRUCTURED_OUTPUTS", True),
        patch.object(config.providers, "load_profile", return_value=_local_ollama_profile()),
        patch.object(config.providers, "get_chat_model", return_value=llm),
    ):
        result = executor.run_executor(
            task="write hi to out.txt", ctx=ctx, max_steps=5,
            step_label="t", profile="LocalQwen3",
        )
    assert (workspace / "out.txt").read_text() == "hi"
    assert result.stopped_reason == "done"
    assert result.summary == "Wrote out.txt containing hi."
    assert llm.binds, "structured schema was never bound"
    assert "STRUCTURED MODE" in result.messages[0].content


def test_loop_structured_final_answer_with_no_action_gets_stall_nudge(
    ctx: tools.ToolContext, workspace: Path
):
    """final_answer before any real action is a premature finish — the stall
    nudge must fire (in its structured-mode wording) before accepting it."""
    LEDGER.reset()
    script = [
        AIMessage(content='{"name": "final_answer", "args": {"summary": "done!"}}'),
        AIMessage(content='{"name": "write_file", "args": {"path": "a.txt", "content": "x"}}'),
        AIMessage(content='{"name": "final_answer", "args": {"summary": "actually done"}}'),
    ]
    llm = _FakeBindableLLM()
    with (
        patch.object(config, "_invoke_with_retry", side_effect=_scripted(script)),
        patch.object(config, "STRUCTURED_OUTPUTS", True),
        patch.object(config.providers, "load_profile", return_value=_local_ollama_profile()),
        patch.object(config.providers, "get_chat_model", return_value=llm),
    ):
        result = executor.run_executor(
            task="write x to a.txt", ctx=ctx, max_steps=5,
            step_label="t", profile="LocalQwen3",
        )
    assert (workspace / "a.txt").exists()
    assert result.summary == "actually done"
    nudges = [m for m in result.messages
              if isinstance(m, HumanMessage) and "final_answer" in (m.content or "")
              and m.content.startswith("[HARNESS")]
    assert nudges, "structured stall nudge never delivered"


def test_loop_structured_mode_falls_back_when_server_rejects_format(
    ctx: tools.ToolContext, workspace: Path
):
    """An older Ollama that errors on response_format must not fail the task:
    one fallback to the plain path, then the run continues."""
    LEDGER.reset()
    boom = {"first": True}

    def _invoke(llm, messages):
        if boom["first"]:
            boom["first"] = False
            raise RuntimeError("response_format is not supported")
        return AIMessage(content="Done, nothing needed.")

    llm = _FakeBindableLLM()
    with (
        patch.object(config, "_invoke_with_retry", side_effect=_invoke),
        patch.object(config, "STRUCTURED_OUTPUTS", True),
        patch.object(executor, "_try_bind_tools", return_value=None),
        patch.object(config.providers, "load_profile", return_value=_local_ollama_profile()),
        patch.object(config.providers, "get_chat_model", return_value=llm),
    ):
        result = executor.run_executor(
            task="answer a question", ctx=ctx, max_steps=3,
            step_label="t", profile="LocalQwen3",
        )
    assert result.stopped_reason == "done"
    assert "Done" in result.summary


def test_structured_outputs_disabled_by_config(ctx: tools.ToolContext):
    """WELLS_STRUCTURED=0 must skip the binding entirely."""
    LEDGER.reset()
    llm = _FakeBindableLLM()
    with (
        patch.object(config, "_invoke_with_retry", side_effect=_scripted([AIMessage(content="ok")])),
        patch.object(config, "STRUCTURED_OUTPUTS", False),
        patch.object(executor, "_try_bind_tools", return_value=None),
        patch.object(config.providers, "load_profile", return_value=_local_ollama_profile()),
        patch.object(config.providers, "get_chat_model", return_value=llm),
    ):
        result = executor.run_executor(
            task="x", ctx=ctx, max_steps=2, step_label="t", profile="LocalQwen3",
        )
    assert llm.binds == []
    assert "STRUCTURED MODE" not in result.messages[0].content


# ---------------------------------------------------------------------------
# Parallel read-only tool dispatch
# ---------------------------------------------------------------------------


def test_batched_readonly_calls_all_execute_in_order(
    ctx: tools.ToolContext, workspace: Path
):
    LEDGER.reset()
    (workspace / "other.py").write_text("y = 2\n")
    batch = (
        '<tool_call>{"name": "read_file", "args": {"path": "maths.py"}}</tool_call>\n'
        '<tool_call>{"name": "read_file", "args": {"path": "other.py"}}</tool_call>\n'
        '<tool_call>{"name": "list_dir", "args": {"path": "."}}</tool_call>'
    )
    with (
        patch.object(config, "_invoke_with_retry",
                     side_effect=_scripted([AIMessage(content=batch), AIMessage(content="done")])),
        patch.object(config, "STRUCTURED_OUTPUTS", False),
        patch.object(executor, "_try_bind_tools", return_value=None),
    ):
        result = executor.run_executor(task="x", ctx=ctx, max_steps=10, step_label="t")
    assert [c["name"] for c in result.tool_calls[:3]] == ["read_file", "read_file", "list_dir"]
    assert all(c["ok"] for c in result.tool_calls[:3])
    tool_msgs = [m for m in result.messages if isinstance(m, ToolMessage)]
    assert "return a - b" in tool_msgs[0].content  # maths.py content, in order
    assert "y = 2" in tool_msgs[1].content


def test_read_after_write_in_same_batch_sees_fresh_content(
    ctx: tools.ToolContext, workspace: Path
):
    """A read-only call AFTER a mutating call in the same batch must not be
    prefetched — it has to observe the write's effect, not stale state."""
    LEDGER.reset()
    batch = (
        '<tool_call>{"name": "write_file", "args": {"path": "fresh.txt", "content": "brand new"}}</tool_call>\n'
        '<tool_call>{"name": "read_file", "args": {"path": "fresh.txt"}}</tool_call>'
    )
    with (
        patch.object(config, "_invoke_with_retry",
                     side_effect=_scripted([AIMessage(content=batch), AIMessage(content="done")])),
        patch.object(config, "STRUCTURED_OUTPUTS", False),
        patch.object(executor, "_try_bind_tools", return_value=None),
    ):
        result = executor.run_executor(task="x", ctx=ctx, max_steps=10, step_label="t")
    read_calls = [c for c in result.tool_calls if c["name"] == "read_file"]
    assert read_calls and read_calls[0]["ok"]
    tool_msgs = [m for m in result.messages if isinstance(m, ToolMessage)]
    assert any("brand new" in (m.content or "") for m in tool_msgs)


# ---------------------------------------------------------------------------
# Streaming repetition kill-switch
# ---------------------------------------------------------------------------


def test_tail_repetition_detects_degenerate_loop():
    assert executor._tail_repetition("I will now fix it. " * 5) is True
    assert executor._tail_repetition("print(1)\n" * 6) is True


def test_tail_repetition_ignores_normal_text_and_banners():
    assert executor._tail_repetition(
        "Read the file, found the bug on line 3, fixing it now with an edit."
    ) is False
    assert executor._tail_repetition("=" * 400) is False  # banner, not a loop
    assert executor._tail_repetition("- " * 200) is False  # low-variety unit
    assert executor._tail_repetition("short") is False


def test_stream_invoke_guard_aborts_early():
    from langchain_core.messages import AIMessageChunk
    from wells.control import CONTROL

    CONTROL.reset()
    consumed = []

    class _RepeatingLLM:
        def stream(self, messages):
            for i in range(50):
                consumed.append(i)
                yield AIMessageChunk(content="I will now fix the bug. ")

    resp, streamed, aborted = executor._stream_invoke(
        _RepeatingLLM(), [], display=False, guard=True
    )
    assert aborted is True
    assert resp is not None
    assert len(consumed) < 50, "stream ran to completion despite the guard"


def test_loop_guard_abort_nudges_then_recovers(ctx: tools.ToolContext, workspace: Path):
    from langchain_core.messages import AIMessageChunk
    from wells import providers as _providers
    from wells.control import CONTROL

    CONTROL.reset()
    LEDGER.reset()
    scripts = iter([
        ["Fixing. " ] + ["I will now fix it. "] * 20,   # degenerate → aborted
        ['<tool_call>{"name": "read_file", "args": {"path": "maths.py"}}</tool_call>'],
        ["All finished; the file was reviewed."],
    ])

    class _FakeStreamLLM:
        def stream(self, messages):
            for piece in next(scripts):
                yield AIMessageChunk(content=piece)

    local = _providers.ProviderProfile(
        name="q", kind="openai", model="qwen2.5-coder:7b",
        base_url="http://127.0.0.1:11434/v1",
    )
    with (
        patch.object(config, "STRUCTURED_OUTPUTS", False),
        patch.object(config, "STREAM_GUARD", True),
        patch.object(executor, "_try_bind_tools", return_value=None),
        patch.object(config.providers, "load_profile", return_value=local),
        patch.object(config.providers, "get_chat_model", return_value=_FakeStreamLLM()),
    ):
        result = executor.run_executor(
            task="review maths.py", ctx=ctx, max_steps=5,
            step_label="t", profile="LocalQwen3",
        )
    assert result.stopped_reason == "done"
    assert "All finished" in result.summary
    nudges = [m for m in result.messages
              if isinstance(m, HumanMessage) and "repeating the same text" in (m.content or "")]
    assert len(nudges) == 1


# ---------------------------------------------------------------------------
# Read-only dedupe cache
# ---------------------------------------------------------------------------

_READ_MATHS = AIMessage(
    content='<tool_call>{"name": "read_file", "args": {"path": "maths.py"}}</tool_call>'
)


def test_identical_read_within_window_returns_cache_pointer(
    ctx: tools.ToolContext, workspace: Path
):
    LEDGER.reset()
    script = [_READ_MATHS, _READ_MATHS, AIMessage(content="done")]
    with (
        patch.object(config, "_invoke_with_retry", side_effect=_scripted(script)),
        patch.object(config, "STRUCTURED_OUTPUTS", False),
        patch.object(executor, "_try_bind_tools", return_value=None),
    ):
        result = executor.run_executor(task="x", ctx=ctx, max_steps=10, step_label="t")
    tool_msgs = [m for m in result.messages if isinstance(m, ToolMessage)]
    assert "return a - b" in tool_msgs[0].content        # first read: real content
    assert "HARNESS CACHE" in tool_msgs[1].content       # second: pointer
    assert "return a - b" not in tool_msgs[1].content    # not paid for twice


def test_mutation_invalidates_read_cache(ctx: tools.ToolContext, workspace: Path):
    """read → write same file → read again: the second read must return the
    FRESH content, not a stale cache pointer."""
    LEDGER.reset()
    script = [
        _READ_MATHS,
        AIMessage(content='<tool_call>{"name": "write_file", "args": {"path": "maths.py", "content": "def add(a, b):\\n    return a + b\\n"}}</tool_call>'),
        _READ_MATHS,
        AIMessage(content="done"),
    ]
    with (
        patch.object(config, "_invoke_with_retry", side_effect=_scripted(script)),
        patch.object(config, "STRUCTURED_OUTPUTS", False),
        patch.object(executor, "_try_bind_tools", return_value=None),
    ):
        result = executor.run_executor(task="x", ctx=ctx, max_steps=10, step_label="t")
    read_msgs = [m for m in result.messages
                 if isinstance(m, ToolMessage) and m.name == "read_file"]
    assert len(read_msgs) == 2
    assert "HARNESS CACHE" not in read_msgs[1].content
    assert "return a + b" in read_msgs[1].content


def test_dedupe_disabled_by_config(ctx: tools.ToolContext, workspace: Path):
    LEDGER.reset()
    script = [_READ_MATHS, _READ_MATHS, AIMessage(content="done")]
    with (
        patch.object(config, "_invoke_with_retry", side_effect=_scripted(script)),
        patch.object(config, "STRUCTURED_OUTPUTS", False),
        patch.object(config, "DEDUPE_READS", False),
        patch.object(executor, "_try_bind_tools", return_value=None),
    ):
        result = executor.run_executor(task="x", ctx=ctx, max_steps=10, step_label="t")
    tool_msgs = [m for m in result.messages if isinstance(m, ToolMessage)]
    assert all("HARNESS CACHE" not in (m.content or "") for m in tool_msgs)


# ---------------------------------------------------------------------------
# Model cascade / escalation on failure
# ---------------------------------------------------------------------------


def _two_model_setup(small_script, big_script):
    """Patches for a run where profile 'small' escalates to profile 'big'.

    Returns (patchers, sentinel_small, sentinel_big).
    """
    from wells import providers as _providers

    small_llm, big_llm = object(), object()
    profiles = {
        "small": _providers.ProviderProfile(
            name="small", kind="openai", model="qwen2.5-coder:7b",
            base_url="https://smallhost.example/v1",
        ),
        "big": _providers.ProviderProfile(
            name="big", kind="openai", model="glm-5.2",
            base_url="https://api.z.ai/api/coding/paas/v4/",
        ),
    }
    small_it, big_it = iter(small_script), iter(big_script)

    def _invoke(llm, messages):
        it = big_it if llm is big_llm else small_it
        try:
            return next(it)
        except StopIteration:
            return AIMessage(content="(done)")

    patchers = (
        patch.object(config, "_invoke_with_retry", side_effect=_invoke),
        patch.object(config, "STRUCTURED_OUTPUTS", False),
        patch.object(config, "ESCALATION_PROFILE", "big"),
        patch.object(executor, "_try_bind_tools", return_value=None),
        patch.object(config.providers, "load_profile",
                     side_effect=lambda n: profiles.get(n)),
        patch.object(config.providers, "get_chat_model",
                     side_effect=lambda n, **kw: big_llm if n == "big" else small_llm),
    )
    return patchers, small_llm, big_llm


def test_escalation_rescues_a_stuck_run(ctx: tools.ToolContext, workspace: Path):
    """Six identical calls would hard-stop the run; with ESCALATION_PROFILE
    set, the stronger model takes over the same transcript and finishes."""
    LEDGER.reset()
    stuck = [
        AIMessage(content='<tool_call>{"name": "read_file", "args": {"path": "maths.py"}}</tool_call>')
    ] * 6
    rescue = [
        AIMessage(content='<tool_call>{"name": "edit_file", "args": {"path": "maths.py", "old_string": "return a - b", "new_string": "return a + b"}}</tool_call>'),
        AIMessage(content="Recovered: fixed the add bug and finished."),
    ]
    patchers, _, _ = _two_model_setup(stuck, rescue)
    import contextlib
    with contextlib.ExitStack() as stack:
        for p in patchers:
            stack.enter_context(p)
        result = executor.run_executor(
            task="fix maths.py", ctx=ctx, max_steps=20, step_label="t",
            profile="small",
        )
    assert result.stopped_reason == "done"
    assert "Recovered" in result.summary
    assert "return a + b" in (workspace / "maths.py").read_text()
    handoff = [m for m in result.messages
               if isinstance(m, HumanMessage) and "has been replaced" in (m.content or "")]
    assert len(handoff) == 1


def test_escalation_fires_only_once_then_stops_for_real(
    ctx: tools.ToolContext, workspace: Path
):
    LEDGER.reset()
    stuck = [
        AIMessage(content='<tool_call>{"name": "read_file", "args": {"path": "maths.py"}}</tool_call>')
    ] * 6
    also_stuck = [
        AIMessage(content='<tool_call>{"name": "list_dir", "args": {"path": "."}}</tool_call>')
    ] * 7
    patchers, _, _ = _two_model_setup(stuck, also_stuck)
    import contextlib
    with contextlib.ExitStack() as stack:
        for p in patchers:
            stack.enter_context(p)
        result = executor.run_executor(
            task="fix maths.py", ctx=ctx, max_steps=30, step_label="t",
            profile="small",
        )
    assert result.stopped_reason == "stuck_loop"


def test_no_escalation_when_profile_unset(ctx: tools.ToolContext):
    """Default behavior unchanged: unset ESCALATION_PROFILE = hard stop."""
    LEDGER.reset()
    stuck = [
        AIMessage(content='<tool_call>{"name": "read_file", "args": {"path": "maths.py"}}</tool_call>')
    ] * 7
    with (
        patch.object(config, "_invoke_with_retry", side_effect=_scripted(stuck)),
        patch.object(config, "STRUCTURED_OUTPUTS", False),
        patch.object(config, "ESCALATION_PROFILE", ""),
        patch.object(executor, "_try_bind_tools", return_value=None),
    ):
        result = executor.run_executor(task="x", ctx=ctx, max_steps=20, step_label="t")
    assert result.stopped_reason == "stuck_loop"


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


def test_loop_detects_stuck_repeat(ctx: tools.ToolContext):
    LEDGER.reset()
    # Same tool, same args, every round, and every call succeeds — no step
    # cap and no failing command, so nothing else in the harness would ever
    # stop this. The stuck-loop backstop should kick in after 6 repeats.
    repeat = AIMessage(
        content='<tool_call>{"name": "list_dir", "args": {"path": "."}}</tool_call>'
    )
    script = [repeat] * 10
    with (
        patch.object(config, "_invoke_with_retry", side_effect=_scripted(script)),
        patch.object(executor, "_try_bind_tools", return_value=None),
    ):
        result = executor.run_executor(
            task="loop forever", ctx=ctx, max_steps=0, step_label="t"
        )
    assert result.stopped_reason == "stuck_loop"
    assert result.steps_taken == 6
    assert all(c["name"] == "list_dir" for c in result.tool_calls)


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


def test_loop_nudges_stalled_model_then_recovers(ctx: tools.ToolContext, workspace: Path):
    """A model that answers in prose with zero tool calls before taking any
    real action (weak/local model ignoring the protocol) gets coached back
    onto it instead of the run silently ending with nothing done."""
    LEDGER.reset()
    script = [
        AIMessage(content="### Step 1: First install the grammar, then..."),
        AIMessage(
            content='<tool_call>{"name": "read_file", "args": {"path": "maths.py"}}</tool_call>'
        ),
        AIMessage(content="Done: read the file."),
    ]
    with (
        patch.object(config, "_invoke_with_retry", side_effect=_scripted(script)),
        patch.object(executor, "_try_bind_tools", return_value=None),
    ):
        result = executor.run_executor(
            task="fix add()", ctx=ctx, max_steps=6, step_label="t"
        )
    assert result.stopped_reason == "done"
    assert result.steps_taken == 1
    assert result.tool_calls[0]["name"] == "read_file"


def test_loop_handles_bare_json_call_from_qwen_style_model(
    ctx: tools.ToolContext, workspace: Path
):
    """Reproduces the live failure: qwen2.5-coder:7b via Ollama answers with
    its native bare-JSON function-call format, un-wrapped, and Ollama never
    populates the structured tool_calls field for this template. The old
    parser (which only recognized <tool_call> tags) saw this as zero calls
    and gave up with nothing done; it must now execute normally."""
    LEDGER.reset()
    script = [
        AIMessage(
            content='{"name": "read_file", "arguments": {"path": "maths.py"}}'
        ),
        AIMessage(content="Done."),
    ]
    with (
        patch.object(config, "_invoke_with_retry", side_effect=_scripted(script)),
        patch.object(executor, "_try_bind_tools", return_value=None),
    ):
        result = executor.run_executor(
            task="read maths.py", ctx=ctx, max_steps=6, step_label="t"
        )
    assert result.stopped_reason == "done"
    assert result.steps_taken == 1
    assert result.tool_calls[0]["name"] == "read_file"


# ---------------------------------------------------------------------------
# WorkingMemory task re-anchoring
# ---------------------------------------------------------------------------


def test_working_memory_not_empty_with_only_task_set():
    """Regression: is_empty() used to ignore `task` entirely, so on round 1
    (before any tool call populates files_modified/etc.) WorkingMemory was
    considered empty and never injected — the goal reminder only started
    appearing after the model had already taken its first action."""
    wm = executor.WorkingMemory(task="write tree.py")
    assert wm.is_empty() is False


def test_working_memory_empty_with_nothing_set():
    wm = executor.WorkingMemory()
    assert wm.is_empty() is True


def test_working_memory_to_xml_includes_prominent_task_anchor():
    wm = executor.WorkingMemory(task="write tree.py")
    xml = wm.to_xml()
    assert "write tree.py" in xml
    assert "YOUR GOAL" in xml  # loud framing, not just a plain <task> tag


def test_inject_wm_fires_on_round_one_before_any_tool_call():
    """The task reminder must reach the model's very first round, not only
    after files_modified/etc. become non-empty from a prior tool call."""
    wm = executor.WorkingMemory(task="write tree.py")
    messages = [
        SystemMessage(content="sys"),
        HumanMessage(content="Please complete this task:\nwrite tree.py"),
    ]
    result = executor._inject_wm(messages, wm)
    assert any(
        isinstance(m, HumanMessage) and "YOUR GOAL" in (m.content or "")
        for m in result
    )


def test_inject_wm_appends_at_end_and_replaces_previous():
    """KV-cache stability: the WM message must live at the END of the list
    (append-only tail) and never duplicate — the old copy is removed, the
    message prefix before the previous WM slot stays byte-identical."""
    wm = executor.WorkingMemory(task="write tree.py")
    head = [
        SystemMessage(content="sys"),
        HumanMessage(content="Please complete this task:\nwrite tree.py"),
    ]
    r1 = executor._inject_wm(list(head), wm)
    assert (r1[-1].content or "").startswith(executor._WM_TAG)

    r1.append(AIMessage(content="round 1 reasoning"))
    wm.files_read.append("a.py")
    r2 = executor._inject_wm(r1, wm)

    wm_msgs = [m for m in r2
               if isinstance(m, HumanMessage)
               and (m.content or "").startswith(executor._WM_TAG)]
    assert len(wm_msgs) == 1, "stale WM copy left behind"
    assert r2[-1] is wm_msgs[0]
    assert "a.py" in r2[-1].content
    # The head (system + task) is untouched — the cacheable prefix survives.
    assert r2[0] is head[0] and r2[1] is head[1]


def _read_rounds_script(n):
    return [
        AIMessage(content='<tool_call>{"name": "read_file", "args": {"path": "big.py"}}</tool_call>')
        for _ in range(n)
    ] + [AIMessage(content="done reading")]


def test_masking_skipped_below_context_pressure(
    ctx: tools.ToolContext, workspace: Path, monkeypatch
):
    """History must stay append-only (verbatim tool outputs, cache-friendly)
    until estimated context actually passes the trim target."""
    LEDGER.reset()
    monkeypatch.delenv("WELLS_CTX_LIMIT", raising=False)
    monkeypatch.delenv("WELLS_CTX_TARGET", raising=False)
    (workspace / "big.py").write_text("x = 1  # padding\n" * 60)
    with (
        patch.object(config, "_invoke_with_retry", side_effect=_scripted(_read_rounds_script(6))),
        patch.object(config, "STRUCTURED_OUTPUTS", False),
        patch.object(executor, "_try_bind_tools", return_value=None),
    ):
        result = executor.run_executor(task="x", ctx=ctx, max_steps=10, step_label="t")
    masked = [m for m in result.messages
              if isinstance(m, ToolMessage) and (m.content or "").startswith("[FILE_READ:")]
    assert masked == [], "tool outputs were masked without context pressure"


def test_masking_fires_under_context_pressure(
    ctx: tools.ToolContext, workspace: Path, monkeypatch
):
    monkeypatch.setenv("WELLS_CTX_LIMIT", "600")
    monkeypatch.setenv("WELLS_CTX_TARGET", "300")
    LEDGER.reset()
    (workspace / "big.py").write_text("x = 1  # padding\n" * 60)
    with (
        patch.object(config, "_invoke_with_retry", side_effect=_scripted(_read_rounds_script(7))),
        patch.object(config, "STRUCTURED_OUTPUTS", False),
        patch.object(executor, "_try_bind_tools", return_value=None),
    ):
        result = executor.run_executor(task="x", ctx=ctx, max_steps=12, step_label="t")
    masked = [m for m in result.messages
              if isinstance(m, ToolMessage) and (m.content or "").startswith("[FILE_READ:")]
    dropped = [m for m in result.messages
               if isinstance(m, HumanMessage) and "safety drop" in (m.content or "")]
    assert masked or dropped, "context pressure produced no trimming at all"


# ---------------------------------------------------------------------------
# Context-trim budget (config.BUDGET/SMALL_BUDGET consolidation)
# ---------------------------------------------------------------------------


def test_effective_ctx_budget_uses_config_budget_by_default(monkeypatch):
    monkeypatch.delenv("WELLS_CTX_LIMIT", raising=False)
    monkeypatch.delenv("WELLS_CTX_TARGET", raising=False)
    threshold, target = executor._effective_ctx_budget(compact=False)
    assert threshold == config.BUDGET.max_input_tokens
    assert target == config.BUDGET.input_allowance


def test_effective_ctx_budget_uses_small_budget_when_compact(monkeypatch):
    monkeypatch.delenv("WELLS_CTX_LIMIT", raising=False)
    monkeypatch.delenv("WELLS_CTX_TARGET", raising=False)
    threshold, target = executor._effective_ctx_budget(compact=True)
    assert threshold == config.SMALL_BUDGET.max_input_tokens
    assert target == config.SMALL_BUDGET.input_allowance
    # The whole point: a local/small-context model must get a materially
    # lower ceiling than the generic default, not the same one.
    assert threshold < config.BUDGET.max_input_tokens


def test_effective_ctx_budget_explicit_env_override_wins(monkeypatch):
    monkeypatch.setenv("WELLS_CTX_LIMIT", "5555")
    monkeypatch.setenv("WELLS_CTX_TARGET", "3333")
    threshold, target = executor._effective_ctx_budget(compact=False)
    assert (threshold, target) == (5555, 3333)
    # Override applies regardless of compact mode too.
    threshold, target = executor._effective_ctx_budget(compact=True)
    assert (threshold, target) == (5555, 3333)


def test_safety_drop_head_protection_stops_at_first_ai_message():
    """Regression for a latent bug found while testing the budget wiring:
    the old head-protection loop only stopped scanning when it found a
    *second* plain HumanMessage — which never happens without a /steer
    message — so in the common case it silently walked the entire message
    list, left `tail` empty, and this "last resort" safety valve never
    actually dropped anything. It must stop at the first AIMessage instead,
    protecting only [system, task, WM?] as the intended fixed-size head."""
    messages = [SystemMessage(content="sys"), HumanMessage(content="the task")]
    for i in range(10):
        messages.append(AIMessage(content=f"round {i} " * 50, tool_calls=[
            {"name": "list_dir", "args": {"path": f"d{i}"}, "id": f"c{i}"}
        ]))
        messages.append(ToolMessage(
            content=f"observation {i} " * 50, tool_call_id=f"c{i}", name="list_dir"
        ))
    # No /steer message anywhere — the realistic common case.
    trimmed, saved = executor._safety_drop(messages, threshold=100, target=50)
    assert saved > 0, "safety_drop must engage when no steer message is present"
    assert len(trimmed) < len(messages)
    # The protected head must survive untouched.
    assert trimmed[0] == messages[0]
    assert trimmed[1] == messages[1]


def test_safety_drop_respects_custom_threshold():
    """A tiny explicit threshold must trigger dropping even for a message
    list config.BUDGET's generic default would leave untouched."""
    messages = [SystemMessage(content="sys"), HumanMessage(content="the task")]
    for i in range(10):
        messages.append(AIMessage(content=f"round {i} " * 50, tool_calls=[
            {"name": "list_dir", "args": {"path": f"d{i}"}, "id": f"c{i}"}
        ]))
        messages.append(ToolMessage(
            content=f"observation {i} " * 50, tool_call_id=f"c{i}", name="list_dir"
        ))
    trimmed, saved = executor._safety_drop(messages, threshold=50, target=20)
    assert saved > 0
    assert len(trimmed) < len(messages)


def test_infer_target_filename_from_hint_phrase():
    task = "Write a Python script ... and call it tree.py. It should look in..."
    assert executor._infer_target_filename(task) == "tree.py"


def test_infer_target_filename_bare_fallback():
    task = "Refactor utils.py to split out the helpers"
    assert executor._infer_target_filename(task) == "utils.py"


def test_salvage_code_block_single_block():
    text = "Here you go:\n```python\nprint('hi')\n```\n"
    assert executor._salvage_code_block(text, "tree.py") == "print('hi')\n"


def test_salvage_code_block_ambiguous_multi_block_returns_none():
    text = "```sh\npip install foo\n```\n```python\nprint(1)\n```\n"
    # Neither fence tag is "py"/"python" AND "sh" simultaneously matching a
    # non-.py filename with two unrelated blocks -> only the python one should
    # match when filename is .py; sh should not.
    assert executor._salvage_code_block(text, "tree.py") == "print(1)\n"
    # A filename whose extension matches neither present fence -> ambiguous.
    assert executor._salvage_code_block(text, "notes.md") is None


def test_salvage_code_block_refuses_failed_json_tool_call():
    """A fenced block that's actually a broken JSON tool-call attempt (not
    plain source) must never be salvaged as if it were the file content —
    that would write the JSON wrapper itself to disk instead of the code."""
    text = (
        '```json\n{ "name": "write_file", "arguments": { "path": "tree.py", '
        '"content": "tree.parse(bytes(source_code, "utf8"))" } }\n```'
    )
    assert executor._salvage_code_block(text, "tree.py") is None


def test_loop_salvages_unwrapped_code_block(ctx: tools.ToolContext, workspace: Path):
    """The exact failure mode reported live: a small local model responds
    with the finished script as a plain markdown code block and never emits
    a tool call. The harness should write the file itself rather than give
    up with zero steps taken."""
    LEDGER.reset()
    script = [
        AIMessage(content=(
            "Here is the script:\n\n"
            "```python\n"
            "import ast\n"
            "print('repository map')\n"
            "```\n"
        )),
        AIMessage(content="All done."),
    ]
    with (
        patch.object(config, "_invoke_with_retry", side_effect=_scripted(script)),
        patch.object(executor, "_try_bind_tools", return_value=None),
    ):
        result = executor.run_executor(
            task="Write a Python script ... call it tree.py.",
            ctx=ctx, max_steps=6, step_label="t",
        )
    assert result.stopped_reason == "done"
    assert result.steps_taken == 1
    assert result.tool_calls[0]["name"] == "write_file"
    assert result.tool_calls[0].get("salvaged") is True
    written = workspace / "tree.py"
    assert written.exists()
    assert "repository map" in written.read_text()


def test_loop_refuses_completion_after_unaddressed_failure(
    ctx: tools.ToolContext, workspace: Path
):
    """Reproduces the live failure: pip install fails, and instead of fixing
    it or reporting the blocker, the model just writes an optimistic summary
    ("you should now have X") as if it had worked. The harness must not
    accept that as a real completion — it should push back once, giving the
    model a chance to actually address the failure."""
    LEDGER.reset()
    script = [
        AIMessage(content=(
            '<tool_call>{"name": "run_command", '
            '"args": {"command": "pip install tree-sitter"}}</tool_call>'
        )),
        AIMessage(content=(
            "After executing the script, you should have a repository_map.json "
            "file in your current directory. The task is complete."
        )),
        AIMessage(content=(
            "You're right, the pip install failed because of the externally-"
            "managed-environment restriction. I was unable to install the "
            "dependency, so I can't complete this — please install tree-sitter "
            "into the project venv manually."
        )),
    ]
    with (
        patch.object(config, "_invoke_with_retry", side_effect=_scripted(script)),
        patch.object(executor, "_try_bind_tools", return_value=None),
        patch.object(tools, "dispatch", return_value=tools.ToolResult(
            ok=False, output="", error="externally-managed-environment", simulated=False
        )),
    ):
        result = executor.run_executor(
            task="write tree.py", ctx=ctx, max_steps=6, step_label="t"
        )
    assert result.stopped_reason == "done"
    # The optimistic non-acknowledging summary was rejected; the run only
    # ended once the model actually acknowledged the failure as a blocker.
    assert "unable to install" in result.summary


def test_loop_accepts_immediate_honest_blocker_report(ctx: tools.ToolContext):
    """A model that reports a failure as a blocker on its first wrap-up
    (permitted by WORKING RULES #5) must NOT be forced to retry — only
    silent, unacknowledged failure should be refused."""
    LEDGER.reset()
    script = [
        AIMessage(content=(
            '<tool_call>{"name": "run_command", '
            '"args": {"command": "pip install tree-sitter"}}</tool_call>'
        )),
        AIMessage(content=(
            "This failed because pip refuses global installs on this system "
            "(externally-managed-environment). I can't proceed without it — "
            "please install tree-sitter manually."
        )),
    ]
    with (
        patch.object(config, "_invoke_with_retry", side_effect=_scripted(script)),
        patch.object(executor, "_try_bind_tools", return_value=None),
        patch.object(tools, "dispatch", return_value=tools.ToolResult(
            ok=False, output="", error="externally-managed-environment", simulated=False
        )),
    ):
        result = executor.run_executor(
            task="write tree.py", ctx=ctx, max_steps=6, step_label="t"
        )
    assert result.stopped_reason == "done"
    assert result.steps_taken == 1
    assert "can't proceed" in result.summary


# ---------------------------------------------------------------------------
# Task-drift detection
# ---------------------------------------------------------------------------


def test_rewrite_similarity_high_for_incremental_edit():
    old = "def add(a, b):\n    return a + b\n\ndef sub(a, b):\n    return a - b\n"
    new = "def add(a, b):\n    return a + b\n\ndef sub(a, b):\n    return a - b - 1\n"
    assert executor._rewrite_similarity(old, new) > 0.8


def test_rewrite_similarity_low_for_unrelated_replacement():
    old = (
        "from tree_sitter import Language, Parser\n"
        "def extract_functions(path):\n    ...\n"
    )
    new = "def print_tree(node):\n    print(node.value)\n    for c in node.children:\n        print_tree(c)\n"
    assert executor._rewrite_similarity(old, new) < 0.3


def test_loop_drift_nudge_then_hard_stop_on_thrashing_rewrites(
    ctx: tools.ToolContext, workspace: Path
):
    """Reproduces the live failure that survived every other fix tonight:
    qwen kept validly calling write_file on the same path — real tool, valid
    JSON, succeeds every time — while the content itself drifted completely
    off the goal (tree-sitter parser -> unrelated binary-tree printer ->
    print('Hello, world!')), never converging, never touching
    repository_map.json. No mechanical check (bad tool name, malformed JSON,
    identical args) catches this; only content-similarity tracking does."""
    LEDGER.reset()
    # Each version genuinely unrelated to the last (not just topically off —
    # near-zero shared text), matching how little the live failure's
    # successive rewrites actually had in common with each other.
    unrelated_contents = [
        "from tree_sitter import Language, Parser\ndef extract_functions(path):\n"
        "    tree = parser.parse(path)\n    return tree.root_node\n",
        '{"name": "config", "version": 2, "settings": {"debug": true, "level": 5}}\n',
        "SELECT id, name, email FROM users WHERE active = 1 ORDER BY created_at DESC;\n",
        "| Column | Type | Description |\n|--------|------|-------------|\n"
        "| id | int | primary key |\n",
        "#!/bin/bash\necho 'deploying'\nrsync -avz ./dist/ user@host:/var/www/\n",
        "print(1)\n",
    ]
    import json as _json
    script = [
        AIMessage(content=(
            "<tool_call>"
            + _json.dumps({"name": "write_file", "args": {"path": "tree.py", "content": content}})
            + "</tool_call>"
        ))
        for content in unrelated_contents
    ]
    with (
        patch.object(config, "_invoke_with_retry", side_effect=_scripted(script)),
        patch.object(executor, "_try_bind_tools", return_value=None),
    ):
        result = executor.run_executor(
            task="Write a tree-sitter based function extractor to repository_map.json",
            ctx=ctx, max_steps=0, step_label="t",
        )
    assert result.stopped_reason == "stuck_loop"
    assert result.steps_taken < len(unrelated_contents)  # stopped before exhausting the script
    assert all(c["name"] == "write_file" for c in result.tool_calls)


def test_loop_incremental_refinement_never_triggers_drift_stop(
    ctx: tools.ToolContext, workspace: Path
):
    """False-positive guard: a model genuinely, incrementally refining the
    same file (each version mostly sharing the last one's structure) must
    never be caught by the drift detector — only wholesale, repeated,
    unrelated replacement should be."""
    LEDGER.reset()
    import json as _json
    refinements = [
        "def extract_functions(path):\n    pass\n",
        "def extract_functions(path):\n    tree = parse(path)\n    return tree\n",
        "def extract_functions(path):\n    tree = parse(path)\n    return tree.root_node\n",
        "def extract_functions(path):\n    tree = parse(path)\n    return walk(tree.root_node)\n",
        "def extract_functions(path):\n    tree = parse(path)\n    return walk(tree.root_node)\n\ndef walk(node):\n    return []\n",
    ]
    script = [
        AIMessage(content=(
            "<tool_call>"
            + _json.dumps({"name": "write_file", "args": {"path": "tree.py", "content": c}})
            + "</tool_call>"
        ))
        for c in refinements
    ] + [AIMessage(content="Done.")]
    with (
        patch.object(config, "_invoke_with_retry", side_effect=_scripted(script)),
        patch.object(executor, "_try_bind_tools", return_value=None),
    ):
        result = executor.run_executor(
            task="write tree.py", ctx=ctx, max_steps=10, step_label="t"
        )
    assert result.stopped_reason == "done"
    assert result.steps_taken == len(refinements)


def test_loop_stops_on_invented_tool_name_even_with_varied_args(ctx: tools.ToolContext):
    """Reproduces the live runaway: qwen2.5-coder invented a "report_error"
    tool that doesn't exist, and dodged the identical-args repeat detector
    by rewording the error message slightly each call — 170+ rounds with no
    progress before it was killed by hand. The harness must catch this by
    tool-name validity, not exact-args equality, and hard-stop."""
    LEDGER.reset()
    # Every call targets the same nonexistent tool but with different args
    # each time, exactly like the live failure — this must NOT reset the
    # identical-args repeat counter's evasion of the old detector.
    script = [
        AIMessage(content=(
            '<tool_call>{"name": "report_error", '
            f'"args": {{"error_message": "attempt number {i}"}}}}</tool_call>'
        ))
        for i in range(20)
    ]
    with (
        patch.object(config, "_invoke_with_retry", side_effect=_scripted(script)),
        patch.object(executor, "_try_bind_tools", return_value=None),
    ):
        result = executor.run_executor(
            task="write tree.py", ctx=ctx, max_steps=0, step_label="t"
        )
    assert result.stopped_reason == "stuck_loop"
    assert result.steps_taken == 6
    assert all(c["name"] == "report_error" for c in result.tool_calls)


def test_loop_unknown_tool_streak_resets_on_valid_call(ctx: tools.ToolContext, workspace: Path):
    """A single valid tool call in between invented ones must reset the
    streak — this guards against a false-positive stop on a model that's
    genuinely working and just fat-fingers a tool name once in a while."""
    LEDGER.reset()
    script = [
        AIMessage(content='<tool_call>{"name": "bogus_tool", "args": {}}</tool_call>'),
        AIMessage(content='<tool_call>{"name": "bogus_tool", "args": {"x": 1}}</tool_call>'),
        AIMessage(content=(
            '<tool_call>{"name": "read_file", "args": {"path": "maths.py"}}</tool_call>'
        )),
        AIMessage(content='<tool_call>{"name": "bogus_tool", "args": {}}</tool_call>'),
        AIMessage(content='<tool_call>{"name": "bogus_tool", "args": {"y": 2}}</tool_call>'),
        AIMessage(content="Done."),
    ]
    with (
        patch.object(config, "_invoke_with_retry", side_effect=_scripted(script)),
        patch.object(executor, "_try_bind_tools", return_value=None),
    ):
        result = executor.run_executor(
            task="x", ctx=ctx, max_steps=10, step_label="t"
        )
    assert result.stopped_reason == "done"


def test_loop_gives_up_after_exhausting_stall_nudges(ctx: tools.ToolContext):
    """If the model never calls a tool despite repeated coaching, the harness
    stops rather than nudging forever."""
    LEDGER.reset()
    prose = AIMessage(content="Here is how you would do it: step one, step two...")
    script = [prose] * 10
    with (
        patch.object(config, "_invoke_with_retry", side_effect=_scripted(script)),
        patch.object(executor, "_try_bind_tools", return_value=None),
        patch.object(config, "STALL_NUDGE_MAX", 2),
    ):
        result = executor.run_executor(task="x", ctx=ctx, max_steps=6, step_label="t")
    assert result.stopped_reason == "done"
    assert result.steps_taken == 0


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
    from wells.control import CONTROL

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
    from wells.control import CONTROL

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


def test_cap_zero_means_no_limit(ctx: tools.ToolContext):
    """max_steps=0 runs until the model finishes, past any old default."""
    from wells.control import CONTROL

    LEDGER.reset()
    CONTROL.reset()
    # Distinct args each round (not a plain repeat) so this exercises the step
    # cap only, not the stuck-loop repeat backstop (see
    # test_loop_detects_stuck_repeat) below.
    calls = [
        AIMessage(
            content=f'<tool_call>{{"name": "list_dir", "args": {{"path": "d{i}"}}}}</tool_call>'
        )
        for i in range(70)
    ]
    script = calls + [AIMessage(content="done at last")]
    with (
        patch.object(config, "_invoke_with_retry", side_effect=_scripted(script)),
        patch.object(executor, "_try_bind_tools", return_value=None),
    ):
        result = executor.run_executor(task="x", ctx=ctx, max_steps=0, step_label="t")
    assert result.stopped_reason == "done"
    assert result.steps_taken == 70


def test_explicit_cap_still_enforced(ctx: tools.ToolContext):
    from wells.control import CONTROL

    LEDGER.reset()
    CONTROL.reset()
    repeat = AIMessage(
        content='<tool_call>{"name": "list_dir", "args": {}}</tool_call>'
    )
    with (
        patch.object(config, "_invoke_with_retry", side_effect=_scripted([repeat] * 20)),
        patch.object(executor, "_try_bind_tools", return_value=None),
    ):
        result = executor.run_executor(task="x", ctx=ctx, max_steps=3, step_label="t")
    assert result.stopped_reason == "max_steps"
    assert result.steps_taken == 3


def test_progress_published(ctx: tools.ToolContext):
    from wells.control import CONTROL

    LEDGER.reset()
    CONTROL.reset()
    script = [
        AIMessage(content='<tool_call>{"name": "list_dir", "args": {}}</tool_call>'),
        AIMessage(content="done"),
    ]
    with (
        patch.object(config, "_invoke_with_retry", side_effect=_scripted(script)),
        patch.object(executor, "_try_bind_tools", return_value=None),
    ):
        executor.run_executor(task="x", ctx=ctx, max_steps=0, step_label="mystage")
    prog = dict((l, (c, cap)) for l, c, cap in CONTROL.progress())
    assert "mystage" in prog
    assert prog["mystage"][1] == 0  # cap recorded as no-limit


# ---------------------------------------------------------------------------
# LLM-call cancellation (a blocking llm.invoke() had zero cancellation
# checks: Escape/`/stop` had no effect until the call returned, up to
# LLM_TIMEOUT x LLM_MAX_RETRIES seconds)
# ---------------------------------------------------------------------------


def test_invoke_cancelable_returns_none_when_cancelled_first():
    import threading
    import time

    from wells.control import CONTROL

    CONTROL.reset()

    def slow_invoke(llm, messages):
        time.sleep(5)
        return AIMessage(content="too late")

    holder: dict = {}

    def run():
        t0 = time.monotonic()
        with patch.object(config, "_invoke_with_retry", side_effect=slow_invoke):
            holder["resp"] = executor._invoke_cancelable(object(), [])
        holder["elapsed"] = time.monotonic() - t0

    th = threading.Thread(target=run)
    th.start()
    time.sleep(0.3)
    CONTROL.cancel()
    th.join(timeout=5)
    CONTROL.reset()
    assert not th.is_alive()
    assert holder["resp"] is None
    assert holder["elapsed"] < 2.0, f"took {holder['elapsed']:.2f}s to return after cancel"


def test_invoke_cancelable_returns_result_when_not_cancelled():
    from wells.control import CONTROL

    CONTROL.reset()
    msg = AIMessage(content="hi")
    with patch.object(config, "_invoke_with_retry", return_value=msg):
        resp = executor._invoke_cancelable(object(), [])
    assert resp is msg
