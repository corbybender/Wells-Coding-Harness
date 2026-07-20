"""Agentic tool-calling executor loop (Layer 2).

This is the component that makes the harness *act*: given a task and a toolset,
it runs a model-driven loop of ``model → tool_calls → observe → model`` until
the task is done or the step cap is reached. It reuses the harness's token
accounting (:data:`LEDGER`) and the compressor for tool outputs.

Two calling conventions are supported so *any decent AI* can drive it:

  * **Native tool-calling** — models that support OpenAI/Anthropic-style
    ``tool_calls`` (Z.ai GLM, OpenAI, Claude, Gemini, …). We bind the tool
    schemas via ``model.bind_tools([...])`` and dispatch the returned calls.
  * **Text fallback** — for models without native tool-calling, the same tool
    schemas are described in the system prompt and the model emits calls as
    ``<tool_call>{json}</tool_call>`` blocks, which we parse and dispatch.

The loop auto-detects which mode each response uses, so a single harness run
works across providers without per-model wiring.

The executor is deliberately *not* a LangGraph node — it is a plain callable so
the coder/tester nodes (and subagents) can invoke it and feed its summary back
into the shared :class:`AgentState`.
"""

from __future__ import annotations

import difflib
import json
import re
from dataclasses import dataclass, field

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

import os
import platform
import shutil

from wells import config, tools
from wells.compress import compress_output
from wells.control import CONTROL, ui
from wells.tokens import LEDGER, estimate_tokens


# ---------------------------------------------------------------------------
# Environment detection — computed once at import, injected into every prompt
# ---------------------------------------------------------------------------

_ENV_CONTEXT_CACHE: str | None = None


def _build_env_context() -> str:
    """Return a one-time snapshot of the execution environment.

    Uses only PATH lookups (shutil.which) — no subprocesses — so it is fast
    and safe to call at import time.
    """
    global _ENV_CONTEXT_CACHE
    if _ENV_CONTEXT_CACHE is not None:
        return _ENV_CONTEXT_CACHE

    sys_name = platform.system()  # 'Windows' | 'Linux' | 'Darwin'

    if sys_name == "Darwin":
        os_str = f"macOS {platform.mac_ver()[0] or platform.release()}"
    elif sys_name == "Windows":
        os_str = f"Windows {platform.release()}"
    else:
        # Linux — try /etc/os-release for a friendlier name
        try:
            with open("/etc/os-release") as fh:
                info = dict(line.strip().split("=", 1) for line in fh if "=" in line)
            pretty = info.get("PRETTY_NAME", "").strip('"')
            os_str = pretty or f"Linux {platform.release()}"
        except Exception:
            os_str = f"Linux {platform.release()}"

    # Shell
    if sys_name == "Windows":
        if shutil.which("pwsh"):
            shell = "PowerShell (pwsh)"
        elif shutil.which("powershell"):
            shell = "Windows PowerShell"
        else:
            shell = "cmd.exe"
    else:
        shell = os.environ.get("SHELL", "/bin/sh")

    # CLI tool availability — PATH lookup only, no subprocess overhead
    candidates = [
        "git",
        "az",
        "aws",
        "gcloud",
        "docker",
        "kubectl",
        "helm",
        "terraform",
        "npm",
        "node",
        "bun",
        "deno",
        "python",
        "python3",
        "pip",
        "uv",
        "cargo",
        "rustc",
        "go",
        "java",
        "mvn",
        "gradle",
        "dotnet",
        "curl",
        "wget",
        "gh",
        "hub",
        "make",
        "cmake",
        "jq",
        "yq",
    ]
    available = [t for t in candidates if shutil.which(t)]

    lines = [
        f"OS      : {os_str}",
        f"Shell   : {shell}",
        f"Tools   : {', '.join(available) if available else '(none detected in PATH)'}",
    ]
    _ENV_CONTEXT_CACHE = "\n".join(lines)
    return _ENV_CONTEXT_CACHE


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------


@dataclass
class ExecutorResult:
    """Outcome of one executor run."""

    summary: str  # final natural-language answer from the model
    steps_taken: int = 0  # number of tool-call rounds executed
    tool_calls: list[dict] = field(
        default_factory=list
    )  # [{name, args, ok, output_preview}]
    stopped_reason: str = (
        "done"  # done | max_steps | error | cancelled | budget | stuck_loop
    )
    messages: list[BaseMessage] = field(default_factory=list)
    # True when the final answer was already streamed to the console live
    # (callers should not print result.summary again).
    streamed: bool = False
    # Per-LLM-call usage log for cache-efficiency analysis. Populated by
    # _account_usage; persisted into the run trace by traces.record_run so
    # `wells analyze` can report where the prompt cache breaks. Each entry:
    # {round, input, output, reasoning, cache_read, cache_creation,
    # mask_saved, drop_saved, ctx_tokens_at_call, model}.
    usage_log: list[dict] = field(default_factory=list)


# ---------------------------------------------------------------------------
# System prompt construction
# ---------------------------------------------------------------------------


def _tool_catalog(toolset: list[tools.ToolDef]) -> str:
    """Human-readable tool catalog for the text-fallback system prompt."""
    lines = []
    for t in toolset:
        params = ", ".join(
            f"{k}: {v.get('type', 'string')}"
            for k, v in t.input_schema.get("properties", {}).items()
        )
        req = ", ".join(t.input_schema.get("required", [])) or "—"
        lines.append(f"- {t.name}({params}) [required: {req}]\n    {t.description}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Structured outputs (grammar-constrained tool calls)
# ---------------------------------------------------------------------------
#
# For local Ollama profiles, the model's reply can be constrained to a
# tool-call JSON schema at the token-sampling level: Ollama's native API via
# the "format" field, its OpenAI-compatible shim via response_format
# json_schema (both since Ollama 0.5). Under the grammar the sampler cannot
# emit malformed JSON, prose mixed with a call, an unescaped inner quote, or
# an invented tool name — every failure class the text parsers and salvage
# paths below exist to mop up dies at the source. Those paths remain for
# providers without schema support and as the fallback when a server rejects
# the format.
#
# The schema is deliberately flat and loose ({name: enum, args: object})
# rather than a strict per-tool anyOf: small models follow a simple grammar
# far more reliably, arg validation already happens at dispatch, and a giant
# union schema slows llama.cpp's grammar compilation. "final_answer" is a
# pseudo-tool: with the grammar active the model *cannot* answer in prose, so
# it needs an in-schema way to say "done".

_FINAL_ANSWER_TOOL = "final_answer"


def _structured_output_schema(toolset: list[tools.ToolDef]) -> dict:
    """JSON schema constraining a reply to {"name": <known tool>, "args": {...}}."""
    names = sorted({t.name for t in toolset} | {_FINAL_ANSWER_TOOL})
    return {
        "type": "object",
        "properties": {
            "name": {"type": "string", "enum": names},
            "args": {"type": "object"},
        },
        "required": ["name", "args"],
    }


def _try_bind_structured(llm, profile, toolset: list[tools.ToolDef]):
    """Bind the tool-call schema as the enforced response format, or None.

    Only for profiles that actually talk to a local Ollama server — cloud
    providers get native bind_tools, which is strictly better when real.
    """
    if profile is None:
        return None
    try:
        if not config.providers._looks_like_local_ollama(profile):
            return None
        if profile.kind == "ollama":
            # langchain-ollama: `format` accepts a JSON schema dict.
            return llm.bind(format=_structured_output_schema(toolset))
        # OpenAI-compat shim (ChatOpenAI pointed at :11434/v1).
        return llm.bind(
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "wells_tool_call",
                    "schema": _structured_output_schema(toolset),
                },
            }
        )
    except Exception:
        return None


def _resolve_structured_reply(
    calls: list[dict], llm_text: str
) -> tuple[list[dict], str]:
    """Post-process calls parsed from a structured-mode reply.

    - ``final_answer`` pseudo-calls become the final narrative text (the
      grammar leaves the model no other way to finish).
    - When the reply parsed into real calls, the raw JSON is not narrative —
      blank it so the UI/summary paths don't show the wire format.
    """
    real = [c for c in calls if c.get("name") != _FINAL_ANSWER_TOOL]
    finals = [c for c in calls if c.get("name") == _FINAL_ANSWER_TOOL]
    if finals:
        fa = finals[0].get("args") or {}
        text = str(
            fa.get("summary") or fa.get("answer") or fa.get("text") or ""
        ).strip()
        if text:
            llm_text = text
    elif real and llm_text and _looks_like_tool_call_attempt(llm_text):
        llm_text = ""
    return real, llm_text


def _system_prompt(
    task: str,
    toolset: list[tools.ToolDef],
    *,
    plan_mode: bool,
    workspace: str | None = None,
    compact: bool = False,
    structured: bool = False,
) -> str:
    catalog = _tool_catalog(toolset)
    plan_note = (
        "\n\nIMPORTANT: You are in PLAN MODE. Do NOT make changes. Use read-only tools "
        "(read_file, list_dir, glob, grep) to investigate, then describe exactly what "
        "changes you WOULD make. Write/edit/run tools will simulate."
        if plan_mode
        else ""
    )
    # The harness operating principles (AGENT.md) are always prepended so every
    # executor run is governed by the constitution, regardless of model.
    from wells.principles import inject_into_prompt as inject_principles

    env = _build_env_context()

    # Workspace operating rules (RULES.md) + any open liabilities. The
    # machine-checkable subset is ALSO enforced at the tool boundary — this
    # block is the always-visible layer.
    rules_block = ""
    try:
        from wells import rules as _rules

        if workspace:
            rules_block = _rules.engine_for(workspace).prompt_block(compact=compact)
    except Exception:
        pass
    structured_note = (
        "\n\nOUTPUT FORMAT — STRUCTURED MODE (enforced by the runtime):\n"
        "Your ENTIRE reply must be exactly ONE JSON object of the form\n"
        '{"name": "<tool>", "args": {...}}\n'
        "— no prose, no markdown, no code fences, nothing outside the object.\n"
        "One tool call per reply; you will see its result before your next call.\n"
        "When (and only when) the task is genuinely complete, finish with:\n"
        '{"name": "final_answer", "args": {"summary": "<what you did and verified>"}}\n'
        if structured
        else ""
    )
    # Small models follow a concrete demonstration far more reliably than an
    # abstract format description — two short worked examples cost ~100 tokens
    # and measurably cut protocol violations, so compact mode (which exists
    # for exactly those models) includes them even while trimming everything
    # else. Shaped to whichever format this run actually expects.
    examples_note = ""
    if compact:
        if structured:
            examples_note = (
                "\n\nEXAMPLES — copy this format exactly:\n"
                "To inspect a file, reply with exactly:\n"
                '{"name": "read_file", "args": {"path": "src/app.py"}}\n'
                "To create or overwrite a file, reply with exactly:\n"
                '{"name": "write_file", "args": {"path": "hello.py", '
                '"content": "print(\'hi\')\\n"}}\n'
            )
        else:
            examples_note = (
                "\n\nEXAMPLES — copy this format exactly:\n"
                "To inspect a file, emit:\n"
                '<tool_call>{"name": "read_file", "args": {"path": "src/app.py"}}</tool_call>\n'
                "To create or overwrite a file, emit:\n"
                '<tool_call>{"name": "write_file", "args": {"path": "hello.py", '
                '"content": "print(\'hi\')\\n"}}</tool_call>\n'
            )
    base = f"""You are an autonomous software engineering agent working inside a real code repository.
You operate by calling tools to read files, search code, make edits, and run commands/tests,
then observing the results, until the task is complete.

ENVIRONMENT:
{env}

The tools listed under "Tools" above reflect actual PATH lookups on this machine.
Do NOT pre-emptively claim a tool is missing — if it appears in the list it is
available. If you are unsure, run it via run_command and observe the actual output.

TASK:
{task}

AVAILABLE TOOLS:
{catalog}
{plan_note}
{rules_block}
WORKING RULES:
1. Index-first lookup: when you need to find where a function/class/variable is defined,
   call find_symbol(name) first. It returns the exact file:line instantly. Only fall back
   to grep or read when the symbol isn't in the index or you need surrounding context.
2. No re-reading: if you have already read a file, do NOT read it again. Check what you
   already know from prior tool results; use grep with a specific pattern if you need
   one more piece of info from that file.
3. Investigate before acting: read/list to understand structure, then make focused changes.
4. After each edit, verify (re-read the changed section, run tests/lint) then stop.
5. If you cannot complete the task after reasonable effort, stop and explain the blocker.

TOOL CALLING — MANDATORY:
- You act ONLY by calling tools. Plain prose describing what you would do, could do, or
  are about to do accomplishes NOTHING — the harness cannot execute a sentence. If the
  task requires a file to exist, a command to run, or code to change, you MUST call a
  tool; do not describe the steps instead of taking them.
- Regardless of whether your runtime also supports native/structured function calls,
  ALWAYS additionally emit each call on its own line as:
  <tool_call>{{"name": "...", "args": {{...}}}}</tool_call>
  and nothing else on that line. The harness parses this exact tag from your text on
  every turn, independent of native tool-call support, so it works no matter what your
  runtime does under the hood. Example, to create a file:
  <tool_call>{{"name": "write_file", "args": {{"path": "example.py", "content": "print(1)\\n"}}}}</tool_call>
- If your training makes that tag awkward, a bare JSON object as your entire reply,
  {{"name": "...", "arguments": {{...}}}}, is also recognized — but never mix that
  with explanatory prose in the same reply, or neither will parse.
- Batch related shell commands: chain multiple az/git/curl calls with semicolons in ONE
  run_command call rather than calling run_command once per command. This cuts round trips
  and token cost dramatically.
- The harness injects REQUESTS_CA_BUNDLE / SSL_CERT_FILE / CURL_CA_BUNDLE automatically
  into every subprocess. Do NOT prepend $env:REQUESTS_CA_BUNDLE=... yourself — it's already
  set. If a command fails with an SSL/cert error, report it rather than retrying manually.
- If the same operation has failed 3+ times with similar errors, STOP and report what is
  blocking you rather than continuing to retry.{structured_note}{examples_note}
"""
    # Principles always first (the constitution), then the skills index (so the
    # model knows which load_skill calls are available, without paying for the
    # full bodies up front — progressive disclosure). Skills are an advanced,
    # optional capability — skipped in compact mode to save the (small) cost
    # for a model that's already struggling to fit the essentials in its
    # context window.
    out = inject_principles(base, workspace)
    if not compact:
        from wells import skills as _skills

        out = _skills.inject_into_prompt(out, workspace)
    return out


# ---------------------------------------------------------------------------
# Tool-call parsing (text fallback)
# ---------------------------------------------------------------------------

_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)
_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*\n?(\{.*?\})\s*```", re.DOTALL)
_LOOKS_LIKE_TOOL_CALL_RE = re.compile(
    r'"name"\s*:\s*"[^"]+"\s*,\s*"(?:args|arguments)"\s*:'
)


def _as_call(obj: dict) -> dict | None:
    name = obj.get("name")
    if not name or ("args" not in obj and "arguments" not in obj):
        return None
    return {"name": name, "args": obj.get("args") or obj.get("arguments") or {}}


def _looks_like_tool_call_attempt(blob: str) -> bool:
    """True when text has the shape of an attempted JSON tool call.

    Used to distinguish "this is a malformed tool call — tell the model
    what broke" from "this is unrelated JSON in prose — ignore it". A model
    embedding real source code as a JSON string value commonly forgets to
    escape an inner double quote (e.g. ``"utf8"`` inside Python code), which
    breaks strict JSON parsing; that failure must not be treated the same as
    "no call was attempted at all" (which is what silently continuing here
    used to do — the caller would then fall through to code-block salvage
    and could write the broken JSON wrapper itself to disk as if it were
    the intended file content).
    """
    return bool(_LOOKS_LIKE_TOOL_CALL_RE.search(blob))


def parse_text_tool_calls(text: str) -> list[dict]:
    """Parse tool calls out of a model's raw text reply.

    Recognizes two shapes:
    - ``<tool_call>{...}</tool_call>`` — the format this harness asks for.
    - A bare ``{"name": ..., "arguments": {...}}`` object, whether it's the
      entire reply or fenced in a ```json block — the native function-call
      format many tool-tuned open models (Qwen, Hermes, ...) actually emit.
      Some local runtimes (e.g. Ollama, depending on model/template) put this
      straight in the text content instead of a structured tool_calls field,
      so without this the harness would reject a call the model got right.

    Returns a list of ``{"name": ..., "args": {...}}`` dicts. Malformed
    ``<tool_call>`` blocks are reported back as parse errors; unrecognized
    bare JSON is silently ignored (it's probably just prose, not a call).
    """
    calls: list[dict] = []
    for m in _TOOL_CALL_RE.finditer(text):
        blob = m.group(1).strip()
        try:
            obj = json.loads(blob)
        except json.JSONDecodeError:
            calls.append({"_parse_error": blob})
            continue
        name = obj.get("name")
        args = obj.get("args") or obj.get("arguments") or {}
        if name:
            calls.append({"name": name, "args": args})
    if calls:
        return calls

    # Fenced extractions are tried before the raw whole text: the whole text
    # always fails json.loads() when the reply has fence markers or leading
    # prose around it, so trying it first would make every well-formed
    # ```json call look like a parse failure before the clean extraction
    # (which would actually parse) ever gets a turn.
    candidates = [m.group(1).strip() for m in _JSON_FENCE_RE.finditer(text)]
    candidates.append(text.strip())
    best_failed_attempt: str | None = None
    for blob in candidates:
        if not blob:
            continue
        try:
            obj = json.loads(blob)
        except json.JSONDecodeError:
            if best_failed_attempt is None and _looks_like_tool_call_attempt(blob):
                best_failed_attempt = blob
            continue
        for item in obj if isinstance(obj, list) else [obj]:
            if isinstance(item, dict):
                call = _as_call(item)
                if call:
                    calls.append(call)
        if calls:
            return calls
    if best_failed_attempt is not None:
        # A real attempt that failed to parse on every candidate — most
        # commonly an unescaped quote from embedded source code. Report it
        # as a parse error (like a malformed <tool_call> tag) instead of
        # silently moving on, so the caller doesn't mistake "tried and
        # failed" for "never tried" and salvage the broken JSON itself as
        # if it were the intended file content.
        calls.append({"_parse_error": best_failed_attempt})
    return calls


# ---------------------------------------------------------------------------
# Code-block salvage — last resort for models that ignore the tool-call
# protocol entirely and just answer with the file content as a markdown
# code block (a common habit for small/local chat-tuned models). Rather than
# discard real, usable output because the model didn't wrap it correctly,
# the harness writes it itself when it can confidently infer the target path.
# ---------------------------------------------------------------------------

_FENCE_RE = re.compile(r"```([A-Za-z0-9_+-]*)\r?\n(.*?)```", re.DOTALL)

_FILENAME_HINT_RE = re.compile(
    r"(?:call(?:ed)?|name[d]?|save(?:d)?\s+(?:it\s+)?as|written?\s+to|"
    r"output(?:s)?\s+to|creat(?:e|ing)?\s+(?:a\s+file\s+)?(?:called|named)?)"
    r"\s*[`\"']?([\w./\\-]+\.[A-Za-z0-9]{1,6})[`\"']?",
    re.IGNORECASE,
)
_BARE_FILENAME_RE = re.compile(
    r"\b[\w-]+\.(?:py|ts|tsx|js|jsx|json|md|txt|sh|ps1|yaml|yml|toml|go|rs|"
    r"java|c|cpp|h|hpp|rb|php|sql|css|html)\b",
    re.IGNORECASE,
)
_LANG_EXT_ALIASES = {
    "py": {"python", "py"},
    "js": {"javascript", "js"},
    "ts": {"typescript", "ts"},
    "sh": {"sh", "bash", "shell"},
    "ps1": {"powershell", "ps1"},
    "yml": {"yaml", "yml"},
    "yaml": {"yaml", "yml"},
}


def _infer_target_filename(task: str) -> str | None:
    """Best-effort guess at the file the task asked to be written.

    Looks for an explicit hint phrase first ("call it X", "save as X"), then
    falls back to any bare token with a known code/text extension.
    """
    m = _FILENAME_HINT_RE.search(task)
    if m:
        return m.group(1)
    m = _BARE_FILENAME_RE.search(task)
    if m:
        return m.group(0)
    return None


def _salvage_code_block(llm_text: str, filename: str) -> str | None:
    """Pull the file content out of the model's prose reply, if unambiguous.

    Returns None (no salvage) when there's no fenced code, when there are
    multiple blocks and none can be confidently matched to ``filename`` by
    its fence language tag (guessing wrong would silently write garbage), or
    when the sole candidate block is itself a failed JSON tool-call attempt
    (e.g. broken by an unescaped quote inside embedded source) rather than
    plain source — writing that wrapper verbatim as "the file" is worse than
    not writing anything; parse_text_tool_calls surfaces it as a parse error
    instead so the model can see and fix its own mistake.
    """
    blocks = _FENCE_RE.findall(llm_text)
    if not blocks:
        return None
    if len(blocks) == 1:
        body = blocks[0][1]
        return None if _looks_like_tool_call_attempt(body) else body
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    wanted = _LANG_EXT_ALIASES.get(ext, {ext} if ext else set())
    matches = [body for lang, body in blocks if lang.lower() in wanted]
    if len(matches) != 1:
        return None
    return None if _looks_like_tool_call_attempt(matches[0]) else matches[0]


def _try_salvage_write(task: str, llm_text: str) -> tuple[str, str] | None:
    """Return (path, content) to auto-write, or None if salvage isn't safe."""
    filename = _infer_target_filename(task)
    if not filename:
        return None
    code = _salvage_code_block(llm_text, filename)
    if not code or not code.strip():
        return None
    return filename, code


# ---------------------------------------------------------------------------
# Token accounting helper
# ---------------------------------------------------------------------------


def _message_text(m: BaseMessage) -> str:
    """Plain text out of a message's content, whether it's a string or a
    multimodal content-block list (image attachments) — image blocks
    contribute no text (their token cost isn't estimable this way; usage_
    metadata from the provider is preferred whenever available)."""
    content = getattr(m, "content", "") or ""
    if isinstance(content, list):
        return " ".join(
            b.get("text", "")
            for b in content
            if isinstance(b, dict) and b.get("type") == "text"
        )
    return content


def _account_usage(
    *,
    step: str,
    model: str,
    messages: list[BaseMessage],
    resp: BaseMessage,
    saved_by_trim: int = 0,
    usage_log: list[dict] | None = None,
    round_num: int = 0,
    mask_saved: int = 0,
    drop_saved: int = 0,
) -> None:
    """Record token usage for one executor round into the global ledger.

    Also appends a per-call entry to ``usage_log`` when supplied, so the run
    trace can later report where the prompt cache broke (round-by-round
    cache_read delta) and which context-management ops fired.
    """
    um = getattr(resp, "usage_metadata", None) or {}
    full = "\n".join(_message_text(m) for m in messages)
    input_tokens = um.get("input_tokens") or estimate_tokens(full)
    output_tokens = um.get("output_tokens") or estimate_tokens(
        getattr(resp, "content", "") or ""
    )
    reasoning = ((um.get("output_token_details") or {}).get("reasoning")) or 0
    cache_read = ((um.get("input_token_details") or {}).get("cache_read")) or 0
    cache_creation = ((um.get("input_token_details") or {}).get("cache_creation")) or 0
    LEDGER.record(
        step=step,
        task_type="executor",
        model=model,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        reasoning_tokens=reasoning,
        cache_read_tokens=cache_read,
        category_tokens={"executor_input": input_tokens},
        saved_by_trim=saved_by_trim,
    )
    if usage_log is not None:
        usage_log.append(
            {
                "round": round_num,
                "model": model,
                "input": int(input_tokens),
                "output": int(output_tokens),
                "reasoning": int(reasoning),
                "cache_read": int(cache_read),
                "cache_creation": int(cache_creation),
                "mask_saved": int(mask_saved),
                "drop_saved": int(drop_saved),
                # Estimated prompt size at this call — useful for correlating
                # cache breaks with context-management events.
                "ctx_tokens_at_call": estimate_tokens(full),
            }
        )
    reasoning = ((um.get("output_token_details") or {}).get("reasoning")) or 0
    cache_read = ((um.get("input_token_details") or {}).get("cache_read")) or 0
    LEDGER.record(
        step=step,
        task_type="executor",
        model=model,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        reasoning_tokens=reasoning,
        cache_read_tokens=cache_read,
        category_tokens={"executor_input": input_tokens},
        saved_by_trim=saved_by_trim,
    )


# ---------------------------------------------------------------------------
# Working memory — structural task state maintained across tool calls
# ---------------------------------------------------------------------------

_WM_TAG = "<working_memory>"  # prefix that identifies the WM HumanMessage in the list


@dataclass
class WorkingMemory:
    """Compact structured state updated after every tool call and injected into
    every LLM request.  Never pruned — it IS the compressed truth of the run.

    Prevents the costliest agent failure modes:
    - Re-reading files already processed
    - Re-attempting approaches that already failed
    - Forgetting test state between rounds
    """

    task: str = ""
    files_modified: list[str] = field(default_factory=list)
    files_read: list[str] = field(default_factory=list)
    failed_commands: list[str] = field(default_factory=list)
    test_status: str = ""
    open_liabilities: str = ""  # undischarged rule obligations (never pruned)

    def update_from_tool(
        self, name: str, args: dict, result_text: str, ok: bool
    ) -> None:
        path = (
            args.get("path")
            or args.get("file_path")
            or args.get("filename")
            or args.get("filepath")
            or ""
        )
        if name in {"read_file", "view_file", "cat"} and path:
            if path not in self.files_read:
                self.files_read.append(path)
        elif (
            name
            in {
                "write_file",
                "edit_file",
                "patch_file",
                "str_replace_editor",
                "str_replace",
                "create_file",
            }
            and path
            and ok
        ):
            if path not in self.files_modified:
                self.files_modified.append(path)
            if path not in self.files_read:
                self.files_read.append(path)
        elif name in {"run_tests", "pytest", "test"}:
            lines = [l.strip() for l in result_text.splitlines() if l.strip()]
            for line in lines:
                low = line.lower()
                if any(
                    k in low for k in ("passed", "failed", "error", "ok", "tests ran")
                ):
                    self.test_status = line[:200]
                    break
            else:
                self.test_status = lines[0][:200] if lines else ""
        elif name in {"run_command", "bash", "shell", "run_code"} and not ok:
            cmd = str(args.get("command") or args.get("code", ""))[:60]
            err_lines = result_text.strip().splitlines()
            err = err_lines[-1][:80] if err_lines else ""
            entry = f"{cmd} → {err}"
            if entry not in self.failed_commands:
                self.failed_commands.append(entry)

    def is_empty(self) -> bool:
        # `task` counts as non-empty content: without it, WorkingMemory is
        # never injected until the first tool call succeeds (nothing else is
        # populated yet on round 1), so a model that's about to drift off
        # the goal from its very first move gets no re-anchor at all until
        # it's already one step down the wrong path.
        return not any(
            [
                self.task,
                self.files_modified,
                self.files_read,
                self.failed_commands,
                self.test_status,
                self.open_liabilities,
            ]
        )

    def to_xml(self) -> str:
        parts = [_WM_TAG]
        if self.task:
            # Loud, unmissable framing rather than a small XML tag buried
            # among other fields — observed live, a model can keep making
            # mechanically valid tool calls (real tool, real args, real
            # success) while drifting completely off the actual goal over
            # many rounds; a plain <task> tag it's easy to skim past every
            # round didn't stop that. This doesn't detect drift (see the
            # dedicated drift-detection work), it just makes the goal harder
            # to lose track of in the first place, for near-zero cost.
            parts.append(
                "  ══ YOUR GOAL — do not drift from this, no matter how "
                "many rounds have passed ══\n"
                f"  <task>{self.task[:500]}</task>"
            )
        if self.files_modified:
            parts.append(
                f"  <files_modified>{', '.join(self.files_modified[-12:])}</files_modified>"
            )
        # Only list read-only files (not ones already in files_modified — redundant)
        read_only = [f for f in self.files_read if f not in self.files_modified]
        if read_only:
            parts.append(f"  <files_read>{', '.join(read_only[-12:])}</files_read>")
        if self.failed_commands:
            parts.append("  <failed_approaches>")
            for fc in self.failed_commands[-5:]:
                parts.append(f"    - {fc}")
            parts.append("  </failed_approaches>")
        if self.test_status:
            parts.append(f"  <test_status>{self.test_status}</test_status>")
        if self.open_liabilities:
            parts.append(
                "  <open_liabilities>MUST be discharged before the task is "
                f"complete: {self.open_liabilities}</open_liabilities>"
            )
        parts.append("</working_memory>")
        return "\n".join(parts)


def _is_wm_message(m: BaseMessage) -> bool:
    """True for the working-memory HumanMessage — never a multimodal one
    (WM content is always a plain string), so list content (an image-
    attached seed message) safely short-circuits to False here."""
    if not isinstance(m, HumanMessage):
        return False
    content = m.content
    return isinstance(content, str) and content.startswith(_WM_TAG)


def _inject_wm(messages: list[BaseMessage], wm: WorkingMemory) -> list[BaseMessage]:
    """Remove the previous working-memory HumanMessage and append a fresh one
    at the END of the list (never pruned).

    Position matters for two reasons:
    - **KV-cache stability.** Both llama.cpp/Ollama's prompt cache and cloud
      prompt caching reuse computation only for an unchanged prefix. The WM
      content changes every round; when it lived near the head (right after
      the task message, its original position), every round invalidated the
      cache from message ~2 onward and the entire history was re-prefilled
      on every call — seconds to minutes per round on local hardware. At the
      end, the invalidation point is the *previous* round's WM slot, so the
      whole earlier transcript stays cached.
    - **Recency.** Models attend most reliably to the end of the context —
      the goal anchor lands where a drifting model is most likely to see it.
    """
    if wm.is_empty():
        return messages
    result = [m for m in messages if not _is_wm_message(m)]
    result.append(HumanMessage(content=wm.to_xml()))
    return result


# ---------------------------------------------------------------------------
# Task-drift detection
# ---------------------------------------------------------------------------
#
# Every other backstop in this module catches a model doing something
# mechanically wrong (bad tool name, malformed JSON, identical repeat). None
# of them catch a model doing everything mechanically *right* — real tool,
# valid args, succeeds — while the actual content drifts off the goal.
# Observed live: a 7B model asked to write a tree-sitter-based function
# extractor rewrote the same file's full content across several rounds as a
# tree-sitter parser, then an unrelated binary-tree printer, then a
# height-based tree-printer, then degenerated to alternating `print(1)` /
# `print('Hello, world!')` — every write_file call individually valid, zero
# progress toward the goal.
#
# Signal: successive full-file rewrites of the SAME path via write_file that
# share very little text with the immediately previous version. A model
# incrementally building toward a goal keeps most of its own prior structure
# between writes; one that's thrashing/free-associating tends to replace the
# whole thing with something largely unrelated to what it just wrote,
# repeatedly. difflib.SequenceMatcher's ratio() is a cheap, stdlib-only,
# dependency-free similarity measure — good enough for a coarse "is this even
# related to what I wrote last time" signal; it isn't trying to judge
# relevance to the *task*, just to itself.

_DRIFT_SIMILARITY_THRESHOLD = float(
    __import__("os").environ.get("WELLS_DRIFT_SIMILARITY", "0.3")
)
_DRIFT_NUDGE_AT = 2  # consecutive low-similarity rewrites before nudging
_DRIFT_STOP_AT = 4  # consecutive low-similarity rewrites before hard stop


def _rewrite_similarity(old: str, new: str) -> float:
    """0..1 text similarity between two versions of the same file's content."""
    if not old and not new:
        return 1.0
    return difflib.SequenceMatcher(None, old, new).ratio()


# ---------------------------------------------------------------------------
# Observation masking (primary context management)
# ---------------------------------------------------------------------------

# Keep this many most-recent AI+Tool rounds verbatim; replace content in older ones.
_MASK_KEEP_ROUNDS = int(__import__("os").environ.get("WELLS_KEEP_ROUNDS", "4"))
# Only mask tool outputs larger than this many estimated tokens (small ones are cheap).
_MASK_MIN_TOKENS = int(__import__("os").environ.get("WELLS_MASK_MIN", "120"))
# Fallback drop threshold/target — used only if neither an explicit
# WELLS_CTX_LIMIT/TARGET override nor config.BUDGET is available (should not
# happen in practice; config always loads). See _effective_ctx_budget().
_DROP_THRESHOLD_DEFAULT = 18000
_DROP_TARGET_DEFAULT = 12000


def _effective_ctx_budget(compact: bool) -> tuple[int, int]:
    """Resolve the safety-drop (threshold, target) pair.

    Historically these were hardcoded module constants driven by their own
    undocumented env vars (WELLS_CTX_LIMIT/WELLS_CTX_TARGET), completely
    disconnected from config.BUDGET/SMALL_BUDGET — the knobs actually shown
    in `wells info` and documented for users. Tuning the documented knob had
    zero effect on what was actually enforced. This makes config.BUDGET the
    real source of truth: WELLS_CTX_LIMIT/TARGET remain available as an
    explicit low-level override (back-compat) when BOTH are set, otherwise
    the threshold/target are derived from config.BUDGET (or config.
    SMALL_BUDGET for a local/small-context profile, ``compact=True`` —
    same signal already used to shrink the system prompt) so a model with a
    small context window gets a matching, much lower trim ceiling instead
    of the generic 24000-token default.
    """
    limit_override = __import__("os").environ.get("WELLS_CTX_LIMIT")
    target_override = __import__("os").environ.get("WELLS_CTX_TARGET")
    if limit_override and target_override:
        try:
            return int(limit_override), int(target_override)
        except ValueError:
            pass
    budget = config.SMALL_BUDGET if compact else config.BUDGET
    if budget is None:
        return _DROP_THRESHOLD_DEFAULT, _DROP_TARGET_DEFAULT
    return budget.max_input_tokens, budget.input_allowance


def _ctx_tokens(messages: list[BaseMessage]) -> int:
    """Rough token estimate for the full message list."""
    total = 0
    for m in messages:
        total += estimate_tokens(_message_text(m))
        for tc in getattr(m, "tool_calls", None) or []:
            total += estimate_tokens(str(tc.get("args", {})))
    return total


def _mask_tool_result(name: str, args: dict, content: str) -> str:
    """Compress a large tool result to a typed 1-line summary.

    Type-aware so each summary carries the most useful signal for its tool kind.
    Index tools (find_symbol, etc.) are never masked — their output is already compact.
    """
    path = (
        args.get("path")
        or args.get("file_path")
        or args.get("filename")
        or args.get("pattern")
        or ""
    )
    lines = [l for l in content.splitlines() if l.strip()]
    n = len(lines)

    if name in {"read_file", "view_file", "cat"}:
        return f"[FILE_READ: {path} — {n} lines, content processed]"
    elif name in {"list_dir", "ls", "glob"}:
        return f"[LIST: {path or '.'} — {n} entries]"
    elif name in {"grep", "search", "ripgrep"}:
        pat = args.get("pattern", path) or ""
        matches = [l for l in lines if ":" in l or l.startswith("/")]
        return f"[GREP: '{pat[:60]}' — {len(matches)} matches]"
    elif name == "write_file":
        return f"[WRITE: {path} — {n} lines written, ok]"
    elif name in {
        "edit_file",
        "patch_file",
        "str_replace_editor",
        "str_replace",
        "create_file",
    }:
        return f"[EDIT: {path} — changes applied]"
    elif name in {"run_tests", "pytest", "test"}:
        for line in lines:
            low = line.lower()
            if any(k in low for k in ("passed", "failed", "error", "ok", "test")):
                return f"[TESTS: {line[:140]}]"
        return f"[TESTS: {lines[0][:140]}]" if lines else "[TESTS: complete]"
    elif name in {"run_command", "bash", "shell"}:
        cmd = str(args.get("command", ""))[:50]
        first = lines[0][:100] if lines else "ok"
        return f"[CMD '{cmd}': {first}]"
    elif name == "run_code":
        return f"[CODE: ran {len(lines)} line(s) of output]"
    elif name == "load_skill":
        return f"[SKILL LOADED: {args.get('name', '?')}]"
    elif name.startswith("bg_"):
        return content  # background status/collect are already compact
    elif (
        name.startswith("find_")
        or name.startswith("search_")
        or name.startswith("list_symbol")
    ):
        return content  # index tools are compact; never mask them
    else:
        first = lines[0][:120] if lines else "ok"
        return f"[{name}: {first}]"


def _apply_observation_masking(
    messages: list[BaseMessage],
    tool_meta: dict[str, tuple[str, dict]],
) -> tuple[list[BaseMessage], int]:
    """Replace large ToolMessage content in old rounds with typed 1-line summaries.

    The JetBrains Research finding (NeurIPS 2025 DL4C): masking beats naive drop
    at 52% lower cost with +2.6% solve rate because the AI reasoning turns — which
    describe what was learned and decided — are preserved intact. Only the raw tool
    output (file contents, grep walls, command stdout) is compressed.

    Always keeps:
    - All AIMessages verbatim (reasoning gold, never touched)
    - Last _MASK_KEEP_ROUNDS rounds verbatim (fresh context the model needs)
    - Tool results under _MASK_MIN_TOKENS (cheap to keep)
    - Index tool results (already compact)

    Returns (new_messages, estimated_tokens_saved).
    """
    ai_positions = [i for i, m in enumerate(messages) if isinstance(m, AIMessage)]
    if len(ai_positions) <= _MASK_KEEP_ROUNDS:
        return messages, 0

    cutoff = ai_positions[-_MASK_KEEP_ROUNDS]  # mask everything before this index

    result = list(messages)
    saved = 0
    for i, m in enumerate(messages[:cutoff]):
        if not isinstance(m, ToolMessage):
            continue
        content = m.content or ""
        if isinstance(content, list):
            content = " ".join(
                b.get("text", "") for b in content if isinstance(b, dict)
            )
        if estimate_tokens(content) <= _MASK_MIN_TOKENS:
            continue
        # Skip if already a 1-liner summary
        if content.startswith("[") and "\n" not in content.strip():
            continue
        name, args = tool_meta.get(m.tool_call_id or "", ("", {}))
        masked = _mask_tool_result(name, args, content)
        saved += estimate_tokens(content) - estimate_tokens(masked)
        result[i] = ToolMessage(
            content=masked, tool_call_id=m.tool_call_id, name=m.name
        )

    return result, max(0, saved)


def _safety_drop(
    messages: list[BaseMessage],
    *,
    threshold: int = _DROP_THRESHOLD_DEFAULT,
    target: int = _DROP_TARGET_DEFAULT,
) -> tuple[list[BaseMessage], int]:
    """Absolute last resort: drop complete oldest rounds when context still exceeds limit.

    Should rarely fire after observation masking. Indicates either an extremely long
    run or pathologically large tool outputs that couldn't be masked enough.

    ``threshold``/``target`` come from _effective_ctx_budget() in normal use;
    the defaults here only apply to a caller that doesn't pass them.
    """
    total = _ctx_tokens(messages)
    if total <= threshold or len(messages) <= 3:
        return messages, 0

    # Protect head: system message + task HumanMessage + optional WM HumanMessage.
    # Must stop at the first AIMessage/ToolMessage (where real rounds begin) —
    # scanning for a *second* plain HumanMessage instead (as this used to do)
    # only stops there if something like a /steer message ever appears later.
    # In the common case (no steer), that second Human never comes, so the
    # loop silently walked the entire message list every time, leaving `tail`
    # empty and this "last resort" safety valve a no-op in practice.
    head_count = 0
    seen_human = False
    for m in messages:
        if isinstance(m, SystemMessage):
            head_count += 1
            continue
        if isinstance(m, HumanMessage):
            if not seen_human:
                seen_human = True  # first HumanMessage = task
                head_count += 1
                continue
            if _is_wm_message(m):
                head_count += 1  # WM message — also protect
                continue
        break  # first AIMessage/ToolMessage, or a later plain Human (e.g. /steer)

    head = messages[:head_count]
    tail = list(messages[head_count:])
    saved = 0

    while tail and (total - saved) > target:
        rounds_left = sum(1 for m in tail if isinstance(m, AIMessage))
        if rounds_left <= 1:
            break
        if isinstance(tail[0], ToolMessage):
            saved += estimate_tokens(getattr(tail[0], "content", "") or "")
            tail = tail[1:]
            continue
        if not isinstance(tail[0], AIMessage):
            break
        ai_c = getattr(tail[0], "content", "") or ""
        if isinstance(ai_c, list):
            ai_c = " ".join(b.get("text", "") for b in ai_c if isinstance(b, dict))
        saved += estimate_tokens(ai_c) + sum(
            estimate_tokens(str(tc.get("args", {})))
            for tc in (getattr(tail[0], "tool_calls", None) or [])
        )
        tail = tail[1:]
        while tail and isinstance(tail[0], ToolMessage):
            saved += estimate_tokens(getattr(tail[0], "content", "") or "")
            tail = tail[1:]

    if not saved:
        return messages, 0
    note = HumanMessage(
        content=f"[safety drop: ~{saved:,} tokens of old history removed]"
    )
    return head + [note] + tail, saved


# ---------------------------------------------------------------------------
# Activity display helpers
# ---------------------------------------------------------------------------


def _extract_llm_text(resp: BaseMessage) -> str:
    """Pull narrative text out of an LLM response (the part before/around tool calls)."""
    c = getattr(resp, "content", "") or ""
    if isinstance(c, str):
        return c.strip()
    if isinstance(c, list):
        parts = [
            block.get("text", "") if isinstance(block, dict) else str(block)
            for block in c
            if (isinstance(block, dict) and block.get("type") == "text")
            or isinstance(block, str)
        ]
        return " ".join(parts).strip()
    return ""


def _first_error_line(output: str) -> str:
    """Return the first meaningful error line from command output."""
    _skip = {"", "$", "---", "EXIT="}
    _skip_prefixes = ("[exit", "[stderr", "# ", "PS ", "> ")
    for line in output.splitlines():
        s = line.strip()
        if not s or s in _skip:
            continue
        if any(s.startswith(p) for p in _skip_prefixes):
            continue
        return s[:120]
    return ""


def _strip_env_prefix(cmd: str) -> str:
    """Remove leading $env:VAR='...' or $env:VAR="..." assignments from a command.

    The harness now injects cert bundles (REQUESTS_CA_BUNDLE etc.) into every
    subprocess automatically, so the agent no longer needs to prepend them.
    When it does anyway (e.g., learned from publishing.md), strip from display
    so the user sees the actual command rather than boilerplate.
    """
    # Matches one or more $env:NAME='value'; or $env:NAME="value"; prefixes
    return re.sub(r'^(\$env:\w+=(?:\'[^\']*\'|"[^"]*");\s*)+', "", cmd).strip()


def _short_path(path: str) -> str:
    """Shorten a path relative to the workspace root for display."""
    p = str(path or "?")
    try:
        ws = config.WORKSPACE_ROOT.replace("\\", "/")
        p2 = p.replace("\\", "/")
        if p2.startswith(ws):
            p = p2[len(ws) :].lstrip("/")
    except Exception:
        pass
    return ("…" + p[-55:]) if len(p) > 58 else p


def _activity_line(name: str, args: dict, ok: bool, simulated: bool = False) -> str:
    """Format one tool call as a compact human-readable line."""
    check = (
        "[dim](plan)[/dim]"
        if simulated
        else ("[green]✓[/green]" if ok else "[red]✗[/red]")
    )

    # Read-category tools
    if name == "read_file":
        desc = f"[dim]read    [/dim] {_short_path(args.get('path', '?'))}"
    elif name in ("list_dir", "glob"):
        tgt = args.get("path") or args.get("pattern") or "."
        desc = f"[dim]list    [/dim] {_short_path(tgt)}"
    elif name == "grep":
        pat = str(args.get("pattern") or args.get("query") or "")[:45]
        desc = f"[dim]grep    [/dim] {pat!r}"
    elif name in ("find_symbol", "find_callers", "search_symbols"):
        sym = str(args.get("name") or args.get("symbol") or args.get("query") or "")[
            :45
        ]
        label = {
            "find_symbol": "find",
            "find_callers": "callers",
            "search_symbols": "search",
        }.get(name, name)
        desc = f"[dim]{label:<8}[/dim] {sym}"

    # Write-category tools
    elif name in ("write_file", "create_file"):
        desc = f"[yellow]write   [/yellow] {_short_path(args.get('path', '?'))}"
    elif name in ("edit_file", "patch_file", "str_replace_editor", "str_replace"):
        desc = f"[yellow]edit    [/yellow] {_short_path(args.get('path', '?'))}"
    elif name in ("delete_file", "remove_file"):
        desc = f"[yellow]delete  [/yellow] {_short_path(args.get('path', '?'))}"

    # Execution tools
    elif name in ("run_command", "shell", "bash"):
        cmd = str(args.get("command") or args.get("cmd") or "")
        cmd = _strip_env_prefix(cmd)  # remove $env:VAR='...'; boilerplate
        if len(cmd) > 70:
            cmd = cmd[:67] + "…"
        desc = f"[cyan]run     [/cyan] {cmd}"
    elif name == "run_tests":
        cmd = str(args.get("command") or "auto-detect")[:60]
        desc = f"[cyan]test    [/cyan] {cmd}"

    # CodeAct + skills + background agents
    elif name == "run_code":
        first_line = str(args.get("code", "")).splitlines()[0:1]
        preview = (first_line[0] if first_line else "")[:60]
        desc = f"[cyan]code    [/cyan] {preview}"
    elif name == "load_skill":
        desc = f"[dim]skill   [/dim] {args.get('name', '?')}"
    elif name in ("bg_start", "bg_status", "bg_collect"):
        label = {"bg_start": "bg▶", "bg_status": "bg?", "bg_collect": "bg◀"}.get(
            name, "bg"
        )
        role = args.get("role") or (args.get("id") or "")
        desc = f"[magenta]{label:<8}[/magenta] {role}"

    # Fallback
    else:
        first = str(next(iter(args.values()), ""))[:50] if args else ""
        desc = f"[dim]{name:<8}[/dim] {first}"

    return f"  {check} {desc}"


# ---------------------------------------------------------------------------
# Executor
# ---------------------------------------------------------------------------


def run_executor(
    *,
    task: str,
    ctx: tools.ToolContext,
    **kwargs,
) -> ExecutorResult:
    """Run the agentic loop (see :func:`_run_executor_impl`) and record a trace.

    The trace (task, model replies, tool calls + results, stop reason) lands
    under ``<workspace>/.wells/traces/`` so any run — especially a failed one
    — can be replayed later as a harness regression test via ``wells replay``.
    Recording is best-effort and never affects the run; ``WELLS_TRACE=0``
    disables it.
    """
    result = _run_executor_impl(task=task, ctx=ctx, **kwargs)
    try:
        from wells import traces as _traces

        _traces.record_run(
            task=task,
            workspace=getattr(ctx, "workspace", None),
            step_label=kwargs.get("step_label", "executor"),
            result=result,
        )
    except Exception:
        pass
    return result


def _run_executor_impl(
    *,
    task: str,
    ctx: tools.ToolContext,
    toolset: list[tools.ToolDef] | None = None,
    max_steps: int | None = None,
    profile: str | None = None,
    temperature: float = 0.2,
    system_prefix: str = "",
    seed_messages: list[BaseMessage] | None = None,
    step_label: str = "executor",
    quiet: bool = False,
    stream: bool = False,
    images: list[str] | None = None,
) -> ExecutorResult:
    """Run the agentic tool-calling loop until completion or the step cap.

    Parameters
    ----------
    images : list[str], optional
        Image file paths attached to the initial task message (multimodal
        content blocks — see :mod:`wells.vision`). Ignored when
        ``seed_messages`` is given (a resumed run already has its own
        history; attaching new images there would need to land on a
        specific message, not the run's start).
    task : str
        Natural-language task for the executor (the "what to do").
    ctx : ToolContext
        Workspace/safety/plan-mode context threaded into every tool call.
    toolset : list[ToolDef], optional
        Tools the model may call. Defaults to all tools (read+write+exec); pass
        ``tools.registry(include_mutating=False)`` for read-only subagents.
    max_steps : int, optional
        Cap on tool-call rounds. Defaults to ``config.MAX_TOOL_STEPS``.
        **0 means no limit** (backstops: MAX_RUN_TOKENS, cancellation, and
        the stuck-loop detector).
    profile : str, optional
        Provider profile to use. Defaults to the active profile.
    quiet : bool, optional
        Suppress per-step UI output and activity updates. Used by parallel
        subagents so concurrent runs don't interleave in the log; token
        accounting and cancellation still apply.
    stream : bool, optional
        Stream model text to stdout token-by-token as it is generated (used by
        the top-level auto path for perceived speed). Falls back to a normal
        invoke on any streaming error. Ignored when ``quiet`` is set.
    """
    _ui = (lambda *a, **k: None) if quiet else ui
    _act = (lambda s: None) if quiet else CONTROL.set_activity
    stream = stream and not quiet
    toolset = toolset if toolset is not None else tools.ALL_TOOLS
    tools._ensure_optional_registered()  # CodeAct/bg tools may lazy-register
    cap = max_steps if max_steps is not None else config.MAX_TOOL_STEPS
    profile = profile or config.ACTIVE_PROFILE

    # Per-LLM-call usage log: populated by _account_usage, surfaced on
    # ExecutorResult and persisted into the run trace so `wells analyze`
    # can report cache efficiency round-by-round.
    usage_log: list[dict] = []

    # Local models (commonly Ollama) often load with a small context window
    # (Ollama's own default: 4096) regardless of what the model architecture
    # supports — measured live, Wells' full system prompt alone can reach
    # 4000+ tokens, leaving no room for the task, history, or a response
    # before the runtime silently truncates something to make it fit. Detect
    # this upfront and build a leaner prompt instead of guessing at a fix
    # after the fact (raising the model's actual context window is possible
    # but costs a multi-minute reload — see providers.warm_ollama_context;
    # this is the free, always-safe mitigation).
    model_label = config.providers.load_profile(profile)
    compact_prompt = bool(
        model_label and config.providers._looks_like_local_ollama(model_label)
    )
    # Same signal drives the context-trim ceiling: a small-context local
    # model gets config.SMALL_BUDGET instead of the generic 24000-token
    # default, so the safety-drop pipeline actually matches what it can hold.
    ctx_drop_threshold, ctx_drop_target = _effective_ctx_budget(compact_prompt)

    # Try structured outputs first (local Ollama: grammar-enforced tool-call
    # JSON), then native tool schemas, then text fallback.
    llm = (
        config.get_llm_for_task("coding", temperature=temperature)
        if profile == config.ACTIVE_PROFILE
        else config.providers.get_chat_model(profile, temperature=temperature)
    )
    plain_llm = llm  # unbound original, kept for the structured-mode fallback
    structured_llm = (
        _try_bind_structured(llm, model_label, toolset)
        if config.STRUCTURED_OUTPUTS
        else None
    )
    structured_mode = structured_llm is not None
    if structured_mode:
        llm = structured_llm
        native_tools = False
    else:
        bound = _try_bind_tools(llm, toolset)
        native_tools = bound is not None
        if native_tools:
            llm = bound
    _structured_fallback_used = False

    system = _system_prompt(
        task,
        toolset,
        plan_mode=ctx.plan_mode,
        workspace=ctx.workspace,
        compact=compact_prompt,
        structured=structured_mode,
    )
    if system_prefix:
        system = f"{system_prefix}\n\n{system}"

    messages: list[BaseMessage] = [SystemMessage(content=system)]
    if seed_messages:
        messages.extend(seed_messages)
    elif images:
        from wells import vision as _vision

        try:
            content = _vision.build_multimodal_content(
                f"Please complete this task:\n{task}", images
            )
        except _vision.VisionError as e:
            return ExecutorResult(
                summary=f"[executor error: {e}]",
                stopped_reason="error",
                messages=[SystemMessage(content=system)],
            )
        _vision_model_name = model_label.label() if model_label else profile
        if not _vision.provider_supports_vision(_vision_model_name):
            _ui(
                "warn",
                f"  [yellow]⚠ {len(images)} image(s) attached, but "
                f"'{_vision_model_name}' doesn't look vision-capable — "
                f"sending anyway; the provider may ignore or reject "
                f"it.[/yellow]",
            )
        messages.append(HumanMessage(content=content))
    else:
        messages.append(HumanMessage(content=f"Please complete this task:\n{task}"))

    model_name = model_label.label() if model_label else profile

    # Visible ground truth for "why didn't the model act on that hint?" —
    # text-fallback models only see harness feedback if they correctly
    # re-parse the whole growing transcript each turn; native tool-calling
    # models get it as a structured message. Silent before this, so a
    # bind_tools() that "succeeds" mechanically without real model support
    # was indistinguishable from genuine native support.
    _tc_mode = (
        "structured-json"
        if structured_mode
        else "native"
        if native_tools
        else "text-fallback"
    )
    _ui("round", f"[dim]model: {model_name} · tool-calling: {_tc_mode}[/dim]")

    wm = WorkingMemory(task=task)
    _tool_meta: dict[str, tuple[str, dict]] = {}  # tool_call_id → (name, args)

    # Reset the background-agent registry so slots don't leak across runs.
    # A nonzero count means a previous task's bg_start agent was still
    # running when this new run began — its thread keeps going (Python
    # can't force-kill it) but its slot is gone, so warn rather than let
    # it silently keep editing the workspace with no visibility.
    try:
        from wells import background as _bg

        _n_abandoned_bg = _bg.REGISTRY.reset()
        if _n_abandoned_bg:
            _ui(
                "warn",
                f"  [yellow]⚠ {_n_abandoned_bg} background agent(s) from "
                f"a previous task were still running and have been "
                f"abandoned — they may still be mid-edit in this "
                f"workspace.[/yellow]",
            )
    except Exception:
        pass

    rules_engine = None
    if config.RULES_ENFORCE:
        try:
            from wells import rules as _rules_mod

            rules_engine = _rules_mod.engine_for(ctx.workspace)
        except Exception:
            rules_engine = None

    history: list[dict] = []
    steps = 0
    rounds = 0
    stall_nudges = 0
    failure_ack_nudges = 0
    _rep_abort_nudges = 0  # generations aborted by the repetition guard
    total_saved = 0
    _read_ranges: dict[str, list[tuple[int, int]]] = {}  # path → [(offset, end), ...]
    _fail_patterns: dict[str, int] = {}  # command prefix → fail count
    _last_call_key: str | None = None  # name+args of the previous call
    _last_call_repeat = 0  # consecutive repeats of that call
    _unknown_tool_streak = 0  # consecutive calls to a name not in toolset
    _known_tool_names = {t.name for t in toolset}
    _file_write_history: dict[str, str] = {}  # path → most recent full content written
    _drift_streak = 0  # consecutive low-similarity full rewrites of one path
    _drift_path: str | None = None
    _readonly_names = {t.name for t in toolset if not t.mutating}
    # Read-only dedupe cache: call_key → (round, step) of the last execution.
    # Cleared whenever a mutating call succeeds (conservative: any write may
    # have changed what any read would return).
    _readonly_seen: dict[str, tuple[int, int]] = {}

    def _stopped(reason: str, summary: str) -> ExecutorResult:
        return ExecutorResult(
            summary=summary,
            steps_taken=steps,
            tool_calls=history,
            stopped_reason=reason,
            messages=messages,
            usage_log=usage_log,
        )

    _escalated = False

    def _try_escalate(reason: str) -> bool:
        """Swap in ESCALATION_PROFILE instead of hard-stopping, once per run.

        Every hard-stop below fires only after the model has demonstrably
        failed (ignored two rounds of coaching about the same loop). When a
        stronger profile is configured, that exact moment — not before — is
        when paying for it is worth it: the cheap model did all the work it
        could, and the alternative is throwing the whole run away. The
        stronger model inherits the full transcript, so it starts from
        everything already learned rather than from scratch. One escalation
        per run: if the strong model gets stuck too, the run stops for real.
        """
        nonlocal llm, native_tools, structured_mode, model_name, _escalated
        nonlocal _unknown_tool_streak, _last_call_repeat, _last_call_key
        nonlocal _drift_streak, _drift_path, compact_prompt
        nonlocal ctx_drop_threshold, ctx_drop_target
        if _escalated:
            return False
        esc = (config.ESCALATION_PROFILE or "").strip()
        if not esc or esc == profile:
            return False
        esc_profile = config.providers.load_profile(esc)
        if esc_profile is None:
            return False
        try:
            new_llm = config.providers.get_chat_model(esc, temperature=temperature)
        except Exception:
            return False
        _escalated = True
        structured = (
            _try_bind_structured(new_llm, esc_profile, toolset)
            if config.STRUCTURED_OUTPUTS
            else None
        )
        if structured is not None:
            llm = structured
            structured_mode = True
            native_tools = False
        else:
            structured_mode = False
            bound = _try_bind_tools(new_llm, toolset)
            native_tools = bound is not None
            llm = bound if native_tools else new_llm
        # The new model may have a completely different context budget and
        # calling convention — rebuild the system prompt to match (this
        # invalidates the prompt cache once, which a model swap does anyway).
        compact_prompt = bool(config.providers._looks_like_local_ollama(esc_profile))
        ctx_drop_threshold, ctx_drop_target = _effective_ctx_budget(compact_prompt)
        new_system = _system_prompt(
            task,
            toolset,
            plan_mode=ctx.plan_mode,
            workspace=ctx.workspace,
            compact=compact_prompt,
            structured=structured_mode,
        )
        if system_prefix:
            new_system = f"{system_prefix}\n\n{new_system}"
        messages[0] = SystemMessage(content=new_system)
        model_name = esc_profile.label()
        # Fresh slate for the loop detectors — they described the old model.
        _unknown_tool_streak = 0
        _last_call_repeat = 0
        _last_call_key = None
        _drift_streak = 0
        _drift_path = None
        messages.append(
            HumanMessage(
                content=(
                    f"[HARNESS: The previous model {reason} and has been replaced — "
                    f"you ({model_name}) are now driving this run. Review the "
                    f"transcript above, work out where it went wrong, and continue "
                    f"the ORIGINAL task from the current state of the workspace.]"
                )
            )
        )
        _ui(
            "warn",
            f"  [bold magenta]⬆ escalating to {model_name} — previous "
            f"model {reason}[/bold magenta]",
        )
        return True

    budget = config.MAX_RUN_TOKENS
    budget_warned = False
    cap_s = str(cap) if cap else "∞"

    while cap == 0 or steps < cap:
        CONTROL.set_progress(step_label, steps, cap)
        # ── Cooperative cancellation (set by the TUI on Escape) ─────────────
        if CONTROL.cancelled():
            return _stopped("cancelled", "(cancelled by user)")

        # ── Per-run token budget (ledger is reset at run start) ─────────────
        if budget:
            t = LEDGER.totals()
            used = t["input"] + t["output"]
            if used >= budget:
                _ui(
                    "warn",
                    f"\n[bold red]Token budget reached "
                    f"({used:,}/{budget:,}) — stopping.[/bold red]",
                )
                return _stopped(
                    "budget",
                    f"(stopped: token budget of {budget:,} reached at {used:,} tokens)",
                )
            if not budget_warned and used >= 0.8 * budget:
                budget_warned = True
                _ui(
                    "warn",
                    f"\n[yellow]⚠ {used:,} of {budget:,} run tokens used "
                    f"({used * 100 // budget}%).[/yellow]",
                )

        # ── Mid-run steering: user /steer notes land in the next round ──────
        steers = CONTROL.drain_steers()
        if steers:
            for s in steers:
                messages.append(
                    HumanMessage(
                        content=(
                            f"[USER STEER — read this NOW, mid-task instruction]: {s}\n"
                            f"Adjust your current approach accordingly before continuing."
                        )
                    )
                )
                _ui("warn", f"  [bold cyan]⮕ steer delivered:[/bold cyan] {s[:90]}")

        # ── Context management pipeline (order matters) ──────────────────────
        # Masking/dropping rewrite old history, which invalidates the
        # provider's prompt cache from the first changed message onward — so
        # they run only under real context pressure (estimated tokens past
        # the trim target). Below the target the transcript is append-only,
        # keeping the whole prefix cached (Ollama KV cache / cloud prompt
        # caching): the cheapest tokens are the ones never re-prefilled.
        mask_saved = drop_saved = 0
        if _ctx_tokens(messages) > ctx_drop_target:
            messages, mask_saved = _apply_observation_masking(messages, _tool_meta)
            messages, drop_saved = _safety_drop(
                messages, threshold=ctx_drop_threshold, target=ctx_drop_target
            )
        # WM last: refreshed each round in the append-only tail (cache-stable)
        # and at maximum recency, where a drifting model most reliably attends.
        messages = _inject_wm(messages, wm)
        saved = mask_saved + drop_saved
        if saved:
            total_saved += saved
        # ─────────────────────────────────────────────────────────────────────

        rounds += 1
        _act(f"thinking · round {rounds} · step {steps}/{cap_s}")
        streamed_this_round = False
        gen_aborted = False
        # The repetition guard needs a live stream to act on; it applies to
        # compact (local) profiles, where degenerate loops actually happen
        # and where every wasted generation second is felt.
        _guard_stream = compact_prompt and config.STREAM_GUARD
        try:
            if stream or _guard_stream:
                resp, streamed_this_round, gen_aborted = _stream_invoke(
                    llm, messages, display=stream, guard=_guard_stream
                )
            else:
                resp = _invoke_cancelable(llm, messages)
        except Exception as e:
            if structured_mode and not _structured_fallback_used:
                # Most likely an older Ollama server rejecting the schema'd
                # response_format. Drop to the ordinary path (native bind or
                # text fallback) once per run rather than failing the task
                # over a format negotiation.
                _structured_fallback_used = True
                structured_mode = False
                llm = plain_llm
                bound = _try_bind_tools(llm, toolset)
                native_tools = bound is not None
                if native_tools:
                    llm = bound
                _ui(
                    "warn",
                    "  [yellow]⚠ structured output rejected by the "
                    "server — falling back to "
                    f"{'native' if native_tools else 'text'} tool "
                    "calling[/yellow]",
                )
                continue
            return ExecutorResult(
                summary=f"[executor error: {type(e).__name__}: {e}]",
                steps_taken=steps,
                tool_calls=history,
                stopped_reason="error",
                messages=messages,
                usage_log=usage_log,
            )
        if resp is None:
            # Cancelled while waiting on the model — there's no way to abort
            # an in-flight HTTP request, so the call keeps running on its own
            # thread and its eventual result is simply discarded.
            return _stopped("cancelled", "(cancelled by user)")
        _account_usage(
            step=step_label,
            model=model_name,
            messages=messages,
            resp=resp,
            saved_by_trim=saved,
            usage_log=usage_log,
            round_num=rounds,
            mask_saved=mask_saved,
            drop_saved=drop_saved,
        )
        messages.append(resp)

        calls = _extract_calls(resp, native_tools=native_tools)

        # Show the LLM's reasoning text (if any) before we execute the tool calls.
        # This tells the user WHY the agent is doing what it's about to do.
        llm_text = _extract_llm_text(resp)
        if structured_mode:
            # final_answer pseudo-calls become the final text; raw call JSON
            # is not narrative.
            calls, llm_text = _resolve_structured_reply(calls, llm_text)
        if llm_text and not streamed_this_round:
            # Trim to ~200 chars; collapse newlines so it reads as one line.
            preview = " ".join(llm_text.split())[:200]
            if len(llm_text) > 200:
                preview += "…"
            _ui("llm_text", f"\n[dim italic]{preview}[/dim italic]")
        elif calls and not llm_text:
            # No narrative text — at least show the round number so the user
            # knows progress is being made.
            _ui("round", f"\n[dim]Round {rounds}  (step {steps + 1}/{cap_s})[/dim]")

        if not calls:
            # A guard-aborted reply with no parseable call is repetition
            # garbage, never a final answer — coach once and re-roll rather
            # than accepting (or salvaging from) a degenerate loop's output.
            if gen_aborted:
                _rep_abort_nudges += 1
                _ui(
                    "warn",
                    f"  [bold yellow]⚠ generation aborted — output was "
                    f"repeating itself (attempt {_rep_abort_nudges}/3)"
                    f"[/bold yellow]",
                )
                if _rep_abort_nudges >= 3:
                    return _stopped(
                        "stuck_loop",
                        "(stopped: the model's output degenerated into verbatim "
                        "self-repetition on 3 consecutive replies)",
                    )
                messages.append(
                    HumanMessage(
                        content=(
                            "[HARNESS: Your previous reply was cut off because it was "
                            "repeating the same text over and over. Do not repeat "
                            "yourself. Continue the task with ONE concise reply — a "
                            "tool call if action is needed, or a short final summary "
                            "if the task is done.]"
                        )
                    )
                )
                continue
            # Zero tool calls with zero real action taken so far this run is
            # almost never a genuine finish for a task that requires changes —
            # it's a model (usually a small/local one) that ignored the
            # tool-calling protocol and answered like a plain chatbot instead.
            #
            # Some models won't be talked into the protocol no matter how many
            # times they're nudged — they just answer with the file content as
            # a markdown code block, habitually, every time. If the task names
            # an explicit target file and the reply has exactly one unambiguous
            # code block, the harness writes it directly rather than burning
            # more rounds asking the model to do something it isn't going to
            # do. This still goes through the same rules gate a real write_file
            # call would hit, so it can't bypass confirm/block policy.
            salvage = (
                _try_salvage_write(task, llm_text) if steps == 0 and llm_text else None
            )
            if salvage:
                path, content = salvage
                s_args = {"path": path, "content": content}
                s_decision = (
                    rules_engine.check("write_file", s_args) if rules_engine else None
                )
                if s_decision is not None and not s_decision.allow:
                    obs = "\n".join(s_decision.notes) or "[RULES: action blocked]"
                    _ui(
                        "warn",
                        f"  [bold red]⛔ rule "
                        f"{s_decision.rule.id if s_decision.rule else '?'} "
                        f"blocked salvaged write_file[/bold red]",
                    )
                    messages.append(
                        HumanMessage(
                            content=(
                                f"[HARNESS: Detected an unwrapped code block addressed to "
                                f"'{path}' but a workspace rule blocked writing it: {obs} "
                                f"Call the tool yourself with a compliant alternative.]"
                            )
                        )
                    )
                elif s_decision is not None and s_decision.confirm:
                    from wells import safety as _safety

                    ap = ctx.approver or _safety.get_approver()
                    rid = s_decision.rule.id if s_decision.rule else "rule"
                    approved = (
                        bool(ap(f"rule:{rid}", f"salvaged write_file: {path}"))
                        if ap
                        else False
                    )
                    if not approved:
                        messages.append(
                            HumanMessage(
                                content=(
                                    f"[HARNESS: Detected an unwrapped code block addressed to "
                                    f"'{path}' but writing it requires confirmation that "
                                    f"wasn't granted. Call write_file yourself if you still "
                                    f"want this written.]"
                                )
                            )
                        )
                        salvage = None
                if salvage and (s_decision is None or s_decision.allow):
                    steps += 1
                    _act(f"write_file (harness salvage) · step {steps}/{cap_s}")
                    result = tools.dispatch("write_file", s_args, ctx)
                    obs_text = result.to_model_text()
                    if rules_engine is not None and s_decision is not None:
                        rule_notes = rules_engine.apply_liability(
                            s_decision, ok=result.ok, simulated=result.simulated
                        )
                        if rule_notes:
                            obs_text = obs_text + "\n\n" + "\n".join(rule_notes)
                        wm.open_liabilities = rules_engine.liability_summary()
                    wm.update_from_tool("write_file", s_args, obs_text, result.ok)
                    history.append(
                        {
                            "name": "write_file",
                            "args": {"path": path},
                            "ok": result.ok,
                            "output_preview": (result.output or result.error or "")[
                                :200
                            ],
                            "simulated": result.simulated,
                            "salvaged": True,
                        }
                    )
                    _ui(
                        "warn",
                        f"  [yellow]⚠ model answered with a code block instead "
                        f"of a tool call — harness salvaged it into "
                        f"write_file({path!r})[/yellow]",
                    )
                    messages.append(
                        HumanMessage(
                            content=(
                                f"[HARNESS: You replied with a code block but no tool call. "
                                f"The harness matched it to '{path}' (named in the task) and "
                                f"wrote it directly:\n{obs_text}\n"
                                f"If more work remains, continue with tool calls — you must "
                                f"use them for anything further. If the task is complete, say "
                                f"so with no further action.]"
                            )
                        )
                    )
                CONTROL.set_progress(step_label, steps, cap)
                continue
            if steps == 0 and llm_text and stall_nudges < config.STALL_NUDGE_MAX:
                stall_nudges += 1
                _ui(
                    "warn",
                    f"  [yellow]⚠ model produced text but no tool call "
                    f"(attempt {stall_nudges}/{config.STALL_NUDGE_MAX}) — "
                    f"nudging back onto the tool-calling protocol[/yellow]",
                )
                if structured_mode:
                    nudge = (
                        "[HARNESS: You replied with final_answer before taking any "
                        "action. Nothing was written, run, or changed — the task "
                        "cannot be complete. Reply with a real tool call "
                        '({"name": "<tool>", "args": {...}}) that makes progress; '
                        "only use final_answer once the work is actually done.]"
                    )
                else:
                    nudge = (
                        "[HARNESS: You responded with text but called no tool. Describing "
                        "steps does not perform them — nothing was written, run, or changed. "
                        "If the task requires an action, take it now by emitting a line of "
                        "the exact form: "
                        '<tool_call>{"name": "<tool>", "args": {...}}</tool_call> '
                        "and nothing else on that line. Do not restate a plan in prose — "
                        "call the tool.]"
                    )
                messages.append(HumanMessage(content=nudge))
                continue
            # Verify before accepting completion: the model reporting text
            # feeding the previous tool's error back into its context is not
            # the same as the model actually resolving it. If its very last
            # real action failed and it just moved straight to a wrap-up
            # answer without another attempt or even mentioning the failure,
            # that answer is describing an outcome that never happened —
            # don't let the run end there. A model that explicitly reports
            # the failure as a blocker (permitted by WORKING RULES #5) is
            # left alone — only silent, unacknowledged failure is refused.
            _ack_words = (
                "fail",
                "error",
                "unable",
                "cannot",
                "can't",
                "couldn't",
                "block",
                "issue",
                "problem",
                "did not",
                "didn't",
                "won't",
                "not able",
                "no module",
                "not found",
                "denied",
            )
            _acked_failure = any(w in (llm_text or "").lower() for w in _ack_words)
            if (
                history
                and not history[-1]["ok"]
                and not _acked_failure
                and failure_ack_nudges < config.STALL_NUDGE_MAX
            ):
                failure_ack_nudges += 1
                last = history[-1]
                _ui(
                    "warn",
                    f"  [yellow]⚠ run ending right after a failed "
                    f"{last['name']} with no fix attempted (attempt "
                    f"{failure_ack_nudges}/{config.STALL_NUDGE_MAX}) — "
                    f"refusing to accept as complete[/yellow]",
                )
                messages.append(
                    HumanMessage(
                        content=(
                            f"[HARNESS: Your last action ({last['name']}) failed:\n"
                            f"{last['output_preview']}\n"
                            f"Your reply did not fix this, retry a different approach, or "
                            f"explicitly report it as a blocker — it just moved on as if "
                            f"the task were done. It is not done while the failure stands "
                            f"unaddressed. Either resolve it, try something else, or state "
                            f"clearly that this is blocking completion and why.]"
                        )
                    )
                )
                continue
            # Either the model has already taken real action this run (a
            # normal wrap-up summary), or it exhausted its coaching budget —
            # accept this as the final answer.
            if llm_text and not streamed_this_round:
                _ui("round", "")  # blank line before final summary
            return ExecutorResult(
                summary=llm_text or "(no output)",
                steps_taken=steps,
                tool_calls=history,
                stopped_reason="done",
                messages=messages,
                usage_log=usage_log,
                streamed=streamed_this_round and bool(llm_text),
            )

        # ── Parallel prefetch: leading run of read-only calls ────────────────
        # Only the batch's *prefix* of read-only calls is eligible: a mutating
        # call changes what any later read in the same batch would see, so
        # everything from the first non-read-only call onward stays strictly
        # sequential. Rules must clear a call before it runs — anything the
        # engine would block or gate behind confirmation is left to the
        # sequential path, which never executes it early.
        _prefetched: dict[int, tools.ToolResult] = {}
        if config.PARALLEL_READS and len(calls) > 1:
            _par_idx: list[int] = []
            for _i, _c in enumerate(calls):
                _cname = _c.get("name")
                if not _cname or _cname not in _readonly_names:
                    break
                if rules_engine is not None:
                    _d = rules_engine.check(_cname, _c.get("args") or {})
                    if not _d.allow or _d.confirm:
                        break
                if config.HOOKS_ENABLE:
                    from wells import hooks as _hooks_mod

                    _h_ok, _ = _hooks_mod.fire_pre_tool_use(
                        ctx.workspace, _cname, _c.get("args") or {}
                    )
                    if not _h_ok:
                        break  # sequential path re-runs the check and blocks properly
                _par_idx.append(_i)
            if len(_par_idx) > 1:
                from concurrent.futures import ThreadPoolExecutor as _TPE

                with _TPE(
                    max_workers=min(4, len(_par_idx)),
                    thread_name_prefix="wells-par-read",
                ) as _pool:
                    _futs = {
                        _i: _pool.submit(
                            tools.dispatch,
                            calls[_i]["name"],
                            calls[_i].get("args") or {},
                            ctx,
                        )
                        for _i in _par_idx
                    }
                    for _i, _fut in _futs.items():
                        try:
                            _prefetched[_i] = _fut.result()
                        except Exception:
                            pass  # sequential dispatch below is the fallback

        for _call_idx, call in enumerate(calls):
            if CONTROL.cancelled():
                return _stopped("cancelled", "(cancelled by user)")
            steps += 1
            name = call.get("name")
            args = call.get("args") or {}
            tcid = call.get("id") or f"call_{steps}"

            if not name:
                parse_err = call.get("_parse_error")
                if parse_err:
                    # Most common real-world cause: an inner double quote from
                    # embedded source code (e.g. "utf8") wasn't escaped as \"
                    # inside the JSON string value. Show the model its own
                    # malformed blob so it can see and fix the actual mistake,
                    # instead of a content-free "missing name" message that
                    # gives it nothing to act on.
                    obs_text = (
                        "[error: tool call JSON failed to parse — most likely an "
                        'unescaped " character inside a string value (e.g. source '
                        'code containing "utf8" must be written as \\"utf8\\" '
                        "inside the JSON string). Your text was:\n"
                        f"{parse_err[:1500]}\n"
                        "Re-emit the call with correctly escaped JSON.]"
                    )
                else:
                    obs_text = "[error: malformed tool call — missing name]"
                history.append(
                    {"name": "?", "args": args, "ok": False, "output_preview": obs_text}
                )
                messages.append(
                    _tool_message(obs_text, tool_call_id="parseerr", name="parse_error")
                )
                continue

            _tool_meta[tcid] = (name, args)

            # ── Repeat-call tracking (feeds the stuck-loop backstop below) ──
            # Unlike _fail_patterns (which only counts *failed* shell commands),
            # this catches a model calling the same tool with the same args
            # over and over even when each call *succeeds* — e.g. re-listing
            # the same directory round after round without making progress.
            call_key = f"{name}:{json.dumps(args, sort_keys=True, default=str)}"
            if call_key == _last_call_key:
                _last_call_repeat += 1
            else:
                _last_call_key = call_key
                _last_call_repeat = 1

            # ── Rules gate: deterministic enforcement before execution ─────
            rule_notes: list[str] = []
            decision = None
            if rules_engine is not None:
                decision = rules_engine.check(name, args)
                rule_notes = list(decision.notes)
                if not decision.allow:
                    obs_text = "\n".join(rule_notes) or "[RULES: action blocked]"
                    _ui(
                        "warn",
                        f"  [bold red]⛔ rule "
                        f"{decision.rule.id if decision.rule else '?'} "
                        f"blocked {name}[/bold red]",
                    )
                    history.append(
                        {
                            "name": name,
                            "args": args,
                            "ok": False,
                            "output_preview": obs_text[:200],
                        }
                    )
                    messages.append(
                        _tool_message(
                            obs_text, tool_call_id=tcid, name=name, ai_message=resp
                        )
                    )
                    continue
                if decision.confirm:
                    from wells import safety as _safety

                    ap = ctx.approver or _safety.get_approver()
                    rid = decision.rule.id if decision.rule else "rule"
                    detail = str(args.get("command") or args)[:160]
                    approved = bool(ap(f"rule:{rid}", detail)) if ap else False
                    if not approved:
                        why = (
                            "denied by user"
                            if ap
                            else "no approver available — confirm-severity rules "
                            "require an interactive session"
                        )
                        obs_text = (
                            f"[RULES {rid}: action NOT executed ({why}). "
                            f"{decision.rule.message if decision.rule else ''} "
                            f"Choose a compliant alternative or ask the user.]"
                        )
                        _ui(
                            "warn",
                            f"  [yellow]⛔ rule {rid}: {name} not "
                            f"executed ({why})[/yellow]",
                        )
                        history.append(
                            {
                                "name": name,
                                "args": args,
                                "ok": False,
                                "output_preview": obs_text[:200],
                            }
                        )
                        messages.append(
                            _tool_message(
                                obs_text, tool_call_id=tcid, name=name, ai_message=resp
                            )
                        )
                        continue

            # ── PreToolUse hooks: user-scriptable gate, separate from rules ──
            # Rules (above) are Wells' own declarative policy; hooks are
            # arbitrary user shell scripts (.wells/hooks.yaml). Checked after
            # rules so a call rules already blocked never pays for a
            # subprocess. A prefetched read-only result (dispatched before
            # this call was reached) already passed this same check when it
            # was scheduled — see the prefetch eligibility filter above.
            if config.HOOKS_ENABLE and _call_idx not in _prefetched:
                from wells import hooks as _hooks_mod

                hook_allowed, hook_reason = _hooks_mod.fire_pre_tool_use(
                    ctx.workspace, name, args
                )
                if not hook_allowed:
                    obs_text = f"[HOOK: action blocked — {hook_reason}]"
                    _ui(
                        "warn",
                        f"  [bold red]⛔ hook blocked {name}: {hook_reason[:100]}[/bold red]",
                    )
                    history.append(
                        {
                            "name": name,
                            "args": args,
                            "ok": False,
                            "output_preview": obs_text[:200],
                        }
                    )
                    messages.append(
                        _tool_message(
                            obs_text, tool_call_id=tcid, name=name, ai_message=resp
                        )
                    )
                    continue

            _act(f"{name} · step {steps}/{cap_s}")
            CONTROL.set_progress(step_label, steps, cap)
            result = _prefetched.pop(_call_idx, None)
            # ── Read-only dedupe ────────────────────────────────────────────
            # An identical read-only call whose result is still verbatim in
            # the model's recent context (inside the masking keep-window,
            # nothing mutated since) is answered with a pointer instead of
            # re-dispatching — saves the tokens of a duplicate full output
            # and makes a re-read loop visible to the model immediately.
            _dedupe_hit = False
            if result is None and config.DEDUPE_READS and name in _readonly_names:
                _seen = _readonly_seen.get(call_key)
                if _seen is not None and (rounds - _seen[0]) < _MASK_KEEP_ROUNDS:
                    _dedupe_hit = True
                    result = tools.ToolResult(
                        True,
                        f"[HARNESS CACHE: this exact {name} call already ran at "
                        f"step {_seen[1]} and nothing has been written or "
                        f"executed since — its full output is still in your "
                        f"context above. Reuse it; do not re-request unchanged "
                        f"data.]",
                        "",
                    )
            if result is None:
                result = tools.dispatch(name, args, ctx)
            if name in _readonly_names:
                if not _dedupe_hit and result.ok and not result.simulated:
                    _readonly_seen[call_key] = (rounds, steps)
            elif result.ok and not result.simulated:
                _readonly_seen.clear()  # a successful mutation may change any read
            # dispatch() can block for a while (a shell command, a subagent);
            # check again right after it returns instead of waiting for the
            # top of the next round — a cancel that landed mid-command should
            # stop the run now, not after one more LLM round-trip.
            if CONTROL.cancelled():
                return _stopped("cancelled", "(cancelled by user)")
            obs_text = result.to_model_text()

            # ── PostToolUse hooks ────────────────────────────────────────────
            # Observational only — the call already ran, so a hook can't
            # block it, only annotate the observation the model reads next
            # (same moment-of-relevance placement rules.py uses for its own
            # notes).
            if config.HOOKS_ENABLE:
                from wells import hooks as _hooks_mod

                _hook_notes = _hooks_mod.fire_post_tool_use(
                    ctx.workspace,
                    name,
                    args,
                    ok=result.ok,
                    output_preview=(result.output or result.error or ""),
                )
                if _hook_notes:
                    obs_text = obs_text + "\n\n" + "\n".join(_hook_notes)

            # ── Invalid-tool-name streak ─────────────────────────────────────
            # Keyed on whether `name` is a real tool, not on exact args — a
            # model that invents a tool (e.g. "report_error") can dodge the
            # identical-args repeat detector below just by rewording the
            # message each time while calling the same nonexistent tool over
            # and over. Tracking validity instead of exact args closes that
            # gap: rewording doesn't reset this counter, only actually
            # switching to a real tool does.
            if name not in _known_tool_names:
                _unknown_tool_streak += 1
            else:
                _unknown_tool_streak = 0
            if _unknown_tool_streak == 3:
                catalog_names = ", ".join(sorted(_known_tool_names))
                obs_text = obs_text + (
                    f"\n\n[HARNESS: '{name}' is not a real tool — you have now "
                    f"invented a nonexistent tool name {_unknown_tool_streak} times "
                    f"in a row. The ONLY tools that exist are: {catalog_names}. "
                    f"Pick one of those, or if none fits, stop and describe the "
                    f"blocker in plain text with no tool call.]"
                )
                _ui(
                    "warn",
                    f"  [bold yellow]⚠ invented tool name "
                    f"'{name}' called {_unknown_tool_streak}× in a row — "
                    f"model told the real tool list[/bold yellow]",
                )
            if _unknown_tool_streak >= 6:
                messages.append(
                    _tool_message(
                        obs_text, tool_call_id=tcid, name=name, ai_message=resp
                    )
                )
                if _try_escalate(
                    f"kept inventing nonexistent tool names "
                    f"({_unknown_tool_streak}× in a row)"
                ):
                    break
                _ui(
                    "warn",
                    f"  [bold red]⛔ stuck loop — model kept inventing "
                    f"nonexistent tool names ({_unknown_tool_streak}× in a "
                    f"row), stopping the run[/bold red]",
                )
                return _stopped(
                    "stuck_loop",
                    f"(stopped: model called nonexistent tool names "
                    f"{_unknown_tool_streak} times in a row with no progress)",
                )

            # ── Stuck-loop warning ───────────────────────────────────────────
            # Unlike _fail_patterns below (which only fires on *failed* shell
            # commands), this catches a tool call that keeps succeeding with
            # identical arguments but never moves the task forward — e.g.
            # re-listing the same directory round after round. Warn at 3
            # repeats; the hard stop below fires at 6 if the warning is ignored.
            if _last_call_repeat == 3:
                obs_text = obs_text + (
                    f"\n\n[HARNESS: You have called {name} with identical "
                    f"arguments {_last_call_repeat} times in a row. Repeating "
                    f"it again will not produce new information — change your "
                    f"approach or report what is blocking you.]"
                )
                _ui(
                    "warn",
                    f"  [bold yellow]⚠ {name} repeated "
                    f"{_last_call_repeat}× with identical args — model "
                    f"told to change approach[/bold yellow]",
                )

            if rules_engine is not None and decision is not None:
                # Liability transitions apply only after the command actually
                # ran (a failed `vastai create` starts nothing).
                rule_notes += rules_engine.apply_liability(
                    decision, ok=result.ok, simulated=result.simulated
                )
            if rule_notes:
                # Moment-of-relevance injection: the fired rule lands in the
                # observation the model reads next, not a wall of text at the top.
                obs_text = obs_text + "\n\n" + "\n".join(rule_notes)
                for n in rule_notes:
                    _ui("warn", f"  [magenta]§ {n[:110]}[/magenta]")
            if rules_engine is not None:
                wm.open_liabilities = rules_engine.liability_summary()
            wm.update_from_tool(name, args, obs_text, result.ok)

            history.append(
                {
                    "name": name,
                    "args": args,
                    "ok": result.ok,
                    "output_preview": (result.output or result.error or "")[:200],
                    "simulated": result.simulated,
                }
            )

            # ── Re-read detection ──────────────────────────────────────────
            # Track which line ranges of each file have been read.
            # On the 2nd+ read, inject a harness note INTO obs_text so the
            # model actually sees it (a terminal-only warning is invisible to
            # the model and has no effect on its behaviour).
            if name == "read_file":
                raw_path = str(args.get("path", ""))
                fpath = _short_path(raw_path)
                offset = int(args.get("offset") or 1)
                limit = int(args.get("limit") or 2000)
                ranges = _read_ranges.setdefault(raw_path, [])
                ranges.append((offset, offset + limit - 1))
                count = len(ranges)

                if count >= 2:
                    prior = ", ".join(f"{s}–{e}" for s, e in ranges[:-1])
                    note = (
                        f"\n\n[HARNESS: You have now read '{fpath}' {count} time(s) "
                        f"(previously lines {prior}). "
                        f"Avoid reading this file again — use grep with a specific "
                        f"pattern to find the exact lines you need instead.]"
                    )
                    obs_text = obs_text + note
                    lbl = "re-reading" if count == 2 else f"reading ×{count}"
                    _ui(
                        "warn",
                        f"  [yellow]⚠ {lbl} {fpath} — consider grep or find_symbol instead[/yellow]",
                    )

            # ── Self-heal: fast checker after every successful write/edit ──
            # Ground truth in the same round: a syntax error or undefined name
            # reaches the model immediately instead of surfacing a full tester
            # loop (an LLM call) later.
            if (
                config.SELF_CHECK
                and result.ok
                and not result.simulated
                and name
                in (
                    "write_file",
                    "edit_file",
                    "create_file",
                    "patch_file",
                    "str_replace_editor",
                    "str_replace",
                )
            ):
                _checked_path = str(
                    args.get("path")
                    or args.get("file_path")
                    or args.get("filename")
                    or ""
                )
                if _checked_path:
                    from wells import checkers

                    check_err = checkers.quick_check(_checked_path, ctx.workspace)
                    if check_err:
                        obs_text = obs_text + (
                            f"\n\n[HARNESS CHECK: fast lint/syntax check FAILED for "
                            f"{_short_path(_checked_path)} — fix these before doing "
                            f"anything else:\n{check_err}]"
                        )
                        _ui(
                            "warn",
                            f"  [yellow]⚠ check failed: "
                            f"{_short_path(_checked_path)}[/yellow]",
                        )

            # ── Task-drift detection ────────────────────────────────────────
            # See the module-level comment above _rewrite_similarity for the
            # full rationale. Scoped to write_file specifically (the observed
            # failure mode): a full-content replacement, not an incremental
            # edit_file patch, is what "thrashing" actually looks like.
            if name == "write_file" and result.ok and not result.simulated:
                _wpath = str(args.get("path") or args.get("file_path") or "")
                _new_content = str(args.get("content", ""))
                if _wpath:
                    _prev_content = _file_write_history.get(_wpath)
                    if _prev_content is not None:
                        _sim = _rewrite_similarity(_prev_content, _new_content)
                        if _sim < _DRIFT_SIMILARITY_THRESHOLD:
                            if _drift_path == _wpath:
                                _drift_streak += 1
                            else:
                                _drift_path = _wpath
                                _drift_streak = 1
                        elif _drift_path == _wpath:
                            _drift_streak = 0
                            _drift_path = None
                    _file_write_history[_wpath] = _new_content

                    if _drift_streak == _DRIFT_NUDGE_AT:
                        obs_text = obs_text + (
                            f"\n\n[HARNESS: Your last {_drift_streak} rewrites of "
                            f"'{_short_path(_wpath)}' have each replaced almost "
                            f"everything from the version before — not refining it, "
                            f"replacing it. That pattern usually means you've lost "
                            f"track of the actual goal:\n{task[:400]}\n"
                            f"Stop and check your last write against that goal before "
                            f"writing again. If you're stuck, say so instead of "
                            f"rewriting again.]"
                        )
                        _ui(
                            "warn",
                            f"  [bold yellow]⚠ {_drift_streak} unrelated "
                            f"full rewrites of {_short_path(_wpath)} in a row "
                            f"— model shown the original goal again[/bold yellow]",
                        )

            # ── Stuck-loop detection ───────────────────────────────────────
            # Inject a note into obs_text when the same command fails 3+ times
            # so the model knows to stop retrying and report the blocker.
            if not result.ok and not result.simulated:
                err = _first_error_line(result.output or result.error or "")
                if err:
                    _ui("warn", f"    [dim red]↳ {err}[/dim red]")

                if name in ("run_command", "shell", "bash"):
                    raw_cmd = str(args.get("command") or args.get("cmd") or "")
                    key = _strip_env_prefix(raw_cmd)[:50]
                    _fail_patterns[key] = _fail_patterns.get(key, 0) + 1
                    if _fail_patterns[key] >= 3:
                        blocker_note = (
                            f"\n\n[HARNESS: This command (or a close variant) has failed "
                            f"{_fail_patterns[key]} times. Stop retrying. Report what is "
                            f"blocking you and what you have tried so far.]"
                        )
                        obs_text = obs_text + blocker_note
                        _ui(
                            "warn",
                            f"  [bold yellow]⚠ same command failed {_fail_patterns[key]}× "
                            f"— model told to report blocker[/bold yellow]",
                        )

            _ui(
                "tool_line",
                _activity_line(name, args, result.ok, result.simulated),
                name=name,
                ok=result.ok,
                simulated=result.simulated,
            )

            # Full, untruncated record of what the model actually saw — the
            # TUI/CLI only ever print a one-line summary per round, so this is
            # the only place a user can go read the real command output/error
            # after the fact. See the /log slash command.
            try:
                from wells import logger as _logger

                _logger.log_tool_result(name, args, result.ok, obs_text)
            except Exception:
                pass

            messages.append(
                _tool_message(obs_text, tool_call_id=tcid, name=name, ai_message=resp)
            )

            # ── Stuck-loop hard stop ─────────────────────────────────────────
            # The warning above gives the model a chance to course-correct; if
            # it's ignored, force the run to end instead of looping forever.
            # This is the real backstop the module docstring promises — with
            # MAX_TOOL_STEPS and MAX_RUN_TOKENS both defaulting to 0
            # (unlimited), nothing else stops a model that keeps calling the
            # same read-only tool with the same args.
            if _last_call_repeat >= 6:
                if _try_escalate(
                    f"repeated {name} with identical arguments "
                    f"{_last_call_repeat}× with no progress"
                ):
                    break
                _ui(
                    "warn",
                    f"  [bold red]⛔ stuck loop — {name} repeated "
                    f"{_last_call_repeat}× with identical args, "
                    f"stopping the run[/bold red]",
                )
                return _stopped(
                    "stuck_loop",
                    f"(stopped: {name} was called with identical arguments "
                    f"{_last_call_repeat} times in a row with no progress)",
                )

            # ── Task-drift hard stop ────────────────────────────────────────
            # The nudge above gives the model a chance to notice and correct;
            # if it keeps thrashing anyway, stop rather than burn the rest of
            # the step budget on rewrites that were never converging.
            if _drift_streak >= _DRIFT_STOP_AT:
                if _try_escalate(
                    f"drifted off the goal ({_drift_streak} unrelated full "
                    f"rewrites of '{_short_path(_drift_path or '')}' in a row)"
                ):
                    break
                _ui(
                    "warn",
                    f"  [bold red]⛔ task drift — {_drift_streak} "
                    f"unrelated full rewrites of "
                    f"{_short_path(_drift_path or '')} in a row with no "
                    f"convergence toward the goal, stopping the "
                    f"run[/bold red]",
                )
                return _stopped(
                    "stuck_loop",
                    f"(stopped: {_drift_streak} consecutive rewrites of "
                    f"'{_drift_path}' each replaced almost everything from the "
                    f"version before, with no convergence toward the goal)",
                )

        if cap and steps >= cap:
            # One final round: apply full context pipeline then ask for summary.
            try:
                ms = ds = 0
                if _ctx_tokens(messages) > ctx_drop_target:
                    messages, ms = _apply_observation_masking(messages, _tool_meta)
                    messages, ds = _safety_drop(
                        messages, threshold=ctx_drop_threshold, target=ctx_drop_target
                    )
                messages = _inject_wm(messages, wm)
                final_saved = ms + ds
                final = config._invoke_with_retry(llm, messages)
                _account_usage(
                    step=step_label,
                    model=model_name,
                    messages=messages,
                    resp=final,
                    saved_by_trim=final_saved,
                    usage_log=usage_log,
                    round_num=rounds + 1,
                    mask_saved=ms,
                    drop_saved=ds,
                )
                messages.append(final)
                return ExecutorResult(
                    summary=(getattr(final, "content", "") or "").strip()
                    or "(reached step cap)",
                    steps_taken=steps,
                    tool_calls=history,
                    stopped_reason="max_steps",
                    messages=messages,
                    usage_log=usage_log,
                )
            except Exception:
                return ExecutorResult(
                    summary="(reached step cap)",
                    steps_taken=steps,
                    tool_calls=history,
                    stopped_reason="max_steps",
                    messages=messages,
                    usage_log=usage_log,
                )

    return ExecutorResult(
        summary="(reached step cap)",
        steps_taken=steps,
        tool_calls=history,
        stopped_reason="max_steps",
        messages=messages,
    )


# ---------------------------------------------------------------------------
# Internals: streaming + tool binding + call extraction + tool message
# ---------------------------------------------------------------------------


def _invoke_cancelable(llm, messages) -> BaseMessage | None:
    """Invoke ``llm`` on a helper thread and poll CONTROL every 250ms.

    A plain blocking ``llm.invoke()`` has no cancellation checks at all — it
    just sits there for up to ``LLM_TIMEOUT × LLM_MAX_RETRIES`` seconds
    (minutes) with Escape/`/stop` having zero effect. There is no way to
    abort an in-flight HTTP request from Python, so this doesn't kill the
    underlying call — it stops the executor from waiting on it. The call
    keeps running on its own thread and its result is discarded.

    Returns ``None`` when cancelled before a response arrived.
    """
    import concurrent.futures

    ex = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    fut = ex.submit(config._invoke_with_retry, llm, messages)
    try:
        while True:
            if CONTROL.cancelled():
                return None
            try:
                return fut.result(timeout=0.25)
            except concurrent.futures.TimeoutError:
                continue
    finally:
        ex.shutdown(wait=False)


# Repetition kill-switch tuning: a "unit" of at least _REP_MIN_UNIT chars
# repeated _REP_MIN_REPEATS times verbatim at the very end of the output is
# degenerate looping, not writing. Units are capped so the rolling buffer
# (and each check) stays O(1) per chunk.
_REP_MIN_UNIT = 6
_REP_MAX_UNIT = 120
_REP_MIN_REPEATS = 4


def _tail_repetition(text: str) -> bool:
    """True when ``text`` ends in >= _REP_MIN_REPEATS verbatim repeats of one unit.

    Degenerate generation loops are exactly periodic ("I will now fix it. I
    will now fix it. ..."), so a strict endswith(unit * N) check catches them
    while leaving legitimately repetitive text (tables, banners) alone —
    units with fewer than 3 distinct characters (``====``, whitespace runs)
    are ignored entirely.
    """
    if len(text) < _REP_MIN_UNIT * _REP_MIN_REPEATS:
        return False
    tail = text[-(_REP_MAX_UNIT * _REP_MIN_REPEATS) :]
    max_unit = min(_REP_MAX_UNIT, len(tail) // _REP_MIN_REPEATS)
    for unit_len in range(_REP_MIN_UNIT, max_unit + 1):
        unit = tail[-unit_len:]
        if len(set(unit.strip() or unit)) < 3:
            continue
        if tail.endswith(unit * _REP_MIN_REPEATS):
            return True
    return False


def _stream_invoke(
    llm, messages, *, display: bool = True, guard: bool = False
) -> tuple[BaseMessage | None, bool, bool]:
    """Invoke the model with streaming; optionally display and/or guard.

    Returns ``(response, streamed_text, aborted)``. Chunks are aggregated
    with ``+`` so the final message carries merged tool_calls/usage exactly
    like invoke. Falls back to a normal retry-invoke when the provider can't
    stream. Checks the cancel flag between chunks so Escape lands mid-answer.

    ``guard=True`` watches a rolling tail buffer for verbatim self-repetition
    and closes the stream the moment it appears — the server stops generating,
    so a doomed reply costs seconds instead of the token limit. The partial
    message is returned with ``aborted=True``; the caller decides what to do
    with it.

    Display chunks go through the CONTROL event bus (``llm_chunk``/
    ``llm_done``) when a listener is registered — the TUI renders them live.
    Without a listener (plain CLI), they fall back to stdout.
    """
    import sys as _sys

    try:
        full = None
        emitted = False
        aborted = False
        _buf = ""
        for chunk in llm.stream(messages):
            full = chunk if full is None else full + chunk
            if CONTROL.cancelled():
                break
            content = getattr(chunk, "content", "") or ""
            if isinstance(content, list):
                content = "".join(
                    b.get("text", "") for b in content if isinstance(b, dict)
                )
            if guard and content:
                _buf = (_buf + content)[-(_REP_MAX_UNIT * _REP_MIN_REPEATS) :]
                if _tail_repetition(_buf):
                    aborted = True
                    break
            if display and content:
                if not CONTROL.emit("llm_chunk", content):
                    if not emitted:
                        _sys.stdout.write("\n")
                    _sys.stdout.write(content)
                    _sys.stdout.flush()
                emitted = True
        if emitted and not CONTROL.emit("llm_done"):
            _sys.stdout.write("\n")
            _sys.stdout.flush()
        if full is None:
            if CONTROL.cancelled():
                return None, False, False
            return _invoke_cancelable(llm, messages), False, False
        return full, emitted, aborted
    except Exception:
        if CONTROL.cancelled():
            return None, False, False
        # Provider/transport can't stream — degrade to the retry path.
        return _invoke_cancelable(llm, messages), False, False


def _try_bind_tools(llm, toolset: list[tools.ToolDef]):
    """Try to bind native tool schemas. Returns the bound model or None."""
    try:
        schemas = tools.langchain_tool_schemas(toolset)
        return llm.bind_tools(schemas)
    except (NotImplementedError, AttributeError, TypeError, ValueError):
        # Provider/model doesn't support tool calling -> text fallback.
        return None
    except Exception:
        # Other errors (e.g. provider quirks) -> be conservative, use text mode.
        return None


def _extract_calls(resp: BaseMessage, *, native_tools: bool) -> list[dict]:
    """Pull tool calls out of a model response.

    Prefers native ``tool_calls``; falls back to parsing ``<tool_call>`` blocks
    from the text content so models without native tool support still work.
    """
    calls: list[dict] = []
    if native_tools:
        for tc in getattr(resp, "tool_calls", None) or []:
            calls.append(
                {
                    "name": tc.get("name"),
                    "args": tc.get("args") or tc.get("arguments") or {},
                    "id": tc.get("id"),
                }
            )
        if calls:
            return calls
    # Text fallback.
    text = getattr(resp, "content", "") or ""
    if isinstance(text, list):  # some providers return content blocks
        text = " ".join(b.get("text", "") for b in text if isinstance(b, dict))
    return parse_text_tool_calls(text)


def _tool_message(
    text: str, *, tool_call_id: str, name: str, ai_message: BaseMessage | None = None
) -> ToolMessage:
    """Build a ToolMessage observation, compressing long outputs."""
    if len(text.splitlines()) > 60:
        text = compress_output(text, tail_lines=60)
    return ToolMessage(content=text, tool_call_id=tool_call_id, name=name)
