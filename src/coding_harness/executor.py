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

from coding_harness import config, tools
from coding_harness.compress import compress_output
from coding_harness.control import CONTROL, ui
from coding_harness.tokens import LEDGER, estimate_tokens


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
                info = dict(
                    line.strip().split("=", 1)
                    for line in fh
                    if "=" in line
                )
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
        "git", "az", "aws", "gcloud",
        "docker", "kubectl", "helm", "terraform",
        "npm", "node", "bun", "deno",
        "python", "python3", "pip", "uv",
        "cargo", "rustc", "go",
        "java", "mvn", "gradle",
        "dotnet",
        "curl", "wget",
        "gh", "hub",
        "make", "cmake",
        "jq", "yq",
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
    stopped_reason: str = "done"  # done | max_steps | error | cancelled | budget
    messages: list[BaseMessage] = field(default_factory=list)
    # True when the final answer was already streamed to the console live
    # (callers should not print result.summary again).
    streamed: bool = False


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


def _system_prompt(task: str, toolset: list[tools.ToolDef], *, plan_mode: bool,
                   workspace: str | None = None) -> str:
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
    from coding_harness.principles import inject_into_prompt as inject_principles
    env = _build_env_context()

    # Workspace operating rules (RULES.md) + any open liabilities. The
    # machine-checkable subset is ALSO enforced at the tool boundary — this
    # block is the always-visible layer.
    rules_block = ""
    try:
        from coding_harness import rules as _rules
        if workspace:
            rules_block = _rules.engine_for(workspace).prompt_block()
    except Exception:
        pass
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

TOOL CALLING:
- If your runtime exposes native tool/function calls, use them.
- Otherwise, emit each call on its own line as: <tool_call>{{"name": "...", "args": {{...}}}}</tool_call>
  and nothing else on that line. The harness will execute it and reply with the result.
- Batch related shell commands: chain multiple az/git/curl calls with semicolons in ONE
  run_command call rather than calling run_command once per command. This cuts round trips
  and token cost dramatically.
- The harness injects REQUESTS_CA_BUNDLE / SSL_CERT_FILE / CURL_CA_BUNDLE automatically
  into every subprocess. Do NOT prepend $env:REQUESTS_CA_BUNDLE=... yourself — it's already
  set. If a command fails with an SSL/cert error, report it rather than retrying manually.
- If the same operation has failed 3+ times with similar errors, STOP and report what is
  blocking you rather than continuing to retry.
"""
    return inject_principles(base, workspace)


# ---------------------------------------------------------------------------
# Tool-call parsing (text fallback)
# ---------------------------------------------------------------------------

_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)


def parse_text_tool_calls(text: str) -> list[dict]:
    """Parse ``<tool_call>{...}</tool_call>`` blocks from model text.

    Returns a list of ``{"name": ..., "args": {...}}`` dicts. Malformed blocks
    are skipped (the model will be told the parse error in the observation).
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
    return calls


# ---------------------------------------------------------------------------
# Token accounting helper
# ---------------------------------------------------------------------------


def _account_usage(
    *, step: str, model: str, messages: list[BaseMessage], resp: BaseMessage,
    saved_by_trim: int = 0,
) -> None:
    """Record token usage for one executor round into the global ledger."""
    um = getattr(resp, "usage_metadata", None) or {}
    full = "\n".join(getattr(m, "content", "") or "" for m in messages)
    input_tokens = um.get("input_tokens") or estimate_tokens(full)
    output_tokens = um.get("output_tokens") or estimate_tokens(
        getattr(resp, "content", "") or ""
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

    def update_from_tool(self, name: str, args: dict, result_text: str, ok: bool) -> None:
        path = (
            args.get("path") or args.get("file_path")
            or args.get("filename") or args.get("filepath") or ""
        )
        if name in {"read_file", "view_file", "cat"} and path:
            if path not in self.files_read:
                self.files_read.append(path)
        elif name in {"write_file", "edit_file", "patch_file", "str_replace_editor",
                      "str_replace", "create_file"} and path and ok:
            if path not in self.files_modified:
                self.files_modified.append(path)
            if path not in self.files_read:
                self.files_read.append(path)
        elif name in {"run_tests", "pytest", "test"}:
            lines = [l.strip() for l in result_text.splitlines() if l.strip()]
            for line in lines:
                low = line.lower()
                if any(k in low for k in ("passed", "failed", "error", "ok", "tests ran")):
                    self.test_status = line[:200]
                    break
            else:
                self.test_status = lines[0][:200] if lines else ""
        elif name in {"run_command", "bash", "shell"} and not ok:
            cmd = str(args.get("command", ""))[:60]
            err_lines = result_text.strip().splitlines()
            err = err_lines[-1][:80] if err_lines else ""
            entry = f"{cmd} → {err}"
            if entry not in self.failed_commands:
                self.failed_commands.append(entry)

    def is_empty(self) -> bool:
        return not any([self.files_modified, self.files_read,
                        self.failed_commands, self.test_status,
                        self.open_liabilities])

    def to_xml(self) -> str:
        parts = [_WM_TAG]
        if self.task:
            parts.append(f"  <task>{self.task[:300]}</task>")
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


def _inject_wm(messages: list[BaseMessage], wm: WorkingMemory) -> list[BaseMessage]:
    """Insert or replace the working memory HumanMessage (never pruned).

    Position: immediately after the first HumanMessage (the task).
    """
    if wm.is_empty():
        return messages
    wm_msg = HumanMessage(content=wm.to_xml())
    result = list(messages)
    for i, m in enumerate(result):
        if isinstance(m, HumanMessage) and (m.content or "").startswith(_WM_TAG):
            result[i] = wm_msg
            return result
    # Not found — insert after first HumanMessage
    for i, m in enumerate(result):
        if isinstance(m, HumanMessage):
            result.insert(i + 1, wm_msg)
            return result
    result.append(wm_msg)
    return result


# ---------------------------------------------------------------------------
# Observation masking (primary context management)
# ---------------------------------------------------------------------------

# Keep this many most-recent AI+Tool rounds verbatim; replace content in older ones.
_MASK_KEEP_ROUNDS = int(__import__("os").environ.get("WELLS_KEEP_ROUNDS", "4"))
# Only mask tool outputs larger than this many estimated tokens (small ones are cheap).
_MASK_MIN_TOKENS = int(__import__("os").environ.get("WELLS_MASK_MIN", "120"))
# Absolute drop threshold — safety valve, fires only when masking isn't enough.
_DROP_THRESHOLD = int(__import__("os").environ.get("WELLS_CTX_LIMIT", "18000"))
_DROP_TARGET = int(__import__("os").environ.get("WELLS_CTX_TARGET", "12000"))


def _ctx_tokens(messages: list[BaseMessage]) -> int:
    """Rough token estimate for the full message list."""
    total = 0
    for m in messages:
        content = getattr(m, "content", "") or ""
        if isinstance(content, list):
            content = " ".join(b.get("text", "") for b in content if isinstance(b, dict))
        total += estimate_tokens(content)
        for tc in getattr(m, "tool_calls", None) or []:
            total += estimate_tokens(str(tc.get("args", {})))
    return total


def _mask_tool_result(name: str, args: dict, content: str) -> str:
    """Compress a large tool result to a typed 1-line summary.

    Type-aware so each summary carries the most useful signal for its tool kind.
    Index tools (find_symbol, etc.) are never masked — their output is already compact.
    """
    path = (
        args.get("path") or args.get("file_path")
        or args.get("filename") or args.get("pattern") or ""
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
    elif name in {"edit_file", "patch_file", "str_replace_editor", "str_replace", "create_file"}:
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
    elif name.startswith("find_") or name.startswith("search_") or name.startswith("list_symbol"):
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
            content = " ".join(b.get("text", "") for b in content if isinstance(b, dict))
        if estimate_tokens(content) <= _MASK_MIN_TOKENS:
            continue
        # Skip if already a 1-liner summary
        if content.startswith("[") and "\n" not in content.strip():
            continue
        name, args = tool_meta.get(m.tool_call_id or "", ("", {}))
        masked = _mask_tool_result(name, args, content)
        saved += estimate_tokens(content) - estimate_tokens(masked)
        result[i] = ToolMessage(content=masked, tool_call_id=m.tool_call_id, name=m.name)

    return result, max(0, saved)


def _safety_drop(messages: list[BaseMessage]) -> tuple[list[BaseMessage], int]:
    """Absolute last resort: drop complete oldest rounds when context still exceeds limit.

    Should rarely fire after observation masking. Indicates either an extremely long
    run or pathologically large tool outputs that couldn't be masked enough.
    """
    total = _ctx_tokens(messages)
    if total <= _DROP_THRESHOLD or len(messages) <= 3:
        return messages, 0

    # Protect head: system message + task HumanMessage + optional WM HumanMessage
    head_count = 0
    seen_human = False
    for m in messages:
        head_count += 1
        if isinstance(m, HumanMessage):
            if not seen_human:
                seen_human = True  # first HumanMessage = task
            elif (m.content or "").startswith(_WM_TAG):
                pass  # WM message — also protect
            else:
                head_count -= 1  # not WM, stop protecting here
                break

    head = messages[:head_count]
    tail = list(messages[head_count:])
    saved = 0

    while tail and (total - saved) > _DROP_TARGET:
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
    note = HumanMessage(content=f"[safety drop: ~{saved:,} tokens of old history removed]")
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
            if (isinstance(block, dict) and block.get("type") == "text") or isinstance(block, str)
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
    return re.sub(r'^(\$env:\w+=(?:\'[^\']*\'|"[^"]*");\s*)+', '', cmd).strip()


def _short_path(path: str) -> str:
    """Shorten a path relative to the workspace root for display."""
    p = str(path or "?")
    try:
        ws = config.WORKSPACE_ROOT.replace("\\", "/")
        p2 = p.replace("\\", "/")
        if p2.startswith(ws):
            p = p2[len(ws):].lstrip("/")
    except Exception:
        pass
    return ("…" + p[-55:]) if len(p) > 58 else p


def _activity_line(name: str, args: dict, ok: bool, simulated: bool = False) -> str:
    """Format one tool call as a compact human-readable line."""
    check = "[dim](plan)[/dim]" if simulated else ("[green]✓[/green]" if ok else "[red]✗[/red]")

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
        sym = str(args.get("name") or args.get("symbol") or args.get("query") or "")[:45]
        label = {"find_symbol": "find", "find_callers": "callers", "search_symbols": "search"}.get(name, name)
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
        cmd = _strip_env_prefix(cmd)   # remove $env:VAR='...'; boilerplate
        if len(cmd) > 70:
            cmd = cmd[:67] + "…"
        desc = f"[cyan]run     [/cyan] {cmd}"
    elif name == "run_tests":
        cmd = str(args.get("command") or "auto-detect")[:60]
        desc = f"[cyan]test    [/cyan] {cmd}"

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
    toolset: list[tools.ToolDef] | None = None,
    max_steps: int | None = None,
    profile: str | None = None,
    temperature: float = 0.2,
    system_prefix: str = "",
    seed_messages: list[BaseMessage] | None = None,
    step_label: str = "executor",
    quiet: bool = False,
    stream: bool = False,
) -> ExecutorResult:
    """Run the agentic tool-calling loop until completion or the step cap.

    Parameters
    ----------
    task : str
        Natural-language task for the executor (the "what to do").
    ctx : ToolContext
        Workspace/safety/plan-mode context threaded into every tool call.
    toolset : list[ToolDef], optional
        Tools the model may call. Defaults to all tools (read+write+exec); pass
        ``tools.registry(include_mutating=False)`` for read-only subagents.
    max_steps : int, optional
        Cap on tool-call rounds. Defaults to ``config.MAX_TOOL_STEPS``.
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
    cap = max_steps if max_steps is not None else config.MAX_TOOL_STEPS
    profile = profile or config.ACTIVE_PROFILE

    system = _system_prompt(task, toolset, plan_mode=ctx.plan_mode, workspace=ctx.workspace)
    if system_prefix:
        system = f"{system_prefix}\n\n{system}"

    messages: list[BaseMessage] = [SystemMessage(content=system)]
    if seed_messages:
        messages.extend(seed_messages)
    else:
        messages.append(HumanMessage(content=f"Please complete this task:\n{task}"))

    # Try to bind native tool schemas; fall back to text mode if unsupported.
    llm = (
        config.get_llm_for_task("coding", temperature=temperature)
        if profile == config.ACTIVE_PROFILE
        else config.providers.get_chat_model(profile, temperature=temperature)
    )
    bound = _try_bind_tools(llm, toolset)
    native_tools = bound is not None
    if native_tools:
        llm = bound

    model_label = config.providers.load_profile(profile)
    model_name = model_label.label() if model_label else profile

    wm = WorkingMemory(task=task)
    _tool_meta: dict[str, tuple[str, dict]] = {}  # tool_call_id → (name, args)

    rules_engine = None
    if config.RULES_ENFORCE:
        try:
            from coding_harness import rules as _rules_mod
            rules_engine = _rules_mod.engine_for(ctx.workspace)
        except Exception:
            rules_engine = None

    history: list[dict] = []
    steps = 0
    rounds = 0
    total_saved = 0
    _read_ranges: dict[str, list[tuple[int, int]]] = {}  # path → [(offset, end), ...]
    _fail_patterns: dict[str, int] = {}                  # command prefix → fail count

    def _stopped(reason: str, summary: str) -> ExecutorResult:
        return ExecutorResult(
            summary=summary,
            steps_taken=steps,
            tool_calls=history,
            stopped_reason=reason,
            messages=messages,
        )

    budget = config.MAX_RUN_TOKENS
    budget_warned = False

    while steps < cap:
        # ── Cooperative cancellation (set by the TUI on Escape) ─────────────
        if CONTROL.cancelled():
            return _stopped("cancelled", "(cancelled by user)")

        # ── Per-run token budget (ledger is reset at run start) ─────────────
        if budget:
            t = LEDGER.totals()
            used = t["input"] + t["output"]
            if used >= budget:
                _ui("warn", f"\n[bold red]Token budget reached "
                           f"({used:,}/{budget:,}) — stopping.[/bold red]")
                return _stopped(
                    "budget",
                    f"(stopped: token budget of {budget:,} reached at {used:,} tokens)",
                )
            if not budget_warned and used >= 0.8 * budget:
                budget_warned = True
                _ui("warn", f"\n[yellow]⚠ {used:,} of {budget:,} run tokens used "
                           f"({used * 100 // budget}%).[/yellow]")

        # ── Mid-run steering: user /steer notes land in the next round ──────
        steers = CONTROL.drain_steers()
        if steers:
            for s in steers:
                messages.append(HumanMessage(content=(
                    f"[USER STEER — read this NOW, mid-task instruction]: {s}\n"
                    f"Adjust your current approach accordingly before continuing."
                )))
                _ui("warn", f"  [bold cyan]⮕ steer delivered:[/bold cyan] {s[:90]}")

        # ── Context management pipeline (order matters) ──────────────────────
        messages = _inject_wm(messages, wm)
        messages, mask_saved = _apply_observation_masking(messages, _tool_meta)
        messages, drop_saved = _safety_drop(messages)
        saved = mask_saved + drop_saved
        if saved:
            total_saved += saved
        # ─────────────────────────────────────────────────────────────────────

        rounds += 1
        _act(f"thinking · round {rounds} · step {steps}/{cap}")
        streamed_this_round = False
        try:
            if stream:
                resp, streamed_this_round = _stream_invoke(llm, messages)
            else:
                resp = config._invoke_with_retry(llm, messages)
        except Exception as e:
            return ExecutorResult(
                summary=f"[executor error: {type(e).__name__}: {e}]",
                steps_taken=steps,
                tool_calls=history,
                stopped_reason="error",
                messages=messages,
            )
        _account_usage(step=step_label, model=model_name, messages=messages, resp=resp,
                       saved_by_trim=saved)
        messages.append(resp)

        calls = _extract_calls(resp, native_tools=native_tools)

        # Show the LLM's reasoning text (if any) before we execute the tool calls.
        # This tells the user WHY the agent is doing what it's about to do.
        llm_text = _extract_llm_text(resp)
        if llm_text and not streamed_this_round:
            # Trim to ~200 chars; collapse newlines so it reads as one line.
            preview = " ".join(llm_text.split())[:200]
            if len(llm_text) > 200:
                preview += "…"
            _ui("llm_text", f"\n[dim italic]{preview}[/dim italic]")
        elif calls and not llm_text:
            # No narrative text — at least show the round number so the user
            # knows progress is being made.
            _ui("round", f"\n[dim]Round {rounds}  (step {steps + 1}/{cap})[/dim]")

        if not calls:
            # No tool calls = final answer.
            if llm_text and not streamed_this_round:
                _ui("round", "")  # blank line before final summary
            return ExecutorResult(
                summary=llm_text or "(no output)",
                steps_taken=steps,
                tool_calls=history,
                stopped_reason="done",
                messages=messages,
                streamed=streamed_this_round and bool(llm_text),
            )

        for call in calls:
            if CONTROL.cancelled():
                return _stopped("cancelled", "(cancelled by user)")
            steps += 1
            name = call.get("name")
            args = call.get("args") or {}
            tcid = call.get("id") or f"call_{steps}"

            if not name:
                obs_text = "[error: malformed tool call — missing name]"
                history.append({"name": "?", "args": args, "ok": False,
                                "output_preview": obs_text})
                messages.append(
                    _tool_message(obs_text, tool_call_id="parseerr", name="parse_error")
                )
                continue

            _tool_meta[tcid] = (name, args)

            # ── Rules gate: deterministic enforcement before execution ─────
            rule_notes: list[str] = []
            decision = None
            if rules_engine is not None:
                decision = rules_engine.check(name, args)
                rule_notes = list(decision.notes)
                if not decision.allow:
                    obs_text = "\n".join(rule_notes) or "[RULES: action blocked]"
                    _ui("warn", f"  [bold red]⛔ rule "
                                f"{decision.rule.id if decision.rule else '?'} "
                                f"blocked {name}[/bold red]")
                    history.append({"name": name, "args": args, "ok": False,
                                    "output_preview": obs_text[:200]})
                    messages.append(_tool_message(
                        obs_text, tool_call_id=tcid, name=name, ai_message=resp))
                    continue
                if decision.confirm:
                    from coding_harness import safety as _safety
                    ap = ctx.approver or _safety.get_approver()
                    rid = decision.rule.id if decision.rule else "rule"
                    detail = str(args.get("command") or args)[:160]
                    approved = bool(ap(f"rule:{rid}", detail)) if ap else False
                    if not approved:
                        why = ("denied by user" if ap else
                               "no approver available — confirm-severity rules "
                               "require an interactive session")
                        obs_text = (
                            f"[RULES {rid}: action NOT executed ({why}). "
                            f"{decision.rule.message if decision.rule else ''} "
                            f"Choose a compliant alternative or ask the user.]"
                        )
                        _ui("warn", f"  [yellow]⛔ rule {rid}: {name} not "
                                    f"executed ({why})[/yellow]")
                        history.append({"name": name, "args": args, "ok": False,
                                        "output_preview": obs_text[:200]})
                        messages.append(_tool_message(
                            obs_text, tool_call_id=tcid, name=name, ai_message=resp))
                        continue

            _act(f"{name} · step {steps}/{cap}")
            result = tools.dispatch(name, args, ctx)
            obs_text = result.to_model_text()
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

            history.append({
                "name": name,
                "args": args,
                "ok": result.ok,
                "output_preview": (result.output or result.error or "")[:200],
                "simulated": result.simulated,
            })

            # ── Re-read detection ──────────────────────────────────────────
            # Track which line ranges of each file have been read.
            # On the 2nd+ read, inject a harness note INTO obs_text so the
            # model actually sees it (a terminal-only warning is invisible to
            # the model and has no effect on its behaviour).
            if name == "read_file":
                raw_path = str(args.get("path", ""))
                fpath = _short_path(raw_path)
                offset = int(args.get("offset") or 1)
                limit  = int(args.get("limit")  or 2000)
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
                    _ui("warn", f"  [yellow]⚠ {lbl} {fpath} — consider grep or find_symbol instead[/yellow]")

            # ── Self-heal: fast checker after every successful write/edit ──
            # Ground truth in the same round: a syntax error or undefined name
            # reaches the model immediately instead of surfacing a full tester
            # loop (an LLM call) later.
            if (
                config.SELF_CHECK
                and result.ok
                and not result.simulated
                and name in ("write_file", "edit_file", "create_file", "patch_file",
                             "str_replace_editor", "str_replace")
            ):
                _checked_path = str(
                    args.get("path") or args.get("file_path")
                    or args.get("filename") or ""
                )
                if _checked_path:
                    from coding_harness import checkers
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
                name=name, ok=result.ok, simulated=result.simulated,
            )

            messages.append(
                _tool_message(obs_text, tool_call_id=tcid, name=name, ai_message=resp)
            )

        if steps >= cap:
            # One final round: apply full context pipeline then ask for summary.
            try:
                messages = _inject_wm(messages, wm)
                messages, ms = _apply_observation_masking(messages, _tool_meta)
                messages, ds = _safety_drop(messages)
                final_saved = ms + ds
                final = config._invoke_with_retry(llm, messages)
                _account_usage(step=step_label, model=model_name, messages=messages,
                               resp=final, saved_by_trim=final_saved)
                messages.append(final)
                return ExecutorResult(
                    summary=(getattr(final, "content", "") or "").strip()
                            or "(reached step cap)",
                    steps_taken=steps,
                    tool_calls=history,
                    stopped_reason="max_steps",
                    messages=messages,
                )
            except Exception:
                return ExecutorResult(
                    summary="(reached step cap)",
                    steps_taken=steps,
                    tool_calls=history,
                    stopped_reason="max_steps",
                    messages=messages,
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


def _stream_invoke(llm, messages) -> tuple[BaseMessage, bool]:
    """Invoke the model with live token streaming to stdout.

    Returns ``(response, streamed_text)``. Chunks are aggregated with ``+`` so
    the final message carries merged tool_calls/usage exactly like invoke.
    Falls back to a normal retry-invoke when the provider can't stream.
    Checks the cancel flag between chunks so Escape lands mid-answer.
    """
    import sys as _sys

    try:
        full = None
        emitted = False
        for chunk in llm.stream(messages):
            full = chunk if full is None else full + chunk
            if CONTROL.cancelled():
                break
            content = getattr(chunk, "content", "") or ""
            if isinstance(content, list):
                content = "".join(
                    b.get("text", "") for b in content if isinstance(b, dict)
                )
            if content:
                if not emitted:
                    _sys.stdout.write("\n")
                _sys.stdout.write(content)
                _sys.stdout.flush()
                emitted = True
        if emitted:
            _sys.stdout.write("\n")
            _sys.stdout.flush()
        if full is None:
            return config._invoke_with_retry(llm, messages), False
        return full, emitted
    except Exception:
        # Provider/transport can't stream — degrade to the retry path.
        return config._invoke_with_retry(llm, messages), False


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
