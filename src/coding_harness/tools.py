"""Repo tools: read/list/glob/grep + write/edit + shell/tests.

This is the layer that makes the harness *act*. Every tool is a small, typed
function that goes through :mod:`coding_harness.safety` (path confinement +
safety gate) before touching the filesystem or running a command. Tools are
described as (name, description, JSON schema, handler) triples so they can be
bound to any chat model that supports tool calling, and dispatched uniformly
from the agentic executor loop in :mod:`coding_harness.executor`.

Design notes:
  * Tools are stateless; runtime context (workspace, safety, approver) lives in
    :class:`ToolContext`, passed to every call. This keeps tools trivial to test.
  * Reads/searches are always allowed (they cannot mutate state). Only write and
    shell tools are gated by the safety policy.
  * Tool outputs are plain strings, compressed via the harness compressor before
    being fed back to the model (the executor handles that).
"""

from __future__ import annotations

import base64
import fnmatch
import os
import platform
import re
import shutil
import subprocess
import time as _time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from coding_harness import safety
from coding_harness.compress import compress_output


# ---------------------------------------------------------------------------
# Subprocess environment helpers
# ---------------------------------------------------------------------------

_cached_win_bundle: str | None = None


def _windows_cert_bundle() -> str | None:
    """Export Windows trusted root certs to a PEM file and return its path.

    Uses ssl.enum_certificates() (Windows-only) to pull certs directly from
    the Windows cert store — the same store that az, curl, and browsers use.
    Cached for 24 h in the temp dir so it's only generated once per day.

    Falls back to the certifi bundle if the Windows store isn't accessible.
    """
    global _cached_win_bundle
    if _cached_win_bundle and os.path.exists(_cached_win_bundle):
        return _cached_win_bundle

    cache = Path(os.environ.get("TEMP", "/tmp")) / "wells_winroots.pem"
    if cache.exists() and (_time.time() - cache.stat().st_mtime) < 86400:
        _cached_win_bundle = str(cache)
        return _cached_win_bundle

    try:
        import ssl
        pem: list[bytes] = []
        for store in ("ROOT", "CA"):
            try:
                for cert, enc, trust in ssl.enum_certificates(store):  # type: ignore[attr-defined]
                    if enc == "x509_asn" and trust is True:
                        b64 = base64.encodebytes(cert)
                        pem += [b"-----BEGIN CERTIFICATE-----\n", b64,
                                b"-----END CERTIFICATE-----\n"]
            except Exception:
                continue
        if pem:
            cache.write_bytes(b"".join(pem))
            _cached_win_bundle = str(cache)
            return _cached_win_bundle
    except Exception:
        pass

    # Fallback: certifi (covers public Azure CAs; misses corporate roots)
    try:
        import certifi
        _cached_win_bundle = certifi.where()
        return _cached_win_bundle
    except Exception:
        pass

    return None


def _subprocess_env() -> dict[str, str]:
    """Build env for shell subprocesses.

    On Windows: injects the Windows cert store as REQUESTS_CA_BUNDLE /
    SSL_CERT_FILE / CURL_CA_BUNDLE so that tools like `az` (which use the
    Python requests library internally) trust the same root CAs as the user's
    native shell — no manual $env:REQUESTS_CA_BUNDLE= prefix needed.

    Strips any Wells-internal certifi path that might have leaked in via
    config._configure_ca_bundle(), since the Windows store PEM is better.
    """
    env = os.environ.copy()

    # Remove any certifi path Wells set at startup; replace with Windows store.
    try:
        import certifi
        certifi_path = certifi.where()
        for var in ("SSL_CERT_FILE", "REQUESTS_CA_BUNDLE", "CURL_CA_BUNDLE"):
            if env.get(var) == certifi_path:
                env.pop(var, None)
    except Exception:
        pass

    # On Windows, auto-inject the Windows cert store bundle.
    if _ON_WINDOWS and not env.get("REQUESTS_CA_BUNDLE"):
        bundle = _windows_cert_bundle()
        if bundle:
            env["REQUESTS_CA_BUNDLE"] = bundle
            env.setdefault("SSL_CERT_FILE", bundle)
            env.setdefault("CURL_CA_BUNDLE", bundle)

    return env


# Cached PowerShell executable path (None = not found, falls back to cmd.exe).
_PWSH: str | None = shutil.which("pwsh") or shutil.which("powershell")
_ON_WINDOWS: bool = platform.system() == "Windows"


def _run_shell(command: str, cwd: str, timeout: float) -> subprocess.CompletedProcess:
    """Run *command* in the appropriate shell for the current OS.

    On Windows: use PowerShell (pwsh/powershell) so commands the agent
    generates — which assume PS syntax ($env:VAR, Select-String, ;-chains) —
    actually work. cmd.exe is the shell=True default on Windows and silently
    misinterprets most PS syntax.

    On Linux/macOS: use shell=True (bash/sh) as before.
    """
    env = _subprocess_env()
    if _ON_WINDOWS and _PWSH:
        return subprocess.run(
            [_PWSH, "-NoProfile", "-NonInteractive", "-Command", command],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,
        )
    return subprocess.run(
        command,
        shell=True,
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=timeout,
        env=env,
    )


def _get_index_tools() -> list:
    """Lazy import of index tools (avoids tools <-> index_tools circular import)."""
    try:
        from coding_harness.index_tools import INDEX_TOOLS

        return INDEX_TOOLS
    except Exception:
        return []


# ---------------------------------------------------------------------------
# Tool context
# ---------------------------------------------------------------------------


@dataclass
class ToolContext:
    """Runtime context threaded through every tool call."""

    workspace: str
    safety: str = "auto"
    approver: safety.Approver = None
    shell_timeout: float = 120.0
    # Max bytes/lines a read tool returns before truncating (keeps context small).
    read_max_lines: int = 2000
    # In plan mode, write/shell tools describe instead of executing.
    plan_mode: bool = False
    # True inside a spawned subagent — blocks recursive subagent spawning.
    subagent: bool = False

    @classmethod
    def from_state(cls, state: dict) -> "ToolContext":
        """Build a ToolContext from a (Typed)dict agent state."""
        return cls(
            workspace=state.get("workspace_root")
            or os.environ.get("WORKSPACE_ROOT")
            or os.getcwd(),
            safety=state.get("safety") or safety.policy(),
            shell_timeout=float(os.environ.get("SHELL_TIMEOUT", "120")),
            plan_mode=bool(state.get("plan_mode")),
        )


# ---------------------------------------------------------------------------
# Tool descriptor
# ---------------------------------------------------------------------------


@dataclass
class ToolDef:
    """A tool: name, description, JSON schema, and handler."""

    name: str
    description: str
    input_schema: dict[str, Any]
    handler: Callable[..., "ToolResult"]
    # If True, the tool mutates state and is subject to the safety gate.
    mutating: bool = False


@dataclass
class ToolResult:
    """Standard tool return value."""

    ok: bool
    output: str
    error: str = ""
    # True when the tool described an action without performing it (dry-run / plan).
    simulated: bool = False

    def to_model_text(self, *, compress: bool = True) -> str:
        """Render as compact text for the model (optionally compressed)."""
        prefix = "[DRY-RUN] " if self.simulated else ("[ERROR] " if not self.ok else "")
        body = self.error if not self.ok and self.error else self.output
        if compress and len(body.splitlines()) > 40:
            body = compress_output(body)
        return f"{prefix}{body}"


# ---------------------------------------------------------------------------
# Read-only tools
# ---------------------------------------------------------------------------


def _read_file(
    ctx: ToolContext, path: str, *, offset: int = 1, limit: int | None = None
) -> ToolResult:
    """Read a text file (1-indexed line numbers, like the harness Read tool)."""
    try:
        resolved = safety.resolve_path(path, ctx.workspace)
    except (safety.PathEscapeError, ValueError) as e:
        return ToolResult(False, "", str(e))
    if not resolved.exists():
        return ToolResult(False, "", f"File not found: {path}")
    if resolved.is_dir():
        return ToolResult(False, "", f"Path is a directory, not a file: {path}")
    try:
        text = resolved.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        return ToolResult(False, "", f"Could not read {path}: {e}")

    lines = text.splitlines()
    max_lines = ctx.read_max_lines
    effective_limit = limit if limit is not None else max_lines
    start = max(1, offset) - 1
    end = start + effective_limit
    selected = lines[start:end]
    numbered = "\n".join(f"{i}: {ln}" for i, ln in enumerate(selected, start=start + 1))
    if len(lines) > end:
        numbered += f"\n({len(lines) - end} more lines; use offset to continue)"
    return ToolResult(True, numbered)


def _list_dir(ctx: ToolContext, path: str = ".") -> ToolResult:
    """List directory entries (dirs get a trailing slash)."""
    try:
        resolved = safety.resolve_path(path, ctx.workspace)
    except (safety.PathEscapeError, ValueError) as e:
        return ToolResult(False, "", str(e))
    if not resolved.exists():
        return ToolResult(False, "", f"Not found: {path}")
    if not resolved.is_dir():
        return ToolResult(False, "", f"Not a directory: {path}")
    rows = []
    try:
        for entry in sorted(
            resolved.iterdir(), key=lambda p: (p.is_file(), p.name.lower())
        ):
            if entry.name in (".git", "__pycache__", ".venv", "node_modules"):
                continue
            tag = "/" if entry.is_dir() else ""
            rows.append(f"{entry.name}{tag}")
    except Exception as e:
        return ToolResult(False, "", f"Could not list {path}: {e}")
    return ToolResult(True, "\n".join(rows) or "(empty directory)")


def _glob_tool(ctx: ToolContext, pattern: str, path: str = ".") -> ToolResult:
    """Find files matching a glob pattern (recursive)."""
    try:
        root = safety.resolve_path(path, ctx.workspace)
    except (safety.PathEscapeError, ValueError) as e:
        return ToolResult(False, "", str(e))
    if not root.exists():
        return ToolResult(False, "", f"Not found: {path}")
    matches = []
    for p in root.rglob(pattern):
        # Keep paths relative to workspace for readability + confinement.
        try:
            rel = p.relative_to(ctx.workspace if False else root)
        except ValueError:
            rel = p
        # Skip noisy dirs.
        if any(
            part in (".git", "__pycache__", ".venv", "node_modules") for part in p.parts
        ):
            continue
        matches.append(str(rel))
        if len(matches) >= 200:
            matches.append("... (truncated at 200 matches)")
            break
    matches.sort()
    return ToolResult(True, "\n".join(matches) or "(no matches)")


def _grep_tool(
    ctx: ToolContext, pattern: str, path: str = ".", include: str = ""
) -> ToolResult:
    """Search file contents for a regex; returns file:line matches."""
    try:
        root = safety.resolve_path(path, ctx.workspace)
    except (safety.PathEscapeError, ValueError) as e:
        return ToolResult(False, "", str(e))
    if not root.exists():
        return ToolResult(False, "", f"Not found: {path}")
    try:
        regex = re.compile(pattern)
    except re.error as e:
        return ToolResult(False, "", f"Bad regex {pattern!r}: {e}")

    files = [root] if root.is_file() else [p for p in root.rglob("*") if p.is_file()]
    out: list[str] = []
    for f in files:
        if any(
            part in (".git", "__pycache__", ".venv", "node_modules") for part in f.parts
        ):
            continue
        if include and not fnmatch.fnmatch(f.name, include):
            continue
        try:
            text = f.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue
        for i, line in enumerate(text.splitlines(), 1):
            if regex.search(line):
                try:
                    rel = f.relative_to(root)
                except ValueError:
                    rel = f
                out.append(f"{rel}:{i}: {line.strip()}")
                if len(out) >= 100:
                    out.append("... (truncated at 100 matches)")
                    return ToolResult(True, "\n".join(out))
    return ToolResult(True, "\n".join(out) or "(no matches)")


# ---------------------------------------------------------------------------
# Mutating tools (gated by safety policy)
# ---------------------------------------------------------------------------


def _fuzzy_locate(text: str, old_string: str) -> tuple[tuple[int, int] | None, int]:
    """Whitespace-tolerant unique match of ``old_string`` in ``text``.

    Compares line-by-line with leading/trailing whitespace stripped, so an
    old_string whose indentation the model got wrong still matches. Returns
    ``((start_line, end_line), 1)`` on a unique match (0-based, end exclusive),
    or ``(None, match_count)`` otherwise.
    """
    old_lines = [l.strip() for l in old_string.splitlines()]
    if not old_lines:
        return None, 0
    file_stripped = [l.strip() for l in text.splitlines()]
    n = len(old_lines)
    hits = [
        i
        for i in range(len(file_stripped) - n + 1)
        if file_stripped[i : i + n] == old_lines
    ]
    if len(hits) != 1:
        return None, len(hits)
    return (hits[0], hits[0] + n), 1


def _reindent(new_string: str, file_first_line: str, old_first_line: str) -> str:
    """Shift ``new_string``'s indentation by the delta between the file's actual
    first matched line and the model's old_string first line."""
    file_lead = file_first_line[: len(file_first_line) - len(file_first_line.lstrip())]
    old_lead = old_first_line[: len(old_first_line) - len(old_first_line.lstrip())]
    if file_lead == old_lead:
        return new_string
    out = []
    for line in new_string.splitlines():
        if not line.strip():
            out.append(line)
        elif old_lead and line.startswith(old_lead):
            out.append(file_lead + line[len(old_lead):])
        elif not old_lead:
            out.append(file_lead + line)
        else:
            out.append(line)
    return "\n".join(out)


_DIFF_MAX_LINES = 40


def _emit_diff(path: str, before: str, after: str) -> None:
    """Show a colorized unified diff of an applied edit in the activity log."""
    import difflib

    from rich.markup import escape

    from coding_harness.control import ui

    diff = list(
        difflib.unified_diff(
            before.splitlines(), after.splitlines(), lineterm="", n=2
        )
    )[2:]  # drop the ---/+++ header pair
    if not diff:
        return
    shown = []
    for d in diff[:_DIFF_MAX_LINES]:
        e = escape(d)
        if d.startswith("+"):
            shown.append(f"    [green]{e}[/green]")
        elif d.startswith("-"):
            shown.append(f"    [red]{e}[/red]")
        elif d.startswith("@@"):
            shown.append(f"    [cyan dim]{e}[/cyan dim]")
        else:
            shown.append(f"    [dim]{e}[/dim]")
    if len(diff) > _DIFF_MAX_LINES:
        shown.append(f"    [dim]… {len(diff) - _DIFF_MAX_LINES} more diff lines[/dim]")
    ui("diff", "\n".join(shown), path=path)


def _write_file(ctx: ToolContext, path: str, content: str) -> ToolResult:
    """Create or overwrite a file with ``content``."""
    try:
        resolved = safety.resolve_path(path, ctx.workspace)
    except (safety.PathEscapeError, ValueError) as e:
        return ToolResult(False, "", str(e))

    detail = f"write {len(content)} chars to {path}"
    if ctx.plan_mode:
        return ToolResult(True, f"[plan] would {detail}", simulated=True)
    decision = safety.gate(
        "write_file", detail, safety=ctx.safety, approver=ctx.approver
    )
    if not decision.allowed:
        return ToolResult(True, decision.reason, simulated=decision.simulated)

    before = ""
    if resolved.exists():
        try:
            before = resolved.read_text(encoding="utf-8", errors="replace")
        except Exception:
            before = ""
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text(content, encoding="utf-8")
    _emit_diff(path, before, content)
    return ToolResult(True, f"Wrote {len(content)} chars to {path}")


def _edit_file(
    ctx: ToolContext,
    path: str,
    old_string: str,
    new_string: str,
    replace_all: bool = False,
) -> ToolResult:
    """Replace ``old_string`` with ``new_string`` in ``path`` (str_replace edit).

    Mirrors the semantics of the harness's own Edit tool: fails if old_string is
    absent, or if it matches more than once without ``replace_all``.
    """
    try:
        resolved = safety.resolve_path(path, ctx.workspace)
    except (safety.PathEscapeError, ValueError) as e:
        return ToolResult(False, "", str(e))
    if not resolved.exists():
        return ToolResult(False, "", f"File not found: {path}")
    if not old_string:
        return ToolResult(
            False, "", "old_string is required (use write_file for new files)"
        )

    text = resolved.read_text(encoding="utf-8", errors="replace")
    count = text.count(old_string)
    fuzzy_note = ""
    fuzzy_span: tuple[int, int] | None = None
    if count == 0:
        # Exact match failed — try whitespace-tolerant matching so an
        # indentation slip doesn't burn a round-trip on a failed edit.
        fuzzy_span, fuzzy_count = _fuzzy_locate(text, old_string)
        if fuzzy_span is None:
            hint = (
                f" (it appears {fuzzy_count} times when ignoring whitespace; "
                "provide a more specific old_string)"
                if fuzzy_count > 1
                else ""
            )
            return ToolResult(False, "", f"old_string not found in {path}{hint}")
        count = 1
        fuzzy_note = " [fuzzy-matched: whitespace differences ignored]"
    if count > 1 and not replace_all:
        return ToolResult(
            False,
            "",
            f"old_string matches {count} times in {path}; "
            "pass replace_all=true or provide a more specific old_string",
        )

    detail = (
        f"replace 1 occurrence in {path}"
        if count == 1
        else f"replace all {count} occurrences in {path}"
    )
    if ctx.plan_mode:
        return ToolResult(True, f"[plan] would {detail}", simulated=True)
    decision = safety.gate(
        "edit_file", detail, safety=ctx.safety, approver=ctx.approver
    )
    if not decision.allowed:
        return ToolResult(True, decision.reason, simulated=decision.simulated)

    if fuzzy_span is not None:
        lines = text.splitlines(keepends=True)
        i, j = fuzzy_span
        replacement = _reindent(
            new_string, lines[i], old_string.splitlines()[0]
        )
        # Preserve the trailing newline of the matched block.
        if lines[j - 1].endswith("\n") and not replacement.endswith("\n"):
            replacement += "\n"
        new_text = "".join(lines[:i]) + replacement + "".join(lines[j:])
    else:
        new_text = (
            text.replace(old_string, new_string)
            if replace_all
            else text.replace(old_string, new_string, 1)
        )
    resolved.write_text(new_text, encoding="utf-8")
    _emit_diff(path, text, new_text)
    return ToolResult(True, f"Edited {path} ({detail}){fuzzy_note}")


def _run_command(ctx: ToolContext, command: str) -> ToolResult:
    """Run a shell command in the workspace; return compressed stdout+stderr."""
    if not command or not command.strip():
        return ToolResult(False, "", "command is required")
    try:
        safety.screen_command(command)
    except safety.BlockedCommandError as e:
        return ToolResult(False, "", str(e))

    if ctx.plan_mode:
        return ToolResult(True, f"[plan] would run: {command}", simulated=True)
    decision = safety.gate(
        "run_command", command, safety=ctx.safety, approver=ctx.approver
    )
    if not decision.allowed:
        return ToolResult(True, decision.reason, simulated=decision.simulated)

    try:
        proc = _run_shell(command, cwd=ctx.workspace, timeout=ctx.shell_timeout)
    except subprocess.TimeoutExpired:
        return ToolResult(
            False, "", f"Command timed out after {ctx.shell_timeout}s: {command}"
        )
    except Exception as e:
        return ToolResult(False, "", f"Command failed: {e}")

    combined = proc.stdout or ""
    if proc.stderr:
        combined += ("\n" if combined else "") + f"[stderr]\n{proc.stderr}"
    combined = combined or "(no output)"
    combined = f"$ {command}\n[exit {proc.returncode}]\n{combined}"
    return ToolResult(
        proc.returncode == 0,
        combined,
        "" if proc.returncode == 0 else f"exit {proc.returncode}",
    )


def _run_tests(ctx: ToolContext, command: str = "") -> ToolResult:
    """Run the project test suite. Defaults to detecting pytest/uv."""
    cmd = command.strip() or _autodetect_test_command(ctx)
    result = _run_command(ctx, cmd)
    # Re-label the action for the model's benefit (keep the simulated flag —
    # a dry-run/plan-mode result must never read as a real green suite).
    if result.ok:
        result = ToolResult(
            True,
            result.output.replace("[exit 0]", "[tests passed]"),
            "",
            simulated=result.simulated,
        )
    return result


def _autodetect_test_command(ctx: ToolContext) -> str:
    """Pick a sensible default test command based on repo layout."""
    root = Path(ctx.workspace)
    if (root / "pyproject.toml").exists() and (root / ".venv").exists():
        return "uv run python -m pytest -q"
    if (root / "pyproject.toml").exists():
        return "python -m pytest -q"
    if (root / "package.json").exists():
        return "npm test --silent"
    if (root / "Cargo.toml").exists():
        return "cargo test"
    if (root / "go.mod").exists():
        return "go test ./..."
    return "python -m pytest -q"


def _spawn_subagent(
    ctx: ToolContext, task: str, role: str = "research", max_steps: int = 8
) -> ToolResult:
    """Spawn a focused subagent (research/fix) and return its report.

    Research subagents are read-only; fix subagents may edit. The subagent runs
    in the same workspace with the same safety policy, and returns a compact
    summary the parent can act on. Lets the model delegate parallel investigation
    or scoped edits without bloating its own context.
    """
    if not task or not task.strip():
        return ToolResult(False, "", "task is required")
    if ctx.subagent:
        return ToolResult(False, "", "subagents cannot spawn subagents")
    # Local import to avoid a circular dependency at module load time.
    from coding_harness import subagents

    role = (role or "research").strip().lower()
    name = f"{role}-{abs(hash(task)) % 10000}"
    if role == "fix":
        spec = subagents.fix_subagent(name, task, max_steps=max_steps)
    else:
        spec = subagents.research_subagent(name, task, max_steps=max_steps)
    report = subagents.run_subagent(spec, ctx)
    return ToolResult(
        report.ok, report.as_context_block(), "" if report.ok else report.error
    )


def _parallel_research(
    ctx: ToolContext, questions: list | None = None, max_steps: int = 8
) -> ToolResult:
    """Run 2–4 read-only research subagents concurrently and merge their reports.

    Wall-clock win only: the questions run in parallel threads (LLM calls are
    I/O-bound), so investigation takes ~max(question) instead of the sum.
    Subagents are read-only and run quiet — the combined findings come back as
    one block. Recursion is blocked (a subagent cannot fan out again).
    """
    if ctx.subagent:
        return ToolResult(False, "", "subagents cannot spawn subagents")
    qs = [q.strip() for q in (questions or []) if isinstance(q, str) and q.strip()][:4]
    if not qs:
        return ToolResult(
            False, "", "questions is required: a list of 1-4 focused research questions"
        )

    # Local imports avoid circular dependency at module load time.
    from coding_harness import subagents
    from coding_harness.control import CONTROL, ui

    specs = [
        subagents.research_subagent(f"research-{i}", q, max_steps=max_steps)
        for i, q in enumerate(qs, 1)
    ]
    CONTROL.set_activity(f"parallel research ×{len(specs)}")
    t0 = _time.time()
    reports = subagents.dispatch_subagents(specs, ctx)
    elapsed = _time.time() - t0

    for r in reports:
        mark = "[green]✓[/green]" if r.ok else "[red]✗[/red]"
        ui("tool_line", f"    {mark} [dim]{r.name}[/dim] {r.steps_taken} steps")
    ui("tool_line", f"    [dim]{len(reports)} research agents · {elapsed:.0f}s[/dim]")

    combined = "\n\n".join(r.as_context_block() for r in reports)
    ok = any(r.ok for r in reports)
    return ToolResult(ok, combined, "" if ok else "all research subagents failed")


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

READ_TOOLS: list[ToolDef] = [
    ToolDef(
        name="read_file",
        description="Read a text file with 1-indexed line numbers. Use offset/limit for large files.",
        input_schema={
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "File path relative to workspace root",
                },
                "offset": {
                    "type": "integer",
                    "description": "1-indexed line to start at",
                    "default": 1,
                },
                "limit": {
                    "type": "integer",
                    "description": "Max lines to return",
                    "default": 2000,
                },
            },
            "required": ["path"],
        },
        handler=_read_file,
    ),
    ToolDef(
        name="list_dir",
        description="List entries in a directory (dirs get a trailing slash).",
        input_schema={
            "type": "object",
            "properties": {"path": {"type": "string", "default": "."}},
        },
        handler=_list_dir,
    ),
    ToolDef(
        name="glob",
        description="Find files matching a glob pattern (recursive).",
        input_schema={
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "Glob pattern, e.g. **/*.py",
                },
                "path": {"type": "string", "default": "."},
            },
            "required": ["pattern"],
        },
        handler=_glob_tool,
    ),
    ToolDef(
        name="grep",
        description="Search file contents for a regex; returns file:line: match lines.",
        input_schema={
            "type": "object",
            "properties": {
                "pattern": {"type": "string", "description": "Python regex"},
                "path": {"type": "string", "default": "."},
                "include": {
                    "type": "string",
                    "description": "Filename glob filter, e.g. *.py",
                },
            },
            "required": ["pattern"],
        },
        handler=_grep_tool,
    ),
    ToolDef(
        name="parallel_research",
        description="Run 2-4 focused read-only research questions CONCURRENTLY via parallel "
        "subagents and get their combined findings. Much faster than investigating "
        "sequentially when a goal spans multiple independent areas (e.g. 'how does auth "
        "work' + 'where are the API routes' + 'how is the DB accessed'). Subagents can "
        "only read — they never edit or run commands.",
        input_schema={
            "type": "object",
            "properties": {
                "questions": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "1-4 focused, independent research questions",
                },
                "max_steps": {"type": "integer", "default": 8},
            },
            "required": ["questions"],
        },
        handler=_parallel_research,
        mutating=False,
    ),
] + _get_index_tools()

WRITE_TOOLS: list[ToolDef] = [
    ToolDef(
        name="write_file",
        description="Create or overwrite a file. Creates parent directories.",
        input_schema={
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "content": {"type": "string", "description": "Full file contents"},
            },
            "required": ["path", "content"],
        },
        handler=_write_file,
        mutating=True,
    ),
    ToolDef(
        name="edit_file",
        description="Replace old_string with new_string in a file. Fails if old_string is "
        "absent or matches multiple times (unless replace_all=true).",
        input_schema={
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "old_string": {"type": "string"},
                "new_string": {"type": "string"},
                "replace_all": {"type": "boolean", "default": False},
            },
            "required": ["path", "old_string", "new_string"],
        },
        handler=_edit_file,
        mutating=True,
    ),
]

EXEC_TOOLS: list[ToolDef] = [
    ToolDef(
        name="run_command",
        description="Run a shell command in the workspace (cwd = workspace root). Returns "
        "stdout/stderr (compressed) and exit code.",
        input_schema={
            "type": "object",
            "properties": {"command": {"type": "string"}},
            "required": ["command"],
        },
        handler=_run_command,
        mutating=True,
    ),
    ToolDef(
        name="run_tests",
        description="Run the project test suite. Auto-detects pytest/npm/cargo/go if command omitted.",
        input_schema={
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "Override test command"}
            },
        },
        handler=_run_tests,
        mutating=True,
    ),
    ToolDef(
        name="spawn_subagent",
        description="Spawn a focused subagent for a research or fix task and return its summary. "
        "Research subagents are read-only; fix subagents may edit. Use this to delegate "
        "parallel investigation or scoped edits without bloating your own context.",
        input_schema={
            "type": "object",
            "properties": {
                "task": {
                    "type": "string",
                    "description": "Focused task for the subagent",
                },
                "role": {
                    "type": "string",
                    "enum": ["research", "fix"],
                    "default": "research",
                },
                "max_steps": {"type": "integer", "default": 8},
            },
            "required": ["task"],
        },
        handler=_spawn_subagent,
        mutating=False,  # the subagent's own writes go through its own safety gate
    ),
]

ALL_TOOLS: list[ToolDef] = READ_TOOLS + WRITE_TOOLS + EXEC_TOOLS

_TOOLS_BY_NAME: dict[str, ToolDef] = {t.name: t for t in ALL_TOOLS}


def get_tool(name: str) -> ToolDef | None:
    return _TOOLS_BY_NAME.get(name)


def register_external(defs: list[ToolDef]) -> None:
    """Register externally-provided tools (MCP client) into the default toolset.

    Added to ALL_TOOLS only — read-only registries (planner/subagents) are not
    extended, so external tools stay confined to the main executor loop.
    """
    for t in defs:
        if t.name not in _TOOLS_BY_NAME:
            ALL_TOOLS.append(t)
            _TOOLS_BY_NAME[t.name] = t


def registry(*, include_mutating: bool = True) -> list[ToolDef]:
    """Return the available tools. Pass include_mutating=False for a read-only set."""
    return ALL_TOOLS if include_mutating else READ_TOOLS


def langchain_tool_schemas(tools: list[ToolDef] | None = None) -> list[dict]:
    """Render tools as LangChain/JSON tool schemas for ``model.bind_tools([...])``."""
    tools = tools or ALL_TOOLS
    return [
        {
            "type": "function",
            "function": {
                "name": t.name,
                "description": t.description,
                "parameters": t.input_schema,
            },
        }
        for t in tools
    ]


def dispatch(name: str, args: dict, ctx: ToolContext) -> ToolResult:
    """Look up ``name`` and run it with ``args`` under ``ctx``."""
    tool = _TOOLS_BY_NAME.get(name)
    if tool is None:
        return ToolResult(False, "", f"Unknown tool: {name}")
    try:
        return tool.handler(ctx, **(args or {}))
    except TypeError as e:
        return ToolResult(False, "", f"Bad arguments for {name}: {e}")
    except Exception as e:
        return ToolResult(False, "", f"{name} failed: {type(e).__name__}: {e}")
