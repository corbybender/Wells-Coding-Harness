"""Fast deterministic post-edit checks (the self-heal layer).

After the agent writes or edits a file, the harness — not the model — runs the
quickest available checker for that file type and injects any failure straight
into the agent's next observation. Broken code is caught in milliseconds
instead of a full tester round-trip (an LLM call) later.

Design constraints:
  * Fast only: per-file syntax/error checks, never whole-project builds.
  * Errors only: style findings are noise here (ruff runs with --select E9,F).
  * Never blocking: any checker failure/timeout degrades to "no report".
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from functools import lru_cache
from pathlib import Path

_TIMEOUT = 15  # seconds; a per-file check that takes longer isn't "fast"
_MAX_REPORT_LINES = 15


@lru_cache(maxsize=None)
def _has(cmd: str) -> bool:
    return shutil.which(cmd) is not None


def _run(args: list[str], cwd: str) -> tuple[int, str]:
    proc = subprocess.run(
        args, cwd=cwd, capture_output=True, text=True, timeout=_TIMEOUT
    )
    out = (proc.stdout or "") + (proc.stderr or "")
    return proc.returncode, out.strip()


def _trim(report: str) -> str:
    lines = [l for l in report.splitlines() if l.strip()]
    if len(lines) > _MAX_REPORT_LINES:
        lines = lines[:_MAX_REPORT_LINES] + [f"… {len(lines) - _MAX_REPORT_LINES} more lines"]
    return "\n".join(lines)


def _check_python(path: Path, workspace: str) -> str | None:
    if _has("ruff"):
        # E9 = syntax/runtime errors, F = pyflakes (undefined names, bad imports).
        code, out = _run(
            ["ruff", "check", "--select", "E9,F", "--no-cache", str(path)], workspace
        )
        return _trim(out) if code != 0 and out else None
    # Fallback: syntax check with the interpreter itself (always available).
    code, out = _run([sys.executable, "-m", "py_compile", str(path)], workspace)
    return _trim(out) if code != 0 and out else None


def _check_json(path: Path) -> str | None:
    try:
        json.loads(path.read_text(encoding="utf-8", errors="replace"))
        return None
    except json.JSONDecodeError as e:
        return f"invalid JSON: {e}"


def _check_js(path: Path, workspace: str) -> str | None:
    if not _has("node"):
        return None
    code, out = _run(["node", "--check", str(path)], workspace)
    return _trim(out) if code != 0 and out else None


def quick_check(path: str, workspace: str) -> str | None:
    """Run the fastest available checker for ``path``.

    Returns a short error report when the file fails, else None. Never raises.
    """
    try:
        p = Path(path)
        if not p.is_absolute():
            p = Path(workspace) / p
        if not p.exists():
            return None
        ext = p.suffix.lower()
        if ext in (".py", ".pyw"):
            return _check_python(p, workspace)
        if ext == ".json":
            return _check_json(p)
        if ext in (".js", ".mjs", ".cjs"):
            return _check_js(p, workspace)
    except Exception:
        return None
    return None
