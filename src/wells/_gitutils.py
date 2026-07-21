"""Shared git plumbing for internal harness operations.

Three modules (fleet, worktree, gitops) each had their own ``_git`` /
``is_git_repo`` / ``current_branch`` — three near-identical copies. This
module is the canonical home; the others re-export from here so external
callers (and tests that monkey-patch ``fleet._git`` etc.) keep working.

All operations use ``git`` directly via argv (no shell) so they bypass the
agent safety gate — these are internal harness operations, mirroring the
pattern in :mod:`gitops` snapshot/restore.
"""

from __future__ import annotations

import subprocess


def git(
    cwd: str,
    *args: str,
    timeout: float = 120.0,
    env_extra: dict | None = None,
) -> tuple[bool, str]:
    """Run ``git`` directly (argv, no shell) in ``cwd``. Returns (ok, output).

    ``output`` is stdout + stderr concatenated and stripped. On exception,
    returns ``(False, "<ExceptionType>: <message>")`` so callers get a
    useful error string instead of just the exception message.

    ``env_extra`` is merged over ``os.environ`` (used by gitops' temp-index
    snapshot trick, which sets ``GIT_INDEX_FILE``).
    """
    import os

    env = {**os.environ, **(env_extra or {})}
    try:
        proc = subprocess.run(
            ["git", *args], cwd=cwd, capture_output=True, text=True,
            timeout=timeout, env=env,
        )
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"
    out = (proc.stdout or "") + (proc.stderr or "")
    return proc.returncode == 0, out.strip()


def is_git_repo(path: str) -> bool:
    ok, out = git(path, "rev-parse", "--is-inside-work-tree")
    return ok and "true" in out.lower()


def current_branch(path: str) -> str:
    ok, out = git(path, "rev-parse", "--abbrev-ref", "HEAD")
    return out.strip() if ok else ""


def head_sha(path: str) -> str:
    ok, out = git(path, "rev-parse", "HEAD")
    return out.strip() if ok else ""
