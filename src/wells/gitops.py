"""Git integration: branch, commit, diff, and (optionally) open a PR.

Runs ``git`` via the shell tool layer, so all operations stay inside the
workspace and honour the safety policy. The PR path uses the ``gh`` CLI when
available; if it is missing or the repo has no remote, we degrade gracefully to
"left changes on a local branch".

This is invoked at the end of a successful coder run (when ``review_complete``
is True) — turning the harness into an end-to-end "goal → PR" loop.

Design:
  * Every operation goes through :func:`tools.dispatch` so the same confinement,
    blocklist, and safety gate apply as to the agent's own tool calls.
  * Git is *optional* — a non-git workspace or missing ``gh`` never fails a run;
    the result simply reports what was done locally.
  * Branch names are deterministic and sanitized (``wells/<slug>``).
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from wells.tools import ToolContext


@dataclass
class GitResult:
    """Outcome of a git operation sequence."""

    ok: bool
    branch: str = ""
    commit: str = ""
    pr_url: str = ""
    pr_number: int = 0
    diff_stat: str = ""
    message: str = ""

    def summary(self) -> str:
        parts = [f"branch={self.branch or '(none)'}"]
        if self.commit:
            parts.append(f"commit={self.commit[:8]}")
        if self.diff_stat:
            parts.append(f"changes={self.diff_stat}")
        if self.pr_url:
            parts.append(f"PR={self.pr_url}")
        return (
            "git: " + " | ".join(parts) + (f" ({self.message})" if self.message else "")
        )


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _slugify(text: str, *, max_len: int = 40) -> str:
    """Turn an arbitrary goal string into a branch-safe slug."""
    s = re.sub(r"[^A-Za-z0-9._-]+", "-", text.strip().lower()).strip("-")
    s = re.sub(r"-+", "-", s)
    if not s:
        s = "task"
    ts = time.strftime("%m%d%H%M")
    return f"{s[:max_len]}-{ts}"


def _run(ctx: "ToolContext", command: str) -> tuple[bool, str]:
    """Run a git command via the tool layer; return (ok, output)."""
    from wells.tools import dispatch

    r = dispatch("run_command", {"command": command}, ctx)
    return r.ok, (r.output if r.ok else (r.error or r.output))


def _is_git_repo(ctx: "ToolContext") -> bool:
    ok, out = _run(ctx, "git rev-parse --is-inside-work-tree")
    return ok and "true" in out.lower()


def _current_branch(ctx: "ToolContext") -> str:
    ok, out = _run(ctx, "git rev-parse --abbrev-ref HEAD")
    if not ok:
        return ""
    return out.strip().splitlines()[-1].strip() if out.strip() else ""


def _has_changes(ctx: "ToolContext") -> bool:
    ok, out = _run(ctx, "git status --porcelain")
    return ok and bool(out.strip())


def _has_remote(ctx: "ToolContext") -> bool:
    ok, out = _run(ctx, "git remote")
    return ok and bool(out.strip())


def _gh_available(ctx: "ToolContext") -> bool:
    ok, _ = _run(ctx, "gh --version")
    return ok


def _safe_branch_name(slug: str) -> str:
    return f"wells/{slug}"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def create_branch_and_commit(
    ctx: "ToolContext",
    *,
    goal: str,
    message: str | None = None,
    base: str | None = None,
) -> GitResult:
    """Create a ``wells/<slug>`` branch off ``base`` and commit all changes.

    If the workspace is not a git repo, or there are no changes, returns a
    GitResult with ``ok=True`` and an explanatory message (does not fail).
    """
    if not _is_git_repo(ctx):
        return GitResult(ok=True, message="not a git repo; changes left on disk")

    branch = _safe_branch_name(_slugify(goal))

    # Off the requested base (or current branch) — never touch main directly.
    start = base or _current_branch(ctx) or "HEAD"
    ok, out = _run(ctx, f"git checkout -b {branch} {start}")
    if not ok:
        # Branch may already exist (resume); try just checking it out.
        ok2, _ = _run(ctx, f"git checkout {branch}")
        if not ok2:
            return GitResult(
                ok=False,
                message=f"could not create branch {branch}: {out.strip()[:160]}",
            )

    if not _has_changes(ctx):
        return GitResult(
            ok=True, branch=branch, message="no uncommitted changes to commit"
        )

    # Stage everything in the workspace and commit.
    _run(ctx, "git add -A")
    commit_msg = (message or f"wells: {goal.strip()[:120]}").replace('"', "'")
    ok, out = _run(ctx, f'git commit -m "{commit_msg}" --no-verify')
    if not ok:
        return GitResult(
            ok=False, branch=branch, message=f"commit failed: {(out or '')[:160]}"
        )

    commit_sha = ""
    ok2, out2 = _run(ctx, "git rev-parse HEAD")
    if ok2:
        commit_sha = out2.strip().splitlines()[-1].strip()

    ok3, diff = _run(ctx, "git diff --stat HEAD~1 HEAD")
    diff_stat = ""
    if ok3:
        # Last line of `git diff --stat` is the summary "N files changed, ...".
        lines = [ln for ln in diff.splitlines() if ln.strip()]
        if lines:
            diff_stat = lines[-1].strip()

    return GitResult(
        ok=True,
        branch=branch,
        commit=commit_sha,
        diff_stat=diff_stat,
        message="committed on local branch",
    )


def push_and_open_pr(
    ctx: "ToolContext",
    *,
    result: GitResult,
    goal: str,
    body: str,
    draft: bool = False,
) -> GitResult:
    """Push the branch and open a PR (best-effort via ``gh``).

    Updates ``result`` in place and returns it. If there is no remote, ``gh`` is
    missing, or auth fails, the result keeps the local commit and the message
    explains what was attempted.
    """
    if not result.ok or not result.branch:
        return result
    if not _is_git_repo(ctx) or not _has_remote(ctx):
        result.message = (result.message + "; no remote — kept local").strip("; ")
        return result

    ok, out = _run(ctx, f"git push -u origin {result.branch}")
    if not ok:
        result.message = (result.message + f"; push failed: {out.strip()[:120]}").strip(
            "; "
        )
        return result

    if not _gh_available(ctx):
        result.message = (result.message + "; pushed; gh CLI missing — no PR").strip(
            "; "
        )
        return result

    title = f"wells: {goal.strip()[:80]}"
    body_safe = body.replace('"', "'").replace("`", "'")
    draft_flag = "--draft " if draft else ""
    ok, out = _run(
        ctx,
        f'gh pr create {draft_flag}--title "{title}" --body "{body_safe[:4000]}" --base ""',
    )
    if not ok:
        result.message = (
            result.message + f"; gh pr failed: {out.strip()[:120]}"
        ).strip("; ")
        return result

    # gh prints the PR URL on success.
    url_match = re.search(r"https?://\S+/pull/\d+", out)
    if url_match:
        result.pr_url = url_match.group(0)
        result.pr_number = int(re.search(r"/pull/(\d+)", result.pr_url).group(1))
        result.message = "PR opened"
    else:
        result.message = (result.message + "; pr created (URL not parsed)").strip("; ")
    return result


def diff_summary(ctx: "ToolContext", *, ref: str = "HEAD") -> str:
    """Return a short diff stat for the current working tree vs ``ref``."""
    if not _is_git_repo(ctx):
        return ""
    ok, out = _run(ctx, f"git diff --stat {ref}")
    return out.strip() if ok else ""


# ---------------------------------------------------------------------------
# Working-tree snapshots (run checkpoints for /undo)
#
# These are internal harness operations, so they bypass the tool layer's
# safety gate on purpose: a checkpoint must never trigger an approval prompt,
# and it never touches the real index or working tree (temp-index trick).
# ---------------------------------------------------------------------------

_UNDO_REF = "refs/wells/undo"


def _git(workspace: str, *args: str, env_extra: dict | None = None,
         timeout: int = 120) -> tuple[bool, str]:
    """Run git directly (argv, no shell) in ``workspace``.

    Thin wrapper over :func:`wells._gitutils.git`. Tests and callers
    reference ``gitops._git`` by name; this re-export keeps that contract.
    """
    from wells._gitutils import git as _real_git

    return _real_git(workspace, *args, env_extra=env_extra, timeout=timeout)


def snapshot_worktree(workspace: str) -> str:
    """Snapshot the working tree (tracked + untracked) to a hidden commit.

    Uses a temporary index so neither the real index nor the working tree is
    touched. Returns the commit sha ('' for non-git workspaces or on failure).
    The sha is also stored at refs/wells/undo so it survives restarts.
    """
    ok, out = _git(workspace, "rev-parse", "--is-inside-work-tree")
    if not ok or "true" not in out.lower():
        return ""

    import os
    import tempfile

    fd, tmp_index = tempfile.mkstemp(prefix="wells-undo-index-")
    os.close(fd)
    os.unlink(tmp_index)  # git wants to create it itself
    env = {"GIT_INDEX_FILE": tmp_index}
    try:
        ok, head = _git(workspace, "rev-parse", "--verify", "HEAD")
        head_sha = head.strip() if ok else ""
        if head_sha:
            if not _git(workspace, "read-tree", "HEAD", env_extra=env)[0]:
                return ""
        if not _git(workspace, "add", "-A", ".", env_extra=env)[0]:
            return ""
        ok, tree = _git(workspace, "write-tree", env_extra=env)
        if not ok:
            return ""
        args = ["commit-tree", tree.strip(), "-m", "wells run checkpoint"]
        if head_sha:
            args += ["-p", head_sha]
        ok, sha = _git(workspace, *args)
        if not ok:
            return ""
        sha = sha.strip()
        _git(workspace, "update-ref", _UNDO_REF, sha)
        return sha
    finally:
        try:
            os.unlink(tmp_index)
        except OSError:
            pass


def snapshot_diff_stat(workspace: str, sha: str) -> str:
    """Diff stat of the current working tree vs a snapshot ('' when identical)."""
    if not sha:
        return ""
    ok, out = _git(workspace, "diff", "--stat", sha)
    return out.strip() if ok else ""


def auto_commit(workspace: str, message: str) -> tuple[bool, str]:
    """Stage everything and commit with ``message`` + Wells authorship trailer.

    Used by the opt-in AUTO_COMMIT flow after a successful run. Returns
    (ok, short sha or error). No-op success when there is nothing to commit.
    """
    ok, out = _git(workspace, "status", "--porcelain")
    if not ok:
        return False, out[:120]
    if not out.strip():
        return True, ""
    if not _git(workspace, "add", "-A")[0]:
        return False, "git add failed"
    full = message.rstrip() + "\n\nCo-Authored-By: Wells Agent <wells@noreply.local>"
    ok, out = _git(workspace, "commit", "--no-verify", "-m", full)
    if not ok:
        return False, out[:160]
    ok, sha = _git(workspace, "rev-parse", "--short", "HEAD")
    return True, sha.strip() if ok else ""


def restore_snapshot(workspace: str, sha: str) -> tuple[bool, str]:
    """Restore the working tree to snapshot ``sha``.

    Restores every file recorded in the snapshot (mods and deletions undone)
    and removes files created since. Returns (ok, human message).
    """
    ok, _ = _git(workspace, "rev-parse", "--verify", f"{sha}^{{commit}}")
    if not ok:
        return False, "checkpoint not found (was the run in a git repo?)"

    # Files present now (tracked + untracked, .gitignore honoured).
    ok, now = _git(workspace, "ls-files", "--cached", "--others", "--exclude-standard")
    now_set = set(now.splitlines()) if ok else set()
    ok, snap = _git(workspace, "ls-tree", "-r", "--name-only", sha)
    if not ok:
        return False, f"could not read snapshot: {snap[:160]}"
    snap_set = set(snap.splitlines())

    ok, out = _git(workspace, "restore", "--source", sha, "--worktree", "--", ".")
    if not ok:
        return False, f"restore failed: {out[:200]}"

    removed = 0
    from pathlib import Path
    for rel in sorted(now_set - snap_set):
        try:
            (Path(workspace) / rel).unlink()
            removed += 1
        except OSError:
            pass

    msg = f"working tree restored to checkpoint {sha[:8]}"
    if removed:
        msg += f"; removed {removed} file(s) created since"
    return True, msg
