"""Per-subagent git worktrees for isolated parallel writes.

:mod:`wells.background` exposes ``bg_start`` so the parent agent can fan out
concurrent sub-agents. With ``role="worktree"``, each sub-agent gets its own
``git worktree`` — a real checkout on its own branch sharing the repo's object
store — so multiple writes can happen in parallel without clobbering each
other in the parent's working tree. On ``bg_collect`` the sub-agent's commit
is cherry-picked into the parent; on conflict the cherry-pick is aborted and
the diff is returned to the parent agent to re-apply manually (no surprise
merges, no semantic guesses).

Worktree paths live under ``~/.wells/bg-worktrees/<slot_id>/`` — deliberately
outside any repo (see :mod:`wells.fleet` for the same rationale: a worktree
nested inside the parent repo's tracked tree pollutes ``git status`` and risks
staging sibling admin files into a real commit).

Design:
  * Every operation uses ``git`` directly via :func:`_git` (argv, no shell).
    These are internal harness operations, not agent tool calls — they bypass
    the safety gate on purpose (mirrors :mod:`gitops` snapshot/restore).
  * Primitives are independently testable; :mod:`background` composes them.
  * The worktree handle is registered on the background slot *before* the
    sub-agent thread starts, so ``REGISTRY.reset`` can always reap a
    half-spawned or cancelled worktree.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from pathlib import Path

BG_WORKTREES_DIR = Path.home() / ".wells" / "bg-worktrees"


def _git(cwd: str, *args: str, timeout: float = 60.0) -> tuple[bool, str]:
    """Run ``git`` directly (argv, no shell) in ``cwd``. Returns (ok, output).

    Thin wrapper over :func:`wells._gitutils.git` preserving the historical
    60s default. Tests monkey-patch this name.
    """
    from wells._gitutils import git as _real_git

    return _real_git(cwd, *args, timeout=timeout)


def is_git_repo(path: str) -> bool:
    from wells._gitutils import is_git_repo as _impl
    return _impl(path)


def current_branch(path: str) -> str:
    from wells._gitutils import current_branch as _impl
    return _impl(path)


def head_sha(path: str) -> str:
    from wells._gitutils import head_sha as _impl
    return _impl(path)


def _slugify(text: str, *, max_len: int = 30) -> str:
    s = re.sub(r"[^A-Za-z0-9._-]+", "-", text.strip().lower()).strip("-")
    s = re.sub(r"-+", "-", s)
    return (s or "task")[:max_len]


@dataclass
class WorktreeHandle:
    """A live worktree + its branch, ready to clean up.

    The caller MUST arrange cleanup via :func:`remove_worktree` — the worktree
    and branch stay alive until then so the parent can cherry-pick from them.
    """

    worktree_path: str
    branch: str
    base_sha: str  # parent HEAD at spawn time (cherry-pick range start)
    slot_id: str


def create_worktree(
    parent_workspace: str, *, slot_id: str, task: str
) -> WorktreeHandle:
    """Create a worktree off the parent's current HEAD on a fresh branch.

    Raises ``RuntimeError`` on failure (non-git parent, git error). The caller
    MUST arrange cleanup via :func:`remove_worktree`.
    """
    if not is_git_repo(parent_workspace):
        raise RuntimeError(
            f"{parent_workspace} is not a git repository — role=worktree requires git"
        )
    base = head_sha(parent_workspace)
    if not base:
        raise RuntimeError("could not resolve parent HEAD")
    slug = _slugify(task)
    ts = time.strftime("%m%d%H%M%S")
    branch = f"wells-bg/{slot_id}/{slug}-{ts}"
    wt_path = str(BG_WORKTREES_DIR / slot_id)
    # Parent dir must exist before `git worktree add` will write into it.
    Path(wt_path).parent.mkdir(parents=True, exist_ok=True)
    ok, out = _git(
        parent_workspace,
        "worktree",
        "add",
        "-b",
        branch,
        wt_path,
        base,
    )
    if not ok:
        raise RuntimeError(f"git worktree add failed: {out[:200]}")
    return WorktreeHandle(
        worktree_path=wt_path,
        branch=branch,
        base_sha=base,
        slot_id=slot_id,
    )


@dataclass
class CommitResult:
    """Outcome of committing a worktree's pending edits."""

    ok: bool
    tip_sha: str  # '' when there was nothing to commit
    message: str = ""


def commit_pending(handle: WorktreeHandle, *, message: str) -> CommitResult:
    """Stage + commit any uncommitted changes in the worktree to its branch.

    Returns ``CommitResult(ok=True, tip_sha='')`` when the working tree is
    clean — caller uses the empty sha to skip the cherry-pick.
    """
    ok, out = _git(handle.worktree_path, "status", "--porcelain")
    if not ok:
        return CommitResult(False, "", f"git status failed: {out[:200]}")
    if not out.strip():
        return CommitResult(True, "")  # nothing to commit
    if not _git(handle.worktree_path, "add", "-A")[0]:
        return CommitResult(False, "", "git add failed in worktree")
    msg = message.replace('"', "'")
    ok, out = _git(handle.worktree_path, "commit", "--no-verify", "-m", msg)
    if not ok:
        return CommitResult(False, "", f"commit failed: {out[:200]}")
    ok2, sha = _git(handle.worktree_path, "rev-parse", "HEAD")
    if not ok2:
        return CommitResult(False, "", f"rev-parse failed: {sha[:200]}")
    return CommitResult(True, sha.strip())


@dataclass
class MergeResult:
    """Outcome of cherry-picking a worktree's commits into the parent."""

    ok: bool  # True = cleanly merged (or nothing to merge)
    skipped: bool  # True when the worktree had no commits to merge
    conflict: bool  # True when the cherry-pick was aborted due to conflict
    diff: str  # on conflict, the worktree-vs-base diff for the parent agent
    message: str


def cherry_pick_into_parent(
    parent_workspace: str,
    handle: WorktreeHandle,
    *,
    tip_sha: str,
) -> MergeResult:
    """Cherry-pick the sub-agent's commits (base..tip) into the parent.

    On conflict, aborts the cherry-pick and returns the worktree-vs-base diff
    so the parent agent can re-apply the change manually. The caller then
    discards the worktree regardless.
    """
    if not tip_sha or tip_sha == handle.base_sha:
        return MergeResult(
            ok=True,
            skipped=True,
            conflict=False,
            diff="",
            message="no commits",
        )
    ok, out = _git(
        parent_workspace,
        "cherry-pick",
        f"{handle.base_sha}..{tip_sha}",
    )
    if ok:
        return MergeResult(
            ok=True,
            skipped=False,
            conflict=False,
            diff="",
            message="merged cleanly",
        )
    # Conflict: abort, then capture the diff for the parent agent to re-apply.
    _git(parent_workspace, "cherry-pick", "--abort")
    ok2, diff = _git(
        parent_workspace,
        "diff",
        handle.base_sha,
        tip_sha,
        "--",
    )
    diff_text = (
        diff
        if ok2 and diff.strip()
        else (f"(cherry-pick failed; no diff available)\n{out[:300]}")
    )
    return MergeResult(
        ok=False,
        skipped=False,
        conflict=True,
        diff=diff_text,
        message="cherry-pick conflict — aborted",
    )


def remove_worktree(parent_workspace: str, handle: WorktreeHandle) -> None:
    """Remove the worktree + delete its branch. Best-effort; never raises.

    Safe to call multiple times — subsequent calls are no-ops.
    """
    _git(parent_workspace, "worktree", "remove", "--force", handle.worktree_path)
    _git(parent_workspace, "branch", "-D", handle.branch)
    _git(parent_workspace, "worktree", "prune")


def reap_stray_worktrees(parent_workspace: str) -> int:
    """Prune worktrees whose checkouts no longer exist (e.g. after a crash).

    Best-effort; never raises. Returns how many were pruned. Call from
    :func:`background.REGISTRY.reset` or a doctor self-heal pass.
    """
    if not is_git_repo(parent_workspace):
        return 0
    before_ok, before = _git(parent_workspace, "worktree", "list")
    before_count = len(before.splitlines()) if before_ok else 0
    _git(parent_workspace, "worktree", "prune", "-v")
    after_ok, after = _git(parent_workspace, "worktree", "list")
    after_count = len(after.splitlines()) if after_ok else 0
    return max(0, before_count - after_count)
