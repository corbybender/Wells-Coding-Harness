"""Parallel worktree fleet: spawn N isolated git worktrees, run the same
task in each concurrently — optionally on different model profiles — then
compare outcomes and merge the winner while cleaning up the rest.

Why worktrees, not branches in one checkout: two runs writing to the same
working tree at once (or even sequentially, without a real checkout swap)
would corrupt each other's edits. ``git worktree`` gives each run its own
real directory + branch off the same repo (shared object store — fast,
disk-cheap) — exactly what "compare N independent approaches to a task"
needs, and each is a genuine `wells "<goal>"` run (full planner->coder->
tester->reviewer graph), not a scoped-down subagent.

Manifests persist to ``~/.wells/fleet/<fleet_id>.json`` so ``fleet list``/
``pick``/``drop`` work as separate CLI invocations after the (typically
slow) spawn+run completes. Worktree checkouts themselves live under
``~/.wells/fleet/worktrees/<fleet_id>/<i>`` — deliberately OUTSIDE the repo
being worked on (git allows nesting a worktree inside its own repo's tree,
but doing so pollutes `git status` in the main working tree with a large
untracked directory for the whole fleet run, and risks a broad `git add -A`
there staging sibling worktrees' admin files into a real commit).
"""

from __future__ import annotations

import json
import re
import subprocess
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path

FLEET_DIR = Path.home() / ".wells" / "fleet"
# Worktree checkouts live OUTSIDE any repo, under the user's home directory —
# never nested inside repo_root. A worktree path inside the tracked repo
# tree (an earlier version of this module used <repo_root>/.wells/fleet/...)
# pollutes `git status` in the main working tree with a large untracked
# directory for the whole time a fleet is running, and risks a broad
# `git add -A` run in the main tree mid-fleet staging sibling worktrees'
# admin files into a real commit. Nothing requires the checkout to live
# inside the repo; putting it beside the manifests avoids the problem
# entirely.
FLEET_WORKTREES_DIR = FLEET_DIR / "worktrees"
_MEMBER_TIMEOUT = 3600.0  # 1h wall-clock cap per member; a runaway run must not hang the fleet forever


def _ensure_dir() -> None:
    FLEET_DIR.mkdir(parents=True, exist_ok=True)


def new_fleet_id() -> str:
    ts = time.strftime("%Y%m%d-%H%M%S")
    return f"{ts}-{uuid.uuid4().hex[:6]}"


def _slugify(text: str, *, max_len: int = 30) -> str:
    s = re.sub(r"[^A-Za-z0-9._-]+", "-", text.strip().lower()).strip("-")
    s = re.sub(r"-+", "-", s)
    return (s or "task")[:max_len]


# ---------------------------------------------------------------------------
# Git plumbing
# ---------------------------------------------------------------------------


def _git(cwd: str, *args: str, timeout: float = 60.0) -> tuple[bool, str]:
    """Run git directly (argv, no shell) in ``cwd``.

    Thin wrapper over :func:`wells._gitutils.git` preserving the historical
    60s default (the unified primitive uses 120s — generous for everything,
    but fleet's worktree add/remove are quick and a runaway git op shouldn't
    hold a fleet run for two minutes). Tests monkey-patch this name.
    """
    from wells._gitutils import git as _real_git

    return _real_git(cwd, *args, timeout=timeout)


def is_git_repo(path: str) -> bool:
    from wells._gitutils import is_git_repo as _impl
    return _impl(path)


def current_branch(path: str) -> str:
    from wells._gitutils import current_branch as _impl
    return _impl(path)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class FleetMember:
    index: int
    branch: str
    worktree_path: str
    profile: str = ""  # "" = active profile
    status: str = "pending"  # pending | running | complete | incomplete | error
    summary: str = ""
    diff_stat: str = ""
    files_changed: int = 0
    tokens_total: int = 0
    cost_usd: float | None = None
    duration_seconds: int = 0
    error: str = ""


@dataclass
class FleetManifest:
    fleet_id: str
    repo_root: str
    base_branch: str
    task: str
    created_at: str
    members: list[FleetMember] = field(default_factory=list)
    winner: int | None = None  # index of the picked member, once decided
    resolved: bool = False  # True once pick/drop has run (worktrees gone)

    def to_json(self) -> dict:
        d = asdict(self)
        return d

    @classmethod
    def from_json(cls, d: dict) -> "FleetManifest":
        members = [FleetMember(**m) for m in d.get("members", [])]
        return cls(
            fleet_id=d["fleet_id"], repo_root=d["repo_root"],
            base_branch=d["base_branch"], task=d["task"], created_at=d["created_at"],
            members=members, winner=d.get("winner"), resolved=d.get("resolved", False),
        )


def save_manifest(m: FleetManifest) -> Path:
    _ensure_dir()
    path = FLEET_DIR / f"{m.fleet_id}.json"
    path.write_text(json.dumps(m.to_json(), indent=2, default=str), encoding="utf-8")
    return path


def load_manifest(fleet_id: str) -> FleetManifest | None:
    path = FLEET_DIR / f"{fleet_id}.json"
    if not path.is_file():
        return None
    try:
        return FleetManifest.from_json(json.loads(path.read_text(encoding="utf-8")))
    except Exception:
        return None


def list_manifests() -> list[FleetManifest]:
    _ensure_dir()
    out = []
    for p in sorted(FLEET_DIR.glob("*.json"), key=lambda x: x.stat().st_mtime, reverse=True):
        try:
            out.append(FleetManifest.from_json(json.loads(p.read_text(encoding="utf-8"))))
        except Exception:
            continue
    return out


# ---------------------------------------------------------------------------
# Spawn
# ---------------------------------------------------------------------------


def spawn_worktrees(
    repo_root: str, fleet_id: str, task: str, n: int, *, profiles: list[str] | None = None,
) -> FleetManifest:
    """Create N worktrees on fresh branches off the current HEAD.

    Raises RuntimeError if repo_root isn't a git repo, or a worktree/branch
    fails to create — a fleet that's only partially spawned is worse than
    one that fails loudly before any run starts.
    """
    if not is_git_repo(repo_root):
        raise RuntimeError(f"{repo_root} is not a git repository — fleet requires git.")
    base = current_branch(repo_root)
    if not base:
        raise RuntimeError("Could not determine the current branch (detached HEAD?).")

    slug = _slugify(task)
    members: list[FleetMember] = []
    for i in range(n):
        branch = f"wells-fleet/{fleet_id}/{i}-{slug}"
        wt_path = str(FLEET_WORKTREES_DIR / fleet_id / str(i))
        Path(wt_path).parent.mkdir(parents=True, exist_ok=True)
        ok, out = _git(repo_root, "worktree", "add", "-b", branch, wt_path, base)
        if not ok:
            # Roll back whatever was already created — a half-spawned fleet
            # left on disk would confuse both the user and the next spawn.
            for m in members:
                _git(repo_root, "worktree", "remove", "--force", m.worktree_path)
                _git(repo_root, "branch", "-D", m.branch)
            raise RuntimeError(f"Failed to create worktree {i}: {out}")
        profile = (profiles[i] if profiles and i < len(profiles) else "")
        members.append(FleetMember(index=i, branch=branch, worktree_path=wt_path, profile=profile))

    return FleetManifest(
        fleet_id=fleet_id, repo_root=repo_root, base_branch=base, task=task,
        created_at=time.strftime("%Y-%m-%d %H:%M:%S"), members=members,
    )


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------


def _run_member(manifest: FleetManifest, member: FleetMember) -> None:
    """Run a full headless harness invocation in one member's worktree.

    Runs as a genuine child process (``python -c "from wells.main import
    main; main()" --workspace <worktree> --output-format json -p "<task>"``)
    rather than an in-process ``build_graph().invoke()`` call on a worker
    thread. That in-process design was tried first and rejected: the model
    profile and the token ledger are both process-wide globals
    (``config.ACTIVE_PROFILE``, ``tokens.LEDGER``) — concurrent fleet members
    running in the same process would race on both, silently corrupting
    which profile a member actually used and whose usage its tokens got
    billed to. A subprocess gives each member its own environment, its own
    config module instance, and its own ledger — the isolation this design
    actually needs — and reuses the headless JSON mode (see ``main.py``) to
    get a structured result back for free.
    """
    import os
    import sys

    member.status = "running"
    t0 = time.time()
    env = dict(os.environ)
    if member.profile:
        env["MODEL_PROFILE"] = member.profile

    argv = [
        "--workspace", member.worktree_path,
        "--output-format", "json", "-p", manifest.task,
    ]
    try:
        proc = subprocess.run(
            [sys.executable, "-c", "from wells.main import main; main()", *argv],
            capture_output=True, text=True, env=env, timeout=_MEMBER_TIMEOUT,
        )
        payload = None
        for line in reversed(proc.stdout.splitlines()):
            line = line.strip()
            if line.startswith("{"):
                try:
                    payload = json.loads(line)
                    break
                except json.JSONDecodeError:
                    continue
        if payload is None:
            member.status = "error"
            member.error = (
                f"no JSON result on stdout (exit {proc.returncode}); "
                f"stderr tail: {proc.stderr[-500:]}"
            )
        else:
            member.status = payload.get("status", "error")
            member.summary = (payload.get("summary") or "")[:2000]
            member.diff_stat = payload.get("git_summary", "")
            member.tokens_total = (payload.get("tokens") or {}).get("total", 0)
            member.cost_usd = payload.get("cost_usd")
            if member.status == "error" and not member.error:
                member.error = payload.get("error", "") or payload.get("review_result", "")[:500]
    except subprocess.TimeoutExpired:
        member.status = "error"
        member.error = f"timed out after {_MEMBER_TIMEOUT}s"
    except Exception as e:
        member.status = "error"
        member.error = f"{type(e).__name__}: {e}"

    ok, out = _git(member.worktree_path, "diff", "--stat", "HEAD")
    member.files_changed = len([ln for ln in out.splitlines() if "|" in ln]) if ok else 0
    member.duration_seconds = int(time.time() - t0)


def run_fleet(
    repo_root: str, task: str, n: int, *, profiles: list[str] | None = None,
    max_workers: int = 4,
) -> FleetManifest:
    """Spawn N worktrees and run the goal in each concurrently; saves the manifest."""
    fleet_id = new_fleet_id()
    manifest = spawn_worktrees(repo_root, fleet_id, task, n, profiles=profiles)
    save_manifest(manifest)

    workers = max(1, min(max_workers, n))
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="wells-fleet") as pool:
        futs = [pool.submit(_run_member, manifest, m) for m in manifest.members]
        for f in futs:
            f.result()  # propagate nothing — _run_member never raises out

    save_manifest(manifest)
    return manifest


# ---------------------------------------------------------------------------
# Pick / drop
# ---------------------------------------------------------------------------


def pick_winner(fleet_id: str, index: int) -> tuple[bool, str]:
    """Merge member `index`'s branch into the fleet's base branch, then
    remove every worktree/branch (winner included — its commits now live
    on the base branch, the throwaway branch has no further purpose)."""
    manifest = load_manifest(fleet_id)
    if manifest is None:
        return False, f"No such fleet: {fleet_id}"
    if manifest.resolved:
        return False, f"Fleet {fleet_id} was already resolved (winner={manifest.winner})."
    member = next((m for m in manifest.members if m.index == index), None)
    if member is None:
        return False, f"No member {index} in fleet {fleet_id}."

    ok, out = _git(manifest.repo_root, "checkout", manifest.base_branch)
    if not ok:
        return False, f"Could not check out {manifest.base_branch}: {out}"
    ok, out = _git(
        manifest.repo_root, "merge", "--no-ff", member.branch,
        "-m", f"wells fleet: merge winner #{index} ({fleet_id})",
    )
    if not ok:
        return False, f"Merge of {member.branch} failed (resolve manually): {out}"

    _cleanup_worktrees(manifest)
    manifest.winner = index
    manifest.resolved = True
    save_manifest(manifest)
    return True, f"Merged {member.branch} into {manifest.base_branch}; fleet cleaned up."


def drop_fleet(fleet_id: str) -> tuple[bool, str]:
    """Abandon a fleet: remove every worktree/branch, no merge."""
    manifest = load_manifest(fleet_id)
    if manifest is None:
        return False, f"No such fleet: {fleet_id}"
    if manifest.resolved:
        return False, f"Fleet {fleet_id} was already resolved."
    _cleanup_worktrees(manifest)
    manifest.resolved = True
    save_manifest(manifest)
    return True, f"Dropped fleet {fleet_id} ({len(manifest.members)} worktrees removed)."


def _cleanup_worktrees(manifest: FleetManifest) -> None:
    for m in manifest.members:
        _git(manifest.repo_root, "worktree", "remove", "--force", m.worktree_path)
        _git(manifest.repo_root, "branch", "-D", m.branch)
    _git(manifest.repo_root, "worktree", "prune")
    try:
        (FLEET_WORKTREES_DIR / manifest.fleet_id).rmdir()  # only succeeds if empty
    except OSError:
        pass  # non-empty (a remove above failed) or already gone — either way, harmless
